"""
Publication-grade inference engine for ImageTrust.

Uses the Phase 2 XGBoost meta-classifier trained on 604,589 samples
(including 105k WhatsApp images) with LAC conformal prediction guarantees.

This module replaces the ad-hoc heuristic fusion (max() across models)
with a principled meta-classifier that:
1. Extracts embeddings from 3 ImageNet-pretrained backbones
2. Computes NIQE image quality score
3. Predicts P(AI) via XGBoost trained on these features
4. Applies LAC conformal prediction for coverage guarantees

Tiers (graceful fallback):
  Tier 1: Phase 2 XGBoost on backbone embeddings (primary)
  Tier 2: Calibrated CNN ensemble average (if Phase 2 unavailable)
  Tier 3: HF model average (last resort)
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# Default paths relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PHASE2_MODEL = _PROJECT_ROOT / "models" / "phase2" / "xgboost" / "meta_classifier.json"
_CONFORMAL_CONFIG = _PROJECT_ROOT / "models" / "phase2" / "conformal_all.json"

# Backbone configuration (must match training order exactly)
_BACKBONES = ["resnet50", "efficientnet_b0", "vit_b_16"]
_EMBED_DIMS = {"resnet50": 2048, "efficientnet_b0": 1280, "vit_b_16": 768}


@dataclass
class PublicationPrediction:
    """Result from the publication-grade inference engine."""

    ai_probability: float
    verdict: str                          # "real" | "ai_generated" | "uncertain"
    prediction_set: Set[str]              # {"real"}, {"ai_generated"}, or both
    conformal_coverage: float             # 0.95 for Tier 1, 0.0 otherwise
    is_uncertain: bool
    tier_used: str                        # "phase2_xgboost" | "cnn_ensemble" | "hf_fallback"
    tier_reason: str
    niqe_score: Optional[float] = None
    cnn_ensemble_prob: Optional[float] = None
    hf_probs: Dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0


class Phase2Predictor:
    """
    Phase 2 meta-classifier: XGBoost on backbone embeddings + NIQE.

    Replicates EXACTLY the training-time feature extraction from
    scripts/orchestrator/run_embedding_extraction.py for single-image inference.

    Feature vector (4097-dim):
      [resnet50_0..2047 | efficientnet_b0_0..1279 | vit_b_16_0..767 | niqe]
    """

    def __init__(
        self,
        model_path: Path = _PHASE2_MODEL,
        conformal_path: Path = _CONFORMAL_CONFIG,
        device: str = "auto",
    ):
        import torch

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load XGBoost model
        import xgboost as xgb

        self._booster = xgb.Booster()
        self._booster.load_model(str(model_path))
        logger.info("Loaded Phase 2 XGBoost from %s", model_path)

        # Load conformal config (LAC at alpha=0.05)
        with open(conformal_path, encoding="utf-8") as f:
            conformal_data = json.load(f)
        lac = conformal_data["xgboost"]["lac_alpha0.05"]
        self._lac_threshold = lac["threshold"]   # 0.7652
        self._lac_coverage = lac["coverage"]     # 0.9519
        logger.info(
            "LAC threshold=%.4f coverage=%.4f",
            self._lac_threshold, self._lac_coverage,
        )

        # Build feature names (must match training order exactly)
        self._feature_names = []
        for backbone in _BACKBONES:
            dim = _EMBED_DIMS[backbone]
            self._feature_names.extend([f"{backbone}_{i}" for i in range(dim)])
        self._feature_names.append("niqe")

        # Load backbone models with embedding hooks
        self._backbones = {}
        self._hook_outputs: Dict[str, Optional[object]] = {}
        self._load_backbones(torch)

        # NIQE scorer
        self._niqe_type, self._niqe_model = self._init_niqe(torch)

        # Transform pipelines
        from torchvision import transforms

        self._backbone_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._niqe_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # [0, 1] range, no normalization
        ])

    def _load_backbones(self, torch_module):
        """Load 3 pretrained backbones with embedding hooks."""
        import torchvision.models as models

        backbone_loaders = {
            "resnet50": lambda: models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            ),
            "efficientnet_b0": lambda: models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            ),
            "vit_b_16": lambda: models.vit_b_16(
                weights=models.ViT_B_16_Weights.IMAGENET1K_V1
            ),
        }

        for name in _BACKBONES:
            model = backbone_loaders[name]()
            model.to(self.device)
            model.eval()
            self._hook_outputs[name] = None
            self._register_hook(model, name)
            self._backbones[name] = model
            logger.info("  Loaded %s: embed_dim=%d", name, _EMBED_DIMS[name])

    def _register_hook(self, model, name: str):
        """Register forward hook matching training-time extraction exactly."""

        def make_hook(backbone_name):
            def hook_fn(module, input, output):
                import torch

                if isinstance(output, torch.Tensor):
                    if output.dim() == 4:
                        # Conv features: global average pool
                        self._hook_outputs[backbone_name] = output.mean(dim=[2, 3])
                    elif output.dim() > 2:
                        # Sequence output: take CLS token
                        self._hook_outputs[backbone_name] = output[:, 0]
                    else:
                        self._hook_outputs[backbone_name] = output
            return hook_fn

        if name == "resnet50":
            model.avgpool.register_forward_hook(make_hook(name))
        elif name == "efficientnet_b0":
            model.avgpool.register_forward_hook(make_hook(name))
        elif name == "vit_b_16":
            # ViT encoder outputs (batch, seq_len, hidden_dim).
            # Extract CLS token (position 0) before passing to hook_fn.
            hook = make_hook(name)
            model.encoder.register_forward_hook(
                lambda m, i, o, _h=hook: _h(m, i, o[:, 0])
            )

    def _init_niqe(self, torch_module):
        """Initialize NIQE scorer with PIQ or fallback."""
        try:
            import piq

            if hasattr(piq, "NIQE"):
                niqe = piq.NIQE().to(self.device)
                logger.info("  NIQE: PIQ library (GPU accelerated)")
                return "piq", niqe
            logger.info("  NIQE: PIQ %s has no NIQE class, using MSCN fallback", piq.__version__)
            return "fallback", None
        except Exception:
            logger.info("  NIQE: MSCN fallback (CPU)")
            return "fallback", None

    def _compute_niqe(self, image) -> float:
        """Compute NIQE score for a single PIL Image."""
        import torch

        img_tensor = self._niqe_transform(image.convert("RGB")).unsqueeze(0)

        if self._niqe_type == "piq" and self._niqe_model is not None:
            try:
                img_tensor = img_tensor.to(self.device)
                with torch.no_grad():
                    score = self._niqe_model(img_tensor).item()
                return float(score)
            except Exception:
                pass  # Fall through to MSCN fallback

        # Fallback: MSCN-based approximation (matches training fallback)
        try:
            from scipy.ndimage import gaussian_filter

            img_np = img_tensor.squeeze(0).numpy()
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            gray = np.mean(img_np, axis=2)
            mu = gaussian_filter(gray, sigma=7 / 6)
            sigma = np.sqrt(gaussian_filter((gray - mu) ** 2, sigma=7 / 6))
            sigma = np.maximum(sigma, 1e-6)
            mscn = (gray - mu) / sigma
            return float((np.abs(np.mean(mscn)) + np.abs(1 - np.var(mscn))) * 10)
        except Exception:
            return 50.0

    def predict(self, image) -> PublicationPrediction:
        """
        Run full Phase 2 inference on a single PIL Image.

        Returns PublicationPrediction with conformal prediction guarantees.
        """
        import torch
        import xgboost as xgb

        t0 = time.perf_counter()

        # 1. Prepare image tensor for backbones (ImageNet normalization)
        img_tensor = self._backbone_transform(image.convert("RGB")).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # 2. Extract embeddings from all 3 backbones
        embeddings = []
        with torch.no_grad():
            for name in _BACKBONES:
                _ = self._backbones[name](img_tensor)
                emb = self._hook_outputs[name].cpu().numpy().flatten()
                self._hook_outputs[name] = None
                embeddings.append(emb)

        # 3. Compute NIQE quality score
        niqe_score = self._compute_niqe(image)

        # 4. Concatenate features (exact order from training):
        #    [resnet50_2048 | efficientnet_b0_1280 | vit_b_16_768 | niqe] = 4097
        features = np.concatenate(embeddings + [np.array([niqe_score])])
        features = features.astype(np.float32).reshape(1, -1)

        # 5. Create DMatrix with feature names matching training
        dmatrix = xgb.DMatrix(features, feature_names=self._feature_names)

        # 6. Predict P(AI)
        ai_prob = float(self._booster.predict(dmatrix)[0])

        # 7. LAC conformal prediction (alpha=0.05)
        #    Nonconformity scores: score_k = 1 - p_k
        #    Class included if score_k <= threshold
        score_real = ai_prob          # = 1 - P(real) = 1 - (1 - ai_prob) = ai_prob
        score_ai = 1.0 - ai_prob     # = 1 - P(ai) = 1 - ai_prob

        prediction_set = set()
        if score_real <= self._lac_threshold:
            prediction_set.add("real")
        if score_ai <= self._lac_threshold:
            prediction_set.add("ai_generated")

        # Edge case: empty prediction set (extreme confidence beyond threshold)
        if not prediction_set:
            prediction_set = {"ai_generated"} if ai_prob > 0.5 else {"real"}

        is_uncertain = len(prediction_set) > 1
        if is_uncertain:
            verdict = "uncertain"
        elif "ai_generated" in prediction_set:
            verdict = "ai_generated"
        else:
            verdict = "real"

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return PublicationPrediction(
            ai_probability=ai_prob,
            verdict=verdict,
            prediction_set=prediction_set,
            conformal_coverage=self._lac_coverage,
            is_uncertain=is_uncertain,
            tier_used="phase2_xgboost",
            tier_reason="Phase 2 meta-classifier (604,589 training samples)",
            niqe_score=niqe_score,
            inference_time_ms=elapsed_ms,
        )

    @property
    def lac_threshold(self) -> float:
        return self._lac_threshold

    @property
    def lac_coverage(self) -> float:
        return self._lac_coverage

    @property
    def backbone_names(self):
        return list(self._backbones.keys())


class PublicationInferenceEngine:
    """
    Multi-tier inference engine with graceful fallback.

    Tier 1: Phase 2 XGBoost meta-classifier (604k training samples)
    Tier 2: Calibrated CNN ensemble average (if Phase 2 unavailable)
    Tier 3: HF model weighted average (last resort)

    Critical rule: Forensics NEVER modifies ai_prob.
    Forensic evidence is displayed separately as independent evidence.
    """

    def __init__(self, device: str = "auto"):
        self._phase2: Optional[Phase2Predictor] = None
        self._tier1_available = False
        self._tier1_error = ""

        # Try to load Phase 2
        try:
            if _PHASE2_MODEL.exists() and _CONFORMAL_CONFIG.exists():
                self._phase2 = Phase2Predictor(device=device)
                self._tier1_available = True
                logger.info("Publication engine: Tier 1 (Phase 2 XGBoost) ready")
            else:
                missing = []
                if not _PHASE2_MODEL.exists():
                    missing.append(str(_PHASE2_MODEL))
                if not _CONFORMAL_CONFIG.exists():
                    missing.append(str(_CONFORMAL_CONFIG))
                self._tier1_error = f"Missing: {', '.join(missing)}"
                logger.warning("Phase 2 unavailable: %s", self._tier1_error)
        except Exception as e:
            self._tier1_error = str(e)
            logger.warning("Phase 2 init failed: %s", e)

    @property
    def tier1_available(self) -> bool:
        return self._tier1_available

    @property
    def tier1_error(self) -> str:
        return self._tier1_error

    @property
    def phase2(self) -> Optional[Phase2Predictor]:
        return self._phase2

    def analyze(
        self,
        image,
        cal_ensemble: Optional[dict] = None,
        hf_results: Optional[list] = None,
        source_info: Optional[dict] = None,
    ) -> PublicationPrediction:
        """
        Run publication-grade inference with tiered fallback.

        Args:
            image: PIL Image to analyze
            cal_ensemble: Calibrated CNN ensemble results (from existing detector)
            hf_results: HuggingFace model results (from existing detector)
            source_info: Source/platform analysis results

        Returns:
            PublicationPrediction with verdict and conformal guarantees
        """
        # Tier 1: Phase 2 XGBoost
        if self._tier1_available and self._phase2 is not None:
            try:
                pred = self._phase2.predict(image)
                # Attach reference data from other detectors
                if cal_ensemble:
                    pred.cnn_ensemble_prob = cal_ensemble.get(
                        "ensemble_avg_prob",
                        cal_ensemble.get("ensemble_min_prob"),
                    )
                if hf_results:
                    for r in hf_results:
                        method = r.get("method", "unknown")
                        pred.hf_probs[method] = r.get("ai_probability", 0.5)
                return pred
            except Exception as e:
                logger.error("Phase 2 prediction failed: %s", e)
                # Fall through to Tier 2

        # Tier 2: Calibrated CNN Ensemble
        if cal_ensemble and cal_ensemble.get("calibrated_probs"):
            t0 = time.perf_counter()
            ai_prob = cal_ensemble.get("ensemble_avg_prob", 0.5)

            # Social media: widen UNCERTAIN zone
            is_social = False
            if source_info:
                is_social = (
                    source_info.get("is_social")
                    or source_info.get("likely_compressed", False)
                )

            if is_social:
                low_t, high_t = 0.25, 0.65
            else:
                low_t, high_t = 0.34, 0.54

            if ai_prob >= high_t:
                verdict = "ai_generated"
                pred_set = {"ai_generated"}
            elif ai_prob < low_t:
                verdict = "real"
                pred_set = {"real"}
            else:
                verdict = "uncertain"
                pred_set = {"real", "ai_generated"}

            hf_prob_map = {}
            if hf_results:
                for r in hf_results:
                    hf_prob_map[r.get("method", "?")] = r.get("ai_probability", 0.5)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            reason = self._tier1_error or "Phase 2 models not available"

            return PublicationPrediction(
                ai_probability=ai_prob,
                verdict=verdict,
                prediction_set=pred_set,
                conformal_coverage=0.0,
                is_uncertain=len(pred_set) > 1,
                tier_used="cnn_ensemble",
                tier_reason=f"Fallback: {reason}",
                cnn_ensemble_prob=ai_prob,
                hf_probs=hf_prob_map,
                inference_time_ms=elapsed_ms,
            )

        # Tier 3: HF weighted average (last resort)
        t0 = time.perf_counter()
        hf_prob_map = {}
        if hf_results:
            for r in hf_results:
                hf_prob_map[r.get("method", "?")] = r.get("ai_probability", 0.5)

        ai_prob = (
            sum(hf_prob_map.values()) / len(hf_prob_map)
            if hf_prob_map
            else 0.5
        )

        if ai_prob >= 0.54:
            verdict = "ai_generated"
            pred_set = {"ai_generated"}
        elif ai_prob < 0.34:
            verdict = "real"
            pred_set = {"real"}
        else:
            verdict = "uncertain"
            pred_set = {"real", "ai_generated"}

        elapsed_ms = (time.perf_counter() - t0) * 1000
        reason = self._tier1_error or "Phase 2 and CNN ensemble unavailable"

        return PublicationPrediction(
            ai_probability=ai_prob,
            verdict=verdict,
            prediction_set=pred_set,
            conformal_coverage=0.0,
            is_uncertain=len(pred_set) > 1,
            tier_used="hf_fallback",
            tier_reason=f"Last resort: {reason}",
            hf_probs=hf_prob_map,
            inference_time_ms=elapsed_ms,
        )
