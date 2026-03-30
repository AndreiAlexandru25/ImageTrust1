"""
Calibrated CNN Ensemble Detector

Loads all 3 custom-trained CNN models (ResNet-50, EfficientNetV2-M, ConvNeXt-Base),
applies temperature scaling for calibration, and uses empirically-determined
thresholds with UNCERTAIN region for three-way classification.

Calibration data: 160,705 validation samples, 1,000 bootstrap iterations.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)

# Default calibrated thresholds (from full 160,705-sample calibration)
_DEFAULT_THRESHOLDS = {
    "ResNet-50": {
        "f1_optimal": 0.5172,
        "temperature": 0.6401,
        "uncertain_low": 0.34,
        "uncertain_high": 0.54,
    },
    "EfficientNetV2-M": {
        "f1_optimal": 0.4778,
        "temperature": 0.5641,
        "uncertain_low": 0.34,
        "uncertain_high": 0.54,
    },
    "ConvNeXt-Base": {
        "f1_optimal": 0.5123,
        "temperature": 0.6457,
        "uncertain_low": 0.34,
        "uncertain_high": 0.54,
    },
    "Ensemble (average)": {
        "f1_optimal": 0.5123,
        "temperature": 0.5621,
        "uncertain_low": 0.34,
        "uncertain_high": 0.54,
    },
    "Ensemble (min)": {
        "f1_optimal": 0.4581,
        "temperature": 0.5579,
        "uncertain_low": 0.52,
        "uncertain_high": 0.57,
    },
}


@dataclass
class CalibratedPrediction:
    """Result from a calibrated ensemble prediction."""

    # Per-model raw probabilities P(AI)
    raw_probs: Dict[str, float]
    # Per-model temperature-scaled probabilities
    calibrated_probs: Dict[str, float]
    # Ensemble probability (average strategy)
    ensemble_avg_prob: float
    # Ensemble probability (min strategy - conservative)
    ensemble_min_prob: float
    # Final verdict: "real", "ai_generated", or "uncertain"
    verdict: str
    # Human-readable verdict
    verdict_text: str
    # Which ensemble strategy was used for verdict
    strategy_used: str
    # UNCERTAIN region boundaries used
    uncertain_low: float
    uncertain_high: float
    # Confidence metrics
    ensemble_std: float
    model_agreement: float  # fraction of models agreeing


class CalibratedCNNEnsemble:
    """
    Calibrated ensemble of 3 custom-trained CNN models.

    Applies temperature scaling and uses empirically-calibrated thresholds
    with UNCERTAIN region for reliable three-way classification.

    Models:
        - ResNet-50 (AUC=85.1%, F1=77.9%)
        - EfficientNetV2-M (AUC=85.7%, F1=78.2%)
        - ConvNeXt-Base (AUC=85.3%, F1=78.1%)

    Ensemble (min) achieves:
        - AUC=85.73%, F1=78.3%
        - Confident accuracy=99.5% (outside UNCERTAIN region)
    """

    MODEL_CONFIGS = [
        {
            "name": "ResNet-50",
            "backbone": "resnet50",
            "weight_key": "resnet50",
            "default_path": "outputs/training_resnet50/best_model.pth",
        },
        {
            "name": "EfficientNetV2-M",
            "backbone": "efficientnet_v2_m",
            "weight_key": "efficientnet",
            "default_path": "outputs/training_efficientnet/best_model.pth",
        },
        {
            "name": "ConvNeXt-Base",
            "backbone": "convnext_base",
            "weight_key": "convnext",
            "default_path": "outputs/training_convnext/best_model.pth",
        },
    ]

    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        thresholds_path: Optional[str] = None,
        device: Optional[str] = None,
        strategy: str = "min",
    ):
        """
        Args:
            model_paths: Dict mapping weight_key to checkpoint path.
                         If None, uses default paths.
            thresholds_path: Path to calibrated_thresholds.json.
                             If None, uses built-in defaults.
            device: "cuda" or "cpu". Auto-detect if None.
            strategy: Ensemble strategy - "min" (conservative) or "average".
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.strategy = strategy
        self.models: Dict[str, Any] = {}
        self.thresholds = dict(_DEFAULT_THRESHOLDS)

        # Load thresholds from JSON if available
        if thresholds_path:
            self._load_thresholds(thresholds_path)
        else:
            self._try_load_default_thresholds()

        # Inference transform (must match training validation transform)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Load models
        self._load_models(model_paths or {})

        n_loaded = len(self.models)
        logger.info(
            f"CalibratedCNNEnsemble initialized: {n_loaded}/3 models loaded, "
            f"strategy={strategy}, device={self.device}"
        )

    def _try_load_default_thresholds(self):
        """Try to load thresholds from standard locations."""
        candidates = [
            Path("configs/calibrated_thresholds.json"),
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "configs"
            / "calibrated_thresholds.json",
        ]
        for p in candidates:
            if p.exists():
                self._load_thresholds(str(p))
                return

    def _load_thresholds(self, path: str):
        """Load calibrated thresholds from JSON."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            models = data.get("models", {})
            for model_name, model_data in models.items():
                thresholds = model_data.get("thresholds", {})
                temp = model_data.get("temperature_scaling", {})
                uncertain = model_data.get("uncertain_region", {})

                self.thresholds[model_name] = {
                    "f1_optimal": thresholds.get(
                        "recommended", thresholds.get("f1_optimal", 0.5)
                    ),
                    "temperature": temp.get("temperature", 1.0),
                    "uncertain_low": uncertain.get("low_threshold", 0.34),
                    "uncertain_high": uncertain.get("high_threshold", 0.54),
                }

            n_samples = data.get("metadata", {}).get("n_samples", "unknown")
            logger.info(
                f"Loaded calibrated thresholds from {path} "
                f"(n={n_samples})"
            )
        except Exception as e:
            logger.warning(f"Failed to load thresholds from {path}: {e}")

    def _load_models(self, custom_paths: Dict[str, str]):
        """Load all 3 CNN models."""
        from imagetrust.detection.models.kaggle_detector import load_kaggle_model

        for cfg in self.MODEL_CONFIGS:
            name = cfg["name"]
            backbone = cfg["backbone"]
            key = cfg["weight_key"]

            # Determine path
            path = custom_paths.get(key) or cfg["default_path"]
            # Also check project root relative paths
            if not Path(path).exists():
                alt = (
                    Path(__file__).resolve().parent.parent.parent.parent.parent
                    / path
                )
                if alt.exists():
                    path = str(alt)

            if not Path(path).exists():
                logger.warning(f"Model weights not found for {name}: {path}")
                continue

            try:
                model = load_kaggle_model(
                    checkpoint_path=path,
                    device=self.device,
                    backbone=backbone,
                )
                self.models[name] = model
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                logger.info(f"Loaded {name} from {path} ({size_mb:.0f} MB)")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")

    def _apply_temperature_scaling(
        self, logit: float, temperature: float
    ) -> float:
        """Apply temperature scaling: calibrated_prob = sigmoid(logit / T)."""
        if temperature <= 0:
            temperature = 1.0
        scaled_logit = logit / temperature
        return 1.0 / (1.0 + np.exp(-scaled_logit))

    def predict(self, image: Image.Image) -> CalibratedPrediction:
        """
        Run calibrated ensemble prediction on a PIL image.

        Returns CalibratedPrediction with per-model and ensemble results.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        raw_probs = {}
        calibrated_probs = {}

        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0]
                ai_raw = probs[0].item()  # P(AI/Fake)

            raw_probs[name] = ai_raw

            # Temperature scaling
            # Convert probability to logit, then scale
            eps = 1e-7
            ai_clipped = np.clip(ai_raw, eps, 1 - eps)
            logit = np.log(ai_clipped / (1 - ai_clipped))

            temp = self.thresholds.get(name, {}).get("temperature", 1.0)
            calibrated_prob = self._apply_temperature_scaling(logit, temp)
            calibrated_probs[name] = calibrated_prob

        if not calibrated_probs:
            return CalibratedPrediction(
                raw_probs={},
                calibrated_probs={},
                ensemble_avg_prob=0.5,
                ensemble_min_prob=0.5,
                verdict="uncertain",
                verdict_text="No models available",
                strategy_used="none",
                uncertain_low=0.34,
                uncertain_high=0.54,
                ensemble_std=0.0,
                model_agreement=0.0,
            )

        # Ensemble strategies
        prob_values = list(calibrated_probs.values())
        ensemble_avg = float(np.mean(prob_values))
        ensemble_min = float(np.min(prob_values))
        ensemble_std = float(np.std(prob_values))

        # Model agreement (fraction voting the same way at 0.5 threshold)
        votes_ai = sum(1 for p in prob_values if p > 0.5)
        votes_real = len(prob_values) - votes_ai
        model_agreement = max(votes_ai, votes_real) / len(prob_values)

        # Select ensemble probability based on strategy
        if self.strategy == "min":
            ensemble_key = "Ensemble (min)"
            ensemble_prob = ensemble_min
        else:
            ensemble_key = "Ensemble (average)"
            ensemble_prob = ensemble_avg

        # Get UNCERTAIN region for the chosen strategy
        ens_thresh = self.thresholds.get(ensemble_key, {})
        low_t = ens_thresh.get("uncertain_low", 0.34)
        high_t = ens_thresh.get("uncertain_high", 0.54)

        # Three-way classification
        if ensemble_prob >= high_t:
            verdict = "ai_generated"
            verdict_text = "AI-Generated"
        elif ensemble_prob < low_t:
            verdict = "real"
            verdict_text = "Real Photograph"
        else:
            verdict = "uncertain"
            verdict_text = "Uncertain"

        return CalibratedPrediction(
            raw_probs=raw_probs,
            calibrated_probs=calibrated_probs,
            ensemble_avg_prob=ensemble_avg,
            ensemble_min_prob=ensemble_min,
            verdict=verdict,
            verdict_text=verdict_text,
            strategy_used=self.strategy,
            uncertain_low=low_t,
            uncertain_high=high_t,
            ensemble_std=ensemble_std,
            model_agreement=model_agreement,
        )

    def predict_batch(
        self, images: List[Image.Image], batch_size: int = 16
    ) -> List[CalibratedPrediction]:
        """Run calibrated prediction on a batch of images."""
        results = []
        for img in images:
            results.append(self.predict(img))
        return results

    def get_info(self) -> Dict[str, Any]:
        """Return ensemble configuration and metadata."""
        return {
            "n_models": len(self.models),
            "models_loaded": list(self.models.keys()),
            "strategy": self.strategy,
            "device": self.device,
            "thresholds": self.thresholds,
            "calibration_dataset_size": 160705,
        }
