"""
Ablation study module for systematic component analysis.

Inspired by MLE-STAR methodology for rigorous ablation studies.
Required for academic publication at IEEE WIFS / ACM IH&MMSec.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from imagetrust.evaluation.metrics import compute_metrics, compute_calibration_metrics
from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)


# HuggingFace models for ablation (same as in multi_detector.py)
HUGGINGFACE_MODELS = [
    ("dima806/deepfake_vs_real_image_detection", "Deepfake-vs-Real"),
    ("umm-maybe/AI-image-detector", "AI-Image-Detector"),
    ("Nahrawy/AIorNot", "AIorNot"),
    ("NYUAD-ComNets/NYUAD_AI-generated_images_detector", "NYUAD-2025"),
]


class AblationStudy:
    """
    Ablation study framework inspired by MLE-STAR methodology.

    Systematically analyzes the contribution of each component
    to overall model performance.

    Components analyzed:
    - Backbone architecture (individual HuggingFace models)
    - Ensemble strategy (average, weighted, voting, max, median)
    - Calibration method (none, temperature, platt, isotonic)
    - Preprocessing variations (image size, augmentation)
    - Signal analysis (frequency, noise, texture, edges, color)
    - Model subsets (all 4, top 3, top 2, single best)

    Example:
        >>> ablation = AblationStudy(detector, val_dataset, output_dir="results")
        >>> results = ablation.run_full_study()
        >>> ablation.print_summary()
    """

    def __init__(
        self,
        detector,
        val_dataset: List[Tuple[Any, int]],
        output_dir: Optional[Union[Path, str]] = None,
        verbose: bool = True,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize ablation study.

        Args:
            detector: Main detector (ComprehensiveDetector or compatible)
            val_dataset: List of (image, label) tuples or dataset object
            output_dir: Directory for saving results
            verbose: Print progress information
            device: Device for model inference (cuda/cpu)
        """
        self.detector = detector
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir) if output_dir else Path("results/ablation")
        self.verbose = verbose
        self.device = device or getattr(detector, "device", "cpu")

        self.results: Dict[str, Any] = {}
        self._baseline_metrics: Optional[Dict[str, float]] = None

    def run_full_study(self) -> Dict[str, Any]:
        """
        Run complete ablation study.

        Returns:
            Dictionary with all ablation results
        """
        logger.info("Starting comprehensive ablation study")
        start_time = time.time()

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "num_samples": len(self.val_dataset),
                "device": self.device,
            },
            "baseline": self._evaluate_baseline(),
            "backbone_ablation": self._ablate_backbones(),
            "ensemble_ablation": self._ablate_ensemble_strategies(),
            "calibration_ablation": self._ablate_calibration(),
            "preprocessing_ablation": self._ablate_preprocessing(),
            "signal_analysis_ablation": self._ablate_signal_analysis(),
            "model_subsets_ablation": self._ablate_model_subsets(),
        }

        # Compute component importance
        self.results["component_importance"] = self._compute_importance()

        # Add timing
        self.results["total_time_seconds"] = time.time() - start_time

        logger.info(f"Ablation study completed in {self.results['total_time_seconds']:.1f}s")

        return self.results

    def _evaluate_baseline(self) -> Dict[str, Any]:
        """Evaluate baseline model (full system)."""
        logger.info("Evaluating baseline model (full system)")
        metrics = self._evaluate_detector(self.detector)
        self._baseline_metrics = metrics
        return metrics

    def _evaluate_detector(
        self,
        detector,
        use_calibration: bool = True,
        enable_signals: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a detector on validation data.

        Args:
            detector: Detector instance with detect() or analyze() method
            use_calibration: Whether to use calibration
            enable_signals: Whether to use signal analysis

        Returns:
            Dictionary with evaluation metrics
        """
        labels = []
        preds = []
        probs = []

        for item in self.val_dataset:
            # Handle both (image, label) tuples and dict items
            if isinstance(item, tuple):
                img, label = item
            elif isinstance(item, dict):
                img, label = item["image"], item["label"]
            else:
                continue

            try:
                # Try different detector interfaces
                if hasattr(detector, "analyze"):
                    result = detector.analyze(img)
                    ai_prob = result.get("ai_probability", 0.5)
                elif hasattr(detector, "detect"):
                    result = detector.detect(img, use_calibration=use_calibration)
                    ai_prob = result.get("ai_probability", 0.5)
                elif hasattr(detector, "predict_proba"):
                    result = detector.predict_proba(img)
                    ai_prob = getattr(result, "ai_probability", result)
                else:
                    continue

                labels.append(label)
                probs.append(ai_prob)
                preds.append(1 if ai_prob > 0.5 else 0)
            except Exception as e:
                logger.warning(f"Evaluation failed for sample: {e}")

        if not labels:
            return {"error": "No samples evaluated", "num_samples": 0}

        # Compute metrics
        metrics = compute_metrics(
            np.array(labels),
            np.array(preds),
            np.array(probs),
        )

        # Add calibration metrics
        cal_metrics = compute_calibration_metrics(np.array(labels), np.array(probs))
        metrics.update(cal_metrics)

        metrics["num_samples"] = len(labels)

        return metrics

    def _ablate_backbones(self) -> Dict[str, Any]:
        """
        Ablate: test each HuggingFace model individually.

        This shows the contribution of each backbone to the ensemble.
        """
        logger.info("Ablating backbone architectures (individual models)")

        results = {}

        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
        except ImportError:
            logger.warning("transformers not available for backbone ablation")
            return {"error": "transformers library not available"}

        for model_id, name in HUGGINGFACE_MODELS:
            logger.info(f"  Testing: {name}")
            try:
                # Create single-model detector
                single_detector = SingleModelDetector(
                    model_id=model_id,
                    name=name,
                    device=self.device,
                )
                metrics = self._evaluate_detector(single_detector)
                metrics["model_id"] = model_id
                results[name] = metrics
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                results[name] = {"error": str(e), "model_id": model_id}

        # Rank by F1 score
        ranked = sorted(
            [(k, v.get("f1_score", 0)) for k, v in results.items() if "error" not in v],
            key=lambda x: -x[1],
        )
        results["ranking"] = [{"model": k, "f1": f1} for k, f1 in ranked]

        return results

    def _ablate_ensemble_strategies(self) -> Dict[str, Any]:
        """
        Ablate: test different ensemble voting strategies.

        Strategies: average, weighted, voting, max, median
        """
        logger.info("Ablating ensemble strategies")

        from imagetrust.detection.ensemble_strategies import (
            create_ensemble_strategy,
            get_available_strategies,
        )

        results = {}
        strategies = get_available_strategies()

        # First, collect all individual model predictions
        all_predictions = self._collect_model_predictions()

        if "error" in all_predictions:
            return all_predictions

        for strategy_name in strategies:
            logger.info(f"  Testing strategy: {strategy_name}")
            try:
                strategy = create_ensemble_strategy(strategy_name)
                metrics = self._evaluate_ensemble_strategy(
                    all_predictions,
                    strategy,
                )
                results[strategy_name] = metrics
            except Exception as e:
                logger.warning(f"Failed to evaluate strategy {strategy_name}: {e}")
                results[strategy_name] = {"error": str(e)}

        # Find best strategy
        best_strategy = max(
            [(k, v.get("f1_score", 0)) for k, v in results.items() if "error" not in v],
            key=lambda x: x[1],
            default=("unknown", 0),
        )
        results["best_strategy"] = best_strategy[0]

        return results

    def _collect_model_predictions(self) -> Dict[str, Any]:
        """Collect predictions from all models for ensemble ablation."""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
        except ImportError:
            return {"error": "transformers library not available"}

        # Load all models
        models = {}
        for model_id, name in HUGGINGFACE_MODELS:
            try:
                processor = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModelForImageClassification.from_pretrained(model_id)
                model.to(self.device)
                model.eval()
                models[name] = {
                    "model": model,
                    "processor": processor,
                    "id2label": model.config.id2label,
                }
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")

        if not models:
            return {"error": "No models loaded"}

        # Collect predictions
        predictions = {name: [] for name in models.keys()}
        labels = []
        confidences = {name: [] for name in models.keys()}

        for item in self.val_dataset:
            if isinstance(item, tuple):
                img, label = item
            elif isinstance(item, dict):
                img, label = item["image"], item["label"]
            else:
                continue

            labels.append(label)

            for name, model_info in models.items():
                try:
                    with torch.no_grad():
                        inputs = model_info["processor"](images=img, return_tensors="pt")
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        outputs = model_info["model"](**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)[0]

                    ai_idx = self._find_ai_index(model_info["id2label"])
                    ai_prob = probs[ai_idx].item()
                    confidence = probs.max().item()

                    predictions[name].append(ai_prob)
                    confidences[name].append(confidence)
                except Exception:
                    predictions[name].append(0.5)
                    confidences[name].append(0.5)

        return {
            "predictions": predictions,
            "labels": np.array(labels),
            "confidences": confidences,
            "model_names": list(models.keys()),
        }

    def _find_ai_index(self, id2label: dict) -> int:
        """Find which index corresponds to AI-generated."""
        for idx, label in id2label.items():
            label_lower = label.lower()
            if any(kw in label_lower for kw in ["ai", "artificial", "generated", "fake", "synthetic"]):
                return int(idx)
        return 1

    def _evaluate_ensemble_strategy(
        self,
        all_predictions: Dict[str, Any],
        strategy,
    ) -> Dict[str, Any]:
        """Evaluate a specific ensemble strategy."""
        predictions = all_predictions["predictions"]
        labels = all_predictions["labels"]
        model_names = all_predictions["model_names"]
        confidences = all_predictions["confidences"]

        combined_probs = []
        n_samples = len(labels)

        # Default weights for weighted strategies
        default_weights = [0.4, 0.3, 0.2, 0.1][:len(model_names)]

        for i in range(n_samples):
            probs = [predictions[name][i] for name in model_names]
            confs = [confidences[name][i] for name in model_names]

            result = strategy.combine(
                probabilities=probs,
                weights=default_weights,
                confidences=confs,
            )
            combined_probs.append(result.combined_probability)

        combined_probs = np.array(combined_probs)
        preds = (combined_probs > 0.5).astype(int)

        metrics = compute_metrics(labels, preds, combined_probs)
        cal_metrics = compute_calibration_metrics(labels, combined_probs)
        metrics.update(cal_metrics)

        return metrics

    def _ablate_calibration(self) -> Dict[str, Any]:
        """
        Ablate: test different calibration methods.

        Methods: none, temperature, platt, isotonic
        """
        logger.info("Ablating calibration methods")

        results = {
            "with_calibration": self._evaluate_detector(self.detector, use_calibration=True),
            "without_calibration": self._evaluate_detector(self.detector, use_calibration=False),
        }

        # Additional calibration methods
        calibration_methods = ["temperature", "platt", "isotonic"]

        # Collect uncalibrated predictions first
        uncal_probs = []
        labels = []

        for item in self.val_dataset:
            if isinstance(item, tuple):
                img, label = item
            elif isinstance(item, dict):
                img, label = item["image"], item["label"]
            else:
                continue

            try:
                if hasattr(self.detector, "analyze"):
                    result = self.detector.analyze(img)
                    ai_prob = result.get("ai_probability", 0.5)
                elif hasattr(self.detector, "detect"):
                    result = self.detector.detect(img, use_calibration=False)
                    ai_prob = result.get("ai_probability", 0.5)
                else:
                    continue

                uncal_probs.append(ai_prob)
                labels.append(label)
            except Exception:
                pass

        if len(uncal_probs) < 10:
            results["error"] = "Not enough samples for calibration comparison"
            return results

        uncal_probs = np.array(uncal_probs)
        labels = np.array(labels)

        # Test different calibration methods
        for method in calibration_methods:
            try:
                cal_probs = self._apply_calibration(uncal_probs, labels, method)
                preds = (cal_probs > 0.5).astype(int)

                metrics = compute_metrics(labels, preds, cal_probs)
                cal_metrics = compute_calibration_metrics(labels, cal_probs)
                metrics.update(cal_metrics)
                results[method] = metrics
            except Exception as e:
                logger.warning(f"Calibration method {method} failed: {e}")
                results[method] = {"error": str(e)}

        # Compare ECE before/after
        if "without_calibration" in results and "error" not in results["without_calibration"]:
            ece_before = results["without_calibration"].get("ece", 0)
            for method in calibration_methods:
                if method in results and "error" not in results[method]:
                    results[method]["ece_improvement"] = ece_before - results[method].get("ece", 0)

        return results

    def _apply_calibration(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        method: str,
    ) -> np.ndarray:
        """Apply calibration method to probabilities."""
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        # Split for calibration (70/30)
        split_idx = int(len(probs) * 0.7)
        train_probs, val_probs = probs[:split_idx], probs[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]

        if method == "temperature":
            # Temperature scaling
            from scipy.optimize import minimize_scalar

            def temp_objective(T):
                scaled = 1 / (1 + np.exp(-np.log(train_probs / (1 - train_probs + 1e-10) + 1e-10) / T))
                return np.mean((scaled - train_labels) ** 2)

            result = minimize_scalar(temp_objective, bounds=(0.1, 10), method="bounded")
            T = result.x

            calibrated = 1 / (1 + np.exp(-np.log(probs / (1 - probs + 1e-10) + 1e-10) / T))
            return calibrated

        elif method == "platt":
            # Platt scaling (logistic regression)
            lr = LogisticRegression()
            lr.fit(train_probs.reshape(-1, 1), train_labels)
            calibrated = lr.predict_proba(probs.reshape(-1, 1))[:, 1]
            return calibrated

        elif method == "isotonic":
            # Isotonic regression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(train_probs, train_labels)
            calibrated = iso.predict(probs)
            return calibrated

        return probs

    def _ablate_preprocessing(self) -> Dict[str, Any]:
        """
        Ablate: test different preprocessing configurations.

        Configs: input size (224, 384, 512), test-time augmentation
        """
        logger.info("Ablating preprocessing configurations")

        configs = [
            {"size": 224, "augment": False, "name": "size_224"},
            {"size": 384, "augment": False, "name": "size_384"},
            {"size": 512, "augment": False, "name": "size_512"},
            {"size": 224, "augment": True, "name": "size_224_tta"},
        ]

        results = {}

        for config in configs:
            logger.info(f"  Testing: {config['name']}")
            try:
                metrics = self._evaluate_with_preprocessing(
                    size=config["size"],
                    augment=config["augment"],
                )
                results[config["name"]] = metrics
            except Exception as e:
                logger.warning(f"Preprocessing config {config['name']} failed: {e}")
                results[config["name"]] = {"error": str(e)}

        return results

    def _evaluate_with_preprocessing(
        self,
        size: int,
        augment: bool,
    ) -> Dict[str, Any]:
        """Evaluate with specific preprocessing config."""
        labels = []
        probs = []
        preds = []

        for item in self.val_dataset:
            if isinstance(item, tuple):
                img, label = item
            elif isinstance(item, dict):
                img, label = item["image"], item["label"]
            else:
                continue

            try:
                # Resize image
                if isinstance(img, Image.Image):
                    resized = img.resize((size, size), Image.Resampling.LANCZOS)
                else:
                    resized = img

                # Test-time augmentation
                if augment:
                    ai_prob = self._tta_inference(resized)
                else:
                    if hasattr(self.detector, "analyze"):
                        result = self.detector.analyze(resized)
                    else:
                        result = self.detector.detect(resized)
                    ai_prob = result.get("ai_probability", 0.5)

                labels.append(label)
                probs.append(ai_prob)
                preds.append(1 if ai_prob > 0.5 else 0)
            except Exception:
                pass

        if not labels:
            return {"error": "No samples evaluated"}

        metrics = compute_metrics(np.array(labels), np.array(preds), np.array(probs))
        cal_metrics = compute_calibration_metrics(np.array(labels), np.array(probs))
        metrics.update(cal_metrics)
        metrics["num_samples"] = len(labels)

        return metrics

    def _tta_inference(self, image: Image.Image) -> float:
        """Test-time augmentation inference."""
        augmented_probs = []

        # Original
        if hasattr(self.detector, "analyze"):
            result = self.detector.analyze(image)
        else:
            result = self.detector.detect(image)
        augmented_probs.append(result.get("ai_probability", 0.5))

        # Horizontal flip
        flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if hasattr(self.detector, "analyze"):
            result = self.detector.analyze(flipped)
        else:
            result = self.detector.detect(flipped)
        augmented_probs.append(result.get("ai_probability", 0.5))

        # Average
        return float(np.mean(augmented_probs))

    def _ablate_signal_analysis(self) -> Dict[str, Any]:
        """
        Ablate: test contribution of signal analysis components.

        Configs:
        - full: ML + all signals
        - ml_only: only ML models (no signal analysis)
        - no_frequency: without FFT analysis
        - no_noise: without noise analysis
        """
        logger.info("Ablating signal analysis components")

        results = {}

        configs = [
            ("full", {"enable_frequency": True, "enable_noise": True}),
            ("ml_only", {"enable_frequency": False, "enable_noise": False}),
            ("no_frequency", {"enable_frequency": False, "enable_noise": True}),
            ("no_noise", {"enable_frequency": True, "enable_noise": False}),
        ]

        for config_name, config in configs:
            logger.info(f"  Testing: {config_name}")
            try:
                metrics = self._evaluate_with_signal_config(config)
                results[config_name] = metrics
            except Exception as e:
                logger.warning(f"Signal config {config_name} failed: {e}")
                results[config_name] = {"error": str(e)}

        # Compute contribution of each signal component
        if all(c in results and "error" not in results[c] for c in ["full", "ml_only"]):
            full_f1 = results["full"].get("f1_score", 0)
            ml_only_f1 = results["ml_only"].get("f1_score", 0)
            results["signal_contribution"] = full_f1 - ml_only_f1

        return results

    def _evaluate_with_signal_config(self, config: Dict[str, bool]) -> Dict[str, Any]:
        """Evaluate with specific signal analysis configuration."""
        labels = []
        probs = []
        preds = []

        enable_freq = config.get("enable_frequency", True)
        enable_noise = config.get("enable_noise", True)

        for item in self.val_dataset:
            if isinstance(item, tuple):
                img, label = item
            elif isinstance(item, dict):
                img, label = item["image"], item["label"]
            else:
                continue

            try:
                # Get analysis with filtered results
                if hasattr(self.detector, "analyze"):
                    result = self.detector.analyze(img)

                    # Recalculate with filtered components
                    individual = result.get("individual_results", [])
                    filtered = []

                    for r in individual:
                        method = r.get("method", "")
                        if "Frequency" in method and not enable_freq:
                            continue
                        if "Noise" in method and not enable_noise:
                            continue
                        filtered.append(r)

                    if filtered:
                        total_weight = sum(
                            r["weight"] * r["confidence"] for r in filtered
                        )
                        ai_prob = sum(
                            r["ai_probability"] * r["weight"] * r["confidence"]
                            for r in filtered
                        ) / total_weight
                    else:
                        ai_prob = 0.5
                else:
                    result = self.detector.detect(img)
                    ai_prob = result.get("ai_probability", 0.5)

                labels.append(label)
                probs.append(ai_prob)
                preds.append(1 if ai_prob > 0.5 else 0)
            except Exception:
                pass

        if not labels:
            return {"error": "No samples evaluated"}

        metrics = compute_metrics(np.array(labels), np.array(preds), np.array(probs))
        cal_metrics = compute_calibration_metrics(np.array(labels), np.array(probs))
        metrics.update(cal_metrics)

        return metrics

    def _ablate_model_subsets(self) -> Dict[str, Any]:
        """
        Ablate: test different combinations of ML models.

        Subsets: all 4, top 3, top 2, single best
        """
        logger.info("Ablating model subsets")

        results = {}

        # First get ranking from backbone ablation
        if "backbone_ablation" not in self.results:
            self.results["backbone_ablation"] = self._ablate_backbones()

        backbone_results = self.results.get("backbone_ablation", {})
        ranking = backbone_results.get("ranking", [])

        if not ranking:
            return {"error": "No backbone ranking available"}

        # Test different subsets
        subsets = [
            ("all_4", [r["model"] for r in ranking[:4]]),
            ("top_3", [r["model"] for r in ranking[:3]]),
            ("top_2", [r["model"] for r in ranking[:2]]),
            ("single_best", [r["model"] for r in ranking[:1]]),
        ]

        all_predictions = self._collect_model_predictions()

        if "error" in all_predictions:
            return all_predictions

        from imagetrust.detection.ensemble_strategies import create_ensemble_strategy

        for subset_name, model_names in subsets:
            logger.info(f"  Testing subset: {subset_name} ({', '.join(model_names)})")
            try:
                # Filter predictions to subset
                filtered_preds = {
                    k: v for k, v in all_predictions["predictions"].items()
                    if k in model_names
                }
                filtered_confs = {
                    k: v for k, v in all_predictions["confidences"].items()
                    if k in model_names
                }

                if not filtered_preds:
                    results[subset_name] = {"error": "No models in subset"}
                    continue

                # Combine using weighted average
                labels = all_predictions["labels"]
                n_samples = len(labels)
                combined_probs = []

                strategy = create_ensemble_strategy("weighted")

                for i in range(n_samples):
                    probs_list = [filtered_preds[name][i] for name in filtered_preds.keys()]
                    confs_list = [filtered_confs[name][i] for name in filtered_confs.keys()]

                    result = strategy.combine(probs_list, confidences=confs_list)
                    combined_probs.append(result.combined_probability)

                combined_probs = np.array(combined_probs)
                preds = (combined_probs > 0.5).astype(int)

                metrics = compute_metrics(labels, preds, combined_probs)
                cal_metrics = compute_calibration_metrics(labels, combined_probs)
                metrics.update(cal_metrics)
                metrics["models_used"] = list(model_names)

                results[subset_name] = metrics
            except Exception as e:
                logger.warning(f"Subset {subset_name} failed: {e}")
                results[subset_name] = {"error": str(e)}

        return results

    def _compute_importance(self) -> Dict[str, float]:
        """
        Compute relative importance of each component.

        Uses performance drop when component is removed/modified.
        """
        importance = {}

        baseline_f1 = self._baseline_metrics.get("f1_score", 0) if self._baseline_metrics else 0

        # Calibration importance
        cal = self.results.get("calibration_ablation", {})
        with_cal_ece = cal.get("with_calibration", {}).get("ece", 1.0)
        without_cal_ece = cal.get("without_calibration", {}).get("ece", 1.0)
        importance["calibration"] = max(0, without_cal_ece - with_cal_ece)

        # Signal analysis importance
        signal = self.results.get("signal_analysis_ablation", {})
        signal_contrib = signal.get("signal_contribution", 0)
        importance["signal_analysis"] = max(0, signal_contrib)

        # Backbone variance
        backbone_results = self.results.get("backbone_ablation", {})
        backbone_f1s = [
            r.get("f1_score", 0)
            for k, r in backbone_results.items()
            if isinstance(r, dict) and "f1_score" in r
        ]
        if backbone_f1s:
            importance["backbone_selection"] = float(np.std(backbone_f1s))
            importance["backbone_best_vs_worst"] = max(backbone_f1s) - min(backbone_f1s)

        # Ensemble strategy importance
        ensemble = self.results.get("ensemble_ablation", {})
        ensemble_f1s = [
            r.get("f1_score", 0)
            for k, r in ensemble.items()
            if isinstance(r, dict) and "f1_score" in r
        ]
        if ensemble_f1s:
            importance["ensemble_strategy"] = float(np.std(ensemble_f1s))

        # Model subset importance
        subsets = self.results.get("model_subsets_ablation", {})
        all_4_f1 = subsets.get("all_4", {}).get("f1_score", 0)
        single_best_f1 = subsets.get("single_best", {}).get("f1_score", 0)
        importance["ensemble_benefit"] = all_4_f1 - single_best_f1

        return importance

    def save_results(
        self,
        results: Optional[Dict[str, Any]] = None,
        filename: str = "ablation_results.json",
    ) -> Path:
        """Save results to file."""
        results = results or self.results

        ensure_dir(self.output_dir)
        output_path = self.output_dir / filename

        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=convert)

        logger.info(f"Ablation results saved to {output_path}")
        return output_path

    def print_summary(self) -> None:
        """Print ablation study summary."""
        print("\n" + "=" * 70)
        print("ABLATION STUDY SUMMARY (MLE-STAR Methodology)")
        print("=" * 70)

        # Baseline
        baseline = self.results.get("baseline", {})
        print(f"\n1. BASELINE PERFORMANCE (Full System)")
        print("-" * 50)
        print(f"  Accuracy:     {baseline.get('accuracy', 0):.2%}")
        print(f"  F1 Score:     {baseline.get('f1_score', 0):.2%}")
        print(f"  ROC-AUC:      {baseline.get('roc_auc', 0):.3f}")
        print(f"  ECE:          {baseline.get('ece', 0):.4f}")

        # Backbone ranking
        backbone = self.results.get("backbone_ablation", {})
        ranking = backbone.get("ranking", [])
        if ranking:
            print(f"\n2. BACKBONE RANKING (Individual Models)")
            print("-" * 50)
            for i, item in enumerate(ranking, 1):
                print(f"  #{i} {item['model']}: F1={item['f1']:.2%}")

        # Ensemble strategies
        ensemble = self.results.get("ensemble_ablation", {})
        best_strategy = ensemble.get("best_strategy", "unknown")
        print(f"\n3. ENSEMBLE STRATEGIES")
        print("-" * 50)
        print(f"  Best strategy: {best_strategy}")
        for name, metrics in ensemble.items():
            if isinstance(metrics, dict) and "f1_score" in metrics:
                print(f"  {name}: F1={metrics['f1_score']:.2%}, ECE={metrics.get('ece', 0):.4f}")

        # Calibration
        cal = self.results.get("calibration_ablation", {})
        print(f"\n4. CALIBRATION IMPACT")
        print("-" * 50)
        with_cal = cal.get("with_calibration", {})
        without_cal = cal.get("without_calibration", {})
        print(f"  With calibration:    ECE={with_cal.get('ece', 0):.4f}")
        print(f"  Without calibration: ECE={without_cal.get('ece', 0):.4f}")
        for method in ["temperature", "platt", "isotonic"]:
            if method in cal and isinstance(cal[method], dict) and "ece" in cal[method]:
                improvement = cal[method].get("ece_improvement", 0)
                print(f"  {method.capitalize()}: ECE={cal[method]['ece']:.4f} (Δ={improvement:+.4f})")

        # Signal analysis
        signal = self.results.get("signal_analysis_ablation", {})
        print(f"\n5. SIGNAL ANALYSIS CONTRIBUTION")
        print("-" * 50)
        for name, metrics in signal.items():
            if isinstance(metrics, dict) and "f1_score" in metrics:
                print(f"  {name}: F1={metrics['f1_score']:.2%}")
        contrib = signal.get("signal_contribution", 0)
        print(f"  Signal contribution: ΔF1={contrib:+.2%}")

        # Model subsets
        subsets = self.results.get("model_subsets_ablation", {})
        print(f"\n6. MODEL SUBSET ANALYSIS")
        print("-" * 50)
        for name, metrics in subsets.items():
            if isinstance(metrics, dict) and "f1_score" in metrics:
                n_models = len(metrics.get("models_used", []))
                print(f"  {name} ({n_models} models): F1={metrics['f1_score']:.2%}")

        # Component importance
        importance = self.results.get("component_importance", {})
        if importance:
            print(f"\n7. COMPONENT IMPORTANCE")
            print("-" * 50)
            for component, score in sorted(importance.items(), key=lambda x: -abs(x[1])):
                print(f"  {component}: {score:.4f}")

        print("\n" + "=" * 70)
        print(f"Total study time: {self.results.get('total_time_seconds', 0):.1f}s")
        print("=" * 70 + "\n")

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper (Table 4: Ablation Results)."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Ablation Study Results}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Configuration & Acc & F1 & AUC & ECE \\",
            r"\midrule",
        ]

        # Baseline
        baseline = self.results.get("baseline", {})
        lines.append(
            f"Full System (Baseline) & {baseline.get('accuracy', 0):.1%} & "
            f"{baseline.get('f1_score', 0):.1%} & {baseline.get('roc_auc', 0):.3f} & "
            f"{baseline.get('ece', 0):.3f} \\\\"
        )

        lines.append(r"\midrule")

        # Best individual model
        backbone = self.results.get("backbone_ablation", {})
        ranking = backbone.get("ranking", [])
        if ranking:
            best_model = ranking[0]["model"]
            best_metrics = backbone.get(best_model, {})
            lines.append(
                f"Best Single Model ({best_model[:15]}) & {best_metrics.get('accuracy', 0):.1%} & "
                f"{best_metrics.get('f1_score', 0):.1%} & {best_metrics.get('roc_auc', 0):.3f} & "
                f"{best_metrics.get('ece', 0):.3f} \\\\"
            )

        # Without calibration
        cal = self.results.get("calibration_ablation", {})
        without_cal = cal.get("without_calibration", {})
        lines.append(
            f"Without Calibration & {without_cal.get('accuracy', 0):.1%} & "
            f"{without_cal.get('f1_score', 0):.1%} & {without_cal.get('roc_auc', 0):.3f} & "
            f"{without_cal.get('ece', 0):.3f} \\\\"
        )

        # ML only
        signal = self.results.get("signal_analysis_ablation", {})
        ml_only = signal.get("ml_only", {})
        lines.append(
            f"ML Models Only & {ml_only.get('accuracy', 0):.1%} & "
            f"{ml_only.get('f1_score', 0):.1%} & {ml_only.get('roc_auc', 0):.3f} & "
            f"{ml_only.get('ece', 0):.3f} \\\\"
        )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)


class SingleModelDetector:
    """Wrapper for single HuggingFace model (for backbone ablation)."""

    def __init__(self, model_id: str, name: str, device: str = "cpu"):
        self.model_id = model_id
        self.name = name
        self.device = device
        self._load_model()

    def _load_model(self):
        """Load the model."""
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        import torch

        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def _find_ai_index(self) -> int:
        """Find AI index in labels."""
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if any(kw in label_lower for kw in ["ai", "artificial", "generated", "fake", "synthetic"]):
                return int(idx)
        return 1

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image."""
        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        ai_idx = self._find_ai_index()
        ai_prob = probs[ai_idx].item()

        return {
            "ai_probability": ai_prob,
            "real_probability": 1 - ai_prob,
            "model": self.name,
        }

    def detect(self, image: Image.Image, use_calibration: bool = True) -> Dict[str, Any]:
        """Alias for analyze."""
        return self.analyze(image)
