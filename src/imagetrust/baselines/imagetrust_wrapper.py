"""
ImageTrust Wrapper for Baseline Comparison.

Wraps the main ImageTrust pipeline (ComprehensiveDetector) to match
the BaselineDetector interface for fair comparison.

This is YOUR METHOD in the comparison tables.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from imagetrust.baselines.base import BaselineDetector, BaselineConfig, BaselineResult


class ImageTrustWrapper(BaselineDetector):
    """
    Wrapper for ImageTrust main pipeline.

    Wraps ComprehensiveDetector (4 HuggingFace models + signal analysis)
    to match the baseline interface.

    For the paper, report:
    - Number of models in ensemble (4 HF + signal analysis)
    - Model names: umm-maybe/AI-image-detector, Organika/sdxl-detector,
                   aiornot/aiornot-detector-v2, nyuad/ai-image-detector-2025
    - Ensemble strategy: weighted voting
    - Signal analysis: FFT, texture, noise, edge, color
    - Calibration method used (if any)

    Example:
        >>> wrapper = ImageTrustWrapper()
        >>> result = wrapper.predict_proba(image)
        >>> print(f"AI probability: {result.ai_probability:.2%}")
    """

    def __init__(
        self,
        config: Optional[BaselineConfig] = None,
        use_signal_analysis: bool = True,
        auto_calibration: bool = True,
    ):
        """
        Initialize ImageTrust wrapper.

        Args:
            config: Optional baseline configuration
            use_signal_analysis: Include signal analysis (FFT, noise, etc.)
            auto_calibration: Apply auto-calibration for compressed images
        """
        if config is None:
            config = BaselineConfig(
                name="ImageTrust (Ours)",
                seed=42,
            )

        super().__init__(config)

        self.use_signal_analysis = use_signal_analysis
        self.auto_calibration = auto_calibration

        # Lazy initialization
        self._detector = None

        # Store config for paper reporting
        self.config.model_params.update({
            "num_models": 4,
            "models": [
                "umm-maybe/AI-image-detector",
                "Organika/sdxl-detector",
                "aiornot/aiornot-detector-v2",
                "nyuad/ai-image-detector-2025",
            ],
            "signal_analysis": use_signal_analysis,
            "auto_calibration": auto_calibration,
            "ensemble_strategy": "weighted_voting",
        })

        # Mark as fitted (no training needed - uses pretrained models)
        self.is_fitted = True

    def _init_detector(self) -> None:
        """Initialize the ComprehensiveDetector."""
        from imagetrust.detection.multi_detector import ComprehensiveDetector

        self._detector = ComprehensiveDetector()

    def fit(
        self,
        train_images: List[Union[Image.Image, Path]],
        train_labels: List[int],
        val_images: Optional[List[Union[Image.Image, Path]]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        ImageTrust uses pretrained models - no training needed.

        This method exists for interface compatibility.
        Returns empty history since no training occurs.
        """
        # Initialize detector if not already done
        if self._detector is None:
            self._init_detector()

        return {
            "note": "ImageTrust uses pretrained HuggingFace models - no training performed",
            "train_samples": len(train_images),
        }

    def predict_proba(self, image: Union[Image.Image, np.ndarray, Path]) -> BaselineResult:
        """
        Predict AI probability using ImageTrust pipeline.

        Args:
            image: Input image

        Returns:
            BaselineResult with predictions
        """
        if self._detector is None:
            self._init_detector()

        pil_img = self._load_image(image)

        def _predict():
            # Run full analysis
            result = self._detector.analyze(pil_img)

            # Extract probabilities from individual results
            individual_results = result.get("individual_results", [])

            # Compute combined score (matching frontend logic)
            ml_probs = []
            signal_probs = []

            for r in individual_results:
                method = r.get("method", "")
                prob = r.get("ai_probability", 0.5)

                if method.startswith("ML:"):
                    ml_probs.append(prob)
                elif self.use_signal_analysis:
                    signal_probs.append(prob)

            # Weighted combination
            if ml_probs:
                # Weight by model (NYUAD gets higher weight)
                weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights for fair comparison
                if len(ml_probs) == len(weights):
                    ml_avg = sum(p * w for p, w in zip(ml_probs, weights))
                else:
                    ml_avg = np.mean(ml_probs)
            else:
                ml_avg = 0.5

            if signal_probs and self.use_signal_analysis:
                signal_avg = np.mean(signal_probs)
                # Combine: 80% ML, 20% signal
                combined = ml_avg * 0.8 + signal_avg * 0.2
            else:
                combined = ml_avg

            return combined, result

        (ai_prob, full_result), elapsed_ms = self._timed_predict(_predict)

        # Clamp probability
        ai_prob = float(np.clip(ai_prob, 0.0, 1.0))
        raw_prob = ai_prob

        # Apply calibration if available
        if self._calibrator is not None:
            ai_prob = self._calibrator.calibrate(ai_prob)

        return BaselineResult(
            ai_probability=ai_prob,
            real_probability=1 - ai_prob,
            raw_probability=raw_prob,
            baseline_name=self.name,
            processing_time_ms=elapsed_ms,
            calibrated=self._calibrator is not None,
        )

    def predict_proba_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, Path]],
    ) -> List[BaselineResult]:
        """
        Batch prediction.

        Note: ImageTrust processes images sequentially (HuggingFace models
        don't easily batch different images). For fair timing comparison,
        we time each image individually.
        """
        results = []
        for img in images:
            results.append(self.predict_proba(img))
        return results

    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration (no model weights to save - uses HuggingFace).
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_data = {
            "name": self.name,
            "config": self.get_config_for_paper(),
            "use_signal_analysis": self.use_signal_analysis,
            "auto_calibration": self.auto_calibration,
        }

        with open(path.with_suffix(".json"), "w") as f:
            json.dump(config_data, f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """
        Load configuration.
        """
        import json

        config_path = Path(path).with_suffix(".json")
        if config_path.exists():
            with open(config_path, "r") as f:
                data = json.load(f)

            self.use_signal_analysis = data.get("use_signal_analysis", True)
            self.auto_calibration = data.get("auto_calibration", True)

        # Re-initialize detector
        self._init_detector()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information for the paper.
        """
        return {
            "name": "ImageTrust",
            "type": "Ensemble (4 HF models + signal analysis)",
            "models": [
                {
                    "name": "umm-maybe/AI-image-detector",
                    "task": "Deepfake vs Real",
                    "weight": 0.25,
                },
                {
                    "name": "Organika/sdxl-detector",
                    "task": "SDXL detection",
                    "weight": 0.25,
                },
                {
                    "name": "aiornot/aiornot-detector-v2",
                    "task": "General AI detection",
                    "weight": 0.25,
                },
                {
                    "name": "nyuad/ai-image-detector-2025",
                    "task": "Modern AI detection",
                    "weight": 0.25,
                },
            ],
            "signal_analysis": {
                "enabled": self.use_signal_analysis,
                "methods": ["FFT", "Texture", "Noise", "Edge", "Color"],
                "weight": 0.2 if self.use_signal_analysis else 0,
            },
            "ensemble_strategy": "weighted_average",
            "pretrained": True,
            "trainable": False,
        }
