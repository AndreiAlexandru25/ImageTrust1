"""
Base class for all baseline detectors.

All baselines must implement the same interface for fair comparison.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import numpy as np
from PIL import Image


@dataclass
class BaselineResult:
    """
    Standardized result from any baseline detector.

    All baselines return this structure for unified evaluation.
    """
    # Core predictions
    ai_probability: float  # P(AI-generated), in [0, 1]
    real_probability: float  # P(Real) = 1 - ai_probability

    # Raw outputs (before calibration)
    raw_logits: Optional[np.ndarray] = None  # Shape: (2,) for [real, ai]
    raw_probability: Optional[float] = None  # Uncalibrated P(AI)

    # Metadata
    baseline_name: str = ""
    processing_time_ms: float = 0.0
    calibrated: bool = False

    # Optional: feature vector for B1
    features: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ai_probability": self.ai_probability,
            "real_probability": self.real_probability,
            "raw_probability": self.raw_probability,
            "baseline_name": self.baseline_name,
            "processing_time_ms": self.processing_time_ms,
            "calibrated": self.calibrated,
        }


@dataclass
class BaselineConfig:
    """
    Configuration for a baseline detector.

    Stores all hyperparameters for reproducibility reporting.
    """
    name: str
    seed: int = 42
    device: str = "cpu"

    # Training params (for neural baselines)
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Model-specific params stored as dict
    model_params: Dict[str, Any] = field(default_factory=dict)

    def to_report_dict(self) -> Dict[str, Any]:
        """
        Generate dict for paper reporting.

        Returns all hyperparameters that should be reported.
        """
        return {
            "name": self.name,
            "seed": self.seed,
            "device": self.device,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            **self.model_params,
        }


class BaselineDetector(ABC):
    """
    Abstract base class for all baseline detectors.

    All baselines must implement:
    - predict_proba(): Single image prediction
    - predict_proba_batch(): Batch prediction
    - fit(): Training (if applicable)
    - save() / load(): Persistence

    Example:
        >>> baseline = ClassicalBaseline(config)
        >>> baseline.fit(train_images, train_labels)
        >>> result = baseline.predict_proba(image)
        >>> print(f"AI probability: {result.ai_probability:.2%}")
    """

    def __init__(self, config: BaselineConfig):
        self.config = config
        self.is_fitted = False
        self._calibrator = None

        # Set random seeds for reproducibility
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)

        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

    @property
    def name(self) -> str:
        """Return baseline name for logging/reporting."""
        return self.config.name

    @abstractmethod
    def predict_proba(self, image: Union[Image.Image, np.ndarray, Path]) -> BaselineResult:
        """
        Predict AI probability for a single image.

        Args:
            image: PIL Image, numpy array (H, W, 3), or path

        Returns:
            BaselineResult with ai_probability in [0, 1]
        """
        pass

    def predict_proba_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, Path]]
    ) -> List[BaselineResult]:
        """
        Predict AI probability for multiple images.

        Default implementation calls predict_proba() in a loop.
        Subclasses may override for efficiency.

        Args:
            images: List of images

        Returns:
            List of BaselineResult
        """
        return [self.predict_proba(img) for img in images]

    @abstractmethod
    def fit(
        self,
        train_images: List[Union[Image.Image, Path]],
        train_labels: List[int],
        val_images: Optional[List[Union[Image.Image, Path]]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Train the baseline on labeled data.

        Args:
            train_images: Training images
            train_labels: Labels (0=real, 1=AI)
            val_images: Optional validation images
            val_labels: Optional validation labels

        Returns:
            Training history/metrics dict
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Load model from disk."""
        pass

    def set_calibrator(self, calibrator: Any) -> None:
        """
        Set a calibrator for post-hoc probability calibration.

        Args:
            calibrator: Fitted calibrator (e.g., TemperatureScaling)
        """
        self._calibrator = calibrator

    def get_config_for_paper(self) -> Dict[str, Any]:
        """
        Get configuration dict for paper reporting.

        Returns dict with all hyperparameters to report.
        """
        return self.config.to_report_dict()

    def _load_image(self, image: Union[Image.Image, np.ndarray, Path]) -> Image.Image:
        """Load and convert image to PIL format."""
        if isinstance(image, Path) or isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _timed_predict(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Wrapper to time prediction."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms
