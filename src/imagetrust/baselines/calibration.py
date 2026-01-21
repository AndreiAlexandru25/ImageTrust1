"""
Calibration utilities for baseline detectors.

Provides post-hoc calibration methods and evaluation for fair comparison.
Integrates with the main calibration module from imagetrust.detection.calibration.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationResult:
    """Results from calibration evaluation."""

    method: str
    ece_before: float
    ece_after: float
    mce_before: float
    mce_after: float
    brier_before: float
    brier_after: float
    temperature: Optional[float] = None
    reliability_data: Dict[str, Any] = field(default_factory=dict)

    def improvement(self) -> float:
        """Compute ECE improvement (positive = better)."""
        return self.ece_before - self.ece_after

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "mce_before": self.mce_before,
            "mce_after": self.mce_after,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "temperature": self.temperature,
            "ece_improvement": self.improvement(),
        }


class BaselineCalibrator:
    """
    Calibrator for baseline detectors.

    Supports Temperature Scaling, Platt Scaling, and Isotonic Regression.
    Works with numpy arrays directly (no PyTorch dependency required).
    """

    METHODS = ["temperature", "platt", "isotonic", "none"]

    def __init__(self, method: str = "temperature", n_bins: int = 15):
        """
        Initialize calibrator.

        Args:
            method: Calibration method ("temperature", "platt", "isotonic", "none")
            n_bins: Number of bins for ECE computation
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {self.METHODS}")

        self.method = method
        self.n_bins = n_bins
        self._fitted = False

        # Method-specific parameters
        self._temperature = 1.0
        self._platt_model: Optional[LogisticRegression] = None
        self._isotonic_model: Optional[IsotonicRegression] = None

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        logits: Optional[np.ndarray] = None,
    ) -> "BaselineCalibrator":
        """
        Fit calibrator on validation data.

        Args:
            probabilities: Predicted P(AI) values, shape (N,)
            labels: True labels (0=real, 1=AI), shape (N,)
            logits: Optional raw logits for temperature scaling

        Returns:
            self for chaining
        """
        probabilities = np.asarray(probabilities).flatten()
        labels = np.asarray(labels).flatten()

        if self.method == "temperature":
            self._fit_temperature(probabilities, labels, logits)
        elif self.method == "platt":
            self._fit_platt(probabilities, labels)
        elif self.method == "isotonic":
            self._fit_isotonic(probabilities, labels)
        # "none" requires no fitting

        self._fitted = True
        return self

    def _fit_temperature(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        logits: Optional[np.ndarray] = None,
    ) -> None:
        """Fit temperature scaling."""
        if logits is not None and TORCH_AVAILABLE:
            # Use PyTorch optimization if logits available
            self._fit_temperature_torch(logits, labels)
        else:
            # Simple grid search for temperature
            self._fit_temperature_grid(probabilities, labels)

    def _fit_temperature_torch(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """Fit temperature using PyTorch LBFGS."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)

        # Ensure 2D logits
        if logits_t.dim() == 1:
            logits_t = torch.stack([1 - logits_t, logits_t], dim=1)

        temperature = nn.Parameter(torch.tensor([1.5]))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / temperature.clamp(min=0.01)
            loss = criterion(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self._temperature = temperature.item()

    def _fit_temperature_grid(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit temperature via grid search (fallback)."""
        best_ece = float("inf")
        best_temp = 1.0

        # Convert probabilities to logits
        eps = 1e-7
        probs_clipped = np.clip(probabilities, eps, 1 - eps)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        for temp in np.linspace(0.1, 5.0, 50):
            scaled_logits = logits / temp
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            ece = self._compute_ece(scaled_probs, labels)

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self._temperature = best_temp

    def _fit_platt(self, probabilities: np.ndarray, labels: np.ndarray) -> None:
        """Fit Platt scaling (logistic regression)."""
        self._platt_model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self._platt_model.fit(probabilities.reshape(-1, 1), labels)

    def _fit_isotonic(self, probabilities: np.ndarray, labels: np.ndarray) -> None:
        """Fit isotonic regression."""
        self._isotonic_model = IsotonicRegression(out_of_bounds="clip")
        self._isotonic_model.fit(probabilities, labels)

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probabilities.

        Args:
            probabilities: Uncalibrated P(AI) values

        Returns:
            Calibrated probabilities
        """
        probabilities = np.asarray(probabilities).flatten()

        if not self._fitted or self.method == "none":
            return probabilities

        if self.method == "temperature":
            return self._apply_temperature(probabilities)
        elif self.method == "platt":
            return self._apply_platt(probabilities)
        elif self.method == "isotonic":
            return self._apply_isotonic(probabilities)

        return probabilities

    def _apply_temperature(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        eps = 1e-7
        probs_clipped = np.clip(probabilities, eps, 1 - eps)
        logits = np.log(probs_clipped / (1 - probs_clipped))
        scaled_logits = logits / self._temperature
        return 1 / (1 + np.exp(-scaled_logits))

    def _apply_platt(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        if self._platt_model is None:
            return probabilities
        return self._platt_model.predict_proba(probabilities.reshape(-1, 1))[:, 1]

    def _apply_isotonic(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply isotonic regression."""
        if self._isotonic_model is None:
            return probabilities
        return self._isotonic_model.predict(probabilities)

    def _compute_ece(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute Expected Calibration Error."""
        confidences = np.maximum(probabilities, 1 - probabilities)
        predictions = (probabilities > 0.5).astype(int)
        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0

        for i in range(self.n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                acc_in_bin = accuracies[in_bin].mean()
                conf_in_bin = confidences[in_bin].mean()
                ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

        return ece

    def evaluate(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> CalibrationResult:
        """
        Evaluate calibration before and after.

        Args:
            probabilities: Uncalibrated P(AI) values
            labels: True labels

        Returns:
            CalibrationResult with metrics
        """
        probabilities = np.asarray(probabilities).flatten()
        labels = np.asarray(labels).flatten()

        # Before calibration
        ece_before = self._compute_ece(probabilities, labels)
        mce_before = self._compute_mce(probabilities, labels)
        brier_before = self._compute_brier(probabilities, labels)
        reliability_before = self._compute_reliability_diagram(probabilities, labels)

        # After calibration
        calibrated = self.calibrate(probabilities)
        ece_after = self._compute_ece(calibrated, labels)
        mce_after = self._compute_mce(calibrated, labels)
        brier_after = self._compute_brier(calibrated, labels)
        reliability_after = self._compute_reliability_diagram(calibrated, labels)

        return CalibrationResult(
            method=self.method,
            ece_before=ece_before,
            ece_after=ece_after,
            mce_before=mce_before,
            mce_after=mce_after,
            brier_before=brier_before,
            brier_after=brier_after,
            temperature=self._temperature if self.method == "temperature" else None,
            reliability_data={
                "before": reliability_before,
                "after": reliability_after,
            },
        )

    def _compute_mce(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Compute Maximum Calibration Error."""
        confidences = np.maximum(probabilities, 1 - probabilities)
        predictions = (probabilities > 0.5).astype(int)
        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        mce = 0.0

        for i in range(self.n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])

            if in_bin.sum() > 0:
                acc_in_bin = accuracies[in_bin].mean()
                conf_in_bin = confidences[in_bin].mean()
                mce = max(mce, abs(acc_in_bin - conf_in_bin))

        return mce

    def _compute_brier(self, probabilities: np.ndarray, labels: np.ndarray) -> float:
        """Compute Brier score."""
        return np.mean((probabilities - labels) ** 2)

    def _compute_reliability_diagram(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, List[float]]:
        """Compute data for reliability diagram."""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                labels, probabilities, n_bins=self.n_bins, strategy="uniform"
            )
            return {
                "fraction_positives": fraction_of_positives.tolist(),
                "mean_predicted": mean_predicted_value.tolist(),
            }
        except Exception:
            return {"fraction_positives": [], "mean_predicted": []}

    def save(self, path: Union[str, Path]) -> None:
        """Save calibrator parameters."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "method": self.method,
            "n_bins": self.n_bins,
            "fitted": self._fitted,
            "temperature": self._temperature,
        }

        if self._platt_model is not None:
            data["platt_coef"] = self._platt_model.coef_.tolist()
            data["platt_intercept"] = self._platt_model.intercept_.tolist()

        if self._isotonic_model is not None:
            data["isotonic_x"] = self._isotonic_model.X_thresholds_.tolist()
            data["isotonic_y"] = self._isotonic_model.y_thresholds_.tolist()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaselineCalibrator":
        """Load calibrator from file."""
        with open(path, "r") as f:
            data = json.load(f)

        calibrator = cls(method=data["method"], n_bins=data.get("n_bins", 15))
        calibrator._fitted = data.get("fitted", False)
        calibrator._temperature = data.get("temperature", 1.0)

        if "platt_coef" in data:
            calibrator._platt_model = LogisticRegression()
            calibrator._platt_model.coef_ = np.array(data["platt_coef"])
            calibrator._platt_model.intercept_ = np.array(data["platt_intercept"])
            calibrator._platt_model.classes_ = np.array([0, 1])

        if "isotonic_x" in data:
            calibrator._isotonic_model = IsotonicRegression(out_of_bounds="clip")
            calibrator._isotonic_model.X_thresholds_ = np.array(data["isotonic_x"])
            calibrator._isotonic_model.y_thresholds_ = np.array(data["isotonic_y"])
            calibrator._isotonic_model.X_min_ = calibrator._isotonic_model.X_thresholds_[0]
            calibrator._isotonic_model.X_max_ = calibrator._isotonic_model.X_thresholds_[-1]

        return calibrator


def calibrate_baseline(
    baseline,
    val_images: List,
    val_labels: List[int],
    method: str = "temperature",
) -> Tuple[BaselineCalibrator, CalibrationResult]:
    """
    Calibrate a baseline detector using validation data.

    Args:
        baseline: BaselineDetector instance
        val_images: Validation images
        val_labels: Validation labels
        method: Calibration method

    Returns:
        Tuple of (fitted calibrator, calibration results)
    """
    # Get predictions
    probabilities = []
    for img in val_images:
        result = baseline.predict_proba(img)
        probabilities.append(result.ai_probability)

    probabilities = np.array(probabilities)
    labels = np.array(val_labels)

    # Fit calibrator
    calibrator = BaselineCalibrator(method=method)
    calibrator.fit(probabilities, labels)

    # Evaluate
    result = calibrator.evaluate(probabilities, labels)

    return calibrator, result


def compare_calibration_methods(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, CalibrationResult]:
    """
    Compare all calibration methods on the same data.

    Args:
        probabilities: Uncalibrated predictions
        labels: True labels

    Returns:
        Dictionary of method -> CalibrationResult
    """
    results = {}

    for method in ["none", "temperature", "platt", "isotonic"]:
        calibrator = BaselineCalibrator(method=method)

        if method != "none":
            calibrator.fit(probabilities, labels)

        results[method] = calibrator.evaluate(probabilities, labels)

    return results
