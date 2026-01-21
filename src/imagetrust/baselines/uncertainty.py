"""
Uncertainty estimation and selective prediction (abstain) module.

Provides methods to:
1. Quantify prediction uncertainty
2. Decide when to abstain (UNCERTAIN verdict)
3. Analyze coverage-accuracy trade-offs
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


class UncertaintyMethod(Enum):
    """Available uncertainty estimation methods."""

    ENTROPY = "entropy"              # Shannon entropy of probabilities
    MARGIN = "margin"                # Difference between top-2 probabilities
    CONFIDENCE = "confidence"        # 1 - max(probability)
    CALIBRATED = "calibrated"        # Post-calibration uncertainty
    ENSEMBLE_VARIANCE = "ensemble"   # Variance across ensemble members


@dataclass
class UncertaintyResult:
    """Result of uncertainty estimation for a single prediction."""

    ai_probability: float
    uncertainty: float
    uncertainty_method: str
    should_abstain: bool
    abstain_threshold: float
    confidence_level: str  # "very_low", "low", "medium", "high", "very_high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ai_probability": self.ai_probability,
            "uncertainty": self.uncertainty,
            "uncertainty_method": self.uncertainty_method,
            "should_abstain": self.should_abstain,
            "abstain_threshold": self.abstain_threshold,
            "confidence_level": self.confidence_level,
        }


@dataclass
class SelectivePredictionResult:
    """Results from selective prediction analysis."""

    coverage: float  # Fraction of samples not abstained
    accuracy_on_covered: float  # Accuracy on non-abstained samples
    abstain_rate: float  # Fraction of samples abstained
    threshold: float

    # Detailed breakdown
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    abstained_count: int = 0

    # Performance at different coverage levels
    coverage_accuracy_curve: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage": self.coverage,
            "accuracy_on_covered": self.accuracy_on_covered,
            "abstain_rate": self.abstain_rate,
            "threshold": self.threshold,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "abstained_count": self.abstained_count,
        }


class UncertaintyEstimator:
    """
    Estimates prediction uncertainty and supports selective prediction.

    Uncertainty is high when the model is unsure (probability near 0.5).
    When uncertainty exceeds a threshold, the model abstains (UNCERTAIN verdict).
    """

    def __init__(
        self,
        method: Union[str, UncertaintyMethod] = "entropy",
        abstain_threshold: Optional[float] = None,
        target_coverage: Optional[float] = None,
    ):
        """
        Initialize uncertainty estimator.

        Args:
            method: Uncertainty estimation method
            abstain_threshold: Fixed threshold for abstaining (uncertainty > threshold -> abstain)
            target_coverage: Target coverage rate (finds threshold automatically)
        """
        if isinstance(method, str):
            method = UncertaintyMethod(method)
        self.method = method

        self.abstain_threshold = abstain_threshold
        self.target_coverage = target_coverage
        self._fitted_threshold: Optional[float] = None

    def estimate_uncertainty(
        self,
        probability: float,
        ensemble_probs: Optional[List[float]] = None,
    ) -> float:
        """
        Estimate uncertainty for a single prediction.

        Args:
            probability: P(AI) prediction
            ensemble_probs: Optional list of probabilities from ensemble members

        Returns:
            Uncertainty score (higher = more uncertain)
        """
        if self.method == UncertaintyMethod.ENTROPY:
            return self._entropy_uncertainty(probability)
        elif self.method == UncertaintyMethod.MARGIN:
            return self._margin_uncertainty(probability)
        elif self.method == UncertaintyMethod.CONFIDENCE:
            return self._confidence_uncertainty(probability)
        elif self.method == UncertaintyMethod.ENSEMBLE_VARIANCE:
            if ensemble_probs is None:
                return self._entropy_uncertainty(probability)
            return self._ensemble_uncertainty(ensemble_probs)
        elif self.method == UncertaintyMethod.CALIBRATED:
            return self._entropy_uncertainty(probability)
        else:
            return self._entropy_uncertainty(probability)

    def _entropy_uncertainty(self, p: float) -> float:
        """Shannon entropy uncertainty (max at p=0.5)."""
        eps = 1e-10
        p = np.clip(p, eps, 1 - eps)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        return entropy  # Max is 1.0 at p=0.5

    def _margin_uncertainty(self, p: float) -> float:
        """Margin-based uncertainty (1 - |p - 0.5| * 2)."""
        margin = abs(p - 0.5) * 2  # 0 at p=0.5, 1 at p=0 or p=1
        return 1 - margin

    def _confidence_uncertainty(self, p: float) -> float:
        """Confidence-based uncertainty."""
        confidence = max(p, 1 - p)
        return 1 - confidence

    def _ensemble_uncertainty(self, probs: List[float]) -> float:
        """Uncertainty from ensemble variance."""
        if len(probs) < 2:
            return self._entropy_uncertainty(np.mean(probs))
        return np.std(probs)

    def predict_with_uncertainty(
        self,
        probability: float,
        ensemble_probs: Optional[List[float]] = None,
    ) -> UncertaintyResult:
        """
        Make prediction with uncertainty estimation.

        Args:
            probability: P(AI) prediction
            ensemble_probs: Optional ensemble member probabilities

        Returns:
            UncertaintyResult with prediction and uncertainty info
        """
        uncertainty = self.estimate_uncertainty(probability, ensemble_probs)

        # Determine threshold
        threshold = self._get_threshold()
        should_abstain = uncertainty > threshold if threshold is not None else False

        # Determine confidence level
        confidence_level = self._get_confidence_level(uncertainty)

        return UncertaintyResult(
            ai_probability=probability,
            uncertainty=uncertainty,
            uncertainty_method=self.method.value,
            should_abstain=should_abstain,
            abstain_threshold=threshold or 0.0,
            confidence_level=confidence_level,
        )

    def _get_threshold(self) -> Optional[float]:
        """Get the abstain threshold."""
        if self._fitted_threshold is not None:
            return self._fitted_threshold
        return self.abstain_threshold

    def _get_confidence_level(self, uncertainty: float) -> str:
        """Map uncertainty to confidence level."""
        # Uncertainty ranges from 0 (certain) to 1 (max uncertain)
        if uncertainty < 0.2:
            return "very_high"
        elif uncertainty < 0.4:
            return "high"
        elif uncertainty < 0.6:
            return "medium"
        elif uncertainty < 0.8:
            return "low"
        else:
            return "very_low"

    def fit_threshold(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        target_coverage: Optional[float] = None,
    ) -> float:
        """
        Fit abstain threshold to achieve target coverage.

        Args:
            probabilities: Array of P(AI) predictions
            labels: True labels
            target_coverage: Target fraction of non-abstained samples

        Returns:
            Fitted threshold
        """
        target = target_coverage or self.target_coverage
        if target is None:
            target = 0.9  # Default: keep 90% of samples

        # Compute uncertainties
        uncertainties = np.array([
            self.estimate_uncertainty(p) for p in probabilities
        ])

        # Find threshold that achieves target coverage
        # Sort uncertainties and find the (1-target) percentile
        self._fitted_threshold = np.percentile(uncertainties, target * 100)

        return self._fitted_threshold

    def evaluate_selective_prediction(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        threshold: Optional[float] = None,
    ) -> SelectivePredictionResult:
        """
        Evaluate selective prediction performance.

        Args:
            probabilities: P(AI) predictions
            labels: True labels
            threshold: Abstain threshold (uses fitted if not provided)

        Returns:
            SelectivePredictionResult with metrics
        """
        threshold = threshold or self._get_threshold() or 0.5

        # Compute uncertainties and decisions
        uncertainties = np.array([self.estimate_uncertainty(p) for p in probabilities])
        predictions = (probabilities > 0.5).astype(int)

        # Split into abstained and covered
        abstain_mask = uncertainties > threshold
        covered_mask = ~abstain_mask

        covered_count = covered_mask.sum()
        abstained_count = abstain_mask.sum()
        total = len(probabilities)

        coverage = covered_count / total if total > 0 else 0

        # Accuracy on covered samples
        if covered_count > 0:
            covered_preds = predictions[covered_mask]
            covered_labels = labels[covered_mask]
            accuracy = (covered_preds == covered_labels).mean()

            # Confusion matrix on covered
            tp = ((covered_preds == 1) & (covered_labels == 1)).sum()
            tn = ((covered_preds == 0) & (covered_labels == 0)).sum()
            fp = ((covered_preds == 1) & (covered_labels == 0)).sum()
            fn = ((covered_preds == 0) & (covered_labels == 1)).sum()
        else:
            accuracy = 0.0
            tp = tn = fp = fn = 0

        return SelectivePredictionResult(
            coverage=coverage,
            accuracy_on_covered=accuracy,
            abstain_rate=1 - coverage,
            threshold=threshold,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            abstained_count=int(abstained_count),
        )

    def compute_coverage_accuracy_curve(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_points: int = 20,
    ) -> Dict[str, List[float]]:
        """
        Compute coverage vs accuracy curve.

        This shows the trade-off: more abstaining -> higher accuracy on remaining.

        Args:
            probabilities: P(AI) predictions
            labels: True labels
            n_points: Number of points on the curve

        Returns:
            Dictionary with 'coverage', 'accuracy', 'threshold' lists
        """
        uncertainties = np.array([self.estimate_uncertainty(p) for p in probabilities])
        predictions = (probabilities > 0.5).astype(int)

        # Thresholds from 0 (no abstain) to max uncertainty
        thresholds = np.linspace(0, uncertainties.max() + 0.01, n_points)

        coverages = []
        accuracies = []

        for thresh in thresholds:
            covered_mask = uncertainties <= thresh
            covered_count = covered_mask.sum()

            if covered_count > 0:
                coverage = covered_count / len(probabilities)
                accuracy = (predictions[covered_mask] == labels[covered_mask]).mean()
            else:
                coverage = 0.0
                accuracy = 0.0

            coverages.append(coverage)
            accuracies.append(accuracy)

        return {
            "coverage": coverages,
            "accuracy": accuracies,
            "threshold": thresholds.tolist(),
        }


class SelectivePredictor:
    """
    Wrapper that adds selective prediction (abstain) capability to any baseline.

    Usage:
        >>> predictor = SelectivePredictor(baseline, target_coverage=0.9)
        >>> result = predictor.predict(image)
        >>> if result.should_abstain:
        ...     print("UNCERTAIN - abstaining from prediction")
    """

    def __init__(
        self,
        baseline,
        uncertainty_method: str = "entropy",
        target_coverage: float = 0.9,
        abstain_threshold: Optional[float] = None,
    ):
        """
        Initialize selective predictor.

        Args:
            baseline: BaselineDetector instance
            uncertainty_method: Method for uncertainty estimation
            target_coverage: Target coverage rate (1 - abstain_rate)
            abstain_threshold: Fixed threshold (overrides target_coverage)
        """
        self.baseline = baseline
        self.uncertainty_estimator = UncertaintyEstimator(
            method=uncertainty_method,
            abstain_threshold=abstain_threshold,
            target_coverage=target_coverage,
        )
        self._calibrated = False

    def calibrate(
        self,
        val_images: List,
        val_labels: List[int],
    ) -> float:
        """
        Calibrate abstain threshold using validation data.

        Args:
            val_images: Validation images
            val_labels: Validation labels

        Returns:
            Fitted threshold
        """
        # Get predictions
        probabilities = []
        for img in val_images:
            result = self.baseline.predict_proba(img)
            probabilities.append(result.ai_probability)

        probabilities = np.array(probabilities)
        labels = np.array(val_labels)

        # Fit threshold
        threshold = self.uncertainty_estimator.fit_threshold(probabilities, labels)
        self._calibrated = True

        return threshold

    def predict(self, image) -> UncertaintyResult:
        """
        Make prediction with possible abstain.

        Args:
            image: Input image

        Returns:
            UncertaintyResult with prediction and abstain decision
        """
        result = self.baseline.predict_proba(image)
        return self.uncertainty_estimator.predict_with_uncertainty(result.ai_probability)

    def evaluate(
        self,
        images: List,
        labels: List[int],
    ) -> SelectivePredictionResult:
        """
        Evaluate selective prediction on test set.

        Args:
            images: Test images
            labels: Test labels

        Returns:
            SelectivePredictionResult with metrics
        """
        probabilities = []
        for img in images:
            result = self.baseline.predict_proba(img)
            probabilities.append(result.ai_probability)

        return self.uncertainty_estimator.evaluate_selective_prediction(
            np.array(probabilities),
            np.array(labels),
        )


def compute_risk_coverage_auc(
    probabilities: np.ndarray,
    labels: np.ndarray,
    uncertainty_method: str = "entropy",
) -> float:
    """
    Compute Area Under Risk-Coverage Curve (AURC).

    Lower AURC is better - indicates model abstains on errors.

    Args:
        probabilities: P(AI) predictions
        labels: True labels
        uncertainty_method: Method for uncertainty

    Returns:
        AURC score
    """
    estimator = UncertaintyEstimator(method=uncertainty_method)
    curve = estimator.compute_coverage_accuracy_curve(probabilities, labels)

    # Risk = 1 - accuracy
    risks = [1 - acc for acc in curve["accuracy"]]
    coverages = curve["coverage"]

    # Compute AUC using trapezoidal rule
    aurc = np.trapz(risks, coverages)

    return aurc


def analyze_abstain_characteristics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    uncertainty_method: str = "entropy",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Analyze characteristics of abstained samples.

    Args:
        probabilities: P(AI) predictions
        labels: True labels
        uncertainty_method: Uncertainty method
        threshold: Abstain threshold

    Returns:
        Dictionary with analysis results
    """
    estimator = UncertaintyEstimator(method=uncertainty_method, abstain_threshold=threshold)

    uncertainties = np.array([estimator.estimate_uncertainty(p) for p in probabilities])
    predictions = (probabilities > 0.5).astype(int)
    correct = predictions == labels

    abstain_mask = uncertainties > threshold

    # Analyze abstained samples
    abstained_correct = correct[abstain_mask].sum() if abstain_mask.any() else 0
    abstained_incorrect = (~correct[abstain_mask]).sum() if abstain_mask.any() else 0

    # Analyze covered samples
    covered_mask = ~abstain_mask
    covered_correct = correct[covered_mask].sum() if covered_mask.any() else 0
    covered_incorrect = (~correct[covered_mask]).sum() if covered_mask.any() else 0

    # Error rejection rate: what fraction of errors were abstained
    total_errors = (~correct).sum()
    errors_abstained = abstained_incorrect
    error_rejection_rate = errors_abstained / total_errors if total_errors > 0 else 0

    return {
        "total_samples": len(probabilities),
        "abstained_count": abstain_mask.sum(),
        "covered_count": covered_mask.sum(),
        "abstained_would_be_correct": int(abstained_correct),
        "abstained_would_be_incorrect": int(abstained_incorrect),
        "covered_correct": int(covered_correct),
        "covered_incorrect": int(covered_incorrect),
        "error_rejection_rate": error_rejection_rate,
        "threshold": threshold,
        "mean_uncertainty_abstained": uncertainties[abstain_mask].mean() if abstain_mask.any() else 0,
        "mean_uncertainty_covered": uncertainties[covered_mask].mean() if covered_mask.any() else 0,
    }
