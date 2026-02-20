"""
Conformal Prediction Framework for AI Image Detection.

Provides mathematically rigorous UNCERTAIN regions with coverage guarantees.
Implements Adaptive Prediction Sets (APS) for set-valued predictions.

Key features:
- Distribution-free coverage guarantee: P(y ∈ C(x)) ≥ 1-α
- Set-valued predictions: {Real}, {AI}, or {Real, AI} (uncertain)
- Configurable miscoverage rate (α = 0.05 to 0.20)
- No assumptions about underlying classifier

Reference:
    Romano, Y., Sesia, M., & Candès, E. (2020).
    Classification with Valid and Adaptive Coverage.
    NeurIPS 2020.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


class ConformalMethod(Enum):
    """Conformal prediction methods."""
    LAC = "lac"  # Least Ambiguous set-valued Classifier
    APS = "aps"  # Adaptive Prediction Sets
    RAPS = "raps"  # Regularized APS (penalizes large sets)
    NAIVE = "naive"  # Simple threshold-based


@dataclass
class ConformalPrediction:
    """
    Result from conformal prediction.

    Attributes:
        prediction_set: Set of labels in the prediction set.
        set_size: Number of labels in the prediction set.
        coverage_level: Nominal coverage level (1 - alpha).
        is_uncertain: True if set contains multiple labels.
        probabilities: Original classifier probabilities.
        conformity_scores: Non-conformity scores used.
        threshold: Calibrated threshold (quantile).
    """
    prediction_set: Set[str]
    set_size: int
    coverage_level: float
    is_uncertain: bool
    probabilities: Dict[str, float]
    conformity_scores: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.0

    def get_primary_label(self) -> str:
        """Get the most likely label from the set."""
        if len(self.prediction_set) == 1:
            return list(self.prediction_set)[0]
        # Return label with highest probability
        return max(self.probabilities.items(), key=lambda x: x[1])[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prediction_set": list(self.prediction_set),
            "set_size": self.set_size,
            "coverage_level": self.coverage_level,
            "is_uncertain": self.is_uncertain,
            "probabilities": self.probabilities,
            "conformity_scores": self.conformity_scores,
            "threshold": self.threshold,
            "primary_label": self.get_primary_label(),
        }


@dataclass
class ConformalCalibrationResult:
    """Result from conformal calibration."""
    threshold: float  # Calibrated quantile threshold
    alpha: float  # Miscoverage rate
    coverage_level: float  # 1 - alpha
    n_calibration: int  # Number of calibration samples
    empirical_coverage: float  # Actual coverage on calibration set
    avg_set_size: float  # Average prediction set size
    method: str  # Method used


class ConformalPredictor:
    """
    Conformal Prediction for binary classification with coverage guarantees.

    Implements multiple conformal prediction methods:
    - LAC (Least Ambiguous Classifier): Simple and efficient
    - APS (Adaptive Prediction Sets): Adaptive to difficulty
    - RAPS (Regularized APS): Controls prediction set size

    Usage:
        predictor = ConformalPredictor(alpha=0.1)  # 90% coverage
        predictor.calibrate(cal_probs, cal_labels)
        prediction = predictor.predict(test_prob)

        if prediction.is_uncertain:
            print("Model is uncertain, abstain from prediction")
    """

    def __init__(
        self,
        alpha: float = 0.1,
        method: ConformalMethod = ConformalMethod.APS,
        labels: Tuple[str, str] = ("real", "ai_generated"),
        raps_lambda: float = 0.01,
        raps_k_reg: int = 1,
    ):
        """
        Initialize conformal predictor.

        Args:
            alpha: Miscoverage rate (1-alpha = coverage guarantee).
                   - 0.05 → 95% coverage (conservative, larger sets)
                   - 0.10 → 90% coverage (standard, balanced)
                   - 0.20 → 80% coverage (aggressive, smaller sets)
            method: Conformal prediction method to use.
            labels: Label names for the two classes.
            raps_lambda: Regularization parameter for RAPS method.
            raps_k_reg: Number of labels before penalty kicks in for RAPS.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

        self.alpha = alpha
        self.method = method
        self.labels = labels
        self.raps_lambda = raps_lambda
        self.raps_k_reg = raps_k_reg

        # Calibration state
        self._threshold: Optional[float] = None
        self._calibrated = False
        self._calibration_result: Optional[ConformalCalibrationResult] = None

        logger.info(
            f"ConformalPredictor initialized: alpha={alpha}, "
            f"coverage={1-alpha:.1%}, method={method.value}"
        )

    @property
    def threshold(self) -> float:
        """Get calibrated threshold."""
        if not self._calibrated:
            raise RuntimeError("Predictor not calibrated. Call calibrate() first.")
        return self._threshold

    @property
    def coverage_level(self) -> float:
        """Get nominal coverage level."""
        return 1 - self.alpha

    def _compute_nonconformity_lac(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute LAC non-conformity scores.

        LAC score: s(x, y) = 1 - p(y|x)
        The score is simply 1 minus the probability of the true label.

        Args:
            probs: Predicted probabilities for positive class (AI).
            labels: True labels (0=real, 1=AI). If None, compute for all labels.

        Returns:
            Non-conformity scores.
        """
        if labels is not None:
            # For calibration: score = 1 - p(true_label)
            p_true = np.where(labels == 1, probs, 1 - probs)
            return 1 - p_true
        else:
            # For prediction: return scores for both classes
            # Real: 1 - (1 - probs) = probs
            # AI: 1 - probs
            return np.column_stack([probs, 1 - probs])

    def _compute_nonconformity_aps(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute APS non-conformity scores.

        APS score: s(x, y) = sum_{j: p_j >= p_y} p_j
        Sum of probabilities of all labels with probability >= true label's prob.

        This adaptive score naturally creates larger sets for harder examples.

        Args:
            probs: Predicted probabilities for positive class (AI).
            labels: True labels (0=real, 1=AI).

        Returns:
            Non-conformity scores.
        """
        # Convert to 2-class probabilities
        p_real = 1 - probs
        p_ai = probs

        if labels is not None:
            # Calibration: compute score for true label
            scores = np.zeros(len(probs))
            for i in range(len(probs)):
                if labels[i] == 1:  # True label is AI
                    # Sum probs >= p_ai
                    if p_real[i] >= p_ai[i]:
                        scores[i] = p_real[i] + p_ai[i]  # Both classes
                    else:
                        scores[i] = p_ai[i]  # Only AI class
                else:  # True label is Real
                    # Sum probs >= p_real
                    if p_ai[i] >= p_real[i]:
                        scores[i] = p_ai[i] + p_real[i]  # Both classes
                    else:
                        scores[i] = p_real[i]  # Only Real class
            return scores
        else:
            # Prediction: return scores for both classes
            scores_real = np.where(p_ai >= p_real, p_ai + p_real, p_real)
            scores_ai = np.where(p_real >= p_ai, p_real + p_ai, p_ai)
            return np.column_stack([scores_real, scores_ai])

    def _compute_nonconformity_raps(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute RAPS non-conformity scores.

        RAPS adds a penalty for including additional labels beyond k_reg.
        This regularization helps control prediction set sizes.

        Args:
            probs: Predicted probabilities for positive class (AI).
            labels: True labels (0=real, 1=AI).

        Returns:
            Non-conformity scores.
        """
        # Start with APS scores
        aps_scores = self._compute_nonconformity_aps(probs, labels)

        # Add regularization penalty
        # For binary classification with k_reg=1, penalty applies when both labels included
        if labels is not None:
            # For calibration, the penalty is based on ranking
            p_real = 1 - probs
            p_ai = probs

            penalties = np.zeros(len(probs))
            for i in range(len(probs)):
                if labels[i] == 1:  # AI is true
                    if p_real[i] >= p_ai[i]:
                        # True label is ranked 2nd, penalty applies
                        penalties[i] = self.raps_lambda * 1
                else:  # Real is true
                    if p_ai[i] >= p_real[i]:
                        # True label is ranked 2nd, penalty applies
                        penalties[i] = self.raps_lambda * 1

            return aps_scores + penalties
        else:
            # For prediction, add penalty column
            penalties = np.ones_like(aps_scores) * self.raps_lambda
            penalties[:, 0] = 0  # No penalty for first label added
            return aps_scores + penalties

    def _compute_nonconformity(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute non-conformity scores based on selected method."""
        if self.method == ConformalMethod.LAC:
            return self._compute_nonconformity_lac(probs, labels)
        elif self.method == ConformalMethod.APS:
            return self._compute_nonconformity_aps(probs, labels)
        elif self.method == ConformalMethod.RAPS:
            return self._compute_nonconformity_raps(probs, labels)
        elif self.method == ConformalMethod.NAIVE:
            return self._compute_nonconformity_lac(probs, labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
    ) -> ConformalCalibrationResult:
        """
        Calibrate the conformal predictor on a calibration set.

        Computes the quantile threshold that guarantees (1-alpha) coverage.

        Args:
            cal_probs: Calibration set predicted probabilities for AI class.
            cal_labels: Calibration set true labels (0=real, 1=AI).

        Returns:
            ConformalCalibrationResult with calibration details.
        """
        cal_probs = np.asarray(cal_probs).flatten()
        cal_labels = np.asarray(cal_labels).flatten()

        if len(cal_probs) != len(cal_labels):
            raise ValueError("Probabilities and labels must have same length")

        n = len(cal_probs)

        # Compute non-conformity scores for calibration set
        scores = self._compute_nonconformity(cal_probs, cal_labels)

        # Compute quantile threshold
        # q̂ = ⌈(n+1)(1-α)⌉/n quantile of calibration scores
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(quantile_level, 1.0)  # Cap at 1.0

        self._threshold = np.quantile(scores, quantile_level)
        self._calibrated = True

        # Compute empirical coverage on calibration set (for verification)
        predictions = [self.predict(p) for p in cal_probs]
        empirical_coverage = np.mean([
            self.labels[int(cal_labels[i])] in predictions[i].prediction_set
            for i in range(n)
        ])

        # Average set size
        avg_set_size = np.mean([p.set_size for p in predictions])

        self._calibration_result = ConformalCalibrationResult(
            threshold=self._threshold,
            alpha=self.alpha,
            coverage_level=1 - self.alpha,
            n_calibration=n,
            empirical_coverage=empirical_coverage,
            avg_set_size=avg_set_size,
            method=self.method.value,
        )

        logger.info(
            f"Conformal calibration complete: "
            f"threshold={self._threshold:.4f}, "
            f"empirical_coverage={empirical_coverage:.1%}, "
            f"avg_set_size={avg_set_size:.2f}"
        )

        return self._calibration_result

    def predict(
        self,
        prob: Union[float, np.ndarray],
    ) -> ConformalPrediction:
        """
        Make a conformal prediction with coverage guarantee.

        Args:
            prob: Predicted probability for AI class (single value or 1D array).

        Returns:
            ConformalPrediction with prediction set and metadata.
        """
        if not self._calibrated:
            raise RuntimeError("Predictor not calibrated. Call calibrate() first.")

        prob = float(prob) if np.isscalar(prob) else float(prob[0])

        # Probabilities for both classes
        p_real = 1 - prob
        p_ai = prob

        probabilities = {
            self.labels[0]: p_real,  # "real"
            self.labels[1]: p_ai,  # "ai_generated"
        }

        # Compute non-conformity scores for both classes
        probs_array = np.array([prob])
        scores_both = self._compute_nonconformity(probs_array, labels=None)

        if scores_both.ndim == 2:
            score_real = scores_both[0, 0]
            score_ai = scores_both[0, 1]
        else:
            # LAC method returns 2D for prediction
            score_real = p_ai  # 1 - p_real = p_ai
            score_ai = p_real  # 1 - p_ai = p_real

        conformity_scores = {
            self.labels[0]: score_real,
            self.labels[1]: score_ai,
        }

        # Build prediction set: include label if score <= threshold
        prediction_set = set()

        if score_real <= self._threshold:
            prediction_set.add(self.labels[0])
        if score_ai <= self._threshold:
            prediction_set.add(self.labels[1])

        # If empty set (shouldn't happen with proper calibration), include most likely
        if len(prediction_set) == 0:
            if p_ai >= p_real:
                prediction_set.add(self.labels[1])
            else:
                prediction_set.add(self.labels[0])

        return ConformalPrediction(
            prediction_set=prediction_set,
            set_size=len(prediction_set),
            coverage_level=1 - self.alpha,
            is_uncertain=len(prediction_set) > 1,
            probabilities=probabilities,
            conformity_scores=conformity_scores,
            threshold=self._threshold,
        )

    def predict_batch(
        self,
        probs: np.ndarray,
    ) -> List[ConformalPrediction]:
        """Make conformal predictions for a batch of samples."""
        return [self.predict(p) for p in probs]

    def evaluate_coverage(
        self,
        test_probs: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate coverage on a test set.

        Args:
            test_probs: Test set predicted probabilities.
            test_labels: Test set true labels.

        Returns:
            Dictionary with coverage metrics.
        """
        predictions = self.predict_batch(test_probs)

        # Coverage: fraction of times true label is in prediction set
        coverage = np.mean([
            self.labels[int(test_labels[i])] in predictions[i].prediction_set
            for i in range(len(test_labels))
        ])

        # Average set size
        avg_set_size = np.mean([p.set_size for p in predictions])

        # Fraction uncertain (set size > 1)
        frac_uncertain = np.mean([p.is_uncertain for p in predictions])

        # Size-stratified coverage
        single_set_coverage = np.mean([
            self.labels[int(test_labels[i])] in predictions[i].prediction_set
            for i in range(len(test_labels))
            if predictions[i].set_size == 1
        ]) if any(p.set_size == 1 for p in predictions) else 0.0

        return {
            "coverage": coverage,
            "avg_set_size": avg_set_size,
            "frac_uncertain": frac_uncertain,
            "single_set_coverage": single_set_coverage,
            "nominal_coverage": 1 - self.alpha,
            "coverage_gap": coverage - (1 - self.alpha),
        }


class AdaptiveConformalPredictor(ConformalPredictor):
    """
    Adaptive conformal predictor that adjusts alpha based on difficulty.

    Uses multiple alpha levels and selects based on classifier confidence
    or ensemble disagreement.
    """

    def __init__(
        self,
        alpha_levels: List[float] = [0.05, 0.10, 0.15, 0.20],
        method: ConformalMethod = ConformalMethod.APS,
        labels: Tuple[str, str] = ("real", "ai_generated"),
    ):
        """
        Initialize adaptive conformal predictor.

        Args:
            alpha_levels: List of alpha levels to calibrate.
            method: Conformal prediction method.
            labels: Label names.
        """
        self.alpha_levels = sorted(alpha_levels)
        self.method = method
        self.labels = labels

        # Create predictor for each alpha level
        self._predictors: Dict[float, ConformalPredictor] = {}
        for alpha in self.alpha_levels:
            self._predictors[alpha] = ConformalPredictor(
                alpha=alpha,
                method=method,
                labels=labels,
            )

        self._calibrated = False
        logger.info(
            f"AdaptiveConformalPredictor initialized with alpha levels: {alpha_levels}"
        )

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
    ) -> Dict[float, ConformalCalibrationResult]:
        """Calibrate all alpha levels."""
        results = {}
        for alpha, predictor in self._predictors.items():
            results[alpha] = predictor.calibrate(cal_probs, cal_labels)
        self._calibrated = True
        return results

    def predict_adaptive(
        self,
        prob: float,
        confidence_score: Optional[float] = None,
        ensemble_std: Optional[float] = None,
    ) -> ConformalPrediction:
        """
        Make adaptive prediction based on difficulty indicators.

        Args:
            prob: Predicted probability for AI class.
            confidence_score: Optional classifier confidence (0-1).
            ensemble_std: Optional ensemble standard deviation.

        Returns:
            ConformalPrediction with adaptive alpha selection.
        """
        if not self._calibrated:
            raise RuntimeError("Predictors not calibrated")

        # Determine difficulty level
        # Higher difficulty → lower alpha (more conservative)
        difficulty = 0.0

        # Confidence-based difficulty (lower confidence = higher difficulty)
        if confidence_score is not None:
            difficulty += (1 - confidence_score) * 0.5

        # Probability-based difficulty (closer to 0.5 = higher difficulty)
        prob_difficulty = 1 - 2 * abs(prob - 0.5)
        difficulty += prob_difficulty * 0.3

        # Ensemble disagreement (higher std = higher difficulty)
        if ensemble_std is not None:
            difficulty += min(ensemble_std * 2, 0.2)

        # Map difficulty to alpha level
        # difficulty 0 → highest alpha (smallest sets)
        # difficulty 1 → lowest alpha (largest sets, most conservative)
        alpha_idx = int(difficulty * (len(self.alpha_levels) - 1))
        alpha_idx = max(0, min(alpha_idx, len(self.alpha_levels) - 1))

        # Use more conservative (lower) alpha for higher difficulty
        selected_alpha = self.alpha_levels[-(alpha_idx + 1)]

        return self._predictors[selected_alpha].predict(prob)


def compute_coverage_accuracy_tradeoff(
    predictor: ConformalPredictor,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    alpha_range: np.ndarray = None,
) -> Dict[str, np.ndarray]:
    """
    Compute coverage vs accuracy trade-off curve.

    Useful for visualizing the effect of different alpha values.

    Args:
        predictor: Base conformal predictor (will be recalibrated).
        test_probs: Test probabilities.
        test_labels: Test labels.
        alpha_range: Range of alpha values to evaluate.

    Returns:
        Dictionary with arrays for plotting.
    """
    if alpha_range is None:
        alpha_range = np.linspace(0.01, 0.30, 30)

    coverages = []
    set_sizes = []
    accuracies = []
    uncertain_fracs = []

    # Split test set for calibration/evaluation
    n = len(test_probs)
    n_cal = n // 2
    cal_probs, eval_probs = test_probs[:n_cal], test_probs[n_cal:]
    cal_labels, eval_labels = test_labels[:n_cal], test_labels[n_cal:]

    for alpha in alpha_range:
        temp_predictor = ConformalPredictor(
            alpha=alpha,
            method=predictor.method,
            labels=predictor.labels,
        )
        temp_predictor.calibrate(cal_probs, cal_labels)

        metrics = temp_predictor.evaluate_coverage(eval_probs, eval_labels)
        coverages.append(metrics["coverage"])
        set_sizes.append(metrics["avg_set_size"])
        uncertain_fracs.append(metrics["frac_uncertain"])

        # Compute accuracy on certain predictions only
        predictions = temp_predictor.predict_batch(eval_probs)
        certain_mask = [not p.is_uncertain for p in predictions]

        if sum(certain_mask) > 0:
            certain_preds = [
                1 if predictions[i].get_primary_label() == predictor.labels[1] else 0
                for i in range(len(predictions)) if certain_mask[i]
            ]
            certain_labels = eval_labels[certain_mask]
            accuracy = np.mean(np.array(certain_preds) == certain_labels)
        else:
            accuracy = np.nan

        accuracies.append(accuracy)

    return {
        "alpha": alpha_range,
        "coverage": np.array(coverages),
        "avg_set_size": np.array(set_sizes),
        "accuracy_on_certain": np.array(accuracies),
        "frac_uncertain": np.array(uncertain_fracs),
    }
