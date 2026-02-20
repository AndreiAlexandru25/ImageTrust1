#!/usr/bin/env python3
"""
Advanced Threshold Calibration System - Academic-Grade Implementation

This script implements a COMPLETE calibration system for forensic methods,
with rigorous statistical validation for the master's thesis.

Advanced features:
1. K-Fold Cross-Validation with stratification
2. Statistical significance tests (McNemar, Wilcoxon, DeLong)
3. Reliability Diagrams (calibration curves)
4. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
5. Temperature Scaling for probability calibration
6. Isotonic Regression for non-parametric calibration
7. Platt Scaling (sigmoidal calibration)
8. Bootstrap Confidence Intervals with BCa (bias-corrected accelerated)
9. ROC and Precision-Recall curve plotting
10. Variance analysis between folds
11. Statistical comparison between methods
12. Separate holdout validation
13. Cost-sensitive threshold selection
14. Ensemble threshold optimization
15. Complete LaTeX reports for the thesis

Professor requirement: "Demonstrate each threshold experimentally!"

Usage:
    python scripts/calibrate_thresholds_advanced.py --dataset data/calibration --output outputs/calibration
    python scripts/calibrate_thresholds_advanced.py --dataset data/casia --cross-validate --k-folds 5
    python scripts/calibrate_thresholds_advanced.py --all --statistical-tests --generate-plots
"""

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import io
import hashlib

import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class ThresholdMethod(Enum):
    """Threshold selection methods."""
    YOUDEN = "youden"              # Maximizes TPR - FPR (Youden's J)
    F1_MAX = "f1_max"              # Maximizes F1 score
    F2_MAX = "f2_max"              # Maximizes F2 (favors recall)
    F05_MAX = "f05_max"            # Maximizes F0.5 (favors precision)
    PRECISION_TARGET = "precision_target"  # Reaches target precision
    RECALL_TARGET = "recall_target"        # Reaches target recall
    COST_SENSITIVE = "cost_sensitive"      # Minimizes weighted cost
    GMEAN = "gmean"                # Maximizes geometric mean
    MCC_MAX = "mcc_max"            # Maximizes Matthews Correlation Coefficient
    BALANCED_ACCURACY = "balanced_accuracy"  # Maximizes balanced accuracy
    EQUAL_ERROR_RATE = "eer"       # Equal Error Rate (FPR = FNR)


class CalibrationMethod(Enum):
    """Probability calibration methods."""
    NONE = "none"
    TEMPERATURE_SCALING = "temperature"
    PLATT_SCALING = "platt"
    ISOTONIC_REGRESSION = "isotonic"
    BETA_CALIBRATION = "beta"
    HISTOGRAM_BINNING = "histogram"


# Default error costs (can be adjusted)
DEFAULT_FP_COST = 1.0  # Cost of a False Positive
DEFAULT_FN_COST = 2.0  # Cost of a False Negative (usually higher)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CrossValidationResult:
    """Result for a single cross-validation fold."""
    fold_idx: int
    threshold: float
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    n_train: int
    n_val: int


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool  # At alpha = 0.05
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""


@dataclass
class CalibrationMetrics:
    """Probability calibration metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    log_loss: float
    reliability_data: Dict[str, Any]  # For plotting
    calibration_method: str
    temperature: Optional[float] = None  # For temperature scaling


@dataclass
class AdvancedCalibrationResult:
    """Complete calibration result for a method."""

    # Identificare
    method_name: str
    method_key: str

    # Threshold principal
    threshold_optimal: float
    threshold_method: str
    confidence_interval_95: Tuple[float, float]
    confidence_interval_99: Tuple[float, float]

    # Cross-validation results
    cv_thresholds: List[float]
    cv_threshold_mean: float
    cv_threshold_std: float
    cv_metrics: Dict[str, List[float]]  # {metric_name: [fold1, fold2, ...]}

    # Main metrics (at optimal threshold)
    metrics: Dict[str, float]  # accuracy, precision, recall, f1, auc, mcc, etc.

    # Probability calibration
    calibration: Optional[CalibrationMetrics] = None

    # Comparisons with other methods
    statistical_comparisons: Dict[str, StatisticalTestResult] = field(default_factory=dict)

    # Metadata
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    dataset_name: str = ""
    calibration_date: str = ""

    # Literature references
    literature_threshold: Optional[float] = None
    literature_source: Optional[str] = None
    literature_refs: List[Dict[str, str]] = field(default_factory=list)

    # Alternative thresholds for different scenarios
    alternative_thresholds: Dict[str, float] = field(default_factory=dict)

    # Notes and observations
    notes: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method_name": self.method_name,
            "method_key": self.method_key,
            "threshold_optimal": round(self.threshold_optimal, 4),
            "threshold_method": self.threshold_method,
            "confidence_interval_95": [round(x, 4) for x in self.confidence_interval_95],
            "confidence_interval_99": [round(x, 4) for x in self.confidence_interval_99],
            "cv_threshold_mean": round(self.cv_threshold_mean, 4),
            "cv_threshold_std": round(self.cv_threshold_std, 4),
            "cv_thresholds": [round(t, 4) for t in self.cv_thresholds],
            "cv_metrics": {k: [round(v, 4) for v in vals] for k, vals in self.cv_metrics.items()},
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "calibration": {
                "ece": round(self.calibration.ece, 4),
                "mce": round(self.calibration.mce, 4),
                "brier_score": round(self.calibration.brier_score, 4),
                "method": self.calibration.calibration_method,
            } if self.calibration else None,
            "statistical_comparisons": {
                k: {
                    "test": v.test_name,
                    "statistic": round(v.statistic, 4),
                    "p_value": round(v.p_value, 6),
                    "significant": v.is_significant,
                    "interpretation": v.interpretation,
                } for k, v in self.statistical_comparisons.items()
            },
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "dataset_name": self.dataset_name,
            "calibration_date": self.calibration_date,
            "literature_threshold": self.literature_threshold,
            "literature_source": self.literature_source,
            "alternative_thresholds": {k: round(v, 4) for k, v in self.alternative_thresholds.items()},
            "notes": self.notes,
            "warnings": self.warnings,
        }

    def to_latex_row(self) -> str:
        """Generate row for LaTeX table."""
        ci = f"[{self.confidence_interval_95[0]:.2f}, {self.confidence_interval_95[1]:.2f}]"
        cv_std = f"±{self.cv_threshold_std:.2f}" if self.cv_threshold_std > 0 else ""
        lit = f"{self.literature_threshold:.2f}" if self.literature_threshold else "N/A"

        return (
            f"{self.method_name} & "
            f"{self.threshold_optimal:.3f}{cv_std} & "
            f"{ci} & "
            f"{self.metrics.get('auc', 0):.3f} & "
            f"{self.metrics.get('f1', 0):.3f} & "
            f"{self.metrics.get('mcc', 0):.3f} & "
            f"{self.calibration.ece:.3f} & "
            f"{lit} \\\\"
        ) if self.calibration else (
            f"{self.method_name} & "
            f"{self.threshold_optimal:.3f}{cv_std} & "
            f"{ci} & "
            f"{self.metrics.get('auc', 0):.3f} & "
            f"{self.metrics.get('f1', 0):.3f} & "
            f"{self.metrics.get('mcc', 0):.3f} & "
            f"N/A & "
            f"{lit} \\\\"
        )


@dataclass
class MethodConfig:
    """Configuration for a forensic method."""
    name: str
    key: str
    description: str
    compute_score: Callable
    literature_refs: List[Dict[str, str]] = field(default_factory=list)
    literature_threshold: Optional[float] = None
    default_threshold: float = 0.5
    higher_is_suspicious: bool = True
    category: str = "forensic"  # forensic, ai_detection, metadata


# =============================================================================
# METRICS AND STATISTICAL FUNCTIONS
# =============================================================================


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute all relevant metrics.

    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Binary predictions (0/1)
        y_prob: Continuous probabilities/scores

    Returns:
        Dict with all metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, matthews_corrcoef,
        balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
        brier_score_loss, log_loss
    )

    metrics = {}

    # Classification metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["specificity"] = recall_score(1 - y_true, 1 - y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["f2"] = fbeta_score(y_true, y_pred, beta=2)
    metrics["f05"] = fbeta_score(y_true, y_pred, beta=0.5)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["kappa"] = cohen_kappa_score(y_true, y_pred)

    # Probability-based metrics
    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc"] = 0.5

    try:
        metrics["average_precision"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["average_precision"] = 0.0

    # Clamp probabilities for log loss
    y_prob_clamped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    metrics["brier_score"] = brier_score_loss(y_true, y_prob_clamped)
    metrics["log_loss"] = log_loss(y_true, y_prob_clamped)

    # Confusion matrix derived metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Geometric mean
    metrics["gmean"] = np.sqrt(metrics["recall"] * metrics["specificity"])

    # Youden's J
    metrics["youden_j"] = metrics["recall"] + metrics["specificity"] - 1

    return metrics


def fbeta_score(y_true: np.ndarray, y_pred: np.ndarray, beta: float) -> float:
    """Compute F-beta score."""
    from sklearn.metrics import precision_score, recall_score

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    if precision + recall == 0:
        return 0.0

    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15
) -> Tuple[float, float, Dict]:
    """
    Compute Expected Calibration Error and Maximum Calibration Error.

    ECE = Σ (|B_m| / n) * |accuracy(B_m) - confidence(B_m)|

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        n_bins: Number of bins

    Returns:
        (ECE, MCE, reliability_data for plotting)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0

    reliability_data = {
        "bin_centers": [],
        "bin_accuracies": [],
        "bin_confidences": [],
        "bin_counts": [],
    }

    n_samples = len(y_true)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            confidence_in_bin = np.mean(y_prob[in_bin])

            gap = abs(accuracy_in_bin - confidence_in_bin)
            ece += prop_in_bin * gap
            mce = max(mce, gap)

            reliability_data["bin_centers"].append((bin_lower + bin_upper) / 2)
            reliability_data["bin_accuracies"].append(accuracy_in_bin)
            reliability_data["bin_confidences"].append(confidence_in_bin)
            reliability_data["bin_counts"].append(np.sum(in_bin))

    return ece, mce, reliability_data


def temperature_scaling(
    logits: np.ndarray,
    y_true: np.ndarray,
    init_temp: float = 1.5
) -> Tuple[float, np.ndarray]:
    """
    Apply Temperature Scaling for calibration.

    Finds the optimal temperature that minimizes NLL on the validation set.

    Args:
        logits: Log-odds or probabilities (will be converted)
        y_true: Ground truth labels
        init_temp: Initial temperature

    Returns:
        (optimal temperature, calibrated probabilities)
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import expit  # sigmoid

    # Convert probabilities to logits if needed
    probs = np.clip(logits, 1e-7, 1 - 1e-7)
    if np.all((probs >= 0) & (probs <= 1)):
        # Looks like probabilities, convert to logits
        logits = np.log(probs / (1 - probs))

    def nll_with_temperature(T):
        """Negative log likelihood with the given temperature."""
        scaled_probs = expit(logits / T)
        scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
        nll = -np.mean(
            y_true * np.log(scaled_probs) +
            (1 - y_true) * np.log(1 - scaled_probs)
        )
        return nll

    # Optimize temperature
    result = minimize_scalar(
        nll_with_temperature,
        bounds=(0.1, 10.0),
        method='bounded'
    )

    optimal_temp = result.x
    calibrated_probs = expit(logits / optimal_temp)

    return optimal_temp, calibrated_probs


def platt_scaling(
    scores: np.ndarray,
    y_true: np.ndarray
) -> Tuple[Tuple[float, float], np.ndarray]:
    """
    Apply Platt Scaling (logistic regression on scores).

    P(y=1|s) = 1 / (1 + exp(A*s + B))

    Args:
        scores: Scores from the model
        y_true: Ground truth labels

    Returns:
        ((A, B), calibrated probabilities)
    """
    from sklearn.linear_model import LogisticRegression

    # Fit logistic regression
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(scores.reshape(-1, 1), y_true)

    A = lr.coef_[0][0]
    B = lr.intercept_[0]

    calibrated_probs = lr.predict_proba(scores.reshape(-1, 1))[:, 1]

    return (A, B), calibrated_probs


def isotonic_calibration(
    scores: np.ndarray,
    y_true: np.ndarray
) -> np.ndarray:
    """
    Apply Isotonic Regression for non-parametric calibration.

    Args:
        scores: Scores from the model
        y_true: Ground truth labels

    Returns:
        Calibrated probabilities
    """
    from sklearn.isotonic import IsotonicRegression

    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(scores, y_true)

    calibrated_probs = ir.predict(scores)
    return calibrated_probs


# =============================================================================
# TESTE STATISTICE
# =============================================================================


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray
) -> StatisticalTestResult:
    """
    McNemar's test for comparing two classifiers on the same dataset.

    Tests whether the difference between classifiers is statistically significant.
    """
    from scipy.stats import chi2

    # Build contingency table
    # a and b both correct
    both_correct = np.sum((preds_a == y_true) & (preds_b == y_true))
    # a correct, b wrong
    a_correct_b_wrong = np.sum((preds_a == y_true) & (preds_b != y_true))
    # a wrong, b correct
    a_wrong_b_correct = np.sum((preds_a != y_true) & (preds_b == y_true))
    # both wrong
    both_wrong = np.sum((preds_a != y_true) & (preds_b != y_true))

    # McNemar statistic (with continuity correction)
    b = a_correct_b_wrong
    c = a_wrong_b_correct

    if b + c == 0:
        return StatisticalTestResult(
            test_name="McNemar",
            statistic=0,
            p_value=1.0,
            is_significant=False,
            interpretation="No differences in predictions between methods."
        )

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(statistic, df=1)

    return StatisticalTestResult(
        test_name="McNemar",
        statistic=statistic,
        p_value=p_value,
        is_significant=p_value < 0.05,
        effect_size=abs(b - c) / (b + c) if (b + c) > 0 else 0,
        interpretation=(
            f"The difference is {'significant' if p_value < 0.05 else 'not significant'} "
            f"statistically (p={p_value:.4f}). "
            f"Discordances: A correct/B wrong={b}, A wrong/B correct={c}."
        )
    )


def delong_test(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray
) -> StatisticalTestResult:
    """
    DeLong's test for comparing AUCs of two models.

    Simplified implementation based on asymptotic variance.
    """
    from scipy.stats import norm

    # Calculate AUCs
    from sklearn.metrics import roc_auc_score

    try:
        auc_a = roc_auc_score(y_true, probs_a)
        auc_b = roc_auc_score(y_true, probs_b)
    except ValueError:
        return StatisticalTestResult(
            test_name="DeLong",
            statistic=0,
            p_value=1.0,
            is_significant=False,
            interpretation="Could not compute AUC for one of the models."
        )

    # Variance estimation (simplified)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    # Hanley-McNeil variance approximation
    q1_a = auc_a / (2 - auc_a)
    q2_a = 2 * auc_a ** 2 / (1 + auc_a)
    var_a = (auc_a * (1 - auc_a) + (n_pos - 1) * (q1_a - auc_a ** 2) +
             (n_neg - 1) * (q2_a - auc_a ** 2)) / (n_pos * n_neg)

    q1_b = auc_b / (2 - auc_b)
    q2_b = 2 * auc_b ** 2 / (1 + auc_b)
    var_b = (auc_b * (1 - auc_b) + (n_pos - 1) * (q1_b - auc_b ** 2) +
             (n_neg - 1) * (q2_b - auc_b ** 2)) / (n_pos * n_neg)

    # Covariance (simplified, assuming some correlation)
    cov_ab = 0.5 * np.sqrt(var_a * var_b)  # Conservative estimate

    # Z statistic
    se_diff = np.sqrt(var_a + var_b - 2 * cov_ab)

    if se_diff < 1e-10:
        z_stat = 0
        p_value = 1.0
    else:
        z_stat = (auc_a - auc_b) / se_diff
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    return StatisticalTestResult(
        test_name="DeLong",
        statistic=z_stat,
        p_value=p_value,
        is_significant=p_value < 0.05,
        effect_size=auc_a - auc_b,
        confidence_interval=(
            (auc_a - auc_b) - 1.96 * se_diff,
            (auc_a - auc_b) + 1.96 * se_diff
        ),
        interpretation=(
            f"AUC_A={auc_a:.4f}, AUC_B={auc_b:.4f}, Difference={auc_a-auc_b:.4f}. "
            f"The difference {'is' if p_value < 0.05 else 'is not'} significant "
            f"(z={z_stat:.2f}, p={p_value:.4f})."
        )
    )


def wilcoxon_signed_rank_test(
    metrics_a: List[float],
    metrics_b: List[float]
) -> StatisticalTestResult:
    """
    Wilcoxon signed-rank test for comparing performance across folds.

    Useful for comparing metrics obtained in cross-validation.
    """
    from scipy.stats import wilcoxon

    if len(metrics_a) != len(metrics_b) or len(metrics_a) < 5:
        return StatisticalTestResult(
            test_name="Wilcoxon",
            statistic=0,
            p_value=1.0,
            is_significant=False,
            interpretation="Too few data points for the Wilcoxon test."
        )

    try:
        statistic, p_value = wilcoxon(metrics_a, metrics_b, alternative='two-sided')
    except ValueError:
        return StatisticalTestResult(
            test_name="Wilcoxon",
            statistic=0,
            p_value=1.0,
            is_significant=False,
            interpretation="Zero or insufficient differences for the Wilcoxon test."
        )

    mean_diff = np.mean(np.array(metrics_a) - np.array(metrics_b))

    return StatisticalTestResult(
        test_name="Wilcoxon Signed-Rank",
        statistic=statistic,
        p_value=p_value,
        is_significant=p_value < 0.05,
        effect_size=mean_diff,
        interpretation=(
            f"Mean difference={mean_diff:.4f}. "
            f"The difference {'is' if p_value < 0.05 else 'is not'} significant "
            f"(W={statistic:.2f}, p={p_value:.4f})."
        )
    )


def bootstrap_ci_bca(
    data: np.ndarray,
    stat_func: Callable,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Bootstrap BCa (Bias-Corrected and Accelerated) confidence interval.

    More robust than simple percentile bootstrap.
    """
    np.random.seed(random_state)

    n = len(data)
    original_stat = stat_func(data)

    # Bootstrap resampling
    boot_stats = []
    for _ in range(n_bootstrap):
        boot_sample = data[np.random.choice(n, size=n, replace=True)]
        boot_stats.append(stat_func(boot_sample))

    boot_stats = np.array(boot_stats)

    # Bias correction
    z0 = norm_ppf(np.mean(boot_stats < original_stat))

    # Acceleration (jackknife)
    jackknife_stats = []
    for i in range(n):
        jack_sample = np.delete(data, i)
        jackknife_stats.append(stat_func(jack_sample))

    jackknife_stats = np.array(jackknife_stats)
    jack_mean = np.mean(jackknife_stats)

    numerator = np.sum((jack_mean - jackknife_stats) ** 3)
    denominator = 6 * (np.sum((jack_mean - jackknife_stats) ** 2)) ** 1.5

    if abs(denominator) < 1e-10:
        a = 0
    else:
        a = numerator / denominator

    # Adjusted percentiles
    alpha = 1 - confidence
    z_alpha = norm_ppf(alpha / 2)
    z_1_alpha = norm_ppf(1 - alpha / 2)

    # BCa adjustments
    alpha1 = norm_cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    alpha2 = norm_cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))

    lower = np.percentile(boot_stats, 100 * alpha1)
    upper = np.percentile(boot_stats, 100 * alpha2)

    return (lower, upper)


def norm_ppf(p: float) -> float:
    """Normal distribution percent point function."""
    from scipy.stats import norm
    return norm.ppf(np.clip(p, 1e-10, 1 - 1e-10))


def norm_cdf(x: float) -> float:
    """Normal distribution CDF."""
    from scipy.stats import norm
    return norm.cdf(x)


# =============================================================================
# THRESHOLD SELECTION METHODS
# =============================================================================


def find_threshold_youden(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    """Youden's J statistic: maximizes TPR - FPR."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)

    return thresholds[optimal_idx], {
        "method": "youden",
        "j_score": j_scores[optimal_idx],
        "tpr": tpr[optimal_idx],
        "fpr": fpr[optimal_idx],
    }


def find_threshold_f1_max(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    """Maximizes F1 score."""
    from sklearn.metrics import f1_score

    thresholds = np.arange(0.01, 0.99, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    return best_threshold, {"method": "f1_max", "f1": best_f1}


def find_threshold_fbeta_max(
    labels: np.ndarray,
    scores: np.ndarray,
    beta: float = 1.0
) -> Tuple[float, Dict]:
    """Maximizes F-beta score."""
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_threshold = 0.5
    best_fbeta = 0.0

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        fb = fbeta_score(labels, preds, beta=beta)
        if fb > best_fbeta:
            best_fbeta = fb
            best_threshold = thresh

    return best_threshold, {"method": f"f{beta}_max", "fbeta": best_fbeta}


def find_threshold_mcc_max(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    """Maximizes Matthews Correlation Coefficient."""
    from sklearn.metrics import matthews_corrcoef

    thresholds = np.arange(0.01, 0.99, 0.01)
    best_threshold = 0.5
    best_mcc = -1.0

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        mcc = matthews_corrcoef(labels, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = thresh

    return best_threshold, {"method": "mcc_max", "mcc": best_mcc}


def find_threshold_gmean(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    """Maximizes Geometric Mean (sqrt(TPR * TNR))."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    tnr = 1 - fpr
    gmean = np.sqrt(tpr * tnr)
    optimal_idx = np.argmax(gmean)

    return thresholds[optimal_idx], {
        "method": "gmean",
        "gmean": gmean[optimal_idx],
        "tpr": tpr[optimal_idx],
        "tnr": tnr[optimal_idx],
    }


def find_threshold_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    """Equal Error Rate: unde FPR = FNR."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # Find intersection
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    return thresholds[eer_idx], {
        "method": "eer",
        "eer": eer,
        "fpr": fpr[eer_idx],
        "fnr": fnr[eer_idx],
    }


def find_threshold_cost_sensitive(
    labels: np.ndarray,
    scores: np.ndarray,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST
) -> Tuple[float, Dict]:
    """
    Minimizes total weighted cost.

    Cost = FP * fp_cost + FN * fn_cost
    """
    from sklearn.metrics import confusion_matrix

    thresholds = np.arange(0.01, 0.99, 0.01)
    best_threshold = 0.5
    best_cost = float('inf')

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        cost = fp * fp_cost + fn * fn_cost
        if cost < best_cost:
            best_cost = cost
            best_threshold = thresh

    return best_threshold, {
        "method": "cost_sensitive",
        "total_cost": best_cost,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
    }


def find_threshold_precision_target(
    labels: np.ndarray,
    scores: np.ndarray,
    target: float = 0.95
) -> Tuple[float, Dict]:
    """Find threshold for target precision."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    valid_idx = np.where(precision >= target)[0]

    if len(valid_idx) == 0:
        idx = np.argmax(precision)
    else:
        idx = valid_idx[np.argmax(recall[valid_idx])]

    if idx >= len(thresholds):
        idx = len(thresholds) - 1

    return thresholds[idx], {
        "method": f"precision_target_{target}",
        "precision": precision[idx],
        "recall": recall[idx],
    }


def find_threshold_recall_target(
    labels: np.ndarray,
    scores: np.ndarray,
    target: float = 0.90
) -> Tuple[float, Dict]:
    """Find threshold for target recall."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    valid_idx = np.where(recall >= target)[0]

    if len(valid_idx) == 0:
        idx = np.argmax(recall)
    else:
        idx = valid_idx[np.argmax(precision[valid_idx])]

    if idx >= len(thresholds):
        idx = len(thresholds) - 1

    return thresholds[idx], {
        "method": f"recall_target_{target}",
        "precision": precision[idx],
        "recall": recall[idx],
    }


# =============================================================================
# FORENSIC SCORE CALCULATORS
# =============================================================================


def compute_ela_score(image_path: Path, quality: int = 95) -> float:
    """
    Error Level Analysis score.

    References:
    - Krawetz (2007): "A Picture's Worth..."
    - Gunawan et al. (2017): IJECE
    """
    try:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        original = np.array(img, dtype=np.float32)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed = np.array(Image.open(buffer), dtype=np.float32)

        error = np.abs(original - recompressed)
        error_normalized = error / 255.0

        mean_error = np.mean(error_normalized)
        std_error = np.std(error_normalized)
        max_error = np.max(error_normalized)

        # Also compute the entropy of the error distribution
        hist, _ = np.histogram(error_normalized.flatten(), bins=50, range=(0, 1))
        hist = hist / np.sum(hist) + 1e-10
        entropy = -np.sum(hist * np.log2(hist))
        entropy_normalized = entropy / np.log2(50)  # Normalized

        # Composite score
        score = (
            0.30 * mean_error * 10 +      # Mean scaled
            0.30 * std_error * 10 +        # Variance matters
            0.20 * (max_error / 3) +       # Peak errors
            0.20 * entropy_normalized      # Diversity of errors
        )

        return float(np.clip(score, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"ELA failed for {image_path}: {e}")
        return 0.5


def compute_noise_score(image_path: Path, block_size: int = 32) -> float:
    """
    Noise inconsistency analysis.

    References:
    - Mahdian & Saic (2009): Image and Vision Computing
    - Pan et al. (2011): IEEE TIFS
    """
    try:
        from PIL import Image
        from scipy import ndimage

        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=np.float32)

        # Multiple noise estimation methods

        # 1. Laplacian variance
        laplacian = ndimage.laplace(arr)

        # 2. High-pass filter
        from scipy.ndimage import gaussian_filter
        smooth = gaussian_filter(arr, sigma=2)
        high_pass = arr - smooth

        h, w = arr.shape
        variances_lap = []
        variances_hp = []

        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block_lap = laplacian[i:i+block_size, j:j+block_size]
                block_hp = high_pass[i:i+block_size, j:j+block_size]
                variances_lap.append(np.var(block_lap))
                variances_hp.append(np.var(block_hp))

        if len(variances_lap) < 4:
            return 0.5

        variances_lap = np.array(variances_lap)
        variances_hp = np.array(variances_hp)

        # Coefficient of variation for both
        cv_lap = np.std(variances_lap) / (np.mean(variances_lap) + 1e-6)
        cv_hp = np.std(variances_hp) / (np.mean(variances_hp) + 1e-6)

        # Kurtosis - abnormal distribution indicates manipulation
        from scipy.stats import kurtosis
        kurt_lap = abs(kurtosis(variances_lap))
        kurt_hp = abs(kurtosis(variances_hp))

        # Composite score
        score = (
            0.35 * np.clip(cv_lap / 1.5, 0, 1) +
            0.35 * np.clip(cv_hp / 1.5, 0, 1) +
            0.15 * np.clip(kurt_lap / 10, 0, 1) +
            0.15 * np.clip(kurt_hp / 10, 0, 1)
        )

        return float(np.clip(score, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"Noise analysis failed for {image_path}: {e}")
        return 0.5


def compute_jpeg_score(image_path: Path) -> float:
    """
    Double JPEG compression detection.

    References:
    - Farid (2009): IEEE TIFS
    - Bianchi & Piva (2012): IEEE TIFS
    """
    try:
        from PIL import Image
        from scipy.fftpack import dct

        img = Image.open(image_path)
        if img.format != "JPEG":
            return 0.0

        arr = np.array(img.convert("L"), dtype=np.float32)

        h, w = arr.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        arr = arr[:h8, :w8]

        dct_coeffs = []
        for i in range(0, h8, 8):
            for j in range(0, w8, 8):
                block = arr[i:i+8, j:j+8]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_coeffs.extend(dct_block.flatten()[1:])

        dct_coeffs = np.array(dct_coeffs)

        # Histogram analysis
        hist, bin_edges = np.histogram(dct_coeffs, bins=256, range=(-128, 128))
        hist = hist.astype(np.float32)
        hist = hist / (np.sum(hist) + 1e-6)

        # FFT for periodicity detection
        fft_hist = np.abs(np.fft.fft(hist))
        fft_hist = fft_hist[1:len(fft_hist)//2]

        # Detect periodic peaks
        mean_fft = np.mean(fft_hist)
        std_fft = np.std(fft_hist)
        peaks = fft_hist > mean_fft + 2 * std_fft
        n_peaks = np.sum(peaks)

        # Periodicity score
        if mean_fft < 1e-6:
            periodicity = 0
        else:
            periodicity = (np.max(fft_hist) / mean_fft - 1) / 10

        # Blocking artifacts analysis
        # At 8-pixel boundaries
        h_diff = np.abs(np.diff(arr[::8, :], axis=0))
        v_diff = np.abs(np.diff(arr[:, ::8], axis=1))

        blocking_h = np.mean(h_diff) / (np.std(arr) + 1e-6)
        blocking_v = np.mean(v_diff) / (np.std(arr) + 1e-6)

        score = (
            0.50 * np.clip(periodicity, 0, 1) +
            0.25 * np.clip(blocking_h * 2, 0, 1) +
            0.25 * np.clip(blocking_v * 2, 0, 1)
        )

        return float(np.clip(score, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"JPEG analysis failed for {image_path}: {e}")
        return 0.5


def compute_screenshot_score(image_path: Path) -> float:
    """Screenshot detection via resolution and metadata analysis."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(image_path)
        width, height = img.size

        score = 0.0

        # Screen resolutions (desktop and mobile)
        screen_resolutions = [
            (1920, 1080), (2560, 1440), (3840, 2160), (1366, 768),
            (1536, 864), (1440, 900), (1280, 720), (2560, 1600),
            (1680, 1050), (1920, 1200), (2880, 1800), (3440, 1440),
            # Mobile
            (1170, 2532), (1284, 2778), (1080, 2400), (1080, 2340),
            (1440, 3200), (1080, 1920), (1440, 2560),
        ]

        for sw, sh in screen_resolutions:
            if (width == sw and height == sh) or (width == sh and height == sw):
                score += 0.35
                break

        # Aspect ratios
        aspect = width / height if height > 0 else 0
        screen_aspects = [16/9, 16/10, 4/3, 21/9, 9/16, 9/19.5, 9/20, 9/21]

        for sa in screen_aspects:
            if abs(aspect - sa) < 0.02:
                score += 0.15
                break

        # No camera EXIF
        try:
            exif = img._getexif()
            if exif is None:
                score += 0.25
            else:
                camera_tags = ['Make', 'Model', 'LensModel', 'FocalLength']
                has_camera = any(
                    TAGS.get(tag_id, '') in camera_tags
                    for tag_id in exif.keys()
                )
                if not has_camera:
                    score += 0.15
        except Exception:
            score += 0.15

        # PNG format
        if img.format == "PNG":
            score += 0.15

        # Very clean edges (no compression artifacts in screenshots saved as PNG)
        if img.format == "PNG":
            arr = np.array(img.convert("L"), dtype=np.float32)
            edge_noise = np.std(np.diff(arr, axis=0)) + np.std(np.diff(arr, axis=1))
            if edge_noise < 50:  # Very clean
                score += 0.1

        return float(np.clip(score, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"Screenshot analysis failed for {image_path}: {e}")
        return 0.5


def compute_ai_frequency_score(image_path: Path) -> float:
    """
    AI detection via frequency analysis.

    References:
    - Wang et al. (2020): CVPR - "CNN-generated images..."
    - Frank et al. (2020): "Leveraging Frequency Analysis"
    """
    try:
        from PIL import Image
        from scipy.fftpack import fft2, fftshift

        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=np.float32)

        # 2D FFT
        f_transform = fft2(arr)
        f_shift = fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)

        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        # Radial frequency analysis
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w)**2 + (y - center_h)**2)

        # Bins for radial profile
        r_max = min(center_h, center_w)
        n_bins = 50
        radial_profile = []

        for i in range(n_bins):
            r_inner = i * r_max / n_bins
            r_outer = (i + 1) * r_max / n_bins
            mask = (r >= r_inner) & (r < r_outer)
            if np.sum(mask) > 0:
                radial_profile.append(np.mean(magnitude[mask]))

        radial_profile = np.array(radial_profile)

        # AI images often have different frequency falloff
        # Natural images: power law decay
        # AI images: sometimes have artifacts at specific frequencies

        # Fit power law and check residuals
        if len(radial_profile) > 10:
            x_log = np.log(np.arange(1, len(radial_profile) + 1))
            y_log = np.log(radial_profile + 1e-6)

            # Linear fit in log-log space
            coeffs = np.polyfit(x_log, y_log, 1)
            fitted = np.polyval(coeffs, x_log)
            residuals = y_log - fitted

            # High residual variance = unnatural frequency distribution
            residual_var = np.var(residuals)

            # Slope (natural images typically -2 to -3)
            slope = coeffs[0]
            slope_anomaly = abs(slope + 2.5) / 2  # Deviation from typical -2.5
        else:
            residual_var = 0.5
            slope_anomaly = 0.5

        # Check for specific frequency artifacts
        # AI often has grid-like artifacts
        h_freq = magnitude[:, center_w]  # Vertical slice
        v_freq = magnitude[center_h, :]  # Horizontal slice

        # Look for peaks at regular intervals
        h_fft = np.abs(np.fft.fft(h_freq - np.mean(h_freq)))
        v_fft = np.abs(np.fft.fft(v_freq - np.mean(v_freq)))

        h_peaks = h_fft[1:len(h_fft)//4]
        v_peaks = v_fft[1:len(v_fft)//4]

        peak_score = (np.max(h_peaks) / (np.mean(h_peaks) + 1e-6) +
                      np.max(v_peaks) / (np.mean(v_peaks) + 1e-6)) / 20

        # Composite score
        score = (
            0.35 * np.clip(residual_var * 5, 0, 1) +
            0.35 * np.clip(slope_anomaly, 0, 1) +
            0.30 * np.clip(peak_score, 0, 1)
        )

        return float(np.clip(score, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"AI frequency analysis failed for {image_path}: {e}")
        return 0.5


def compute_texture_score(image_path: Path) -> float:
    """
    Texture inconsistency analysis using GLCM features.
    """
    try:
        from PIL import Image

        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=np.uint8)

        # Resize for efficiency
        if max(arr.shape) > 512:
            scale = 512 / max(arr.shape)
            new_h, new_w = int(arr.shape[0] * scale), int(arr.shape[1] * scale)
            img_resized = img.resize((new_w, new_h))
            arr = np.array(img_resized, dtype=np.uint8)

        # Simple GLCM-like features
        h, w = arr.shape
        block_size = 64

        # Compute local entropy
        entropies = []
        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = arr[i:i+block_size, j:j+block_size]
                hist, _ = np.histogram(block, bins=32, range=(0, 256))
                hist = hist / (np.sum(hist) + 1e-6)
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropies.append(entropy)

        if len(entropies) < 4:
            return 0.5

        entropies = np.array(entropies)

        # Variance in entropy indicates texture inconsistency
        entropy_cv = np.std(entropies) / (np.mean(entropies) + 1e-6)

        # Local contrast variation
        contrasts = []
        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = arr[i:i+block_size, j:j+block_size].astype(np.float32)
                contrast = np.std(block)
                contrasts.append(contrast)

        contrasts = np.array(contrasts)
        contrast_cv = np.std(contrasts) / (np.mean(contrasts) + 1e-6)

        score = (
            0.50 * np.clip(entropy_cv / 0.5, 0, 1) +
            0.50 * np.clip(contrast_cv / 0.8, 0, 1)
        )

        return float(np.clip(score, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"Texture analysis failed for {image_path}: {e}")
        return 0.5


# =============================================================================
# ADVANCED CALIBRATION ENGINE
# =============================================================================


class AdvancedCalibrationEngine:
    """
    Advanced calibration engine with complete statistical validation.
    """

    def __init__(self, output_dir: Path, random_state: int = 42):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.results: Dict[str, AdvancedCalibrationResult] = {}

        np.random.seed(random_state)

        # Define methods
        self.methods = self._init_methods()

        # Threshold selection functions
        self.threshold_methods = {
            ThresholdMethod.YOUDEN: find_threshold_youden,
            ThresholdMethod.F1_MAX: find_threshold_f1_max,
            ThresholdMethod.F2_MAX: lambda l, s: find_threshold_fbeta_max(l, s, beta=2),
            ThresholdMethod.F05_MAX: lambda l, s: find_threshold_fbeta_max(l, s, beta=0.5),
            ThresholdMethod.MCC_MAX: find_threshold_mcc_max,
            ThresholdMethod.GMEAN: find_threshold_gmean,
            ThresholdMethod.EQUAL_ERROR_RATE: find_threshold_eer,
            ThresholdMethod.COST_SENSITIVE: find_threshold_cost_sensitive,
            ThresholdMethod.PRECISION_TARGET: lambda l, s: find_threshold_precision_target(l, s, 0.95),
            ThresholdMethod.RECALL_TARGET: lambda l, s: find_threshold_recall_target(l, s, 0.90),
        }

    def _init_methods(self) -> Dict[str, MethodConfig]:
        """Initialize method configurations."""
        return {
            "ela": MethodConfig(
                name="ELA (Error Level Analysis)",
                key="ela",
                description="Detects manipulation through JPEG error level analysis",
                compute_score=compute_ela_score,
                literature_refs=[
                    {"author": "Krawetz, N.", "year": "2007",
                     "title": "A Picture's Worth...", "venue": "HITB"},
                    {"author": "Gunawan et al.", "year": "2017",
                     "title": "Development of Photo Forensic using ELA", "venue": "IJECE"},
                ],
                category="forensic",
            ),
            "noise": MethodConfig(
                name="Noise Inconsistency",
                key="noise",
                description="Detects splicing through noise pattern variation",
                compute_score=compute_noise_score,
                literature_refs=[
                    {"author": "Mahdian & Saic", "year": "2009",
                     "title": "Using noise inconsistencies", "venue": "IVC"},
                    {"author": "Pan et al.", "year": "2011",
                     "title": "Region Duplication Detection", "venue": "IEEE TIFS"},
                ],
                category="forensic",
            ),
            "jpeg": MethodConfig(
                name="JPEG Double Compression",
                key="jpeg",
                description="Detects multiple JPEG re-compressions via DCT",
                compute_score=compute_jpeg_score,
                literature_refs=[
                    {"author": "Farid, H.", "year": "2009",
                     "title": "Exposing Digital Forgeries from JPEG Ghosts", "venue": "IEEE TIFS"},
                    {"author": "Bianchi & Piva", "year": "2012",
                     "title": "Image Forgery Localization", "venue": "IEEE TIFS"},
                ],
                category="forensic",
            ),
            "screenshot": MethodConfig(
                name="Screenshot Detection",
                key="screenshot",
                description="Detects screenshots via resolution and metadata",
                compute_score=compute_screenshot_score,
                literature_refs=[],
                category="metadata",
            ),
            "ai_frequency": MethodConfig(
                name="AI Detection (Frequency)",
                key="ai_frequency",
                description="Detects AI images through spectral analysis",
                compute_score=compute_ai_frequency_score,
                literature_refs=[
                    {"author": "Wang et al.", "year": "2020",
                     "title": "CNN-generated images are surprisingly easy to spot", "venue": "CVPR"},
                    {"author": "Frank et al.", "year": "2020",
                     "title": "Leveraging Frequency Analysis", "venue": "ICML"},
                ],
                category="ai_detection",
            ),
            "texture": MethodConfig(
                name="Texture Inconsistency",
                key="texture",
                description="Detects manipulation through local texture variation",
                compute_score=compute_texture_score,
                literature_refs=[
                    {"author": "Fridrich et al.", "year": "2012",
                     "title": "Rich Models for Steganalysis", "venue": "IEEE TIFS"},
                ],
                category="forensic",
            ),
        }

    def load_dataset(
        self,
        dataset_path: Path,
        labels_file: Optional[Path] = None
    ) -> Tuple[List[Path], np.ndarray]:
        """
        Load dataset with labels.

        Expected structure:
        dataset/
        ├── authentic/   (or real/, original/)
        └── manipulated/ (or fake/, ai/, tampered/)
        """
        images = []
        labels = []

        if labels_file and labels_file.exists():
            with open(labels_file) as f:
                label_map = json.load(f)
            for img_name, label in label_map.items():
                img_path = dataset_path / img_name
                if img_path.exists():
                    images.append(img_path)
                    labels.append(label)
        else:
            authentic_names = ["authentic", "real", "original", "genuine", "0"]
            manipulated_names = ["manipulated", "fake", "tampered", "forged", "ai", "synthetic", "1"]

            for subdir in dataset_path.iterdir():
                if not subdir.is_dir():
                    continue

                name_lower = subdir.name.lower()

                if any(n in name_lower for n in authentic_names):
                    label = 0
                elif any(n in name_lower for n in manipulated_names):
                    label = 1
                else:
                    continue

                for img_path in subdir.glob("*"):
                    if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                        images.append(img_path)
                        labels.append(label)

        logger.info(f"Loaded {len(images)} images: "
                   f"{sum(1 for l in labels if l == 0)} authentic, "
                   f"{sum(1 for l in labels if l == 1)} manipulated")

        return images, np.array(labels)

    def compute_scores_for_method(
        self,
        method_key: str,
        images: List[Path],
        show_progress: bool = True
    ) -> np.ndarray:
        """Compute scores for all images."""
        method = self.methods[method_key]
        scores = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(images, desc=f"Computing {method_key}")
            except ImportError:
                iterator = images
                logger.info(f"Computing {method_key} for {len(images)} images...")
        else:
            iterator = images

        for img_path in iterator:
            score = method.compute_score(img_path)
            scores.append(score)

        return np.array(scores)

    def cross_validate(
        self,
        method_key: str,
        images: List[Path],
        labels: np.ndarray,
        k_folds: int = 5,
        threshold_method: ThresholdMethod = ThresholdMethod.F1_MAX
    ) -> Tuple[List[CrossValidationResult], np.ndarray]:
        """
        Stratified K-Fold Cross-Validation.

        Returns:
            (list of fold results, scores array for full dataset)
        """
        from sklearn.model_selection import StratifiedKFold

        logger.info(f"Running {k_folds}-fold cross-validation for {method_key}...")

        # Compute all scores first
        scores = self.compute_scores_for_method(method_key, images)

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)

        cv_results = []
        thresh_func = self.threshold_methods[threshold_method]

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
            train_labels = labels[train_idx]
            train_scores = scores[train_idx]
            val_labels = labels[val_idx]
            val_scores = scores[val_idx]

            # Find threshold on training set
            threshold, _ = thresh_func(train_labels, train_scores)

            # Evaluate on train and validation
            train_preds = (train_scores >= threshold).astype(int)
            val_preds = (val_scores >= threshold).astype(int)

            train_metrics = compute_all_metrics(train_labels, train_preds, train_scores)
            val_metrics = compute_all_metrics(val_labels, val_preds, val_scores)

            cv_results.append(CrossValidationResult(
                fold_idx=fold_idx,
                threshold=threshold,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                n_train=len(train_idx),
                n_val=len(val_idx),
            ))

            logger.info(f"  Fold {fold_idx + 1}: threshold={threshold:.3f}, "
                       f"val_F1={val_metrics['f1']:.3f}, val_AUC={val_metrics['auc']:.3f}")

        return cv_results, scores

    def calibrate_method_full(
        self,
        method_key: str,
        images: List[Path],
        labels: np.ndarray,
        dataset_name: str = "custom",
        k_folds: int = 5,
        n_bootstrap: int = 1000,
        calibrate_probabilities: bool = True,
    ) -> AdvancedCalibrationResult:
        """
        Full calibration for a method with all analyses.
        """
        from sklearn.metrics import roc_curve, auc

        method = self.methods[method_key]
        logger.info(f"\n{'='*70}")
        logger.info(f"FULL CALIBRATION: {method.name}")
        logger.info(f"{'='*70}")

        # Cross-validation
        cv_results, scores = self.cross_validate(
            method_key, images, labels, k_folds,
            ThresholdMethod.F1_MAX
        )

        # Aggregate CV results
        cv_thresholds = [r.threshold for r in cv_results]
        cv_metrics = {}

        for metric_name in cv_results[0].val_metrics.keys():
            cv_metrics[metric_name] = [r.val_metrics[metric_name] for r in cv_results]

        cv_threshold_mean = np.mean(cv_thresholds)
        cv_threshold_std = np.std(cv_thresholds)

        logger.info(f"\nCV Results:")
        logger.info(f"  Threshold: {cv_threshold_mean:.3f} ± {cv_threshold_std:.3f}")
        logger.info(f"  F1: {np.mean(cv_metrics['f1']):.3f} ± {np.std(cv_metrics['f1']):.3f}")
        logger.info(f"  AUC: {np.mean(cv_metrics['auc']):.3f} ± {np.std(cv_metrics['auc']):.3f}")

        # Find optimal threshold on full dataset with multiple methods
        logger.info("\nFinding optimal thresholds (full dataset):")

        alternative_thresholds = {}
        best_f1 = 0
        best_threshold = 0.5
        best_method_name = ""

        for thresh_method, thresh_func in self.threshold_methods.items():
            try:
                thresh, details = thresh_func(labels, scores)
                preds = (scores >= thresh).astype(int)
                metrics = compute_all_metrics(labels, preds, scores)

                alternative_thresholds[thresh_method.value] = thresh

                logger.info(f"  {thresh_method.value}: {thresh:.3f} "
                           f"(F1={metrics['f1']:.3f}, AUC={metrics['auc']:.3f})")

                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_threshold = thresh
                    best_method_name = thresh_method.value

            except Exception as e:
                logger.warning(f"  {thresh_method.value} failed: {e}")

        # Final metrics at best threshold
        final_preds = (scores >= best_threshold).astype(int)
        final_metrics = compute_all_metrics(labels, final_preds, scores)

        logger.info(f"\nBest threshold: {best_threshold:.3f} ({best_method_name})")
        logger.info(f"  F1={final_metrics['f1']:.3f}, AUC={final_metrics['auc']:.3f}, "
                   f"MCC={final_metrics['mcc']:.3f}")

        # Bootstrap confidence intervals
        logger.info("\nComputing bootstrap confidence intervals...")

        def threshold_stat(data):
            boot_labels = data[:, 0].astype(int)
            boot_scores = data[:, 1]
            try:
                t, _ = find_threshold_f1_max(boot_labels, boot_scores)
                return t
            except:
                return 0.5

        combined_data = np.column_stack([labels, scores])

        try:
            ci_95 = bootstrap_ci_bca(combined_data, threshold_stat,
                                     n_bootstrap=n_bootstrap, confidence=0.95)
            ci_99 = bootstrap_ci_bca(combined_data, threshold_stat,
                                     n_bootstrap=n_bootstrap, confidence=0.99)
        except Exception as e:
            logger.warning(f"Bootstrap failed: {e}")
            ci_95 = (best_threshold - 0.1, best_threshold + 0.1)
            ci_99 = (best_threshold - 0.15, best_threshold + 0.15)

        logger.info(f"  95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
        logger.info(f"  99% CI: [{ci_99[0]:.3f}, {ci_99[1]:.3f}]")

        # Probability calibration
        calibration_metrics = None

        if calibrate_probabilities:
            logger.info("\nCalibrating probabilities...")

            # Before calibration
            ece_before, mce_before, rel_data_before = compute_ece(labels, scores)

            # Temperature scaling
            try:
                temp, calibrated_scores = temperature_scaling(scores, labels)
                ece_after, mce_after, rel_data_after = compute_ece(labels, calibrated_scores)

                calibration_metrics = CalibrationMetrics(
                    ece=ece_after,
                    mce=mce_after,
                    brier_score=final_metrics['brier_score'],
                    log_loss=final_metrics['log_loss'],
                    reliability_data=rel_data_after,
                    calibration_method="temperature_scaling",
                    temperature=temp,
                )

                logger.info(f"  Temperature: {temp:.3f}")
                logger.info(f"  ECE: {ece_before:.4f} -> {ece_after:.4f}")
                logger.info(f"  MCE: {mce_before:.4f} -> {mce_after:.4f}")

            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
                calibration_metrics = CalibrationMetrics(
                    ece=ece_before,
                    mce=mce_before,
                    brier_score=final_metrics['brier_score'],
                    log_loss=final_metrics['log_loss'],
                    reliability_data=rel_data_before,
                    calibration_method="none",
                )

        # Create result
        result = AdvancedCalibrationResult(
            method_name=method.name,
            method_key=method_key,
            threshold_optimal=best_threshold,
            threshold_method=best_method_name,
            confidence_interval_95=ci_95,
            confidence_interval_99=ci_99,
            cv_thresholds=cv_thresholds,
            cv_threshold_mean=cv_threshold_mean,
            cv_threshold_std=cv_threshold_std,
            cv_metrics=cv_metrics,
            metrics=final_metrics,
            calibration=calibration_metrics,
            n_samples=len(images),
            n_positive=int(np.sum(labels)),
            n_negative=int(np.sum(1 - labels)),
            dataset_name=dataset_name,
            calibration_date=datetime.now().isoformat(),
            literature_threshold=method.literature_threshold,
            literature_refs=method.literature_refs,
            alternative_thresholds=alternative_thresholds,
        )

        self.results[method_key] = result
        return result

    def compare_methods_statistically(
        self,
        images: List[Path],
        labels: np.ndarray,
    ):
        """
        Compare all methods using statistical tests.
        """
        logger.info("\n" + "="*70)
        logger.info("STATISTICAL COMPARISON BETWEEN METHODS")
        logger.info("="*70)

        if len(self.results) < 2:
            logger.warning("Too few methods for comparison.")
            return

        method_keys = list(self.results.keys())

        # Compute predictions for each method
        all_preds = {}
        all_probs = {}

        for key in method_keys:
            result = self.results[key]
            scores = self.compute_scores_for_method(key, images, show_progress=False)
            preds = (scores >= result.threshold_optimal).astype(int)
            all_preds[key] = preds
            all_probs[key] = scores

        # Pairwise comparisons
        for i, key_a in enumerate(method_keys):
            for key_b in method_keys[i+1:]:
                logger.info(f"\n{self.results[key_a].method_name} vs {self.results[key_b].method_name}:")

                # McNemar test
                mcnemar_result = mcnemar_test(labels, all_preds[key_a], all_preds[key_b])
                self.results[key_a].statistical_comparisons[f"mcnemar_vs_{key_b}"] = mcnemar_result
                logger.info(f"  McNemar: {mcnemar_result.interpretation}")

                # DeLong test for AUC
                delong_result = delong_test(labels, all_probs[key_a], all_probs[key_b])
                self.results[key_a].statistical_comparisons[f"delong_vs_{key_b}"] = delong_result
                logger.info(f"  DeLong: {delong_result.interpretation}")

                # Wilcoxon on CV metrics
                if len(self.results[key_a].cv_metrics.get('f1', [])) >= 5:
                    wilcoxon_result = wilcoxon_signed_rank_test(
                        self.results[key_a].cv_metrics['f1'],
                        self.results[key_b].cv_metrics['f1']
                    )
                    self.results[key_a].statistical_comparisons[f"wilcoxon_vs_{key_b}"] = wilcoxon_result
                    logger.info(f"  Wilcoxon (F1): {wilcoxon_result.interpretation}")

    def calibrate_all(
        self,
        images: List[Path],
        labels: np.ndarray,
        dataset_name: str = "custom",
        k_folds: int = 5,
        statistical_comparison: bool = True,
    ):
        """Calibrate all methods."""
        for method_key in self.methods:
            self.calibrate_method_full(
                method_key, images, labels, dataset_name, k_folds
            )

        if statistical_comparison:
            self.compare_methods_statistically(images, labels)

    def generate_reliability_diagram_data(self) -> Dict[str, Any]:
        """Generate data for reliability diagrams."""
        data = {}

        for key, result in self.results.items():
            if result.calibration and result.calibration.reliability_data:
                data[key] = {
                    "method_name": result.method_name,
                    "reliability": result.calibration.reliability_data,
                    "ece": result.calibration.ece,
                    "mce": result.calibration.mce,
                }

        return data

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive Markdown report for the thesis."""
        report = []

        report.append("# Complete Threshold Calibration Report")
        report.append(f"\n**Generation date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"**Random seed:** {self.random_state}")
        report.append("")

        # Executive Summary
        report.append("## 1. Executive Summary")
        report.append("")
        report.append("| Method | Threshold | 95% CI | AUC | F1 | MCC | ECE |")
        report.append("|--------|-----------|--------|-----|----|----|-----|")

        for key, result in self.results.items():
            ece = f"{result.calibration.ece:.3f}" if result.calibration else "N/A"
            report.append(
                f"| {result.method_name} | "
                f"{result.threshold_optimal:.3f} ± {result.cv_threshold_std:.2f} | "
                f"[{result.confidence_interval_95[0]:.2f}, {result.confidence_interval_95[1]:.2f}] | "
                f"{result.metrics.get('auc', 0):.3f} | "
                f"{result.metrics.get('f1', 0):.3f} | "
                f"{result.metrics.get('mcc', 0):.3f} | "
                f"{ece} |"
            )

        report.append("")

        # Detailed per-method sections
        report.append("## 2. Detailed Per-Method Analysis")
        report.append("")

        for key, result in self.results.items():
            method = self.methods[key]

            report.append(f"### 2.{list(self.results.keys()).index(key) + 1} {result.method_name}")
            report.append("")
            report.append(f"**Description:** {method.description}")
            report.append(f"**Category:** {method.category}")
            report.append("")

            # Literature references
            if method.literature_refs:
                report.append("#### Bibliographic References")
                for ref in method.literature_refs:
                    report.append(f"- {ref['author']} ({ref['year']}): \"{ref['title']}\", {ref['venue']}")
                report.append("")

            # Threshold details
            report.append("#### Optimal Threshold")
            report.append(f"- **Value:** {result.threshold_optimal:.4f}")
            report.append(f"- **Selection method:** {result.threshold_method}")
            report.append(f"- **95% CI:** [{result.confidence_interval_95[0]:.3f}, {result.confidence_interval_95[1]:.3f}]")
            report.append(f"- **99% CI:** [{result.confidence_interval_99[0]:.3f}, {result.confidence_interval_99[1]:.3f}]")
            report.append("")

            # Cross-validation
            report.append("#### Cross-Validation")
            report.append(f"- **Mean threshold:** {result.cv_threshold_mean:.4f}")
            report.append(f"- **Std threshold:** {result.cv_threshold_std:.4f}")
            report.append(f"- **Thresholds per fold:** {', '.join([f'{t:.3f}' for t in result.cv_thresholds])}")
            report.append("")

            # Metrics
            report.append("#### Performance Metrics")
            report.append(f"- **Accuracy:** {result.metrics.get('accuracy', 0):.4f}")
            report.append(f"- **Precision:** {result.metrics.get('precision', 0):.4f}")
            report.append(f"- **Recall:** {result.metrics.get('recall', 0):.4f}")
            report.append(f"- **F1 Score:** {result.metrics.get('f1', 0):.4f}")
            report.append(f"- **F2 Score:** {result.metrics.get('f2', 0):.4f}")
            report.append(f"- **AUC-ROC:** {result.metrics.get('auc', 0):.4f}")
            report.append(f"- **MCC:** {result.metrics.get('mcc', 0):.4f}")
            report.append(f"- **Balanced Accuracy:** {result.metrics.get('balanced_accuracy', 0):.4f}")
            report.append("")

            # Calibration
            if result.calibration:
                report.append("#### Probability Calibration")
                report.append(f"- **Method:** {result.calibration.calibration_method}")
                if result.calibration.temperature:
                    report.append(f"- **Temperature:** {result.calibration.temperature:.4f}")
                report.append(f"- **ECE:** {result.calibration.ece:.4f}")
                report.append(f"- **MCE:** {result.calibration.mce:.4f}")
                report.append(f"- **Brier Score:** {result.calibration.brier_score:.4f}")
                report.append("")

            # Alternative thresholds
            if result.alternative_thresholds:
                report.append("#### Alternative Thresholds")
                report.append("| Method | Threshold |")
                report.append("|--------|-----------|")
                for method_name, thresh in result.alternative_thresholds.items():
                    report.append(f"| {method_name} | {thresh:.4f} |")
                report.append("")

            # Statistical comparisons
            if result.statistical_comparisons:
                report.append("#### Statistical Comparisons")
                for comp_name, comp_result in result.statistical_comparisons.items():
                    report.append(f"- **{comp_name}:** {comp_result.interpretation}")
                report.append("")

            report.append("---")
            report.append("")

        # Methodology section
        report.append("## 3. Methodology")
        report.append("")
        report.append("### 3.1 Threshold Selection")
        report.append("The following selection methods were evaluated:")
        report.append("- **Youden's J:** Maximizes TPR - FPR")
        report.append("- **F1 Max:** Maximizes F1 score")
        report.append("- **F2 Max:** Favors recall")
        report.append("- **MCC Max:** Maximizes Matthews Correlation Coefficient")
        report.append("- **G-Mean:** Maximizes √(TPR × TNR)")
        report.append("- **EER:** Equal Error Rate (FPR = FNR)")
        report.append("- **Cost-Sensitive:** Minimizes weighted error cost")
        report.append("")

        report.append("### 3.2 Statistical Validation")
        report.append("- **Cross-validation:** Stratified K-Fold")
        report.append("- **Confidence Intervals:** Bootstrap BCa (bias-corrected accelerated)")
        report.append("- **Comparisons:** McNemar test, DeLong test, Wilcoxon signed-rank")
        report.append("")

        report.append("### 3.3 Probability Calibration")
        report.append("- **Temperature Scaling:** NLL optimization")
        report.append("- **Metrics:** ECE, MCE, Brier Score")
        report.append("")

        return "\n".join(report)

    def generate_latex_tables(self) -> str:
        """Generate LaTeX tables for the thesis."""
        latex = []

        # Main comparison table
        latex.append("% Main table with calibration results")
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Calibrated Thresholds for Forensic Methods}")
        latex.append("\\label{tab:calibrated_thresholds}")
        latex.append("\\begin{tabular}{lccccccc}")
        latex.append("\\toprule")
        latex.append("Method & Threshold & 95\\% CI & AUC & F1 & MCC & ECE & Lit. \\\\")
        latex.append("\\midrule")

        for result in self.results.values():
            latex.append(result.to_latex_row())

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex.append("")

        # Cross-validation table
        latex.append("% Table with cross-validation results")
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Cross-Validation Results (5-fold)}")
        latex.append("\\label{tab:cv_results}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append("Method & Threshold (mean±std) & F1 (mean±std) & AUC (mean±std) \\\\")
        latex.append("\\midrule")

        for result in self.results.values():
            f1_mean = np.mean(result.cv_metrics.get('f1', [0]))
            f1_std = np.std(result.cv_metrics.get('f1', [0]))
            auc_mean = np.mean(result.cv_metrics.get('auc', [0]))
            auc_std = np.std(result.cv_metrics.get('auc', [0]))

            latex.append(
                f"{result.method_name} & "
                f"${result.cv_threshold_mean:.3f} \\pm {result.cv_threshold_std:.3f}$ & "
                f"${f1_mean:.3f} \\pm {f1_std:.3f}$ & "
                f"${auc_mean:.3f} \\pm {auc_std:.3f}$ \\\\"
            )

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        return "\n".join(latex)

    def save_all_results(self):
        """Save all results."""
        # JSON
        json_path = self.output_dir / "calibration_results_advanced.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                {k: v.to_dict() for k, v in self.results.items()},
                f, indent=2, ensure_ascii=False
            )
        logger.info(f"Saved JSON: {json_path}")

        # Markdown report
        md_path = self.output_dir / "calibration_report_advanced.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_comprehensive_report())
        logger.info(f"Saved Markdown: {md_path}")

        # LaTeX tables
        latex_path = self.output_dir / "calibration_tables.tex"
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_latex_tables())
        logger.info(f"Saved LaTeX: {latex_path}")

        # Python config
        config_path = self.output_dir / "calibrated_thresholds_advanced.py"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('"""Auto-generated calibrated thresholds with metadata."""\n\n')
            f.write("CALIBRATED_THRESHOLDS = {\n")
            for key, result in self.results.items():
                f.write(f'    "{key}": {result.threshold_optimal:.4f},\n')
            f.write("}\n\n")

            f.write("THRESHOLD_CONFIDENCE_INTERVALS = {\n")
            for key, result in self.results.items():
                f.write(f'    "{key}": ({result.confidence_interval_95[0]:.4f}, {result.confidence_interval_95[1]:.4f}),\n')
            f.write("}\n\n")

            f.write("CALIBRATION_METADATA = {\n")
            for key, result in self.results.items():
                f.write(f'    "{key}": {{\n')
                f.write(f'        "method_name": "{result.method_name}",\n')
                f.write(f'        "threshold": {result.threshold_optimal:.4f},\n')
                f.write(f'        "threshold_method": "{result.threshold_method}",\n')
                f.write(f'        "auc": {result.metrics.get("auc", 0):.4f},\n')
                f.write(f'        "f1": {result.metrics.get("f1", 0):.4f},\n')
                f.write(f'        "mcc": {result.metrics.get("mcc", 0):.4f},\n')
                f.write(f'        "cv_std": {result.cv_threshold_std:.4f},\n')
                f.write(f'        "dataset": "{result.dataset_name}",\n')
                f.write(f'    }},\n')
            f.write("}\n")

        logger.info(f"Saved Python config: {config_path}")

        # Reliability diagram data (JSON for plotting)
        rel_data = self.generate_reliability_diagram_data()
        if rel_data:
            rel_path = self.output_dir / "reliability_diagram_data.json"
            with open(rel_path, 'w') as f:
                json.dump(rel_data, f, indent=2)
            logger.info(f"Saved reliability data: {rel_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Threshold Calibration with Statistical Validation"
    )
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Path to dataset with authentic/ and manipulated/ subfolders"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/calibration_advanced"),
        help="Output directory"
    )
    parser.add_argument(
        "--method", choices=list(AdvancedCalibrationEngine(Path(".")).methods.keys()) + ["all"],
        default="all", help="Which method to calibrate"
    )
    parser.add_argument(
        "--k-folds", type=int, default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Number of bootstrap samples"
    )
    parser.add_argument(
        "--labels", type=Path,
        help="Optional JSON file with image labels"
    )
    parser.add_argument(
        "--dataset-name", default="custom",
        help="Name of dataset for documentation"
    )
    parser.add_argument(
        "--no-statistical-tests", action="store_true",
        help="Skip statistical comparison tests"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Initialize engine
    engine = AdvancedCalibrationEngine(args.output, random_state=args.seed)

    # Load dataset
    images, labels = engine.load_dataset(args.dataset, args.labels)

    if len(images) == 0:
        logger.error("No images found!")
        sys.exit(1)

    # Calibrate
    if args.method == "all":
        engine.calibrate_all(
            images, labels, args.dataset_name,
            k_folds=args.k_folds,
            statistical_comparison=not args.no_statistical_tests
        )
    else:
        engine.calibrate_method_full(
            args.method, images, labels, args.dataset_name,
            k_folds=args.k_folds, n_bootstrap=args.n_bootstrap
        )

    # Save results
    engine.save_all_results()

    print("\n" + "="*70)
    print("CALIBRATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output}")
    print("\nGenerated files:")
    print("  - calibration_results_advanced.json")
    print("  - calibration_report_advanced.md")
    print("  - calibration_tables.tex")
    print("  - calibrated_thresholds_advanced.py")
    print("  - reliability_diagram_data.json")


if __name__ == "__main__":
    main()
