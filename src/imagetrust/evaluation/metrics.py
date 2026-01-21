"""
Evaluation metrics for AI detection models.
"""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_proba: Predicted probabilities for positive class
        threshold: Classification threshold
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }
    
    # ROC-AUC (requires both classes present)
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["average_precision"] = average_precision_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = 0.0
        metrics["average_precision"] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)
        
        # Specificity
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def compute_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute ROC-AUC score.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        
    Returns:
        ROC-AUC score
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return roc_auc_score(y_true, y_proba)


def compute_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    return roc_curve(y_true, y_proba)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute calibration metrics.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of calibration bins
        
    Returns:
        Dictionary with calibration metrics
    """
    # Brier score
    brier = np.mean((y_proba - y_true) ** 2)
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )
    
    # Expected Calibration Error (ECE)
    bin_counts = np.histogram(y_proba, bins=n_bins, range=(0, 1))[0]
    total_samples = len(y_true)
    
    ece = 0.0
    for i in range(len(prob_true)):
        if bin_counts[i] > 0:
            ece += np.abs(prob_true[i] - prob_pred[i]) * (bin_counts[i] / total_samples)
    
    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0
    
    return {
        "brier_score": float(brier),
        "ece": float(ece),
        "mce": float(mce),
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
    }


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ("f1", "accuracy", "youden")
        
    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        elif metric == "youden":
            # Youden's J statistic
            recall = recall_score(y_true, y_pred, zero_division=0)
            specificity = recall_score(1 - y_true, 1 - y_pred, zero_division=0)
            score = recall + specificity - 1
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score
