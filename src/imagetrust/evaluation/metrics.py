"""
Evaluation metrics for AI detection models.
"""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
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


def compute_metrics_with_confidence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute metrics with bootstrap confidence intervals.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary with metrics and confidence intervals
    """
    from imagetrust.evaluation.statistical_tests import bootstrap_ci

    # Point estimates
    metrics = compute_metrics(y_true, y_pred, y_proba)

    # Add confidence intervals for key metrics
    metric_fns = {
        "accuracy": lambda yt, yp: accuracy_score(yt, (yp > 0.5).astype(int)),
        "f1_score": lambda yt, yp: f1_score(yt, (yp > 0.5).astype(int), zero_division=0),
        "roc_auc": lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.5,
    }

    confidence_intervals = {}
    for name, fn in metric_fns.items():
        try:
            ci_result = bootstrap_ci(
                fn,
                y_true,
                y_proba,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
            )
            confidence_intervals[name] = {
                "estimate": ci_result.estimate,
                "ci_lower": ci_result.ci_lower,
                "ci_upper": ci_result.ci_upper,
                "std_error": ci_result.std_error,
            }
        except Exception as e:
            logger.warning(f"Could not compute CI for {name}: {e}")

    metrics["confidence_intervals"] = confidence_intervals
    return metrics


def compute_metrics_with_significance(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    probabilities_dict: Dict[str, np.ndarray],
    reference_model: str = "ImageTrust",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute metrics for multiple models with pairwise significance tests.

    This function produces publication-ready comparison tables with:
    - Per-model metrics (Accuracy, F1, AUC, ECE)
    - McNemar's test for accuracy comparison
    - DeLong's test for AUC comparison
    - Bonferroni-corrected significance

    Args:
        y_true: Ground truth labels
        predictions_dict: Dictionary of {model_name: predictions}
        probabilities_dict: Dictionary of {model_name: probabilities}
        reference_model: Model to use as reference for comparisons
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with metrics and significance tests
    """
    from imagetrust.evaluation.statistical_tests import (
        compute_pairwise_significance,
        mcnemar_test,
        delong_test,
    )

    results = {
        "per_model_metrics": {},
        "significance_tests": {},
        "summary_table": [],
    }

    # Compute metrics for each model
    for model_name in predictions_dict.keys():
        y_pred = predictions_dict[model_name]
        y_proba = probabilities_dict.get(model_name)

        if y_proba is None:
            y_proba = y_pred.astype(float)

        metrics = compute_metrics(y_true, y_pred, y_proba)

        # Add calibration metrics
        cal_metrics = compute_calibration_metrics(y_true, y_proba)
        metrics.update(cal_metrics)

        results["per_model_metrics"][model_name] = metrics

        # Build summary row
        results["summary_table"].append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "ece": metrics["ece"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })

    # Compute significance tests
    significance = compute_pairwise_significance(
        y_true,
        predictions_dict,
        probabilities_dict,
        reference_model=reference_model if reference_model in predictions_dict else None,
        alpha=alpha,
    )

    results["significance_tests"] = significance

    # Add significance markers to summary
    for row in results["summary_table"]:
        model = row["model"]
        if model != reference_model and reference_model in predictions_dict:
            key = f"{reference_model}_vs_{model}"
            alt_key = f"{model}_vs_{reference_model}"

            mcnemar_result = (
                significance["mcnemar"].get(key) or
                significance["mcnemar"].get(alt_key)
            )
            if mcnemar_result:
                row["significant_vs_ref"] = mcnemar_result.significant
                row["p_value_mcnemar"] = mcnemar_result.p_value

            delong_result = (
                significance["delong"].get(key) or
                significance["delong"].get(alt_key)
            )
            if delong_result:
                row["p_value_delong"] = delong_result.p_value
                row["auc_diff"] = delong_result.auc_difference

    return results


def format_results_table(
    results: Dict[str, Any],
    include_ci: bool = False,
    latex: bool = False,
) -> str:
    """
    Format results as a publication-ready table.

    Args:
        results: Output from compute_metrics_with_significance
        include_ci: Include confidence intervals
        latex: Output LaTeX format

    Returns:
        Formatted table string
    """
    if latex:
        return _format_latex_table(results)
    else:
        return _format_markdown_table(results)


def _format_markdown_table(results: Dict[str, Any]) -> str:
    """Format results as markdown table."""
    rows = results["summary_table"]

    headers = ["Model", "Accuracy", "F1", "AUC", "ECE", "Precision", "Recall", "Sig."]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]

    for row in rows:
        sig = ""
        if row.get("significant_vs_ref"):
            sig = "*"
        if row.get("p_value_mcnemar", 1.0) < 0.01:
            sig = "**"
        if row.get("p_value_mcnemar", 1.0) < 0.001:
            sig = "***"

        line = f"| {row['model']} | {row['accuracy']:.3f} | {row['f1_score']:.3f} | "
        line += f"{row['roc_auc']:.3f} | {row['ece']:.3f} | {row['precision']:.3f} | "
        line += f"{row['recall']:.3f} | {sig} |"
        lines.append(line)

    lines.append("")
    lines.append("*p<0.05, **p<0.01, ***p<0.001 (McNemar's test vs reference)")

    return "\n".join(lines)


def _format_latex_table(results: Dict[str, Any]) -> str:
    """Format results as LaTeX table."""
    rows = results["summary_table"]

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Model comparison on test set}",
        "\\label{tab:results}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Model & Accuracy & F1 & AUC & ECE & Precision & Recall \\\\",
        "\\midrule",
    ]

    for row in rows:
        sig = ""
        if row.get("significant_vs_ref"):
            sig = "$^{*}$"

        line = f"{row['model']} & {row['accuracy']:.3f}{sig} & {row['f1_score']:.3f} & "
        line += f"{row['roc_auc']:.3f} & {row['ece']:.3f} & {row['precision']:.3f} & "
        line += f"{row['recall']:.3f} \\\\"
        lines.append(line)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)
