"""
Statistical Tests for Publication-Ready Evaluation.

Implements statistical significance tests required for academic publications:
- McNemar's test: Compare two classifiers on the same test set
- DeLong's test: Compare two ROC-AUC values
- Bootstrap confidence intervals: For any metric
- Permutation tests: Model-agnostic significance testing

Reference:
    DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
    Comparing the areas under two or more correlated receiver operating
    characteristic curves: a nonparametric approach.
    Biometrics, 837-845.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class McNemarResult:
    """Result from McNemar's test."""

    chi2_statistic: float
    p_value: float
    significant: bool  # p < 0.05
    n_disagreements: int  # Total disagreements
    n_01: int  # A wrong, B correct
    n_10: int  # A correct, B wrong
    odds_ratio: float  # Odds of A being correct when they disagree
    interpretation: str


@dataclass
class DeLongResult:
    """Result from DeLong's test for comparing AUCs."""

    auc_a: float
    auc_b: float
    auc_difference: float
    z_statistic: float
    p_value: float
    significant: bool
    ci_lower: float  # 95% CI lower bound for difference
    ci_upper: float  # 95% CI upper bound for difference
    interpretation: str


@dataclass
class BootstrapCIResult:
    """Result from bootstrap confidence interval estimation."""

    estimate: float  # Point estimate
    ci_lower: float  # Lower bound
    ci_upper: float  # Upper bound
    std_error: float  # Bootstrap standard error
    n_bootstrap: int
    confidence_level: float


@dataclass
class PermutationTestResult:
    """Result from permutation test."""

    observed_difference: float
    p_value: float
    significant: bool
    n_permutations: int
    null_distribution_mean: float
    null_distribution_std: float
    interpretation: str


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    alpha: float = 0.05,
    continuity_correction: bool = True,
) -> McNemarResult:
    """
    McNemar's test for comparing two classifiers.

    Tests whether two classifiers have the same error rate.
    Null hypothesis: Both classifiers make errors at the same rate.

    Args:
        y_true: Ground truth labels.
        pred_a: Predictions from classifier A.
        pred_b: Predictions from classifier B.
        alpha: Significance level (default 0.05).
        continuity_correction: Apply Edwards' continuity correction.

    Returns:
        McNemarResult with test statistics and interpretation.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    # Compute contingency table
    # correct_a[i] = 1 if pred_a[i] == y_true[i]
    correct_a = (pred_a == y_true).astype(int)
    correct_b = (pred_b == y_true).astype(int)

    # Count disagreements
    # n_01: A wrong, B correct
    n_01 = np.sum((correct_a == 0) & (correct_b == 1))
    # n_10: A correct, B wrong
    n_10 = np.sum((correct_a == 1) & (correct_b == 0))

    n_disagreements = n_01 + n_10

    # McNemar's chi-squared statistic
    if n_disagreements == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        if continuity_correction:
            # Edwards' continuity correction
            chi2 = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
        else:
            chi2 = (n_01 - n_10) ** 2 / (n_01 + n_10)

        # P-value from chi-squared distribution with 1 df
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    # Odds ratio
    odds_ratio = (n_10 + 0.5) / (n_01 + 0.5)

    # Interpretation
    significant = p_value < alpha
    if significant:
        if n_10 > n_01:
            interpretation = (
                f"Classifier A significantly outperforms B "
                f"(p={p_value:.4f}, A correct when B wrong: {n_10}, "
                f"B correct when A wrong: {n_01})"
            )
        else:
            interpretation = (
                f"Classifier B significantly outperforms A "
                f"(p={p_value:.4f}, B correct when A wrong: {n_01}, "
                f"A correct when B wrong: {n_10})"
            )
    else:
        interpretation = (
            f"No significant difference between classifiers "
            f"(p={p_value:.4f})"
        )

    return McNemarResult(
        chi2_statistic=chi2,
        p_value=p_value,
        significant=significant,
        n_disagreements=n_disagreements,
        n_01=n_01,
        n_10=n_10,
        odds_ratio=odds_ratio,
        interpretation=interpretation,
    )


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks for DeLong test."""
    n = len(x)
    sorted_indices = np.argsort(x)
    ranks = np.empty(n)

    i = 0
    while i < n:
        j = i
        while j < n and x[sorted_indices[j]] == x[sorted_indices[i]]:
            j += 1
        # Average rank for ties
        avg_rank = (i + j - 1) / 2 + 1
        for k in range(i, j):
            ranks[sorted_indices[k]] = avg_rank
        i = j

    return ranks


def _fast_delong(
    predictions_sorted_transposed: np.ndarray,
    label_1_count: int,
) -> Tuple[float, np.ndarray]:
    """
    Fast DeLong AUC computation.

    Based on: https://github.com/yandexdataschool/roc_comparison
    """
    n = predictions_sorted_transposed.shape[1]
    m = label_1_count
    n_negatives = n - m

    # Compute ranks
    aucs = []
    delongcov = []

    for i in range(predictions_sorted_transposed.shape[0]):
        ordered_sample = predictions_sorted_transposed[i]
        ranks = _compute_midrank(ordered_sample)

        positive_ranks = ranks[:m]
        auc = (np.sum(positive_ranks) - m * (m + 1) / 2) / (m * n_negatives)
        aucs.append(auc)

        # Placement values for covariance
        positive_placement = positive_ranks - np.arange(1, m + 1)
        negative_placement = (
            ranks[m:] - np.arange(m + 1, n + 1)
        )

        delongcov.append((positive_placement, negative_placement))

    return np.array(aucs), delongcov


def delong_test(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    alpha: float = 0.05,
) -> DeLongResult:
    """
    DeLong's test for comparing two ROC-AUC values.

    Tests whether two classifiers have different AUC values.
    Uses DeLong's variance estimation for correlated ROC curves.

    Args:
        y_true: Ground truth labels (0 or 1).
        prob_a: Predicted probabilities from classifier A.
        prob_b: Predicted probabilities from classifier B.
        alpha: Significance level (default 0.05).

    Returns:
        DeLongResult with test statistics and confidence intervals.
    """
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)

    # Sort by label
    order = np.argsort(y_true)[::-1]  # Positives first
    y_sorted = y_true[order]
    prob_a_sorted = prob_a[order]
    prob_b_sorted = prob_b[order]

    m = int(np.sum(y_true))  # Number of positives

    # Stack predictions
    predictions = np.vstack([prob_a_sorted, prob_b_sorted])

    # Compute AUCs and covariance components
    aucs, delongcov = _fast_delong(predictions, m)
    auc_a, auc_b = aucs[0], aucs[1]

    # Compute covariance matrix
    n = len(y_true)
    n_negatives = n - m

    # Placement values
    pos_a, neg_a = delongcov[0]
    pos_b, neg_b = delongcov[1]

    # Variance components
    var_a = (np.var(pos_a) / m + np.var(neg_a) / n_negatives)
    var_b = (np.var(pos_b) / m + np.var(neg_b) / n_negatives)

    # Covariance
    cov_ab = (
        np.cov(pos_a, pos_b)[0, 1] / m +
        np.cov(neg_a, neg_b)[0, 1] / n_negatives
    )

    # Variance of difference
    var_diff = var_a + var_b - 2 * cov_ab

    # Z statistic
    auc_diff = auc_a - auc_b

    if var_diff > 0:
        z = auc_diff / np.sqrt(var_diff)
    else:
        z = 0.0

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # 95% confidence interval for difference
    z_crit = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(var_diff) if var_diff > 0 else 0.0
    ci_lower = auc_diff - z_crit * se
    ci_upper = auc_diff + z_crit * se

    # Interpretation
    significant = p_value < alpha
    if significant:
        if auc_diff > 0:
            interpretation = (
                f"Classifier A has significantly higher AUC "
                f"({auc_a:.4f} vs {auc_b:.4f}, p={p_value:.4f})"
            )
        else:
            interpretation = (
                f"Classifier B has significantly higher AUC "
                f"({auc_b:.4f} vs {auc_a:.4f}, p={p_value:.4f})"
            )
    else:
        interpretation = (
            f"No significant difference in AUC "
            f"({auc_a:.4f} vs {auc_b:.4f}, p={p_value:.4f})"
        )

    return DeLongResult(
        auc_a=auc_a,
        auc_b=auc_b,
        auc_difference=auc_diff,
        z_statistic=z,
        p_value=p_value,
        significant=significant,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        interpretation=interpretation,
    )


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> BootstrapCIResult:
    """
    Bootstrap confidence intervals for any metric.

    Uses the percentile bootstrap method to estimate confidence
    intervals for arbitrary metrics.

    Args:
        metric_fn: Function that computes metric from (y_true, y_pred).
        y_true: Ground truth labels.
        y_pred: Predictions (labels or probabilities).
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level (default 0.95).
        random_state: Random seed for reproducibility.

    Returns:
        BootstrapCIResult with estimate and confidence interval.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    rng = np.random.RandomState(random_state)

    # Point estimate
    point_estimate = metric_fn(y_true, y_pred)

    # Bootstrap samples
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        bootstrap_y_true = y_true[indices]
        bootstrap_y_pred = y_pred[indices]

        try:
            estimate = metric_fn(bootstrap_y_true, bootstrap_y_pred)
            bootstrap_estimates.append(estimate)
        except Exception:
            # Skip failed bootstrap samples
            continue

    bootstrap_estimates = np.array(bootstrap_estimates)

    # Percentile confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_estimates, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)

    # Standard error
    std_error = np.std(bootstrap_estimates)

    return BootstrapCIResult(
        estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def permutation_test(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> PermutationTestResult:
    """
    Permutation test for comparing two sets of predictions.

    Tests the null hypothesis that the two prediction sets come from
    the same distribution (i.e., performance difference is due to chance).

    Args:
        metric_fn: Function that computes metric from (y_true, y_pred).
        y_true: Ground truth labels.
        pred_a: Predictions from model A.
        pred_b: Predictions from model B.
        n_permutations: Number of permutations.
        alpha: Significance level.
        random_state: Random seed.

    Returns:
        PermutationTestResult with p-value and interpretation.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    rng = np.random.RandomState(random_state)

    # Observed difference
    metric_a = metric_fn(y_true, pred_a)
    metric_b = metric_fn(y_true, pred_b)
    observed_diff = metric_a - metric_b

    # Permutation distribution
    # Under null: A and B are exchangeable
    all_preds = np.vstack([pred_a, pred_b])
    n = len(y_true)

    null_diffs = []
    for _ in range(n_permutations):
        # Randomly swap predictions between A and B
        swap_mask = rng.randint(0, 2, size=n)

        perm_a = np.where(swap_mask, pred_b, pred_a)
        perm_b = np.where(swap_mask, pred_a, pred_b)

        try:
            diff = metric_fn(y_true, perm_a) - metric_fn(y_true, perm_b)
            null_diffs.append(diff)
        except Exception:
            continue

    null_diffs = np.array(null_diffs)

    # P-value: proportion of permutations with difference >= observed
    # Two-tailed test
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))

    # Interpretation
    significant = p_value < alpha
    if significant:
        if observed_diff > 0:
            interpretation = (
                f"Model A significantly outperforms Model B "
                f"(diff={observed_diff:.4f}, p={p_value:.4f})"
            )
        else:
            interpretation = (
                f"Model B significantly outperforms Model A "
                f"(diff={-observed_diff:.4f}, p={p_value:.4f})"
            )
    else:
        interpretation = (
            f"No significant difference between models "
            f"(p={p_value:.4f})"
        )

    return PermutationTestResult(
        observed_difference=observed_diff,
        p_value=p_value,
        significant=significant,
        n_permutations=n_permutations,
        null_distribution_mean=np.mean(null_diffs),
        null_distribution_std=np.std(null_diffs),
        interpretation=interpretation,
    )


def compute_pairwise_significance(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    probabilities_dict: Optional[Dict[str, np.ndarray]] = None,
    reference_model: Optional[str] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute pairwise statistical significance between all models.

    Args:
        y_true: Ground truth labels.
        predictions_dict: Dictionary of {model_name: predictions}.
        probabilities_dict: Optional dictionary of {model_name: probabilities}.
        reference_model: If specified, only compare to this model.
        alpha: Significance level.

    Returns:
        Dictionary with McNemar results and optional DeLong results.
    """
    results = {
        "mcnemar": {},
        "delong": {},
        "summary": {},
    }

    model_names = list(predictions_dict.keys())

    if reference_model and reference_model in model_names:
        # Compare all models to reference
        pairs = [(reference_model, m) for m in model_names if m != reference_model]
    else:
        # All pairwise comparisons
        pairs = []
        for i, m1 in enumerate(model_names):
            for m2 in model_names[i + 1:]:
                pairs.append((m1, m2))

    # McNemar tests
    for m1, m2 in pairs:
        key = f"{m1}_vs_{m2}"
        results["mcnemar"][key] = mcnemar_test(
            y_true,
            predictions_dict[m1],
            predictions_dict[m2],
            alpha=alpha,
        )

    # DeLong tests (if probabilities available)
    if probabilities_dict:
        for m1, m2 in pairs:
            if m1 in probabilities_dict and m2 in probabilities_dict:
                key = f"{m1}_vs_{m2}"
                results["delong"][key] = delong_test(
                    y_true,
                    probabilities_dict[m1],
                    probabilities_dict[m2],
                    alpha=alpha,
                )

    # Summary
    significant_mcnemar = sum(
        1 for r in results["mcnemar"].values() if r.significant
    )
    total_comparisons = len(pairs)

    results["summary"] = {
        "total_comparisons": total_comparisons,
        "significant_mcnemar": significant_mcnemar,
        "significant_delong": sum(
            1 for r in results["delong"].values() if r.significant
        ),
        "bonferroni_alpha": alpha / total_comparisons if total_comparisons > 0 else alpha,
    }

    return results


def format_significance_table(
    results: Dict[str, Any],
    include_delong: bool = True,
) -> str:
    """
    Format significance results as a markdown table.

    Args:
        results: Output from compute_pairwise_significance.
        include_delong: Include DeLong test results.

    Returns:
        Markdown-formatted table string.
    """
    lines = [
        "| Comparison | McNemar χ² | McNemar p | Significant |",
        "|------------|------------|-----------|-------------|",
    ]

    for key, result in results["mcnemar"].items():
        sig_mark = "✓" if result.significant else "✗"
        lines.append(
            f"| {key} | {result.chi2_statistic:.2f} | "
            f"{result.p_value:.4f} | {sig_mark} |"
        )

    if include_delong and results["delong"]:
        lines.extend([
            "",
            "| Comparison | AUC Diff | DeLong z | DeLong p | 95% CI | Significant |",
            "|------------|----------|----------|----------|--------|-------------|",
        ])

        for key, result in results["delong"].items():
            sig_mark = "✓" if result.significant else "✗"
            ci = f"[{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
            lines.append(
                f"| {key} | {result.auc_difference:.4f} | "
                f"{result.z_statistic:.2f} | {result.p_value:.4f} | "
                f"{ci} | {sig_mark} |"
            )

    # Summary
    lines.extend([
        "",
        f"**Summary**: {results['summary']['significant_mcnemar']}/"
        f"{results['summary']['total_comparisons']} McNemar tests significant",
    ])

    if results["delong"]:
        lines.append(
            f"{results['summary']['significant_delong']}/"
            f"{results['summary']['total_comparisons']} DeLong tests significant"
        )

    return "\n".join(lines)
