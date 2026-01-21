#!/usr/bin/env python3
"""
Statistical Significance Testing for ImageTrust Thesis.

Provides rigorous statistical comparison between methods:
1. McNemar's test - paired comparison of binary predictions
2. DeLong test - comparison of AUC-ROC curves
3. Paired t-test - comparison of cross-validation scores
4. Bootstrap confidence intervals
5. Multiple comparison correction (Bonferroni, Holm)

Usage:
    python scripts/statistical_tests.py --results ./outputs/baselines/{timestamp}
    python scripts/statistical_tests.py --demo  # Run with synthetic data

Output:
    outputs/paper/statistics/
    ├── mcnemar_results.json
    ├── delong_results.json
    ├── bootstrap_ci.json
    ├── significance_summary.tex
    └── significance_report.md

Author: ImageTrust Team
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imagetrust.utils.helpers import ensure_dir


# =============================================================================
# McNemar's Test
# =============================================================================

def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    correction: bool = True,
) -> Dict[str, float]:
    """
    McNemar's test for paired binary predictions.

    Tests whether two classifiers have significantly different error rates.

    Args:
        y_true: Ground truth labels (0 or 1)
        pred_a: Predictions from classifier A
        pred_b: Predictions from classifier B
        correction: Apply continuity correction

    Returns:
        Dict with statistic, p_value, and contingency table
    """
    # Build contingency table
    # n00: both wrong, n01: A wrong B correct, n10: A correct B wrong, n11: both correct
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    n00 = np.sum(~correct_a & ~correct_b)  # Both wrong
    n01 = np.sum(~correct_a & correct_b)   # A wrong, B correct
    n10 = np.sum(correct_a & ~correct_b)   # A correct, B wrong
    n11 = np.sum(correct_a & correct_b)    # Both correct

    # McNemar's statistic (with or without continuity correction)
    if correction:
        # Edwards' continuity correction
        statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0
    else:
        statistic = (n01 - n10) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0

    # Chi-squared distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "test": "McNemar",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "contingency_table": {
            "both_wrong": int(n00),
            "a_wrong_b_correct": int(n01),
            "a_correct_b_wrong": int(n10),
            "both_correct": int(n11),
        },
        "n_discordant": int(n01 + n10),
    }


def exact_mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Dict[str, float]:
    """
    Exact McNemar's test using binomial distribution.

    More accurate for small sample sizes (n_discordant < 25).
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    n01 = np.sum(~correct_a & correct_b)
    n10 = np.sum(correct_a & ~correct_b)

    n_discordant = n01 + n10

    if n_discordant == 0:
        return {
            "test": "Exact McNemar",
            "statistic": 0.0,
            "p_value": 1.0,
            "n_discordant": 0,
        }

    # Two-sided binomial test
    # Under null hypothesis, P(A wrong, B correct) = P(A correct, B wrong) = 0.5
    p_value = stats.binom_test(n01, n_discordant, 0.5, alternative='two-sided')

    return {
        "test": "Exact McNemar (Binomial)",
        "statistic": float(min(n01, n10)),
        "p_value": float(p_value),
        "n_discordant": int(n_discordant),
    }


# =============================================================================
# DeLong Test for AUC Comparison
# =============================================================================

def delong_test(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
) -> Dict[str, float]:
    """
    DeLong test for comparing two correlated AUC-ROC curves.

    Based on: DeLong et al. (1988) "Comparing the Areas under Two or
    More Correlated Receiver Operating Characteristic Curves"

    Args:
        y_true: Ground truth labels (0 or 1)
        prob_a: Predicted probabilities from model A
        prob_b: Predicted probabilities from model B

    Returns:
        Dict with z-statistic, p_value, and AUCs
    """
    from sklearn.metrics import roc_auc_score

    # Compute AUCs
    auc_a = roc_auc_score(y_true, prob_a)
    auc_b = roc_auc_score(y_true, prob_b)

    # Separate positive and negative samples
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    pos_a = prob_a[pos_mask]
    neg_a = prob_a[neg_mask]
    pos_b = prob_b[pos_mask]
    neg_b = prob_b[neg_mask]

    m = len(pos_a)  # Number of positives
    n = len(neg_a)  # Number of negatives

    # Compute placement values (structural components)
    def compute_placements(pos_scores, neg_scores):
        """Compute placement values for DeLong variance."""
        placements_pos = np.zeros(len(pos_scores))
        placements_neg = np.zeros(len(neg_scores))

        for i, p in enumerate(pos_scores):
            placements_pos[i] = np.mean(p > neg_scores) + 0.5 * np.mean(p == neg_scores)

        for j, n_val in enumerate(neg_scores):
            placements_neg[j] = np.mean(pos_scores > n_val) + 0.5 * np.mean(pos_scores == n_val)

        return placements_pos, placements_neg

    v10_a, v01_a = compute_placements(pos_a, neg_a)
    v10_b, v01_b = compute_placements(pos_b, neg_b)

    # Variance and covariance estimation
    var_a = (np.var(v10_a, ddof=1) / m + np.var(v01_a, ddof=1) / n)
    var_b = (np.var(v10_b, ddof=1) / m + np.var(v01_b, ddof=1) / n)

    cov_ab = (np.cov(v10_a, v10_b, ddof=1)[0, 1] / m +
              np.cov(v01_a, v01_b, ddof=1)[0, 1] / n)

    # Variance of difference
    var_diff = var_a + var_b - 2 * cov_ab

    if var_diff <= 0:
        # Degenerate case
        z_stat = 0.0
        p_value = 1.0
    else:
        # Z-statistic
        z_stat = (auc_a - auc_b) / np.sqrt(var_diff)
        # Two-sided p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "test": "DeLong",
        "statistic": float(z_stat),
        "p_value": float(p_value),
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "auc_difference": float(auc_a - auc_b),
        "std_error": float(np.sqrt(var_diff)) if var_diff > 0 else 0.0,
    }


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        metric_fn: Function(y_true, y_prob) -> float
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed

    Returns:
        Dict with point estimate, lower/upper bounds, std error
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Point estimate
    point_estimate = metric_fn(y_true, y_prob)

    # Bootstrap samples
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        score = metric_fn(y_true[indices], y_prob[indices])
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)

    # Confidence interval (percentile method)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

    return {
        "point_estimate": float(point_estimate),
        "lower": float(lower),
        "upper": float(upper),
        "std_error": float(np.std(bootstrap_scores, ddof=1)),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_difference_ci(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for difference between two methods.

    Args:
        y_true: Ground truth labels
        prob_a: Probabilities from method A
        prob_b: Probabilities from method B
        metric_fn: Metric function
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Dict with difference estimate and CI
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Point estimates
    score_a = metric_fn(y_true, prob_a)
    score_b = metric_fn(y_true, prob_b)
    point_diff = score_a - score_b

    # Bootstrap differences
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        diff = (metric_fn(y_true[indices], prob_a[indices]) -
                metric_fn(y_true[indices], prob_b[indices]))
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    # Check if CI contains zero (significant if it doesn't)
    significant = (lower > 0) or (upper < 0)

    return {
        "difference": float(point_diff),
        "lower": float(lower),
        "upper": float(upper),
        "std_error": float(np.std(bootstrap_diffs, ddof=1)),
        "significant": significant,
        "confidence": confidence,
        "method_a_score": float(score_a),
        "method_b_score": float(score_b),
    }


# =============================================================================
# Multiple Comparison Correction
# =============================================================================

def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values
        alpha: Significance level

    Returns:
        Dict with corrected alpha, adjusted p-values, and significance
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    adjusted = [min(p * n_tests, 1.0) for p in p_values]
    significant = [p < corrected_alpha for p in p_values]

    return {
        "method": "Bonferroni",
        "original_alpha": alpha,
        "corrected_alpha": corrected_alpha,
        "n_tests": n_tests,
        "adjusted_p_values": adjusted,
        "significant": significant,
        "n_significant": sum(significant),
    }


def holm_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Holm-Bonferroni step-down correction for multiple comparisons.

    Less conservative than Bonferroni while controlling FWER.

    Args:
        p_values: List of p-values
        alpha: Significance level

    Returns:
        Dict with adjusted p-values and significance
    """
    n_tests = len(p_values)

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = [p_values[i] for i in sorted_indices]

    # Holm adjustment
    adjusted = []
    for i, p in enumerate(sorted_p):
        adj_p = min(p * (n_tests - i), 1.0)
        # Ensure monotonicity
        if adjusted:
            adj_p = max(adj_p, adjusted[-1])
        adjusted.append(adj_p)

    # Restore original order
    adjusted_original = [0.0] * n_tests
    for i, idx in enumerate(sorted_indices):
        adjusted_original[idx] = adjusted[i]

    significant = [p < alpha for p in adjusted_original]

    return {
        "method": "Holm-Bonferroni",
        "original_alpha": alpha,
        "n_tests": n_tests,
        "adjusted_p_values": adjusted_original,
        "significant": significant,
        "n_significant": sum(significant),
    }


# =============================================================================
# Comprehensive Statistical Analysis
# =============================================================================

def run_all_tests(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    reference_method: str = "ImageTrust (Ours)",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Run comprehensive statistical tests.

    Args:
        predictions: Dict mapping method name to (probabilities, labels)
        reference_method: Method to compare against
        alpha: Significance level

    Returns:
        Complete statistical analysis results
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    results = {
        "reference_method": reference_method,
        "alpha": alpha,
        "mcnemar_tests": {},
        "delong_tests": {},
        "bootstrap_ci": {},
        "pairwise_comparisons": [],
    }

    if reference_method not in predictions:
        print(f"Warning: Reference method '{reference_method}' not found")
        return results

    ref_prob, ref_labels = predictions[reference_method]
    ref_pred = (ref_prob > 0.5).astype(int)

    p_values_mcnemar = []
    p_values_delong = []
    comparison_names = []

    for method_name, (prob, labels) in predictions.items():
        if method_name == reference_method:
            continue

        pred = (prob > 0.5).astype(int)

        # McNemar's test
        mcnemar_result = mcnemar_test(labels, ref_pred, pred)
        results["mcnemar_tests"][method_name] = mcnemar_result
        p_values_mcnemar.append(mcnemar_result["p_value"])

        # DeLong test
        delong_result = delong_test(labels, ref_prob, prob)
        results["delong_tests"][method_name] = delong_result
        p_values_delong.append(delong_result["p_value"])

        comparison_names.append(method_name)

        # Bootstrap CI for AUC difference
        def auc_fn(y, p):
            return roc_auc_score(y, p)

        bootstrap_result = bootstrap_difference_ci(
            labels, ref_prob, prob, auc_fn,
            n_bootstrap=1000, confidence=0.95,
        )

        results["pairwise_comparisons"].append({
            "comparison": f"{reference_method} vs {method_name}",
            "mcnemar": mcnemar_result,
            "delong": delong_result,
            "bootstrap_auc_diff": bootstrap_result,
        })

    # Multiple comparison corrections
    if p_values_mcnemar:
        results["correction_mcnemar"] = {
            "bonferroni": bonferroni_correction(p_values_mcnemar, alpha),
            "holm": holm_correction(p_values_mcnemar, alpha),
        }

    if p_values_delong:
        results["correction_delong"] = {
            "bonferroni": bonferroni_correction(p_values_delong, alpha),
            "holm": holm_correction(p_values_delong, alpha),
        }

    # Bootstrap CIs for each method's AUC
    for method_name, (prob, labels) in predictions.items():
        ci = bootstrap_ci(labels, prob, roc_auc_score, n_bootstrap=1000)
        results["bootstrap_ci"][method_name] = ci

    return results


# =============================================================================
# Report Generation
# =============================================================================

def generate_significance_report(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate markdown report of statistical significance tests."""
    ensure_dir(output_dir)

    lines = [
        "# Statistical Significance Report",
        "",
        f"**Reference Method:** {results.get('reference_method', 'N/A')}",
        f"**Significance Level:** α = {results.get('alpha', 0.05)}",
        "",
        "---",
        "",
        "## Summary",
        "",
    ]

    # Summary table
    lines.append("| Comparison | McNemar p-value | DeLong p-value | AUC Diff | Significant |")
    lines.append("|------------|-----------------|----------------|----------|-------------|")

    for comp in results.get("pairwise_comparisons", []):
        mcnemar_p = comp["mcnemar"]["p_value"]
        delong_p = comp["delong"]["p_value"]
        auc_diff = comp["bootstrap_auc_diff"]["difference"]
        sig = "Yes" if (mcnemar_p < 0.05 or delong_p < 0.05) else "No"

        lines.append(
            f"| {comp['comparison']} | {mcnemar_p:.4f} | {delong_p:.4f} | "
            f"{auc_diff:+.4f} | {sig} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## McNemar's Test Results",
        "",
        "Tests whether classifiers have significantly different error rates.",
        "",
    ])

    for method, test_result in results.get("mcnemar_tests", {}).items():
        lines.append(f"### vs {method}")
        lines.append(f"- **Statistic:** {test_result['statistic']:.3f}")
        lines.append(f"- **p-value:** {test_result['p_value']:.4f}")
        lines.append(f"- **Discordant pairs:** {test_result['n_discordant']}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## DeLong Test Results",
        "",
        "Compares AUC-ROC curves between methods.",
        "",
    ])

    for method, test_result in results.get("delong_tests", {}).items():
        lines.append(f"### vs {method}")
        lines.append(f"- **Z-statistic:** {test_result['statistic']:.3f}")
        lines.append(f"- **p-value:** {test_result['p_value']:.4f}")
        lines.append(f"- **AUC difference:** {test_result['auc_difference']:.4f}")
        lines.append(f"- **Standard error:** {test_result['std_error']:.4f}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Bootstrap Confidence Intervals (95%)",
        "",
    ])

    for method, ci in results.get("bootstrap_ci", {}).items():
        lines.append(f"### {method}")
        lines.append(f"- **AUC:** {ci['point_estimate']:.4f}")
        lines.append(f"- **95% CI:** [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        lines.append(f"- **Std Error:** {ci['std_error']:.4f}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Multiple Comparison Correction",
        "",
    ])

    for test_type in ["mcnemar", "delong"]:
        corrections = results.get(f"correction_{test_type}", {})
        if corrections:
            lines.append(f"### {test_type.title()} Test Corrections")
            lines.append("")

            for method, corr in corrections.items():
                lines.append(f"**{method}:**")
                lines.append(f"- Corrected α: {corr.get('corrected_alpha', 'N/A')}")
                lines.append(f"- Significant tests: {corr.get('n_significant', 0)}/{corr.get('n_tests', 0)}")
                lines.append("")

    # Write report
    report_path = output_dir / "significance_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated: {report_path}")


def generate_significance_latex(results: Dict[str, Any], output_path: Path) -> None:
    """Generate LaTeX table for significance tests."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical significance tests comparing ImageTrust against baseline methods. "
        r"McNemar's test compares prediction accuracy; DeLong test compares AUC. $\alpha = 0.05$.}",
        r"\label{tab:significance}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Comparison & McNemar $\chi^2$ & p-value & DeLong $z$ & p-value \\",
        r"\midrule",
    ]

    for comp in results.get("pairwise_comparisons", []):
        comparison = comp["comparison"].replace("_", r"\_")
        mc_stat = comp["mcnemar"]["statistic"]
        mc_p = comp["mcnemar"]["p_value"]
        dl_stat = comp["delong"]["statistic"]
        dl_p = comp["delong"]["p_value"]

        # Bold significant p-values
        mc_p_str = f"\\textbf{{{mc_p:.4f}}}" if mc_p < 0.05 else f"{mc_p:.4f}"
        dl_p_str = f"\\textbf{{{dl_p:.4f}}}" if dl_p < 0.05 else f"{dl_p:.4f}"

        lines.append(
            f"{comparison} & {mc_stat:.2f} & {mc_p_str} & {dl_stat:.2f} & {dl_p_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated: {output_path}")


# =============================================================================
# Demo with Synthetic Data
# =============================================================================

def run_demo(output_dir: Path) -> None:
    """Run statistical tests with synthetic data."""
    ensure_dir(output_dir)
    np.random.seed(42)

    print("Running statistical tests with synthetic data...\n")

    n_samples = 1000
    labels = np.random.randint(0, 2, n_samples)

    # Simulate predictions from different methods
    predictions = {}

    # Classical (weakest)
    noise = np.random.normal(0, 0.3, n_samples)
    prob = labels * 0.6 + (1 - labels) * 0.4 + noise
    predictions["Classical (LogReg)"] = (np.clip(prob, 0, 1), labels)

    # CNN (medium)
    noise = np.random.normal(0, 0.25, n_samples)
    prob = labels * 0.7 + (1 - labels) * 0.3 + noise
    predictions["CNN (ResNet-50)"] = (np.clip(prob, 0, 1), labels)

    # ViT (good)
    noise = np.random.normal(0, 0.2, n_samples)
    prob = labels * 0.75 + (1 - labels) * 0.25 + noise
    predictions["ViT-B/16"] = (np.clip(prob, 0, 1), labels)

    # ImageTrust (best)
    noise = np.random.normal(0, 0.15, n_samples)
    prob = labels * 0.85 + (1 - labels) * 0.15 + noise
    predictions["ImageTrust (Ours)"] = (np.clip(prob, 0, 1), labels)

    # Run all tests
    results = run_all_tests(predictions, reference_method="ImageTrust (Ours)")

    # Save results
    results_path = output_dir / "statistical_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {results_path}")

    # Generate reports
    generate_significance_report(results, output_dir)
    generate_significance_latex(results, output_dir / "table_significance.tex")

    print(f"\n✅ Statistical analysis complete. Results in: {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run statistical significance tests for ImageTrust thesis",
    )

    parser.add_argument(
        "--results", "-r",
        type=str,
        default=None,
        help="Path to results directory with predictions",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/paper/statistics",
        help="Output directory",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic data",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.demo:
        run_demo(output_dir)
    elif args.results:
        results_dir = Path(args.results)
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            sys.exit(1)

        # Load predictions from results
        main_results_path = results_dir / "main_results.json"
        if not main_results_path.exists():
            print(f"Error: main_results.json not found in {results_dir}")
            sys.exit(1)

        with open(main_results_path) as f:
            main_results = json.load(f)

        # Extract predictions
        predictions = {}
        for method, metrics in main_results.items():
            if "_predictions" in metrics:
                pred_data = metrics["_predictions"]
                probs = np.array(pred_data["probabilities"])
                labels = np.array(pred_data["labels"])
                predictions[method] = (probs, labels)

        if not predictions:
            print("Error: No prediction data found in results")
            print("Tip: Re-run evaluation with return_predictions=True")
            sys.exit(1)

        # Run tests
        ensure_dir(output_dir)
        results = run_all_tests(predictions, alpha=args.alpha)

        # Save and generate reports
        results_path = output_dir / "statistical_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        generate_significance_report(results, output_dir)
        generate_significance_latex(results, output_dir / "table_significance.tex")

        print(f"\n✅ Statistical analysis complete. Results in: {output_dir}")
    else:
        print("Error: Either --results or --demo must be specified")
        sys.exit(1)


if __name__ == "__main__":
    main()
