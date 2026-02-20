#!/usr/bin/env python3
"""
Calibration Visualization Script - Paper-Ready Plots

Generates figures for the thesis:
1. Reliability Diagrams (calibration curves)
2. ROC Curves with comparison between methods
3. Precision-Recall Curves
4. Confusion Matrix Heatmaps
5. Threshold Analysis Plots
6. Cross-Validation Box Plots
7. ECE Bar Charts

Usage:
    python scripts/generate_calibration_plots.py --results outputs/calibration_advanced --output outputs/figures
    python scripts/generate_calibration_plots.py --results outputs/calibration_advanced --format pdf
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def setup_matplotlib():
    """Configure matplotlib for paper-quality plots."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Use a clean style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Paper-ready settings
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.figsize': (6, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })

    return plt


def plot_reliability_diagram(
    reliability_data: Dict[str, Any],
    output_path: Path,
    method_name: str = "",
    format: str = "png"
):
    """
    Plot reliability diagram (calibration curve).

    Shows how well the predicted probabilities match actual frequencies.
    Perfect calibration = diagonal line.
    """
    plt = setup_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Reliability diagram
    bin_centers = reliability_data.get('bin_centers', [])
    bin_accuracies = reliability_data.get('bin_accuracies', [])
    bin_confidences = reliability_data.get('bin_confidences', [])
    bin_counts = reliability_data.get('bin_counts', [])

    if not bin_centers:
        logger.warning(f"No reliability data for {method_name}")
        plt.close()
        return

    # Normalize bin counts for bar width
    max_count = max(bin_counts) if bin_counts else 1
    bar_widths = [0.05] * len(bin_centers)

    # Plot bars for accuracy
    ax1.bar(bin_centers, bin_accuracies, width=0.05, alpha=0.7,
            color='steelblue', edgecolor='navy', label='Accuracy per bin')

    # Plot diagonal (perfect calibration)
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')

    # Plot confidence line
    ax1.plot(bin_centers, bin_confidences, 'ro-', markersize=4, alpha=0.8,
             label='Mean confidence')

    ax1.set_xlabel('Predicted Probability (Confidence)')
    ax1.set_ylabel('Observed Frequency (Accuracy)')
    ax1.set_title(f'Reliability Diagram - {method_name}')
    ax1.legend(loc='upper left')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Right plot: Histogram of predictions
    ax2.bar(bin_centers, bin_counts, width=0.05, alpha=0.7,
            color='forestgreen', edgecolor='darkgreen')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Number of Predictions')
    ax2.set_title('Prediction Distribution')
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = output_path / f"reliability_diagram_{method_name.lower().replace(' ', '_')}.{format}"
    plt.savefig(save_path, format=format)
    plt.close()

    logger.info(f"Saved reliability diagram: {save_path}")


def plot_combined_reliability_diagrams(
    all_reliability_data: Dict[str, Dict],
    output_path: Path,
    format: str = "png"
):
    """Plot reliability diagrams for all methods in one figure."""
    plt = setup_matplotlib()

    n_methods = len(all_reliability_data)
    if n_methods == 0:
        return

    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if n_methods == 1:
        axes = np.array([axes])

    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, (method_key, data) in enumerate(all_reliability_data.items()):
        ax = axes[idx]

        rel = data.get('reliability', {})
        bin_centers = rel.get('bin_centers', [])
        bin_accuracies = rel.get('bin_accuracies', [])

        if bin_centers:
            ax.bar(bin_centers, bin_accuracies, width=0.05, alpha=0.7,
                   color='steelblue', edgecolor='navy')
            ax.plot([0, 1], [0, 1], 'k--', lw=1)

        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f"{data.get('method_name', method_key)}\nECE={data.get('ece', 0):.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(len(all_reliability_data), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    save_path = output_path / f"reliability_diagrams_combined.{format}"
    plt.savefig(save_path, format=format)
    plt.close()

    logger.info(f"Saved combined reliability diagrams: {save_path}")


def plot_roc_curves(
    results: Dict[str, Dict],
    scores_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    format: str = "png"
):
    """
    Plot ROC curves for all methods.

    scores_data: {method_key: (labels, scores)}
    """
    plt = setup_matplotlib()
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(scores_data)))

    for (method_key, (labels, scores)), color in zip(scores_data.items(), colors):
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        method_name = results.get(method_key, {}).get('method_name', method_key)

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{method_name} (AUC = {roc_auc:.3f})')

    # Diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curves - Method Comparison')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    save_path = output_path / f"roc_curves_comparison.{format}"
    plt.savefig(save_path, format=format)
    plt.close()

    logger.info(f"Saved ROC curves: {save_path}")


def plot_precision_recall_curves(
    results: Dict[str, Dict],
    scores_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    format: str = "png"
):
    """Plot Precision-Recall curves for all methods."""
    plt = setup_matplotlib()
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(figsize=(8, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(scores_data)))

    for (method_key, (labels, scores)), color in zip(scores_data.items(), colors):
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)

        method_name = results.get(method_key, {}).get('method_name', method_key)

        ax.plot(recall, precision, color=color, lw=2,
                label=f'{method_name} (AP = {ap:.3f})')

    # Baseline (prevalence)
    if scores_data:
        first_labels = list(scores_data.values())[0][0]
        baseline = np.mean(first_labels)
        ax.axhline(y=baseline, color='gray', linestyle='--', lw=1,
                   label=f'Baseline (prevalence = {baseline:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves - Method Comparison')
    ax.legend(loc='lower left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = output_path / f"precision_recall_curves.{format}"
    plt.savefig(save_path, format=format)
    plt.close()

    logger.info(f"Saved PR curves: {save_path}")


def plot_cv_boxplots(
    results: Dict[str, Dict],
    output_path: Path,
    format: str = "png"
):
    """Plot box plots of cross-validation results."""
    plt = setup_matplotlib()

    # Collect CV metrics
    methods = []
    f1_scores = []
    auc_scores = []
    thresholds = []

    for method_key, result in results.items():
        methods.append(result.get('method_name', method_key))
        cv_metrics = result.get('cv_metrics', {})
        f1_scores.append(cv_metrics.get('f1', [0]))
        auc_scores.append(cv_metrics.get('auc', [0]))
        thresholds.append(result.get('cv_thresholds', [0]))

    if not methods:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # F1 Score boxplot
    bp1 = axes[0].boxplot(f1_scores, labels=methods, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('Cross-Validation F1 Scores')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')

    # AUC boxplot
    bp2 = axes[1].boxplot(auc_scores, labels=methods, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
    axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title('Cross-Validation AUC Scores')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Threshold boxplot
    bp3 = axes[2].boxplot(thresholds, labels=methods, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('lightyellow')
    axes[2].set_ylabel('Threshold')
    axes[2].set_title('Cross-Validation Thresholds')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    save_path = output_path / f"cv_boxplots.{format}"
    plt.savefig(save_path, format=format)
    plt.close()

    logger.info(f"Saved CV boxplots: {save_path}")


def plot_ece_comparison(
    results: Dict[str, Dict],
    output_path: Path,
    format: str = "png"
):
    """Plot ECE comparison bar chart."""
    plt = setup_matplotlib()

    methods = []
    ece_values = []
    mce_values = []

    for method_key, result in results.items():
        methods.append(result.get('method_name', method_key))
        calibration = result.get('calibration', {})
        ece_values.append(calibration.get('ece', 0) if calibration else 0)
        mce_values.append(calibration.get('mce', 0) if calibration else 0)

    if not methods:
        return

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, ece_values, width, label='ECE', color='steelblue', edgecolor='navy')
    bars2 = ax.bar(x + width/2, mce_values, width, label='MCE', color='coral', edgecolor='darkred')

    ax.set_xlabel('Method')
    ax.set_ylabel('Calibration Error')
    ax.set_title('ECE and MCE Comparison Between Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    save_path = output_path / f"ece_comparison.{format}"
    plt.savefig(save_path, format=format)
    plt.close()

    logger.info(f"Saved ECE comparison: {save_path}")


def plot_metrics_heatmap(
    results: Dict[str, Dict],
    output_path: Path,
    format: str = "png"
):
    """Plot heatmap of metrics across methods."""
    plt = setup_matplotlib()

    metrics_to_show = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc']
    methods = []
    data = []

    for method_key, result in results.items():
        methods.append(result.get('method_name', method_key))
        metrics = result.get('metrics', {})
        row = [metrics.get(m, 0) for m in metrics_to_show]
        data.append(row)

    if not methods:
        return

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(10, max(4, len(methods) * 0.6)))

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Valoare', rotation=-90, va="bottom")

    # Labels
    ax.set_xticks(np.arange(len(metrics_to_show)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([m.upper() for m in metrics_to_show])
    ax.set_yticklabels(methods)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(metrics_to_show)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center",
                          color="white" if data[i, j] < 0.5 else "black",
                          fontsize=9)

    ax.set_title('Performance Metrics per Method')

    plt.tight_layout()

    save_path = output_path / f"metrics_heatmap.{format}"
    plt.savefig(save_path, format=format)
    plt.close()

    logger.info(f"Saved metrics heatmap: {save_path}")


def plot_threshold_comparison(
    results: Dict[str, Dict],
    output_path: Path,
    format: str = "png"
):
    """Plot threshold comparison with confidence intervals."""
    plt = setup_matplotlib()

    methods = []
    thresholds = []
    ci_lower = []
    ci_upper = []
    cv_stds = []

    for method_key, result in results.items():
        methods.append(result.get('method_name', method_key))
        thresholds.append(result.get('threshold_optimal', 0.5))

        ci = result.get('confidence_interval_95', (0, 1))
        ci_lower.append(ci[0] if ci else 0)
        ci_upper.append(ci[1] if ci else 1)

        cv_stds.append(result.get('cv_threshold_std', 0))

    if not methods:
        return

    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Error bars from CI
    yerr_lower = np.array(thresholds) - np.array(ci_lower)
    yerr_upper = np.array(ci_upper) - np.array(thresholds)

    ax.errorbar(x, thresholds, yerr=[yerr_lower, yerr_upper],
                fmt='o', markersize=10, capsize=5, capthick=2,
                color='steelblue', ecolor='gray', elinewidth=2,
                label='Threshold optimal (95% CI)')

    # Add CV std as shaded region
    ax.fill_between(x,
                    np.array(thresholds) - np.array(cv_stds),
                    np.array(thresholds) + np.array(cv_stds),
                    alpha=0.2, color='steelblue',
                    label='±1 std (CV)')

    ax.set_xlabel('Method')
    ax.set_ylabel('Threshold')
    ax.set_title('Optimal Thresholds with Confidence Intervals')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    plt.tight_layout()

    save_path = output_path / f"threshold_comparison.{format}"
    plt.savefig(save_path, format=format)
    plt.close()

    logger.info(f"Saved threshold comparison: {save_path}")


def plot_statistical_significance_matrix(
    results: Dict[str, Dict],
    output_path: Path,
    format: str = "png"
):
    """Plot matrix of statistical significance between methods."""
    plt = setup_matplotlib()

    methods = list(results.keys())
    n = len(methods)

    if n < 2:
        return

    # Build significance matrix
    significance_matrix = np.zeros((n, n))
    p_value_matrix = np.ones((n, n))

    for i, method_key in enumerate(methods):
        result = results[method_key]
        comparisons = result.get('statistical_comparisons', {})

        for comp_name, comp_data in comparisons.items():
            # Parse comparison name to find target method
            for j, other_key in enumerate(methods):
                if other_key in comp_name and 'mcnemar' in comp_name.lower():
                    p_value = comp_data.get('p_value', 1.0)
                    p_value_matrix[i, j] = p_value
                    significance_matrix[i, j] = 1 if p_value < 0.05 else 0

    # Make symmetric
    for i in range(n):
        for j in range(i + 1, n):
            if significance_matrix[i, j] == 0 and significance_matrix[j, i] == 1:
                significance_matrix[i, j] = 1
            if p_value_matrix[i, j] == 1 and p_value_matrix[j, i] < 1:
                p_value_matrix[i, j] = p_value_matrix[j, i]

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(p_value_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('p-value', rotation=-90, va="bottom")

    # Labels
    method_names = [results[k].get('method_name', k) for k in methods]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.set_yticklabels(method_names)

    # Add annotations
    for i in range(n):
        for j in range(n):
            if i != j:
                p = p_value_matrix[i, j]
                sig = '*' if p < 0.05 else ''
                color = 'white' if p < 0.05 else 'black'
                ax.text(j, i, f'{p:.3f}{sig}',
                       ha='center', va='center', color=color, fontsize=8)
            else:
                ax.text(j, i, '-', ha='center', va='center', fontsize=8)

    ax.set_title('Statistical Significance (McNemar Test)\n* p < 0.05')

    plt.tight_layout()

    save_path = output_path / f"statistical_significance.{format}"
    plt.savefig(save_path, format=format)
    plt.close()

    logger.info(f"Saved significance matrix: {save_path}")


def generate_all_plots(
    results_dir: Path,
    output_dir: Path,
    format: str = "png"
):
    """Generate all plots from calibration results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load main results
    results_file = results_dir / "calibration_results_advanced.json"
    if not results_file.exists():
        # Try alternative name
        results_file = results_dir / "calibration_results.json"

    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return

    with open(results_file) as f:
        results = json.load(f)

    logger.info(f"Loaded results for {len(results)} methods")

    # Load reliability data if available
    reliability_file = results_dir / "reliability_diagram_data.json"
    reliability_data = {}

    if reliability_file.exists():
        with open(reliability_file) as f:
            reliability_data = json.load(f)

    # Generate plots
    logger.info("\nGenerating plots...")

    # 1. Reliability diagrams
    if reliability_data:
        plot_combined_reliability_diagrams(reliability_data, output_dir, format)

        for method_key, data in reliability_data.items():
            plot_reliability_diagram(
                data.get('reliability', {}),
                output_dir,
                data.get('method_name', method_key),
                format
            )

    # 2. CV boxplots
    plot_cv_boxplots(results, output_dir, format)

    # 3. ECE comparison
    plot_ece_comparison(results, output_dir, format)

    # 4. Metrics heatmap
    plot_metrics_heatmap(results, output_dir, format)

    # 5. Threshold comparison
    plot_threshold_comparison(results, output_dir, format)

    # 6. Statistical significance
    plot_statistical_significance_matrix(results, output_dir, format)

    logger.info(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate calibration plots")
    parser.add_argument(
        "--results", type=Path, required=True,
        help="Path to calibration results directory"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/figures"),
        help="Output directory for plots"
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "svg"], default="png",
        help="Output format for plots"
    )

    args = parser.parse_args()

    generate_all_plots(args.results, args.output, args.format)


if __name__ == "__main__":
    main()
