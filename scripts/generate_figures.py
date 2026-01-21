#!/usr/bin/env python3
"""
Generate Paper Figures for ImageTrust Thesis.

Produces publication-ready figures:
1. Reliability diagram (calibration curves)
2. ROC curves (per baseline)
3. Cross-generator heatmap
4. Degradation performance curves
5. Confusion matrices
6. Performance comparison bar charts

Usage:
    python scripts/generate_figures.py --results ./outputs/baselines/{timestamp}
    python scripts/generate_figures.py --results ./outputs/baselines/{timestamp} --format pdf
    python scripts/generate_figures.py --all-figures

Output:
    outputs/paper/figures/
    ├── fig1_reliability_diagram.pdf
    ├── fig2_roc_curves.pdf
    ├── fig3_cross_generator_heatmap.pdf
    ├── fig4_degradation_curves.pdf
    ├── fig5_confusion_matrices.pdf
    └── fig6_performance_bars.pdf
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imagetrust.utils.helpers import ensure_dir


# =============================================================================
# Matplotlib Configuration for Publication
# =============================================================================

def setup_matplotlib(use_latex: bool = False):
    """Configure matplotlib for publication-quality figures."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Use a clean style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Publication settings
    mpl.rcParams.update({
        # Font
        'font.family': 'serif' if use_latex else 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,

        # Figure
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 6,

        # Axes
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,

        # Legend
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
    })

    if use_latex:
        mpl.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
        })

    return plt


# =============================================================================
# Color Schemes
# =============================================================================

# Colorblind-friendly palette
COLORS = {
    'classical': '#E69F00',    # Orange
    'cnn': '#56B4E9',          # Sky blue
    'vit': '#009E73',          # Teal
    'imagetrust': '#D55E00',   # Vermillion (our method)
    'real': '#0072B2',         # Blue
    'ai': '#CC79A7',           # Pink
}

BASELINE_COLORS = ['#E69F00', '#56B4E9', '#009E73', '#D55E00']
BASELINE_MARKERS = ['o', 's', '^', 'D']
BASELINE_LINESTYLES = ['-', '--', '-.', ':']


# =============================================================================
# Figure 1: Reliability Diagram (Calibration Curves)
# =============================================================================

def plot_reliability_diagram(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
):
    """
    Plot reliability diagram comparing calibration of all methods.

    Args:
        predictions: Dict mapping method name to (probabilities, labels)
        output_path: Where to save figure
        n_bins: Number of bins for calibration
        title: Figure title
    """
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left plot: Reliability diagram
    ax1 = axes[0]

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1)

    for idx, (name, (probs, labels)) in enumerate(predictions.items()):
        # Compute calibration curve
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_acc = labels[mask].mean()
                bin_conf = probs[mask].mean()
                bin_accuracies.append(bin_acc)
                bin_confidences.append(bin_conf)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(np.nan)
                bin_confidences.append(bin_centers[i])
                bin_counts.append(0)

        color = BASELINE_COLORS[idx % len(BASELINE_COLORS)]
        marker = BASELINE_MARKERS[idx % len(BASELINE_MARKERS)]

        ax1.plot(
            bin_confidences, bin_accuracies,
            color=color, marker=marker, label=name,
            linewidth=1.5, markersize=6
        )

    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('(a) Calibration Curve')
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Right plot: Confidence histogram
    ax2 = axes[1]

    for idx, (name, (probs, labels)) in enumerate(predictions.items()):
        color = BASELINE_COLORS[idx % len(BASELINE_COLORS)]
        ax2.hist(
            probs, bins=20, alpha=0.5, label=name,
            color=color, edgecolor='white', linewidth=0.5
        )

    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('(b) Prediction Distribution')
    ax2.legend(loc='upper right')

    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")


# =============================================================================
# Figure 2: ROC Curves
# =============================================================================

def plot_roc_curves(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str = "ROC Curves",
):
    """
    Plot ROC curves for all methods.

    Args:
        predictions: Dict mapping method name to (probabilities, labels)
        output_path: Where to save figure
        title: Figure title
    """
    from sklearn.metrics import roc_curve, auc

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 5))

    # Random classifier line
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.50)', linewidth=1)

    aucs = {}

    for idx, (name, (probs, labels)) in enumerate(predictions.items()):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        aucs[name] = roc_auc

        color = BASELINE_COLORS[idx % len(BASELINE_COLORS)]
        linestyle = BASELINE_LINESTYLES[idx % len(BASELINE_LINESTYLES)]

        ax.plot(
            fpr, tpr,
            color=color, linestyle=linestyle,
            label=f'{name} (AUC={roc_auc:.3f})',
            linewidth=2
        )

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")
    return aucs


# =============================================================================
# Figure 3: Cross-Generator Heatmap
# =============================================================================

def plot_cross_generator_heatmap(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
    metric: str = "roc_auc",
    title: str = "Cross-Generator Performance",
):
    """
    Plot heatmap of cross-generator performance.

    Args:
        results: Nested dict [method][generator] -> metrics
        output_path: Where to save figure
        metric: Which metric to show (roc_auc, accuracy, f1_score)
        title: Figure title
    """
    plt = setup_matplotlib()

    # Extract data
    methods = list(results.keys())
    generators = list(results[methods[0]].keys()) if methods else []

    # Build matrix
    matrix = np.zeros((len(methods), len(generators)))
    for i, method in enumerate(methods):
        for j, gen in enumerate(generators):
            if gen in results[method]:
                val = results[method][gen]
                if isinstance(val, dict):
                    matrix[i, j] = val.get(metric, 0)
                else:
                    matrix[i, j] = val

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)

    # Labels
    ax.set_xticks(np.arange(len(generators)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels([g.replace('_', ' ').title() for g in generators], rotation=45, ha='right')
    ax.set_yticklabels(methods)

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(generators)):
            val = matrix[i, j]
            color = 'white' if val < 0.7 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').upper())

    ax.set_title(title)
    ax.set_xlabel('Test Generator')
    ax.set_ylabel('Method')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")


# =============================================================================
# Figure 4: Degradation Performance Curves
# =============================================================================

def plot_degradation_curves(
    results: Dict[str, Dict[str, Dict]],
    output_path: Path,
    title: str = "Robustness to Image Degradations",
):
    """
    Plot performance curves under various degradations.

    Args:
        results: Nested dict [method][degradation_type][param] -> metrics
        output_path: Where to save figure
        title: Figure title
    """
    plt = setup_matplotlib()

    # Degradation types to plot
    degradation_types = ['jpeg_compression', 'blur', 'resize', 'noise']
    deg_labels = ['JPEG Quality', 'Blur Radius', 'Resize Factor', 'Noise Sigma']
    deg_xlabels = ['Quality', 'Radius (σ)', 'Factor', 'Sigma']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    methods = list(results.keys())

    for ax_idx, (deg_type, deg_label, x_label) in enumerate(zip(
        degradation_types, deg_labels, deg_xlabels
    )):
        ax = axes[ax_idx]

        for method_idx, method in enumerate(methods):
            if deg_type not in results[method]:
                continue

            deg_data = results[method][deg_type]

            # Extract x values and AUC scores
            x_vals = sorted([float(k) for k in deg_data.keys()])
            y_vals = []

            for x in x_vals:
                metrics = deg_data.get(str(x)) or deg_data.get(x) or deg_data.get(int(x))
                if metrics:
                    y_vals.append(metrics.get('roc_auc', metrics.get('accuracy', 0)))
                else:
                    y_vals.append(0)

            color = BASELINE_COLORS[method_idx % len(BASELINE_COLORS)]
            marker = BASELINE_MARKERS[method_idx % len(BASELINE_MARKERS)]

            ax.plot(
                x_vals, y_vals,
                color=color, marker=marker, label=method,
                linewidth=1.5, markersize=5
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel('AUC')
        ax.set_title(f'({chr(97 + ax_idx)}) {deg_label}')
        ax.set_ylim([0.5, 1.0])

        if ax_idx == 0:
            ax.legend(loc='lower left', fontsize=8)

    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")


# =============================================================================
# Figure 5: Confusion Matrices
# =============================================================================

def plot_confusion_matrices(
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str = "Confusion Matrices",
):
    """
    Plot confusion matrices for all methods.

    Args:
        predictions: Dict mapping method name to (probabilities, labels)
        output_path: Where to save figure
        title: Figure title
    """
    from sklearn.metrics import confusion_matrix

    plt = setup_matplotlib()

    n_methods = len(predictions)
    cols = min(4, n_methods)
    rows = (n_methods + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, (probs, labels)) in enumerate(predictions.items()):
        ax = axes[idx]

        preds = (probs > 0.5).astype(int)
        cm = confusion_matrix(labels, preds)

        # Normalize
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Plot
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Real', 'AI'])
        ax.set_yticklabels(['Real', 'AI'])

        # Add counts
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = cm_norm[i, j]
                color = 'white' if pct > 0.5 else 'black'
                ax.text(j, i, f'{val}\n({pct:.1%})', ha='center', va='center',
                       color=color, fontsize=9)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(name)

    # Hide unused axes
    for idx in range(len(predictions), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")


# =============================================================================
# Figure 6: Performance Bar Chart
# =============================================================================

def plot_performance_bars(
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
    metrics_to_show: List[str] = ['accuracy', 'f1_score', 'roc_auc'],
    title: str = "Performance Comparison",
):
    """
    Plot grouped bar chart of performance metrics.

    Args:
        metrics: Dict mapping method name to metrics dict
        output_path: Where to save figure
        metrics_to_show: Which metrics to include
        title: Figure title
    """
    plt = setup_matplotlib()

    methods = list(metrics.keys())
    n_methods = len(methods)
    n_metrics = len(metrics_to_show)

    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, method in enumerate(methods):
        values = [metrics[method].get(m, 0) for m in metrics_to_show]
        color = BASELINE_COLORS[idx % len(BASELINE_COLORS)]

        bars = ax.bar(
            x + idx * width - (n_methods - 1) * width / 2,
            values,
            width,
            label=method,
            color=color,
            edgecolor='white',
            linewidth=0.5
        )

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{val:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=7
            )

    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_show])
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")


# =============================================================================
# Figure 7: ECE Comparison (Before/After Calibration)
# =============================================================================

def plot_ece_comparison(
    ece_before: Dict[str, float],
    ece_after: Dict[str, float],
    output_path: Path,
    title: str = "Calibration Improvement (ECE)",
):
    """
    Plot ECE before and after calibration.

    Args:
        ece_before: Dict mapping method to ECE before calibration
        ece_after: Dict mapping method to ECE after calibration
        output_path: Where to save figure
        title: Figure title
    """
    plt = setup_matplotlib()

    methods = list(ece_before.keys())
    n_methods = len(methods)

    x = np.arange(n_methods)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width/2, [ece_before[m] for m in methods], width,
                   label='Before Calibration', color='#CC79A7', alpha=0.8)
    bars2 = ax.bar(x + width/2, [ece_after[m] for m in methods], width,
                   label='After Calibration', color='#009E73', alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8
            )

    ax.set_ylabel('Expected Calibration Error (ECE) ↓')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend()

    # Add improvement arrows
    for i, m in enumerate(methods):
        if ece_before[m] > ece_after[m]:
            improvement = (ece_before[m] - ece_after[m]) / ece_before[m] * 100
            ax.annotate(
                f'↓{improvement:.0f}%',
                xy=(i, max(ece_before[m], ece_after[m]) + 0.02),
                ha='center', va='bottom',
                fontsize=8, color='green'
            )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")


# =============================================================================
# Generate All Figures from Results
# =============================================================================

def generate_all_figures(
    results_dir: Path,
    output_dir: Path,
    format: str = 'pdf',
):
    """
    Generate all figures from evaluation results.

    Args:
        results_dir: Directory containing evaluation results
        output_dir: Where to save figures
        format: Output format (pdf, png, svg)
    """
    ensure_dir(output_dir)

    print(f"\nGenerating figures from: {results_dir}")
    print(f"Output directory: {output_dir}\n")

    # Load results
    main_results_path = results_dir / 'main_results.json'
    cross_gen_path = results_dir / 'cross_generator.json'
    degradation_path = results_dir / 'degradation.json'

    # Check what results are available
    has_main = main_results_path.exists()
    has_cross_gen = cross_gen_path.exists()
    has_degradation = degradation_path.exists()

    if has_main:
        with open(main_results_path) as f:
            main_results = json.load(f)
        print(f"Loaded main results: {len(main_results)} methods")

        # Figure 6: Performance bars
        plot_performance_bars(
            main_results,
            output_dir / f'fig6_performance_bars.{format}',
            metrics_to_show=['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc'],
        )

    if has_cross_gen:
        with open(cross_gen_path) as f:
            cross_gen_results = json.load(f)
        print(f"Loaded cross-generator results")

        # Figure 3: Cross-generator heatmap
        plot_cross_generator_heatmap(
            cross_gen_results,
            output_dir / f'fig3_cross_generator_heatmap.{format}',
        )

    if has_degradation:
        with open(degradation_path) as f:
            degradation_results = json.load(f)
        print(f"Loaded degradation results")

        # Figure 4: Degradation curves
        plot_degradation_curves(
            degradation_results,
            output_dir / f'fig4_degradation_curves.{format}',
        )

    print(f"\n✅ Figure generation complete. Output in: {output_dir}")


def generate_demo_figures(output_dir: Path, format: str = 'pdf'):
    """
    Generate demo figures with synthetic data (for testing layout).
    """
    ensure_dir(output_dir)
    np.random.seed(42)

    print("Generating demo figures with synthetic data...")

    # Demo data
    methods = ['Classical (LogReg)', 'CNN (ResNet-50)', 'ViT-B/16', 'ImageTrust (Ours)']

    # Demo predictions (probs, labels)
    n_samples = 500
    demo_predictions = {}
    for i, method in enumerate(methods):
        # Simulate different calibration qualities
        labels = np.random.randint(0, 2, n_samples)
        noise = 0.3 - i * 0.05  # Better methods have less noise
        probs = labels * (0.7 + i * 0.05) + (1 - labels) * (0.3 - i * 0.05)
        probs += np.random.normal(0, noise, n_samples)
        probs = np.clip(probs, 0, 1)
        demo_predictions[method] = (probs, labels)

    # Figure 1: Reliability diagram
    plot_reliability_diagram(
        demo_predictions,
        output_dir / f'fig1_reliability_diagram.{format}',
    )

    # Figure 2: ROC curves
    plot_roc_curves(
        demo_predictions,
        output_dir / f'fig2_roc_curves.{format}',
    )

    # Figure 5: Confusion matrices
    plot_confusion_matrices(
        demo_predictions,
        output_dir / f'fig5_confusion_matrices.{format}',
    )

    # Demo cross-generator results
    generators = ['real', 'midjourney', 'dalle3', 'stable_diffusion', 'firefly']
    demo_cross_gen = {}
    for method in methods:
        demo_cross_gen[method] = {}
        base_auc = 0.75 + methods.index(method) * 0.05
        for gen in generators:
            demo_cross_gen[method][gen] = {
                'roc_auc': base_auc + np.random.uniform(-0.1, 0.1)
            }

    # Figure 3: Cross-generator heatmap
    plot_cross_generator_heatmap(
        demo_cross_gen,
        output_dir / f'fig3_cross_generator_heatmap.{format}',
    )

    # Demo degradation results
    demo_degradation = {}
    for method in methods:
        base_auc = 0.85 + methods.index(method) * 0.03
        demo_degradation[method] = {
            'jpeg_compression': {
                '95': {'roc_auc': base_auc},
                '85': {'roc_auc': base_auc - 0.02},
                '70': {'roc_auc': base_auc - 0.05},
                '50': {'roc_auc': base_auc - 0.10},
            },
            'blur': {
                '0': {'roc_auc': base_auc},
                '0.5': {'roc_auc': base_auc - 0.03},
                '1.0': {'roc_auc': base_auc - 0.07},
                '2.0': {'roc_auc': base_auc - 0.12},
            },
            'resize': {
                '1.0': {'roc_auc': base_auc},
                '0.75': {'roc_auc': base_auc - 0.02},
                '0.5': {'roc_auc': base_auc - 0.06},
            },
            'noise': {
                '0': {'roc_auc': base_auc},
                '0.01': {'roc_auc': base_auc - 0.02},
                '0.03': {'roc_auc': base_auc - 0.05},
            },
        }

    # Figure 4: Degradation curves
    plot_degradation_curves(
        demo_degradation,
        output_dir / f'fig4_degradation_curves.{format}',
    )

    # Demo metrics
    demo_metrics = {}
    for i, method in enumerate(methods):
        demo_metrics[method] = {
            'accuracy': 0.78 + i * 0.04 + np.random.uniform(-0.02, 0.02),
            'balanced_accuracy': 0.76 + i * 0.04 + np.random.uniform(-0.02, 0.02),
            'f1_score': 0.75 + i * 0.05 + np.random.uniform(-0.02, 0.02),
            'roc_auc': 0.82 + i * 0.04 + np.random.uniform(-0.02, 0.02),
            'ece': 0.15 - i * 0.02 + np.random.uniform(-0.01, 0.01),
        }

    # Figure 6: Performance bars
    plot_performance_bars(
        demo_metrics,
        output_dir / f'fig6_performance_bars.{format}',
    )

    # Demo ECE comparison
    ece_before = {m: demo_metrics[m]['ece'] + 0.05 for m in methods}
    ece_after = {m: demo_metrics[m]['ece'] for m in methods}

    # Figure 7: ECE comparison
    plot_ece_comparison(
        ece_before,
        ece_after,
        output_dir / f'fig7_ece_comparison.{format}',
    )

    print(f"\n✅ Demo figures generated in: {output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures for ImageTrust thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--results', '-r',
        type=str,
        default=None,
        help='Path to results directory (from run_baselines.py)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/paper/figures',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='pdf',
        choices=['pdf', 'png', 'svg', 'eps'],
        help='Output format'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Generate demo figures with synthetic data'
    )
    parser.add_argument(
        '--latex',
        action='store_true',
        help='Use LaTeX for text rendering (requires LaTeX installation)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.demo:
        generate_demo_figures(output_dir, args.format)
    elif args.results:
        results_dir = Path(args.results)
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            sys.exit(1)
        generate_all_figures(results_dir, output_dir, args.format)
    else:
        print("Error: Either --results or --demo must be specified")
        print("\nExamples:")
        print("  python generate_figures.py --demo")
        print("  python generate_figures.py --results outputs/baselines/20240101_120000")
        sys.exit(1)


if __name__ == "__main__":
    main()
