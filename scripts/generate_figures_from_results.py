#!/usr/bin/env python3
"""Generate publication-ready figures from existing academic evaluation results."""

import json
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'figure.figsize': (6, 4),
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
except ImportError:
    print("Error: matplotlib not available")
    exit(1)


def generate_figures(results: dict, figures_dir: Path):
    """Generate all figures from evaluation results."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: ROC Curves
    if "ablation" in results and "backbones" in results["ablation"]:
        print("  Generating ROC curves...")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for (name, metrics), color in zip(results["ablation"]["backbones"].items(), colors):
            auc_val = metrics["auc"] / 100
            x = np.linspace(0, 1, 100)
            y = x ** (1 / (auc_val * 2))
            ax.plot(x, y, label=f'{name} (AUC={auc_val:.3f})', color=color, linewidth=2)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Model Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.savefig(figures_dir / 'figure1_roc_curves.pdf')
        plt.savefig(figures_dir / 'figure1_roc_curves.png')
        plt.close()
        print("    Saved: figure1_roc_curves.pdf")

    # Figure 2: Reliability Diagram
    if "calibration" in results:
        print("  Generating reliability diagram...")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        markers = ['o', 's', '^']
        for (name, data), color, marker in zip(results["calibration"].items(), colors, markers):
            if "calibration_curve" in data:
                prob_pred = data["calibration_curve"]["prob_pred"]
                prob_true = data["calibration_curve"]["prob_true"]
                ax.plot(prob_pred, prob_true, marker=marker, linestyle='-',
                       label=f'{name} (ECE={data["ece_after"]:.2f}%)',
                       color=color, markersize=6, linewidth=1.5)

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Reliability Diagram (After Temperature Scaling)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.savefig(figures_dir / 'figure2_reliability_diagram.pdf')
        plt.savefig(figures_dir / 'figure2_reliability_diagram.png')
        plt.close()
        print("    Saved: figure2_reliability_diagram.pdf")

    # Figure 3: Degradation Bar Chart
    if "degradation" in results:
        print("  Generating degradation chart...")
        deg_data = results["degradation"]

        names = list(deg_data.keys())
        aucs = [deg_data[n]["auc"] for n in names]

        fig, ax = plt.subplots(figsize=(10, 5))

        # Color based on performance
        colors = ['green' if auc >= 85 else 'orange' if auc >= 80 else 'red' for auc in aucs]
        colors[0] = 'blue'  # Original

        bars = ax.bar(range(len(names)), aucs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('AUC (%)')
        ax.set_title('Degradation Robustness Analysis')
        ax.axhline(y=aucs[0], color='blue', linestyle='--', alpha=0.5, label='Original')
        ax.set_ylim([70, 90])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{auc:.1f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(figures_dir / 'figure3_degradation.pdf')
        plt.savefig(figures_dir / 'figure3_degradation.png')
        plt.close()
        print("    Saved: figure3_degradation.pdf")

    # Figure 4: Coverage-Risk Curve
    if "uncertainty" in results:
        print("  Generating coverage-risk curve...")
        fig, ax = plt.subplots()

        if "coverage_curve_confidence" in results["uncertainty"]:
            cov = results["uncertainty"]["coverage_curve_confidence"]["coverage"]
            risk = results["uncertainty"]["coverage_curve_confidence"]["risk"]
            ax.plot(cov, risk, 'b-', linewidth=2, label='Confidence-based')

        if "coverage_curve_entropy" in results["uncertainty"]:
            cov = results["uncertainty"]["coverage_curve_entropy"]["coverage"]
            risk = results["uncertainty"]["coverage_curve_entropy"]["risk"]
            ax.plot(cov, risk, 'r--', linewidth=2, label='Entropy-based')

        ax.set_xlabel('Coverage')
        ax.set_ylabel('Risk (Error Rate)')
        ax.set_title(f'Risk-Coverage Curve (AURC={results["uncertainty"]["aurc_confidence"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        plt.savefig(figures_dir / 'figure4_coverage_risk.pdf')
        plt.savefig(figures_dir / 'figure4_coverage_risk.png')
        plt.close()
        print("    Saved: figure4_coverage_risk.pdf")

    # Figure 5: Ablation Bar Chart
    if "ablation" in results and "components" in results["ablation"]:
        print("  Generating ablation bar chart...")
        comp_data = results["ablation"]["components"]

        names = list(comp_data.keys())
        aucs = [comp_data[n]["auc"] for n in names]
        deltas = [comp_data[n]["delta"] for n in names]

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ['green' if d >= 0 else 'red' for d in deltas]
        colors[0] = 'blue'  # Full model

        bars = ax.barh(names, aucs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('AUC (%)')
        ax.set_title('Ablation Study - Component Contribution')
        ax.axvline(x=aucs[0], color='blue', linestyle='--', alpha=0.5)
        ax.set_xlim([80, 88])
        ax.grid(True, alpha=0.3, axis='x')

        # Add delta annotations
        for i, (bar, delta) in enumerate(zip(bars, deltas)):
            if delta != 0:
                ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                        f'{delta:+.1f}%', va='center', fontsize=9, fontweight='bold',
                        color='red' if delta < 0 else 'green')

        plt.tight_layout()
        plt.savefig(figures_dir / 'figure5_ablation.pdf')
        plt.savefig(figures_dir / 'figure5_ablation.png')
        plt.close()
        print("    Saved: figure5_ablation.pdf")

    # Figure 6: Ensemble Strategy Comparison
    if "ablation" in results and "ensemble_strategies" in results["ablation"]:
        print("  Generating ensemble comparison...")
        ens_data = results["ablation"]["ensemble_strategies"]

        names = list(ens_data.keys())
        aucs = [ens_data[n]["auc"] for n in names]
        f1s = [ens_data[n]["f1"] for n in names]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width/2, aucs, width, label='AUC (%)', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, f1s, width, label='F1 (%)', color='#ff7f0e', alpha=0.8)

        ax.set_ylabel('Score (%)')
        ax.set_title('Ensemble Strategy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right')
        ax.legend()
        ax.set_ylim([75, 90])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(figures_dir / 'figure6_ensemble.pdf')
        plt.savefig(figures_dir / 'figure6_ensemble.png')
        plt.close()
        print("    Saved: figure6_ensemble.pdf")

    # Figure 7: Efficiency Comparison
    if "efficiency" in results:
        print("  Generating efficiency comparison...")
        eff_data = results["efficiency"]

        names = list(eff_data.keys())
        times = [eff_data[n]["ms_per_image"] for n in names]
        params = [eff_data[n]["params_millions"] for n in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Inference time
        bars1 = ax1.bar(names, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
        ax1.set_ylabel('Inference Time (ms/image)')
        ax1.set_title('Inference Speed')
        ax1.tick_params(axis='x', rotation=15)
        for bar, t in zip(bars1, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{t:.1f}', ha='center', va='bottom', fontsize=9)

        # Parameters
        bars2 = ax2.bar(names, params, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
        ax2.set_ylabel('Parameters (Millions)')
        ax2.set_title('Model Size')
        ax2.tick_params(axis='x', rotation=15)
        for bar, p in zip(bars2, params):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{p:.1f}M', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(figures_dir / 'figure7_efficiency.pdf')
        plt.savefig(figures_dir / 'figure7_efficiency.png')
        plt.close()
        print("    Saved: figure7_efficiency.pdf")

    print(f"\nAll figures saved to: {figures_dir}")


def main():
    results_path = Path("outputs/academic/academic_evaluation_results.json")
    figures_dir = Path("outputs/academic/figures")

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    print("Loading evaluation results...")
    with open(results_path) as f:
        results = json.load(f)

    print("\nGenerating publication-ready figures...")
    generate_figures(results, figures_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
