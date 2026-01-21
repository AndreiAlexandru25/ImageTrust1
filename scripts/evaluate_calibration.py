#!/usr/bin/env python3
"""
Calibration and uncertainty evaluation for ImageTrust baselines.

This script evaluates:
1. Calibration quality (ECE, MCE, Brier) before and after calibration
2. Uncertainty estimation quality
3. Selective prediction (abstain) trade-offs
4. Generates reliability diagrams and coverage-accuracy curves

Usage:
    # Evaluate calibration on all baselines
    python scripts/evaluate_calibration.py --splits-dir ./data/splits

    # Compare calibration methods
    python scripts/evaluate_calibration.py --compare-methods

    # Analyze selective prediction
    python scripts/evaluate_calibration.py --selective-prediction --target-coverage 0.9

    # Generate figures only
    python scripts/evaluate_calibration.py --generate-figures --results-file ./outputs/calibration/results.json

Author: ImageTrust Team
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def generate_reliability_diagram(
    reliability_data: Dict[str, Any],
    baseline_name: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Generate reliability diagram comparing before/after calibration."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available for figure generation")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for idx, (key, label) in enumerate([("before", "Before"), ("after", "After")]):
        ax = axes[idx]
        data = reliability_data.get(key, {})

        frac_pos = data.get("fraction_positives", [])
        mean_pred = data.get("mean_predicted", [])

        if frac_pos and mean_pred:
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], "k--", label="Perfect", alpha=0.7)

            # Actual calibration
            ax.plot(mean_pred, frac_pos, "o-", color="C0", label=baseline_name)
            ax.fill_between(
                mean_pred,
                frac_pos,
                mean_pred,
                alpha=0.2,
                color="C0",
            )

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"{label} Calibration")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc="lower right")
        ax.set_aspect("equal")

    fig.suptitle(title or f"Reliability Diagram - {baseline_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_coverage_accuracy_curve(
    curve_data: Dict[str, List[float]],
    output_path: Path,
    title: str = "Coverage vs Accuracy",
) -> None:
    """Generate coverage-accuracy trade-off curve."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    coverages = curve_data.get("coverage", [])
    accuracies = curve_data.get("accuracy", [])

    if coverages and accuracies:
        ax.plot(coverages, accuracies, "o-", color="C0", linewidth=2, markersize=4)
        ax.fill_between(coverages, 0, accuracies, alpha=0.2, color="C0")

        # Mark 90% coverage point
        for cov, acc in zip(coverages, accuracies):
            if cov >= 0.9:
                ax.axvline(x=cov, color="red", linestyle="--", alpha=0.5, label=f"90% coverage: {acc:.2%} acc")
                ax.scatter([cov], [acc], color="red", s=100, zorder=5)
                break

    ax.set_xlabel("Coverage (1 - Abstain Rate)")
    ax.set_ylabel("Accuracy on Covered Samples")
    ax.set_title(title)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_calibration_comparison(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Generate bar chart comparing ECE across methods."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = list(results.keys())
    ece_before = [results[m].get("ece_before", 0) for m in methods]
    ece_after = [results[m].get("ece_after", 0) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width/2, ece_before, width, label="Before", color="C1", alpha=0.8)
    bars2 = ax.bar(x + width/2, ece_after, width, label="After", color="C0", alpha=0.8)

    ax.set_xlabel("Calibration Method")
    ax.set_ylabel("ECE (Lower is Better)")
    ax.set_title("Expected Calibration Error Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in methods])
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_ece_improvement_table(
    baseline_results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate LaTeX table for ECE improvement."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Calibration improvement using temperature scaling.}",
        r"\label{tab:calibration}",
        r"\begin{tabular}{l|cc|cc|c}",
        r"\toprule",
        r"& \multicolumn{2}{c|}{ECE} & \multicolumn{2}{c|}{Brier} & \\",
        r"Method & Before & After & Before & After & Temp. \\",
        r"\midrule",
    ]

    for baseline, data in baseline_results.items():
        temp = data.get("temperature", "-")
        temp_str = f"{temp:.2f}" if isinstance(temp, float) else "-"

        lines.append(
            f"{baseline} & {data.get('ece_before', 0):.3f} & {data.get('ece_after', 0):.3f} & "
            f"{data.get('brier_before', 0):.3f} & {data.get('brier_after', 0):.3f} & "
            f"{temp_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def run_calibration_evaluation(
    splits_dir: Path,
    output_dir: Path,
    baselines: List[str],
    calibration_method: str = "temperature",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run calibration evaluation on all baselines.

    Returns dictionary with all results.
    """
    from imagetrust.data.splits import load_split
    from imagetrust.baselines.calibration import BaselineCalibrator, compare_calibration_methods

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load split
    split_path = splits_dir / "default_split.json"
    if not split_path.exists():
        print(f"Error: Split file not found: {split_path}")
        return {}

    split = load_split(split_path)

    if verbose:
        print("=" * 60)
        print("CALIBRATION EVALUATION")
        print("=" * 60)
        print(f"Split: {split.name}")
        print(f"Val samples: {len(split.val)}")
        print(f"Test samples: {len(split.test)}")
        print(f"Method: {calibration_method}")
        print("-" * 60)

    results = {
        "metadata": {
            "split": split.name,
            "calibration_method": calibration_method,
            "val_samples": len(split.val),
            "test_samples": len(split.test),
        },
        "baselines": {},
        "method_comparison": {},
    }

    # Simulate evaluation (in real usage, load actual baseline predictions)
    for baseline in baselines:
        if verbose:
            print(f"\nEvaluating {baseline}...")

        # Placeholder: In real usage, load predictions from baseline
        # Here we generate synthetic data for demonstration
        np.random.seed(42)
        n_val = len(split.val)
        n_test = len(split.test)

        # Simulate uncalibrated probabilities (overconfident)
        val_labels = np.array([item["label"] for item in split.val])
        test_labels = np.array([item["label"] for item in split.test])

        # Generate synthetic predictions (replace with actual baseline predictions)
        val_probs = _generate_synthetic_predictions(val_labels, overconfidence=0.15)
        test_probs = _generate_synthetic_predictions(test_labels, overconfidence=0.15)

        # Fit calibrator on validation set
        calibrator = BaselineCalibrator(method=calibration_method)
        calibrator.fit(val_probs, val_labels)

        # Evaluate on test set
        cal_result = calibrator.evaluate(test_probs, test_labels)

        results["baselines"][baseline] = cal_result.to_dict()
        results["baselines"][baseline]["reliability_data"] = cal_result.reliability_data

        if verbose:
            print(f"  ECE: {cal_result.ece_before:.4f} -> {cal_result.ece_after:.4f}")
            print(f"  Improvement: {cal_result.improvement():.4f}")
            if cal_result.temperature:
                print(f"  Temperature: {cal_result.temperature:.4f}")

        # Compare all calibration methods for first baseline
        if baseline == baselines[0]:
            method_results = compare_calibration_methods(test_probs, test_labels)
            results["method_comparison"] = {
                m: r.to_dict() for m, r in method_results.items()
            }

    return results


def run_selective_prediction_evaluation(
    splits_dir: Path,
    output_dir: Path,
    baselines: List[str],
    target_coverage: float = 0.9,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run selective prediction (abstain) evaluation."""
    from imagetrust.data.splits import load_split
    from imagetrust.baselines.uncertainty import (
        UncertaintyEstimator,
        analyze_abstain_characteristics,
        compute_risk_coverage_auc,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = splits_dir / "default_split.json"
    if not split_path.exists():
        return {}

    split = load_split(split_path)

    if verbose:
        print("\n" + "=" * 60)
        print("SELECTIVE PREDICTION (ABSTAIN) EVALUATION")
        print("=" * 60)
        print(f"Target coverage: {target_coverage:.0%}")
        print("-" * 60)

    results = {
        "metadata": {
            "target_coverage": target_coverage,
        },
        "baselines": {},
    }

    for baseline in baselines:
        if verbose:
            print(f"\nEvaluating {baseline}...")

        # Generate synthetic predictions (replace with actual)
        test_labels = np.array([item["label"] for item in split.test])
        test_probs = _generate_synthetic_predictions(test_labels, overconfidence=0.1)

        # Fit uncertainty estimator
        estimator = UncertaintyEstimator(
            method="entropy",
            target_coverage=target_coverage,
        )

        val_labels = np.array([item["label"] for item in split.val])
        val_probs = _generate_synthetic_predictions(val_labels, overconfidence=0.1)
        threshold = estimator.fit_threshold(val_probs, val_labels)

        # Evaluate
        sel_result = estimator.evaluate_selective_prediction(test_probs, test_labels)
        curve = estimator.compute_coverage_accuracy_curve(test_probs, test_labels)
        characteristics = analyze_abstain_characteristics(
            test_probs, test_labels, "entropy", threshold
        )
        aurc = compute_risk_coverage_auc(test_probs, test_labels)

        results["baselines"][baseline] = {
            **sel_result.to_dict(),
            "coverage_accuracy_curve": curve,
            "abstain_characteristics": characteristics,
            "aurc": aurc,
            "fitted_threshold": threshold,
        }

        if verbose:
            print(f"  Coverage: {sel_result.coverage:.2%}")
            print(f"  Accuracy on covered: {sel_result.accuracy_on_covered:.2%}")
            print(f"  Abstain rate: {sel_result.abstain_rate:.2%}")
            print(f"  AURC: {aurc:.4f}")
            print(f"  Error rejection rate: {characteristics['error_rejection_rate']:.2%}")

    return results


def _generate_synthetic_predictions(
    labels: np.ndarray,
    overconfidence: float = 0.1,
    noise: float = 0.2,
) -> np.ndarray:
    """Generate synthetic predictions for demonstration."""
    np.random.seed(42)
    n = len(labels)

    # Base predictions correlated with labels
    base = labels * 0.7 + 0.15 + np.random.randn(n) * noise

    # Add overconfidence
    base = np.clip(base, 0.05, 0.95)
    base = np.where(base > 0.5, base + overconfidence, base - overconfidence)
    base = np.clip(base, 0.01, 0.99)

    return base


def main():
    parser = argparse.ArgumentParser(
        description="Calibration and uncertainty evaluation for ImageTrust",
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing split files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/calibration"),
        help="Output directory",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        default=["classical", "cnn", "vit", "imagetrust"],
        help="Baselines to evaluate",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["temperature", "platt", "isotonic"],
        default="temperature",
        help="Calibration method",
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Compare all calibration methods",
    )
    parser.add_argument(
        "--selective-prediction",
        action="store_true",
        help="Run selective prediction analysis",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.9,
        help="Target coverage for selective prediction",
    )
    parser.add_argument(
        "--generate-figures",
        action="store_true",
        help="Generate figures from results",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help="Load results from file instead of running evaluation",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
    )

    args = parser.parse_args()
    verbose = args.verbose and not args.quiet

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load or compute results
    if args.results_file and args.results_file.exists():
        with open(args.results_file) as f:
            results = json.load(f)
    else:
        results = {}

        # Run calibration evaluation
        cal_results = run_calibration_evaluation(
            args.splits_dir,
            args.output_dir,
            args.baseline,
            args.calibration_method,
            verbose,
        )
        results["calibration"] = cal_results

        # Run selective prediction evaluation
        if args.selective_prediction:
            sel_results = run_selective_prediction_evaluation(
                args.splits_dir,
                args.output_dir,
                args.baseline,
                args.target_coverage,
                verbose,
            )
            results["selective_prediction"] = sel_results

        # Save results
        results_path = args.output_dir / "calibration_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        if verbose:
            print(f"\nResults saved to: {results_path}")

    # Generate figures
    if args.generate_figures:
        if verbose:
            print("\nGenerating figures...")

        figures_dir = args.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Reliability diagrams
        cal_data = results.get("calibration", {}).get("baselines", {})
        for baseline, data in cal_data.items():
            if "reliability_data" in data:
                generate_reliability_diagram(
                    data["reliability_data"],
                    baseline,
                    figures_dir / f"reliability_{baseline}.pdf",
                )

        # Method comparison
        method_data = results.get("calibration", {}).get("method_comparison", {})
        if method_data:
            generate_calibration_comparison(
                method_data,
                figures_dir / "ece_comparison.pdf",
            )

        # Coverage-accuracy curves
        sel_data = results.get("selective_prediction", {}).get("baselines", {})
        for baseline, data in sel_data.items():
            if "coverage_accuracy_curve" in data:
                generate_coverage_accuracy_curve(
                    data["coverage_accuracy_curve"],
                    figures_dir / f"coverage_accuracy_{baseline}.pdf",
                    title=f"Coverage-Accuracy Trade-off - {baseline}",
                )

        # LaTeX tables
        tables_dir = args.output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        generate_ece_improvement_table(cal_data, tables_dir / "table_calibration.tex")

        if verbose:
            print(f"Figures saved to: {figures_dir}")
            print(f"Tables saved to: {tables_dir}")

    if verbose:
        print("\n" + "=" * 60)
        print("CALIBRATION EVALUATION COMPLETE")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
