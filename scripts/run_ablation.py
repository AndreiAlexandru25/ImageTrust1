#!/usr/bin/env python3
"""
Ablation study evaluation script for ImageTrust.

Runs systematic ablation experiments to evaluate component importance:
1. Baseline (full model)
2. Backbone architectures (EfficientNet, ConvNext, ViT)
3. Ensemble strategies (average, weighted, voting, max)
4. Calibration methods (with/without)
5. Signal analysis components

Usage:
    # Run full ablation study
    python scripts/run_ablation.py --splits-dir ./data/splits

    # Run specific ablation type
    python scripts/run_ablation.py --ablation-type backbone

    # Generate LaTeX tables
    python scripts/run_ablation.py --generate-tables

Author: ImageTrust Team
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def run_ablation_study(
    dataset_dir: Path,
    output_dir: Path,
    ablation_types: List[str],
    device: Optional[str] = None,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run ablation study experiments using the real AblationStudy class.

    Args:
        dataset_dir: Directory containing test images (real/ and ai_generated/)
        output_dir: Output directory for results
        ablation_types: Types of ablation to run
        device: Device (cuda/cpu)
        max_samples: Maximum samples to use (for testing)
        verbose: Verbose output

    Returns:
        Dictionary with all ablation results
    """
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("IMAGETRUST ABLATION STUDY")
        print("=" * 60)
        print(f"Dataset: {dataset_dir}")
        print(f"Ablation types: {', '.join(ablation_types)}")
        print("-" * 60)

    # Load dataset
    dataset = load_dataset(dataset_dir, max_samples=max_samples)

    if not dataset:
        print("Error: No images found in dataset")
        return {}

    if verbose:
        print(f"Loaded {len(dataset)} images")

    # Initialize detector
    if verbose:
        print("\nInitializing ComprehensiveDetector...")

    from imagetrust.detection.multi_detector import ComprehensiveDetector
    detector = ComprehensiveDetector(device=device)

    # Run ablation study
    if verbose:
        print("Running ablation study...")

    from imagetrust.evaluation.ablation import AblationStudy

    ablation = AblationStudy(
        detector=detector,
        val_dataset=dataset,
        output_dir=output_dir,
        verbose=verbose,
        device=device or detector.device,
    )

    results = ablation.run_full_study()

    # Print summary
    if verbose:
        ablation.print_summary()

    # Save results
    results_path = ablation.save_results()

    if verbose:
        print(f"\nResults saved to: {results_path}")

    # Generate LaTeX table
    latex_table = ablation.generate_latex_table()
    latex_path = output_dir / "table_ablation.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)

    if verbose:
        print(f"LaTeX table saved to: {latex_path}")

    return results


def load_dataset(dataset_path: Path, max_samples: Optional[int] = None) -> List[Tuple[Any, int]]:
    """
    Load dataset from directory structure.

    Expected structure:
        dataset_path/
            real/
                image1.jpg
                ...
            ai_generated/
                image1.jpg
                ...
    """
    from PIL import Image

    dataset = []

    # Load real images (label=0)
    real_dir = dataset_path / "real"
    if real_dir.exists():
        for img_path in real_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                try:
                    img = Image.open(img_path).convert("RGB")
                    dataset.append((img, 0))
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")

    # Load AI images (label=1)
    for ai_dir_name in ["ai_generated", "ai", "fake", "synthetic"]:
        ai_dir = dataset_path / ai_dir_name
        if ai_dir.exists():
            for img_path in ai_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        dataset.append((img, 1))
                    except Exception as e:
                        print(f"Warning: Failed to load {img_path}: {e}")

    if max_samples and len(dataset) > max_samples:
        import random
        random.seed(42)
        dataset = random.sample(dataset, max_samples)

    return dataset


def run_ablation_study_simulated(
    splits_dir: Path,
    output_dir: Path,
    ablation_types: List[str],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run ablation study with simulated data (for demo/testing).

    Args:
        splits_dir: Directory containing split files
        output_dir: Output directory for results
        ablation_types: Types of ablation to run
        verbose: Verbose output

    Returns:
        Dictionary with all ablation results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("IMAGETRUST ABLATION STUDY (SIMULATED DATA)")
        print("=" * 60)
        print(f"Ablation types: {', '.join(ablation_types)}")
        print("-" * 60)

    # Create synthetic test labels
    n_samples = 1000
    test_labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_samples": n_samples,
            "simulated": True,
        },
        "baseline": {},
        "ablations": {},
    }

    # 1. Baseline (full ImageTrust)
    if verbose:
        print("\n[1] Evaluating baseline (full ImageTrust)...")

    baseline_metrics = _simulate_evaluation(test_labels, accuracy=0.92)
    results["baseline"] = baseline_metrics

    if verbose:
        print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
        print(f"  AUC: {baseline_metrics['auc']:.4f}")
        print(f"  F1: {baseline_metrics['f1']:.4f}")

    # 2. Backbone ablation
    if "backbone" in ablation_types or "all" in ablation_types:
        if verbose:
            print("\n[2] Backbone architecture ablation...")

        backbones = {
            "Deepfake-vs-Real": 0.91,
            "AI-Image-Detector": 0.88,
            "AIorNot": 0.87,
            "NYUAD-2025": 0.89,
        }

        results["backbone_ablation"] = {}
        for backbone, target_acc in backbones.items():
            metrics = _simulate_evaluation(test_labels, accuracy=target_acc)
            results["backbone_ablation"][backbone] = metrics

            if verbose:
                print(f"  {backbone}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")

        # Add ranking
        ranked = sorted(
            [(k, v["f1"]) for k, v in results["backbone_ablation"].items()],
            key=lambda x: -x[1],
        )
        results["backbone_ablation"]["ranking"] = [
            {"model": k, "f1": f1} for k, f1 in ranked
        ]

    # 3. Ensemble strategy ablation
    if "ensemble" in ablation_types or "all" in ablation_types:
        if verbose:
            print("\n[3] Ensemble strategy ablation...")

        strategies = {
            "average": 0.90,
            "weighted": 0.92,
            "voting": 0.89,
            "max": 0.87,
            "median": 0.88,
        }

        results["ensemble_ablation"] = {}
        for strategy, target_acc in strategies.items():
            metrics = _simulate_evaluation(test_labels, accuracy=target_acc)
            results["ensemble_ablation"][strategy] = metrics

            if verbose:
                print(f"  {strategy}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")

        results["ensemble_ablation"]["best_strategy"] = "weighted"

    # 4. Calibration ablation
    if "calibration" in ablation_types or "all" in ablation_types:
        if verbose:
            print("\n[4] Calibration method ablation...")

        results["calibration_ablation"] = {
            "with_calibration": _simulate_evaluation(test_labels, accuracy=0.92),
            "without_calibration": _simulate_evaluation(test_labels, accuracy=0.91),
        }
        results["calibration_ablation"]["with_calibration"]["ece"] = 0.031
        results["calibration_ablation"]["without_calibration"]["ece"] = 0.078

        for method, ece_target in [("temperature", 0.035), ("platt", 0.042), ("isotonic", 0.028)]:
            metrics = _simulate_evaluation(test_labels, accuracy=0.91)
            metrics["ece"] = ece_target
            metrics["ece_improvement"] = 0.078 - ece_target
            results["calibration_ablation"][method] = metrics

            if verbose:
                print(f"  {method}: ECE={ece_target:.4f}")

    # 5. Signal analysis ablation
    if "components" in ablation_types or "all" in ablation_types:
        if verbose:
            print("\n[5] Signal analysis ablation...")

        results["signal_analysis_ablation"] = {
            "full": _simulate_evaluation(test_labels, accuracy=0.92),
            "ml_only": _simulate_evaluation(test_labels, accuracy=0.91),
            "no_frequency": _simulate_evaluation(test_labels, accuracy=0.915),
            "no_noise": _simulate_evaluation(test_labels, accuracy=0.918),
        }
        results["signal_analysis_ablation"]["signal_contribution"] = 0.01

        if verbose:
            for name, metrics in results["signal_analysis_ablation"].items():
                if isinstance(metrics, dict):
                    print(f"  {name}: Acc={metrics.get('accuracy', 0):.4f}")

    # 6. Compute component importance
    if verbose:
        print("\n[6] Computing component importance...")

    results["component_importance"] = {
        "calibration": 0.047,
        "signal_analysis": 0.01,
        "backbone_selection": 0.02,
        "ensemble_strategy": 0.015,
        "ensemble_benefit": 0.03,
    }

    if verbose:
        print("\nComponent Importance:")
        for component, score in sorted(results["component_importance"].items(), key=lambda x: -x[1]):
            print(f"  {component}: {score:.4f}")

    return results


def _simulate_evaluation(
    labels: np.ndarray,
    accuracy: float = 0.90,
    noise: float = 0.01,
) -> Dict[str, float]:
    """Simulate evaluation metrics for demonstration."""
    np.random.seed(42)

    # Add small noise to make results more realistic
    accuracy = min(1.0, max(0.5, accuracy + np.random.randn() * noise))
    auc = min(1.0, max(0.5, accuracy + 0.02 + np.random.randn() * noise))
    f1 = min(1.0, max(0.5, accuracy - 0.01 + np.random.randn() * noise))
    precision = min(1.0, max(0.5, accuracy + np.random.randn() * noise))
    recall = min(1.0, max(0.5, accuracy + np.random.randn() * noise))

    return {
        "accuracy": accuracy,
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "ece": max(0.01, 0.05 + np.random.randn() * 0.02),
    }


def _compute_importance(results: Dict[str, Any]) -> Dict[str, float]:
    """Compute component importance from ablation results."""
    baseline_acc = results["baseline"]["accuracy"]
    importance = {}

    components = results.get("ablations", {}).get("components", {})
    for name, metrics in components.items():
        if name != "full_model":
            # Importance = how much accuracy drops when removed
            importance[name.replace("without_", "")] = baseline_acc - metrics["accuracy"]

    return importance


def generate_ablation_tables(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate LaTeX tables from ablation results."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table: Component Ablation
    _generate_component_table(results, tables_dir)

    # Table: Ensemble Strategy Comparison
    _generate_ensemble_table(results, tables_dir)

    # Table: Calibration Comparison
    _generate_calibration_ablation_table(results, tables_dir)


def _generate_component_table(results: Dict[str, Any], tables_dir: Path) -> None:
    """Generate component ablation table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Component ablation study. Each row shows performance when",
        r"that component is removed. $\Delta$ shows accuracy drop from full model.}",
        r"\label{tab:ablation_components}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Configuration & Accuracy & AUC & $\Delta$ Acc \\",
        r"\midrule",
    ]

    baseline = results.get("baseline", {})
    baseline_acc = baseline.get("accuracy", 0)
    lines.append(
        f"Full ImageTrust & \\textbf{{{baseline_acc:.3f}}} & "
        f"\\textbf{{{baseline.get('auc', 0):.3f}}} & -- \\\\"
    )
    lines.append(r"\midrule")

    components = results.get("ablations", {}).get("components", {})
    for name, metrics in components.items():
        if name == "full_model":
            continue

        display_name = name.replace("without_", "-- ").replace("_", " ").title()
        delta = baseline_acc - metrics.get("accuracy", 0)
        delta_str = f"-{delta:.3f}" if delta > 0 else f"+{abs(delta):.3f}"

        lines.append(
            f"{display_name} & {metrics.get('accuracy', 0):.3f} & "
            f"{metrics.get('auc', 0):.3f} & {delta_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(tables_dir / "table_ablation_components.tex", "w") as f:
        f.write("\n".join(lines))


def _generate_ensemble_table(results: Dict[str, Any], tables_dir: Path) -> None:
    """Generate ensemble strategy comparison table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of ensemble strategies.}",
        r"\label{tab:ablation_ensemble}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Strategy & Accuracy & AUC & F1 \\",
        r"\midrule",
    ]

    ensemble = results.get("ablations", {}).get("ensemble", {})

    # Find best for bolding
    best_acc = max(m.get("accuracy", 0) for m in ensemble.values()) if ensemble else 0

    for strategy, metrics in ensemble.items():
        acc = metrics.get("accuracy", 0)
        acc_str = f"\\textbf{{{acc:.3f}}}" if abs(acc - best_acc) < 0.001 else f"{acc:.3f}"

        lines.append(
            f"{strategy.title()} & {acc_str} & "
            f"{metrics.get('auc', 0):.3f} & {metrics.get('f1', 0):.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(tables_dir / "table_ablation_ensemble.tex", "w") as f:
        f.write("\n".join(lines))


def _generate_calibration_ablation_table(results: Dict[str, Any], tables_dir: Path) -> None:
    """Generate calibration ablation table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Effect of calibration method on accuracy and ECE.}",
        r"\label{tab:ablation_calibration}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Calibration & Accuracy & ECE ($\downarrow$) \\",
        r"\midrule",
    ]

    calibration = results.get("ablations", {}).get("calibration", {})

    # Find best ECE for bolding
    best_ece = min(m.get("ece", 1) for m in calibration.values()) if calibration else 0

    for method, metrics in calibration.items():
        ece = metrics.get("ece", 0)
        ece_str = f"\\textbf{{{ece:.3f}}}" if abs(ece - best_ece) < 0.001 else f"{ece:.3f}"

        lines.append(
            f"{method.title()} & {metrics.get('accuracy', 0):.3f} & {ece_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(tables_dir / "table_ablation_calibration.tex", "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study for ImageTrust",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with real data
    python scripts/run_ablation.py --dataset ./data/test --output ./outputs/ablation

    # Run with simulated data (demo)
    python scripts/run_ablation.py --demo --output ./outputs/ablation_demo

    # Run specific ablation types
    python scripts/run_ablation.py --dataset ./data/test --ablation-type backbone ensemble

    # Generate tables from existing results
    python scripts/run_ablation.py --results-file ./outputs/ablation/ablation_results.json --generate-tables
        """,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        help="Directory containing test images (real/ and ai_generated/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ablation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--ablation-type",
        nargs="+",
        choices=["all", "backbone", "ensemble", "calibration", "components"],
        default=["all"],
        help="Types of ablation to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to use (for quick testing)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with simulated data (for demo/testing)",
    )
    parser.add_argument(
        "--generate-tables",
        action="store_true",
        help="Generate LaTeX tables from results",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help="Load results from file instead of running ablation",
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

    # Load or run ablation
    if args.results_file and args.results_file.exists():
        with open(args.results_file) as f:
            results = json.load(f)
    elif args.demo:
        # Run with simulated data
        results = run_ablation_study_simulated(
            splits_dir=Path("data/splits"),
            output_dir=args.output_dir,
            ablation_types=args.ablation_type,
            verbose=verbose,
        )

        # Save results
        results_path = args.output_dir / "ablation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if verbose:
            print(f"\nResults saved to: {results_path}")
    elif args.dataset:
        # Run with real data
        results = run_ablation_study(
            dataset_dir=args.dataset,
            output_dir=args.output_dir,
            ablation_types=args.ablation_type,
            device=args.device,
            max_samples=args.max_samples,
            verbose=verbose,
        )
    else:
        print("Error: Either --dataset or --demo must be specified")
        print("Use --help for usage examples")
        return 1

    # Generate tables
    if args.generate_tables:
        if verbose:
            print("\nGenerating LaTeX tables...")
        generate_ablation_tables(results, args.output_dir)
        if verbose:
            print(f"Tables saved to: {args.output_dir / 'tables'}")

    if verbose:
        print("\n" + "=" * 60)
        print("ABLATION STUDY COMPLETE")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
