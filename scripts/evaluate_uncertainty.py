#!/usr/bin/env python
"""
Evaluate uncertainty estimation and selective prediction (abstain) for thesis.

Generates:
- Coverage vs Accuracy trade-off curves
- AURC (Area Under Risk-Coverage Curve) metrics
- Analysis of abstained samples
- Figures for paper (Figure 5)

Usage:
    python scripts/evaluate_uncertainty.py --dataset ./data/test --output ./outputs/uncertainty
    python scripts/evaluate_uncertainty.py --dataset ./data/test --thresholds 0.3,0.4,0.5,0.6,0.7
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)


def load_dataset(dataset_path: Path) -> List[Tuple[Any, int]]:
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
                    logger.warning(f"Failed to load {img_path}: {e}")

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
                        logger.warning(f"Failed to load {img_path}: {e}")

    logger.info(f"Loaded {len(dataset)} images from {dataset_path}")
    return dataset


def get_predictions(
    detector,
    dataset: List[Tuple[Any, int]],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get predictions from detector."""
    probabilities = []
    labels = []

    for i, (img, label) in enumerate(dataset):
        try:
            result = detector.analyze(img, return_uncertainty=False)
            probabilities.append(result["ai_probability"])
            labels.append(label)

            if verbose and (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(dataset)} images")
        except Exception as e:
            logger.warning(f"Failed to process image: {e}")

    return np.array(probabilities), np.array(labels)


def evaluate_selective_prediction(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float],
    uncertainty_method: str = "entropy",
) -> Dict[str, Any]:
    """
    Evaluate selective prediction at different thresholds.

    Returns coverage, accuracy, and confusion matrix at each threshold.
    """
    from imagetrust.baselines.uncertainty import (
        UncertaintyEstimator,
        compute_risk_coverage_auc,
        analyze_abstain_characteristics,
    )

    results = {
        "thresholds": thresholds,
        "method": uncertainty_method,
        "per_threshold": [],
    }

    # Compute AURC
    aurc = compute_risk_coverage_auc(probabilities, labels, uncertainty_method)
    results["aurc"] = float(aurc)

    # Evaluate at each threshold
    for threshold in thresholds:
        estimator = UncertaintyEstimator(
            method=uncertainty_method,
            abstain_threshold=threshold,
        )

        # Get selective prediction metrics
        eval_result = estimator.evaluate_selective_prediction(
            probabilities, labels, threshold
        )

        # Get abstain characteristics
        abstain_analysis = analyze_abstain_characteristics(
            probabilities, labels, uncertainty_method, threshold
        )

        results["per_threshold"].append({
            "threshold": threshold,
            "coverage": eval_result.coverage,
            "accuracy_on_covered": eval_result.accuracy_on_covered,
            "abstain_rate": eval_result.abstain_rate,
            "true_positives": eval_result.true_positives,
            "true_negatives": eval_result.true_negatives,
            "false_positives": eval_result.false_positives,
            "false_negatives": eval_result.false_negatives,
            "abstained_count": eval_result.abstained_count,
            "error_rejection_rate": abstain_analysis["error_rejection_rate"],
        })

    return results


def compute_coverage_accuracy_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    uncertainty_method: str = "entropy",
    n_points: int = 50,
) -> Dict[str, List[float]]:
    """Compute full coverage-accuracy curve."""
    from imagetrust.baselines.uncertainty import UncertaintyEstimator

    estimator = UncertaintyEstimator(method=uncertainty_method)
    return estimator.compute_coverage_accuracy_curve(probabilities, labels, n_points)


def compare_uncertainty_methods(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Compare all uncertainty estimation methods."""
    from imagetrust.baselines.uncertainty import (
        UncertaintyMethod,
        compute_risk_coverage_auc,
    )

    methods = ["entropy", "margin", "confidence", "ensemble"]
    results = {}

    for method in methods:
        try:
            aurc = compute_risk_coverage_auc(probabilities, labels, method)
            curve = compute_coverage_accuracy_curve(probabilities, labels, method)

            results[method] = {
                "aurc": float(aurc),
                "coverage_accuracy_curve": curve,
            }
        except Exception as e:
            logger.warning(f"Method {method} failed: {e}")
            results[method] = {"error": str(e)}

    return results


def generate_coverage_accuracy_figure(
    curve_data: Dict[str, List[float]],
    output_path: Path,
    title: str = "Coverage vs Accuracy Trade-off",
) -> None:
    """Generate Figure 5: Coverage-accuracy trade-off curve."""
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.size": 10,
            "font.family": "serif",
            "figure.figsize": (6, 4),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        })

        fig, ax = plt.subplots()

        coverages = curve_data["coverage"]
        accuracies = curve_data["accuracy"]

        ax.plot(coverages, accuracies, "b-", linewidth=2)
        ax.fill_between(coverages, accuracies, alpha=0.2)

        # Add diagonal reference
        ax.plot([0, 1], [0.5, 0.5], "k--", alpha=0.3, label="Random")

        ax.set_xlabel("Coverage (1 - Abstain Rate)")
        ax.set_ylabel("Accuracy on Covered Samples")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.savefig(output_path)
        plt.close()

        logger.info(f"Saved figure to {output_path}")
    except ImportError:
        logger.warning("matplotlib not available for plotting")


def generate_method_comparison_figure(
    method_results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate comparison of uncertainty methods."""
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.size": 10,
            "font.family": "serif",
            "figure.figsize": (8, 5),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        })

        fig, ax = plt.subplots()

        colors = ["b", "r", "g", "orange"]
        labels = {
            "entropy": "Entropy",
            "margin": "Margin",
            "confidence": "Confidence",
            "ensemble": "Ensemble Variance",
        }

        for i, (method, data) in enumerate(method_results.items()):
            if "error" in data:
                continue

            curve = data.get("coverage_accuracy_curve", {})
            coverages = curve.get("coverage", [])
            accuracies = curve.get("accuracy", [])
            aurc = data.get("aurc", 0)

            if coverages and accuracies:
                ax.plot(
                    coverages,
                    accuracies,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    label=f"{labels.get(method, method)} (AURC={aurc:.4f})",
                )

        ax.set_xlabel("Coverage (1 - Abstain Rate)")
        ax.set_ylabel("Accuracy on Covered Samples")
        ax.set_title("Comparison of Uncertainty Methods")
        ax.set_xlim(0, 1)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.savefig(output_path)
        plt.close()

        logger.info(f"Saved comparison figure to {output_path}")
    except ImportError:
        logger.warning("matplotlib not available for plotting")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty estimation for AI image detection"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/uncertainty",
        help="Output directory for results",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.3,0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated list of abstain thresholds",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="entropy",
        choices=["entropy", "margin", "confidence", "ensemble"],
        help="Primary uncertainty method",
    )
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Compare all uncertainty methods",
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
        help="Maximum samples to evaluate (for testing)",
    )

    args = parser.parse_args()

    # Parse thresholds
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    # Setup output
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    # Load dataset
    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    if not dataset:
        logger.error("No images found in dataset")
        return

    logger.info(f"Loaded {len(dataset)} images")

    # Initialize detector
    logger.info("Initializing detector...")
    from imagetrust.detection.multi_detector import ComprehensiveDetector

    detector = ComprehensiveDetector(device=args.device)

    # Get predictions
    logger.info("Getting predictions...")
    probabilities, labels = get_predictions(detector, dataset)

    logger.info(f"Got {len(probabilities)} predictions")

    # Evaluate selective prediction
    logger.info("Evaluating selective prediction...")
    selective_results = evaluate_selective_prediction(
        probabilities, labels, thresholds, args.method
    )

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "num_samples": len(dataset),
        "primary_method": args.method,
        "selective_prediction": selective_results,
    }

    # Compare methods if requested
    if args.compare_methods:
        logger.info("Comparing uncertainty methods...")
        method_comparison = compare_uncertainty_methods(probabilities, labels)
        results["method_comparison"] = method_comparison

        # Generate comparison figure
        generate_method_comparison_figure(
            method_comparison,
            output_dir / "uncertainty_method_comparison.pdf",
        )

    # Generate coverage-accuracy curve
    logger.info("Computing coverage-accuracy curve...")
    curve_data = compute_coverage_accuracy_curve(
        probabilities, labels, args.method
    )
    results["coverage_accuracy_curve"] = curve_data

    # Generate figure
    generate_coverage_accuracy_figure(
        curve_data,
        output_dir / "coverage_accuracy_curve.pdf",
        title=f"Coverage vs Accuracy ({args.method.capitalize()} Uncertainty)",
    )

    # Save JSON results
    results_file = output_dir / "uncertainty_evaluation.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("UNCERTAINTY EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nDataset: {dataset_path}")
    print(f"Samples: {len(dataset)}")
    print(f"AURC ({args.method}): {selective_results['aurc']:.4f}")
    print("\nSelective Prediction Results:")
    print("-" * 60)
    print(f"{'Threshold':<12} {'Coverage':<12} {'Accuracy':<12} {'Error Rej.':<12}")
    print("-" * 60)
    for t in selective_results["per_threshold"]:
        print(
            f"{t['threshold']:<12.2f} "
            f"{t['coverage']:<12.1%} "
            f"{t['accuracy_on_covered']:<12.1%} "
            f"{t['error_rejection_rate']:<12.1%}"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
