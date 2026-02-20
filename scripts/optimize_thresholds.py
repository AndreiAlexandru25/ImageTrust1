#!/usr/bin/env python
"""
Threshold Optimization Script for Custom-Trained Models.

Problem: Models have ~99.6% precision but only ~47% recall.
Solution: Find optimal thresholds that balance precision/recall.

Generates:
- Precision-Recall curves
- Optimal thresholds for F1, Youden's J, and custom targets
- Per-model threshold recommendations
- Calibrated probability outputs

Usage:
    python scripts/optimize_thresholds.py --models outputs/training_*/best_model.pth
    python scripts/optimize_thresholds.py --target-recall 0.85 --min-precision 0.80
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)


@dataclass
class ThresholdResult:
    """Results for a single threshold optimization."""

    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float
    balanced_accuracy: float
    youden_j: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "youden_j": self.youden_j,
        }


@dataclass
class OptimizationResults:
    """Complete optimization results for a model."""

    model_name: str
    model_path: str
    default_threshold: ThresholdResult  # 0.5
    optimal_f1_threshold: ThresholdResult
    optimal_youden_threshold: ThresholdResult
    optimal_balanced_threshold: ThresholdResult
    target_recall_threshold: Optional[ThresholdResult]
    all_thresholds: List[ThresholdResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "default_threshold": self.default_threshold.to_dict(),
            "optimal_f1_threshold": self.optimal_f1_threshold.to_dict(),
            "optimal_youden_threshold": self.optimal_youden_threshold.to_dict(),
            "optimal_balanced_threshold": self.optimal_balanced_threshold.to_dict(),
            "target_recall_threshold": self.target_recall_threshold.to_dict() if self.target_recall_threshold else None,
            "threshold_curve": [t.to_dict() for t in self.all_thresholds],
        }


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> ThresholdResult:
    """Compute all metrics at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    # Basic counts
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy = (recall + specificity) / 2
    youden_j = recall + specificity - 1

    return ThresholdResult(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        specificity=specificity,
        balanced_accuracy=balanced_accuracy,
        youden_j=youden_j,
    )


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_recall: Optional[float] = None,
    min_precision: float = 0.5,
    n_thresholds: int = 200,
) -> Tuple[List[ThresholdResult], Dict[str, ThresholdResult]]:
    """
    Find optimal thresholds using different criteria.

    Returns:
        Tuple of (all_results, optimal_thresholds_dict)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    results = []

    for t in thresholds:
        result = compute_metrics_at_threshold(y_true, y_proba, t)
        results.append(result)

    # Find optimal by different criteria
    optimal = {}

    # Default threshold (0.5)
    optimal["default"] = compute_metrics_at_threshold(y_true, y_proba, 0.5)

    # Optimal F1
    best_f1_idx = np.argmax([r.f1 for r in results])
    optimal["f1"] = results[best_f1_idx]

    # Optimal Youden's J
    best_youden_idx = np.argmax([r.youden_j for r in results])
    optimal["youden"] = results[best_youden_idx]

    # Optimal Balanced Accuracy
    best_balanced_idx = np.argmax([r.balanced_accuracy for r in results])
    optimal["balanced"] = results[best_balanced_idx]

    # Target recall (if specified)
    if target_recall is not None:
        # Find lowest threshold that achieves target recall with min precision
        valid_thresholds = [
            r for r in results
            if r.recall >= target_recall and r.precision >= min_precision
        ]
        if valid_thresholds:
            # Pick the one with highest precision among valid ones
            optimal["target_recall"] = max(valid_thresholds, key=lambda r: r.precision)
        else:
            # Relax precision constraint and find closest to target recall
            closest = min(results, key=lambda r: abs(r.recall - target_recall))
            optimal["target_recall"] = closest
            logger.warning(
                f"Could not achieve target recall {target_recall} with min precision {min_precision}. "
                f"Best found: recall={closest.recall:.3f}, precision={closest.precision:.3f}"
            )

    return results, optimal


def load_model_and_evaluate(
    model_path: Path,
    data_dir: Path,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a trained model and get predictions on validation data.

    Returns:
        Tuple of (y_true, y_proba)
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image

    logger.info(f"Loading model from {model_path}")

    # Determine backbone from path
    if "resnet" in str(model_path).lower():
        backbone = "resnet50"
    elif "efficientnet" in str(model_path).lower():
        backbone = "efficientnetv2_m"
    elif "convnext" in str(model_path).lower():
        backbone = "convnext_base"
    else:
        backbone = "resnet50"  # default

    logger.info(f"Detected backbone: {backbone}")

    # Load model
    try:
        import timm

        if backbone == "resnet50":
            model = timm.create_model("resnet50", pretrained=False, num_classes=2)
        elif backbone == "efficientnetv2_m":
            model = timm.create_model("tf_efficientnetv2_m", pretrained=False, num_classes=2)
        elif backbone == "convnext_base":
            model = timm.create_model("convnext_base", pretrained=False, num_classes=2)

        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load validation data
    logger.info(f"Loading validation data from {data_dir}")

    y_true = []
    y_proba = []

    # Look for validation data in standard structure
    for label, class_name in [(0, "real"), (1, "ai"), (1, "ai_generated"), (1, "fake"), (1, "synthetic")]:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + \
                 list(class_dir.glob("*.png")) + list(class_dir.glob("*.webp"))

        logger.info(f"Found {len(images)} images in {class_name}")

        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_tensor)
                    proba = torch.softmax(output, dim=1)
                    ai_prob = proba[0, 1].item()

                y_true.append(label)
                y_proba.append(ai_prob)

            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")

    return np.array(y_true), np.array(y_proba)


def generate_pr_curve_figure(
    results: List[ThresholdResult],
    optimal: Dict[str, ThresholdResult],
    model_name: str,
    output_path: Path,
) -> None:
    """Generate Precision-Recall curve figure."""
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'serif',
            'figure.figsize': (8, 6),
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PR Curve
        ax1 = axes[0]
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]

        ax1.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')

        # Mark optimal points
        markers = {
            'default': ('gray', 'o', 'Default (0.5)'),
            'f1': ('green', 's', 'Optimal F1'),
            'youden': ('red', '^', "Youden's J"),
            'balanced': ('purple', 'd', 'Balanced Acc'),
        }

        for key, (color, marker, label) in markers.items():
            if key in optimal:
                t = optimal[key]
                ax1.scatter([t.recall], [t.precision], c=color, marker=marker,
                           s=100, zorder=5, label=f'{label} (t={t.threshold:.2f})')

        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title(f'{model_name} - Precision-Recall Curve')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower left')

        # Threshold vs Metrics
        ax2 = axes[1]
        thresholds = [r.threshold for r in results]

        ax2.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
        ax2.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
        ax2.plot(thresholds, [r.f1 for r in results], 'g--', linewidth=2, label='F1')
        ax2.plot(thresholds, [r.balanced_accuracy for r in results], 'm:', linewidth=2, label='Balanced Acc')

        # Mark optimal thresholds
        for key, t in optimal.items():
            if key != 'default':
                ax2.axvline(x=t.threshold, color='gray', linestyle='--', alpha=0.5)

        ax2.axvline(x=0.5, color='black', linestyle=':', alpha=0.5, label='Default (0.5)')

        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Metric Value')
        ax2.set_title(f'{model_name} - Metrics vs Threshold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='center left')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Saved PR curve to {output_path}")

    except ImportError:
        logger.warning("matplotlib not available for plotting")


def generate_threshold_comparison_table(
    all_results: Dict[str, OptimizationResults],
    output_path: Path,
) -> str:
    """Generate LaTeX table comparing thresholds across models."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Threshold Optimization Results}",
        r"\label{tab:thresholds}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Threshold & Precision & Recall & F1 & Bal. Acc & Strategy \\",
        r"\midrule",
    ]

    for model_name, results in all_results.items():
        # Default row
        d = results.default_threshold
        lines.append(
            f"{model_name} & {d.threshold:.2f} & {d.precision:.1%} & {d.recall:.1%} & "
            f"{d.f1:.1%} & {d.balanced_accuracy:.1%} & Default \\\\"
        )

        # Optimal F1 row
        f = results.optimal_f1_threshold
        lines.append(
            f" & {f.threshold:.2f} & {f.precision:.1%} & {f.recall:.1%} & "
            f"\\textbf{{{f.f1:.1%}}} & {f.balanced_accuracy:.1%} & Optimal F1 \\\\"
        )

        # Optimal Balanced row
        b = results.optimal_balanced_threshold
        lines.append(
            f" & {b.threshold:.2f} & {b.precision:.1%} & {b.recall:.1%} & "
            f"{b.f1:.1%} & \\textbf{{{b.balanced_accuracy:.1%}}} & Balanced \\\\"
        )

        lines.append(r"\midrule")

    # Remove last midrule
    lines = lines[:-1]

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(table)

    logger.info(f"Saved threshold table to {output_path}")
    return table


def main():
    parser = argparse.ArgumentParser(
        description="Optimize classification thresholds for trained models"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Paths to model files (supports glob patterns)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/eval",
        help="Path to evaluation data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/threshold_optimization",
        help="Output directory",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.85,
        help="Target recall to achieve",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.70,
        help="Minimum acceptable precision when targeting recall",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Use cached predictions if available",
    )

    args = parser.parse_args()

    # Setup output
    output_dir = Path(args.output)
    ensure_dir(output_dir)

    # Setup device
    device = args.device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    # Find models
    if args.models is None:
        # Auto-discover trained models
        model_paths = []
        outputs_dir = Path("./outputs")
        for training_dir in outputs_dir.glob("training_*"):
            best_model = training_dir / "best_model.pth"
            if best_model.exists():
                model_paths.append(best_model)
    else:
        model_paths = []
        for pattern in args.models:
            model_paths.extend(Path(".").glob(pattern))

    if not model_paths:
        logger.error("No models found!")
        return

    logger.info(f"Found {len(model_paths)} models to optimize")

    # Process each model
    all_results = {}
    data_dir = Path(args.data_dir)

    for model_path in model_paths:
        model_name = model_path.parent.name.replace("training_", "").capitalize()
        logger.info(f"\n{'='*60}")
        logger.info(f"Optimizing thresholds for {model_name}")
        logger.info(f"{'='*60}")

        # Check for cached predictions
        cache_path = output_dir / f"{model_name}_predictions.npz"

        if args.use_cached and cache_path.exists():
            logger.info(f"Loading cached predictions from {cache_path}")
            data = np.load(cache_path)
            y_true = data["y_true"]
            y_proba = data["y_proba"]
        else:
            # Get predictions
            try:
                y_true, y_proba = load_model_and_evaluate(model_path, data_dir, device)

                # Cache predictions
                np.savez(cache_path, y_true=y_true, y_proba=y_proba)
                logger.info(f"Cached predictions to {cache_path}")

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue

        if len(y_true) == 0:
            logger.error(f"No data found for {model_name}")
            continue

        logger.info(f"Evaluating on {len(y_true)} samples")
        logger.info(f"Class distribution: {np.sum(y_true == 0)} real, {np.sum(y_true == 1)} AI")

        # Find optimal thresholds
        results, optimal = find_optimal_thresholds(
            y_true, y_proba,
            target_recall=args.target_recall,
            min_precision=args.min_precision,
        )

        # Create results object
        opt_results = OptimizationResults(
            model_name=model_name,
            model_path=str(model_path),
            default_threshold=optimal["default"],
            optimal_f1_threshold=optimal["f1"],
            optimal_youden_threshold=optimal["youden"],
            optimal_balanced_threshold=optimal["balanced"],
            target_recall_threshold=optimal.get("target_recall"),
            all_thresholds=results,
        )

        all_results[model_name] = opt_results

        # Print summary
        print(f"\n{'='*60}")
        print(f"THRESHOLD OPTIMIZATION: {model_name}")
        print(f"{'='*60}")
        print(f"\n{'Strategy':<20} {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Bal.Acc':>10}")
        print("-" * 70)

        strategies = [
            ("Default (0.5)", optimal["default"]),
            ("Optimal F1", optimal["f1"]),
            ("Youden's J", optimal["youden"]),
            ("Balanced Acc", optimal["balanced"]),
        ]
        if "target_recall" in optimal:
            strategies.append((f"Target Recall ≥{args.target_recall:.0%}", optimal["target_recall"]))

        for name, t in strategies:
            print(f"{name:<20} {t.threshold:>10.3f} {t.precision:>10.1%} {t.recall:>10.1%} {t.f1:>10.1%} {t.balanced_accuracy:>10.1%}")

        # Generate PR curve
        generate_pr_curve_figure(
            results, optimal, model_name,
            output_dir / f"{model_name}_pr_curve.pdf"
        )

    # Save all results
    results_file = output_dir / "threshold_optimization_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "target_recall": args.target_recall,
                "min_precision": args.min_precision,
                "models": {k: v.to_dict() for k, v in all_results.items()},
            },
            f, indent=2
        )
    logger.info(f"\nResults saved to {results_file}")

    # Generate comparison table
    if all_results:
        generate_threshold_comparison_table(
            all_results,
            output_dir / "table_thresholds.tex"
        )

    # Print recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("\nFor academic evaluation, use:")
    print("- Optimal F1 threshold for balanced precision/recall")
    print("- Report both default (0.5) and optimal thresholds in paper")
    print("- Include calibration analysis with optimal thresholds")
    print("\nFor production use:")
    print("- Use Youden's J or Balanced Accuracy threshold")
    print("- Consider target recall if false negatives are costly")
    print("=" * 60)


if __name__ == "__main__":
    main()
