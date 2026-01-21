#!/usr/bin/env python3
"""
Unified Baseline Evaluation Script.

Runs all baselines (B1, B2, B3) with the same protocol for fair comparison.
Supports: in-domain split, cross-generator, degradation robustness.

Usage:
    python scripts/run_baselines.py --dataset ./data/test --baseline all
    python scripts/run_baselines.py --dataset ./data/test --baseline classical --train
    python scripts/run_baselines.py --dataset ./data/test --cross-generator
    python scripts/run_baselines.py --dataset ./data/test --degradation

Output:
    - results/baselines/{timestamp}/
        - metrics.json           # All metrics in JSON
        - metrics_table.csv      # Paper-ready CSV table
        - metrics_table.tex      # LaTeX table
        - predictions/           # Per-image predictions
        - checkpoints/           # Trained model checkpoints
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

from imagetrust.utils.logging import get_logger, setup_logging
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset_from_directory(
    dataset_path: Path,
    split_file: Optional[Path] = None,
) -> Tuple[List[Path], List[int], Dict[str, List[Path]]]:
    """
    Load dataset from directory structure.

    Expected structure:
        dataset/
            real/           -> label 0
            ai/ or fake/    -> label 1
        OR with generators:
        dataset/
            real/
            midjourney/
            dalle3/
            stable_diffusion/

    Args:
        dataset_path: Path to dataset root
        split_file: Optional JSON file with predefined splits

    Returns:
        (all_images, all_labels, generator_dict)
    """
    all_images = []
    all_labels = []
    generator_dict = {}  # generator_name -> list of paths

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    for subdir in sorted(dataset_path.iterdir()):
        if not subdir.is_dir():
            continue

        name = subdir.name.lower()

        # Determine label
        if name in ["real", "authentic", "genuine"]:
            label = 0
        else:
            label = 1  # All non-real are AI

        # Collect images
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            images.extend(subdir.glob(ext))
            images.extend(subdir.glob(ext.upper()))

        if images:
            all_images.extend(images)
            all_labels.extend([label] * len(images))
            generator_dict[name] = images
            logger.info(f"  {name}: {len(images)} images (label={label})")

    logger.info(f"Total: {len(all_images)} images")

    return all_images, all_labels, generator_dict


def create_train_val_test_split(
    images: List[Path],
    labels: List[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[List[Path], List[int]]]:
    """
    Create stratified train/val/test split.

    Returns:
        Dict with 'train', 'val', 'test' keys, each -> (images, labels)
    """
    from sklearn.model_selection import train_test_split

    # First split: train vs (val + test)
    train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
        images, labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
        temp_imgs, temp_lbls,
        train_size=relative_val,
        stratify=temp_lbls,
        random_state=seed,
    )

    return {
        "train": (train_imgs, train_lbls),
        "val": (val_imgs, val_lbls),
        "test": (test_imgs, test_lbls),
    }


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_baseline(
    baseline,
    images: List[Path],
    labels: List[int],
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a baseline on a dataset.

    Returns dict with all metrics, optionally including raw predictions.
    """
    from imagetrust.evaluation.metrics import compute_metrics, compute_calibration_metrics

    # Get predictions
    results = baseline.predict_proba_batch(images)

    probs = np.array([r.ai_probability for r in results])
    preds = (probs > 0.5).astype(int)
    labels_arr = np.array(labels)

    # Compute metrics
    metrics = compute_metrics(labels_arr, preds, probs)

    # Compute calibration metrics
    cal_metrics = compute_calibration_metrics(labels_arr, probs)
    metrics.update(cal_metrics)

    # Add timing info
    times = [r.processing_time_ms for r in results if r.processing_time_ms > 0]
    if times:
        metrics["avg_time_ms"] = np.mean(times)
        metrics["std_time_ms"] = np.std(times)

    # Optionally return raw predictions for figure generation
    if return_predictions:
        metrics["_predictions"] = {
            "probabilities": probs.tolist(),
            "labels": labels_arr.tolist(),
            "predictions": preds.tolist(),
        }

    return metrics


def evaluate_cross_generator(
    baseline,
    generator_dict: Dict[str, List[Path]],
    train_on: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Cross-generator evaluation.

    If train_on is specified, trains only on that generator and tests on others.
    Otherwise, evaluates on each generator separately (assuming pre-trained).
    """
    results = {}

    for gen_name, gen_images in generator_dict.items():
        # Determine labels (real=0, others=1)
        label = 0 if gen_name in ["real", "authentic", "genuine"] else 1
        labels = [label] * len(gen_images)

        metrics = evaluate_baseline(baseline, gen_images, labels)
        results[gen_name] = metrics
        logger.info(f"  {gen_name}: Acc={metrics['accuracy']:.2%}, AUC={metrics['roc_auc']:.3f}")

    return results


def evaluate_degradation(
    baseline,
    images: List[Path],
    labels: List[int],
    jpeg_qualities: List[int] = [95, 85, 70, 50],
    resize_factors: List[float] = [1.0, 0.75, 0.5],
    blur_radii: List[float] = [0, 0.5, 1.0, 2.0],
    noise_sigmas: List[float] = [0, 0.01, 0.03],
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate robustness to degradations.
    """
    from imagetrust.evaluation.degradation import DegradationEvaluator

    evaluator = DegradationEvaluator(verbose=True)
    evaluator.add_images(images, labels)

    # Run evaluation
    results = evaluator.evaluate(
        baseline,
        jpeg_qualities=jpeg_qualities,
        blur_radii=blur_radii,
        resize_factors=resize_factors,
        noise_levels=noise_sigmas,
    )

    return results


# =============================================================================
# Results Formatting
# =============================================================================

def format_results_table(
    results: Dict[str, Dict[str, Any]],
    metrics_to_show: List[str] = ["accuracy", "balanced_accuracy", "f1_score", "roc_auc", "ece"],
) -> str:
    """
    Format results as a table (CSV format).
    """
    lines = []

    # Header
    header = ["Baseline"] + [m.replace("_", " ").title() for m in metrics_to_show]
    lines.append(",".join(header))

    # Rows
    for baseline_name, metrics in results.items():
        row = [baseline_name]
        for m in metrics_to_show:
            val = metrics.get(m, 0)
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        lines.append(",".join(row))

    return "\n".join(lines)


def format_latex_table(
    results: Dict[str, Dict[str, Any]],
    metrics_to_show: List[str] = ["accuracy", "balanced_accuracy", "f1_score", "roc_auc", "ece"],
    caption: str = "Baseline Comparison Results",
    label: str = "tab:baselines",
) -> str:
    """
    Format results as LaTeX table.
    """
    # Column spec
    cols = "l" + "c" * len(metrics_to_show)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
    ]

    # Header
    header_cells = ["Method"] + [m.replace("_", " ").title() for m in metrics_to_show]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # Find best values for each metric (for bolding)
    best_values = {}
    for m in metrics_to_show:
        values = [results[b].get(m, 0) for b in results]
        if m == "ece":  # Lower is better
            best_values[m] = min(values)
        else:  # Higher is better
            best_values[m] = max(values)

    # Rows
    for baseline_name, metrics in results.items():
        row = [baseline_name]
        for m in metrics_to_show:
            val = metrics.get(m, 0)
            if isinstance(val, float):
                formatted = f"{val:.4f}"
                # Bold if best
                if abs(val - best_values[m]) < 1e-6:
                    formatted = rf"\textbf{{{formatted}}}"
                row.append(formatted)
            else:
                row.append(str(val))
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def save_results(
    output_dir: Path,
    results: Dict[str, Any],
    prefix: str = "metrics",
) -> None:
    """Save results in multiple formats."""
    ensure_dir(output_dir)

    # JSON (full data)
    json_path = output_dir / f"{prefix}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved JSON: {json_path}")

    # CSV table
    if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
        csv_path = output_dir / f"{prefix}_table.csv"
        csv_content = format_results_table(results)
        with open(csv_path, "w") as f:
            f.write(csv_content)
        logger.info(f"Saved CSV: {csv_path}")

        # LaTeX table
        tex_path = output_dir / f"{prefix}_table.tex"
        tex_content = format_latex_table(results)
        with open(tex_path, "w") as f:
            f.write(tex_content)
        logger.info(f"Saved LaTeX: {tex_path}")


# =============================================================================
# Main Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline evaluations for ImageTrust thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train and evaluate classical baseline
    python run_baselines.py --dataset ./data/test --baseline classical --train

    # Evaluate all baselines (pre-trained)
    python run_baselines.py --dataset ./data/test --baseline all

    # Cross-generator evaluation
    python run_baselines.py --dataset ./data/test --cross-generator

    # Degradation robustness
    python run_baselines.py --dataset ./data/test --degradation

    # Full evaluation suite
    python run_baselines.py --dataset ./data/test --baseline all --train --cross-generator --degradation
        """,
    )

    # Data arguments
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Path to dataset")
    parser.add_argument("--split-file", type=str, default=None, help="JSON file with splits")

    # Baseline selection
    parser.add_argument(
        "--baseline", "-b",
        type=str,
        default="all",
        choices=["all", "baselines", "classical", "cnn", "vit", "imagetrust", "ours"],
        help="Which baseline(s) to run. 'all' includes ImageTrust, 'baselines' excludes it.",
    )
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to baselines.yaml")

    # Actions
    parser.add_argument("--train", action="store_true", help="Train baselines (otherwise load)")
    parser.add_argument("--cross-generator", action="store_true", help="Run cross-generator eval")
    parser.add_argument("--degradation", action="store_true", help="Run degradation eval")

    # Output
    parser.add_argument("--output-dir", "-o", type=str, default="outputs/baselines", help="Output directory")
    parser.add_argument("--generate-figures", action="store_true", help="Generate paper figures after evaluation")
    parser.add_argument("--figure-format", type=str, default="pdf", choices=["pdf", "png", "svg"], help="Figure format")

    # Training params (override config)
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    setup_logging(level="INFO")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    ensure_dir(output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Load dataset
    logger.info(f"Loading dataset from: {args.dataset}")
    images, labels, generator_dict = load_dataset_from_directory(Path(args.dataset))

    if len(images) == 0:
        logger.error("No images found in dataset!")
        return

    # Create splits
    splits = create_train_val_test_split(images, labels, seed=args.seed)
    train_imgs, train_lbls = splits["train"]
    val_imgs, val_lbls = splits["val"]
    test_imgs, test_lbls = splits["test"]

    logger.info(f"Splits: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    # Determine which baselines to run
    if args.baseline == "all":
        baseline_names = ["classical", "cnn", "vit", "imagetrust"]
    elif args.baseline == "baselines":
        baseline_names = ["classical", "cnn", "vit"]  # Excludes ImageTrust
    elif args.baseline in ["imagetrust", "ours"]:
        baseline_names = ["imagetrust"]
    else:
        baseline_names = [args.baseline]

    # Load or create baselines
    from imagetrust.baselines import get_baseline
    from imagetrust.baselines.base import BaselineConfig

    all_results = {}
    cross_gen_results = {}
    degradation_results = {}

    for baseline_name in baseline_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {baseline_name.upper()}")
        logger.info(f"{'='*60}")

        # Create baseline with appropriate config
        if baseline_name == "classical":
            config = BaselineConfig(
                name="Classical (LogReg)",
                seed=args.seed,
                epochs=1,
                batch_size=args.batch_size or 32,
            )
            baseline = get_baseline(baseline_name, config, classifier="logistic_regression")

        elif baseline_name == "cnn":
            config = BaselineConfig(
                name="CNN (ResNet-50)",
                seed=args.seed,
                epochs=args.epochs or 10,
                batch_size=args.batch_size or 32,
                learning_rate=1e-4,
                weight_decay=1e-4,
            )
            baseline = get_baseline(baseline_name, config, backbone="resnet50")

        elif baseline_name == "vit":
            config = BaselineConfig(
                name="ViT-B/16",
                seed=args.seed,
                epochs=args.epochs or 10,
                batch_size=args.batch_size or 16,
                learning_rate=1e-5,
                weight_decay=1e-2,
            )
            baseline = get_baseline(baseline_name, config, architecture="vit")

        elif baseline_name == "imagetrust":
            # ImageTrust uses pretrained HF models - no training needed
            config = BaselineConfig(
                name="ImageTrust (Ours)",
                seed=args.seed,
            )
            baseline = get_baseline(baseline_name, config)
            logger.info("ImageTrust uses pretrained HuggingFace models (no training required)")

        else:
            logger.warning(f"Unknown baseline: {baseline_name}")
            continue

        # Train if requested (and baseline supports it)
        checkpoint_dir = output_dir / "checkpoints"
        ensure_dir(checkpoint_dir)
        checkpoint_path = checkpoint_dir / f"{baseline_name}.pth"

        # ImageTrust doesn't need training - it uses pretrained models
        if baseline_name == "imagetrust":
            # Just initialize (loads HF models on first prediction)
            pass

        elif args.train:
            logger.info(f"Training {baseline_name}...")
            history = baseline.fit(train_imgs, train_lbls, val_imgs, val_lbls)

            # Save checkpoint
            baseline.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Save training history
            history_path = output_dir / f"{baseline_name}_history.json"
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2, default=str)

        elif checkpoint_path.exists():
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            baseline.load(checkpoint_path)

        elif baseline_name != "imagetrust":
            logger.warning(f"No checkpoint found and --train not specified. Skipping {baseline_name}.")
            continue

        # Evaluate on test set
        logger.info(f"Evaluating {baseline_name} on test set...")
        metrics = evaluate_baseline(baseline, test_imgs, test_lbls, return_predictions=True)
        all_results[baseline.name] = metrics

        logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")
        logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  ECE: {metrics['ece']:.4f}")

        # Cross-generator evaluation
        if args.cross_generator:
            logger.info(f"Running cross-generator evaluation for {baseline_name}...")
            cross_gen = evaluate_cross_generator(baseline, generator_dict)
            cross_gen_results[baseline.name] = cross_gen

        # Degradation evaluation
        if args.degradation:
            logger.info(f"Running degradation evaluation for {baseline_name}...")
            deg_results = evaluate_degradation(baseline, test_imgs, test_lbls)
            degradation_results[baseline.name] = deg_results

    # Save all results
    logger.info(f"\n{'='*60}")
    logger.info("Saving results...")
    logger.info(f"{'='*60}")

    save_results(output_dir, all_results, prefix="main_results")

    if cross_gen_results:
        save_results(output_dir, cross_gen_results, prefix="cross_generator")

    if degradation_results:
        save_results(output_dir, degradation_results, prefix="degradation")

    # Save reproducibility info
    repro_info = {
        "timestamp": timestamp,
        "seed": args.seed,
        "dataset": str(args.dataset),
        "baselines": baseline_names,
        "train_samples": len(train_imgs),
        "val_samples": len(val_imgs),
        "test_samples": len(test_imgs),
        "args": vars(args),
    }
    with open(output_dir / "reproducibility.json", "w") as f:
        json.dump(repro_info, f, indent=2)

    # Generate figures if requested
    if args.generate_figures:
        logger.info(f"\n{'='*60}")
        logger.info("Generating paper figures...")
        logger.info(f"{'='*60}")

        try:
            from scripts.generate_figures import generate_all_figures

            figures_dir = output_dir / "figures"
            generate_all_figures(output_dir, figures_dir, args.figure_format)

        except ImportError:
            # Try alternative import
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from generate_figures import generate_all_figures

                figures_dir = output_dir / "figures"
                generate_all_figures(output_dir, figures_dir, args.figure_format)
            except Exception as e:
                logger.warning(f"Could not generate figures: {e}")
                logger.info("Run manually: python scripts/generate_figures.py --results " + str(output_dir))

    logger.info(f"\n✅ All evaluations complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
