#!/usr/bin/env python
"""
Evaluate Novel Contributions for Academic Paper.

Generates Tables 7 and 8:
- Table 7: Screenshot Detection Performance
- Table 8: Social Media Platform Detection Performance

These are the original contributions of the thesis.

Usage:
    python scripts/evaluate_novel_contributions.py --screenshot-dataset data/screenshots --social-dataset data/social_media
    python scripts/evaluate_novel_contributions.py --output outputs/novel_contributions
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)


def load_images_from_dir(
    dir_path: Path,
    max_samples: Optional[int] = None,
) -> List[Tuple[Image.Image, str]]:
    """Load images from directory."""
    images = []
    extensions = [".jpg", ".jpeg", ".png", ".webp"]

    for ext in extensions:
        for img_path in dir_path.glob(f"*{ext}"):
            try:
                img = Image.open(img_path).convert("RGB")
                images.append((img, img_path.name))
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")

            if max_samples and len(images) >= max_samples:
                return images

    return images


def evaluate_screenshot_detection(
    screenshot_dir: Path,
    original_dir: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate Screenshot Detection module.

    Args:
        screenshot_dir: Directory with screenshot images (label=1)
        original_dir: Directory with original images (label=0)
        max_samples: Maximum samples per class

    Returns:
        Evaluation results for Table 7
    """
    from imagetrust.forensics.source_detection import ScreenshotDetector

    detector = ScreenshotDetector()

    y_true = []
    y_pred = []
    y_scores = []

    # Load screenshots (positive class)
    logger.info(f"Loading screenshots from {screenshot_dir}")
    screenshots = load_images_from_dir(screenshot_dir, max_samples)
    logger.info(f"Loaded {len(screenshots)} screenshots")

    for img, name in screenshots:
        try:
            result = detector.analyze(img)
            score = result.score if hasattr(result, 'score') else result.get('score', 0)
            detected = result.detected if hasattr(result, 'detected') else result.get('detected', False)

            y_true.append(1)
            y_pred.append(1 if detected else 0)
            y_scores.append(score)
        except Exception as e:
            logger.warning(f"Failed to analyze {name}: {e}")

    # Load originals (negative class)
    logger.info(f"Loading originals from {original_dir}")
    originals = load_images_from_dir(original_dir, max_samples)
    logger.info(f"Loaded {len(originals)} originals")

    for img, name in originals:
        try:
            result = detector.analyze(img)
            score = result.score if hasattr(result, 'score') else result.get('score', 0)
            detected = result.detected if hasattr(result, 'detected') else result.get('detected', False)

            y_true.append(0)
            y_pred.append(1 if detected else 0)
            y_scores.append(score)
        except Exception as e:
            logger.warning(f"Failed to analyze {name}: {e}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Compute metrics
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(1 - y_true, 1 - y_pred, zero_division=0)),
    }

    # AUC if we have both classes
    if len(np.unique(y_true)) > 1:
        results["roc_auc"] = float(roc_auc_score(y_true, y_scores))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        results["confusion_matrix"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

    results["num_screenshots"] = int(np.sum(y_true == 1))
    results["num_originals"] = int(np.sum(y_true == 0))

    return results


def evaluate_platform_detection(
    social_media_dir: Path,
    original_dir: Path,
    platforms: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate Platform Detection module.

    Args:
        social_media_dir: Directory with platform subdirectories
        original_dir: Directory with original images
        platforms: Platforms to evaluate (None = all found)
        max_samples: Maximum samples per platform

    Returns:
        Evaluation results for Table 8
    """
    from imagetrust.forensics.source_detection import PlatformDetector

    detector = PlatformDetector()

    # Find all platform subdirectories
    if platforms is None:
        platforms = [d.name for d in social_media_dir.iterdir()
                    if d.is_dir() and d.name != "original"]

    logger.info(f"Evaluating platforms: {platforms}")

    results = {
        "per_platform": {},
        "overall": {},
    }

    all_y_true = []  # Platform labels
    all_y_pred = []  # Predicted platforms
    all_detected = []  # Whether any platform was detected
    all_true_detected = []  # True label is social media (1) or original (0)

    # Evaluate each platform
    for platform in platforms:
        platform_dir = social_media_dir / platform
        if not platform_dir.exists():
            logger.warning(f"Platform directory not found: {platform_dir}")
            continue

        logger.info(f"Evaluating {platform}")
        images = load_images_from_dir(platform_dir, max_samples)

        correct = 0
        total = 0
        detected_any = 0

        for img, name in images:
            try:
                result = detector.analyze(img)
                details = result.details if hasattr(result, 'details') else result.get('details', {})
                detected = result.detected if hasattr(result, 'detected') else result.get('detected', False)
                best_platform = details.get('best_platform', '')

                all_y_true.append(platform)
                all_y_pred.append(best_platform if detected else 'original')
                all_detected.append(1 if detected else 0)
                all_true_detected.append(1)  # These are all social media

                if detected:
                    detected_any += 1
                    if best_platform == platform:
                        correct += 1

                total += 1

            except Exception as e:
                logger.warning(f"Failed to analyze {name}: {e}")

        if total > 0:
            results["per_platform"][platform] = {
                "correct_platform": correct,
                "detected_any": detected_any,
                "total": total,
                "platform_accuracy": correct / total,
                "detection_rate": detected_any / total,
            }

    # Evaluate originals (should NOT be detected as social media)
    logger.info("Evaluating originals (should not be detected)")
    originals = load_images_from_dir(original_dir, max_samples)

    false_detections = 0
    total_originals = 0

    for img, name in originals:
        try:
            result = detector.analyze(img)
            detected = result.detected if hasattr(result, 'detected') else result.get('detected', False)
            details = result.details if hasattr(result, 'details') else result.get('details', {})
            best_platform = details.get('best_platform', '')

            all_y_true.append('original')
            all_y_pred.append(best_platform if detected else 'original')
            all_detected.append(1 if detected else 0)
            all_true_detected.append(0)  # These are originals

            if detected:
                false_detections += 1

            total_originals += 1

        except Exception as e:
            logger.warning(f"Failed to analyze {name}: {e}")

    results["originals"] = {
        "false_detections": false_detections,
        "total": total_originals,
        "false_positive_rate": false_detections / total_originals if total_originals > 0 else 0,
    }

    # Overall binary detection metrics (social media vs original)
    all_detected = np.array(all_detected)
    all_true_detected = np.array(all_true_detected)

    results["overall"]["binary_detection"] = {
        "accuracy": float(accuracy_score(all_true_detected, all_detected)),
        "precision": float(precision_score(all_true_detected, all_detected, zero_division=0)),
        "recall": float(recall_score(all_true_detected, all_detected, zero_division=0)),
        "f1_score": float(f1_score(all_true_detected, all_detected, zero_division=0)),
    }

    # Multi-class platform accuracy
    correct_platform_count = sum(
        1 for t, p in zip(all_y_true, all_y_pred) if t == p
    )
    results["overall"]["platform_accuracy"] = correct_platform_count / len(all_y_true) if all_y_true else 0

    return results


def generate_table_7_latex(results: Dict[str, Any], output_path: Path) -> str:
    """Generate Table 7: Screenshot Detection Results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Screenshot Detection Performance}",
        r"\label{tab:screenshot}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Accuracy & {results['accuracy']:.1%} \\\\",
        f"Precision & {results['precision']:.1%} \\\\",
        f"Recall (Sensitivity) & {results['recall']:.1%} \\\\",
        f"Specificity & {results['specificity']:.1%} \\\\",
        f"F1-Score & {results['f1_score']:.1%} \\\\",
    ]

    if "roc_auc" in results:
        lines.append(f"ROC-AUC & {results['roc_auc']:.3f} \\\\")

    lines.extend([
        r"\midrule",
        f"Screenshots (Positive) & {results['num_screenshots']} \\\\",
        f"Originals (Negative) & {results['num_originals']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(table)

    return table


def generate_table_8_latex(results: Dict[str, Any], output_path: Path) -> str:
    """Generate Table 8: Social Media Platform Detection Results."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Social Media Platform Detection Performance}",
        r"\label{tab:platform}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Platform & Detection Rate & Platform Acc. & Samples \\",
        r"\midrule",
    ]

    for platform, metrics in results.get("per_platform", {}).items():
        lines.append(
            f"{platform.replace('_', ' ').title()} & "
            f"{metrics['detection_rate']:.1%} & "
            f"{metrics['platform_accuracy']:.1%} & "
            f"{metrics['total']} \\\\"
        )

    # Originals row
    orig = results.get("originals", {})
    fpr = orig.get("false_positive_rate", 0)
    lines.append(
        f"Original (FPR) & {fpr:.1%} & - & {orig.get('total', 0)} \\\\"
    )

    lines.extend([
        r"\midrule",
    ])

    # Overall metrics
    overall = results.get("overall", {})
    binary = overall.get("binary_detection", {})
    lines.extend([
        r"\multicolumn{4}{l}{\textit{Overall Binary Detection (Social Media vs Original):}} \\",
        f"\\quad Accuracy & \\multicolumn{{3}}{{c}}{{{binary.get('accuracy', 0):.1%}}} \\\\",
        f"\\quad Precision & \\multicolumn{{3}}{{c}}{{{binary.get('precision', 0):.1%}}} \\\\",
        f"\\quad Recall & \\multicolumn{{3}}{{c}}{{{binary.get('recall', 0):.1%}}} \\\\",
        f"\\quad F1-Score & \\multicolumn{{3}}{{c}}{{{binary.get('f1_score', 0):.1%}}} \\\\",
    ])

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(table)

    return table


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate novel contributions for academic paper"
    )
    parser.add_argument(
        "--screenshot-dataset",
        type=str,
        default=None,
        help="Path to screenshot dataset (with /screenshot and /original subdirs)",
    )
    parser.add_argument(
        "--social-dataset",
        type=str,
        default=None,
        help="Path to social media dataset (with platform subdirs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/novel_contributions",
        help="Output directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per category",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    ensure_dir(output_dir)

    all_results = {
        "timestamp": datetime.now().isoformat(),
    }

    # Evaluate Screenshot Detection
    if args.screenshot_dataset:
        screenshot_dir = Path(args.screenshot_dataset)
        logger.info("=" * 60)
        logger.info("EVALUATING SCREENSHOT DETECTION")
        logger.info("=" * 60)

        screenshot_results = evaluate_screenshot_detection(
            screenshot_dir=screenshot_dir / "screenshot",
            original_dir=screenshot_dir / "original",
            max_samples=args.max_samples,
        )

        all_results["screenshot_detection"] = screenshot_results

        # Generate Table 7
        table_7 = generate_table_7_latex(screenshot_results, output_dir / "table_7_screenshot.tex")

        print("\n" + "=" * 60)
        print("TABLE 7: SCREENSHOT DETECTION RESULTS")
        print("=" * 60)
        print(f"Accuracy:    {screenshot_results['accuracy']:.1%}")
        print(f"Precision:   {screenshot_results['precision']:.1%}")
        print(f"Recall:      {screenshot_results['recall']:.1%}")
        print(f"F1-Score:    {screenshot_results['f1_score']:.1%}")
        if "roc_auc" in screenshot_results:
            print(f"ROC-AUC:     {screenshot_results['roc_auc']:.3f}")
        print("=" * 60)

    # Evaluate Platform Detection
    if args.social_dataset:
        social_dir = Path(args.social_dataset)
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING PLATFORM DETECTION")
        logger.info("=" * 60)

        platform_results = evaluate_platform_detection(
            social_media_dir=social_dir,
            original_dir=social_dir / "original",
            max_samples=args.max_samples,
        )

        all_results["platform_detection"] = platform_results

        # Generate Table 8
        table_8 = generate_table_8_latex(platform_results, output_dir / "table_8_platform.tex")

        print("\n" + "=" * 60)
        print("TABLE 8: PLATFORM DETECTION RESULTS")
        print("=" * 60)
        print("\nPer-Platform:")
        for platform, metrics in platform_results.get("per_platform", {}).items():
            print(f"  {platform}: Detection={metrics['detection_rate']:.1%}, "
                  f"Platform Acc={metrics['platform_accuracy']:.1%}")

        overall = platform_results.get("overall", {}).get("binary_detection", {})
        print(f"\nOverall Binary Detection:")
        print(f"  Accuracy:  {overall.get('accuracy', 0):.1%}")
        print(f"  Precision: {overall.get('precision', 0):.1%}")
        print(f"  Recall:    {overall.get('recall', 0):.1%}")
        print(f"  F1-Score:  {overall.get('f1_score', 0):.1%}")
        print("=" * 60)

    # Save all results
    results_file = output_dir / "novel_contributions_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    if args.screenshot_dataset:
        logger.info(f"Table 7 saved to {output_dir / 'table_7_screenshot.tex'}")
    if args.social_dataset:
        logger.info(f"Table 8 saved to {output_dir / 'table_8_platform.tex'}")


if __name__ == "__main__":
    main()
