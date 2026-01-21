#!/usr/bin/env python3
"""
ImageTrust Quick Demo Script.

One-click demonstration of ImageTrust capabilities.
Creates synthetic test images and demonstrates all major features.

Usage:
    python scripts/quick_demo.py
    python scripts/quick_demo.py --feature detection
    python scripts/quick_demo.py --feature calibration
    python scripts/quick_demo.py --feature all

Author: ImageTrust Team
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Demo Image Generation
# =============================================================================

def create_demo_images(output_dir: Path, n_real: int = 5, n_ai: int = 5) -> dict:
    """Create synthetic demo images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    real_dir = output_dir / "real"
    ai_dir = output_dir / "ai"
    real_dir.mkdir(exist_ok=True)
    ai_dir.mkdir(exist_ok=True)

    np.random.seed(42)

    images = {"real": [], "ai": []}

    # Create "real-like" images (smooth gradients, natural patterns)
    for i in range(n_real):
        arr = np.zeros((256, 256, 3), dtype=np.uint8)

        # Create smooth gradient
        for y in range(256):
            for x in range(256):
                arr[y, x] = [
                    int(100 + 100 * np.sin(x / 30 + i)),
                    int(100 + 100 * np.sin(y / 30 + i)),
                    int(100 + 50 * np.sin((x + y) / 40)),
                ]

        # Add some natural noise
        noise = np.random.randint(-10, 10, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)
        path = real_dir / f"real_{i+1:03d}.jpg"
        img.save(path, "JPEG", quality=95)
        images["real"].append(path)

    # Create "AI-like" images (uniform regions, sharp transitions)
    for i in range(n_ai):
        arr = np.zeros((256, 256, 3), dtype=np.uint8)

        # Create blocky patterns (typical of AI)
        block_size = 32
        for by in range(0, 256, block_size):
            for bx in range(0, 256, block_size):
                color = np.random.randint(50, 200, 3)
                arr[by:by+block_size, bx:bx+block_size] = color

        # Add uniform noise (AI-like)
        noise = np.random.randint(-5, 5, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)
        path = ai_dir / f"ai_{i+1:03d}.jpg"
        img.save(path, "JPEG", quality=85)
        images["ai"].append(path)

    return images


# =============================================================================
# Feature Demos
# =============================================================================

def demo_basic_detection(images: dict):
    """Demonstrate basic AI detection."""
    print("\n" + "=" * 60)
    print("DEMO: Basic AI Detection")
    print("=" * 60)

    try:
        from imagetrust.detection import AIDetector

        detector = AIDetector()

        print("\nAnalyzing images...")
        print("-" * 50)

        all_paths = images["real"][:3] + images["ai"][:3]
        labels = ["Real"] * 3 + ["AI"] * 3

        for path, label in zip(all_paths, labels):
            result = detector.detect(str(path))
            prob = result["ai_probability"]
            verdict = result["verdict"].value

            # Visual bar
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))

            correct = (verdict == "ai_generated" and label == "AI") or \
                     (verdict == "real" and label == "Real")
            status = "✓" if correct else "✗"

            print(f"{status} {path.name:15s} [{bar}] {prob:5.1%} → {verdict:12s} (actual: {label})")

        print("\n✓ Basic detection demo complete!")

    except ImportError as e:
        print(f"\n⚠ Detection not available: {e}")
        print("  This demo requires the full ImageTrust installation.")
        _mock_detection_demo(images)


def _mock_detection_demo(images: dict):
    """Mock detection demo when models aren't available."""
    print("\n  [Mock Demo - No models loaded]")
    print("-" * 50)

    for category, paths in images.items():
        for path in paths[:2]:
            prob = 0.8 if category == "ai" else 0.2
            prob += np.random.uniform(-0.1, 0.1)
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            print(f"  {path.name:15s} [{bar}] {prob:5.1%}")


def demo_baseline_comparison(images: dict):
    """Demonstrate baseline comparison."""
    print("\n" + "=" * 60)
    print("DEMO: Baseline Comparison")
    print("=" * 60)

    try:
        from imagetrust.baselines import get_baseline, list_baselines

        available = list_baselines()
        print(f"\nAvailable baselines: {available}")

        # Prepare data
        all_paths = images["real"] + images["ai"]
        labels = [0] * len(images["real"]) + [1] * len(images["ai"])

        print("\nEvaluating baselines...")
        print("-" * 50)

        for baseline_name in ["imagetrust"]:
            if baseline_name not in available:
                continue

            baseline = get_baseline(baseline_name)

            correct = 0
            total = 0

            for path, label in zip(all_paths, labels):
                result = baseline.predict_proba(str(path))
                pred = 1 if result.ai_probability > 0.5 else 0
                if pred == label:
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0
            print(f"  {baseline_name:20s}: Accuracy = {accuracy:.1%} ({correct}/{total})")

        print("\n✓ Baseline comparison demo complete!")

    except ImportError as e:
        print(f"\n⚠ Baselines not available: {e}")
        _mock_baseline_demo()


def _mock_baseline_demo():
    """Mock baseline demo."""
    print("\n  [Mock Demo - Baselines not loaded]")
    print("-" * 50)
    print("  Classical (LogReg):     Accuracy = 78.0%")
    print("  CNN (ResNet-50):        Accuracy = 86.0%")
    print("  ViT-B/16:               Accuracy = 88.0%")
    print("  ImageTrust (Ours):      Accuracy = 92.0%")


def demo_calibration():
    """Demonstrate probability calibration."""
    print("\n" + "=" * 60)
    print("DEMO: Probability Calibration")
    print("=" * 60)

    np.random.seed(42)

    # Simulate uncalibrated predictions
    n_samples = 100
    labels = np.random.randint(0, 2, n_samples)

    # Overconfident predictions
    raw_probs = np.zeros(n_samples)
    for i in range(n_samples):
        if labels[i] == 1:
            raw_probs[i] = np.clip(0.85 + np.random.normal(0, 0.1), 0.5, 0.99)
        else:
            raw_probs[i] = np.clip(0.15 + np.random.normal(0, 0.1), 0.01, 0.5)

    # Calculate ECE
    def calc_ece(probs, labels, n_bins=10):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_acc = labels[mask].mean()
                bin_conf = probs[mask].mean()
                ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
        return ece

    ece_before = calc_ece(raw_probs, labels)

    # Apply temperature scaling
    def apply_temp(probs, temp):
        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs / (1 - probs))
        scaled = logits / temp
        return 1 / (1 + np.exp(-scaled))

    # Find optimal temperature
    best_temp = 1.0
    best_ece = ece_before
    for temp in np.arange(0.5, 3.0, 0.1):
        cal_probs = apply_temp(raw_probs, temp)
        ece = calc_ece(cal_probs, labels)
        if ece < best_ece:
            best_ece = ece
            best_temp = temp

    calibrated_probs = apply_temp(raw_probs, best_temp)
    ece_after = calc_ece(calibrated_probs, labels)

    print("\nCalibration Results:")
    print("-" * 50)
    print(f"  Optimal Temperature:  {best_temp:.2f}")
    print(f"  ECE Before:           {ece_before:.4f}")
    print(f"  ECE After:            {ece_after:.4f}")
    print(f"  ECE Improvement:      {(ece_before - ece_after) / ece_before * 100:.1f}%")

    print("\n  Key Insight: Calibration improves confidence reliability")
    print("               without changing accuracy!")

    print("\n✓ Calibration demo complete!")


def demo_batch_processing(images: dict):
    """Demonstrate batch processing."""
    print("\n" + "=" * 60)
    print("DEMO: Batch Processing")
    print("=" * 60)

    all_paths = images["real"] + images["ai"]

    print(f"\nProcessing {len(all_paths)} images...")
    print("-" * 50)

    start_time = time.time()

    try:
        from imagetrust.detection import AIDetector

        detector = AIDetector()
        results = []

        for i, path in enumerate(all_paths):
            result = detector.detect(str(path))
            results.append({
                "file": path.name,
                "ai_probability": result["ai_probability"],
                "verdict": result["verdict"].value,
            })

            # Progress
            progress = (i + 1) / len(all_paths)
            bar = "█" * int(progress * 30) + "░" * (30 - int(progress * 30))
            print(f"\r  Progress: [{bar}] {progress:.0%}", end="", flush=True)

        print()  # New line after progress

    except ImportError:
        # Mock results
        results = []
        for i, path in enumerate(all_paths):
            prob = np.random.uniform(0.1, 0.9)
            results.append({
                "file": path.name,
                "ai_probability": prob,
                "verdict": "ai_generated" if prob > 0.5 else "real",
            })

            progress = (i + 1) / len(all_paths)
            bar = "█" * int(progress * 30) + "░" * (30 - int(progress * 30))
            print(f"\r  Progress: [{bar}] {progress:.0%}", end="", flush=True)
            time.sleep(0.05)  # Simulate processing

        print()

    elapsed = time.time() - start_time

    # Summary
    ai_count = sum(1 for r in results if r["verdict"] == "ai_generated")
    real_count = len(results) - ai_count

    print(f"\nResults:")
    print(f"  Total:        {len(results)} images")
    print(f"  AI Detected:  {ai_count} ({ai_count/len(results):.0%})")
    print(f"  Real:         {real_count} ({real_count/len(results):.0%})")
    print(f"  Time:         {elapsed:.2f}s ({elapsed/len(results)*1000:.0f}ms/image)")

    print("\n✓ Batch processing demo complete!")


def demo_evaluation_metrics():
    """Demonstrate evaluation metrics."""
    print("\n" + "=" * 60)
    print("DEMO: Evaluation Metrics")
    print("=" * 60)

    np.random.seed(42)

    # Simulate predictions
    n = 100
    y_true = np.random.randint(0, 2, n)
    y_prob = np.zeros(n)
    for i in range(n):
        if y_true[i] == 1:
            y_prob[i] = np.clip(0.75 + np.random.normal(0, 0.2), 0, 1)
        else:
            y_prob[i] = np.clip(0.25 + np.random.normal(0, 0.2), 0, 1)

    y_pred = (y_prob > 0.5).astype(int)

    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score,
        recall_score, f1_score, roc_auc_score
    )

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_prob),
    }

    print("\nMetrics (simulated data):")
    print("-" * 50)
    for name, value in metrics.items():
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"  {name:20s} [{bar}] {value:.3f}")

    print("\n✓ Evaluation metrics demo complete!")


# =============================================================================
# Main Demo Runner
# =============================================================================

def run_all_demos():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("ImageTrust Quick Demo")
    print("=" * 60)
    print("\nThis demo showcases the main features of ImageTrust.")
    print("Synthetic test images will be created for demonstration.")

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="imagetrust_demo_"))
    print(f"\nDemo directory: {temp_dir}")

    try:
        # Create demo images
        print("\n[1/5] Creating demo images...")
        images = create_demo_images(temp_dir, n_real=5, n_ai=5)
        print(f"  Created {len(images['real'])} real + {len(images['ai'])} AI images")

        # Run demos
        print("\n[2/5] Running detection demo...")
        demo_basic_detection(images)

        print("\n[3/5] Running baseline comparison...")
        demo_baseline_comparison(images)

        print("\n[4/5] Running calibration demo...")
        demo_calibration()

        print("\n[5/5] Running evaluation metrics demo...")
        demo_evaluation_metrics()

        # Summary
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print("\nImageTrust successfully demonstrated:")
        print("  ✓ AI image detection")
        print("  ✓ Baseline comparison framework")
        print("  ✓ Probability calibration")
        print("  ✓ Evaluation metrics")
        print("\nFor more examples, see: examples/")
        print("For full documentation, see: docs/")

    finally:
        # Cleanup
        print(f"\nCleaning up {temp_dir}...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="ImageTrust quick demo",
    )
    parser.add_argument(
        "--feature",
        choices=["detection", "baseline", "calibration", "metrics", "batch", "all"],
        default="all",
        help="Which feature to demo",
    )

    args = parser.parse_args()

    if args.feature == "all":
        run_all_demos()
    else:
        # Create temp images
        temp_dir = Path(tempfile.mkdtemp(prefix="imagetrust_demo_"))
        try:
            images = create_demo_images(temp_dir, n_real=5, n_ai=5)

            if args.feature == "detection":
                demo_basic_detection(images)
            elif args.feature == "baseline":
                demo_baseline_comparison(images)
            elif args.feature == "calibration":
                demo_calibration()
            elif args.feature == "metrics":
                demo_evaluation_metrics()
            elif args.feature == "batch":
                demo_batch_processing(images)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
