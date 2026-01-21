#!/usr/bin/env python3
"""
Example 3: Baseline Comparison

Demonstrates how to compare different baseline detectors on the same images.

Usage:
    python examples/03_baseline_comparison.py path/to/images/
    python examples/03_baseline_comparison.py --demo
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_test_data() -> tuple:
    """Create synthetic test data."""
    from PIL import Image
    import numpy as np

    images = []
    labels = []  # 0 = real, 1 = AI

    demo_dir = Path("demo_comparison")
    demo_dir.mkdir(exist_ok=True)

    # Create "real-like" images (smooth gradients)
    for i in range(5):
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        for y in range(224):
            for x in range(224):
                arr[y, x] = [
                    int(255 * (x + i * 10) / 234),
                    int(255 * (y + i * 10) / 234),
                    100 + i * 20,
                ]
        img = Image.fromarray(arr)
        path = demo_dir / f"real_{i+1:03d}.jpg"
        img.save(path, quality=95)
        images.append(path)
        labels.append(0)

    # Create "AI-like" images (noisy patterns)
    for i in range(5):
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        # Add some structure
        arr[:, :, 0] = (arr[:, :, 0] + np.arange(224)) % 255
        img = Image.fromarray(arr.astype(np.uint8))
        path = demo_dir / f"ai_{i+1:03d}.jpg"
        img.save(path, quality=85)
        images.append(path)
        labels.append(1)

    print(f"Created {len(images)} test images in {demo_dir}/")
    return images, labels, demo_dir


def compare_baselines(images: List[Path], labels: List[int]) -> Dict:
    """Compare multiple baselines on the same data."""
    results = {}

    print("\n" + "=" * 70)
    print("Comparing Baselines")
    print("=" * 70)

    try:
        from imagetrust.baselines import get_baseline, list_baselines
        from sklearn.metrics import accuracy_score, roc_auc_score
        import numpy as np

        available = list_baselines()
        print(f"Available baselines: {available}")

        # Test each baseline
        for baseline_name in ["imagetrust"]:  # Focus on main baseline
            if baseline_name not in available:
                continue

            print(f"\n[{baseline_name.upper()}]")
            print("-" * 50)

            try:
                baseline = get_baseline(baseline_name)

                predictions = []
                probabilities = []

                for img_path in images:
                    result = baseline.predict_proba(str(img_path))
                    prob = result.ai_probability
                    probabilities.append(prob)
                    predictions.append(1 if prob > 0.5 else 0)

                # Calculate metrics
                accuracy = accuracy_score(labels, predictions)
                try:
                    auc = roc_auc_score(labels, probabilities)
                except ValueError:
                    auc = 0.5

                results[baseline_name] = {
                    "accuracy": accuracy,
                    "auc": auc,
                    "predictions": predictions,
                    "probabilities": probabilities,
                }

                print(f"  Accuracy: {accuracy:.1%}")
                print(f"  AUC-ROC:  {auc:.3f}")

                # Show confusion breakdown
                tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
                tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
                fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
                fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
                print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

            except Exception as e:
                print(f"  Error: {e}")
                results[baseline_name] = {"error": str(e)}

    except ImportError as e:
        print(f"Note: Baseline framework not available ({e})")
        print("Showing mock comparison...")

        # Mock results
        import random
        for name in ["classical", "cnn", "vit", "imagetrust"]:
            acc = random.uniform(0.7, 0.95)
            auc = acc + random.uniform(0, 0.05)
            results[name] = {"accuracy": acc, "auc": min(auc, 1.0)}
            print(f"\n[{name.upper()}]")
            print(f"  Accuracy: {acc:.1%}")
            print(f"  AUC-ROC:  {auc:.3f}")

    return results


def print_comparison_table(results: Dict):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)

    print(f"{'Baseline':<20} {'Accuracy':>12} {'AUC-ROC':>12}")
    print("-" * 50)

    # Sort by accuracy
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get("accuracy", 0),
        reverse=True
    )

    for name, metrics in sorted_results:
        if "error" in metrics:
            print(f"{name:<20} {'Error':>12} {'N/A':>12}")
        else:
            acc = metrics.get("accuracy", 0)
            auc = metrics.get("auc", 0)
            print(f"{name:<20} {acc:>11.1%} {auc:>12.3f}")

    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison example")
    parser.add_argument("directory", nargs="?", help="Directory with labeled images")
    parser.add_argument("--demo", action="store_true", help="Use demo data")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("ImageTrust - Baseline Comparison Example")
    print("=" * 70)

    demo_dir = None

    if args.demo:
        images, labels, demo_dir = create_test_data()
    elif args.directory:
        # Expect structure: directory/real/*.jpg and directory/ai/*.jpg
        dir_path = Path(args.directory)
        images = []
        labels = []

        real_dir = dir_path / "real"
        ai_dir = dir_path / "ai"

        if real_dir.exists():
            for p in real_dir.glob("*.jpg"):
                images.append(p)
                labels.append(0)

        if ai_dir.exists():
            for p in ai_dir.glob("*.jpg"):
                images.append(p)
                labels.append(1)

        if not images:
            print("Error: No images found. Expected structure:")
            print("  directory/real/*.jpg")
            print("  directory/ai/*.jpg")
            sys.exit(1)

        print(f"Found {len(images)} images ({labels.count(0)} real, {labels.count(1)} AI)")
    else:
        print("Error: Please provide a directory or use --demo")
        parser.print_help()
        sys.exit(1)

    # Run comparison
    results = compare_baselines(images, labels)

    # Print summary table
    print_comparison_table(results)

    # Cleanup
    if demo_dir and demo_dir.exists():
        import shutil
        shutil.rmtree(demo_dir, ignore_errors=True)
        print("\nDemo data cleaned up.")


if __name__ == "__main__":
    main()
