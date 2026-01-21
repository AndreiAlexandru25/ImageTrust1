#!/usr/bin/env python3
"""
Example 2: Batch Image Processing

Demonstrates how to efficiently process multiple images with ImageTrust.

Usage:
    python examples/02_batch_processing.py path/to/images/
    python examples/02_batch_processing.py --demo  # Creates test images
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_demo_images(n: int = 5) -> List[Path]:
    """Create test images for demonstration."""
    from PIL import Image
    import numpy as np

    paths = []
    demo_dir = Path("demo_images")
    demo_dir.mkdir(exist_ok=True)

    for i in range(n):
        # Create varied images
        width, height = 256, 256
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        img = Image.fromarray(arr)
        path = demo_dir / f"test_image_{i+1:03d}.jpg"
        img.save(path, quality=90)
        paths.append(path)

    print(f"Created {len(paths)} demo images in {demo_dir}/")
    return paths


def process_batch(image_paths: List[Path]) -> dict:
    """Process a batch of images and return results."""
    results = {
        "total_images": len(image_paths),
        "ai_detected": 0,
        "real_detected": 0,
        "uncertain": 0,
        "processing_time_ms": 0,
        "details": [],
    }

    try:
        from imagetrust.detection import AIDetector

        detector = AIDetector()

        print("\nProcessing images...")
        print("-" * 50)

        start_time = time.time()

        for i, path in enumerate(image_paths, 1):
            result = detector.detect(str(path))

            # Categorize result
            verdict = result["verdict"].value
            if verdict == "ai_generated":
                results["ai_detected"] += 1
            elif verdict == "real":
                results["real_detected"] += 1
            else:
                results["uncertain"] += 1

            results["details"].append({
                "filename": path.name,
                "ai_probability": result["ai_probability"],
                "verdict": verdict,
                "confidence": result["confidence"].value,
            })

            # Progress indicator
            prob = result["ai_probability"]
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            print(f"[{i:3d}/{len(image_paths)}] {path.name:30s} [{bar}] {prob:6.1%} {verdict}")

        results["processing_time_ms"] = (time.time() - start_time) * 1000

    except ImportError as e:
        print(f"Note: Full detector not available ({e})")
        print("Using mock results...")

        import random
        for path in image_paths:
            prob = random.uniform(0.1, 0.9)
            verdict = "ai_generated" if prob > 0.5 else "real"
            results["details"].append({
                "filename": path.name,
                "ai_probability": prob,
                "verdict": verdict,
            })
            if verdict == "ai_generated":
                results["ai_detected"] += 1
            else:
                results["real_detected"] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch image processing example")
    parser.add_argument("directory", nargs="?", help="Directory containing images")
    parser.add_argument("--demo", action="store_true", help="Use demo images")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--limit", "-n", type=int, default=100, help="Max images to process")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ImageTrust - Batch Processing Example")
    print("=" * 60)

    # Get image paths
    if args.demo:
        image_paths = create_demo_images(10)
    elif args.directory:
        dir_path = Path(args.directory)
        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            sys.exit(1)

        # Find images
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_paths = [
            p for p in dir_path.iterdir()
            if p.suffix.lower() in extensions
        ][:args.limit]

        if not image_paths:
            print(f"No images found in {dir_path}")
            sys.exit(1)

        print(f"Found {len(image_paths)} images")
    else:
        print("Error: Please provide a directory or use --demo")
        parser.print_help()
        sys.exit(1)

    # Process batch
    results = process_batch(image_paths)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total Images:     {results['total_images']}")
    print(f"AI Generated:     {results['ai_detected']} ({results['ai_detected']/results['total_images']:.1%})")
    print(f"Real Images:      {results['real_detected']} ({results['real_detected']/results['total_images']:.1%})")
    print(f"Uncertain:        {results['uncertain']}")
    print(f"Processing Time:  {results['processing_time_ms']:.1f} ms")
    print(f"Avg per Image:    {results['processing_time_ms']/results['total_images']:.1f} ms")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Cleanup
    if args.demo:
        import shutil
        shutil.rmtree("demo_images", ignore_errors=True)
        print("\nDemo images cleaned up.")


if __name__ == "__main__":
    main()
