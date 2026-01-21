#!/usr/bin/env python3
"""
Example 1: Basic AI Image Detection

Demonstrates how to use ImageTrust for simple AI image detection.

Usage:
    python examples/01_basic_detection.py path/to/image.jpg
    python examples/01_basic_detection.py --demo  # Uses synthetic test image
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_demo_image() -> Path:
    """Create a simple test image for demonstration."""
    from PIL import Image
    import numpy as np

    # Create a simple gradient image
    width, height = 512, 512
    arr = np.zeros((height, width, 3), dtype=np.uint8)

    # Create gradient
    for y in range(height):
        for x in range(width):
            arr[y, x] = [
                int(255 * x / width),
                int(255 * y / height),
                128,
            ]

    img = Image.fromarray(arr)
    demo_path = Path("demo_image.jpg")
    img.save(demo_path, quality=95)
    print(f"Created demo image: {demo_path}")
    return demo_path


def main():
    parser = argparse.ArgumentParser(description="Basic AI image detection example")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--demo", action="store_true", help="Use demo image")
    args = parser.parse_args()

    # Get image path
    if args.demo:
        image_path = create_demo_image()
    elif args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)
    else:
        print("Error: Please provide an image path or use --demo")
        parser.print_help()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ImageTrust - Basic Detection Example")
    print("=" * 60)

    # Method 1: Using AIDetector (high-level API)
    print("\n[Method 1] Using AIDetector:")
    print("-" * 40)

    try:
        from imagetrust.detection import AIDetector

        detector = AIDetector()
        result = detector.detect(str(image_path))

        print(f"Image: {image_path}")
        print(f"AI Probability: {result['ai_probability']:.1%}")
        print(f"Real Probability: {result['real_probability']:.1%}")
        print(f"Verdict: {result['verdict'].value}")
        print(f"Confidence: {result['confidence'].value}")
        print(f"Calibrated: {result['calibrated']}")
        print(f"Processing Time: {result['processing_time_ms']:.1f} ms")

    except ImportError as e:
        print(f"Note: Full detector not available ({e})")
        print("Showing mock result...")

        # Mock result for demonstration
        print(f"Image: {image_path}")
        print("AI Probability: 85.2%")
        print("Real Probability: 14.8%")
        print("Verdict: ai_generated")
        print("Confidence: high")

    # Method 2: Using Baselines (research API)
    print("\n[Method 2] Using Baseline Framework:")
    print("-" * 40)

    try:
        from imagetrust.baselines import get_baseline, list_baselines

        print(f"Available baselines: {list_baselines()}")

        # Use ImageTrust baseline (pretrained ensemble)
        baseline = get_baseline("imagetrust")
        result = baseline.predict_proba(str(image_path))

        print(f"AI Probability: {result.ai_probability:.1%}")
        print(f"Calibrated: {result.calibrated}")
        print(f"Processing Time: {result.processing_time_ms:.1f} ms")

    except ImportError as e:
        print(f"Note: Baseline framework not available ({e})")

    # Method 3: Direct PIL Image input
    print("\n[Method 3] Direct PIL Image Input:")
    print("-" * 40)

    try:
        from PIL import Image
        from imagetrust.detection import AIDetector

        # Load image with PIL
        pil_image = Image.open(image_path)
        print(f"Image size: {pil_image.size}")
        print(f"Image mode: {pil_image.mode}")

        # Detect directly from PIL Image
        detector = AIDetector()
        result = detector.detect(pil_image)
        print(f"AI Probability: {result['ai_probability']:.1%}")

    except ImportError:
        print("Note: Direct PIL input demo skipped")

    print("\n" + "=" * 60)
    print("Detection Complete")
    print("=" * 60)

    # Cleanup demo image
    if args.demo and image_path.exists():
        image_path.unlink()
        print("\nDemo image cleaned up.")


if __name__ == "__main__":
    main()
