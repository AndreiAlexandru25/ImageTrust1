#!/usr/bin/env python
"""
Generate Synthetic Robustness Dataset.

Creates synthetic variants of existing images to train robust AI detectors:
- Social media compressed versions (Instagram, WhatsApp, Twitter)
- Screenshot variants with UI overlays
- Various degradation combinations

This augments the existing dataset to improve model robustness against
False Positives on social media images and screenshots.

Usage:
    python scripts/generate_synthetic_robustness_data.py \
        --input_dir data/train \
        --output_dir data/synthetic \
        --platforms instagram whatsapp twitter \
        --include_screenshots

Output Structure:
    data/synthetic/
    ├── social_media/
    │   ├── instagram/
    │   ├── whatsapp/
    │   └── twitter/
    ├── screenshots/
    │   ├── windows/
    │   ├── macos/
    │   └── mobile/
    └── manifest.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imagetrust.detection.augmentation import (
    Platform,
    ScreenshotType,
    SocialMediaSimulator,
    ScreenshotSimulator,
)


def find_images(input_dir: Path, extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")) -> List[Path]:
    """Find all image files in directory."""
    images = []
    for ext in extensions:
        images.extend(input_dir.rglob(f"*{ext}"))
        images.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(images)


def get_label_from_path(image_path: Path) -> int:
    """
    Infer label from image path.

    Assumes directory structure like:
    - data/train/real/*.jpg -> 0
    - data/train/ai/*.jpg -> 1
    - data/train/fake/*.jpg -> 1
    """
    path_parts = [p.lower() for p in image_path.parts]

    # Check for AI/fake indicators
    ai_indicators = ["ai", "fake", "generated", "synthetic", "deepfake", "midjourney", "dalle", "sd"]
    for indicator in ai_indicators:
        if any(indicator in part for part in path_parts):
            return 1

    # Check for real indicators
    real_indicators = ["real", "authentic", "photo", "natural", "camera"]
    for indicator in real_indicators:
        if any(indicator in part for part in path_parts):
            return 0

    # Default to real if unclear
    return 0


def generate_social_media_variants(
    input_dir: Path,
    output_dir: Path,
    platforms: List[str],
    num_variants_per_image: int = 1,
    compression_rounds: int = 2,
) -> List[Dict]:
    """
    Generate social media compressed versions of images.

    Args:
        input_dir: Directory containing original images.
        output_dir: Directory to save compressed variants.
        platforms: List of platform names.
        num_variants_per_image: Number of variants to generate per image.
        compression_rounds: Number of compression passes.

    Returns:
        List of manifest entries.
    """
    # Convert platform names to enum
    platform_map = {
        "instagram": Platform.INSTAGRAM,
        "whatsapp": Platform.WHATSAPP,
        "twitter": Platform.TWITTER,
        "facebook": Platform.FACEBOOK,
        "telegram": Platform.TELEGRAM,
        "generic": Platform.GENERIC,
    }

    platform_enums = [platform_map.get(p.lower(), Platform.GENERIC) for p in platforms]

    simulator = SocialMediaSimulator(
        platforms=platform_enums,
        compression_rounds=compression_rounds,
    )

    images = find_images(input_dir)
    manifest = []

    print(f"Found {len(images)} images in {input_dir}")
    print(f"Generating {num_variants_per_image} variant(s) per image for platforms: {platforms}")

    for image_path in tqdm(images, desc="Generating social media variants"):
        try:
            image = Image.open(image_path).convert("RGB")
            original_label = get_label_from_path(image_path)

            for platform in platform_enums:
                for variant_idx in range(num_variants_per_image):
                    # Apply simulation
                    degraded, metadata = simulator.simulate(image, platform=platform)

                    # Create output path
                    platform_dir = output_dir / "social_media" / platform.value
                    platform_dir.mkdir(parents=True, exist_ok=True)

                    # Generate filename
                    stem = image_path.stem
                    suffix = f"_v{variant_idx}" if num_variants_per_image > 1 else ""
                    output_path = platform_dir / f"{stem}{suffix}.jpg"

                    # Save
                    degraded.save(output_path, "JPEG", quality=95)

                    # Add to manifest
                    manifest.append({
                        "path": str(output_path.relative_to(output_dir)),
                        "original_path": str(image_path),
                        "label": original_label,
                        "augmentation_type": "social_media",
                        "platform": platform.value,
                        "variant_idx": variant_idx,
                        "metadata": metadata,
                    })

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    return manifest


def generate_screenshot_variants(
    input_dir: Path,
    output_dir: Path,
    screenshot_types: Optional[List[str]] = None,
    num_variants_per_image: int = 1,
) -> List[Dict]:
    """
    Generate screenshot versions with UI overlays.

    Args:
        input_dir: Directory containing original images.
        output_dir: Directory to save screenshot variants.
        screenshot_types: List of screenshot types.
        num_variants_per_image: Number of variants per image.

    Returns:
        List of manifest entries.
    """
    # Convert type names to enum
    type_map = {
        "windows": ScreenshotType.WINDOWS,
        "macos": ScreenshotType.MACOS,
        "mobile_ios": ScreenshotType.MOBILE_IOS,
        "mobile_android": ScreenshotType.MOBILE_ANDROID,
        "mobile": ScreenshotType.MOBILE_IOS,
        "browser": ScreenshotType.BROWSER,
        "generic": ScreenshotType.GENERIC,
    }

    if screenshot_types:
        type_enums = [type_map.get(t.lower(), ScreenshotType.GENERIC) for t in screenshot_types]
    else:
        type_enums = list(ScreenshotType)

    simulator = ScreenshotSimulator(
        screenshot_types=type_enums,
        add_ui_elements=True,
        add_borders=True,
        add_text_overlays=True,
    )

    images = find_images(input_dir)
    manifest = []

    print(f"Found {len(images)} images in {input_dir}")
    print(f"Generating screenshot variants for types: {[t.value for t in type_enums]}")

    for image_path in tqdm(images, desc="Generating screenshot variants"):
        try:
            image = Image.open(image_path).convert("RGB")
            original_label = get_label_from_path(image_path)

            for ss_type in type_enums:
                for variant_idx in range(num_variants_per_image):
                    # Apply simulation
                    screenshot, metadata = simulator.simulate(image, screenshot_type=ss_type)

                    # Create output path
                    type_dir = output_dir / "screenshots" / ss_type.value
                    type_dir.mkdir(parents=True, exist_ok=True)

                    # Generate filename
                    stem = image_path.stem
                    suffix = f"_v{variant_idx}" if num_variants_per_image > 1 else ""
                    output_path = type_dir / f"{stem}{suffix}.png"

                    # Save as PNG (typical for screenshots)
                    screenshot.save(output_path, "PNG")

                    # Add to manifest
                    manifest.append({
                        "path": str(output_path.relative_to(output_dir)),
                        "original_path": str(image_path),
                        "label": original_label,
                        "augmentation_type": "screenshot",
                        "screenshot_type": ss_type.value,
                        "variant_idx": variant_idx,
                        "metadata": metadata,
                    })

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    return manifest


def create_robustness_splits(
    original_dir: Path,
    synthetic_dir: Path,
    output_manifest: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict:
    """
    Create train/val/test splits mixing original + synthetic.

    Strategy:
    - Train: Uses heavy augmentation mix of original + all synthetic
    - Val: Uses original + synthetic for threshold calibration
    - Test: Uses only original (clean evaluation)

    Args:
        original_dir: Directory with original images.
        synthetic_dir: Directory with synthetic variants.
        output_manifest: Path to save manifest JSON.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.

    Returns:
        Manifest dictionary.
    """
    import random

    random.seed(42)

    # Load synthetic manifest
    synthetic_manifest_path = synthetic_dir / "manifest.json"
    if synthetic_manifest_path.exists():
        with open(synthetic_manifest_path) as f:
            synthetic_entries = json.load(f)
    else:
        synthetic_entries = []

    # Find original images
    original_images = find_images(original_dir)

    # Create entries for original images
    original_entries = []
    for img_path in original_images:
        original_entries.append({
            "path": str(img_path),
            "label": get_label_from_path(img_path),
            "augmentation_type": "original",
            "split": None,  # Will be assigned below
        })

    # Shuffle and split original images
    random.shuffle(original_entries)
    n_total = len(original_entries)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    for i, entry in enumerate(original_entries):
        if i < n_train:
            entry["split"] = "train"
        elif i < n_train + n_val:
            entry["split"] = "val"
        else:
            entry["split"] = "test"

    # Assign splits to synthetic (follow original's split)
    original_path_to_split = {
        e["path"]: e["split"] for e in original_entries
    }

    for entry in synthetic_entries:
        original_path = entry.get("original_path", "")
        # Match based on filename
        matched_split = None
        for orig_path, split in original_path_to_split.items():
            if Path(original_path).name == Path(orig_path).name:
                matched_split = split
                break
        entry["split"] = matched_split or "train"  # Default to train

    # Combine
    all_entries = original_entries + synthetic_entries

    # Create split summary
    splits = {"train": [], "val": [], "test": []}
    for entry in all_entries:
        split = entry.get("split", "train")
        splits[split].append(entry)

    manifest = {
        "created_at": str(Path(__file__).stat().st_mtime),
        "original_count": len(original_entries),
        "synthetic_count": len(synthetic_entries),
        "total_count": len(all_entries),
        "splits": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "entries": all_entries,
    }

    # Save
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {output_manifest}")
    print(f"  Original images: {len(original_entries)}")
    print(f"  Synthetic images: {len(synthetic_entries)}")
    print(f"  Train: {manifest['splits']['train']}")
    print(f"  Val: {manifest['splits']['val']}")
    print(f"  Test: {manifest['splits']['test']}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic robustness dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing original images",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Output directory for synthetic data",
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        default=["instagram", "whatsapp", "twitter"],
        help="Social media platforms to simulate",
    )
    parser.add_argument(
        "--include_screenshots",
        action="store_true",
        help="Include screenshot variants",
    )
    parser.add_argument(
        "--screenshot_types",
        nargs="+",
        default=["windows", "macos", "mobile"],
        help="Screenshot types to generate",
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=1,
        help="Number of variants per image per augmentation type",
    )
    parser.add_argument(
        "--compression_rounds",
        type=int,
        default=2,
        help="Number of JPEG compression rounds for social media",
    )
    parser.add_argument(
        "--create_splits",
        action="store_true",
        help="Create train/val/test splits",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_manifest_entries = []

    # Generate social media variants
    if args.platforms:
        print("\n=== Generating Social Media Variants ===")
        entries = generate_social_media_variants(
            args.input_dir,
            args.output_dir,
            args.platforms,
            num_variants_per_image=args.num_variants,
            compression_rounds=args.compression_rounds,
        )
        all_manifest_entries.extend(entries)
        print(f"Generated {len(entries)} social media variants")

    # Generate screenshot variants
    if args.include_screenshots:
        print("\n=== Generating Screenshot Variants ===")
        entries = generate_screenshot_variants(
            args.input_dir,
            args.output_dir,
            args.screenshot_types,
            num_variants_per_image=args.num_variants,
        )
        all_manifest_entries.extend(entries)
        print(f"Generated {len(entries)} screenshot variants")

    # Save manifest
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_manifest_entries, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Total synthetic images: {len(all_manifest_entries)}")

    # Create splits if requested
    if args.create_splits:
        print("\n=== Creating Train/Val/Test Splits ===")
        create_robustness_splits(
            args.input_dir,
            args.output_dir,
            args.output_dir / "splits_manifest.json",
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
