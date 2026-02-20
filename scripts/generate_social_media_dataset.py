#!/usr/bin/env python
"""
Social Media Recompression Dataset Generator for Novel Contribution Evaluation.

Simulates image processing pipelines of major social media platforms:
- WhatsApp: JPEG Q~75, resize to 1600px max, strip EXIF
- Instagram: JPEG Q~80, specific dimensions, sharpening
- Facebook: JPEG Q~85, resize to 2048px max
- Twitter/X: Variable JPEG compression
- Telegram: Minimal compression, preserves quality

This enables evaluation of the Platform Detection module (Table 8).

Usage:
    python scripts/generate_social_media_dataset.py --input data/real --output data/social_media
    python scripts/generate_social_media_dataset.py --input data/real --output data/social_media --platforms all
"""

import argparse
import io
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)


# Platform-specific processing parameters (based on real-world analysis)
PLATFORM_CONFIGS = {
    "whatsapp": {
        "description": "WhatsApp (aggressive compression, EXIF stripping)",
        "max_dimension": 1600,
        "jpeg_quality_range": (72, 82),
        "strip_exif": True,
        "resize_method": "lanczos",
        "color_subsampling": "4:2:0",
        "additional_processing": ["slight_sharpen"],
    },
    "instagram_feed": {
        "description": "Instagram Feed (square crops, specific sizing)",
        "max_dimension": 1080,
        "jpeg_quality_range": (70, 85),
        "strip_exif": True,
        "resize_method": "lanczos",
        "color_subsampling": "4:2:0",
        "preferred_aspect_ratios": [(1, 1), (4, 5), (1.91, 1)],  # Square, Portrait, Landscape
        "additional_processing": ["sharpen", "slight_saturation_boost"],
    },
    "instagram_story": {
        "description": "Instagram Story (9:16 aspect, lower quality)",
        "max_dimension": 1080,
        "target_resolution": (1080, 1920),
        "jpeg_quality_range": (65, 80),
        "strip_exif": True,
        "resize_method": "lanczos",
        "color_subsampling": "4:2:0",
        "additional_processing": ["sharpen"],
    },
    "facebook": {
        "description": "Facebook (moderate compression)",
        "max_dimension": 2048,
        "jpeg_quality_range": (78, 88),
        "strip_exif": True,
        "resize_method": "lanczos",
        "color_subsampling": "4:2:0",
        "additional_processing": [],
    },
    "twitter": {
        "description": "Twitter/X (variable quality)",
        "max_dimension": 4096,
        "jpeg_quality_range": (75, 90),
        "strip_exif": True,
        "resize_method": "lanczos",
        "color_subsampling": "4:2:0",
        "large_image_threshold": 5242880,  # 5MB - larger images get more compression
        "additional_processing": [],
    },
    "telegram": {
        "description": "Telegram (minimal compression, preserves quality)",
        "max_dimension": 1280,
        "jpeg_quality_range": (85, 95),
        "strip_exif": False,  # Telegram preserves some metadata
        "resize_method": "lanczos",
        "color_subsampling": "4:2:2",  # Less aggressive subsampling
        "additional_processing": [],
    },
    "discord": {
        "description": "Discord (file size based compression)",
        "max_dimension": 2048,
        "jpeg_quality_range": (75, 90),
        "strip_exif": True,
        "resize_method": "lanczos",
        "color_subsampling": "4:2:0",
        "max_file_size": 8388608,  # 8MB
        "additional_processing": [],
    },
    "reddit": {
        "description": "Reddit (hosted images)",
        "max_dimension": 3024,
        "jpeg_quality_range": (80, 92),
        "strip_exif": True,
        "resize_method": "lanczos",
        "color_subsampling": "4:2:0",
        "additional_processing": [],
    },
}


def apply_platform_resize(
    image: Image.Image,
    config: Dict[str, Any],
) -> Image.Image:
    """
    Resize image according to platform rules.
    """
    max_dim = config.get("max_dimension", 2048)
    target_res = config.get("target_resolution")

    if target_res:
        # Fixed target resolution (e.g., stories)
        return image.resize(target_res, Image.Resampling.LANCZOS)

    # Resize if exceeds max dimension
    width, height = image.size
    max_current = max(width, height)

    if max_current > max_dim:
        scale = max_dim / max_current
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def apply_aspect_ratio_crop(
    image: Image.Image,
    config: Dict[str, Any],
) -> Image.Image:
    """
    Optionally crop to platform's preferred aspect ratio.
    """
    preferred_ratios = config.get("preferred_aspect_ratios")
    if not preferred_ratios:
        return image

    # Randomly decide whether to crop
    if random.random() > 0.5:
        return image

    width, height = image.size
    current_ratio = width / height

    # Find closest preferred ratio
    target_ratio_tuple = min(preferred_ratios, key=lambda r: abs(r[0] / r[1] - current_ratio))
    target_ratio = target_ratio_tuple[0] / target_ratio_tuple[1]

    if abs(current_ratio - target_ratio) < 0.05:
        return image

    # Crop to target ratio
    if target_ratio > current_ratio:
        # Need to crop height
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        return image.crop((0, top, width, top + new_height))
    else:
        # Need to crop width
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))


def apply_additional_processing(
    image: Image.Image,
    processing_list: List[str],
) -> Image.Image:
    """
    Apply platform-specific additional processing.
    """
    for proc in processing_list:
        if proc == "sharpen":
            image = image.filter(ImageFilter.SHARPEN)
        elif proc == "slight_sharpen":
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
        elif proc == "slight_saturation_boost":
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.1)
        elif proc == "contrast_boost":
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)

    return image


def compress_to_target_size(
    image: Image.Image,
    max_size: int,
    initial_quality: int,
) -> Tuple[bytes, int]:
    """
    Iteratively compress until file size is under target.
    """
    quality = initial_quality

    while quality > 30:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()

        if len(data) <= max_size:
            return data, quality

        quality -= 5

    # Return whatever we got at lowest quality
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=30, optimize=True)
    return buffer.getvalue(), 30


def simulate_platform_processing(
    image: Image.Image,
    platform: str,
    config: Dict[str, Any],
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Simulate full platform processing pipeline.

    Args:
        image: Original image
        platform: Platform name
        config: Platform configuration

    Returns:
        Tuple of (processed_image, metadata)
    """
    metadata = {
        "platform": platform,
        "original_size": image.size,
        "original_format": image.format,
    }

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 1. Apply aspect ratio crop (if applicable)
    image = apply_aspect_ratio_crop(image, config)
    metadata["after_crop_size"] = image.size

    # 2. Resize according to platform rules
    image = apply_platform_resize(image, config)
    metadata["after_resize_size"] = image.size

    # 3. Apply additional processing
    additional = config.get("additional_processing", [])
    if additional:
        image = apply_additional_processing(image, additional)
        metadata["additional_processing"] = additional

    # 4. JPEG compression with platform-specific quality
    q_min, q_max = config.get("jpeg_quality_range", (75, 85))
    quality = random.randint(q_min, q_max)

    # Handle file size limits
    max_file_size = config.get("max_file_size")
    large_threshold = config.get("large_image_threshold")

    buffer = io.BytesIO()

    if max_file_size:
        data, actual_quality = compress_to_target_size(image, max_file_size, quality)
        quality = actual_quality
        buffer = io.BytesIO(data)
    else:
        image.save(buffer, format="JPEG", quality=quality, optimize=True)

    # Large image penalty (Twitter-style)
    if large_threshold:
        file_size = len(buffer.getvalue())
        if file_size > large_threshold:
            quality = max(60, quality - 15)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)

    metadata["jpeg_quality"] = quality
    metadata["file_size_bytes"] = len(buffer.getvalue())

    # 5. Re-open as if freshly downloaded
    buffer.seek(0)
    processed_image = Image.open(buffer).copy()

    # Strip EXIF if platform does so
    if config.get("strip_exif", True):
        # Create new image without metadata
        clean_image = Image.new("RGB", processed_image.size)
        clean_image.paste(processed_image)
        processed_image = clean_image
        metadata["exif_stripped"] = True

    metadata["final_size"] = processed_image.size

    return processed_image, metadata


def generate_social_media_dataset(
    input_dir: Path,
    output_dir: Path,
    platforms: List[str],
    num_per_platform: int = 1,
) -> Dict[str, Any]:
    """
    Generate social media processed dataset.

    Args:
        input_dir: Directory with original images
        output_dir: Output directory
        platforms: Platforms to simulate
        num_per_platform: Number of variations per platform

    Returns:
        Dataset metadata
    """
    ensure_dir(output_dir)
    ensure_dir(output_dir / "original")

    for platform in platforms:
        ensure_dir(output_dir / platform)

    # Find all images
    extensions = [".jpg", ".jpeg", ".png", ".webp"]
    images = []
    for ext in extensions:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))

    logger.info(f"Found {len(images)} images in {input_dir}")
    logger.info(f"Platforms: {platforms}")

    dataset_meta = {
        "timestamp": datetime.now().isoformat(),
        "source_dir": str(input_dir),
        "platforms": platforms,
        "samples": [],
    }

    for i, img_path in enumerate(images):
        logger.info(f"Processing {i+1}/{len(images)}: {img_path.name}")

        try:
            image = Image.open(img_path).convert("RGB")

            # Save original reference
            orig_name = f"orig_{i:05d}.jpg"
            image.save(output_dir / "original" / orig_name, quality=95)

            # Process for each platform
            for platform in platforms:
                if platform not in PLATFORM_CONFIGS:
                    logger.warning(f"Unknown platform: {platform}")
                    continue

                config = PLATFORM_CONFIGS[platform]

                for j in range(num_per_platform):
                    try:
                        processed, meta = simulate_platform_processing(
                            image.copy(), platform, config
                        )

                        # Save processed image
                        processed_name = f"{platform}_{i:05d}_{j:02d}.jpg"
                        processed.save(output_dir / platform / processed_name, quality=95)

                        # Record metadata
                        sample_meta = {
                            "original": orig_name,
                            "processed": processed_name,
                            "platform": platform,
                            "details": meta,
                        }
                        dataset_meta["samples"].append(sample_meta)

                    except Exception as e:
                        logger.warning(f"Failed to process {img_path} for {platform}: {e}")

        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")

    # Statistics
    dataset_meta["num_originals"] = len(images)
    dataset_meta["num_processed"] = len(dataset_meta["samples"])

    per_platform = {}
    for sample in dataset_meta["samples"]:
        p = sample["platform"]
        per_platform[p] = per_platform.get(p, 0) + 1
    dataset_meta["per_platform_counts"] = per_platform

    # Save metadata
    with open(output_dir / "dataset_metadata.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    logger.info(f"Generated {dataset_meta['num_processed']} processed images")

    return dataset_meta


def main():
    parser = argparse.ArgumentParser(
        description="Generate social media processed dataset for evaluation"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with original images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/social_media",
        help="Output directory",
    )
    parser.add_argument(
        "--platforms",
        type=str,
        default="all",
        help="Comma-separated platforms or 'all'",
    )
    parser.add_argument(
        "--num-per-platform",
        type=int,
        default=1,
        help="Number of variations per platform per image",
    )

    args = parser.parse_args()

    # Parse platforms
    if args.platforms == "all":
        platforms = list(PLATFORM_CONFIGS.keys())
    else:
        platforms = [p.strip() for p in args.platforms.split(",")]

    # Validate platforms
    valid_platforms = []
    for p in platforms:
        if p in PLATFORM_CONFIGS:
            valid_platforms.append(p)
        else:
            logger.warning(f"Unknown platform '{p}', skipping")

    if not valid_platforms:
        logger.error("No valid platforms specified")
        return

    # Generate dataset
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    logger.info(f"Generating social media dataset")
    logger.info(f"Platforms: {valid_platforms}")

    metadata = generate_social_media_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        platforms=valid_platforms,
        num_per_platform=args.num_per_platform,
    )

    print("\n" + "=" * 60)
    print("SOCIAL MEDIA DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Originals: {metadata['num_originals']}")
    print(f"Processed images: {metadata['num_processed']}")
    print(f"\nPer-platform counts:")
    for platform, count in metadata["per_platform_counts"].items():
        print(f"  {platform}: {count}")
    print(f"\nOutput: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
