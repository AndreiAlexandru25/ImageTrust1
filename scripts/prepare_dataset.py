#!/usr/bin/env python3
"""
GenImage Dataset Downloader & Organizer for Dissertation Benchmarking

Downloads a balanced subset of the GenImage dataset from HuggingFace and
organizes it into a flat directory structure ready for benchmark_tool.py.

Dataset: https://huggingface.co/datasets/poloclub/diffusiondb (GenImage alternative)
         https://huggingface.co/datasets/ILSVRC/imagenet-1k (for real images)

Structure Created:
    C:/Licenta/Data/Benchmark/
    ├── Real/          # ~1000 real photographs
    │   ├── image_0001.jpg
    │   ├── image_0002.jpg
    │   └── ...
    └── Fake/          # ~1000 AI-generated images
        ├── image_0001.png
        ├── image_0002.png
        └── ...

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --real-count 500 --fake-count 500
    python scripts/prepare_dataset.py --output-dir D:/MyData/Benchmark

Author: ImageTrust Research Team
License: MIT
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import random
import hashlib

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_OUTPUT_DIR = "C:/Licenta/Data/Benchmark"
DEFAULT_REAL_COUNT = 1000
DEFAULT_FAKE_COUNT = 1000

# HuggingFace dataset options for FAKE images (AI-generated)
# Using DiffusionDB which has high-quality Stable Diffusion images
FAKE_DATASETS = [
    {
        "repo_id": "poloclub/diffusiondb",
        "subset": "2m_first_1k",  # Small subset for testing
        "description": "DiffusionDB - Stable Diffusion generated images",
    },
]

# For REAL images, we'll use multiple sources
REAL_DATASETS = [
    {
        "repo_id": "imagenet-1k",  # Requires authentication
        "description": "ImageNet-1K real photographs",
    },
]

# Alternative: Download from URLs (more reliable, no auth required)
SAMPLE_REAL_URLS = [
    # Using Unsplash Source API (free, high-quality real photos)
    "https://picsum.photos/512/512",  # Lorem Picsum - random real photos
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_files(directory: Path, extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp')) -> List[Path]:
    """Recursively find all image files in a directory."""
    files = []
    if not directory.exists():
        return files

    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
        files.extend(directory.rglob(f"*{ext.upper()}"))

    return sorted(set(files))


def flatten_images(source_dir: Path, target_dir: Path, prefix: str, max_count: int) -> int:
    """
    Move images from nested directories to a flat structure.

    Args:
        source_dir: Directory containing nested image folders
        target_dir: Target flat directory
        prefix: Filename prefix (e.g., "real_" or "fake_")
        max_count: Maximum number of images to copy

    Returns:
        Number of images copied
    """
    ensure_dir(target_dir)

    # Find all images
    images = get_image_files(source_dir)
    random.shuffle(images)  # Randomize selection

    copied = 0
    for img_path in images:
        if copied >= max_count:
            break

        # Generate new filename
        ext = img_path.suffix.lower()
        new_name = f"{prefix}{copied + 1:05d}{ext}"
        target_path = target_dir / new_name

        try:
            shutil.copy2(img_path, target_path)
            copied += 1

            if copied % 100 == 0:
                print(f"  Copied {copied}/{max_count} images...")
        except Exception as e:
            print(f"  Warning: Could not copy {img_path.name}: {e}")

    return copied


def download_with_huggingface(
    repo_id: str,
    local_dir: Path,
    allow_patterns: Optional[List[str]] = None,
    max_files: int = 2000
) -> Path:
    """
    Download dataset from HuggingFace using huggingface_hub.

    Args:
        repo_id: HuggingFace dataset repository ID
        local_dir: Local directory to save files
        allow_patterns: Glob patterns to filter files
        max_files: Maximum number of files to download

    Returns:
        Path to downloaded files
    """
    try:
        from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files

        print(f"  Connecting to HuggingFace: {repo_id}")

        # Download snapshot with filters
        download_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=allow_patterns or ["*.jpg", "*.png", "*.jpeg"],
            ignore_patterns=["*.txt", "*.json", "*.md", "*.parquet"],
            max_workers=4,
        )

        return Path(download_path)

    except ImportError:
        print("  ERROR: huggingface_hub not installed. Installing...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        return download_with_huggingface(repo_id, local_dir, allow_patterns, max_files)
    except Exception as e:
        print(f"  ERROR downloading from HuggingFace: {e}")
        return None


def download_sample_images_from_urls(
    target_dir: Path,
    count: int,
    prefix: str,
    category: str = "nature"
) -> int:
    """
    Download sample images from free image APIs.
    This is a fallback when HuggingFace datasets require authentication.

    Args:
        target_dir: Directory to save images
        count: Number of images to download
        prefix: Filename prefix
        category: Image category for search

    Returns:
        Number of images downloaded
    """
    import urllib.request
    import time

    ensure_dir(target_dir)
    downloaded = 0

    print(f"  Downloading {count} images from Lorem Picsum (real photos)...")

    for i in range(count):
        try:
            # Lorem Picsum provides real photographs
            # Adding random seed to get different images
            url = f"https://picsum.photos/seed/{random.randint(1, 100000)}/512/512"

            filename = f"{prefix}{i + 1:05d}.jpg"
            filepath = target_dir / filename

            # Download with retry
            for attempt in range(3):
                try:
                    urllib.request.urlretrieve(url, filepath)
                    downloaded += 1
                    break
                except Exception:
                    time.sleep(0.5)

            if downloaded % 50 == 0:
                print(f"    Downloaded {downloaded}/{count}...")

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"    Warning: Failed to download image {i + 1}: {e}")

    return downloaded


def download_diffusiondb_subset(
    target_dir: Path,
    count: int
) -> int:
    """
    Download AI-generated images from DiffusionDB dataset.
    Uses the parquet files and extracts images.

    Args:
        target_dir: Directory to save images
        count: Number of images to download

    Returns:
        Number of images downloaded
    """
    try:
        from huggingface_hub import hf_hub_download
        import io
        from PIL import Image

        ensure_dir(target_dir)

        print(f"  Downloading DiffusionDB subset ({count} images)...")

        # DiffusionDB stores images in parquet format
        # We'll download a small subset

        # Try to use datasets library for easier access
        try:
            from datasets import load_dataset

            print("  Loading DiffusionDB dataset (this may take a moment)...")

            # Load a small streaming subset
            dataset = load_dataset(
                "poloclub/diffusiondb",
                "2m_random_1k",  # Small 1K subset
                split="train",
                streaming=True
            )

            downloaded = 0
            for i, item in enumerate(dataset):
                if downloaded >= count:
                    break

                try:
                    # Get image from dataset
                    img = item.get("image")
                    if img is None:
                        continue

                    # Save image
                    filename = f"fake_{downloaded + 1:05d}.png"
                    filepath = target_dir / filename
                    img.save(filepath)
                    downloaded += 1

                    if downloaded % 100 == 0:
                        print(f"    Saved {downloaded}/{count} images...")

                except Exception as e:
                    print(f"    Warning: Error processing image {i}: {e}")

            return downloaded

        except ImportError:
            print("  Installing datasets library...")
            os.system(f"{sys.executable} -m pip install datasets")
            return download_diffusiondb_subset(target_dir, count)

    except Exception as e:
        print(f"  ERROR: {e}")
        return 0


def download_civitai_samples(target_dir: Path, count: int) -> int:
    """
    Download AI-generated image samples from alternative sources.
    Uses publicly available AI art samples.
    """
    import urllib.request
    import time

    ensure_dir(target_dir)
    downloaded = 0

    print(f"  Downloading AI-generated samples ({count} images)...")

    # Using ThisPersonDoesNotExist style API or similar
    # These are GAN/Diffusion generated faces/images

    sources = [
        # StyleGAN generated faces (fake but realistic)
        ("https://thispersondoesnotexist.com/", "face"),
    ]

    for i in range(count):
        try:
            url = "https://thispersondoesnotexist.com/"

            filename = f"fake_{i + 1:05d}.jpg"
            filepath = target_dir / filename

            # Create request with headers (some sites block raw urllib)
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )

            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=10) as response:
                        with open(filepath, 'wb') as f:
                            f.write(response.read())
                    downloaded += 1
                    break
                except Exception:
                    time.sleep(1)

            if downloaded % 50 == 0:
                print(f"    Downloaded {downloaded}/{count}...")

            # Rate limiting (important for these APIs)
            time.sleep(0.3)

        except Exception as e:
            if downloaded % 100 == 0:
                print(f"    Note: Some downloads may fail, continuing... ({e})")

    return downloaded


# =============================================================================
# ALTERNATIVE: USE LOCAL EXISTING DATASETS
# =============================================================================

def find_existing_datasets() -> dict:
    """
    Search for existing image datasets on the system.
    Returns paths to found datasets.
    """
    common_paths = [
        Path("C:/Users") / os.getenv("USERNAME", "User") / "Downloads",
        Path("C:/Data"),
        Path("D:/Data"),
        Path("D:/Datasets"),
        Path.home() / "datasets",
    ]

    found = {"real": [], "fake": []}

    keywords_real = ["real", "authentic", "original", "imagenet", "coco", "photos"]
    keywords_fake = ["fake", "ai", "generated", "synthetic", "diffusion", "gan", "midjourney", "dalle"]

    for base_path in common_paths:
        if not base_path.exists():
            continue

        for subdir in base_path.iterdir():
            if not subdir.is_dir():
                continue

            name_lower = subdir.name.lower()

            # Check for real image indicators
            if any(kw in name_lower for kw in keywords_real):
                images = get_image_files(subdir)
                if len(images) > 10:
                    found["real"].append((subdir, len(images)))

            # Check for fake image indicators
            if any(kw in name_lower for kw in keywords_fake):
                images = get_image_files(subdir)
                if len(images) > 10:
                    found["fake"].append((subdir, len(images)))

    return found


# =============================================================================
# MAIN DOWNLOAD & ORGANIZE FUNCTION
# =============================================================================

def prepare_benchmark_dataset(
    output_dir: Path,
    real_count: int = 1000,
    fake_count: int = 1000,
    use_hf: bool = True,
    use_urls: bool = True
) -> Tuple[int, int]:
    """
    Main function to prepare the benchmark dataset.

    Args:
        output_dir: Base output directory
        real_count: Number of real images to collect
        fake_count: Number of fake images to collect
        use_hf: Try to use HuggingFace datasets
        use_urls: Fall back to URL downloads

    Returns:
        Tuple of (real_count, fake_count) actually downloaded
    """
    print("=" * 60)
    print("GenImage Benchmark Dataset Preparation")
    print("=" * 60)

    # Setup directories
    real_dir = ensure_dir(output_dir / "Real")
    fake_dir = ensure_dir(output_dir / "Fake")
    cache_dir = ensure_dir(output_dir / ".cache")

    print(f"\nOutput directory: {output_dir}")
    print(f"  Real images -> {real_dir}")
    print(f"  Fake images -> {fake_dir}")
    print(f"  Target: {real_count} real + {fake_count} fake images")

    total_real = 0
    total_fake = 0

    # =========================================================================
    # STEP 1: Check for existing local datasets
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 1: Checking for existing local datasets...")

    existing = find_existing_datasets()

    if existing["real"]:
        print(f"  Found {len(existing['real'])} potential REAL image sources:")
        for path, count in existing["real"][:3]:
            print(f"    - {path.name}: {count} images")

    if existing["fake"]:
        print(f"  Found {len(existing['fake'])} potential FAKE image sources:")
        for path, count in existing["fake"][:3]:
            print(f"    - {path.name}: {count} images")

    # Use existing if available
    for source_path, img_count in existing.get("real", []):
        if total_real >= real_count:
            break
        needed = real_count - total_real
        print(f"\n  Copying from {source_path.name}...")
        copied = flatten_images(source_path, real_dir, "real_", min(needed, img_count))
        total_real += copied
        print(f"  Copied {copied} real images (total: {total_real})")

    for source_path, img_count in existing.get("fake", []):
        if total_fake >= fake_count:
            break
        needed = fake_count - total_fake
        print(f"\n  Copying from {source_path.name}...")
        copied = flatten_images(source_path, fake_dir, "fake_", min(needed, img_count))
        total_fake += copied
        print(f"  Copied {copied} fake images (total: {total_fake})")

    # =========================================================================
    # STEP 2: Download from HuggingFace (if needed and enabled)
    # =========================================================================
    if use_hf and (total_real < real_count or total_fake < fake_count):
        print("\n" + "-" * 40)
        print("STEP 2: Downloading from HuggingFace...")

        # Download fake images from DiffusionDB
        if total_fake < fake_count:
            needed = fake_count - total_fake
            print(f"\n  Downloading {needed} AI-generated images from DiffusionDB...")

            downloaded = download_diffusiondb_subset(fake_dir, needed)
            total_fake += downloaded
            print(f"  Downloaded {downloaded} fake images (total: {total_fake})")

    # =========================================================================
    # STEP 3: Download from URLs (fallback)
    # =========================================================================
    if use_urls and (total_real < real_count or total_fake < fake_count):
        print("\n" + "-" * 40)
        print("STEP 3: Downloading from web APIs (fallback)...")

        # Download real images from Lorem Picsum
        if total_real < real_count:
            needed = real_count - total_real
            print(f"\n  Downloading {needed} real photographs...")
            downloaded = download_sample_images_from_urls(real_dir, needed, "real_")
            total_real += downloaded
            print(f"  Downloaded {downloaded} real images (total: {total_real})")

        # Download fake images from ThisPersonDoesNotExist
        if total_fake < fake_count:
            needed = fake_count - total_fake
            print(f"\n  Downloading {needed} AI-generated images...")
            downloaded = download_civitai_samples(fake_dir, needed)
            total_fake += downloaded
            print(f"  Downloaded {downloaded} fake images (total: {total_fake})")

    # =========================================================================
    # STEP 4: Verify and report
    # =========================================================================
    print("\n" + "-" * 40)
    print("STEP 4: Verification...")

    final_real = len(get_image_files(real_dir))
    final_fake = len(get_image_files(fake_dir))

    print(f"\n  Real images in {real_dir}: {final_real}")
    print(f"  Fake images in {fake_dir}: {final_fake}")

    # Calculate total size
    total_size = 0
    for img in get_image_files(real_dir):
        total_size += img.stat().st_size
    for img in get_image_files(fake_dir):
        total_size += img.stat().st_size

    print(f"  Total size: {total_size / (1024*1024):.1f} MB")

    # Cleanup cache
    if cache_dir.exists() and any(cache_dir.iterdir()):
        print(f"\n  Cleaning up cache directory...")
        try:
            shutil.rmtree(cache_dir)
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nYou can now run the benchmark with:")
    print(f"  python -m imagetrust.tools.benchmark_tool \\")
    print(f"      --real \"{real_dir}\" \\")
    print(f"      --fake \"{fake_dir}\" \\")
    print(f"      --output results/genimage_benchmark.csv")
    print()

    return final_real, final_fake


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download and organize GenImage dataset for benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default 1000+1000 images
  python prepare_dataset.py

  # Download smaller subset
  python prepare_dataset.py --real-count 200 --fake-count 200

  # Custom output directory
  python prepare_dataset.py --output-dir D:/Benchmark/Data

  # Skip HuggingFace (use only URL downloads)
  python prepare_dataset.py --no-huggingface
        """
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--real-count", "-r",
        type=int,
        default=DEFAULT_REAL_COUNT,
        help=f"Number of real images to download (default: {DEFAULT_REAL_COUNT})"
    )
    parser.add_argument(
        "--fake-count", "-f",
        type=int,
        default=DEFAULT_FAKE_COUNT,
        help=f"Number of fake images to download (default: {DEFAULT_FAKE_COUNT})"
    )
    parser.add_argument(
        "--no-huggingface",
        action="store_true",
        help="Skip HuggingFace datasets (use only URL downloads)"
    )
    parser.add_argument(
        "--no-urls",
        action="store_true",
        help="Skip URL downloads (use only HuggingFace)"
    )

    args = parser.parse_args()

    # Run preparation
    real_count, fake_count = prepare_benchmark_dataset(
        output_dir=args.output_dir,
        real_count=args.real_count,
        fake_count=args.fake_count,
        use_hf=not args.no_huggingface,
        use_urls=not args.no_urls
    )

    # Exit with error if we didn't get enough images
    if real_count < args.real_count // 2 or fake_count < args.fake_count // 2:
        print("\nWARNING: Could not download enough images!")
        print("Try running with different options or check your internet connection.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
