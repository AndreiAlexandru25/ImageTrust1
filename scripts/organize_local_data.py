#!/usr/bin/env python3
"""
================================================================================
DATASET ORGANIZER FOR LOCAL TRAINING — ImageTrust
================================================================================

Extracts ZIP archives from datasets/ and organizes images into the canonical
structure expected by create_splits.py and train_kaggle_deepfake.py.

Target structure:
    data/
    ├── raw/
    │   ├── real/
    │   │   ├── coco/
    │   │   ├── ffhq/
    │   │   ├── cifake_real/
    │   │   └── other_real/
    │   └── ai_generated/
    │       ├── cifake_sd/
    │       ├── sfhq/
    │       ├── deepfake/
    │       ├── deepfake_faces/
    │       ├── fake_faces/
    │       └── hard_fakes/
    └── train/
        ├── Real/   ← flat folder (for train_kaggle_deepfake.py)
        └── Fake/   ← flat folder

Usage:
    python scripts/organize_local_data.py
    python scripts/organize_local_data.py --datasets-dir ./datasets --data-dir ./data
    python scripts/organize_local_data.py --extract-only
    python scripts/organize_local_data.py --skip-extract
    python scripts/organize_local_data.py --dry-run

Author: ImageTrust Team
================================================================================
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Image extensions to look for
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif",
}


# =============================================================================
# DATASET DEFINITIONS
# =============================================================================

# Each dataset defines:
#   zip_pattern: glob pattern to find the ZIP in datasets/
#   priority: extraction order (lower = first)
#   category: "mixed" (has both real/fake), "real", or "ai_generated"
#   mappings: list of (source_subpath_pattern, target_folder, label)
#       source_subpath_pattern: relative path inside extracted dir to search
#       target_folder: destination under data/raw/{real|ai_generated}/
#       label: "real" or "fake"

DATASET_CONFIGS = {
    "cifake": {
        "zip_pattern": "CIFAKE*",
        "priority": 1,
        "description": "CIFAKE Real and AI-Generated Synthetic Images",
        "mappings": [
            {"search_dirs": ["train/REAL", "test/REAL", "REAL", "real"],
             "target": "cifake_real", "label": "real"},
            {"search_dirs": ["train/FAKE", "test/FAKE", "FAKE", "fake"],
             "target": "cifake_sd", "label": "fake"},
        ],
    },
    "deepfake_real_images": {
        "zip_pattern": "deepfake and real images*",
        "priority": 2,
        "description": "Deepfake and Real Images dataset",
        "mappings": [
            {"search_dirs": ["real", "Real", "real_images", "real-images"],
             "target": "other_real", "label": "real"},
            {"search_dirs": ["fake", "Fake", "deepfake", "fake_images", "fake-images"],
             "target": "deepfake", "label": "fake"},
        ],
    },
    "real_fake_face_detection": {
        "zip_pattern": "Real and Fake Face Detection*",
        "priority": 3,
        "description": "Real and Fake Face Detection dataset",
        "mappings": [
            {"search_dirs": ["real_and_fake_face/training_real",
                             "real_and_fake_face/training_fake",
                             "training_real", "real", "Real"],
             "target": "other_real", "label": "real",
             "filter_dirs": ["real", "Real", "training_real"]},
            {"search_dirs": ["real_and_fake_face/training_fake",
                             "training_fake", "fake", "Fake"],
             "target": "fake_faces", "label": "fake"},
        ],
    },
    "fake_vs_real_hard": {
        "zip_pattern": "Fake-Vs-Real-Faces*",
        "priority": 4,
        "description": "Fake vs Real Faces (Hard) dataset",
        "mappings": [
            {"search_dirs": ["real", "Real"], "target": "other_real", "label": "real"},
            {"search_dirs": ["fake", "Fake"], "target": "hard_fakes", "label": "fake"},
        ],
    },
    "coco": {
        "zip_pattern": "COCO*",
        "priority": 5,
        "description": "COCO 2017 dataset (real photos)",
        "mappings": [
            {"search_dirs": ["train2017", "val2017", "test2017", "."],
             "target": "coco", "label": "real"},
        ],
    },
    "ffhq": {
        "zip_pattern": "Flickr-Faces*",
        "priority": 6,
        "description": "Flickr Faces HQ (FFHQ) dataset (real faces)",
        "mappings": [
            {"search_dirs": [".", "images", "thumbnails128x128", "images1024x1024"],
             "target": "ffhq", "label": "real"},
        ],
    },
    "sfhq": {
        "zip_pattern": "Synthetic Faces*",
        "priority": 7,
        "description": "Synthetic Faces High Quality (SFHQ) dataset",
        "mappings": [
            {"search_dirs": [".", "images", "aligned", "raw"],
             "target": "sfhq", "label": "fake"},
        ],
    },
    "deepfake_faces": {
        "zip_pattern": "deepfake_faces*",
        "priority": 8,
        "description": "Deepfake Faces dataset",
        "mappings": [
            {"search_dirs": ["real", "Real", "original"],
             "target": "other_real", "label": "real"},
            {"search_dirs": ["fake", "Fake", "deepfake", "manipulated"],
             "target": "deepfake_faces", "label": "fake"},
        ],
    },
}

# ZIPs to skip
SKIP_ZIPS = [
    "artifacts*",
    "Anime Face*",
    "Training_Results*",
]


def find_7zip() -> str:
    """Find 7-Zip executable."""
    candidates = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
        "7z",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    # Try running 7z from PATH
    try:
        result = subprocess.run(["7z"], capture_output=True, timeout=5)
        return "7z"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def find_zip_file(datasets_dir: Path, pattern: str) -> Optional[Path]:
    """Find a ZIP file matching the pattern in datasets_dir."""
    import fnmatch
    for f in datasets_dir.iterdir():
        if f.suffix.lower() == ".zip" and fnmatch.fnmatch(f.name, pattern + ".zip"):
            return f
        if f.suffix.lower() == ".zip" and fnmatch.fnmatch(f.stem, pattern):
            return f
    # Broader match
    for f in datasets_dir.iterdir():
        if f.suffix.lower() == ".zip" and fnmatch.fnmatch(f.name.lower(), pattern.lower() + "*"):
            return f
    return None


def extract_zip(
    zip_path: Path,
    extract_dir: Path,
    seven_zip: str,
    dry_run: bool = False,
) -> bool:
    """Extract a ZIP archive using 7-Zip or Python zipfile."""
    target = extract_dir / zip_path.stem
    if target.exists() and any(target.iterdir()):
        print(f"    Already extracted: {target.name}")
        return True

    if dry_run:
        print(f"    [DRY RUN] Would extract to: {target}")
        return True

    target.mkdir(parents=True, exist_ok=True)

    size_gb = zip_path.stat().st_size / (1024 ** 3)
    print(f"    Extracting {zip_path.name} ({size_gb:.1f} GB)...")

    if seven_zip:
        # Use 7-Zip for large files (faster, handles edge cases)
        cmd = [seven_zip, "x", str(zip_path), f"-o{str(target)}", "-y", "-bso0", "-bsp1"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            if result.returncode == 0:
                print(f"    Extracted OK: {target.name}")
                return True
            else:
                print(f"    7-Zip error: {result.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"    7-Zip timeout (>2h)")
    else:
        # Fallback: Python zipfile (slower but always available)
        import zipfile
        try:
            with zipfile.ZipFile(str(zip_path), 'r') as zf:
                zf.extractall(str(target))
            print(f"    Extracted OK: {target.name}")
            return True
        except Exception as e:
            print(f"    zipfile error: {e}")

    return False


def count_images(directory: Path) -> int:
    """Count image files in directory (recursive)."""
    count = 0
    if not directory.exists():
        return 0
    for f in directory.rglob("*"):
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
            count += 1
    return count


def find_image_dir(base: Path, search_dirs: List[str]) -> Optional[Path]:
    """Find the first existing directory with images from search_dirs list."""
    # Try each search dir pattern
    for sd in search_dirs:
        if sd == ".":
            candidate = base
        else:
            candidate = base / sd

        if candidate.exists() and candidate.is_dir():
            n = count_images(candidate)
            if n > 0:
                return candidate

    # If nothing found, try case-insensitive search
    if base.exists():
        subdirs = {d.name.lower(): d for d in base.iterdir() if d.is_dir()}
        for sd in search_dirs:
            if sd == ".":
                continue
            key = sd.lower().replace("/", os.sep).split(os.sep)[0]
            if key in subdirs:
                candidate = subdirs[key]
                # Check for nested path
                parts = sd.split("/")
                if len(parts) > 1:
                    for part in parts[1:]:
                        sub = {d.name.lower(): d for d in candidate.iterdir() if d.is_dir()}
                        if part.lower() in sub:
                            candidate = sub[part.lower()]
                n = count_images(candidate)
                if n > 0:
                    return candidate

    # Last resort: check if base itself has images
    if base.exists() and count_images(base) > 0:
        return base

    return None


def copy_images(
    src_dir: Path,
    dst_dir: Path,
    max_images: int = 0,
    dry_run: bool = False,
) -> int:
    """Copy image files from src to dst (flat copy, no subdirectory structure)."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images
    images = []
    for f in src_dir.rglob("*"):
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
            images.append(f)

    if max_images > 0 and len(images) > max_images:
        # Deterministic sample
        import random
        rng = random.Random(42)
        rng.shuffle(images)
        images = images[:max_images]

    if dry_run:
        print(f"      [DRY RUN] Would copy {len(images)} images to {dst_dir.name}")
        return len(images)

    copied = 0
    # Use prefix to avoid name collisions across sources
    prefix = src_dir.name + "_"
    for img in images:
        dst_name = prefix + img.name
        dst_path = dst_dir / dst_name
        # Handle duplicates
        if dst_path.exists():
            stem = dst_path.stem
            suffix = dst_path.suffix
            counter = 1
            while dst_path.exists():
                dst_path = dst_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        try:
            shutil.copy2(str(img), str(dst_path))
            copied += 1
        except (OSError, shutil.SameFileError) as e:
            pass  # Skip problematic files silently

        if copied % 5000 == 0 and copied > 0:
            print(f"      ... copied {copied}/{len(images)}")

    return copied


def create_train_structure(
    raw_dir: Path,
    train_dir: Path,
    dry_run: bool = False,
    max_per_class: int = 0,
) -> Tuple[int, int]:
    """
    Create flat train/Real and train/Fake from raw/ structure.
    Uses symlinks on Linux, copies on Windows.
    """
    real_dir = raw_dir / "real"
    fake_dir = raw_dir / "ai_generated"
    train_real = train_dir / "Real"
    train_fake = train_dir / "Fake"

    if dry_run:
        n_real = count_images(real_dir) if real_dir.exists() else 0
        n_fake = count_images(fake_dir) if fake_dir.exists() else 0
        print(f"\n  [DRY RUN] Would create train/Real ({n_real}) and train/Fake ({n_fake})")
        return n_real, n_fake

    train_real.mkdir(parents=True, exist_ok=True)
    train_fake.mkdir(parents=True, exist_ok=True)

    print("\n  Creating train/Real and train/Fake (flat copy)...")

    n_real = 0
    if real_dir.exists():
        for subdir in sorted(real_dir.iterdir()):
            if subdir.is_dir():
                n = copy_images(subdir, train_real, max_images=max_per_class)
                print(f"    Real/{subdir.name}: {n} images")
                n_real += n

    n_fake = 0
    if fake_dir.exists():
        for subdir in sorted(fake_dir.iterdir()):
            if subdir.is_dir():
                n = copy_images(subdir, train_fake, max_images=max_per_class)
                print(f"    Fake/{subdir.name}: {n} images")
                n_fake += n

    return n_real, n_fake


def verify_sample(raw_dir: Path, n_samples: int = 50) -> Tuple[int, int]:
    """Verify a sample of images can be opened with PIL."""
    try:
        from PIL import Image
    except ImportError:
        print("  PIL not available, skipping verification")
        return 0, 0

    import random
    rng = random.Random(42)

    all_images = []
    for f in raw_dir.rglob("*"):
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
            all_images.append(f)

    if not all_images:
        return 0, 0

    sample = rng.sample(all_images, min(n_samples, len(all_images)))

    ok = 0
    bad = 0
    for img_path in sample:
        try:
            img = Image.open(str(img_path))
            img.verify()
            ok += 1
        except Exception:
            bad += 1
            print(f"    Corrupt: {img_path}")

    return ok, bad


def print_stats(raw_dir: Path, train_dir: Path):
    """Print dataset statistics."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    total_real = 0
    total_fake = 0

    real_dir = raw_dir / "real"
    fake_dir = raw_dir / "ai_generated"

    if real_dir.exists():
        print("\n  REAL images (data/raw/real/):")
        for subdir in sorted(real_dir.iterdir()):
            if subdir.is_dir():
                n = count_images(subdir)
                print(f"    {subdir.name:30s} {n:>8,} images")
                total_real += n

    if fake_dir.exists():
        print("\n  AI-GENERATED images (data/raw/ai_generated/):")
        for subdir in sorted(fake_dir.iterdir()):
            if subdir.is_dir():
                n = count_images(subdir)
                print(f"    {subdir.name:30s} {n:>8,} images")
                total_fake += n

    print(f"\n  {'TOTAL REAL':30s} {total_real:>8,}")
    print(f"  {'TOTAL FAKE':30s} {total_fake:>8,}")
    print(f"  {'TOTAL':30s} {total_real + total_fake:>8,}")

    if total_real > 0 and total_fake > 0:
        ratio = total_fake / total_real
        print(f"\n  Balance ratio (fake/real): {ratio:.2f}")
        if ratio < 0.5 or ratio > 2.0:
            print("  WARNING: Dataset is significantly imbalanced!")

    # Train structure stats
    train_real = train_dir / "Real"
    train_fake = train_dir / "Fake"
    if train_real.exists() or train_fake.exists():
        n_tr = count_images(train_real) if train_real.exists() else 0
        n_tf = count_images(train_fake) if train_fake.exists() else 0
        print(f"\n  TRAIN STRUCTURE:")
        print(f"    train/Real:  {n_tr:>8,}")
        print(f"    train/Fake:  {n_tf:>8,}")


def main():
    parser = argparse.ArgumentParser(
        description="Organize local datasets for ImageTrust training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets-dir", type=Path, default=Path("datasets"),
        help="Directory containing ZIP archives (default: datasets/)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Output data directory (default: data/)",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Only extract ZIPs, don't organize",
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip extraction, only organize from already-extracted data",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip creating train/Real and train/Fake flat structure",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=0,
        help="Max images per source when creating train/ (0 = unlimited)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify sample of images with PIL after organizing",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=True,
        help="Verbose output",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ImageTrust Dataset Organizer")
    print("=" * 70)
    print(f"\n  Datasets dir : {args.datasets_dir.resolve()}")
    print(f"  Data dir     : {args.data_dir.resolve()}")

    # Find 7-Zip
    seven_zip = find_7zip()
    if seven_zip:
        print(f"  7-Zip        : {seven_zip}")
    else:
        print("  7-Zip        : NOT FOUND (will use Python zipfile, slower)")

    # Check datasets dir
    if not args.datasets_dir.exists():
        print(f"\nERROR: Datasets directory not found: {args.datasets_dir}")
        return 1

    # List available ZIPs
    zips = sorted([f for f in args.datasets_dir.iterdir() if f.suffix.lower() == ".zip"])
    print(f"\n  Found {len(zips)} ZIP archives:")
    for z in zips:
        size_gb = z.stat().st_size / (1024 ** 3)
        print(f"    {z.name:55s} {size_gb:>6.1f} GB")

    # Create directories
    extract_dir = args.data_dir / "extracted"
    raw_dir = args.data_dir / "raw"
    real_dir = raw_dir / "real"
    fake_dir = raw_dir / "ai_generated"
    train_dir = args.data_dir / "train"

    if not args.dry_run:
        for d in [extract_dir, real_dir, fake_dir, train_dir]:
            d.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # =========================================================================
    # STEP 1: Extract ZIPs
    # =========================================================================
    if not args.skip_extract:
        print("\n" + "-" * 70)
        print("STEP 1: EXTRACTING ARCHIVES")
        print("-" * 70)

        # Sort by priority
        sorted_configs = sorted(DATASET_CONFIGS.items(), key=lambda x: x[1]["priority"])

        for name, config in sorted_configs:
            zip_file = find_zip_file(args.datasets_dir, config["zip_pattern"])
            if zip_file is None:
                print(f"\n  [{name}] ZIP not found (pattern: {config['zip_pattern']})")
                continue

            print(f"\n  [{name}] {config['description']}")
            print(f"    ZIP: {zip_file.name}")

            extract_zip(zip_file, extract_dir, seven_zip, dry_run=args.dry_run)

    if args.extract_only:
        print("\n  --extract-only: stopping after extraction.")
        elapsed = time.time() - start_time
        print(f"\n  Elapsed: {elapsed / 60:.1f} minutes")
        return 0

    # =========================================================================
    # STEP 2: Organize into raw/real/ and raw/ai_generated/
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: ORGANIZING INTO CANONICAL STRUCTURE")
    print("-" * 70)

    sorted_configs = sorted(DATASET_CONFIGS.items(), key=lambda x: x[1]["priority"])
    stats = defaultdict(int)

    for name, config in sorted_configs:
        # Find extracted directory
        zip_file = find_zip_file(args.datasets_dir, config["zip_pattern"])
        if zip_file is None:
            continue

        extracted_base = extract_dir / zip_file.stem

        # Also check for nested extraction (some ZIPs extract into a subfolder)
        if not extracted_base.exists():
            # Try finding any directory in extract_dir matching the name
            candidates = [d for d in extract_dir.iterdir()
                         if d.is_dir() and zip_file.stem.lower().startswith(d.name.lower()[:10])]
            if candidates:
                extracted_base = candidates[0]
            else:
                print(f"\n  [{name}] Extracted dir not found, skipping")
                continue

        print(f"\n  [{name}] Processing {config['description']}...")
        print(f"    Source: {extracted_base}")

        for mapping in config["mappings"]:
            src = find_image_dir(extracted_base, mapping["search_dirs"])
            if src is None:
                # Try one level deeper (some ZIPs nest in a folder)
                for subdir in extracted_base.iterdir():
                    if subdir.is_dir():
                        src = find_image_dir(subdir, mapping["search_dirs"])
                        if src:
                            break

            if src is None:
                print(f"    {mapping['label']:5s} -> {mapping['target']:20s} : NOT FOUND")
                continue

            if mapping["label"] == "real":
                dst = real_dir / mapping["target"]
            else:
                dst = fake_dir / mapping["target"]

            n = copy_images(src, dst, dry_run=args.dry_run)
            label_key = f"real" if mapping["label"] == "real" else "fake"
            stats[label_key] += n
            print(f"    {mapping['label']:5s} -> {mapping['target']:20s} : {n:>8,} images")

    # =========================================================================
    # STEP 3: Create train/Real + train/Fake flat structure
    # =========================================================================
    if not args.skip_train:
        print("\n" + "-" * 70)
        print("STEP 3: CREATING TRAIN STRUCTURE")
        print("-" * 70)

        n_real, n_fake = create_train_structure(
            raw_dir, train_dir,
            dry_run=args.dry_run,
            max_per_class=args.max_per_class,
        )

    # =========================================================================
    # STEP 4: Verify sample
    # =========================================================================
    if args.verify and not args.dry_run:
        print("\n" + "-" * 70)
        print("STEP 4: VERIFYING SAMPLE IMAGES")
        print("-" * 70)
        ok, bad = verify_sample(raw_dir)
        print(f"  Verified: {ok} OK, {bad} corrupt out of sample")

    # =========================================================================
    # Print final statistics
    # =========================================================================
    if not args.dry_run:
        print_stats(raw_dir, train_dir)

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed / 60:.1f} minutes")
    print("\nDone! Next steps:")
    print("  1. python scripts/create_splits.py --data-dir ./data/raw --output-dir ./data/splits")
    print("  2. python scripts/train_kaggle_deepfake.py --config configs/rtx5080_resnet50.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
