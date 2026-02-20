#!/usr/bin/env python
"""
Resource-Aware Synthetic Data Generation Orchestrator.

Generates synthetic robustness data (Instagram, WhatsApp, Screenshots)
with CPU resource control to allow concurrent PC usage.

Features:
- Configurable CPU core usage (default: 6/8 cores for 7800X3D)
- Multiprocessing with controlled worker pool
- Progress tracking with tqdm
- Checkpoint/resume support
- Memory-efficient batch processing

Usage:
    python scripts/orchestrator/run_synthetic_generation.py \
        --input_dir data/train \
        --output_dir data/synthetic \
        --num_workers 6 \
        --batch_size 100

For low-priority execution on Windows:
    start /LOW /BELOWNORMAL python scripts/orchestrator/run_synthetic_generation.py ...

For Linux:
    nice -n 10 python scripts/orchestrator/run_synthetic_generation.py ...
"""

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation."""

    input_dir: Path
    output_dir: Path
    checkpoint_dir: Path

    # CPU resource control
    num_workers: int = 6  # Use 6/8 cores on 7800X3D
    batch_size: int = 100  # Images per batch for progress tracking

    # Generation settings
    platforms: List[str] = field(default_factory=lambda: ["instagram", "whatsapp", "twitter"])
    include_screenshots: bool = True
    screenshot_types: List[str] = field(default_factory=lambda: ["windows", "macos", "mobile"])

    # Augmentation settings
    num_variants_per_platform: int = 1
    compression_rounds: int = 2

    # Memory management
    gc_interval: int = 500  # Run garbage collection every N images


@dataclass
class GenerationProgress:
    """Tracks generation progress for checkpointing."""

    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    completed_platforms: List[str] = field(default_factory=list)
    completed_screenshot_types: List[str] = field(default_factory=list)
    processed_files: set = field(default_factory=set)
    start_time: Optional[str] = None
    last_checkpoint: Optional[str] = None
    estimated_completion: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "total_images": self.total_images,
            "processed_images": self.processed_images,
            "failed_images": self.failed_images,
            "completed_platforms": self.completed_platforms,
            "completed_screenshot_types": self.completed_screenshot_types,
            "processed_files": list(self.processed_files),
            "start_time": self.start_time,
            "last_checkpoint": self.last_checkpoint,
            "estimated_completion": self.estimated_completion,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GenerationProgress":
        progress = cls()
        progress.total_images = data.get("total_images", 0)
        progress.processed_images = data.get("processed_images", 0)
        progress.failed_images = data.get("failed_images", 0)
        progress.completed_platforms = data.get("completed_platforms", [])
        progress.completed_screenshot_types = data.get("completed_screenshot_types", [])
        progress.processed_files = set(data.get("processed_files", []))
        progress.start_time = data.get("start_time")
        progress.last_checkpoint = data.get("last_checkpoint")
        progress.estimated_completion = data.get("estimated_completion")
        return progress


def get_image_id(image_path: Path) -> str:
    """Generate unique ID for an image based on path."""
    return hashlib.md5(str(image_path).encode()).hexdigest()[:16]


def find_all_images(
    input_dir: Path,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
) -> List[Path]:
    """Find all image files in directory recursively."""
    images = []
    for ext in extensions:
        images.extend(input_dir.rglob(f"*{ext}"))
        images.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def process_single_image_social_media(args: Tuple) -> Dict[str, Any]:
    """
    Process a single image for social media simulation.

    Worker function for multiprocessing pool.
    """
    from PIL import Image

    image_path, output_dir, platform, variant_idx, compression_rounds = args

    try:
        # Import here to avoid pickling issues
        from imagetrust.detection.augmentation import Platform, SocialMediaSimulator

        platform_enum = Platform(platform)
        simulator = SocialMediaSimulator(
            platforms=[platform_enum],
            compression_rounds=compression_rounds,
        )

        # Load and process image
        image = Image.open(image_path).convert("RGB")
        degraded, metadata = simulator.simulate(image, platform=platform_enum)

        # Create output path
        platform_dir = output_dir / "social_media" / platform
        platform_dir.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem
        suffix = f"_v{variant_idx}" if variant_idx > 0 else ""
        output_path = platform_dir / f"{stem}{suffix}.jpg"

        # Save
        degraded.save(output_path, "JPEG", quality=95)

        # Clean up
        del image, degraded
        gc.collect()

        return {
            "success": True,
            "input_path": str(image_path),
            "output_path": str(output_path),
            "platform": platform,
            "metadata": metadata,
        }

    except Exception as e:
        return {
            "success": False,
            "input_path": str(image_path),
            "error": str(e),
            "platform": platform,
        }


def process_single_image_screenshot(args: Tuple) -> Dict[str, Any]:
    """
    Process a single image for screenshot simulation.

    Worker function for multiprocessing pool.
    """
    from PIL import Image

    image_path, output_dir, screenshot_type, variant_idx = args

    try:
        # Import here to avoid pickling issues
        from imagetrust.detection.augmentation import ScreenshotSimulator, ScreenshotType

        type_enum = ScreenshotType(screenshot_type)
        simulator = ScreenshotSimulator(
            screenshot_types=[type_enum],
            add_ui_elements=True,
            add_borders=True,
            add_text_overlays=True,
        )

        # Load and process image
        image = Image.open(image_path).convert("RGB")
        screenshot, metadata = simulator.simulate(image, screenshot_type=type_enum)

        # Create output path
        type_dir = output_dir / "screenshots" / screenshot_type
        type_dir.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem
        suffix = f"_v{variant_idx}" if variant_idx > 0 else ""
        output_path = type_dir / f"{stem}{suffix}.png"

        # Save as PNG
        screenshot.save(output_path, "PNG")

        # Clean up
        del image, screenshot
        gc.collect()

        return {
            "success": True,
            "input_path": str(image_path),
            "output_path": str(output_path),
            "screenshot_type": screenshot_type,
            "metadata": metadata,
        }

    except Exception as e:
        return {
            "success": False,
            "input_path": str(image_path),
            "error": str(e),
            "screenshot_type": screenshot_type,
        }


class SyntheticGenerationOrchestrator:
    """
    Orchestrates synthetic data generation with resource control.

    Features:
    - Multiprocessing with controlled worker pool
    - Checkpoint/resume support
    - Progress tracking
    - Memory management
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.progress = GenerationProgress()
        self.manifest_entries: List[Dict] = []

        # Create directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint file
        self.checkpoint_file = self.config.checkpoint_dir / "generation_checkpoint.json"

    def load_checkpoint(self) -> bool:
        """Load progress from checkpoint if exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    data = json.load(f)
                self.progress = GenerationProgress.from_dict(data["progress"])
                self.manifest_entries = data.get("manifest_entries", [])
                print(f"Resumed from checkpoint: {self.progress.processed_images} images processed")
                return True
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
        return False

    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        self.progress.last_checkpoint = datetime.now().isoformat()

        # Estimate completion time
        if self.progress.processed_images > 0 and self.progress.start_time:
            start = datetime.fromisoformat(self.progress.start_time)
            elapsed = (datetime.now() - start).total_seconds()
            rate = self.progress.processed_images / elapsed
            remaining = self.progress.total_images - self.progress.processed_images
            if rate > 0:
                eta_seconds = remaining / rate
                eta = datetime.now() + timedelta(seconds=eta_seconds)
                self.progress.estimated_completion = eta.isoformat()

        data = {
            "progress": self.progress.to_dict(),
            "manifest_entries": self.manifest_entries,
        }

        # Atomic write
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.checkpoint_file)

    def generate_social_media_variants(self, images: List[Path]) -> List[Dict]:
        """Generate social media variants for all images."""
        results = []

        for platform in self.config.platforms:
            if platform in self.progress.completed_platforms:
                print(f"Skipping {platform} (already completed)")
                continue

            print(f"\n{'='*60}")
            print(f"Generating {platform.upper()} variants...")
            print(f"{'='*60}")

            # Build work items, skipping already processed
            work_items = []
            for image_path in images:
                image_id = f"{get_image_id(image_path)}_{platform}"
                if image_id in self.progress.processed_files:
                    continue

                for variant_idx in range(self.config.num_variants_per_platform):
                    work_items.append((
                        image_path,
                        self.config.output_dir,
                        platform,
                        variant_idx,
                        self.config.compression_rounds,
                    ))

            if not work_items:
                print(f"  All {platform} images already processed")
                self.progress.completed_platforms.append(platform)
                continue

            print(f"  Processing {len(work_items)} images with {self.config.num_workers} workers...")

            # Process with multiprocessing pool
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = {executor.submit(process_single_image_social_media, item): item for item in work_items}

                pbar = tqdm(
                    as_completed(futures),
                    total=len(work_items),
                    desc=f"  {platform}",
                    unit="img",
                    ncols=100,
                )

                batch_count = 0
                for future in pbar:
                    result = future.result()
                    results.append(result)

                    if result["success"]:
                        self.progress.processed_images += 1
                        image_id = f"{get_image_id(Path(result['input_path']))}_{platform}"
                        self.progress.processed_files.add(image_id)

                        self.manifest_entries.append({
                            "path": result["output_path"],
                            "original_path": result["input_path"],
                            "augmentation_type": "social_media",
                            "platform": platform,
                            "label": self._infer_label(Path(result["input_path"])),
                        })
                    else:
                        self.progress.failed_images += 1
                        tqdm.write(f"    Failed: {result.get('error', 'Unknown error')}")

                    batch_count += 1

                    # Checkpoint and GC periodically
                    if batch_count % self.config.gc_interval == 0:
                        self.save_checkpoint()
                        gc.collect()

                    # Update progress bar postfix
                    pbar.set_postfix({
                        "done": self.progress.processed_images,
                        "failed": self.progress.failed_images,
                    })

            self.progress.completed_platforms.append(platform)
            self.save_checkpoint()

        return results

    def generate_screenshot_variants(self, images: List[Path]) -> List[Dict]:
        """Generate screenshot variants for all images."""
        if not self.config.include_screenshots:
            return []

        results = []

        for ss_type in self.config.screenshot_types:
            if ss_type in self.progress.completed_screenshot_types:
                print(f"Skipping {ss_type} screenshots (already completed)")
                continue

            print(f"\n{'='*60}")
            print(f"Generating {ss_type.upper()} screenshot variants...")
            print(f"{'='*60}")

            # Build work items
            work_items = []
            for image_path in images:
                image_id = f"{get_image_id(image_path)}_screenshot_{ss_type}"
                if image_id in self.progress.processed_files:
                    continue

                for variant_idx in range(self.config.num_variants_per_platform):
                    work_items.append((
                        image_path,
                        self.config.output_dir,
                        ss_type,
                        variant_idx,
                    ))

            if not work_items:
                print(f"  All {ss_type} screenshots already processed")
                self.progress.completed_screenshot_types.append(ss_type)
                continue

            print(f"  Processing {len(work_items)} images with {self.config.num_workers} workers...")

            # Process with multiprocessing pool
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = {executor.submit(process_single_image_screenshot, item): item for item in work_items}

                pbar = tqdm(
                    as_completed(futures),
                    total=len(work_items),
                    desc=f"  {ss_type}",
                    unit="img",
                    ncols=100,
                )

                batch_count = 0
                for future in pbar:
                    result = future.result()
                    results.append(result)

                    if result["success"]:
                        self.progress.processed_images += 1
                        image_id = f"{get_image_id(Path(result['input_path']))}_screenshot_{ss_type}"
                        self.progress.processed_files.add(image_id)

                        self.manifest_entries.append({
                            "path": result["output_path"],
                            "original_path": result["input_path"],
                            "augmentation_type": "screenshot",
                            "screenshot_type": ss_type,
                            "label": self._infer_label(Path(result["input_path"])),
                        })
                    else:
                        self.progress.failed_images += 1

                    batch_count += 1

                    if batch_count % self.config.gc_interval == 0:
                        self.save_checkpoint()
                        gc.collect()

                    pbar.set_postfix({
                        "done": self.progress.processed_images,
                        "failed": self.progress.failed_images,
                    })

            self.progress.completed_screenshot_types.append(ss_type)
            self.save_checkpoint()

        return results

    def _infer_label(self, image_path: Path) -> int:
        """Infer label from image path."""
        path_parts = [p.lower() for p in image_path.parts]
        ai_indicators = ["ai", "fake", "generated", "synthetic", "deepfake", "midjourney", "dalle", "sd"]
        for indicator in ai_indicators:
            if any(indicator in part for part in path_parts):
                return 1
        return 0

    def run(self) -> Dict[str, Any]:
        """Run the complete synthetic generation pipeline."""
        print("\n" + "=" * 70)
        print("  IMAGETRUST v2.0 - SYNTHETIC DATA GENERATION ORCHESTRATOR")
        print("=" * 70)

        # Load checkpoint if available
        resumed = self.load_checkpoint()

        # Find all source images
        print(f"\nScanning for images in: {self.config.input_dir}")
        images = find_all_images(self.config.input_dir)
        print(f"Found {len(images)} source images")

        if not resumed:
            # Calculate total work
            num_platforms = len(self.config.platforms)
            num_screenshot_types = len(self.config.screenshot_types) if self.config.include_screenshots else 0
            total_variants = len(images) * (num_platforms + num_screenshot_types) * self.config.num_variants_per_platform
            self.progress.total_images = total_variants
            self.progress.start_time = datetime.now().isoformat()

        print(f"\nConfiguration:")
        print(f"  - Workers: {self.config.num_workers} / {mp.cpu_count()} available cores")
        print(f"  - Platforms: {', '.join(self.config.platforms)}")
        print(f"  - Screenshots: {self.config.include_screenshots}")
        print(f"  - Total synthetic images to generate: ~{self.progress.total_images:,}")
        print(f"  - Already processed: {self.progress.processed_images:,}")
        print(f"  - Checkpoint dir: {self.config.checkpoint_dir}")

        start_time = time.time()

        # Generate social media variants
        self.generate_social_media_variants(images)

        # Generate screenshot variants
        self.generate_screenshot_variants(images)

        elapsed = time.time() - start_time

        # Save final manifest
        manifest_path = self.config.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.manifest_entries, f, indent=2)

        # Final summary
        print("\n" + "=" * 70)
        print("  GENERATION COMPLETE")
        print("=" * 70)
        print(f"  Total processed: {self.progress.processed_images:,}")
        print(f"  Failed: {self.progress.failed_images:,}")
        print(f"  Time elapsed: {elapsed / 3600:.1f} hours")
        print(f"  Rate: {self.progress.processed_images / elapsed:.1f} images/sec")
        print(f"  Manifest saved to: {manifest_path}")

        # Clean up checkpoint on success
        if self.progress.failed_images == 0:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                print("  Checkpoint cleaned up (success)")

        return {
            "total_processed": self.progress.processed_images,
            "failed": self.progress.failed_images,
            "elapsed_seconds": elapsed,
            "manifest_path": str(manifest_path),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Resource-aware synthetic data generation for ImageTrust v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/synthetic_generation"),
        help="Directory for checkpoint files",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="Number of CPU workers (default: 6 for 8-core CPU)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for progress tracking",
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        default=["instagram", "whatsapp", "twitter"],
        help="Social media platforms to simulate",
    )
    parser.add_argument(
        "--no_screenshots",
        action="store_true",
        help="Skip screenshot generation",
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
        "--gc_interval",
        type=int,
        default=500,
        help="Run garbage collection every N images",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Create config
    config = GenerationConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        platforms=args.platforms,
        include_screenshots=not args.no_screenshots,
        screenshot_types=args.screenshot_types,
        num_variants_per_platform=args.num_variants,
        compression_rounds=args.compression_rounds,
        gc_interval=args.gc_interval,
    )

    # Run orchestrator
    orchestrator = SyntheticGenerationOrchestrator(config)
    result = orchestrator.run()

    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
