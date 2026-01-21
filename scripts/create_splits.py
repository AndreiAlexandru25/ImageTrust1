#!/usr/bin/env python3
"""
Create deterministic dataset splits for ImageTrust evaluation.

This script generates reproducible train/val/test splits for:
1. In-domain evaluation (standard 70/15/15 split)
2. Cross-generator evaluation (leave-one-out)
3. Cross-dataset evaluation (train on A, test on B)

Usage:
    # Create all splits from raw data directory
    python scripts/create_splits.py --data-dir ./data/raw --output-dir ./data/splits

    # Create specific split type
    python scripts/create_splits.py --split-type in-domain --data-dir ./data/raw

    # Create cross-generator split for specific generator
    python scripts/create_splits.py --split-type cross-generator --held-out midjourney

    # Dry run to see what would be created
    python scripts/create_splits.py --data-dir ./data/raw --dry-run

Author: ImageTrust Team
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from imagetrust.data.dataset import (
    DatasetManager,
    create_manifest_from_directory,
    save_manifest,
)
from imagetrust.data.splits import (
    SplitConfig,
    create_default_split,
    create_cross_generator_splits,
    create_cross_dataset_split,
    create_all_splits,
    save_split,
    get_evaluation_matrix,
)
from imagetrust.data.generators import (
    GENERATOR_IDS,
    GENERATOR_DISPLAY_NAMES,
    get_generator_id,
)


def discover_datasets(data_dir: Path, verbose: bool = True) -> DatasetManager:
    """
    Discover datasets from directory structure.

    Expected structure:
        data_dir/
        ├── real/
        │   ├── coco/
        │   ├── imagenet/
        │   └── ...
        └── ai_generated/
            ├── midjourney/
            ├── dalle3/
            └── ...
    """
    manager = DatasetManager()

    real_dir = data_dir / "real"
    ai_dir = data_dir / "ai_generated"

    # Discover real image sources
    if real_dir.exists():
        for source_dir in real_dir.iterdir():
            if source_dir.is_dir():
                name = f"real_{source_dir.name}"
                if verbose:
                    print(f"  Found real dataset: {source_dir.name}")
                manager.add_from_directory(
                    directory=source_dir,
                    name=name,
                    generator="real",
                    description=f"Real images from {source_dir.name}",
                )

    # Discover AI-generated image sources
    if ai_dir.exists():
        for gen_dir in ai_dir.iterdir():
            if gen_dir.is_dir():
                generator_id = get_generator_id(gen_dir.name)
                name = f"ai_{gen_dir.name}"
                if verbose:
                    print(f"  Found AI dataset: {gen_dir.name} -> {generator_id}")
                manager.add_from_directory(
                    directory=gen_dir,
                    name=name,
                    generator=generator_id,
                    description=f"AI images from {gen_dir.name}",
                )

    return manager


def create_manifests(data_dir: Path, output_dir: Path, verbose: bool = True) -> None:
    """Create manifest.json files for all discovered datasets."""
    manager = discover_datasets(data_dir, verbose=verbose)

    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for name, manifest in manager.manifests.items():
        manifest_path = manifest_dir / f"{name}_manifest.json"
        save_manifest(manifest, manifest_path)
        if verbose:
            print(f"  Saved: {manifest_path}")


def create_in_domain_split(
    manager: DatasetManager,
    output_dir: Path,
    config: SplitConfig,
    verbose: bool = True,
) -> Path:
    """Create standard in-domain split."""
    all_images = manager.get_all_images()

    if verbose:
        print(f"\nCreating in-domain split:")
        print(f"  Total images: {len(all_images)}")
        print(f"  Ratios: {config.train_ratio:.0%}/{config.val_ratio:.0%}/{config.test_ratio:.0%}")
        print(f"  Seed: {config.seed}")

    split = create_default_split(all_images, config, name="default_split")

    output_path = output_dir / "default_split.json"
    save_split(split, output_path)

    if verbose:
        print(f"  Train: {len(split.train)} images")
        print(f"  Val: {len(split.val)} images")
        print(f"  Test: {len(split.test)} images")
        print(f"  Saved: {output_path}")

    return output_path


def create_cross_gen_splits(
    manager: DatasetManager,
    output_dir: Path,
    config: SplitConfig,
    held_out: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Create leave-one-generator-out splits."""
    all_images = manager.get_all_images()

    cross_gen_dir = output_dir / "cross_generator"
    cross_gen_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nCreating cross-generator splits:")

    splits = create_cross_generator_splits(all_images, held_out, config)
    paths = {}

    for gen_id, split in splits.items():
        output_path = cross_gen_dir / f"leave_{gen_id}.json"
        save_split(split, output_path)
        paths[gen_id] = output_path

        if verbose:
            display_name = GENERATOR_DISPLAY_NAMES.get(gen_id, gen_id)
            print(f"  {display_name}: train={len(split.train)}, test={len(split.test)}")

    if verbose:
        print(f"  Saved {len(splits)} splits to: {cross_gen_dir}")

    return paths


def print_evaluation_matrix():
    """Print the evaluation matrix for paper."""
    matrix = get_evaluation_matrix()

    print("\n" + "=" * 60)
    print("EVALUATION MATRIX")
    print("=" * 60)

    for eval_type, config in matrix.items():
        print(f"\n{eval_type.upper()}")
        print("-" * 40)
        print(f"  Description: {config['description']}")
        print(f"  Metrics: {', '.join(config['metrics'])}")

        if eval_type == "cross_generator":
            print(f"  Generators: {', '.join(config['generators'])}")

        if eval_type == "degradation":
            print("  Degradations:")
            for deg_type, params in config["degradations"].items():
                for param_name, values in params.items():
                    print(f"    - {deg_type}: {param_name}={values}")


def print_summary(manager: DatasetManager, output_dir: Path):
    """Print summary of created splits."""
    print("\n" + "=" * 60)
    print("SPLIT CREATION SUMMARY")
    print("=" * 60)

    print("\nDataset Statistics:")
    print(manager.summary())

    print(f"\nOutput directory: {output_dir}")
    print("\nCreated files:")

    for split_file in output_dir.rglob("*.json"):
        rel_path = split_file.relative_to(output_dir)
        print(f"  - {rel_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic dataset splits for ImageTrust",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create all splits
  python scripts/create_splits.py --data-dir ./data/raw --output-dir ./data/splits

  # Create only cross-generator splits
  python scripts/create_splits.py --split-type cross-generator --data-dir ./data/raw

  # Specify held-out generators
  python scripts/create_splits.py --split-type cross-generator --held-out midjourney dalle3

  # Custom split ratios
  python scripts/create_splits.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing raw data (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Output directory for split files (default: data/splits)",
    )
    parser.add_argument(
        "--split-type",
        choices=["all", "in-domain", "cross-generator", "manifests"],
        default="all",
        help="Type of split to create (default: all)",
    )
    parser.add_argument(
        "--held-out",
        nargs="+",
        help="Generators to hold out for cross-generator splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training set ratio (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified splitting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without writing files",
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        help="Show evaluation matrix and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    # Show evaluation matrix and exit
    if args.show_matrix:
        print_evaluation_matrix()
        return 0

    # Validate data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print("\nExpected structure:")
        print("  data/raw/")
        print("  ├── real/")
        print("  │   ├── coco/")
        print("  │   └── imagenet/")
        print("  └── ai_generated/")
        print("      ├── midjourney/")
        print("      ├── dalle3/")
        print("      └── sdxl/")
        return 1

    # Create split config
    config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by_generator=not args.no_stratify,
        stratify_by_label=not args.no_stratify,
    )

    if verbose:
        print("=" * 60)
        print("ImageTrust Dataset Split Creator")
        print("=" * 60)
        print(f"\nData directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Split type: {args.split_type}")
        print(f"Seed: {config.seed}")

    # Discover datasets
    if verbose:
        print("\nDiscovering datasets...")

    manager = discover_datasets(args.data_dir, verbose=verbose)

    if len(manager.manifests) == 0:
        print("Error: No datasets found in data directory")
        return 1

    if args.dry_run:
        print("\n[DRY RUN] Would create the following:")
        print(manager.summary())
        print_evaluation_matrix()
        return 0

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create splits based on type
    if args.split_type in ["all", "manifests"]:
        if verbose:
            print("\nCreating manifests...")
        create_manifests(args.data_dir, args.output_dir, verbose=verbose)

    if args.split_type in ["all", "in-domain"]:
        create_in_domain_split(manager, args.output_dir, config, verbose=verbose)

    if args.split_type in ["all", "cross-generator"]:
        create_cross_gen_splits(
            manager, args.output_dir, config,
            held_out=args.held_out,
            verbose=verbose,
        )

    # Print summary
    if verbose:
        print_summary(manager, args.output_dir)
        print_evaluation_matrix()

    return 0


if __name__ == "__main__":
    sys.exit(main())
