#!/usr/bin/env python
"""
High-Speed GPU-Optimized Embedding Factory.

Extracts embeddings from all images using multiple backbones (ResNet-50,
EfficientNet-B0, ViT-B/16) with integrated NIQE quality scoring.

Features:
- Large batch sizes (256) for GPU throughput
- Mixed precision (AMP) for speed and memory efficiency
- GPU memory management (85% limit for responsiveness)
- Checkpoint/resume support
- Compressed .npz output format
- Integrated NIQE quality scoring

Hardware Target: RTX 5080 (16GB VRAM)

Usage:
    python scripts/orchestrator/run_embedding_extraction.py \
        --input_dir data/all_images \
        --output_dir data/embeddings \
        --batch_size 256 \
        --gpu_memory_fraction 0.85

For low-priority execution on Windows:
    start /LOW /BELOWNORMAL python scripts/orchestrator/run_embedding_extraction.py ...

For Linux:
    nice -n 10 python scripts/orchestrator/run_embedding_extraction.py ...
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@dataclass
class ExtractionConfig:
    """Configuration for embedding extraction."""

    input_dirs: List[Path]  # Can include multiple directories
    output_dir: Path
    checkpoint_dir: Path

    # GPU settings
    batch_size: int = 256
    gpu_memory_fraction: float = 0.85
    use_amp: bool = True
    num_workers: int = 8  # DataLoader workers

    # Backbones to use
    backbones: List[str] = field(default_factory=lambda: ["resnet50", "efficientnet_b0", "vit_b_16"])

    # Quality assessment
    compute_niqe: bool = True

    # Checkpointing
    checkpoint_interval: int = 1000  # Save every N images
    shard_size: int = 10000  # Images per .npz file

    # Processing
    input_size: int = 224
    prefetch_factor: int = 4

    # Delay between batches (ms) for system responsiveness
    batch_delay_ms: int = 10


@dataclass
class ExtractionProgress:
    """Tracks extraction progress for checkpointing."""

    total_images: int = 0
    processed_images: int = 0
    failed_images: int = 0
    current_shard: int = 0
    processed_files: set = field(default_factory=set)
    start_time: Optional[str] = None
    last_checkpoint: Optional[str] = None
    backbone_status: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "total_images": self.total_images,
            "processed_images": self.processed_images,
            "failed_images": self.failed_images,
            "current_shard": self.current_shard,
            "processed_files": list(self.processed_files),
            "start_time": self.start_time,
            "last_checkpoint": self.last_checkpoint,
            "backbone_status": self.backbone_status,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExtractionProgress":
        progress = cls()
        progress.total_images = data.get("total_images", 0)
        progress.processed_images = data.get("processed_images", 0)
        progress.failed_images = data.get("failed_images", 0)
        progress.current_shard = data.get("current_shard", 0)
        progress.processed_files = set(data.get("processed_files", []))
        progress.start_time = data.get("start_time")
        progress.last_checkpoint = data.get("last_checkpoint")
        progress.backbone_status = data.get("backbone_status", {})
        return progress


def get_image_id(image_path: Path) -> str:
    """Generate unique ID for an image."""
    return hashlib.md5(str(image_path).encode()).hexdigest()[:16]


def find_all_images(
    input_dirs: List[Path],
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
) -> List[Path]:
    """Find all image files in directories."""
    images = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        for ext in extensions:
            images.extend(input_dir.rglob(f"*{ext}"))
            images.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


class ImageDataset(Dataset):
    """Dataset for efficient image loading."""

    def __init__(
        self,
        image_paths: List[Path],
        transform: transforms.Compose,
        return_raw: bool = False,  # For NIQE computation
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.return_raw = return_raw

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_paths[idx]
        image_id = get_image_id(image_path)

        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image)

            result = {
                "tensor": tensor,
                "image_id": image_id,
                "path": str(image_path),
                "success": True,
            }

            if self.return_raw:
                # Return numpy array for NIQE computation
                result["raw"] = np.array(image)

            return result

        except Exception as e:
            # Return zeros on failure
            return {
                "tensor": torch.zeros(3, 224, 224),
                "image_id": image_id,
                "path": str(image_path),
                "success": False,
                "error": str(e),
            }


class BackboneEmbedder:
    """
    Efficient backbone embedding extractor with AMP support.

    Supports: ResNet-50, EfficientNet-B0, ViT-B/16
    """

    BACKBONE_INFO = {
        "resnet50": {"embed_dim": 2048, "module": "torchvision.models"},
        "efficientnet_b0": {"embed_dim": 1280, "module": "torchvision.models"},
        "vit_b_16": {"embed_dim": 768, "module": "torchvision.models"},
    }

    def __init__(
        self,
        backbone_name: str,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        self.backbone_name = backbone_name
        self.device = device
        self.use_amp = use_amp

        self.model = self._load_backbone()
        self.model.to(device)
        self.model.eval()

        self.embed_dim = self.BACKBONE_INFO[backbone_name]["embed_dim"]

        # Register hook for embedding extraction
        self._embeddings = None
        self._register_hook()

        print(f"  Loaded {backbone_name}: embed_dim={self.embed_dim}")

    def _load_backbone(self) -> nn.Module:
        """Load pretrained backbone model."""
        import torchvision.models as models

        if self.backbone_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif self.backbone_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif self.backbone_name == "vit_b_16":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

        return model

    def _register_hook(self):
        """Register forward hook on the penultimate layer."""

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                if output.dim() > 2:
                    # Global average pooling
                    self._embeddings = output.mean(dim=[2, 3]) if output.dim() == 4 else output[:, 0]
                else:
                    self._embeddings = output

        # Find target layer based on architecture
        if self.backbone_name == "resnet50":
            self.model.avgpool.register_forward_hook(hook_fn)
        elif self.backbone_name == "efficientnet_b0":
            self.model.avgpool.register_forward_hook(hook_fn)
        elif self.backbone_name == "vit_b_16":
            # ViT: use the output of the encoder
            self.model.encoder.register_forward_hook(
                lambda m, i, o: hook_fn(m, i, o[:, 0])  # CLS token
            )

    @torch.no_grad()
    def extract(self, batch: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from a batch of images."""
        batch = batch.to(self.device)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                _ = self.model(batch)
        else:
            _ = self.model(batch)

        embeddings = self._embeddings.clone()
        self._embeddings = None

        return embeddings.cpu()


class NIQEComputer:
    """
    Fast NIQE quality scoring.

    Computes NIQE score for each image in the batch.
    Lower scores indicate better quality.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._piq_available = False

        try:
            import piq
            self._niqe = piq.NIQE().to(device)
            self._piq_available = True
            print("  NIQE: Using PIQ library (GPU accelerated)")
        except ImportError:
            print("  NIQE: Using fallback implementation (CPU)")

    def compute_batch(self, images: torch.Tensor) -> np.ndarray:
        """Compute NIQE scores for a batch of images."""
        if self._piq_available:
            return self._compute_piq(images)
        else:
            return self._compute_fallback(images)

    def _compute_piq(self, images: torch.Tensor) -> np.ndarray:
        """Compute NIQE using PIQ library."""
        images = images.to(self.device)

        # NIQE expects values in [0, 1]
        if images.max() > 1.0:
            images = images / 255.0

        scores = []
        with torch.no_grad():
            for i in range(images.shape[0]):
                try:
                    score = self._niqe(images[i:i+1]).item()
                    scores.append(score)
                except Exception:
                    scores.append(50.0)  # Default high score for failures

        return np.array(scores, dtype=np.float32)

    def _compute_fallback(self, images: torch.Tensor) -> np.ndarray:
        """Fallback NIQE approximation."""
        from scipy.ndimage import gaussian_filter

        scores = []

        for i in range(images.shape[0]):
            try:
                img = images[i].numpy()
                if img.shape[0] == 3:  # CHW -> HWC
                    img = np.transpose(img, (1, 2, 0))

                # Convert to grayscale
                gray = np.mean(img, axis=2)

                # MSCN computation
                mu = gaussian_filter(gray, sigma=7/6)
                sigma = np.sqrt(gaussian_filter((gray - mu) ** 2, sigma=7/6))
                sigma = np.maximum(sigma, 1e-6)
                mscn = (gray - mu) / sigma

                # Simple NIQE approximation
                niqe_approx = np.abs(np.mean(mscn)) + np.abs(1 - np.var(mscn))
                scores.append(float(niqe_approx * 10))
            except Exception:
                scores.append(50.0)

        return np.array(scores, dtype=np.float32)


class EmbeddingFactory:
    """
    High-speed embedding factory with GPU optimization.

    Features:
    - Multi-backbone extraction
    - Mixed precision (AMP)
    - Memory management
    - Checkpoint/resume
    - Compressed .npz output
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set GPU memory fraction
        if self.device == "cuda":
            self._setup_gpu_memory()

        # Initialize progress
        self.progress = ExtractionProgress()
        self.checkpoint_file = config.checkpoint_dir / "extraction_checkpoint.json"

        # Create output directories
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize backbones
        self.backbones: Dict[str, BackboneEmbedder] = {}

        # Initialize NIQE computer
        self.niqe_computer = None
        if config.compute_niqe:
            self.niqe_computer = NIQEComputer(self.device)

        # Storage for current shard
        self.current_shard_data: Dict[str, Dict] = {}

        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _setup_gpu_memory(self):
        """Configure GPU memory usage."""
        try:
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            target_memory = int(total_memory * self.config.gpu_memory_fraction)

            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)

            print(f"\nGPU Configuration:")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Total Memory: {total_memory / 1e9:.1f} GB")
            print(f"  Allocated Limit: {target_memory / 1e9:.1f} GB ({self.config.gpu_memory_fraction*100:.0f}%)")
            print(f"  AMP (Mixed Precision): {self.config.use_amp}")

        except Exception as e:
            print(f"Warning: Could not configure GPU memory: {e}")

    def load_checkpoint(self) -> bool:
        """Load progress from checkpoint."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    data = json.load(f)
                self.progress = ExtractionProgress.from_dict(data["progress"])
                print(f"Resumed from checkpoint: {self.progress.processed_images} images processed")
                return True
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
        return False

    def save_checkpoint(self):
        """Save current progress to checkpoint."""
        self.progress.last_checkpoint = datetime.now().isoformat()

        data = {
            "progress": self.progress.to_dict(),
        }

        temp_file = self.checkpoint_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self.checkpoint_file)

    def _load_backbones(self):
        """Load all backbone models."""
        print("\nLoading backbone models...")

        for backbone_name in self.config.backbones:
            if backbone_name not in self.backbones:
                self.backbones[backbone_name] = BackboneEmbedder(
                    backbone_name,
                    device=self.device,
                    use_amp=self.config.use_amp,
                )
                self.progress.backbone_status[backbone_name] = "loaded"

    def _save_shard(self, shard_idx: int):
        """Save current shard to compressed .npz file."""
        if not self.current_shard_data:
            return

        shard_path = self.config.output_dir / f"embeddings_shard_{shard_idx:05d}.npz"

        # Prepare arrays
        image_ids = []
        paths = []
        labels = []

        # Embedding arrays per backbone
        embeddings = {name: [] for name in self.config.backbones}

        # Quality scores
        niqe_scores = []

        for image_id, data in self.current_shard_data.items():
            image_ids.append(image_id)
            paths.append(data["path"])
            labels.append(data.get("label", -1))

            for backbone_name in self.config.backbones:
                embeddings[backbone_name].append(data["embeddings"].get(backbone_name, np.zeros(2048)))

            niqe_scores.append(data.get("niqe_score", 50.0))

        # Create save dict
        save_dict = {
            "image_ids": np.array(image_ids),
            "paths": np.array(paths),
            "labels": np.array(labels, dtype=np.int32),
            "niqe_scores": np.array(niqe_scores, dtype=np.float32),
        }

        for backbone_name, emb_list in embeddings.items():
            save_dict[f"embeddings_{backbone_name}"] = np.stack(emb_list).astype(np.float16)  # Half precision to save space

        # Save compressed
        np.savez_compressed(shard_path, **save_dict)
        print(f"  Saved shard {shard_idx}: {len(image_ids)} images -> {shard_path.name}")

        # Clear shard data
        self.current_shard_data.clear()
        gc.collect()

    def _infer_label(self, image_path: Path) -> int:
        """Infer label from image path."""
        path_parts = [p.lower() for p in image_path.parts]
        ai_indicators = ["ai", "fake", "generated", "synthetic", "deepfake", "midjourney", "dalle", "sd"]
        for indicator in ai_indicators:
            if any(indicator in part for part in path_parts):
                return 1
        return 0

    def extract_all(self) -> Dict[str, Any]:
        """Extract embeddings from all images."""
        print("\n" + "=" * 70)
        print("  IMAGETRUST v2.0 - EMBEDDING FACTORY")
        print("=" * 70)

        # Load checkpoint
        resumed = self.load_checkpoint()

        # Find all images
        print(f"\nScanning for images in: {self.config.input_dirs}")
        all_images = find_all_images(self.config.input_dirs)
        print(f"Found {len(all_images)} total images")

        # Filter out already processed
        if resumed:
            images_to_process = [
                img for img in all_images
                if get_image_id(img) not in self.progress.processed_files
            ]
            print(f"Remaining to process: {len(images_to_process)}")
        else:
            images_to_process = all_images
            self.progress.total_images = len(all_images)
            self.progress.start_time = datetime.now().isoformat()

        if not images_to_process:
            print("All images already processed!")
            return {"status": "complete", "total": self.progress.processed_images}

        # Load backbones
        self._load_backbones()

        # Create dataset and dataloader
        dataset = ImageDataset(
            images_to_process,
            self.transform,
            return_raw=self.config.compute_niqe,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=True,
        )

        print(f"\nExtraction Configuration:")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  DataLoader workers: {self.config.num_workers}")
        print(f"  Backbones: {', '.join(self.config.backbones)}")
        print(f"  NIQE scoring: {self.config.compute_niqe}")
        print(f"  Shard size: {self.config.shard_size}")
        print(f"  Batch delay: {self.config.batch_delay_ms}ms")

        start_time = time.time()
        images_in_current_shard = 0

        # Main extraction loop
        pbar = tqdm(
            dataloader,
            total=len(dataloader),
            desc="Extracting",
            unit="batch",
            ncols=120,
        )

        for batch_data in pbar:
            batch_tensors = batch_data["tensor"]
            batch_ids = batch_data["image_id"]
            batch_paths = batch_data["path"]
            batch_success = batch_data["success"]

            # Extract embeddings from each backbone
            batch_embeddings = {}
            for backbone_name, backbone in self.backbones.items():
                embeddings = backbone.extract(batch_tensors)
                batch_embeddings[backbone_name] = embeddings.numpy()

            # Compute NIQE scores
            batch_niqe = None
            if self.config.compute_niqe and self.niqe_computer:
                # Use raw images if available, otherwise use tensors
                if "raw" in batch_data:
                    raw_batch = torch.stack([
                        torch.from_numpy(r).permute(2, 0, 1).float() / 255.0
                        for r in batch_data["raw"]
                    ])
                else:
                    raw_batch = batch_tensors
                batch_niqe = self.niqe_computer.compute_batch(raw_batch)

            # Store results
            for i in range(len(batch_ids)):
                if not batch_success[i]:
                    self.progress.failed_images += 1
                    continue

                image_id = batch_ids[i]
                image_path = Path(batch_paths[i])

                self.current_shard_data[image_id] = {
                    "path": str(image_path),
                    "label": self._infer_label(image_path),
                    "embeddings": {
                        name: batch_embeddings[name][i]
                        for name in self.config.backbones
                    },
                    "niqe_score": float(batch_niqe[i]) if batch_niqe is not None else 50.0,
                }

                self.progress.processed_images += 1
                self.progress.processed_files.add(image_id)
                images_in_current_shard += 1

            # Save shard if full
            if images_in_current_shard >= self.config.shard_size:
                self._save_shard(self.progress.current_shard)
                self.progress.current_shard += 1
                images_in_current_shard = 0
                self.save_checkpoint()

            # Checkpoint periodically
            if self.progress.processed_images % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

            # Update progress bar
            elapsed = time.time() - start_time
            rate = self.progress.processed_images / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                "done": f"{self.progress.processed_images:,}",
                "shard": self.progress.current_shard,
                "rate": f"{rate:.1f}/s",
            })

            # Small delay for system responsiveness
            if self.config.batch_delay_ms > 0:
                time.sleep(self.config.batch_delay_ms / 1000.0)

        # Save final shard
        if self.current_shard_data:
            self._save_shard(self.progress.current_shard)

        elapsed = time.time() - start_time

        # Create index file
        index_path = self.config.output_dir / "embedding_index.json"
        index_data = {
            "total_images": self.progress.processed_images,
            "total_shards": self.progress.current_shard + 1,
            "backbones": self.config.backbones,
            "embed_dims": {name: self.backbones[name].embed_dim for name in self.backbones},
            "has_niqe": self.config.compute_niqe,
            "shard_files": [
                f"embeddings_shard_{i:05d}.npz"
                for i in range(self.progress.current_shard + 1)
            ],
            "created_at": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
        }

        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        # Final summary
        print("\n" + "=" * 70)
        print("  EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"  Total processed: {self.progress.processed_images:,}")
        print(f"  Failed: {self.progress.failed_images:,}")
        print(f"  Time elapsed: {elapsed / 3600:.2f} hours")
        print(f"  Rate: {self.progress.processed_images / elapsed:.1f} images/sec")
        print(f"  Shards created: {self.progress.current_shard + 1}")
        print(f"  Index file: {index_path}")

        # Estimate storage
        total_size = sum(
            f.stat().st_size
            for f in self.config.output_dir.glob("*.npz")
        )
        print(f"  Total storage: {total_size / 1e9:.2f} GB")

        # Clean up checkpoint on success
        if self.progress.failed_images == 0 and self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("  Checkpoint cleaned up (success)")

        return {
            "total_processed": self.progress.processed_images,
            "failed": self.progress.failed_images,
            "elapsed_seconds": elapsed,
            "shards": self.progress.current_shard + 1,
        }


def main():
    parser = argparse.ArgumentParser(
        description="GPU-optimized embedding extraction for ImageTrust v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input_dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Input directories containing images (can specify multiple)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/embeddings"),
        help="Output directory for embedding files",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints/embedding_extraction"),
        help="Directory for checkpoint files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for GPU processing (default: 256 for RTX 5080)",
    )
    parser.add_argument(
        "--gpu_memory_fraction",
        type=float,
        default=0.85,
        help="Fraction of GPU memory to use (default: 0.85)",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable mixed precision (AMP)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers (default: 8)",
    )
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["resnet50", "efficientnet_b0", "vit_b_16"],
        help="Backbone models to use",
    )
    parser.add_argument(
        "--no_niqe",
        action="store_true",
        help="Skip NIQE quality scoring",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="Save checkpoint every N images",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10000,
        help="Images per .npz shard file",
    )
    parser.add_argument(
        "--batch_delay_ms",
        type=int,
        default=10,
        help="Delay between batches in ms (for system responsiveness)",
    )

    args = parser.parse_args()

    # Validate inputs
    for input_dir in args.input_dirs:
        if not input_dir.exists():
            print(f"Warning: Input directory does not exist: {input_dir}")

    # Create config
    config = ExtractionConfig(
        input_dirs=args.input_dirs,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        gpu_memory_fraction=args.gpu_memory_fraction,
        use_amp=not args.no_amp,
        num_workers=args.num_workers,
        backbones=args.backbones,
        compute_niqe=not args.no_niqe,
        checkpoint_interval=args.checkpoint_interval,
        shard_size=args.shard_size,
        batch_delay_ms=args.batch_delay_ms,
    )

    # Run factory
    factory = EmbeddingFactory(config)
    result = factory.extract_all()

    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
