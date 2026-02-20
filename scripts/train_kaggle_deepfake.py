#!/usr/bin/env python3
"""
================================================================================
PROFESSIONAL DEEPFAKE DETECTION TRAINING PIPELINE FOR KAGGLE
================================================================================
Senior ML Engineer Production-Grade Implementation

Target: Kaggle GPU T4 x2
Dataset: datasets.rar (auto-extracted from /kaggle/input)

Advanced Features:
- Multi-backbone support (ResNet50, EfficientNetV2-M, ConvNeXt-Base)
- Exponential Moving Average (EMA) for stable predictions
- Stochastic Weight Averaging (SWA) for better generalization
- Mixup & CutMix augmentation for regularization
- Label Smoothing for calibration
- RandAugment (13 automated augmentations)
- Cosine Annealing with Warm Restarts
- Gradient Accumulation for larger effective batch size
- Class Balancing (weighted sampler + weighted loss)
- Early Stopping with patience
- Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC-ROC, AP)
- TensorBoard logging
- Automatic Mixed Precision (AMP) for speed
- Multi-GPU with DataParallel
- Robust checkpointing with resume capability
- Graceful disk quota handling

Author: Senior ML Engineer
Version: 2.0.0
================================================================================

USAGE ON KAGGLE:
    1. Create a new Kaggle notebook
    2. Add your dataset (datasets.rar) as input
    3. Enable GPU T4 x2 in notebook settings
    4. Copy-paste this entire script into a cell
    5. Run the cell

The script will automatically:
    - Find and extract datasets.rar
    - Handle disk quota issues gracefully
    - Find the correct folder structure (Real/Fake)
    - Train with all advanced techniques
    - Save best_model.pth to /kaggle/working
================================================================================
"""

# =============================================================================
# CELL 1: SYSTEM SETUP & DEPENDENCIES
# =============================================================================
import subprocess
import sys
import os

# Guard: only print in main process (not in DataLoader workers)
_IS_MAIN_PROCESS = os.environ.get("_IMAGETRUST_WORKER") is None
if _IS_MAIN_PROCESS:
    os.environ["_IMAGETRUST_WORKER"] = "1"

# Install required system packages (Linux/Kaggle only)
if _IS_MAIN_PROCESS:
    print("=" * 70)
    print("CHECKING SYSTEM DEPENDENCIES")
    print("=" * 70)

    if sys.platform != "win32":
        subprocess.run(["apt-get", "update", "-qq"], check=False, capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "unrar"], check=False, capture_output=True)
        print("unrar installed")
    else:
        print("Windows detected")

# =============================================================================
# IMPORTS
# =============================================================================
import gc
import glob
import shutil
import time
import random
import math
import json
import warnings
import copy
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from collections import defaultdict, OrderedDict
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import (
    DataLoader, Dataset, WeightedRandomSampler,
    Subset, ConcatDataset, random_split
)
from torch.cuda.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    if _IS_MAIN_PROCESS:
        print("TensorBoard not available, logging to console only")

import torchvision
from torchvision import datasets, transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet101, ResNet101_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    convnext_large, ConvNeXt_Large_Weights,
)

# Sklearn for metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, roc_curve
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Print versions (main process only)
if _IS_MAIN_PROCESS:
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Production-grade training configuration.
    All hyperparameters tuned for deepfake detection.
    """

    # ==========================================================================
    # PATHS
    # ==========================================================================
    # Auto-detect: Kaggle vs local Windows
    input_dir: str = (
        "/kaggle/input" if os.path.exists("/kaggle/input")
        else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    )
    extract_dir: str = (
        "/kaggle/temp/extracted_data" if os.path.exists("/kaggle")
        else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "extracted")
    )
    output_dir: str = (
        "/kaggle/working" if os.path.exists("/kaggle")
        else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "training")
    )
    rar_filename: str = "datasets.rar"
    experiment_name: str = "deepfake_detector_v2"

    # ==========================================================================
    # MODEL ARCHITECTURE
    # ==========================================================================
    backbone: str = "resnet50"  # Options: resnet50, resnet101, efficientnet_v2_m,
                                 #          efficientnet_v2_l, convnext_base, convnext_large
    num_classes: int = 2
    pretrained: bool = True
    dropout_rate: float = 0.5
    use_attention: bool = True

    # ==========================================================================
    # TRAINING HYPERPARAMETERS
    # ==========================================================================
    num_epochs: int = 30
    batch_size: int = 32              # Per GPU (effective = batch_size * num_gpus * grad_accum)
    gradient_accumulation_steps: int = 4  # Effective batch = 32 * 2 * 4 = 256

    # Optimizer
    optimizer: str = "adamw"          # Options: adamw, sgd, adam
    learning_rate: float = 1e-4
    backbone_lr_multiplier: float = 0.1  # Backbone LR = learning_rate * 0.1
    weight_decay: float = 0.01
    momentum: float = 0.9             # For SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam/AdamW

    # Learning Rate Schedule
    scheduler: str = "cosine_warmup"  # Options: cosine_warmup, cosine_restarts,
                                       #          onecycle, step, plateau
    warmup_epochs: int = 3
    warmup_lr: float = 1e-7
    min_lr: float = 1e-7

    # For step scheduler
    step_size: int = 10
    step_gamma: float = 0.1

    # ==========================================================================
    # REGULARIZATION
    # ==========================================================================
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    mixup_prob: float = 0.5           # Probability of applying mixup/cutmix per batch

    # Dropout already in model
    use_dropblock: bool = False
    dropblock_prob: float = 0.1

    # ==========================================================================
    # EXPONENTIAL MOVING AVERAGE (EMA)
    # ==========================================================================
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_every: int = 1         # Update EMA every N steps

    # ==========================================================================
    # STOCHASTIC WEIGHT AVERAGING (SWA)
    # ==========================================================================
    use_swa: bool = True
    swa_start_epoch: int = 20         # Start SWA after this epoch
    swa_lr: float = 1e-5
    swa_anneal_epochs: int = 5

    # ==========================================================================
    # EARLY STOPPING
    # ==========================================================================
    use_early_stopping: bool = True
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 0.001
    early_stopping_metric: str = "val_acc"  # Options: val_acc, val_auc, val_f1, val_loss

    # ==========================================================================
    # DATA CONFIGURATION
    # ==========================================================================
    image_size: int = 224
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    num_workers: int = 0 if sys.platform == "win32" else 4  # 0 on Windows = no multiprocessing stalls
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2

    # Class balancing
    use_class_weights: bool = True
    use_weighted_sampler: bool = True

    # Augmentation strength
    augmentation_strength: str = "strong"  # Options: light, medium, strong
    use_randaugment: bool = True
    randaugment_n: int = 2            # Number of augmentations
    randaugment_m: int = 10           # Magnitude (0-30)

    # Test Time Augmentation
    use_tta: bool = True
    tta_transforms: int = 5           # Number of TTA transforms

    # ==========================================================================
    # GRADIENT & OPTIMIZATION
    # ==========================================================================
    max_grad_norm: float = 1.0        # Gradient clipping
    use_amp: bool = True              # Automatic Mixed Precision

    # ==========================================================================
    # CHECKPOINTING & LOGGING
    # ==========================================================================
    save_every_n_epochs: int = 5
    log_every_n_steps: int = 25
    eval_every_n_epochs: int = 1

    # ==========================================================================
    # MISC
    # ==========================================================================
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False           # Set True for fixed input sizes
    min_images_required: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def __post_init__(self):
        """Validate configuration."""
        valid_backbones = [
            'resnet50', 'resnet101',
            'efficientnet_v2_m', 'efficientnet_v2_l',
            'convnext_base', 'convnext_large'
        ]
        if self.backbone not in valid_backbones:
            raise ValueError(f"Invalid backbone: {self.backbone}. Choose from {valid_backbones}")

        valid_optimizers = ['adamw', 'sgd', 'adam']
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")

        valid_schedulers = ['cosine_warmup', 'cosine_restarts', 'onecycle', 'step', 'plateau']
        if self.scheduler not in valid_schedulers:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")


# Default configuration (can be overridden by CLI --config)
CONFIG = TrainingConfig()

# Print config summary
print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)
print(f"Backbone:        {CONFIG.backbone}")
print(f"Epochs:          {CONFIG.num_epochs}")
print(f"Batch size:      {CONFIG.batch_size} (effective: {CONFIG.batch_size * CONFIG.gradient_accumulation_steps})")
print(f"Learning rate:   {CONFIG.learning_rate}")
print(f"Scheduler:       {CONFIG.scheduler}")
print(f"EMA:             {CONFIG.use_ema} (decay: {CONFIG.ema_decay})")
print(f"SWA:             {CONFIG.use_swa} (start: epoch {CONFIG.swa_start_epoch})")
print(f"Mixup/CutMix:    {CONFIG.mixup_prob}")
print(f"Label smoothing: {CONFIG.label_smoothing}")
print("=" * 70)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        # For PyTorch >= 1.8
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        try:
            torch.use_deterministic_algorithms(True)
        except:
            pass
    else:
        torch.backends.cudnn.benchmark = True

    print(f"Random seed set to {seed} (deterministic={deterministic})")


# Set seed immediately
set_seed(CONFIG.seed, CONFIG.deterministic)


# =============================================================================
# UTILITY CLASSES
# =============================================================================

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = "", fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class EarlyStopping:
    """
    Early stopping handler with customizable metric and patience.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        mode: str = "max",
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"   EarlyStopping: {self.counter}/{self.patience} "
                      f"(best: {self.best_score:.4f} at epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ExponentialMovingAverage:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model weights that is updated
    with an exponential moving average, providing more stable
    predictions during inference.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.num_updates = 0

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update shadow weights with current model weights."""
        self.num_updates += 1

        # Compute decay with warmup
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)

    def apply_shadow(self, model: nn.Module):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, Any]:
        return {
            'shadow': self.shadow,
            'num_updates': self.num_updates,
            'decay': self.decay
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.shadow = state_dict['shadow']
        self.num_updates = state_dict['num_updates']
        self.decay = state_dict['decay']


class ProgressBar:
    """Simple progress bar for training."""

    def __init__(self, total: int, prefix: str = "", width: int = 40):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0

    def update(self, current: int, suffix: str = ""):
        self.current = current
        filled = int(self.width * current / self.total)
        bar = "=" * filled + "-" * (self.width - filled)
        percent = 100 * current / self.total
        print(f"\r{self.prefix} [{bar}] {percent:5.1f}% {suffix}", end="", flush=True)
        if current >= self.total:
            print()


# =============================================================================
# GPU & MEMORY UTILITIES
# =============================================================================

def get_gpu_info() -> Dict[str, Any]:
    """Get comprehensive GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    info = {
        "available": True,
        "count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "gpus": []
    }

    for i in range(info["count"]):
        props = torch.cuda.get_device_properties(i)
        gpu_info = {
            "index": i,
            "name": props.name,
            "total_memory_gb": props.total_memory / (1024**3),
            "allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
            "free_gb": (props.total_memory - torch.cuda.memory_reserved(i)) / (1024**3),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        }
        info["gpus"].append(gpu_info)

    return info


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"   GPU {i}: {allocated:.2f}/{total:.2f} GB allocated, "
                  f"{reserved:.2f} GB reserved")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_disk_usage(path: str) -> Dict[str, float]:
    """Get disk usage statistics."""
    try:
        total, used, free = shutil.disk_usage(path)
        return {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "percent_used": (used / total) * 100
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent_used": 0}


# =============================================================================
# STEP 1: RAR EXTRACTION
# =============================================================================

def find_rar_file(search_dir: str, filename: str) -> Optional[str]:
    """
    Recursively search for the RAR file in the input directory.

    Args:
        search_dir: Directory to search in
        filename: Name of the RAR file to find

    Returns:
        Path to the RAR file or None if not found
    """
    print(f"\nSearching for '{filename}' in '{search_dir}'...")

    # Direct search for exact filename
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.lower() == filename.lower():
                path = os.path.join(root, file)
                size_gb = os.path.getsize(path) / (1024**3)
                print(f"Found: {path} ({size_gb:.2f} GB)")
                return path

    # Fallback: find any .rar file
    rar_files = glob.glob(os.path.join(search_dir, "**/*.rar"), recursive=True)
    if rar_files:
        print(f"Exact file not found. Available RAR files:")
        for f in rar_files[:10]:
            size_gb = os.path.getsize(f) / (1024**3)
            print(f"   - {f} ({size_gb:.2f} GB)")

        # Use the largest RAR file (likely the main dataset)
        largest = max(rar_files, key=os.path.getsize)
        size_gb = os.path.getsize(largest) / (1024**3)
        print(f"Using largest RAR: {largest} ({size_gb:.2f} GB)")
        return largest

    print("ERROR: No RAR files found!")
    return None


def extract_rar_with_fallback(rar_path: str, extract_to: str) -> Tuple[bool, str]:
    """
    Extract RAR archive with graceful handling of disk quota errors.

    If disk fills up during extraction, the function will:
    1. Stop extracting
    2. Print a warning
    3. Return success=True to allow training with partial data

    Args:
        rar_path: Path to the RAR file
        extract_to: Directory to extract to

    Returns:
        Tuple of (success, message)
    """
    os.makedirs(extract_to, exist_ok=True)

    # Print disk space before extraction
    disk_before = get_disk_usage("/kaggle")
    print(f"\nDisk space before extraction: {disk_before['free_gb']:.2f} GB free")
    print(f"Extracting to: {extract_to}")
    print("This may take a while for large archives...")

    try:
        # Use unrar with options:
        # -o+ : overwrite existing files
        # -y  : assume yes to all queries
        result = subprocess.run(
            ["unrar", "x", "-o+", "-y", rar_path, extract_to],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        if result.returncode == 0:
            disk_after = get_disk_usage("/kaggle")
            print(f"Extraction completed successfully!")
            print(f"Disk space after: {disk_after['free_gb']:.2f} GB free")
            return True, "Extraction completed successfully"

        # Check for disk quota errors
        output = (result.stderr + result.stdout).lower()
        disk_errors = ["disk quota", "no space", "disk full", "write error", "cannot create"]

        if any(err in output for err in disk_errors):
            print("\n" + "=" * 60)
            print("WARNING: DISK QUOTA EXCEEDED")
            print("Training will proceed with partially extracted data")
            print("=" * 60)
            return True, "Partial extraction (disk quota reached)"

        # Other errors - still try to proceed
        print(f"Extraction completed with warnings: {result.stderr[:500]}")
        return True, "Extraction completed with warnings"

    except subprocess.TimeoutExpired:
        print("\nExtraction timeout - proceeding with partial data")
        return True, "Partial extraction (timeout)"

    except Exception as e:
        error_str = str(e).lower()
        if any(err in error_str for err in ["disk quota", "no space", "disk full"]):
            print("\n" + "=" * 60)
            print("WARNING: DISK QUOTA EXCEEDED")
            print("Training will proceed with partially extracted data")
            print("=" * 60)
            return True, "Partial extraction (disk quota)"

        print(f"Extraction error: {e}")
        return False, str(e)


# =============================================================================
# STEP 2: ADVANCED DATA AUGMENTATION
# =============================================================================

class RandAugment:
    """
    RandAugment: Practical automated data augmentation with reduced search space.

    Paper: https://arxiv.org/abs/1909.13719

    Applies N random augmentations from a predefined set, each with magnitude M.
    """

    def __init__(self, n: int = 2, m: int = 10, prob: float = 1.0):
        """
        Args:
            n: Number of augmentations to apply
            m: Magnitude of augmentations (0-30)
            prob: Probability of applying RandAugment
        """
        self.n = n
        self.m = m
        self.prob = prob

        # Available augmentations
        self.augmentations = [
            self._identity,
            self._auto_contrast,
            self._equalize,
            self._rotate,
            self._solarize,
            self._color,
            self._posterize,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
        ]

    def __call__(self, img):
        if random.random() > self.prob:
            return img

        ops = random.choices(self.augmentations, k=self.n)
        for op in ops:
            img = op(img, self.m)
        return img

    def _identity(self, img, _):
        return img

    def _auto_contrast(self, img, _):
        from PIL import ImageOps
        return ImageOps.autocontrast(img)

    def _equalize(self, img, _):
        from PIL import ImageOps
        return ImageOps.equalize(img)

    def _rotate(self, img, m):
        from PIL import Image
        angle = (m / 30) * 30
        if random.random() > 0.5:
            angle = -angle
        return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    def _solarize(self, img, m):
        from PIL import ImageOps
        threshold = 256 - int((m / 30) * 256)
        return ImageOps.solarize(img, threshold)

    def _color(self, img, m):
        from PIL import ImageEnhance
        factor = 1.0 + (m / 30) * 0.9 * random.choice([-1, 1])
        factor = max(0.1, min(1.9, factor))
        return ImageEnhance.Color(img).enhance(factor)

    def _posterize(self, img, m):
        from PIL import ImageOps
        bits = 8 - int((m / 30) * 4)
        bits = max(1, min(8, bits))
        return ImageOps.posterize(img, bits)

    def _contrast(self, img, m):
        from PIL import ImageEnhance
        factor = 1.0 + (m / 30) * 0.9 * random.choice([-1, 1])
        factor = max(0.1, min(1.9, factor))
        return ImageEnhance.Contrast(img).enhance(factor)

    def _brightness(self, img, m):
        from PIL import ImageEnhance
        factor = 1.0 + (m / 30) * 0.9 * random.choice([-1, 1])
        factor = max(0.1, min(1.9, factor))
        return ImageEnhance.Brightness(img).enhance(factor)

    def _sharpness(self, img, m):
        from PIL import ImageEnhance
        factor = 1.0 + (m / 30) * 0.9 * random.choice([-1, 1])
        factor = max(0.1, min(1.9, factor))
        return ImageEnhance.Sharpness(img).enhance(factor)

    def _shear_x(self, img, m):
        from PIL import Image
        shear = (m / 30) * 0.3 * random.choice([-1, 1])
        return img.transform(
            img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0),
            resample=Image.BILINEAR, fillcolor=(128, 128, 128)
        )

    def _shear_y(self, img, m):
        from PIL import Image
        shear = (m / 30) * 0.3 * random.choice([-1, 1])
        return img.transform(
            img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0),
            resample=Image.BILINEAR, fillcolor=(128, 128, 128)
        )

    def _translate_x(self, img, m):
        from PIL import Image
        pixels = int((m / 30) * img.size[0] * 0.3) * random.choice([-1, 1])
        return img.transform(
            img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
            resample=Image.BILINEAR, fillcolor=(128, 128, 128)
        )

    def _translate_y(self, img, m):
        from PIL import Image
        pixels = int((m / 30) * img.size[1] * 0.3) * random.choice([-1, 1])
        return img.transform(
            img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
            resample=Image.BILINEAR, fillcolor=(128, 128, 128)
        )


class GridMask:
    """
    GridMask augmentation.

    Paper: https://arxiv.org/abs/2001.04086
    """

    def __init__(self, d1: int = 96, d2: int = 224, rotate: float = 1, ratio: float = 0.5, prob: float = 0.5):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.prob = prob

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return img

        _, h, w = img.shape

        # Random grid size
        d = random.randint(self.d1, self.d2)

        # Grid mask
        mask = torch.ones(h, w, dtype=img.dtype, device=img.device)

        st_h = random.randint(0, d)
        st_w = random.randint(0, d)

        for i in range(-1, h // d + 1):
            s = d * i + st_h
            t = s + int(d * self.ratio)
            s = max(0, s)
            t = min(h, t)
            mask[s:t, :] = 0

        for i in range(-1, w // d + 1):
            s = d * i + st_w
            t = s + int(d * self.ratio)
            s = max(0, s)
            t = min(w, t)
            mask[:, s:t] = 0

        return img * mask


class SafeImageFolder(datasets.ImageFolder):
    """
    ImageFolder that skips corrupt/unreadable images instead of crashing.
    Critical for long training runs (50+ hours) on large datasets (1M+ images).
    Returns None for bad images; collate_fn filters them out.
    """

    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception:
            # Return None — will be filtered in collate_fn
            return None


def safe_collate_fn(batch):
    """Filter out None entries from corrupt images, then use default collate."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def get_train_transforms(config: TrainingConfig) -> transforms.Compose:
    """
    Get training transforms based on augmentation strength.

    Args:
        config: Training configuration

    Returns:
        Composed transforms for training
    """
    size = config.image_size

    # Base transforms
    base_transforms = [
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    # Augmentation based on strength
    if config.augmentation_strength == "light":
        aug_transforms = [
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    elif config.augmentation_strength == "medium":
        aug_transforms = [
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
        ]
    else:  # strong
        aug_transforms = [
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
            ),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        ]

    # Add RandAugment if enabled
    if config.use_randaugment:
        aug_transforms.append(
            RandAugment(n=config.randaugment_n, m=config.randaugment_m)
        )

    # Final transforms
    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ]

    return transforms.Compose(base_transforms + aug_transforms + final_transforms)


def get_val_transforms(config: TrainingConfig) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        config: Training configuration

    Returns:
        Composed transforms for validation
    """
    size = config.image_size

    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_tta_transforms(config: TrainingConfig) -> List[transforms.Compose]:
    """
    Get Test Time Augmentation transforms.

    Args:
        config: Training configuration

    Returns:
        List of transform compositions for TTA
    """
    size = config.image_size
    base_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    tta_list = [
        # Original
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            base_norm,
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            base_norm,
        ]),
        # Scale up
        transforms.Compose([
            transforms.Resize((size + 32, size + 32)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            base_norm,
        ]),
        # Scale down
        transforms.Compose([
            transforms.Resize((size - 32, size - 32)),
            transforms.Pad(16),
            transforms.ToTensor(),
            base_norm,
        ]),
        # Rotation
        transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomRotation(degrees=(5, 5)),
            transforms.ToTensor(),
            base_norm,
        ]),
    ]

    return tta_list[:config.tta_transforms]


# =============================================================================
# MIXUP & CUTMIX
# =============================================================================

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup augmentation.

    Paper: https://arxiv.org/abs/1710.09412

    Args:
        x: Input images [B, C, H, W]
        y: Labels [B]
        alpha: Beta distribution parameter

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix augmentation.

    Paper: https://arxiv.org/abs/1905.04899

    Args:
        x: Input images [B, C, H, W]
        y: Labels [B]
        alpha: Beta distribution parameter

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape

    # Compute bounding box
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply cutmix
    x_clone = x.clone()
    x_clone[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to match actual pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]

    return x_clone, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """Compute mixed loss for mixup/cutmix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# STEP 2B: DATA LOADING
# =============================================================================

def find_imagefolder_root(base_dir: str) -> Optional[str]:
    """
    Recursively find the correct root folder for ImageFolder.

    Searches for a directory containing class subfolders with images.

    Args:
        base_dir: Base directory to search from

    Returns:
        Path to the ImageFolder root or None
    """
    print(f"\nSearching for valid dataset structure in: {base_dir}")

    # Common class folder name patterns
    valid_patterns = [
        {"real", "fake"},
        {"real", "ai"},
        {"real", "synthetic"},
        {"authentic", "fake"},
        {"genuine", "forged"},
        {"original", "manipulated"},
        {"0", "1"},
        {"class_0", "class_1"},
        {"negative", "positive"},
    ]

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.gif'}

    def count_images(folder: str) -> int:
        """Count images in a folder."""
        count = 0
        try:
            for item in os.listdir(folder):
                if Path(item).suffix.lower() in image_extensions:
                    count += 1
        except (PermissionError, OSError):
            pass
        return count

    def check_folder(path: str) -> Tuple[bool, int]:
        """Check if path is a valid ImageFolder root."""
        try:
            subdirs = os.listdir(path)
            subdirs_lower = {d.lower() for d in subdirs if os.path.isdir(os.path.join(path, d))}

            # Check against known patterns
            for pattern in valid_patterns:
                if pattern.issubset(subdirs_lower):
                    total_images = 0
                    for subdir in subdirs:
                        subdir_path = os.path.join(path, subdir)
                        if os.path.isdir(subdir_path):
                            total_images += count_images(subdir_path)
                    if total_images > 0:
                        return True, total_images

            # Also accept any 2-folder structure with images
            dir_subdirs = [d for d in subdirs if os.path.isdir(os.path.join(path, d))]
            if len(dir_subdirs) == 2:
                total_images = 0
                for subdir in dir_subdirs:
                    total_images += count_images(os.path.join(path, subdir))
                if total_images > 0:
                    return True, total_images

        except (PermissionError, OSError):
            pass

        return False, 0

    # BFS to find all candidates
    candidates = []

    for root, dirs, files in os.walk(base_dir):
        is_valid, img_count = check_folder(root)
        if is_valid:
            depth = root.replace(base_dir, '').count(os.sep)
            candidates.append((root, img_count, depth))
            print(f"   Found: {root} ({img_count:,} images, depth {depth})")

    if candidates:
        # Prefer shallower directories with more images
        candidates.sort(key=lambda x: (-x[1], x[2]))
        best = candidates[0]
        print(f"\nSelected: {best[0]} ({best[1]:,} images)")
        return best[0]

    print("ERROR: No valid ImageFolder structure found!")
    return None


def analyze_dataset(data_root: str) -> Dict[str, Any]:
    """
    Analyze dataset statistics.

    Args:
        data_root: Root directory of the dataset

    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "root": data_root,
        "classes": [],
        "total_samples": 0,
        "class_counts": {},
        "class_weights": {},
        "imbalance_ratio": 1.0,
    }

    try:
        dataset = datasets.ImageFolder(root=data_root)
        stats["classes"] = dataset.classes
        stats["total_samples"] = len(dataset)

        # Count per class
        labels = [s[1] for s in dataset.samples]
        for i, cls in enumerate(dataset.classes):
            count = labels.count(i)
            stats["class_counts"][cls] = count

        # Compute class weights
        max_count = max(stats["class_counts"].values())
        min_count = min(stats["class_counts"].values())
        stats["imbalance_ratio"] = max_count / min_count if min_count > 0 else float('inf')

        for i, cls in enumerate(dataset.classes):
            count = stats["class_counts"][cls]
            # Inverse frequency weighting
            stats["class_weights"][i] = stats["total_samples"] / (len(dataset.classes) * count)

    except Exception as e:
        print(f"Error analyzing dataset: {e}")

    return stats


def create_data_loaders(
    data_root: str,
    config: TrainingConfig
) -> Tuple[DataLoader, DataLoader, List[str], Dict[int, float]]:
    """
    Create train and validation data loaders with stratified split.

    Args:
        data_root: Root directory of the dataset
        config: Training configuration

    Returns:
        train_loader, val_loader, class_names, class_weights
    """
    print(f"\n{'=' * 60}")
    print("LOADING DATASET")
    print(f"{'=' * 60}")

    # Analyze dataset
    stats = analyze_dataset(data_root)
    class_names = stats["classes"]

    print(f"Root: {data_root}")
    print(f"Classes: {class_names}")
    print(f"Total samples: {stats['total_samples']:,}")
    for cls, count in stats["class_counts"].items():
        print(f"   {cls}: {count:,} images")
    print(f"Imbalance ratio: {stats['imbalance_ratio']:.2f}")

    # Load dataset for splitting
    full_dataset = datasets.ImageFolder(root=data_root)
    labels = [s[1] for s in full_dataset.samples]
    indices = list(range(len(full_dataset)))

    # Stratified split
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config.val_ratio,
        stratify=labels,
        random_state=config.seed
    )

    print(f"\nSplit: {len(train_idx):,} train, {len(val_idx):,} validation")

    # Create datasets with appropriate transforms (SafeImageFolder skips corrupt images)
    train_dataset = SafeImageFolder(
        root=data_root,
        transform=get_train_transforms(config)
    )
    val_dataset = SafeImageFolder(
        root=data_root,
        transform=get_val_transforms(config)
    )

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    # Weighted sampler for class balance
    sampler = None
    shuffle = True

    if config.use_weighted_sampler:
        train_labels = [labels[i] for i in train_idx]
        sample_weights = [stats["class_weights"][l] for l in train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True
        )
        shuffle = False
        print("Using weighted random sampler for class balance")

    # Create data loaders
    # Fix prefetch_factor: only pass when num_workers > 0
    loader_kwargs = dict(
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["timeout"] = 120  # 2 min timeout — restart if worker hangs

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        collate_fn=safe_collate_fn,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size * 2,  # Larger batch for validation (no gradients)
        shuffle=False,
        collate_fn=safe_collate_fn,
        **loader_kwargs,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    return train_loader, val_loader, class_names, stats["class_weights"]


# =============================================================================
# STEP 3: MODEL ARCHITECTURE
# =============================================================================

class AttentionPool(nn.Module):
    """
    Attention-based pooling layer.

    Computes attention weights over spatial features and produces
    a weighted average.
    """

    def __init__(self, in_features: int, hidden_features: int = None):
        super().__init__()
        hidden_features = hidden_features or in_features // 4

        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C] or [B, C]
        if x.dim() == 2:
            return x

        attn_weights = self.attention(x)  # [B, N, 1]
        attn_weights = F.softmax(attn_weights, dim=1)

        return (x * attn_weights).sum(dim=1)  # [B, C]


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.

    Paper: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DeepfakeDetector(nn.Module):
    """
    Production-grade Deepfake Detection model.

    Features:
    - Multiple backbone support
    - Attention-based feature refinement
    - SE blocks for channel attention
    - Multi-dropout for better regularization
    - Gradient checkpointing support
    """

    BACKBONES = {
        'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V1, 2048),
        'resnet101': (resnet101, ResNet101_Weights.IMAGENET1K_V1, 2048),
        'efficientnet_v2_m': (efficientnet_v2_m, EfficientNet_V2_M_Weights.IMAGENET1K_V1, 1280),
        'efficientnet_v2_l': (efficientnet_v2_l, EfficientNet_V2_L_Weights.IMAGENET1K_V1, 1280),
        'convnext_base': (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024),
        'convnext_large': (convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1, 1536),
    }

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Get backbone configuration
        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}. "
                           f"Choose from {list(self.BACKBONES.keys())}")

        model_fn, weights, num_features = self.BACKBONES[backbone]

        # Load backbone
        weights_arg = weights if pretrained else None
        self.backbone = model_fn(weights=weights_arg)

        # Remove original classifier
        if 'resnet' in backbone:
            self.backbone.fc = nn.Identity()
        elif 'efficientnet' in backbone:
            self.backbone.classifier = nn.Identity()
        elif 'convnext' in backbone:
            self.backbone.classifier = nn.Identity()

        self.num_features = num_features

        # Attention-based feature refinement
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(num_features, num_features // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(num_features // 4, num_features),
                nn.Sigmoid()
            )

        # SE block for channel attention
        self.se = nn.Sequential(
            nn.Linear(num_features, num_features // 16),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // 16, num_features),
            nn.Sigmoid()
        )

        # Classification head with multi-sample dropout
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(256, num_classes)
        )

        # Multi-sample dropout layers for training
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(5)
        ])

        # Initialize weights
        self._init_weights()

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nModel: {backbone}")
        print(f"   Feature dimension: {num_features}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

    def _init_weights(self):
        """Initialize classifier weights using truncated normal."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, use_multi_dropout: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, C, H, W]
            use_multi_dropout: If True, use multi-sample dropout (training only)

        Returns:
            Logits [B, num_classes]
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Handle different output shapes
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        elif features.dim() == 3:
            features = features.mean(dim=1)

        # Apply attention
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights

        # Apply SE
        se_weights = self.se(features)
        features = features * se_weights

        # Multi-sample dropout (training regularization)
        if use_multi_dropout and self.training:
            outputs = []
            for dropout in self.dropout_layers:
                dropped = dropout(features)
                out = self.classifier(dropped)
                outputs.append(out)
            return torch.mean(torch.stack(outputs), dim=0)

        # Standard classification
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification (for visualization)."""
        features = self.backbone(x)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")


def create_model(config: TrainingConfig) -> DeepfakeDetector:
    """Create the detection model."""
    return DeepfakeDetector(
        backbone=config.backbone,
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
        pretrained=config.pretrained,
        use_attention=config.use_attention,
    )


def setup_multi_gpu(model: nn.Module) -> Tuple[nn.Module, torch.device, int]:
    """
    Setup model for multi-GPU training using DataParallel.

    Args:
        model: The model to setup

    Returns:
        model, device, num_gpus
    """
    print(f"\n{'=' * 60}")
    print("GPU SETUP")
    print(f"{'=' * 60}")

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA available: {num_gpus} GPU(s)")

        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}: {props.name}")
            print(f"      Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"      Compute: {props.major}.{props.minor}")
            print(f"      SMs: {props.multi_processor_count}")

        device = torch.device("cuda:0")
        model = model.to(device)

        if num_gpus > 1:
            model = nn.DataParallel(model)
            print(f"\nDataParallel enabled with {num_gpus} GPUs")

        return model, device, num_gpus

    print("CUDA not available, using CPU")
    return model, torch.device("cpu"), 0


# =============================================================================
# STEP 4: TRAINING ENGINE
# =============================================================================

class Metrics:
    """Compute and store training/validation metrics."""

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.preds = []
        self.probs = []
        self.labels = []

    def update(self, preds: torch.Tensor, probs: torch.Tensor, labels: torch.Tensor):
        self.preds.extend(preds.cpu().numpy())
        self.probs.extend(probs.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        preds = np.array(self.preds)
        probs = np.array(self.probs)
        labels = np.array(self.labels)

        # Handle binary classification
        if probs.ndim == 2 and probs.shape[1] == 2:
            probs_positive = probs[:, 1]
        else:
            probs_positive = probs

        metrics = {
            "accuracy": accuracy_score(labels, preds) * 100,
            "precision": precision_score(labels, preds, average='binary', zero_division=0) * 100,
            "recall": recall_score(labels, preds, average='binary', zero_division=0) * 100,
            "f1": f1_score(labels, preds, average='binary', zero_division=0) * 100,
            "specificity": self._specificity(labels, preds) * 100,
        }

        # AUC and AP (need probability scores)
        try:
            if len(np.unique(labels)) > 1:
                metrics["auc"] = roc_auc_score(labels, probs_positive) * 100
                metrics["ap"] = average_precision_score(labels, probs_positive) * 100
            else:
                metrics["auc"] = 0.0
                metrics["ap"] = 0.0
        except:
            metrics["auc"] = 0.0
            metrics["ap"] = 0.0

        return metrics

    def _specificity(self, labels: np.ndarray, preds: np.ndarray) -> float:
        """Compute specificity (true negative rate)."""
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    def confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(self.labels, self.preds)

    def classification_report(self) -> str:
        """Get full classification report."""
        return classification_report(
            self.labels, self.preds,
            target_names=self.class_names,
            digits=4
        )


class TrainingEngine:
    """
    Production-grade training engine with all advanced features.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
        num_gpus: int,
        class_names: List[str],
        class_weights: Dict[int, float],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.num_gpus = num_gpus
        self.class_names = class_names

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Loss function with label smoothing and class weights
        weight_tensor = None
        if config.use_class_weights:
            weight_tensor = torch.tensor(
                [class_weights[i] for i in range(len(class_names))],
                dtype=torch.float32, device=device
            )

        self.criterion = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=config.label_smoothing
        )

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # EMA
        self.ema = None
        if config.use_ema:
            base_model = model.module if hasattr(model, 'module') else model
            self.ema = ExponentialMovingAverage(base_model, config.ema_decay)
            print(f"EMA enabled (decay={config.ema_decay})")

        # SWA
        self.swa_model = None
        self.swa_scheduler = None
        if config.use_swa:
            self.swa_model = AveragedModel(model)
            self.swa_scheduler = SWALR(
                self.optimizer,
                swa_lr=config.swa_lr,
                anneal_epochs=config.swa_anneal_epochs
            )
            print(f"SWA enabled (start epoch {config.swa_start_epoch})")

        # Early stopping
        self.early_stopping = None
        if config.use_early_stopping:
            mode = "min" if "loss" in config.early_stopping_metric else "max"
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                mode=mode
            )
            print(f"Early stopping enabled (patience={config.early_stopping_patience})")

        # TensorBoard
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            log_dir = os.path.join(config.output_dir, "logs", config.experiment_name)
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging to: {log_dir}")

        # Metrics
        self.train_metrics = Metrics(class_names)
        self.val_metrics = Metrics(class_names)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.best_val_f1 = 0.0
        self.history = defaultdict(list)

        # Save config
        config.save(os.path.join(config.output_dir, "config.json"))

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with layer-wise learning rate."""
        config = self.config
        base_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Separate parameters into groups
        backbone_params = []
        head_params = []

        for name, param in base_model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {
                'params': backbone_params,
                'lr': config.learning_rate * config.backbone_lr_multiplier,
                'name': 'backbone'
            },
            {
                'params': head_params,
                'lr': config.learning_rate,
                'name': 'head'
            }
        ]

        if config.optimizer == "adamw":
            return optim.AdamW(
                param_groups,
                weight_decay=config.weight_decay,
                betas=config.betas
            )
        elif config.optimizer == "adam":
            return optim.Adam(
                param_groups,
                weight_decay=config.weight_decay,
                betas=config.betas
            )
        elif config.optimizer == "sgd":
            return optim.SGD(
                param_groups,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        config = self.config
        steps_per_epoch = len(self.train_loader) // config.gradient_accumulation_steps
        total_steps = steps_per_epoch * config.num_epochs
        warmup_steps = steps_per_epoch * config.warmup_epochs

        if config.scheduler == "cosine_warmup":
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup
                    return (config.warmup_lr / config.learning_rate) + \
                           (1 - config.warmup_lr / config.learning_rate) * (step / warmup_steps)
                # Cosine annealing
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return max(
                    config.min_lr / config.learning_rate,
                    0.5 * (1 + math.cos(math.pi * progress))
                )
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        elif config.scheduler == "cosine_restarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=steps_per_epoch * 5,
                T_mult=2,
                eta_min=config.min_lr
            )

        elif config.scheduler == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[
                    config.learning_rate * config.backbone_lr_multiplier,
                    config.learning_rate
                ],
                total_steps=total_steps,
                pct_start=config.warmup_epochs / config.num_epochs,
                anneal_strategy='cos',
                final_div_factor=config.learning_rate / config.min_lr
            )

        elif config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.step_size * steps_per_epoch,
                gamma=config.step_gamma
            )

        elif config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                min_lr=config.min_lr
            )

        else:
            raise ValueError(f"Unknown scheduler: {config.scheduler}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        config = self.config

        loss_meter = AverageMeter("loss")
        data_time = AverageMeter("data")
        batch_time = AverageMeter("batch")

        self.optimizer.zero_grad(set_to_none=True)

        end = time.time()
        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                if batch is None:
                    continue  # Skip empty batch (all images were corrupt)
                inputs, labels = batch
            except Exception as e:
                print(f"\n  [WARN] Skipping batch {batch_idx}: {e}")
                continue
            data_time.update(time.time() - end)

            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Apply Mixup or CutMix
            use_mixup = random.random() < config.mixup_prob
            if use_mixup:
                if random.random() > 0.5:
                    inputs, labels_a, labels_b, lam = mixup_data(
                        inputs, labels, config.mixup_alpha
                    )
                else:
                    inputs, labels_a, labels_b, lam = cutmix_data(
                        inputs, labels, config.cutmix_alpha
                    )

            # Forward pass with mixed precision
            with autocast(enabled=config.use_amp):
                outputs = self.model(inputs, use_multi_dropout=True)

                if use_mixup:
                    loss = mixup_criterion(
                        self.criterion, outputs, labels_a, labels_b, lam
                    )
                else:
                    loss = self.criterion(outputs, labels)

                loss = loss / config.gradient_accumulation_steps

            # Backward pass
            if config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    config.max_grad_norm
                )

                if config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                # Update EMA
                if self.ema is not None:
                    base_model = self.model.module if hasattr(self.model, 'module') else self.model
                    self.ema.update(base_model)

                # Update scheduler (for step-based schedulers)
                if config.scheduler not in ["plateau"]:
                    self.scheduler.step()

                self.global_step += 1

            # Update metrics
            loss_meter.update(loss.item() * config.gradient_accumulation_steps, inputs.size(0))

            if not use_mixup:
                with torch.no_grad():
                    probs = F.softmax(outputs, dim=1)
                    _, preds = outputs.max(1)
                    self.train_metrics.update(preds, probs, labels)

            batch_time.update(time.time() - end)
            end = time.time()

            # Logging
            if (batch_idx + 1) % config.log_every_n_steps == 0:
                current_lr = self.optimizer.param_groups[1]['lr']  # Head LR
                samples_processed = (batch_idx + 1) * config.batch_size * max(1, self.num_gpus)
                total_samples = len(self.train_loader) * config.batch_size * max(1, self.num_gpus)
                progress = 100 * (batch_idx + 1) / len(self.train_loader)

                print(f"   [{batch_idx+1:4d}/{len(self.train_loader)}] "
                      f"({progress:5.1f}%) "
                      f"Loss: {loss_meter.avg:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {batch_time.avg:.3f}s")

                if self.writer:
                    self.writer.add_scalar("train/loss_step", loss_meter.val, self.global_step)
                    self.writer.add_scalar("train/lr", current_lr, self.global_step)

        # Epoch metrics
        epoch_time = time.time() - epoch_start
        metrics = self.train_metrics.compute()
        metrics["loss"] = loss_meter.avg
        metrics["time"] = epoch_time

        return metrics

    @torch.no_grad()
    def validate(self, use_ema: bool = True) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            use_ema: Whether to use EMA weights for validation

        Returns:
            Dictionary of validation metrics
        """
        base_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Apply EMA weights if available
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(base_model)

        self.model.eval()
        self.val_metrics.reset()

        loss_meter = AverageMeter("val_loss")

        for batch in self.val_loader:
            if batch is None:
                continue
            inputs, labels = batch
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.config.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            self.val_metrics.update(preds, probs, labels)
            loss_meter.update(loss.item(), inputs.size(0))

        # Restore original weights if EMA was used
        if use_ema and self.ema is not None:
            self.ema.restore(base_model)

        metrics = self.val_metrics.compute()
        metrics["loss"] = loss_meter.avg

        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        filename: str = "checkpoint.pth"
    ):
        """Save model checkpoint."""
        base_model = self.model.module if hasattr(self.model, 'module') else self.model

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "metrics": metrics,
            "class_names": self.class_names,
            "config": self.config.to_dict(),
            "best_val_acc": self.best_val_acc,
            "best_val_auc": self.best_val_auc,
            "best_val_f1": self.best_val_f1,
        }

        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

        if self.swa_model is not None:
            checkpoint["swa_state_dict"] = self.swa_model.state_dict()

        filepath = os.path.join(self.config.output_dir, filename)
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = os.path.join(self.config.output_dir, "best_model.pth")
            shutil.copy(filepath, best_path)
            print(f"   New best model saved! (Acc: {metrics['accuracy']:.2f}%, "
                  f"AUC: {metrics['auc']:.2f}%, F1: {metrics['f1']:.2f}%)")

    def train(self) -> Dict[str, List[float]]:
        """
        Full training loop.

        Returns:
            Training history dictionary
        """
        config = self.config

        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Model:           {config.backbone}")
        print(f"Epochs:          {config.num_epochs}")
        print(f"Batch size:      {config.batch_size} x {max(1, self.num_gpus)} GPUs "
              f"x {config.gradient_accumulation_steps} accum = "
              f"{config.batch_size * max(1, self.num_gpus) * config.gradient_accumulation_steps}")
        print(f"Learning rate:   {config.learning_rate} (backbone: x{config.backbone_lr_multiplier})")
        print(f"Scheduler:       {config.scheduler}")
        print(f"Mixed precision: {config.use_amp}")
        print(f"EMA:             {config.use_ema}")
        print(f"SWA:             {config.use_swa} (start: epoch {config.swa_start_epoch})")
        print("=" * 70)

        total_start = time.time()

        for epoch in range(1, config.num_epochs + 1):
            self.current_epoch = epoch

            print(f"\n{'=' * 70}")
            print(f"EPOCH {epoch}/{config.num_epochs}")
            print(f"{'=' * 70}")

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(use_ema=config.use_ema)

            # Update SWA
            if config.use_swa and epoch >= config.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()

            # Update plateau scheduler
            if config.scheduler == "plateau":
                self.scheduler.step(val_metrics["accuracy"])

            # Log history
            for key, value in train_metrics.items():
                if key != "time":
                    self.history[f"train_{key}"].append(value)
            for key, value in val_metrics.items():
                self.history[f"val_{key}"].append(value)

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalars("loss", {
                    "train": train_metrics["loss"],
                    "val": val_metrics["loss"]
                }, epoch)
                self.writer.add_scalars("accuracy", {
                    "train": train_metrics["accuracy"],
                    "val": val_metrics["accuracy"]
                }, epoch)
                self.writer.add_scalar("val/auc", val_metrics["auc"], epoch)
                self.writer.add_scalar("val/f1", val_metrics["f1"], epoch)
                self.writer.add_scalar("val/precision", val_metrics["precision"], epoch)
                self.writer.add_scalar("val/recall", val_metrics["recall"], epoch)

            # Check for best model
            is_best = val_metrics["accuracy"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics["accuracy"]
            if val_metrics["auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc"]
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best, "latest.pth")

            if epoch % config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, val_metrics, False, f"epoch_{epoch}.pth")

            # Print epoch summary
            print(f"\n{'─' * 70}")
            print(f"EPOCH {epoch} SUMMARY")
            print(f"{'─' * 70}")
            print(f"   Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"   Val Loss:   {val_metrics['loss']:.4f} | "
                  f"Val Acc:   {val_metrics['accuracy']:.2f}%")
            print(f"   Val AUC:    {val_metrics['auc']:.2f}% | "
                  f"Val F1:    {val_metrics['f1']:.2f}%")
            print(f"   Precision:  {val_metrics['precision']:.2f}% | "
                  f"Recall:    {val_metrics['recall']:.2f}%")
            print(f"   Specificity:{val_metrics['specificity']:.2f}%")
            print(f"{'─' * 70}")
            print(f"   Best Val Acc: {self.best_val_acc:.2f}%")
            print(f"   Best Val AUC: {self.best_val_auc:.2f}%")
            print(f"   Best Val F1:  {self.best_val_f1:.2f}%")
            print(f"   Epoch time:   {train_metrics['time']/60:.1f} min")

            # Print confusion matrix
            cm = self.val_metrics.confusion_matrix()
            print(f"\n   Confusion Matrix:")
            print(f"                  Predicted")
            print(f"                  {self.class_names[0]:>10} {self.class_names[1]:>10}")
            print(f"   Actual")
            print(f"   {self.class_names[0]:>10}  {cm[0,0]:>10} {cm[0,1]:>10}")
            print(f"   {self.class_names[1]:>10}  {cm[1,0]:>10} {cm[1,1]:>10}")

            # GPU memory
            print_gpu_memory()

            # Early stopping
            if self.early_stopping is not None:
                metric_value = val_metrics.get(
                    config.early_stopping_metric.replace("val_", ""),
                    val_metrics["accuracy"]
                )
                if self.early_stopping(metric_value, epoch):
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

        # Final SWA batch norm update
        if config.use_swa and self.swa_model is not None:
            print("\nUpdating SWA batch normalization statistics...")
            torch.optim.swa_utils.update_bn(
                self.train_loader, self.swa_model, device=self.device
            )

            # Save SWA model
            swa_path = os.path.join(config.output_dir, "swa_model.pth")
            swa_state = self.swa_model.module.state_dict()
            torch.save({
                "model_state_dict": swa_state,
                "class_names": self.class_names,
                "config": config.to_dict()
            }, swa_path)
            print(f"SWA model saved to: {swa_path}")

        # Training complete
        total_time = time.time() - total_start

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"   Total time:     {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"   Best Val Acc:   {self.best_val_acc:.2f}%")
        print(f"   Best Val AUC:   {self.best_val_auc:.2f}%")
        print(f"   Best Val F1:    {self.best_val_f1:.2f}%")
        print(f"   Model saved:    {config.output_dir}/best_model.pth")
        print("=" * 70)

        if self.writer:
            self.writer.close()

        return dict(self.history)


# =============================================================================
# FINAL EVALUATION
# =============================================================================

def final_evaluation(
    model_path: str,
    val_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    config: TrainingConfig
) -> Dict[str, float]:
    """
    Comprehensive final evaluation on the best model.

    Args:
        model_path: Path to the best model checkpoint
        val_loader: Validation data loader
        class_names: Class names
        device: Device to use
        config: Training configuration

    Returns:
        Dictionary of final metrics
    """
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Create model
    model = DeepfakeDetector(
        backbone=config.backbone,
        num_classes=config.num_classes,
        dropout_rate=config.dropout_rate,
        pretrained=False,
        use_attention=config.use_attention,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Collect predictions
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)

            if config.use_tta:
                # Test Time Augmentation
                tta_transforms = get_tta_transforms(config)
                tta_probs = []

                # We need to apply TTA to PIL images, so we need to handle this differently
                # For simplicity, we'll just use the standard forward pass
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
            else:
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)

            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds) * 100,
        "precision": precision_score(all_labels, all_preds, average='binary') * 100,
        "recall": recall_score(all_labels, all_preds, average='binary') * 100,
        "f1": f1_score(all_labels, all_preds, average='binary') * 100,
        "auc": roc_auc_score(all_labels, all_probs[:, 1]) * 100,
        "ap": average_precision_score(all_labels, all_probs[:, 1]) * 100,
    }

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Print confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:")
    print(f"              Predicted")
    print(f"              {class_names[0]:>10} {class_names[1]:>10}")
    print(f"Actual")
    print(f"{class_names[0]:>10}  {cm[0,0]:>10} {cm[0,1]:>10}")
    print(f"{class_names[1]:>10}  {cm[1,0]:>10} {cm[1,1]:>10}")

    # Print metrics summary
    print(f"\nFinal Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"   Precision: {metrics['precision']:.2f}%")
    print(f"   Recall:    {metrics['recall']:.2f}%")
    print(f"   F1 Score:  {metrics['f1']:.2f}%")
    print(f"   AUC-ROC:   {metrics['auc']:.2f}%")
    print(f"   AP:        {metrics['ap']:.2f}%")

    # Save metrics
    metrics_path = os.path.join(config.output_dir, "final_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    return metrics


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def scan_kaggle_inputs() -> List[str]:
    """
    Scan input directories and find valid ImageFolder structures.
    Returns list of paths that contain Real/Fake or similar class folders.
    Works on both Kaggle and local Windows/Linux.
    """
    input_dir = CONFIG.input_dir
    valid_roots = []

    print(f"\nScanning inputs in: {input_dir}")
    print("-" * 50)

    if not os.path.exists(input_dir):
        print(f"ERROR: {input_dir} does not exist!")
        print(f"Please create it and place your dataset there (Real/Fake subfolders).")
        return []

    # List all datasets
    datasets = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            datasets.append(item_path)
            print(f"  📁 {item}/")

    print(f"\nFound {len(datasets)} dataset directories")
    print("\nSearching for valid image folders (Real/Fake structure)...")

    # Search each dataset for valid ImageFolder structure
    for dataset_path in datasets:
        result = find_imagefolder_root(dataset_path)
        if result:
            # Count images
            try:
                ds = datasets_module.ImageFolder(root=result)
                count = len(ds)
                valid_roots.append((result, count, ds.classes))
                print(f"  ✅ {result}")
                print(f"     Classes: {ds.classes}, Images: {count:,}")
            except Exception as e:
                print(f"  ⚠️ {result} - Error: {e}")

    return valid_roots


def select_best_dataset(valid_roots: List[Tuple[str, int, List[str]]]) -> Optional[str]:
    """
    Select the best dataset based on image count and class structure.
    Prefers datasets with more images and binary classification (Real/Fake).
    """
    if not valid_roots:
        return None

    # Sort by image count (descending)
    valid_roots.sort(key=lambda x: x[1], reverse=True)

    # Prefer binary classification datasets
    binary_datasets = [r for r in valid_roots if len(r[2]) == 2]

    if binary_datasets:
        best = binary_datasets[0]
        print(f"\nSelected dataset: {best[0]}")
        print(f"   Classes: {best[2]}")
        print(f"   Total images: {best[1]:,}")
        return best[0]

    # Fallback to largest dataset
    best = valid_roots[0]
    print(f"\nSelected dataset (fallback): {best[0]}")
    print(f"   Classes: {best[2]}")
    print(f"   Total images: {best[1]:,}")
    return best[0]


# Alias for ImageFolder to avoid name conflict
datasets_module = datasets


def _prevent_sleep():
    """Prevent Windows from sleeping during training using SetThreadExecutionState."""
    if sys.platform == "win32":
        try:
            import ctypes
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_AWAYMODE_REQUIRED = 0x00000040
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            )
            print("[POWER] Windows sleep prevention ENABLED — PC will stay awake during training")
        except Exception as e:
            print(f"[POWER] Warning: Could not prevent sleep: {e}")


def _restore_sleep():
    """Restore normal Windows sleep behavior."""
    if sys.platform == "win32":
        try:
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            print("[POWER] Windows sleep prevention DISABLED — normal power management restored")
        except Exception:
            pass


def main(config_override: Optional[TrainingConfig] = None):
    """Main training pipeline."""
    global CONFIG
    if config_override is not None:
        CONFIG = config_override

    # Prevent Windows from sleeping during training
    _prevent_sleep()

    print("\n" + "=" * 70)
    print("DEEPFAKE DETECTION TRAINING PIPELINE")
    print("Senior ML Engineer Production Implementation")
    print("=" * 70)

    start_time = time.time()

    # Ensure output directory exists
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    os.makedirs(CONFIG.input_dir, exist_ok=True)

    # =========================================================================
    # STEP 1: Scan inputs for datasets
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: SCANNING DATASETS")
    print("-" * 70)

    # First, try to find already-extracted datasets (standard Kaggle setup)
    valid_roots = scan_kaggle_inputs()

    data_root = None

    if valid_roots:
        # Found valid dataset structures - select the best one
        data_root = select_best_dataset(valid_roots)
    else:
        # No valid structures found - try to extract RAR if exists
        print("\nNo ready-to-use datasets found. Looking for archives...")

        rar_path = find_rar_file(CONFIG.input_dir, CONFIG.rar_filename)

        if rar_path:
            print(f"Found archive: {rar_path}")
            success, message = extract_rar_with_fallback(rar_path, CONFIG.extract_dir)
            print(f"Extraction result: {message}")

            if success:
                data_root = find_imagefolder_root(CONFIG.extract_dir)
        else:
            # Also look for ZIP files
            zip_files = glob.glob(os.path.join(CONFIG.input_dir, "**/*.zip"), recursive=True)
            if zip_files:
                print(f"Found ZIP files: {zip_files[:5]}")
                # Could add ZIP extraction here if needed

    if data_root is None:
        print("\n" + "=" * 70)
        print("ERROR: Could not find valid dataset!")
        print("=" * 70)
        print("\nExpected structure:")
        print(f"  {CONFIG.input_dir}/your-dataset/")
        print("      Real/")
        print("          image1.jpg")
        print("          image2.jpg")
        print("      Fake/")
        print("          image1.jpg")
        print("          image2.jpg")
        if os.path.exists(CONFIG.input_dir):
            print(f"\nAvailable items in {CONFIG.input_dir}:")
            for item in os.listdir(CONFIG.input_dir):
                print(f"  - {item}")
        return

    # =========================================================================
    # STEP 2: Create data loaders
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: DATA LOADING")
    print("-" * 70)

    # Create data loaders
    try:
        train_loader, val_loader, class_names, class_weights = create_data_loaders(
            data_root=data_root,
            config=CONFIG
        )
    except Exception as e:
        print(f"ERROR creating data loaders: {e}")
        return

    # Check minimum data
    total_samples = len(train_loader.dataset) + len(val_loader.dataset)
    if total_samples < CONFIG.min_images_required:
        print(f"ERROR: Insufficient data ({total_samples} images)")
        print(f"Minimum required: {CONFIG.min_images_required}")
        return

    # =========================================================================
    # STEP 3: Create model
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: MODEL CREATION")
    print("-" * 70)

    model = create_model(CONFIG)
    model, device, num_gpus = setup_multi_gpu(model)

    # Clear memory
    clear_gpu_memory()

    # =========================================================================
    # STEP 4: Training
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: TRAINING")
    print("-" * 70)

    engine = TrainingEngine(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=CONFIG,
        device=device,
        num_gpus=num_gpus,
        class_names=class_names,
        class_weights=class_weights,
    )

    history = engine.train()

    # =========================================================================
    # STEP 5: Final evaluation
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: FINAL EVALUATION")
    print("-" * 70)

    best_model_path = os.path.join(CONFIG.output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        final_metrics = final_evaluation(
            model_path=best_model_path,
            val_loader=val_loader,
            class_names=class_names,
            device=device,
            config=CONFIG
        )

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total pipeline time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")

    print(f"\nOutput files in {CONFIG.output_dir}:")
    if os.path.exists(CONFIG.output_dir):
        for f in sorted(os.listdir(CONFIG.output_dir)):
            filepath = os.path.join(CONFIG.output_dir, f)
            if os.path.isfile(filepath):
                size_mb = os.path.getsize(filepath) / (1024**2)
                print(f"   {f}: {size_mb:.1f} MB")

    if TENSORBOARD_AVAILABLE:
        print(f"\nTensorBoard logs: {CONFIG.output_dir}/logs/")
        print(f"   Run: tensorboard --logdir {CONFIG.output_dir}/logs")

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)

    # Restore normal Windows sleep behavior
    _restore_sleep()


# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================
def parse_args():
    """Parse command-line arguments for local training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deepfake Detection Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python scripts/train_kaggle_deepfake.py

  # Train with specific config file
  python scripts/train_kaggle_deepfake.py --config configs/rtx5080_resnet50.json

  # Override backbone and batch size
  python scripts/train_kaggle_deepfake.py --backbone efficientnet_v2_m --batch-size 32

  # Specify custom data directory
  python scripts/train_kaggle_deepfake.py --data-dir ./data/train --config configs/rtx5080_convnext.json
        """,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file (overrides all defaults)",
    )
    parser.add_argument(
        "--backbone", type=str, default=None,
        choices=["resnet50", "resnet101", "efficientnet_v2_m",
                 "efficientnet_v2_l", "convnext_base", "convnext_large"],
        help="Model backbone architecture",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Training batch size per GPU",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to dataset root (containing Real/ and Fake/ folders)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Path to save outputs (checkpoints, logs, metrics)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--grad-accum", type=int, default=None,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )

    return parser.parse_args()


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    args = parse_args()

    # Build config: start from file or defaults, then apply CLI overrides
    if args.config:
        config = TrainingConfig.load(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        config = TrainingConfig()

    # Apply CLI overrides
    if args.backbone is not None:
        config.backbone = args.backbone
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.data_dir is not None:
        config.input_dir = args.data_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.no_amp:
        config.use_amp = False
    if args.grad_accum is not None:
        config.gradient_accumulation_steps = args.grad_accum

    main(config_override=config)
