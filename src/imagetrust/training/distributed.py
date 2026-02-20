"""
Distributed Training and Experiment Logging.

Provides:
- Multi-GPU training with PyTorch Distributed Data Parallel (DDP)
- Weights & Biases (WandB) integration for experiment tracking
- Comprehensive logging utilities

These utilities are essential for:
- Scaling training to multiple GPUs
- Reproducible experiment tracking
- Publication-ready visualizations
"""

import os
import socket
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


# Check for WandB availability
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available. Install with: pip install wandb")


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    # DDP settings
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    # Multi-node settings
    master_addr: str = "localhost"
    master_port: str = "29500"

    # Sync settings
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    gradient_as_bucket_view: bool = True


def get_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_distributed(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: Optional[str] = None,
) -> DistributedConfig:
    """
    Initialize distributed training environment.

    Supports:
    - Single GPU (no-op)
    - Multi-GPU on single machine
    - Multi-node cluster (via SLURM or manual)

    Args:
        rank: Process rank. If None, read from environment.
        world_size: Total number of processes. If None, read from environment.
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU).
        master_addr: Master node address.
        master_port: Master node port. If None, find free port.

    Returns:
        DistributedConfig with setup details.
    """
    # Check for SLURM environment
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        logger.info(f"SLURM environment detected: rank={rank}, world_size={world_size}")
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Default: single process
        rank = rank if rank is not None else 0
        world_size = world_size if world_size is not None else 1
        local_rank = 0

    config = DistributedConfig(
        backend=backend,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port or str(get_free_port()),
    )

    # Skip distributed init for single process
    if world_size == 1:
        logger.info("Single process mode, skipping distributed init")
        return config

    # Set environment variables
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=config.init_method,
            world_size=world_size,
            rank=rank,
        )
        logger.info(
            f"Distributed init complete: rank={rank}/{world_size}, "
            f"backend={backend}"
        )

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return config


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get current process rank."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def wrap_model_ddp(
    model: nn.Module,
    device_id: Optional[int] = None,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
) -> Union[nn.Module, DDP]:
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: PyTorch model to wrap.
        device_id: GPU device ID. If None, use LOCAL_RANK.
        find_unused_parameters: Find unused parameters (slower but safer).
        broadcast_buffers: Broadcast buffers every forward pass.

    Returns:
        DDP-wrapped model (or original if single process).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return model

    if device_id is None:
        device_id = int(os.environ.get("LOCAL_RANK", 0))

    model = model.to(device_id)

    ddp_model = DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )

    logger.info(f"Model wrapped with DDP on device {device_id}")

    return ddp_model


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Reduce tensor across all processes and compute mean."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor

    world_size = dist.get_world_size()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / world_size

    return tensor


def all_gather_list(data: Any) -> List[Any]:
    """Gather arbitrary data from all processes."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [data]

    import pickle

    # Serialize data
    buffer = pickle.dumps(data)
    buffer_tensor = torch.ByteTensor(list(buffer)).cuda()

    # Gather sizes
    local_size = torch.tensor([buffer_tensor.numel()], dtype=torch.long).cuda()
    all_sizes = [torch.zeros(1, dtype=torch.long).cuda() for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)

    max_size = max(s.item() for s in all_sizes)

    # Pad to max size
    padded_tensor = torch.zeros(max_size, dtype=torch.uint8).cuda()
    padded_tensor[:buffer_tensor.numel()] = buffer_tensor

    # Gather all data
    all_tensors = [
        torch.zeros(max_size, dtype=torch.uint8).cuda()
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_tensors, padded_tensor)

    # Deserialize
    all_data = []
    for i, tensor in enumerate(all_tensors):
        size = all_sizes[i].item()
        buffer = bytes(tensor[:size].cpu().numpy())
        all_data.append(pickle.loads(buffer))

    return all_data


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases logging."""

    project: str = "imagetrust"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    resume: bool = False
    mode: str = "online"  # "online", "offline", "disabled"
    dir: Optional[str] = None


class WandBLogger:
    """
    Weights & Biases integration for experiment tracking.

    Provides:
    - Metric logging
    - Confusion matrix visualization
    - Calibration curves
    - Generalization gap tracking
    - Model checkpointing
    """

    def __init__(
        self,
        config: Optional[WandBConfig] = None,
        project: str = "imagetrust",
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
        run_config: Optional[Dict[str, Any]] = None,
        mode: str = "online",
    ):
        """
        Initialize WandB logger.

        Args:
            config: WandBConfig object. If provided, other args are ignored.
            project: Project name.
            name: Run name. If None, auto-generated.
            tags: List of tags for the run.
            notes: Run notes/description.
            run_config: Hyperparameters to log.
            mode: Logging mode ("online", "offline", "disabled").
        """
        if not WANDB_AVAILABLE:
            logger.warning("WandB not available. Logging disabled.")
            self._enabled = False
            return

        self._enabled = True

        if config is None:
            config = WandBConfig(
                project=project,
                name=name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=tags or [],
                notes=notes,
                config=run_config or {},
                mode=mode,
            )

        self.config = config
        self._run = None
        self._step = 0

        # Only initialize on main process
        if is_main_process():
            self._init_wandb()

    def _init_wandb(self):
        """Initialize WandB run."""
        self._run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.name,
            group=self.config.group,
            tags=self.config.tags,
            notes=self.config.notes,
            config=self.config.config,
            resume=self.config.resume,
            mode=self.config.mode,
            dir=self.config.dir,
        )
        logger.info(f"WandB initialized: {self._run.url}")

    @property
    def run(self):
        """Get WandB run object."""
        return self._run

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics to WandB.

        Args:
            metrics: Dictionary of metric names and values.
            step: Step number. If None, auto-increment.
            commit: Whether to commit this log immediately.
        """
        if not self._enabled or not is_main_process():
            return

        if step is None:
            step = self._step

        wandb.log(metrics, step=step, commit=commit)
        self._step = step + 1

    def log_generalization_gap(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """
        Log generalization gap metrics.

        Computes and logs the difference between train and validation metrics.

        Args:
            train_metrics: Training metrics (accuracy, loss, etc.).
            val_metrics: Validation metrics.
            step: Step number.
        """
        if not self._enabled or not is_main_process():
            return

        gap_metrics = {}

        # Common metric pairs
        metric_pairs = [
            ("accuracy", "accuracy"),
            ("loss", "loss"),
            ("auc", "auc"),
            ("f1", "f1"),
        ]

        for train_key, val_key in metric_pairs:
            train_val = train_metrics.get(f"train_{train_key}") or train_metrics.get(train_key)
            val_val = val_metrics.get(f"val_{val_key}") or val_metrics.get(val_key)

            if train_val is not None and val_val is not None:
                gap_metrics[f"gap_{train_key}"] = train_val - val_val

        if gap_metrics:
            self.log_metrics(gap_metrics, step=step)

    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        step: Optional[int] = None,
    ):
        """
        Log confusion matrix to WandB.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            class_names: Names for each class.
            title: Plot title.
            step: Step number.
        """
        if not self._enabled or not is_main_process():
            return

        if class_names is None:
            class_names = ["Real", "AI-Generated"]

        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        wandb.log({title: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def log_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        title: str = "Calibration Curve",
        step: Optional[int] = None,
    ):
        """
        Log reliability diagram (calibration curve) to WandB.

        Args:
            y_true: Ground truth labels.
            y_proba: Predicted probabilities.
            n_bins: Number of calibration bins.
            title: Plot title.
            step: Step number.
        """
        if not self._enabled or not is_main_process():
            return

        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

        # Calibration curve
        ax.plot(prob_pred, prob_true, "o-", label="Model")

        # Histogram of predicted probabilities
        ax2 = ax.twinx()
        ax2.hist(y_proba, bins=n_bins, alpha=0.3, color="gray")
        ax2.set_ylabel("Count")

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        wandb.log({title: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def log_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve",
        step: Optional[int] = None,
    ):
        """
        Log ROC curve to WandB.

        Args:
            y_true: Ground truth labels.
            y_proba: Predicted probabilities.
            title: Plot title.
            step: Step number.
        """
        if not self._enabled or not is_main_process():
            return

        from sklearn.metrics import roc_curve, roc_auc_score
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", label="Random")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        wandb.log({title: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def log_image(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        caption: Optional[str] = None,
        key: str = "images",
        step: Optional[int] = None,
    ):
        """
        Log images to WandB.

        Args:
            images: Single image or list of images.
            caption: Image caption.
            key: Log key.
            step: Step number.
        """
        if not self._enabled or not is_main_process():
            return

        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        wandb_images = [wandb.Image(img, caption=caption) for img in images]
        wandb.log({key: wandb_images}, step=step)

    def log_table(
        self,
        data: Dict[str, List],
        key: str = "table",
        step: Optional[int] = None,
    ):
        """
        Log table data to WandB.

        Args:
            data: Dictionary with column names as keys and lists as values.
            key: Log key.
            step: Step number.
        """
        if not self._enabled or not is_main_process():
            return

        table = wandb.Table(columns=list(data.keys()))
        n_rows = len(list(data.values())[0])

        for i in range(n_rows):
            row = [data[col][i] for col in data.keys()]
            table.add_data(*row)

        wandb.log({key: table}, step=step)

    def log_model(
        self,
        model: nn.Module,
        name: str = "model",
        aliases: Optional[List[str]] = None,
    ):
        """
        Log model checkpoint as artifact.

        Args:
            model: PyTorch model to save.
            name: Artifact name.
            aliases: Artifact aliases (e.g., ["best", "latest"]).
        """
        if not self._enabled or not is_main_process():
            return

        # Save model to temp file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(f.name, name=f"{name}.pt")
            wandb.log_artifact(artifact, aliases=aliases)

    def finish(self):
        """Finish WandB run."""
        if self._enabled and is_main_process() and self._run is not None:
            self._run.finish()
            logger.info("WandB run finished")


def log_training_run(
    model: nn.Module,
    train_metrics: List[Dict[str, float]],
    val_metrics: List[Dict[str, float]],
    config: Dict[str, Any],
    final_predictions: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    project: str = "imagetrust",
    name: Optional[str] = None,
):
    """
    Convenience function to log a complete training run.

    Args:
        model: Trained model.
        train_metrics: List of training metrics per epoch.
        val_metrics: List of validation metrics per epoch.
        config: Training configuration.
        final_predictions: Tuple of (y_true, y_pred, y_proba) for final evaluation.
        project: WandB project name.
        name: Run name.
    """
    logger = WandBLogger(
        project=project,
        name=name,
        run_config=config,
    )

    # Log metrics for each epoch
    for epoch, (train, val) in enumerate(zip(train_metrics, val_metrics)):
        all_metrics = {}
        all_metrics.update({f"train_{k}": v for k, v in train.items()})
        all_metrics.update({f"val_{k}": v for k, v in val.items()})
        logger.log_metrics(all_metrics, step=epoch)
        logger.log_generalization_gap(train, val, step=epoch)

    # Log final evaluation
    if final_predictions is not None:
        y_true, y_pred, y_proba = final_predictions
        logger.log_confusion_matrix(y_true, y_pred)
        logger.log_calibration_curve(y_true, y_proba)
        logger.log_roc_curve(y_true, y_proba)

    # Log model
    logger.log_model(model, aliases=["final"])

    logger.finish()
