"""
Advanced Training Strategy for Robust AI Detection.

Implements:
- Consistency regularization (KL divergence between clean and augmented)
- Hard negative mining (track and upweight misclassified screenshots)
- Gradient accumulation for large effective batch sizes
- Mixed precision training
- Learning rate scheduling with warmup

These techniques are critical for training models robust to
social media compression and screenshot artifacts.
"""

import os
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from tqdm import tqdm

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for advanced training."""

    # Basic training params
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Optimizer
    optimizer: str = "adamw"  # "adamw", "sgd"
    momentum: float = 0.9  # For SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # For AdamW

    # Learning rate schedule
    scheduler: str = "cosine"  # "cosine", "step", "plateau", "warmup_cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Consistency regularization
    use_consistency: bool = True
    consistency_weight: float = 0.5
    consistency_rampup_epochs: int = 10

    # Hard negative mining
    use_hard_negatives: bool = True
    hard_negative_memory_size: int = 1000
    hard_negative_weight: float = 3.0
    hard_negative_update_freq: int = 1  # Update every N epochs

    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    save_best: bool = True

    # Reproducibility
    seed: int = 42

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    epoch: int = 0
    train_loss: float = 0.0
    train_ce_loss: float = 0.0
    train_consistency_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    val_auc: float = 0.0
    learning_rate: float = 0.0
    hard_negatives_count: int = 0
    generalization_gap: float = 0.0


class ConsistencyLoss(nn.Module):
    """
    Consistency regularization loss.

    Enforces that predictions on clean and augmented versions
    of the same image should be similar.

    L_consistency = KL(p_clean || p_aug) + KL(p_aug || p_clean)

    This symmetric KL divergence encourages smooth predictions
    under input perturbations.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        reduction: str = "mean",
    ):
        """
        Initialize consistency loss.

        Args:
            temperature: Softmax temperature for softening predictions.
            reduction: Loss reduction ("mean", "sum", "none").
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        logits_clean: torch.Tensor,
        logits_aug: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute consistency loss.

        Args:
            logits_clean: Logits from clean images (B, num_classes).
            logits_aug: Logits from augmented images (B, num_classes).

        Returns:
            Consistency loss value.
        """
        # Softmax with temperature
        p_clean = F.softmax(logits_clean / self.temperature, dim=1)
        p_aug = F.softmax(logits_aug / self.temperature, dim=1)

        # Log probabilities
        log_p_clean = F.log_softmax(logits_clean / self.temperature, dim=1)
        log_p_aug = F.log_softmax(logits_aug / self.temperature, dim=1)

        # Symmetric KL divergence
        kl_clean_aug = F.kl_div(log_p_aug, p_clean, reduction="none").sum(dim=1)
        kl_aug_clean = F.kl_div(log_p_clean, p_aug, reduction="none").sum(dim=1)

        loss = (kl_clean_aug + kl_aug_clean) / 2

        # Scale by temperature^2 (standard practice for distillation)
        loss = loss * (self.temperature ** 2)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class HardNegativeMiner:
    """
    Track and manage hard negative examples.

    Hard negatives are samples that the model consistently misclassifies.
    For AI detection, these are typically:
    - Real photos misclassified as AI (screenshots, compressed)
    - AI images misclassified as real

    These samples are upweighted during training to improve robustness.
    """

    def __init__(
        self,
        memory_size: int = 1000,
        min_confidence_for_hard: float = 0.3,
        max_hard_ratio: float = 0.3,
    ):
        """
        Initialize hard negative miner.

        Args:
            memory_size: Maximum number of hard negatives to track.
            min_confidence_for_hard: Minimum confidence to consider as hard.
            max_hard_ratio: Maximum ratio of hard negatives in a batch.
        """
        self.memory_size = memory_size
        self.min_confidence_for_hard = min_confidence_for_hard
        self.max_hard_ratio = max_hard_ratio

        # Memory bank: {sample_id: (loss, times_wrong, metadata)}
        self._memory: Dict[int, Tuple[float, int, Dict]] = {}

        # Statistics
        self._total_added = 0
        self._total_removed = 0

    def update(
        self,
        sample_ids: List[int],
        losses: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
    ) -> int:
        """
        Update hard negative memory with batch results.

        Args:
            sample_ids: Unique IDs for samples in batch.
            losses: Per-sample losses.
            predictions: Model predictions (probabilities).
            labels: True labels.
            metadata: Optional metadata for each sample.

        Returns:
            Number of new hard negatives added.
        """
        losses = losses.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        added = 0

        for i, sample_id in enumerate(sample_ids):
            pred_label = int(predictions[i] > 0.5)
            true_label = int(labels[i])

            # Check if misclassified
            if pred_label != true_label:
                # Check confidence (hard negatives are confident but wrong)
                confidence = abs(predictions[i] - 0.5) * 2

                if confidence >= self.min_confidence_for_hard:
                    meta = metadata[i] if metadata else {}

                    if sample_id in self._memory:
                        # Update existing entry
                        old_loss, times_wrong, old_meta = self._memory[sample_id]
                        self._memory[sample_id] = (
                            losses[i],
                            times_wrong + 1,
                            meta,
                        )
                    else:
                        # Add new entry
                        self._memory[sample_id] = (losses[i], 1, meta)
                        added += 1
                        self._total_added += 1

        # Prune if over capacity
        if len(self._memory) > self.memory_size:
            # Remove samples with lowest times_wrong
            sorted_items = sorted(
                self._memory.items(),
                key=lambda x: x[1][1],  # Sort by times_wrong
                reverse=True,
            )
            self._memory = dict(sorted_items[:self.memory_size])
            self._total_removed += len(sorted_items) - self.memory_size

        return added

    def get_hard_negative_ids(self) -> List[int]:
        """Get list of hard negative sample IDs."""
        return list(self._memory.keys())

    def get_sample_weight(self, sample_id: int, base_weight: float = 1.0) -> float:
        """Get training weight for a sample."""
        if sample_id in self._memory:
            _, times_wrong, _ = self._memory[sample_id]
            # Weight increases with number of times misclassified
            return base_weight * (1 + min(times_wrong, 5))
        return base_weight

    def get_weights_for_batch(
        self,
        sample_ids: List[int],
        hard_negative_weight: float = 3.0,
    ) -> torch.Tensor:
        """Get training weights for a batch of samples."""
        weights = []
        for sample_id in sample_ids:
            if sample_id in self._memory:
                weights.append(hard_negative_weight)
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float32)

    def clear(self):
        """Clear the memory bank."""
        self._memory.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get mining statistics."""
        if not self._memory:
            return {
                "memory_size": 0,
                "total_added": self._total_added,
                "total_removed": self._total_removed,
            }

        times_wrong = [v[1] for v in self._memory.values()]
        losses = [v[0] for v in self._memory.values()]

        return {
            "memory_size": len(self._memory),
            "total_added": self._total_added,
            "total_removed": self._total_removed,
            "avg_times_wrong": np.mean(times_wrong),
            "max_times_wrong": max(times_wrong),
            "avg_loss": np.mean(losses),
        }


class AdvancedTrainer:
    """
    Advanced training loop with all robustness enhancements.

    Features:
    - Dual forward pass (clean + augmented) for consistency
    - Hard negative mining and upweighting
    - Gradient accumulation for large effective batch sizes
    - Mixed precision training (AMP)
    - Comprehensive logging and checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: TrainingConfig,
        augmentation_fn: Optional[Callable] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize advanced trainer.

        Args:
            model: PyTorch model to train.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            config: Training configuration.
            augmentation_fn: Function to apply heavy augmentation.
            device: Device to train on.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.augmentation_fn = augmentation_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Set seed for reproducibility
        self._set_seed(config.seed)

        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_criterion()
        self._setup_dataloaders()

        # Consistency loss
        self.consistency_loss = ConsistencyLoss(temperature=1.0)

        # Hard negative miner
        self.hard_negative_miner = HardNegativeMiner(
            memory_size=config.hard_negative_memory_size,
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        self.history: List[TrainingMetrics] = []

        # Early stopping
        self._early_stop_counter = 0

        logger.info(f"AdvancedTrainer initialized on {self.device}")

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_optimizer(self):
        """Initialize optimizer."""
        if self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _setup_scheduler(self):
        """Initialize learning rate scheduler."""
        if self.config.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler == "warmup_cosine":
            # Warmup + cosine annealing
            def lr_lambda(epoch):
                if epoch < self.config.warmup_epochs:
                    return (epoch + 1) / self.config.warmup_epochs
                else:
                    progress = (epoch - self.config.warmup_epochs) / (
                        self.config.epochs - self.config.warmup_epochs
                    )
                    return max(
                        self.config.min_lr / self.config.learning_rate,
                        0.5 * (1 + np.cos(np.pi * progress)),
                    )

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda
            )
        elif self.config.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1,
            )
        elif self.config.scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=self.config.min_lr,
            )
        else:
            self.scheduler = None

    def _setup_criterion(self):
        """Initialize loss criterion."""
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def _setup_dataloaders(self):
        """Initialize data loaders."""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def _get_consistency_weight(self, epoch: int) -> float:
        """Get consistency loss weight with rampup."""
        if not self.config.use_consistency:
            return 0.0

        if epoch < self.config.consistency_rampup_epochs:
            # Linear rampup
            return self.config.consistency_weight * (
                epoch / self.config.consistency_rampup_epochs
            )
        return self.config.consistency_weight

    def _training_step(
        self,
        batch: Dict[str, torch.Tensor],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step with consistency regularization.

        Args:
            batch: Batch dictionary with 'image', 'label', and optionally 'sample_id'.
            epoch: Current epoch number.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)
        sample_ids = batch.get("sample_id", list(range(len(labels))))

        batch_size = images.size(0)

        # Forward pass on clean images
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            logits_clean = self.model(images)

            # Ensure logits have correct shape for binary classification
            if logits_clean.dim() == 1:
                logits_clean = logits_clean.unsqueeze(1)
            if logits_clean.size(1) == 1:
                # Single output: convert to 2-class logits
                logits_clean = torch.cat([-logits_clean, logits_clean], dim=1)

            # Cross-entropy loss
            ce_loss_per_sample = self.criterion(logits_clean, labels.long())

            # Apply hard negative weights
            if self.config.use_hard_negatives:
                weights = self.hard_negative_miner.get_weights_for_batch(
                    sample_ids if isinstance(sample_ids, list) else sample_ids.tolist(),
                    hard_negative_weight=self.config.hard_negative_weight,
                ).to(self.device)
                ce_loss = (ce_loss_per_sample * weights).mean()
            else:
                ce_loss = ce_loss_per_sample.mean()

            # Consistency regularization
            consistency_loss = torch.tensor(0.0, device=self.device)
            if self.config.use_consistency and self.augmentation_fn is not None:
                # Apply heavy augmentation
                images_aug = self.augmentation_fn(images)
                logits_aug = self.model(images_aug)

                if logits_aug.dim() == 1:
                    logits_aug = logits_aug.unsqueeze(1)
                if logits_aug.size(1) == 1:
                    logits_aug = torch.cat([-logits_aug, logits_aug], dim=1)

                consistency_loss = self.consistency_loss(logits_clean, logits_aug)

            # Total loss
            consistency_weight = self._get_consistency_weight(epoch)
            total_loss = ce_loss + consistency_weight * consistency_loss

        # Update hard negative miner
        if self.config.use_hard_negatives:
            with torch.no_grad():
                probs = F.softmax(logits_clean, dim=1)[:, 1]
                self.hard_negative_miner.update(
                    sample_ids if isinstance(sample_ids, list) else sample_ids.tolist(),
                    ce_loss_per_sample,
                    probs,
                    labels,
                )

        loss_dict = {
            "ce_loss": ce_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_consistency_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Training step
            loss, loss_dict = self._training_step(batch, epoch)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.config.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate metrics
            total_loss += loss_dict["total_loss"]
            total_ce_loss += loss_dict["ce_loss"]
            total_consistency_loss += loss_dict["consistency_loss"]

            # Accuracy
            with torch.no_grad():
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = self.model(images)
                if logits.dim() == 1 or logits.size(1) == 1:
                    preds = (logits.squeeze() > 0).long()
                else:
                    preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss']:.4f}",
                "acc": f"{correct / total:.4f}",
            })

        n_batches = len(self.train_loader)
        return {
            "train_loss": total_loss / n_batches,
            "train_ce_loss": total_ce_loss / n_batches,
            "train_consistency_loss": total_consistency_loss / n_batches,
            "train_accuracy": correct / total,
        }

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(images)

            # Handle different output formats
            if logits.dim() == 1:
                logits = logits.unsqueeze(1)
            if logits.size(1) == 1:
                logits = torch.cat([-logits, logits], dim=1)

            loss = F.cross_entropy(logits, labels.long())
            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)[:, 1]
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        accuracy = np.mean(all_preds == all_labels)

        # AUC
        from sklearn.metrics import roc_auc_score

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5

        return {
            "val_loss": total_loss / len(self.val_loader),
            "val_accuracy": accuracy,
            "val_auc": auc,
        }

    def train(self) -> List[TrainingMetrics]:
        """
        Run the full training loop.

        Returns:
            List of TrainingMetrics for each epoch.
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate()

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["val_auc"])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Compute generalization gap
            gen_gap = train_metrics["train_accuracy"] - val_metrics["val_accuracy"]

            # Create metrics object
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics["train_loss"],
                train_ce_loss=train_metrics["train_ce_loss"],
                train_consistency_loss=train_metrics["train_consistency_loss"],
                train_accuracy=train_metrics["train_accuracy"],
                val_loss=val_metrics["val_loss"],
                val_accuracy=val_metrics["val_accuracy"],
                val_auc=val_metrics["val_auc"],
                learning_rate=current_lr,
                hard_negatives_count=len(self.hard_negative_miner.get_hard_negative_ids()),
                generalization_gap=gen_gap,
            )
            self.history.append(metrics)

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs}: "
                f"train_loss={metrics.train_loss:.4f}, "
                f"train_acc={metrics.train_accuracy:.4f}, "
                f"val_loss={metrics.val_loss:.4f}, "
                f"val_acc={metrics.val_accuracy:.4f}, "
                f"val_auc={metrics.val_auc:.4f}, "
                f"gen_gap={metrics.generalization_gap:.4f}"
            )

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch, "periodic")

            # Save best model
            if self.config.save_best and metrics.val_auc > self.best_val_metric:
                self.best_val_metric = metrics.val_auc
                self._save_checkpoint(epoch, "best")
                self._early_stop_counter = 0
            else:
                self._early_stop_counter += 1

            # Early stopping
            if self._early_stop_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}"
                )
                break

        logger.info(
            f"Training complete. Best val AUC: {self.best_val_metric:.4f}"
        )

        return self.history

    def _save_checkpoint(self, epoch: int, tag: str):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_metric": self.best_val_metric,
            "history": self.history,
            "config": self.config,
        }

        path = checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: Union[str, Path]):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_metric = checkpoint["best_val_metric"]
        self.history = checkpoint["history"]

        logger.info(f"Checkpoint loaded from {path}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.history:
            return {}

        best_epoch = max(self.history, key=lambda x: x.val_auc)

        return {
            "total_epochs": len(self.history),
            "best_epoch": best_epoch.epoch,
            "best_val_auc": best_epoch.val_auc,
            "best_val_accuracy": best_epoch.val_accuracy,
            "final_train_accuracy": self.history[-1].train_accuracy,
            "final_val_accuracy": self.history[-1].val_accuracy,
            "final_generalization_gap": self.history[-1].generalization_gap,
            "hard_negatives_final": self.history[-1].hard_negatives_count,
            "hard_negative_stats": self.hard_negative_miner.get_statistics(),
        }


def create_augmentation_fn(device: str = "cuda") -> Callable:
    """
    Create augmentation function for consistency regularization.

    Returns a callable that applies robustness augmentations.
    """
    try:
        from imagetrust.detection.augmentation import RobustnessAugmentor

        augmentor = RobustnessAugmentor(
            input_size=224,
            social_media_prob=0.4,
            screenshot_prob=0.3,
        )

        def augment_batch(images: torch.Tensor) -> torch.Tensor:
            # Images are already tensors (B, C, H, W)
            # Apply augmentation (this is simplified - in practice would need
            # to handle tensor -> PIL -> augment -> tensor conversion)
            import torchvision.transforms as T

            aug_transform = T.Compose([
                T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])

            return aug_transform(images)

        return augment_batch

    except ImportError:
        # Fallback: simple augmentation
        import torchvision.transforms as T

        aug_transform = T.Compose([
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ])

        def simple_augment(images: torch.Tensor) -> torch.Tensor:
            return aug_transform(images)

        return simple_augment
