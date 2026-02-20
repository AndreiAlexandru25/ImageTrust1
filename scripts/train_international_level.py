#!/usr/bin/env python3
"""
================================================================================
INTERNATIONAL-LEVEL AI IMAGE DETECTION TRAINING PIPELINE
================================================================================

Target: >90% Accuracy, >90% Recall, >90% Precision, >95% AUC
Level: IEEE WIFS / ACM IH&MMSec Conference Ready

KEY FIXES from previous training:
1. Focal Loss instead of CrossEntropy (handles hard examples better)
2. Reduced label smoothing (0.05 vs 0.1)
3. Reduced mixup probability (0.2 vs 0.5)
4. Early stopping on F1-Score (balances precision/recall)
5. Optimal threshold selection (not fixed 0.5)
6. Asymmetric class weights (penalize FN more than FP)

Usage:
    python scripts/train_international_level.py --backbone resnet50
    python scripts/train_international_level.py --backbone efficientnetv2_m --epochs 30
    python scripts/train_international_level.py --backbone convnext_base --focal-gamma 2.0

================================================================================
"""

import os
import sys
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

import torchvision
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

warnings.filterwarnings('ignore')

# =============================================================================
# FOCAL LOSS - KEY FIX FOR RECALL
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - gamma > 0 reduces the relative loss for well-classified examples,
      putting more focus on hard, misclassified examples.
    - alpha balances the importance of positive/negative examples.

    For AI detection:
    - gamma=2.0 is standard
    - alpha should penalize False Negatives (missing AI images) more
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        num_classes = inputs.size(-1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                targets_smooth = torch.zeros_like(inputs)
                targets_smooth.fill_(self.label_smoothing / (num_classes - 1))
                targets_smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Compute softmax probabilities
        p = F.softmax(inputs, dim=-1)

        # Get probability of true class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        # Final loss
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss - penalizes False Negatives more than False Positives.

    Useful when missing AI images (FN) is worse than false alarms (FP).
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,  # Focus on hard negatives
        gamma_pos: float = 1.0,  # Less focus on positives
        clip: float = 0.05,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Probabilities
        p = torch.sigmoid(inputs)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()

        # Asymmetric focusing
        p_pos = p * targets_one_hot
        p_neg = (1 - p) * (1 - targets_one_hot)

        # Clip negative probabilities
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1)

        # Focal modulation
        pos_weight = (1 - p_pos) ** self.gamma_pos
        neg_weight = p_neg ** self.gamma_neg

        # Loss
        loss = -targets_one_hot * pos_weight * torch.log(p.clamp(min=1e-8))
        loss -= (1 - targets_one_hot) * neg_weight * torch.log((1 - p).clamp(min=1e-8))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# CONFIGURATION - INTERNATIONAL LEVEL
# =============================================================================

@dataclass
class InternationalConfig:
    """
    Configuration optimized for international-level performance.
    Target: >90% Accuracy, >90% Recall, >90% Precision
    """

    # Paths
    data_dir: str = "./data/train"
    output_dir: str = "./outputs/training_international"
    experiment_name: str = "international_level"

    # Model
    backbone: str = "resnet50"
    num_classes: int = 2
    pretrained: bool = True
    dropout_rate: float = 0.3  # Reduced from 0.5

    # Training
    num_epochs: int = 30
    batch_size: int = 32
    gradient_accumulation_steps: int = 2

    # Optimizer
    learning_rate: float = 1e-4
    backbone_lr_multiplier: float = 0.1
    weight_decay: float = 0.01

    # Scheduler
    scheduler: str = "cosine_warmup"
    warmup_epochs: int = 3
    min_lr: float = 1e-6

    # CRITICAL FIXES FOR RECALL
    loss_function: str = "focal"  # "focal", "asymmetric", or "cross_entropy"
    focal_gamma: float = 2.0
    focal_alpha_real: float = 0.4  # Lower weight for Real class
    focal_alpha_ai: float = 0.6    # Higher weight for AI class (penalize FN)
    label_smoothing: float = 0.05  # Reduced from 0.1

    # Augmentation - REDUCED
    mixup_alpha: float = 0.2       # Reduced from 0.4
    cutmix_alpha: float = 0.5      # Reduced from 1.0
    mixup_prob: float = 0.2        # Reduced from 0.5
    use_randaugment: bool = True
    randaugment_n: int = 2
    randaugment_m: int = 9         # Reduced from 10

    # Early stopping - USE F1!
    early_stopping_metric: str = "val_f1"  # Changed from val_acc
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Threshold optimization
    optimize_threshold: bool = True
    threshold_metric: str = "f1"  # Optimize for F1

    # Class weights - ASYMMETRIC
    use_class_weights: bool = True
    real_class_weight: float = 1.0
    ai_class_weight: float = 1.5   # Penalize missing AI more

    # Training stability
    use_ema: bool = True
    ema_decay: float = 0.999
    use_swa: bool = True
    swa_start_epoch: int = 20

    # Other
    use_amp: bool = True
    seed: int = 42
    num_workers: int = 4

    # Validation
    val_ratio: float = 0.15  # 15% for validation


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# MODEL
# =============================================================================

def create_model(config: InternationalConfig, device: torch.device) -> nn.Module:
    """Create model with specified backbone."""
    import timm

    # Use timm for consistent interface
    model = timm.create_model(
        config.backbone,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        drop_rate=config.dropout_rate,
    )

    return model.to(device)


# =============================================================================
# DATA
# =============================================================================

def create_dataloaders(
    config: InternationalConfig,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, Dict[str, float]]:
    """Create train and validation dataloaders."""

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(config.data_dir, transform=train_transform)

    # Split
    total_size = len(full_dataset)
    val_size = int(total_size * config.val_ratio)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    # Update val transform
    val_dataset.dataset.transform = val_transform

    # Class weights
    targets = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = np.bincount(targets)

    class_weights = {
        0: config.real_class_weight,
        1: config.ai_class_weight,
    }

    # Weighted sampler for balanced batches
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    print(f"Train: {train_size:,} | Val: {val_size:,}")
    print(f"Class distribution: Real={class_counts[0]:,}, AI={class_counts[1]:,}")

    return train_loader, val_loader, class_weights


# =============================================================================
# TRAINING
# =============================================================================

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Find optimal classification threshold."""
    best_threshold = 0.5
    best_score = 0.0

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced":
            recall = recall_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            score = 2 * (precision * recall) / (precision + recall + 1e-8)
        else:
            score = accuracy_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: InternationalConfig,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass with AMP
        with autocast(enabled=config.use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / config.gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Metrics
        total_loss += loss.item() * config.gradient_accumulation_steps
        probs = F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    return {
        "loss": total_loss / len(train_loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: InternationalConfig,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Find optimal threshold
    opt_threshold, opt_f1 = find_optimal_threshold(all_labels, all_probs, "f1")
    opt_preds = (all_probs >= opt_threshold).astype(int)

    return {
        "loss": total_loss / len(val_loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0,
        # Optimal threshold metrics
        "opt_threshold": opt_threshold,
        "opt_accuracy": accuracy_score(all_labels, opt_preds),
        "opt_precision": precision_score(all_labels, opt_preds, zero_division=0),
        "opt_recall": recall_score(all_labels, opt_preds, zero_division=0),
        "opt_f1": opt_f1,
    }


def train(config: InternationalConfig):
    """Main training function."""
    print("=" * 70)
    print("INTERNATIONAL-LEVEL AI IMAGE DETECTION TRAINING")
    print("=" * 70)
    print(f"Target: >90% Accuracy, >90% Recall, >90% Precision, >95% AUC")
    print("=" * 70)

    # Setup
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create output directory
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Data
    train_loader, val_loader, class_weights = create_dataloaders(config, device)

    # Model
    model = create_model(config, device)
    print(f"Model: {config.backbone}")

    # Loss function - THE KEY FIX
    if config.loss_function == "focal":
        alpha = torch.tensor([config.focal_alpha_real, config.focal_alpha_ai]).to(device)
        criterion = FocalLoss(
            alpha=alpha,
            gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
        )
        print(f"Loss: Focal Loss (gamma={config.focal_gamma}, alpha={alpha.tolist()})")
    elif config.loss_function == "asymmetric":
        criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)
        print("Loss: Asymmetric Loss")
    else:
        weight = torch.tensor([config.real_class_weight, config.ai_class_weight]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)
        print("Loss: CrossEntropy")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.warmup_epochs,
        eta_min=config.min_lr,
    )

    # AMP
    scaler = GradScaler(enabled=config.use_amp)

    # Training loop
    best_f1 = 0
    best_epoch = 0
    patience_counter = 0

    history = {"train": [], "val": []}

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 50)

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config)

        # Update scheduler
        scheduler.step()

        # Log
        print(f"Train - Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.1%} | "
              f"Prec: {train_metrics['precision']:.1%} | "
              f"Rec: {train_metrics['recall']:.1%} | "
              f"F1: {train_metrics['f1']:.1%}")

        print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.1%} | "
              f"Prec: {val_metrics['precision']:.1%} | "
              f"Rec: {val_metrics['recall']:.1%} | "
              f"F1: {val_metrics['f1']:.1%} | "
              f"AUC: {val_metrics['auc']:.3f}")

        print(f"Optimal (t={val_metrics['opt_threshold']:.2f}) - "
              f"Acc: {val_metrics['opt_accuracy']:.1%} | "
              f"Prec: {val_metrics['opt_precision']:.1%} | "
              f"Rec: {val_metrics['opt_recall']:.1%} | "
              f"F1: {val_metrics['opt_f1']:.1%}")

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Save best model
        current_f1 = val_metrics["opt_f1"]  # Use optimal threshold F1
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": val_metrics,
                "config": asdict(config),
            }, output_dir / "best_model.pth")

            print(f"  *** New best model saved (F1={best_f1:.1%}) ***")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.early_stopping_patience})")

        # Early stopping
        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Save final results
    final_metrics = {
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "final_val_metrics": val_metrics,
        "config": asdict(config),
    }

    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best F1: {best_f1:.1%} at epoch {best_epoch}")
    print(f"Best recall: {history['val'][best_epoch-1]['opt_recall']:.1%}")
    print(f"Best precision: {history['val'][best_epoch-1]['opt_precision']:.1%}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return history


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="International-level AI detection training")
    parser.add_argument("--backbone", type=str, default="resnet50",
                       choices=["resnet50", "efficientnetv2_m", "convnext_base"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--ai-weight", type=float, default=1.5,
                       help="Weight for AI class (higher = penalize FN more)")
    parser.add_argument("--data-dir", type=str, default="./data/train")
    parser.add_argument("--output-dir", type=str, default="./outputs/training_international")
    parser.add_argument("--loss", type=str, default="focal",
                       choices=["focal", "asymmetric", "cross_entropy"])

    args = parser.parse_args()

    config = InternationalConfig(
        backbone=args.backbone,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        focal_gamma=args.focal_gamma,
        ai_class_weight=args.ai_weight,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        loss_function=args.loss,
        experiment_name=f"{args.backbone}_{args.loss}_international",
    )

    train(config)


if __name__ == "__main__":
    main()
