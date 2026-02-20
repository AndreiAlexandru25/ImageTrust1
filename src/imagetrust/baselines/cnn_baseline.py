"""
B2: CNN Baseline - ResNet-50 or EfficientNet-B0 classifier.

Single-model CNN baseline for AI image detection.
Uses ImageNet pretrained weights and fine-tunes on the detection task.

For the paper, report:
- Backbone architecture (ResNet-50, EfficientNet-B0)
- Pretrained weights source (ImageNet)
- Fine-tuning strategy (full or head-only)
- Optimizer, LR, scheduler
- Batch size, epochs
- Data augmentation used
- Training time per epoch
- GPU memory usage
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from imagetrust.baselines.base import BaselineDetector, BaselineConfig, BaselineResult


class CNNBaseline(BaselineDetector):
    """
    CNN baseline using ResNet-50 or EfficientNet-B0.

    This is a standard deep learning baseline that should be competitive
    with many published methods.

    Example:
        >>> config = BaselineConfig(
        ...     name="CNN (ResNet-50)",
        ...     epochs=10,
        ...     batch_size=32,
        ...     learning_rate=1e-4,
        ... )
        >>> baseline = CNNBaseline(config, backbone="resnet50")
        >>> history = baseline.fit(train_images, train_labels, val_images, val_labels)
        >>> result = baseline.predict_proba(test_image)
    """

    SUPPORTED_BACKBONES = ["resnet50", "resnet18", "efficientnet_b0", "efficientnet_b4"]

    def __init__(
        self,
        config: BaselineConfig,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        input_size: int = 224,
    ):
        """
        Initialize CNN baseline.

        Args:
            config: Baseline configuration
            backbone: Model architecture name
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone, train only classifier head
            input_size: Input image size
        """
        super().__init__(config)

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported: {self.SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.input_size = input_size

        # PyTorch components (lazy init)
        self._model = None
        self._device = None
        self._transform = None

        # Store config for paper reporting
        self.config.model_params.update({
            "backbone": backbone,
            "pretrained": pretrained,
            "freeze_backbone": freeze_backbone,
            "input_size": input_size,
        })

    def _init_model(self) -> None:
        """Initialize PyTorch model and transforms."""
        import torch
        import torch.nn as nn
        import timm

        # Device selection
        if self.config.device == "auto" or self.config.device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(self.config.device)

        # Create model using timm
        self._model = timm.create_model(
            self.backbone_name,
            pretrained=self.pretrained,
            num_classes=2,  # Real vs AI
        )

        # Freeze backbone if requested
        if self.freeze_backbone:
            for name, param in self._model.named_parameters():
                if "classifier" not in name and "fc" not in name and "head" not in name:
                    param.requires_grad = False

        self._model = self._model.to(self._device)

        # Create transforms
        self._transform = self._create_transform(is_train=False)

    def _create_transform(self, is_train: bool = False):
        """Create image transforms."""
        from torchvision import transforms

        if is_train:
            return transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    def fit(
        self,
        train_images: List[Union[Image.Image, Path]],
        train_labels: List[int],
        val_images: Optional[List[Union[Image.Image, Path]]] = None,
        val_labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Train the CNN baseline.

        Args:
            train_images: Training images
            train_labels: Labels (0=real, 1=AI)
            val_images: Optional validation images
            val_labels: Optional validation labels

        Returns:
            Training history dict
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset

        # Initialize model
        self._init_model()

        # Create dataset class
        class ImageDataset(Dataset):
            def __init__(ds_self, images, labels, transform, load_fn):
                ds_self.images = images
                ds_self.labels = labels
                ds_self.transform = transform
                ds_self.load_fn = load_fn

            def __len__(ds_self):
                return len(ds_self.images)

            def __getitem__(ds_self, idx):
                img = ds_self.load_fn(ds_self.images[idx])
                img = ds_self.transform(img)
                label = ds_self.labels[idx]
                return img, label

        # Create dataloaders
        train_transform = self._create_transform(is_train=True)
        val_transform = self._create_transform(is_train=False)

        train_dataset = ImageDataset(
            train_images, train_labels, train_transform, self._load_image
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=True if self._device.type == "cuda" else False,
        )

        val_loader = None
        if val_images is not None and val_labels is not None:
            val_dataset = ImageDataset(
                val_images, val_labels, val_transform, self._load_image
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Setup training
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
        )

        # Training loop
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

        for epoch in range(self.config.epochs):
            # Train
            self._model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_imgs, batch_labels in train_loader:
                batch_imgs = batch_imgs.to(self._device)
                batch_labels = batch_labels.to(self._device)

                optimizer.zero_grad()
                outputs = self._model(batch_imgs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_imgs.size(0)
                _, predicted = outputs.max(1)
                train_total += batch_labels.size(0)
                train_correct += predicted.eq(batch_labels).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["lr"].append(scheduler.get_last_lr()[0])

            # Validate
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
            else:
                val_loss, val_acc = None, None

            scheduler.step()

            # Log progress
            val_str = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}" if val_loss else ""
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}{val_str}"
            )

        self.is_fitted = True

        # Store final metrics
        history["final_train_acc"] = history["train_acc"][-1]
        if history["val_acc"]:
            history["final_val_acc"] = history["val_acc"][-1]
        history["total_epochs"] = self.config.epochs
        history["train_samples"] = len(train_images)

        return history

    def _validate(self, val_loader, criterion) -> Tuple[float, float]:
        """Run validation."""
        import torch

        self._model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(self._device)
                batch_labels = batch_labels.to(self._device)

                outputs = self._model(batch_imgs)
                loss = criterion(outputs, batch_labels)

                val_loss += loss.item() * batch_imgs.size(0)
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()

        return val_loss / val_total, val_correct / val_total

    def predict_proba(self, image: Union[Image.Image, np.ndarray, Path]) -> BaselineResult:
        """
        Predict AI probability for a single image.

        Args:
            image: Input image

        Returns:
            BaselineResult with predictions
        """
        import torch
        import torch.nn.functional as F

        if self._model is None:
            self._init_model()

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first or load() a checkpoint.")

        pil_img = self._load_image(image)

        def _predict():
            self._model.eval()
            with torch.no_grad():
                img_tensor = self._transform(pil_img).unsqueeze(0).to(self._device)
                logits = self._model(img_tensor)
                probs = F.softmax(logits, dim=1)
                return probs.cpu().numpy()[0], logits.cpu().numpy()[0]

        (probs, logits), elapsed_ms = self._timed_predict(_predict)

        # probs[0] = P(real), probs[1] = P(AI)
        ai_prob = float(probs[1])
        real_prob = float(probs[0])
        raw_prob = ai_prob

        # Apply calibration if available
        if self._calibrator is not None:
            ai_prob = self._calibrator.calibrate(ai_prob)
            real_prob = 1 - ai_prob

        return BaselineResult(
            ai_probability=ai_prob,
            real_probability=real_prob,
            raw_logits=logits,
            raw_probability=raw_prob,
            baseline_name=self.name,
            processing_time_ms=elapsed_ms,
            calibrated=self._calibrator is not None,
        )

    def predict_proba_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, Path]],
    ) -> List[BaselineResult]:
        """Batch prediction for efficiency."""
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset

        if self._model is None:
            self._init_model()

        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")

        class SimpleDataset(Dataset):
            def __init__(ds_self, images, transform, load_fn):
                ds_self.images = images
                ds_self.transform = transform
                ds_self.load_fn = load_fn

            def __len__(ds_self):
                return len(ds_self.images)

            def __getitem__(ds_self, idx):
                img = ds_self.load_fn(ds_self.images[idx])
                return ds_self.transform(img)

        dataset = SimpleDataset(images, self._transform, self._load_image)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        all_probs = []
        all_logits = []

        self._model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                logits = self._model(batch)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)

        results = []
        for i in range(len(images)):
            ai_prob = float(all_probs[i, 1])
            raw_prob = ai_prob

            if self._calibrator is not None:
                ai_prob = self._calibrator.calibrate(ai_prob)

            results.append(BaselineResult(
                ai_probability=ai_prob,
                real_probability=1 - ai_prob,
                raw_logits=all_logits[i],
                raw_probability=raw_prob,
                baseline_name=self.name,
                processing_time_ms=0,
                calibrated=self._calibrator is not None,
            ))

        return results

    def save(self, path: Union[str, Path]) -> None:
        """Save model checkpoint."""
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self._model.state_dict() if self._model else None,
            "config": self.config,
            "backbone_name": self.backbone_name,
            "pretrained": self.pretrained,
            "freeze_backbone": self.freeze_backbone,
            "input_size": self.input_size,
            "is_fitted": self.is_fitted,
        }

        torch.save(state, path)

        # Also save config as JSON for readability
        config_path = path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(self.get_config_for_paper(), f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        import torch

        state = torch.load(path, map_location="cpu", weights_only=False)

        self.config = state["config"]
        self.backbone_name = state["backbone_name"]
        self.pretrained = state["pretrained"]
        self.freeze_backbone = state["freeze_backbone"]
        self.input_size = state["input_size"]
        self.is_fitted = state["is_fitted"]

        # Re-initialize and load weights
        self._init_model()
        if state["model_state_dict"]:
            self._model.load_state_dict(state["model_state_dict"])
