"""
B3: Modern Baseline - ViT-B/16 or CLIP Linear Probe.

Vision Transformer baseline representing modern architectures.
Can use either ViT fine-tuning or CLIP linear probe (frozen backbone).

For the paper, report:
- Architecture (ViT-B/16, CLIP ViT-B/32)
- Pretrained weights source (ImageNet-21k, OpenAI CLIP)
- Fine-tuning strategy (full, linear probe, last few layers)
- Optimizer, LR, scheduler (ViT typically needs lower LR)
- Batch size, epochs
- Training time per epoch
- GPU memory usage
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from imagetrust.baselines.base import BaselineDetector, BaselineConfig, BaselineResult


class ViTBaseline(BaselineDetector):
    """
    Vision Transformer baseline (ViT-B/16 or CLIP linear probe).

    Two modes:
    1. ViT fine-tuning: Full fine-tuning of ViT-B/16
    2. CLIP linear probe: Frozen CLIP features + learned classifier

    Example:
        >>> config = BaselineConfig(
        ...     name="ViT-B/16",
        ...     epochs=10,
        ...     batch_size=16,
        ...     learning_rate=1e-5,  # Lower for ViT
        ... )
        >>> baseline = ViTBaseline(config, architecture="vit")
        >>> history = baseline.fit(train_images, train_labels)
    """

    SUPPORTED_ARCHITECTURES = ["vit", "clip"]

    def __init__(
        self,
        config: BaselineConfig,
        architecture: str = "vit",
        model_name: Optional[str] = None,
        freeze_backbone: bool = False,
        input_size: int = 224,
    ):
        """
        Initialize ViT baseline.

        Args:
            config: Baseline configuration
            architecture: "vit" or "clip"
            model_name: Specific model name (default: auto-select)
            freeze_backbone: Freeze backbone (for linear probe)
            input_size: Input image size
        """
        super().__init__(config)

        if architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Supported: {self.SUPPORTED_ARCHITECTURES}"
            )

        self.architecture = architecture
        self.freeze_backbone = freeze_backbone
        self.input_size = input_size

        # Default model names
        if model_name is None:
            if architecture == "vit":
                model_name = "vit_base_patch16_224"
            else:  # clip
                model_name = "ViT-B/32"
        self.model_name = model_name

        # PyTorch components (lazy init)
        self._model = None
        self._classifier = None  # For CLIP linear probe
        self._device = None
        self._transform = None
        self._clip_model = None
        self._clip_preprocess = None

        # Store config for paper reporting
        self.config.model_params.update({
            "architecture": architecture,
            "model_name": model_name,
            "freeze_backbone": freeze_backbone,
            "input_size": input_size,
        })

    def _init_model(self) -> None:
        """Initialize model based on architecture."""
        import torch

        # Device selection
        if self.config.device == "auto" or self.config.device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(self.config.device)

        if self.architecture == "vit":
            self._init_vit()
        else:
            self._init_clip()

    def _init_vit(self) -> None:
        """Initialize ViT model."""
        import torch
        import torch.nn as nn
        import timm

        # Create ViT model
        self._model = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=2,
        )

        # Freeze backbone if requested
        if self.freeze_backbone:
            for name, param in self._model.named_parameters():
                if "head" not in name and "fc" not in name:
                    param.requires_grad = False

        self._model = self._model.to(self._device)

        # Create transforms
        self._transform = self._create_vit_transform(is_train=False)

    def _init_clip(self) -> None:
        """Initialize CLIP model for linear probe."""
        import torch
        import torch.nn as nn

        try:
            import clip
        except ImportError:
            raise ImportError(
                "CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git"
            )

        # Load CLIP model (frozen)
        self._clip_model, self._clip_preprocess = clip.load(
            self.model_name, device=self._device
        )
        self._clip_model.eval()

        # Freeze CLIP
        for param in self._clip_model.parameters():
            param.requires_grad = False

        # Get feature dimension
        with torch.no_grad():
            dummy_img = torch.zeros(1, 3, 224, 224).to(self._device)
            features = self._clip_model.encode_image(dummy_img)
            feature_dim = features.shape[1]

        # Create linear classifier
        self._classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        ).to(self._device)

        self._transform = self._clip_preprocess

    def _create_vit_transform(self, is_train: bool = False):
        """Create ViT transforms."""
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
        Train the ViT/CLIP baseline.

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

        # Create dataset
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
                return img, ds_self.labels[idx]

        # For ViT, use train transform; for CLIP, use CLIP's preprocess
        train_transform = (
            self._create_vit_transform(is_train=True)
            if self.architecture == "vit"
            else self._transform
        )

        train_dataset = ImageDataset(
            train_images, train_labels, train_transform, self._load_image
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self._device.type == "cuda" else False,
        )

        val_loader = None
        if val_images is not None and val_labels is not None:
            val_transform = self._transform
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

        if self.architecture == "vit":
            # ViT fine-tuning
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self._model.parameters()),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            # CLIP linear probe (only classifier is trainable)
            optimizer = torch.optim.AdamW(
                self._classifier.parameters(),
                lr=self.config.learning_rate * 10,  # Higher LR for linear probe
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
            if self.architecture == "vit":
                self._model.train()
            else:
                self._classifier.train()

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_imgs, batch_labels in train_loader:
                batch_imgs = batch_imgs.to(self._device)
                batch_labels = batch_labels.to(self._device)

                optimizer.zero_grad()

                if self.architecture == "vit":
                    outputs = self._model(batch_imgs)
                else:
                    # CLIP: extract features then classify
                    with torch.no_grad():
                        features = self._clip_model.encode_image(batch_imgs)
                        features = features.float()  # CLIP returns float16
                    outputs = self._classifier(features)

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

            # Log
            val_str = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}" if val_loss else ""
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}{val_str}"
            )

        self.is_fitted = True

        history["final_train_acc"] = history["train_acc"][-1]
        if history["val_acc"]:
            history["final_val_acc"] = history["val_acc"][-1]
        history["total_epochs"] = self.config.epochs
        history["train_samples"] = len(train_images)

        return history

    def _validate(self, val_loader, criterion) -> Tuple[float, float]:
        """Run validation."""
        import torch

        if self.architecture == "vit":
            self._model.eval()
        else:
            self._classifier.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(self._device)
                batch_labels = batch_labels.to(self._device)

                if self.architecture == "vit":
                    outputs = self._model(batch_imgs)
                else:
                    features = self._clip_model.encode_image(batch_imgs).float()
                    outputs = self._classifier(features)

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

        if self._model is None and self._clip_model is None:
            self._init_model()

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first or load() a checkpoint.")

        pil_img = self._load_image(image)

        def _predict():
            if self.architecture == "vit":
                self._model.eval()
                with torch.no_grad():
                    img_tensor = self._transform(pil_img).unsqueeze(0).to(self._device)
                    logits = self._model(img_tensor)
                    probs = F.softmax(logits, dim=1)
                    return probs.cpu().numpy()[0], logits.cpu().numpy()[0]
            else:
                self._classifier.eval()
                with torch.no_grad():
                    img_tensor = self._transform(pil_img).unsqueeze(0).to(self._device)
                    features = self._clip_model.encode_image(img_tensor).float()
                    logits = self._classifier(features)
                    probs = F.softmax(logits, dim=1)
                    return probs.cpu().numpy()[0], logits.cpu().numpy()[0]

        (probs, logits), elapsed_ms = self._timed_predict(_predict)

        ai_prob = float(probs[1])
        real_prob = float(probs[0])
        raw_prob = ai_prob

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
        """Batch prediction."""
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset

        if self._model is None and self._clip_model is None:
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

        if self.architecture == "vit":
            self._model.eval()
        else:
            self._classifier.eval()

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)

                if self.architecture == "vit":
                    logits = self._model(batch)
                else:
                    features = self._clip_model.encode_image(batch).float()
                    logits = self._classifier(features)

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
            "architecture": self.architecture,
            "model_name": self.model_name,
            "freeze_backbone": self.freeze_backbone,
            "input_size": self.input_size,
            "config": self.config,
            "is_fitted": self.is_fitted,
        }

        if self.architecture == "vit":
            state["model_state_dict"] = self._model.state_dict() if self._model else None
        else:
            state["classifier_state_dict"] = self._classifier.state_dict() if self._classifier else None

        torch.save(state, path)

        # Save config as JSON
        config_path = path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(self.get_config_for_paper(), f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """Load model checkpoint."""
        import torch

        state = torch.load(path, map_location="cpu")

        self.architecture = state["architecture"]
        self.model_name = state["model_name"]
        self.freeze_backbone = state["freeze_backbone"]
        self.input_size = state["input_size"]
        self.config = state["config"]
        self.is_fitted = state["is_fitted"]

        # Re-initialize and load weights
        self._init_model()

        if self.architecture == "vit" and state.get("model_state_dict"):
            self._model.load_state_dict(state["model_state_dict"])
        elif self.architecture == "clip" and state.get("classifier_state_dict"):
            self._classifier.load_state_dict(state["classifier_state_dict"])
