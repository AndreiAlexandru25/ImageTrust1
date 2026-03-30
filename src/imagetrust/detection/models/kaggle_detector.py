"""
Custom-trained Deepfake Detector from Kaggle training pipeline.

Reproduces the exact DeepfakeDetector architecture trained on
140k images (Fake vs Real) with ResNet50 backbone, attention,
SE blocks, and multi-sample dropout.

Trained model: best_model.pth (~97% accuracy, 99.6% AUC)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    resnet101,
    ResNet101_Weights,
    efficientnet_v2_m,
    EfficientNet_V2_M_Weights,
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
    convnext_base,
    ConvNeXt_Base_Weights,
    convnext_large,
    ConvNeXt_Large_Weights,
)
from PIL import Image

from imagetrust.detection.models.base import BaseDetector
from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


class KaggleDeepfakeDetector(BaseDetector):
    """
    Production-grade Deepfake Detector trained on Kaggle.

    Architecture matches the training script exactly:
    - ResNet50 backbone (or other supported backbones)
    - Attention-based feature refinement
    - Squeeze-and-Excitation (SE) channel attention
    - Multi-layer classifier head with LayerNorm + GELU
    - Multi-sample dropout for training regularization

    Expected checkpoint format (best_model.pth):
        {
            "model_state_dict": ...,
            "config": {"backbone": "resnet50", "num_classes": 2, ...},
            "best_val_acc": 97.10,
            "best_val_auc": 99.61,
            "class_names": ["Fake", "Real"],
        }
    """

    BACKBONES = {
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1, 2048),
        "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V1, 2048),
        "efficientnet_v2_m": (
            efficientnet_v2_m,
            EfficientNet_V2_M_Weights.IMAGENET1K_V1,
            1280,
        ),
        "efficientnet_v2_l": (
            efficientnet_v2_l,
            EfficientNet_V2_L_Weights.IMAGENET1K_V1,
            1280,
        ),
        "convnext_base": (convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024),
        "convnext_large": (
            convnext_large,
            ConvNeXt_Large_Weights.IMAGENET1K_V1,
            1536,
        ),
    }

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        pretrained: bool = False,
        use_attention: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint: Optional[Path] = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            device=device,
            checkpoint=None,  # We handle loading ourselves
        )

        self.dropout_rate = dropout_rate
        self.use_attention = use_attention

        if backbone not in self.BACKBONES:
            raise ValueError(
                f"Unknown backbone: {backbone}. "
                f"Choose from {list(self.BACKBONES.keys())}"
            )

        model_fn, weights, num_features = self.BACKBONES[backbone]
        self.num_features = num_features

        # Load backbone (no pretrained weights - we load from checkpoint)
        weights_arg = weights if pretrained else None
        self._backbone = model_fn(weights=weights_arg)

        # Remove original classifier
        if "resnet" in backbone:
            self._backbone.fc = nn.Identity()
        elif "efficientnet" in backbone:
            self._backbone.classifier = nn.Identity()
        elif "convnext" in backbone:
            self._backbone.classifier = nn.Identity()

        # Attention-based feature refinement
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(num_features, num_features // 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(num_features // 4, num_features),
                nn.Sigmoid(),
            )

        # SE block for channel attention
        self.se = nn.Sequential(
            nn.Linear(num_features, num_features // 16),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // 16, num_features),
            nn.Sigmoid(),
        )

        # Classification head (exact match to training script)
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
            nn.Linear(256, num_classes),
        )

        # Multi-sample dropout layers for training
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(5)]
        )

        self.input_size = 224

        # Load checkpoint if provided
        if checkpoint:
            self._load_kaggle_checkpoint(checkpoint)

        logger.info(
            f"KaggleDeepfakeDetector created: {backbone}, "
            f"features={num_features}, attention={use_attention}"
        )

    def _load_kaggle_checkpoint(self, checkpoint_path: Union[Path, str]) -> None:
        """Load weights from Kaggle training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading Kaggle checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self._device, weights_only=False
        )

        # Extract model state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # The training script uses self.backbone, but we use self._backbone
        # to avoid conflict with BaseDetector.backbone_name.
        # Remap keys: "backbone.xxx" -> "_backbone.xxx"
        remapped = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                new_key = "_backbone." + key[len("backbone."):]
                remapped[new_key] = value
            else:
                remapped[key] = value

        self.load_state_dict(remapped, strict=False)

        # Log checkpoint info
        if "best_val_acc" in checkpoint:
            logger.info(
                f"Checkpoint stats: Acc={checkpoint['best_val_acc']:.2f}%, "
                f"AUC={checkpoint.get('best_val_auc', 0):.2f}%, "
                f"F1={checkpoint.get('best_val_f1', 0):.2f}%"
            )
        if "class_names" in checkpoint:
            logger.info(f"Class names: {checkpoint['class_names']}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Logits [B, num_classes]
        """
        features = self._backbone(x)

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

        # Standard classification
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        features = self._backbone(x)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return features

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate feature maps (for Grad-CAM)."""
        if hasattr(self._backbone, "layer4"):
            # ResNet: get output of last residual block
            x = self._backbone.conv1(x)
            x = self._backbone.bn1(x)
            x = self._backbone.relu(x)
            x = self._backbone.maxpool(x)
            x = self._backbone.layer1(x)
            x = self._backbone.layer2(x)
            x = self._backbone.layer3(x)
            x = self._backbone.layer4(x)
            return x
        return self._backbone(x)

    def get_target_layer(self):
        """Get target layer for Grad-CAM."""
        if hasattr(self._backbone, "layer4"):
            return self._backbone.layer4
        elif hasattr(self._backbone, "features"):
            return self._backbone.features[-1]
        return None

    def predict_image(self, image: Image.Image) -> tuple:
        """
        Predict on a PIL Image directly.

        Args:
            image: PIL RGB Image

        Returns:
            (ai_probability, real_probability)
        """
        from torchvision import transforms

        preprocess = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = preprocess(image).unsqueeze(0).to(self._device)

        self.eval()
        with torch.no_grad():
            logits = self.forward(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        # Training used classes: ["Fake", "Real"]
        # Index 0 = Fake (AI), Index 1 = Real
        ai_prob = probs[0].item()
        real_prob = probs[1].item()

        return ai_prob, real_prob

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = super().get_model_info()
        info.update({
            "type": "kaggle_trained",
            "use_attention": self.use_attention,
            "dropout_rate": self.dropout_rate,
            "feature_dim": self.num_features,
        })
        return info


def load_kaggle_model(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    backbone: str = "resnet50",
) -> KaggleDeepfakeDetector:
    """
    Convenience function to load the Kaggle-trained model.

    Args:
        checkpoint_path: Path to best_model.pth from Kaggle
        device: Device to load on (auto-detect if None)
        backbone: Backbone architecture (must match training)

    Returns:
        Ready-to-use KaggleDeepfakeDetector in eval mode
    """
    checkpoint_path = Path(checkpoint_path)

    # Try to read backbone from checkpoint config
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "config" in ckpt and isinstance(ckpt["config"], dict):
            backbone = ckpt["config"].get("backbone", backbone)
        del ckpt
    except Exception:
        pass

    model = KaggleDeepfakeDetector(
        backbone=backbone,
        num_classes=2,
        pretrained=False,
        use_attention=True,
        device=device,
        checkpoint=checkpoint_path,
    )
    model.to(model._device)
    model.eval()

    return model
