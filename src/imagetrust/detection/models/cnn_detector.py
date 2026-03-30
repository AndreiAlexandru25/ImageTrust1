"""
CNN-based detectors for AI image detection.

Supports various CNN backbones via timm library.
"""

from pathlib import Path
from typing import Optional, Union

import timm
import torch
import torch.nn as nn

from imagetrust.detection.models.base import BaseDetector
from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


class CNNDetector(BaseDetector):
    """
    CNN-based AI image detector.
    
    Uses timm library to load various pretrained backbones.
    
    Supported backbones:
        - efficientnet_b0 to efficientnet_b7
        - resnet18, resnet34, resnet50, resnet101, resnet152
        - convnext_tiny, convnext_small, convnext_base, convnext_large
        - densenet121, densenet169, densenet201
        - And many more from timm
        
    Example:
        >>> detector = CNNDetector(backbone="convnext_base")
        >>> logits = detector(image_tensor)
    """

    def __init__(
        self,
        backbone: str = "convnext_base",
        num_classes: int = 2,
        pretrained: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint: Optional[Path] = None,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            device=device,
            checkpoint=checkpoint,
        )
        
        self.drop_rate = drop_rate
        
        # Load backbone from timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            drop_rate=drop_rate,
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if self._needs_pooling(backbone) else nn.Identity(),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(512, num_classes),
        )
        
        # Set input size based on backbone
        self.input_size = self._get_input_size(backbone)
        
        # Load checkpoint if provided
        if checkpoint:
            self.load_weights(checkpoint)
        
        logger.info(f"CNNDetector created: {backbone}, features={self.feature_dim}")

    def _needs_pooling(self, backbone: str) -> bool:
        """Check if backbone output needs pooling."""
        # Most timm models return pooled features, but some don't
        return False  # timm handles this

    def _get_input_size(self, backbone: str) -> int:
        """Get default input size for backbone."""
        # EfficientNet variants
        if "efficientnet_b0" in backbone:
            return 224
        elif "efficientnet_b1" in backbone:
            return 240
        elif "efficientnet_b2" in backbone:
            return 260
        elif "efficientnet_b3" in backbone:
            return 300
        elif "efficientnet_b4" in backbone:
            return 380
        elif "efficientnet_b5" in backbone:
            return 456
        elif "efficientnet_b6" in backbone:
            return 528
        elif "efficientnet_b7" in backbone:
            return 600
        # ConvNeXt
        elif "convnext" in backbone:
            return 224
        # ResNet
        elif "resnet" in backbone:
            return 224
        # Default
        return 224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Logits (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Feature tensor (B, feature_dim)
        """
        return self.backbone(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate feature maps (for Grad-CAM).
        
        Args:
            x: Input tensor
            
        Returns:
            Feature maps before pooling
        """
        # Use forward_features if available
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)
        return self.backbone(x)

    def get_target_layer(self):
        """Get target layer for Grad-CAM."""
        # Try common layer names
        if hasattr(self.backbone, "stages"):
            return self.backbone.stages[-1]
        elif hasattr(self.backbone, "layer4"):
            return self.backbone.layer4
        elif hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1]
        elif hasattr(self.backbone, "features"):
            return self.backbone.features[-1]
        return None
