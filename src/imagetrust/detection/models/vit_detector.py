"""
Vision Transformer-based detectors for AI image detection.

Supports ViT, DeiT, Swin Transformer via timm library.
"""

from pathlib import Path
from typing import Optional, Union

import timm
import torch
import torch.nn as nn

from imagetrust.detection.models.base import BaseDetector
from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


class ViTDetector(BaseDetector):
    """
    Vision Transformer-based AI image detector.
    
    Uses timm library to load various ViT architectures.
    
    Supported backbones:
        - vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
        - vit_large_patch16_224, vit_huge_patch14_224
        - deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224
        - swin_tiny_patch4_window7_224, swin_small_patch4_window7_224
        - swin_base_patch4_window7_224, swin_large_patch4_window12_384
        
    Example:
        >>> detector = ViTDetector(backbone="vit_base_patch16_224")
        >>> logits = detector(image_tensor)
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint: Optional[Path] = None,
        drop_rate: float = 0.1,
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
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(drop_rate),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, num_classes),
        )
        
        # Set input size based on backbone
        self.input_size = self._get_input_size(backbone)
        
        # Load checkpoint if provided
        if checkpoint:
            self.load_weights(checkpoint)
        
        logger.info(f"ViTDetector created: {backbone}, features={self.feature_dim}")

    def _get_input_size(self, backbone: str) -> int:
        """Get default input size for backbone."""
        if "384" in backbone:
            return 384
        elif "512" in backbone:
            return 512
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

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps from the transformer.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention maps
        """
        # This requires hooks for most ViT implementations
        # Simplified version - returns None if not supported
        if hasattr(self.backbone, "get_attention_map"):
            return self.backbone.get_attention_map(x)
        return None

    def get_target_layer(self):
        """Get target layer for attention visualization."""
        # For ViT models
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1]
        # For Swin
        elif hasattr(self.backbone, "layers"):
            return self.backbone.layers[-1]
        return None
