"""
Base detector class for AI detection models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from PIL import Image


class BaseDetector(ABC, nn.Module):
    """
    Abstract base class for all AI detection models.
    
    Defines the interface that all detector implementations must follow.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int = 2,
        pretrained: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint
        self.input_size = 224  # Default, can be overridden
        self.version = "1.0"
        
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self._device = torch.device(device)
        else:
            self._device = device

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Logits tensor (B, num_classes)
        """
        pass

    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Feature tensor
        """
        pass

    def predict(
        self,
        image: Union[Image.Image, torch.Tensor],
        preprocessor=None,
    ) -> Dict[str, Any]:
        """
        Make a prediction on a single image.
        
        Args:
            image: PIL Image or preprocessed tensor
            preprocessor: Optional preprocessor to use
            
        Returns:
            Dictionary with prediction results
        """
        self.eval()
        
        with torch.no_grad():
            if isinstance(image, Image.Image):
                if preprocessor is None:
                    raise ValueError("Preprocessor required for PIL Image input")
                tensor = preprocessor(image).unsqueeze(0).to(self._device)
            else:
                tensor = image.to(self._device)
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
            
            logits = self.forward(tensor)
            probs = torch.softmax(logits, dim=1)
            
            ai_prob = probs[0, 1].item()
            real_prob = probs[0, 0].item()
        
        return {
            "ai_probability": ai_prob,
            "real_probability": real_prob,
            "logits": logits[0].cpu().numpy(),
        }

    def load_weights(self, checkpoint_path: Union[Path, str]) -> None:
        """
        Load model weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        state_dict = torch.load(
            checkpoint_path, map_location=self._device, weights_only=False
        )
        
        # Handle different checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        self.load_state_dict(state_dict, strict=False)

    def save_weights(self, checkpoint_path: Union[Path, str]) -> None:
        """
        Save model weights to a checkpoint.
        
        Args:
            checkpoint_path: Path to save the checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.state_dict(),
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "version": self.version,
        }, checkpoint_path)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "version": self.version,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def get_input_size(self) -> int:
        """Get the expected input size for this model."""
        return self.input_size

    def to(self, device: Union[str, torch.device]) -> "BaseDetector":
        """Move model to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return super().to(device)
