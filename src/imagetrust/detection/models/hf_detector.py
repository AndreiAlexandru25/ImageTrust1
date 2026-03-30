"""
Hugging Face pretrained AI detection model.

Uses models specifically trained to detect AI-generated images.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from imagetrust.detection.models.base import BaseDetector
from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceDetector(BaseDetector):
    """
    AI Image Detector using Hugging Face pretrained models.
    
    This uses models specifically trained to distinguish between
    AI-generated and real photographs.
    
    Recommended models:
        - "umm-maybe/AI-image-detector" (best accuracy)
        - "Organika/sdxl-detector"
    """

    # Mapping of friendly names to HF model IDs
    MODEL_REGISTRY = {
        "ai-detector": "umm-maybe/AI-image-detector",
        "sdxl-detector": "Organika/sdxl-detector", 
        "aiornot": "arnabdhar/ai-or-not-image-classifier",
    }

    def __init__(
        self,
        model_name: str = "ai-detector",
        device: Optional[Union[str, torch.device]] = None,
        checkpoint: Optional[Path] = None,
    ) -> None:
        # Get HF model ID
        if model_name in self.MODEL_REGISTRY:
            hf_model_id = self.MODEL_REGISTRY[model_name]
        else:
            hf_model_id = model_name  # Assume it's a full HF model ID
        
        super().__init__(
            backbone=hf_model_id,
            num_classes=2,
            pretrained=True,
            device=device,
            checkpoint=checkpoint,
        )
        
        self.hf_model_id = hf_model_id
        self.input_size = 224
        
        # Load processor and model from Hugging Face
        logger.info(f"Loading HuggingFace model: {hf_model_id}")
        
        self.processor = AutoImageProcessor.from_pretrained(hf_model_id)
        self.model = AutoModelForImageClassification.from_pretrained(hf_model_id)
        
        # Move to device
        self.model.to(self._device)
        self.model.eval()
        
        # Get label mapping
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        # Find AI and Real indices
        self._ai_idx = self._find_ai_index()
        self._real_idx = 1 - self._ai_idx
        
        logger.info(f"HuggingFaceDetector loaded: {hf_model_id}")
        logger.info(f"Labels: {self.id2label}, AI index: {self._ai_idx}")

    def _find_ai_index(self) -> int:
        """Find which output index corresponds to AI-generated."""
        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if any(kw in label_lower for kw in ["ai", "artificial", "generated", "fake", "synthetic"]):
                return int(idx)
        # Default: assume index 1 is AI
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - expects preprocessed tensor.
        
        Note: For HF models, use predict() or predict_image() instead.
        """
        with torch.no_grad():
            outputs = self.model(x)
            logits = outputs.logits
            
            # Ensure AI is at index 1 for consistency
            if self._ai_idx == 0:
                # Swap columns
                logits = logits[:, [1, 0]]
            
            return logits

    def predict_image(self, image: Image.Image) -> Tuple[float, float]:
        """
        Predict AI probability for a PIL Image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (ai_probability, real_probability)
        """
        # Preprocess with HF processor
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            
            ai_prob = probs[self._ai_idx].item()
            real_prob = probs[self._real_idx].item()
            
        return ai_prob, real_prob

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the model."""
        with torch.no_grad():
            # Get hidden states if available
            outputs = self.model(x, output_hidden_states=True)
            if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                return outputs.hidden_states[-1]
            return outputs.logits

    def get_target_layer(self):
        """Get target layer for Grad-CAM."""
        # For ViT-based models
        if hasattr(self.model, "vit"):
            return self.model.vit.encoder.layer[-1]
        # For CNN-based models
        if hasattr(self.model, "convnext"):
            return self.model.convnext.encoder.stages[-1]
        if hasattr(self.model, "resnet"):
            return self.model.resnet.layer4[-1]
        return None

    def __repr__(self) -> str:
        return f"HuggingFaceDetector(model='{self.hf_model_id}')"
