"""
Ensemble detector for combining multiple AI detection models.

Supports various ensemble strategies including averaging, weighted voting,
and learned stacking.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from imagetrust.detection.models.base import BaseDetector
from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


class EnsembleDetector(nn.Module):
    """
    Ensemble of AI detection models.
    
    Combines predictions from multiple detectors using various strategies.
    
    Strategies:
        - "average": Simple average of probabilities
        - "weighted": Weighted average with learnable or fixed weights
        - "voting": Hard voting based on individual predictions
        - "stacking": Learn a meta-classifier on detector outputs
        - "max": Take maximum confidence prediction
        
    Example:
        >>> detectors = [CNNDetector("convnext_base"), ViTDetector("vit_base")]
        >>> ensemble = EnsembleDetector(detectors, strategy="weighted")
        >>> logits = ensemble(image_tensor)
    """

    def __init__(
        self,
        detectors: List[BaseDetector],
        strategy: Literal["average", "weighted", "voting", "stacking", "max"] = "weighted",
        weights: Optional[List[float]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        
        self.detectors = nn.ModuleList(detectors)
        self.strategy = strategy
        self.num_detectors = len(detectors)
        
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self._device = torch.device(device)
        else:
            self._device = device
        
        # Initialize weights
        if weights is not None:
            assert len(weights) == self.num_detectors
            self.register_buffer(
                "weights",
                torch.tensor(weights, dtype=torch.float32)
            )
        else:
            # Equal weights by default
            self.weights = nn.Parameter(
                torch.ones(self.num_detectors) / self.num_detectors
            )
        
        # For stacking strategy
        if strategy == "stacking":
            # Meta-classifier takes all detector outputs
            num_classes = detectors[0].num_classes if detectors else 2
            self.meta_classifier = nn.Sequential(
                nn.Linear(self.num_detectors * num_classes, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            )
        
        # Get input size from first detector
        self.input_size = detectors[0].input_size if detectors else 224
        
        logger.info(
            f"EnsembleDetector created: {self.num_detectors} models, "
            f"strategy={strategy}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            Combined logits (B, num_classes)
        """
        # Collect predictions from all detectors
        all_logits = []
        for detector in self.detectors:
            with torch.no_grad() if not self.training else torch.enable_grad():
                logits = detector(x)
                all_logits.append(logits)
        
        # Stack: (num_detectors, B, num_classes)
        stacked = torch.stack(all_logits, dim=0)
        
        # Apply ensemble strategy
        if self.strategy == "average":
            probs = F.softmax(stacked, dim=-1)
            combined = probs.mean(dim=0)
            return torch.log(combined + 1e-8)  # Convert back to logits
        
        elif self.strategy == "weighted":
            probs = F.softmax(stacked, dim=-1)
            weights = F.softmax(self.weights, dim=0)
            # Weighted average: (num_detectors,) @ (num_detectors, B, C)
            combined = torch.einsum("d,dbc->bc", weights, probs)
            return torch.log(combined + 1e-8)
        
        elif self.strategy == "voting":
            # Hard voting
            predictions = stacked.argmax(dim=-1)  # (num_detectors, B)
            # Count votes for each class
            batch_size = x.size(0)
            num_classes = stacked.size(-1)
            votes = torch.zeros(batch_size, num_classes, device=x.device)
            for i in range(self.num_detectors):
                votes.scatter_add_(1, predictions[i].unsqueeze(1), 
                                   torch.ones(batch_size, 1, device=x.device))
            return votes  # Higher = more votes
        
        elif self.strategy == "stacking":
            # Concatenate all logits and pass through meta-classifier
            batch_size = x.size(0)
            stacked_flat = stacked.permute(1, 0, 2).reshape(batch_size, -1)
            return self.meta_classifier(stacked_flat)
        
        elif self.strategy == "max":
            # Take prediction from most confident detector
            probs = F.softmax(stacked, dim=-1)
            max_probs = probs.max(dim=-1)[0]  # (num_detectors, B)
            most_confident = max_probs.argmax(dim=0)  # (B,)
            
            # Gather the logits from most confident detector
            batch_indices = torch.arange(x.size(0), device=x.device)
            combined = stacked[most_confident, batch_indices]
            return combined
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def get_individual_predictions(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions from each individual detector.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary mapping detector name to its predictions
        """
        results = {}
        for i, detector in enumerate(self.detectors):
            name = getattr(detector, "backbone_name", f"detector_{i}")
            with torch.no_grad():
                logits = detector(x)
                probs = F.softmax(logits, dim=-1)
            results[name] = {
                "logits": logits,
                "probabilities": probs,
            }
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the ensemble."""
        return {
            "type": "ensemble",
            "strategy": self.strategy,
            "num_detectors": self.num_detectors,
            "detectors": [
                d.get_model_info() if hasattr(d, "get_model_info") else str(d)
                for d in self.detectors
            ],
            "weights": self.weights.tolist() if hasattr(self.weights, "tolist") else None,
        }

    def to(self, device: Union[str, torch.device]) -> "EnsembleDetector":
        """Move ensemble to device."""
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        return super().to(device)
