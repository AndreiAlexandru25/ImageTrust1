"""
Image preprocessing for AI detection models.

Handles resizing, normalization, and conversion to PyTorch tensors.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from imagetrust.utils.image_utils import convert_to_rgb


class ImagePreprocessor:
    """
    Standard image preprocessor for AI detection models.
    
    Applies resizing, converts to RGB, converts to tensor, and normalizes
    pixel values using ImageNet statistics.
    
    Args:
        input_size: Target size for images.
        mean: Normalization mean (ImageNet defaults).
        std: Normalization std (ImageNet defaults).
        
    Example:
        >>> preprocessor = ImagePreprocessor(input_size=224)
        >>> image = Image.open("photo.jpg")
        >>> tensor = preprocessor.preprocess(image)
        >>> batch = preprocessor.preprocess_batch([img1, img2, img3])
    """

    def __init__(
        self,
        input_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.input_size = input_size
        self.mean = mean
        self.std = std

        # Standard transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        
        # Augmented transform for training
        self.train_transform = transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray],
        augment: bool = False,
    ) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image: PIL Image or numpy array.
            augment: Whether to apply data augmentation.
            
        Returns:
            Preprocessed tensor (C, H, W).
        """
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Ensure RGB
        image = convert_to_rgb(image)
        
        # Apply transforms
        transform = self.train_transform if augment else self.transform
        return transform(image)

    def preprocess_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        augment: bool = False,
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of PIL Images or numpy arrays.
            augment: Whether to apply data augmentation.
            
        Returns:
            Batch tensor (B, C, H, W).
        """
        tensors = [self.preprocess(img, augment=augment) for img in images]
        return torch.stack(tensors)

    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization to get pixel values back to [0, 1].
        
        Args:
            tensor: Normalized tensor.
            
        Returns:
            Denormalized tensor.
        """
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        
        if tensor.device != mean.device:
            mean = mean.to(tensor.device)
            std = std.to(tensor.device)
        
        return tensor * std + mean

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        augment: bool = False,
    ) -> torch.Tensor:
        """Allow preprocessor to be called directly."""
        return self.preprocess(image, augment=augment)

    def get_config(self) -> dict:
        """Get preprocessor configuration."""
        return {
            "input_size": self.input_size,
            "mean": self.mean,
            "std": self.std,
        }
