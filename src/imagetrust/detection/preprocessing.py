"""
Image preprocessing for AI detection models.

Handles resizing, normalization, and conversion to PyTorch tensors.
Supports both torchvision transforms and Albumentations for advanced augmentation.
"""

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from imagetrust.utils.image_utils import convert_to_rgb

# Check for Albumentations availability
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


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


class AlbumentationsPreprocessor:
    """
    Advanced image preprocessor using Albumentations.

    Provides more sophisticated augmentation pipeline including:
    - JPEG compression simulation
    - Screenshot artifacts
    - Social media degradation

    Args:
        input_size: Target size for images.
        mean: Normalization mean (ImageNet defaults).
        std: Normalization std (ImageNet defaults).
        use_robustness_augmentation: Use robustness augmentor for training.
        social_media_prob: Probability of social media augmentation.
        screenshot_prob: Probability of screenshot augmentation.

    Example:
        >>> preprocessor = AlbumentationsPreprocessor(input_size=224)
        >>> image = Image.open("photo.jpg")
        >>> tensor = preprocessor.preprocess(image, mode="train")
    """

    def __init__(
        self,
        input_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_robustness_augmentation: bool = True,
        social_media_prob: float = 0.3,
        screenshot_prob: float = 0.2,
    ) -> None:
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError(
                "Albumentations is required for AlbumentationsPreprocessor. "
                "Install with: pip install albumentations"
            )

        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.use_robustness = use_robustness_augmentation

        # Try to use RobustnessAugmentor if available
        self._robustness_augmentor = None
        if use_robustness_augmentation:
            try:
                from imagetrust.detection.augmentation import RobustnessAugmentor
                self._robustness_augmentor = RobustnessAugmentor(
                    input_size=input_size,
                    mean=mean,
                    std=std,
                    social_media_prob=social_media_prob,
                    screenshot_prob=screenshot_prob,
                )
            except ImportError:
                pass

        # Build Albumentations transforms
        self._build_transforms()

    def _build_transforms(self) -> None:
        """Build Albumentations transform pipelines."""
        # Validation transform (minimal)
        self.val_transform = A.Compose([
            A.Resize(height=self.input_size, width=self.input_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

        # Training transform (standard augmentation)
        self.train_transform = A.Compose([
            A.RandomResizedCrop(
                height=self.input_size,
                width=self.input_size,
                scale=(0.8, 1.0),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03,
                p=0.5,
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            A.OneOf([
                A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
                A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
            ], p=0.3),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

        # Heavy augmentation (for robustness training)
        self.heavy_transform = A.Compose([
            A.RandomResizedCrop(
                height=self.input_size,
                width=self.input_size,
                scale=(0.7, 1.0),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                p=0.3,
            ),
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            ], p=0.5),
            # JPEG compression (crucial for robustness)
            A.ImageCompression(quality_lower=30, quality_upper=85, p=0.5),
            # Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.3),
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.3),
            # Downscale-upscale
            A.Downscale(scale_min=0.5, scale_max=0.75, p=0.2),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray],
        mode: str = "val",
    ) -> torch.Tensor:
        """
        Preprocess a single image.

        Args:
            image: PIL Image or numpy array.
            mode: Preprocessing mode ("val", "train", "heavy", "robustness").

        Returns:
            Preprocessed tensor (C, H, W).
        """
        # Convert to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        # Use robustness augmentor if available and requested
        if mode == "robustness" and self._robustness_augmentor is not None:
            return self._robustness_augmentor(image, mode="train")

        # Select transform
        if mode == "val":
            transform = self.val_transform
        elif mode == "train":
            transform = self.train_transform
        elif mode == "heavy":
            transform = self.heavy_transform
        else:
            transform = self.val_transform

        # Apply transform
        augmented = transform(image=image)
        return augmented["image"]

    def preprocess_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        mode: str = "val",
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.

        Args:
            images: List of PIL Images or numpy arrays.
            mode: Preprocessing mode.

        Returns:
            Batch tensor (B, C, H, W).
        """
        tensors = [self.preprocess(img, mode=mode) for img in images]
        return torch.stack(tensors)

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mode: str = "val",
    ) -> torch.Tensor:
        """Allow preprocessor to be called directly."""
        return self.preprocess(image, mode=mode)

    def get_config(self) -> dict:
        """Get preprocessor configuration."""
        return {
            "input_size": self.input_size,
            "mean": self.mean,
            "std": self.std,
            "use_robustness": self.use_robustness,
            "backend": "albumentations",
        }


def create_preprocessor(
    input_size: int = 224,
    backend: str = "torchvision",
    use_robustness: bool = False,
    **kwargs,
) -> Union[ImagePreprocessor, "AlbumentationsPreprocessor"]:
    """
    Factory function to create appropriate preprocessor.

    Args:
        input_size: Target image size.
        backend: "torchvision" or "albumentations".
        use_robustness: Enable robustness augmentation (albumentations only).
        **kwargs: Additional arguments passed to preprocessor.

    Returns:
        Preprocessor instance.
    """
    if backend == "albumentations":
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError(
                "Albumentations not available. "
                "Install with: pip install albumentations"
            )
        return AlbumentationsPreprocessor(
            input_size=input_size,
            use_robustness_augmentation=use_robustness,
            **kwargs,
        )
    else:
        return ImagePreprocessor(input_size=input_size, **kwargs)
