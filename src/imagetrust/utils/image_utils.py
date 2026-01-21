"""
Image processing utilities for ImageTrust.

Provides functions for common image manipulations and conversions.
"""

import base64
from io import BytesIO
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert a PIL Image to RGB mode.
    
    Args:
        image: Input PIL Image.
        
    Returns:
        RGB PIL Image.
    """
    if image.mode == "RGB":
        return image
    
    if image.mode == "RGBA":
        # Composite on white background
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    
    return image.convert("RGB")


def resize_image(
    image: Image.Image,
    size: Union[int, Tuple[int, int]],
    resample: int = Image.Resampling.LANCZOS,
    keep_aspect: bool = False,
) -> Image.Image:
    """
    Resize a PIL Image.
    
    Args:
        image: Input PIL Image.
        size: Target size (int for square, or (width, height) tuple).
        resample: Resampling filter.
        keep_aspect: Whether to maintain aspect ratio.
        
    Returns:
        Resized PIL Image.
    """
    if isinstance(size, int):
        size = (size, size)
    
    if keep_aspect:
        # Resize keeping aspect ratio
        image.thumbnail(size, resample)
        return image
    
    return image.resize(size, resample)


def normalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Normalize a tensor with ImageNet statistics.
    
    Args:
        tensor: Input tensor (C, H, W) or (B, C, H, W).
        mean: Normalization mean.
        std: Normalization std.
        
    Returns:
        Normalized tensor.
    """
    normalize = transforms.Normalize(mean=mean, std=std)
    
    if tensor.dim() == 3:
        return normalize(tensor)
    elif tensor.dim() == 4:
        return torch.stack([normalize(t) for t in tensor])
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Denormalize a tensor (reverse ImageNet normalization).
    
    Args:
        tensor: Normalized tensor.
        mean: Normalization mean used.
        std: Normalization std used.
        
    Returns:
        Denormalized tensor.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    return tensor * std + mean


def image_to_tensor(
    image: Image.Image,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Convert a PIL Image to a PyTorch Tensor.
    
    Args:
        image: Input PIL Image.
        normalize: Whether to apply normalization.
        mean: Normalization mean.
        std: Normalization std.
        
    Returns:
        PyTorch Tensor (C, H, W).
    """
    # Convert to RGB if needed
    image = convert_to_rgb(image)
    
    # Convert to tensor [0, 1]
    tensor = transforms.ToTensor()(image)
    
    # Normalize if requested
    if normalize:
        tensor = normalize_tensor(tensor, mean, std)
    
    return tensor


def tensor_to_image(
    tensor: torch.Tensor,
    denormalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Image.Image:
    """
    Convert a PyTorch Tensor to a PIL Image.
    
    Args:
        tensor: Input tensor (C, H, W) or (B, C, H, W).
        denormalize: Whether to reverse normalization.
        mean: Normalization mean used.
        std: Normalization std used.
        
    Returns:
        PIL Image.
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Move to CPU if needed
    tensor = tensor.cpu()
    
    # Denormalize if requested
    if denormalize:
        tensor = denormalize_tensor(tensor, mean, std)
    
    # Clamp to [0, 1]
    tensor = tensor.clamp(0, 1)
    
    # Convert to PIL
    return transforms.ToPILImage()(tensor)


def image_to_base64(
    image: Image.Image,
    format: str = "PNG",
    quality: int = 95,
) -> str:
    """
    Convert a PIL Image to a base64 encoded string.
    
    Args:
        image: Input PIL Image.
        format: Output format (PNG, JPEG).
        quality: JPEG quality.
        
    Returns:
        Base64 encoded string.
    """
    buffer = BytesIO()
    
    save_kwargs = {"format": format}
    if format.upper() == "JPEG":
        save_kwargs["quality"] = quality
    
    image.save(buffer, **save_kwargs)
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode("utf-8")


def base64_to_image(b64_string: str) -> Image.Image:
    """
    Convert a base64 encoded string to a PIL Image.
    
    Args:
        b64_string: Base64 encoded image string.
        
    Returns:
        PIL Image.
    """
    # Remove data URL prefix if present
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data))


def create_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """
    Create a heatmap overlay on an image.
    
    Args:
        image: Base PIL Image.
        heatmap: 2D numpy array (values 0-1).
        alpha: Overlay transparency.
        colormap: Matplotlib colormap name.
        
    Returns:
        PIL Image with heatmap overlay.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    # Ensure image is RGB
    image = convert_to_rgb(image)
    
    # Resize heatmap to image size
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            image.size, Image.Resampling.BILINEAR
        )
    ) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap_colored = (cmap(heatmap_resized) * 255).astype(np.uint8)[:, :, :3]
    heatmap_pil = Image.fromarray(heatmap_colored)
    
    # Blend images
    return Image.blend(image, heatmap_pil, alpha)
