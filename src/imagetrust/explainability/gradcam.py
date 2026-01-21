"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Visualizes which regions of an image contribute most to AI detection.

This module provides:
1. Standard Grad-CAM
2. Grad-CAM++ (improved version)
3. Multi-layer analysis
4. Heatmap overlay generation
"""

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import io
import base64


@dataclass
class GradCAMResult:
    """Result of Grad-CAM analysis."""
    heatmap: np.ndarray  # Raw heatmap (0-1)
    overlay: Image.Image  # Overlaid on original
    highlighted_regions: List[Dict]  # Top suspicious regions
    activation_score: float  # Overall activation
    layer_name: str  # Which layer was used


class GradCAMAnalyzer:
    """
    Generates Grad-CAM visualizations for AI detection models.
    
    Supports multiple backends:
    - HuggingFace transformers
    - Custom CNN models
    - PyTorch models
    """
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.gradients = None
        self.activations = None
        
    def load_model(self, model_id: str = "umm-maybe/AI-image-detector"):
        """Load a HuggingFace model for Grad-CAM analysis."""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageClassification.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            
            # Register hooks for gradient capture
            self._register_hooks()
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture gradients and activations."""
        if self.model is None:
            return
        
        # Find the last convolutional layer
        self.target_layer = None
        
        # For ViT models
        if hasattr(self.model, 'vit'):
            # Use the last encoder layer
            self.target_layer = self.model.vit.encoder.layer[-1]
        elif hasattr(self.model, 'base_model'):
            # Generic transformer
            for name, module in self.model.base_model.named_modules():
                if 'layer' in name.lower() or 'block' in name.lower():
                    self.target_layer = module
        
        if self.target_layer is None:
            # Fallback: use the model itself
            self.target_layer = self.model
        
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def analyze(self, image: Image.Image, target_class: int = None) -> GradCAMResult:
        """
        Generate Grad-CAM visualization for an image.
        
        Args:
            image: PIL Image to analyze
            target_class: Class index to visualize (default: predicted class)
            
        Returns:
            GradCAMResult with heatmap and overlay
        """
        if self.model is None:
            # Use fallback analysis without model
            return self._fallback_analysis(image)
        
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Enable gradients
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()
        
        # Backward pass
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Generate heatmap
        if self.gradients is not None and self.activations is not None:
            heatmap = self._compute_gradcam(self.gradients, self.activations)
        else:
            # Fallback to frequency-based heatmap
            heatmap = self._frequency_based_heatmap(image)
        
        # Resize heatmap to image size
        heatmap_resized = self._resize_heatmap(heatmap, image.size)
        
        # Create overlay
        overlay = self._create_overlay(image, heatmap_resized)
        
        # Find highlighted regions
        regions = self._find_suspicious_regions(heatmap_resized)
        
        # Calculate activation score
        activation_score = float(np.mean(heatmap_resized[heatmap_resized > 0.5]))
        
        return GradCAMResult(
            heatmap=heatmap_resized,
            overlay=overlay,
            highlighted_regions=regions,
            activation_score=activation_score if not np.isnan(activation_score) else 0.5,
            layer_name=str(type(self.target_layer).__name__) if self.target_layer else "fallback"
        )
    
    def _compute_gradcam(self, gradients: torch.Tensor, activations: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM heatmap from gradients and activations."""
        # Global average pooling of gradients
        if len(gradients.shape) == 4:
            # CNN-style: (B, C, H, W)
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
        elif len(gradients.shape) == 3:
            # Transformer-style: (B, N, D)
            weights = torch.mean(gradients, dim=1, keepdim=True)
            cam = torch.sum(weights * activations, dim=2)
            
            # Reshape to 2D (assume square)
            n_tokens = cam.shape[1]
            side = int(np.sqrt(n_tokens))
            if side * side == n_tokens:
                cam = cam.reshape(1, 1, side, side)
            else:
                # Handle CLS token
                side = int(np.sqrt(n_tokens - 1))
                cam = cam[:, 1:].reshape(1, 1, side, side)
        else:
            # Fallback
            cam = gradients.mean(dim=-1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        if cam.size > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros((14, 14))
        
        return cam
    
    def _fallback_analysis(self, image: Image.Image) -> GradCAMResult:
        """Fallback analysis using frequency-based method when model unavailable."""
        heatmap = self._frequency_based_heatmap(image)
        heatmap_resized = self._resize_heatmap(heatmap, image.size)
        overlay = self._create_overlay(image, heatmap_resized)
        regions = self._find_suspicious_regions(heatmap_resized)
        
        return GradCAMResult(
            heatmap=heatmap_resized,
            overlay=overlay,
            highlighted_regions=regions,
            activation_score=float(np.mean(heatmap_resized)),
            layer_name="frequency_fallback"
        )
    
    def _frequency_based_heatmap(self, image: Image.Image) -> np.ndarray:
        """Generate heatmap based on frequency analysis (AI artifact detection)."""
        # Convert to grayscale numpy array
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_array = np.array(image.convert('L'))
        
        # Divide into blocks
        block_size = 32
        h, w = img_array.shape
        heatmap = np.zeros((h // block_size + 1, w // block_size + 1))
        
        for i, y in enumerate(range(0, h - block_size, block_size)):
            for j, x in enumerate(range(0, w - block_size, block_size)):
                block = img_array[y:y+block_size, x:x+block_size]
                
                # FFT of block
                fft = np.fft.fft2(block)
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.abs(fft_shift)
                
                # High frequency energy ratio (AI artifacts)
                center = block_size // 2
                high_freq_mask = np.ones_like(magnitude, dtype=bool)
                high_freq_mask[center-4:center+4, center-4:center+4] = False
                
                high_energy = np.sum(magnitude[high_freq_mask])
                total_energy = np.sum(magnitude) + 1e-10
                
                # Smoothness measure (AI tends to be smoother)
                smoothness = 1.0 - (np.std(block) / 128.0)
                
                # Combined score
                heatmap[i, j] = (high_energy / total_energy) * 0.5 + smoothness * 0.5
        
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def _resize_heatmap(self, heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize heatmap to target size with smooth interpolation."""
        from scipy.ndimage import zoom
        
        target_w, target_h = target_size
        current_h, current_w = heatmap.shape
        
        zoom_h = target_h / current_h
        zoom_w = target_w / current_w
        
        resized = zoom(heatmap, (zoom_h, zoom_w), order=1)
        
        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        resized = gaussian_filter(resized, sigma=target_h // 50)
        
        return resized
    
    def _create_overlay(self, image: Image.Image, heatmap: np.ndarray) -> Image.Image:
        """Create visualization overlay of heatmap on original image."""
        # Convert heatmap to colormap
        import matplotlib.cm as cm
        
        # Use 'jet' colormap (blue -> green -> yellow -> red)
        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(heatmap)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        
        heatmap_img = Image.fromarray(heatmap_colored)
        heatmap_img = heatmap_img.resize(image.size, Image.Resampling.LANCZOS)
        
        # Convert original to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Blend with original
        overlay = Image.blend(image, heatmap_img, alpha=0.4)
        
        return overlay
    
    def _find_suspicious_regions(self, heatmap: np.ndarray, num_regions: int = 5) -> List[Dict]:
        """Find the most suspicious (high activation) regions."""
        regions = []
        
        # Threshold for suspicious regions
        threshold = np.percentile(heatmap, 90)
        
        # Find connected components above threshold
        from scipy import ndimage
        
        binary = heatmap > threshold
        labeled, num_features = ndimage.label(binary)
        
        # Get region properties
        for i in range(1, min(num_features + 1, num_regions + 1)):
            region_mask = labeled == i
            region_coords = np.where(region_mask)
            
            if len(region_coords[0]) > 0:
                y_min, y_max = region_coords[0].min(), region_coords[0].max()
                x_min, x_max = region_coords[1].min(), region_coords[1].max()
                
                avg_activation = float(np.mean(heatmap[region_mask]))
                
                regions.append({
                    "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                    "center": [int((x_min + x_max) // 2), int((y_min + y_max) // 2)],
                    "activation": avg_activation,
                    "area": int(np.sum(region_mask)),
                    "description": self._describe_region(avg_activation)
                })
        
        # Sort by activation
        regions.sort(key=lambda x: x["activation"], reverse=True)
        
        return regions[:num_regions]
    
    def _describe_region(self, activation: float) -> str:
        """Generate human-readable description of region."""
        if activation > 0.8:
            return "Highly suspicious - strong AI artifacts detected"
        elif activation > 0.6:
            return "Moderately suspicious - possible AI generation"
        elif activation > 0.4:
            return "Slightly suspicious - minor anomalies"
        else:
            return "Low suspicion - appears natural"


class GradCAMPlusPlus(GradCAMAnalyzer):
    """
    Grad-CAM++ implementation for improved visualization.
    Better handles multiple instances and provides smoother heatmaps.
    """
    
    def _compute_gradcam(self, gradients: torch.Tensor, activations: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM++ heatmap."""
        if len(gradients.shape) == 4:
            # CNN-style
            # Grad-CAM++ weights
            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)
            
            sum_activations = torch.sum(activations, dim=[2, 3], keepdim=True)
            alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + 1e-10)
            
            weights = torch.sum(alpha * F.relu(gradients), dim=[2, 3], keepdim=True)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
        else:
            # Fallback to standard Grad-CAM
            return super()._compute_gradcam(gradients, activations)
        
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        if cam.size > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros((14, 14))
        
        return cam


def analyze_with_gradcam(image: Image.Image, use_model: bool = True) -> GradCAMResult:
    """
    Convenience function to analyze image with Grad-CAM.
    
    Args:
        image: PIL Image to analyze
        use_model: Whether to try loading ML model (slower but more accurate)
        
    Returns:
        GradCAMResult with heatmap and overlay
    """
    analyzer = GradCAMAnalyzer()
    
    if use_model:
        success = analyzer.load_model()
        if not success:
            print("Warning: Could not load model, using fallback analysis")
    
    return analyzer.analyze(image)


def heatmap_to_base64(heatmap_image: Image.Image) -> str:
    """Convert heatmap image to base64 for web display."""
    buffer = io.BytesIO()
    heatmap_image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')
