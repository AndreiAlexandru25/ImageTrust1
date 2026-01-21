"""
Visualization utilities for explainability.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imagetrust.core.types import PatchScore, ExplainabilityAnalysis
from imagetrust.utils.logging import get_logger
from imagetrust.utils.image_utils import image_to_base64, create_heatmap_overlay

logger = get_logger(__name__)


class ExplainabilityVisualizer:
    """
    Creates visualizations for AI detection explanations.
    
    Generates:
    - Heatmap overlays
    - Patch score visualizations
    - Frequency spectrum displays
    - Combined explanation panels
    
    Example:
        >>> visualizer = ExplainabilityVisualizer()
        >>> panel = visualizer.create_panel(image, analysis)
    """

    def __init__(
        self,
        colormap: str = "jet",
        alpha: float = 0.5,
    ) -> None:
        self.colormap = colormap
        self.alpha = alpha

    def create_heatmap_overlay(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Image.Image:
        """Create heatmap overlay on image."""
        return create_heatmap_overlay(
            image,
            heatmap,
            alpha=alpha or self.alpha,
            colormap=self.colormap,
        )

    def visualize_patches(
        self,
        image: Image.Image,
        patch_scores: List[PatchScore],
        show_scores: bool = True,
        threshold: float = 0.5,
    ) -> Image.Image:
        """
        Visualize patch scores on image.
        
        Args:
            image: Original image
            patch_scores: List of patch scores
            show_scores: Whether to show score values
            threshold: Threshold for highlighting
            
        Returns:
            Image with patch visualization
        """
        # Create copy
        vis_image = image.copy().convert("RGBA")
        draw = ImageDraw.Draw(vis_image)
        
        for patch in patch_scores:
            if patch.score < threshold:
                continue
            
            # Color based on score (green=low, red=high)
            intensity = int((patch.score - threshold) / (1 - threshold) * 255)
            color = (intensity, 255 - intensity, 0, 128)
            
            # Draw rectangle
            draw.rectangle(
                [patch.x, patch.y, patch.x + patch.width, patch.y + patch.height],
                outline=color,
                width=2,
            )
            
            # Draw score
            if show_scores:
                try:
                    draw.text(
                        (patch.x + 2, patch.y + 2),
                        f"{patch.score:.2f}",
                        fill=(255, 255, 255, 255),
                    )
                except Exception:
                    pass
        
        return vis_image.convert("RGB")

    def visualize_top_regions(
        self,
        image: Image.Image,
        top_regions: List[Dict[str, Any]],
    ) -> Image.Image:
        """Visualize top scoring regions."""
        vis_image = image.copy().convert("RGBA")
        draw = ImageDraw.Draw(vis_image)
        
        for i, region in enumerate(top_regions):
            # Color gradient from red (rank 1) to yellow
            r = 255
            g = min(255, i * 50)
            color = (r, g, 0, 200)
            
            draw.rectangle(
                [region["x"], region["y"], 
                 region["x"] + region["width"], region["y"] + region["height"]],
                outline=color,
                width=3,
            )
            
            # Label
            label = f"#{i+1}: {region['score']:.2f}"
            draw.text(
                (region["x"], region["y"] - 15),
                label,
                fill=(255, 255, 255, 255),
            )
        
        return vis_image.convert("RGB")

    def create_panel(
        self,
        image: Image.Image,
        analysis: ExplainabilityAnalysis,
        include_original: bool = True,
        include_heatmap: bool = True,
        include_patches: bool = True,
        panel_width: int = 1200,
    ) -> Image.Image:
        """
        Create a combined explanation panel.
        
        Args:
            image: Original image
            analysis: ExplainabilityAnalysis object
            include_original: Include original image
            include_heatmap: Include Grad-CAM heatmap
            include_patches: Include patch visualization
            panel_width: Width of output panel
            
        Returns:
            Combined panel image
        """
        panels = []
        
        # Original
        if include_original:
            panels.append(("Original", image))
        
        # Grad-CAM overlay
        if include_heatmap and analysis.gradcam_overlay:
            from imagetrust.utils.image_utils import base64_to_image
            try:
                overlay = base64_to_image(analysis.gradcam_overlay)
                panels.append(("Grad-CAM", overlay))
            except Exception as e:
                logger.warning(f"Failed to decode Grad-CAM overlay: {e}")
        
        # Patch visualization
        if include_patches and analysis.patch_scores:
            patch_vis = self.visualize_patches(image, analysis.patch_scores)
            panels.append(("Patch Analysis", patch_vis))
        
        # Top regions
        if analysis.top_regions:
            regions_vis = self.visualize_top_regions(image, analysis.top_regions)
            panels.append(("Top Regions", regions_vis))
        
        if not panels:
            return image
        
        # Calculate panel dimensions
        num_panels = len(panels)
        single_width = panel_width // num_panels
        aspect_ratio = image.height / image.width
        single_height = int(single_width * aspect_ratio)
        
        # Create combined panel
        combined = Image.new("RGB", (panel_width, single_height + 30), "white")
        draw = ImageDraw.Draw(combined)
        
        for i, (title, panel_img) in enumerate(panels):
            # Resize panel
            resized = panel_img.resize((single_width, single_height), Image.Resampling.LANCZOS)
            
            # Paste
            x_offset = i * single_width
            combined.paste(resized, (x_offset, 0))
            
            # Add title
            draw.text(
                (x_offset + 10, single_height + 5),
                title,
                fill="black",
            )
        
        return combined

    def get_panel_base64(
        self,
        image: Image.Image,
        analysis: ExplainabilityAnalysis,
        **kwargs,
    ) -> str:
        """Get panel as base64 string."""
        panel = self.create_panel(image, analysis, **kwargs)
        return image_to_base64(panel)
