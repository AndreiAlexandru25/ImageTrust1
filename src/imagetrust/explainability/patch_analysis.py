"""
Patch-level analysis for AI detection.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from imagetrust.core.types import PatchScore
from imagetrust.utils.logging import get_logger
from imagetrust.utils.image_utils import resize_image

logger = get_logger(__name__)


class PatchAnalyzer:
    """
    Analyzes images at the patch level.
    
    Divides images into patches and scores each patch
    for AI-generated content.
    
    Example:
        >>> analyzer = PatchAnalyzer(detector)
        >>> scores, top_regions = analyzer.analyze(image)
    """

    def __init__(
        self,
        detector,
        patch_size: int = 64,
        stride: int = 32,
        min_patch_score: float = 0.6,
    ) -> None:
        self.detector = detector
        self.patch_size = patch_size
        self.stride = stride
        self.min_patch_score = min_patch_score

    def analyze(
        self,
        image: Image.Image,
        target_class: int = 1,
        batch_size: int = 16,
        verbose: bool = False,
    ) -> Tuple[List[PatchScore], List[Dict[str, Any]]]:
        """
        Analyze image by patches.
        
        Args:
            image: Input PIL Image
            target_class: Class to score (1 = AI-generated)
            batch_size: Batch size for processing
            verbose: Show progress
            
        Returns:
            Tuple of (patch scores, top regions)
        """
        width, height = image.size
        
        # Extract patches
        patches = []
        coords = []
        
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
                patches.append(patch)
                coords.append((x, y))
        
        if not patches:
            return [], []
        
        # Score patches
        scores = []
        iterator = range(0, len(patches), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Analyzing patches")
        
        for i in iterator:
            batch = patches[i:i + batch_size]
            batch_scores = self._score_batch(batch, target_class)
            scores.extend(batch_scores)
        
        # Create PatchScore objects
        patch_scores = []
        for (x, y), score in zip(coords, scores):
            patch_scores.append(PatchScore(
                x=x,
                y=y,
                width=self.patch_size,
                height=self.patch_size,
                score=score,
            ))
        
        # Find top regions
        top_regions = self._find_top_regions(patch_scores)
        
        return patch_scores, top_regions

    def _score_batch(
        self,
        patches: List[Image.Image],
        target_class: int,
    ) -> List[float]:
        """Score a batch of patches."""
        scores = []
        
        for patch in patches:
            try:
                # Resize patch to model input size
                resized = resize_image(patch, self.detector.preprocessor.input_size)
                result = self.detector.detect(resized, use_calibration=False)
                
                if target_class == 1:
                    scores.append(result["ai_probability"])
                else:
                    scores.append(result["real_probability"])
                    
            except Exception as e:
                logger.warning(f"Patch scoring failed: {e}")
                scores.append(0.5)
        
        return scores

    def _find_top_regions(
        self,
        patch_scores: List[PatchScore],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find top scoring regions."""
        # Sort by score
        sorted_patches = sorted(patch_scores, key=lambda p: p.score, reverse=True)
        
        # Get top N above threshold
        top_regions = []
        for patch in sorted_patches[:top_n]:
            if patch.score >= self.min_patch_score:
                top_regions.append({
                    "x": patch.x,
                    "y": patch.y,
                    "width": patch.width,
                    "height": patch.height,
                    "score": patch.score,
                    "center": patch.center,
                })
        
        return top_regions

    def create_score_map(
        self,
        patch_scores: List[PatchScore],
        image_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Create a 2D score map from patch scores.
        
        Args:
            patch_scores: List of patch scores
            image_size: Original image size (width, height)
            
        Returns:
            2D numpy array with scores
        """
        width, height = image_size
        score_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
        for patch in patch_scores:
            y_end = min(patch.y + patch.height, height)
            x_end = min(patch.x + patch.width, width)
            
            score_map[patch.y:y_end, patch.x:x_end] += patch.score
            count_map[patch.y:y_end, patch.x:x_end] += 1
        
        # Average overlapping regions
        count_map[count_map == 0] = 1
        score_map = score_map / count_map
        
        return score_map
