"""
Degradation robustness evaluation.

Tests model performance under various image degradations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from imagetrust.evaluation.benchmark import Benchmark
from imagetrust.evaluation.metrics import compute_metrics
from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import ensure_dir, load_image, save_image

logger = get_logger(__name__)


class DegradationEvaluator:
    """
    Evaluates model robustness against image degradations.
    
    Tests performance under:
    - JPEG compression
    - Gaussian blur
    - Resize operations
    - Gaussian noise
    
    Example:
        >>> evaluator = DegradationEvaluator(output_dir="results")
        >>> evaluator.add_images(image_paths, labels)
        >>> results = evaluator.evaluate(detector)
    """

    def __init__(
        self,
        output_dir: Optional[Union[Path, str]] = None,
        verbose: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("results/degradation")
        self.verbose = verbose
        
        self.images: List[Path] = []
        self.labels: List[int] = []
        self.results: Optional[Dict[str, Any]] = None
        
        # Cache for degraded images
        self.cache_dir = self.output_dir / "cache"

    def add_images(
        self,
        images: List[Union[Path, str]],
        labels: Union[int, List[int]],
    ) -> None:
        """
        Add images for evaluation.
        
        Args:
            images: List of image paths
            labels: Single label or list of labels
        """
        if isinstance(labels, int):
            labels = [labels] * len(images)
        
        self.images.extend([Path(p) for p in images])
        self.labels.extend(labels)
        
        logger.info(f"Added {len(images)} images for degradation testing")

    def _apply_jpeg_compression(
        self,
        image: Image.Image,
        quality: int,
    ) -> Image.Image:
        """Apply JPEG compression."""
        from io import BytesIO
        
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _apply_blur(
        self,
        image: Image.Image,
        radius: float,
    ) -> Image.Image:
        """Apply Gaussian blur."""
        if radius <= 0:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius))

    def _apply_resize(
        self,
        image: Image.Image,
        factor: float,
    ) -> Image.Image:
        """Apply resize and restore."""
        if factor >= 1.0:
            return image
        
        original_size = image.size
        new_size = (int(original_size[0] * factor), int(original_size[1] * factor))
        
        # Downscale then upscale
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        restored = resized.resize(original_size, Image.Resampling.LANCZOS)
        
        return restored

    def _apply_noise(
        self,
        image: Image.Image,
        sigma: float,
    ) -> Image.Image:
        """Apply Gaussian noise."""
        if sigma <= 0:
            return image
        
        arr = np.array(image, dtype=np.float32) / 255.0
        noise = np.random.normal(0, sigma, arr.shape)
        noisy = np.clip(arr + noise, 0, 1) * 255
        
        return Image.fromarray(noisy.astype(np.uint8))

    def evaluate(
        self,
        detector,
        jpeg_qualities: List[int] = [95, 85, 70, 50],
        blur_radii: List[float] = [0, 0.5, 1.0, 2.0],
        resize_factors: List[float] = [1.0, 0.75, 0.5],
        noise_levels: List[float] = [0, 0.01, 0.03],
    ) -> Dict[str, Any]:
        """
        Run degradation evaluation.
        
        Args:
            detector: AIDetector instance
            jpeg_qualities: JPEG quality levels (1-100)
            blur_radii: Gaussian blur radii
            resize_factors: Resize factors (1.0 = original)
            noise_levels: Gaussian noise sigma values
            
        Returns:
            Evaluation results
        """
        logger.info("Running degradation robustness evaluation")
        
        results = {
            "original": {},
            "jpeg_compression": {},
            "blur": {},
            "resize": {},
            "noise": {},
        }
        
        # Evaluate original
        logger.info("Evaluating original images")
        results["original"] = self._evaluate_set(detector, self.images, self.labels)
        
        # JPEG compression
        logger.info("Evaluating JPEG compression robustness")
        for quality in jpeg_qualities:
            degraded = []
            for img_path in tqdm(self.images, desc=f"JPEG Q{quality}", disable=not self.verbose):
                img = load_image(img_path)
                degraded.append(self._apply_jpeg_compression(img, quality))
            
            results["jpeg_compression"][quality] = self._evaluate_pil_images(
                detector, degraded, self.labels
            )
        
        # Blur
        logger.info("Evaluating blur robustness")
        for radius in blur_radii:
            degraded = []
            for img_path in tqdm(self.images, desc=f"Blur r={radius}", disable=not self.verbose):
                img = load_image(img_path)
                degraded.append(self._apply_blur(img, radius))
            
            results["blur"][radius] = self._evaluate_pil_images(
                detector, degraded, self.labels
            )
        
        # Resize
        logger.info("Evaluating resize robustness")
        for factor in resize_factors:
            degraded = []
            for img_path in tqdm(self.images, desc=f"Resize {factor}", disable=not self.verbose):
                img = load_image(img_path)
                degraded.append(self._apply_resize(img, factor))
            
            results["resize"][factor] = self._evaluate_pil_images(
                detector, degraded, self.labels
            )
        
        # Noise
        logger.info("Evaluating noise robustness")
        for sigma in noise_levels:
            degraded = []
            for img_path in tqdm(self.images, desc=f"Noise σ={sigma}", disable=not self.verbose):
                img = load_image(img_path)
                degraded.append(self._apply_noise(img, sigma))
            
            results["noise"][sigma] = self._evaluate_pil_images(
                detector, degraded, self.labels
            )
        
        self.results = results
        return results

    def _evaluate_set(
        self,
        detector,
        images: List[Path],
        labels: List[int],
    ) -> Dict[str, Any]:
        """Evaluate a set of images."""
        preds = []
        probs = []
        
        for img_path in tqdm(images, desc="Evaluating", disable=not self.verbose):
            try:
                result = detector.detect(img_path)
                probs.append(result["ai_probability"])
                preds.append(1 if result["ai_probability"] > 0.5 else 0)
            except Exception as e:
                logger.warning(f"Failed: {e}")
                probs.append(0.5)
                preds.append(0)
        
        return compute_metrics(np.array(labels), np.array(preds), np.array(probs))

    def _evaluate_pil_images(
        self,
        detector,
        images: List[Image.Image],
        labels: List[int],
    ) -> Dict[str, Any]:
        """Evaluate PIL images directly."""
        preds = []
        probs = []
        
        for img in images:
            try:
                result = detector.detect(img)
                probs.append(result["ai_probability"])
                preds.append(1 if result["ai_probability"] > 0.5 else 0)
            except Exception as e:
                logger.warning(f"Failed: {e}")
                probs.append(0.5)
                preds.append(0)
        
        return compute_metrics(np.array(labels), np.array(preds), np.array(probs))

    def save_results(self, filename: str = "degradation_results.json") -> Path:
        """Save results to file."""
        import json
        
        ensure_dir(self.output_dir)
        output_path = self.output_dir / filename
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return output_path

    def print_summary(self) -> None:
        """Print evaluation summary."""
        if self.results is None:
            print("No results available.")
            return
        
        print("\n" + "=" * 60)
        print("DEGRADATION ROBUSTNESS SUMMARY")
        print("=" * 60)
        
        original = self.results.get("original", {})
        print(f"\nOriginal: Acc={original.get('accuracy', 0):.2%}, "
              f"F1={original.get('f1_score', 0):.2%}")
        
        for deg_type in ["jpeg_compression", "blur", "resize", "noise"]:
            print(f"\n{deg_type.replace('_', ' ').title()}")
            print("-" * 40)
            for param, metrics in self.results.get(deg_type, {}).items():
                print(f"  {param}: Acc={metrics.get('accuracy', 0):.2%}, "
                      f"F1={metrics.get('f1_score', 0):.2%}")
        
        print("=" * 60 + "\n")
