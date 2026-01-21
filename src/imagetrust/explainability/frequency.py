"""
Frequency domain analysis for AI detection.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


class FrequencyAnalyzer:
    """
    Frequency domain analysis for AI-generated image detection.
    
    AI-generated images often have distinctive frequency patterns
    due to upsampling artifacts and generation methods.
    
    Example:
        >>> analyzer = FrequencyAnalyzer()
        >>> features = analyzer.analyze(image)
        >>> spectrum = analyzer.get_spectrum(image)
    """

    def __init__(self) -> None:
        pass

    def analyze(
        self,
        image: Image.Image,
    ) -> Dict[str, Any]:
        """
        Analyze image in frequency domain.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Dictionary with frequency features
        """
        # Convert to grayscale numpy array
        gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
        
        # Compute 2D FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Log magnitude for visualization
        log_magnitude = np.log1p(magnitude)
        
        # Compute radial profile
        radial_profile = self._compute_radial_profile(magnitude)
        
        # Compute features
        features = {
            "mean_magnitude": float(magnitude.mean()),
            "std_magnitude": float(magnitude.std()),
            "max_magnitude": float(magnitude.max()),
            "high_freq_ratio": self._compute_high_freq_ratio(magnitude),
            "radial_profile": radial_profile.tolist(),
            "spectral_centroid": self._compute_spectral_centroid(magnitude),
            "spectral_entropy": self._compute_spectral_entropy(magnitude),
        }
        
        # Detect periodic artifacts
        artifacts = self._detect_artifacts(magnitude)
        features["artifacts"] = artifacts
        
        return features

    def get_spectrum(
        self,
        image: Image.Image,
        log_scale: bool = True,
    ) -> np.ndarray:
        """
        Get the magnitude spectrum of an image.
        
        Args:
            image: Input PIL Image
            log_scale: Use log scale for visualization
            
        Returns:
            2D numpy array of magnitude spectrum
        """
        gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
        
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        if log_scale:
            magnitude = np.log1p(magnitude)
        
        # Normalize to [0, 1]
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        return magnitude

    def get_spectrum_image(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """Get spectrum as PIL Image."""
        spectrum = self.get_spectrum(image, log_scale=True)
        spectrum_uint8 = (spectrum * 255).astype(np.uint8)
        return Image.fromarray(spectrum_uint8)

    def _compute_radial_profile(
        self,
        magnitude: np.ndarray,
    ) -> np.ndarray:
        """Compute radial average of magnitude spectrum."""
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Create distance matrix
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        distances = distances.astype(int)
        
        # Compute radial average
        max_radius = min(center_x, center_y)
        profile = np.zeros(max_radius)
        
        for r in range(max_radius):
            mask = distances == r
            if mask.any():
                profile[r] = magnitude[mask].mean()
        
        return profile

    def _compute_high_freq_ratio(
        self,
        magnitude: np.ndarray,
        threshold_ratio: float = 0.25,
    ) -> float:
        """Compute ratio of high frequency energy."""
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Create distance matrix
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        
        max_radius = min(center_x, center_y)
        threshold = max_radius * (1 - threshold_ratio)
        
        high_freq_energy = magnitude[distances > threshold].sum()
        total_energy = magnitude.sum()
        
        return float(high_freq_energy / (total_energy + 1e-8))

    def _compute_spectral_centroid(
        self,
        magnitude: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute spectral centroid."""
        h, w = magnitude.shape
        
        y_coords = np.arange(h).reshape(-1, 1)
        x_coords = np.arange(w).reshape(1, -1)
        
        total = magnitude.sum() + 1e-8
        centroid_y = (magnitude * y_coords).sum() / total
        centroid_x = (magnitude * x_coords).sum() / total
        
        return (float(centroid_x), float(centroid_y))

    def _compute_spectral_entropy(
        self,
        magnitude: np.ndarray,
    ) -> float:
        """Compute spectral entropy."""
        # Normalize to probability distribution
        prob = magnitude / (magnitude.sum() + 1e-8)
        prob = prob[prob > 0]
        
        entropy = -np.sum(prob * np.log2(prob + 1e-8))
        
        return float(entropy)

    def _detect_artifacts(
        self,
        magnitude: np.ndarray,
    ) -> Dict[str, Any]:
        """Detect periodic artifacts in spectrum."""
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Normalize magnitude
        norm_mag = magnitude / (magnitude.max() + 1e-8)
        
        # Look for peaks (potential artifacts)
        # Exclude center (DC component)
        exclude_radius = 5
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        
        mask = distances > exclude_radius
        masked_mag = norm_mag.copy()
        masked_mag[~mask] = 0
        
        # Find peaks above threshold
        threshold = norm_mag[mask].mean() + 3 * norm_mag[mask].std()
        peaks = masked_mag > threshold
        
        return {
            "num_peaks": int(peaks.sum()),
            "peak_threshold": float(threshold),
            "has_periodic_artifacts": int(peaks.sum()) > 10,
        }
