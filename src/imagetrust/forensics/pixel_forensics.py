"""
Pixel Forensics Pack.

Detectors for pixel-level forensic analysis:
- ELA (Error Level Analysis)
- Noise inconsistency
- JPEG artifacts / Double JPEG
- Resampling/resize detection
- Edge/halo artifacts
"""

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from imagetrust.forensics.base import (
    Artifact,
    Confidence,
    ForensicsPlugin,
    ForensicsResult,
    PluginCategory,
    register_plugin,
)
from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


@register_plugin
class ELADetector(ForensicsPlugin):
    """
    Error Level Analysis (ELA) detector.

    Detects local edits and recompression by comparing
    the image with a re-saved version at a specific quality.

    Strong signal for: local edits, splicing, inpainting
    Weak signal for: heavily compressed images, screenshots
    """

    plugin_id = "ela_detector"
    plugin_name = "Error Level Analysis (ELA)"
    category = PluginCategory.PIXEL
    description = "Detects local edits by analyzing JPEG recompression error patterns"
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.quality = config.get("quality", 90) if config else 90
        self.scale_factor = config.get("scale_factor", 15) if config else 15
        self.threshold = config.get("threshold", 0.15) if config else 0.15

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Perform ELA analysis."""
        start_time = time.perf_counter()

        try:
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resave at specified quality
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            resaved = Image.open(buffer)

            # Calculate difference
            original_arr = np.array(image, dtype=np.float32)
            resaved_arr = np.array(resaved, dtype=np.float32)
            diff = np.abs(original_arr - resaved_arr)

            # Create ELA heatmap (scale for visibility)
            ela_map = np.mean(diff, axis=2)  # Average across channels
            ela_scaled = np.clip(ela_map * self.scale_factor, 0, 255).astype(np.uint8)

            # Analyze for inconsistencies
            # High ELA = area was likely edited (doesn't match recompression pattern)
            mean_ela = np.mean(ela_map)
            std_ela = np.std(ela_map)
            max_ela = np.max(ela_map)

            # Detect suspicious regions (significantly different from mean)
            threshold = mean_ela + 2 * std_ela
            suspicious_mask = ela_map > threshold
            suspicious_ratio = np.mean(suspicious_mask)

            # Create colorized heatmap
            heatmap = self._create_heatmap(ela_scaled)

            # Determine detection
            # If there are localized high-ELA regions, it's suspicious
            detected = suspicious_ratio > 0.01 and suspicious_ratio < 0.5
            score = min(1.0, suspicious_ratio * 10)  # Scale up small ratios

            # Confidence based on signal strength
            if std_ela > 10 and max_ela > 50:
                confidence = Confidence.HIGH
            elif std_ela > 5 and max_ela > 30:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            # Build explanation
            if detected:
                explanation = (
                    f"ELA detected {suspicious_ratio:.1%} of pixels with unusual error levels, "
                    f"suggesting possible local edits (mean ELA={mean_ela:.1f}, max={max_ela:.1f})"
                )
            else:
                explanation = (
                    f"ELA shows uniform error distribution (mean={mean_ela:.1f}, std={std_ela:.1f}), "
                    f"consistent with unedited or uniformly processed image"
                )

            limitations = [
                "ELA is less reliable on heavily compressed images",
                "Screenshots and social media images may show false positives",
                "Multiple resaves can mask original edits",
            ]

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=score,
                confidence=confidence,
                detected=detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "mean_ela": float(mean_ela),
                    "std_ela": float(std_ela),
                    "max_ela": float(max_ela),
                    "suspicious_ratio": float(suspicious_ratio),
                    "quality_used": self.quality,
                },
                artifacts=[
                    Artifact(
                        name="ela_heatmap",
                        artifact_type="heatmap",
                        data=heatmap,
                        description="ELA heatmap - bright areas indicate potential edits",
                    ),
                ],
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"ELA analysis failed: {e}")
            return self._create_error_result(str(e))

    def _create_heatmap(self, ela_map: np.ndarray) -> Image.Image:
        """Create a colorized heatmap from ELA values."""
        # Apply colormap (blue=low, red=high)
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Normalize
        normalized = ela_map.astype(np.float32) / 255.0
        colored = cm.jet(normalized)[:, :, :3]  # Drop alpha
        colored = (colored * 255).astype(np.uint8)

        return Image.fromarray(colored)


@register_plugin
class NoiseInconsistencyDetector(ForensicsPlugin):
    """
    Noise inconsistency detector.

    Analyzes local noise patterns to detect regions with
    different noise characteristics (suggesting manipulation).

    Strong signal for: splicing, copy-paste, AI inpainting
    """

    plugin_id = "noise_inconsistency"
    plugin_name = "Noise Inconsistency Analysis"
    category = PluginCategory.PIXEL
    description = "Detects regions with inconsistent noise patterns"
    version = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.block_size = config.get("block_size", 32) if config else 32

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Analyze noise consistency across image regions."""
        start_time = time.perf_counter()

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            img_array = np.array(image, dtype=np.float32)
            h, w = img_array.shape[:2]

            # Estimate noise in blocks using Laplacian
            noise_map = np.zeros((h // self.block_size, w // self.block_size))

            for i in range(noise_map.shape[0]):
                for j in range(noise_map.shape[1]):
                    y1, y2 = i * self.block_size, (i + 1) * self.block_size
                    x1, x2 = j * self.block_size, (j + 1) * self.block_size

                    block = img_array[y1:y2, x1:x2]

                    # Estimate noise per channel using Laplacian
                    noise_levels = []
                    for c in range(3):
                        laplacian = ndimage.laplace(block[:, :, c])
                        sigma = np.median(np.abs(laplacian)) / 0.6745
                        noise_levels.append(sigma)

                    noise_map[i, j] = np.mean(noise_levels)

            # Analyze noise distribution
            mean_noise = np.mean(noise_map)
            std_noise = np.std(noise_map)
            cv_noise = std_noise / (mean_noise + 1e-10)  # Coefficient of variation

            # Find outlier regions
            threshold_high = mean_noise + 2 * std_noise
            threshold_low = mean_noise - 2 * std_noise
            outliers_high = noise_map > threshold_high
            outliers_low = noise_map < threshold_low
            outlier_ratio = (np.mean(outliers_high) + np.mean(outliers_low))

            # Detection logic
            # High CV suggests inconsistent noise = possible manipulation
            detected = cv_noise > 0.3 and outlier_ratio > 0.05
            score = min(1.0, cv_noise) * min(1.0, outlier_ratio * 10)

            if cv_noise > 0.5:
                confidence = Confidence.HIGH
            elif cv_noise > 0.3:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            # Create visualization
            noise_vis = self._create_noise_map_visual(noise_map)

            if detected:
                explanation = (
                    f"Noise analysis found {outlier_ratio:.1%} of regions with unusual noise levels "
                    f"(CV={cv_noise:.2f}), suggesting possible manipulation or splicing"
                )
            else:
                explanation = (
                    f"Noise patterns are consistent across the image (CV={cv_noise:.2f}), "
                    f"suggesting a single source/capture"
                )

            limitations = [
                "Natural images can have varying noise in different regions",
                "Heavy processing/filtering can unify noise artificially",
                "Very smooth or synthetic images may show false positives",
            ]

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=score,
                confidence=confidence,
                detected=detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "mean_noise": float(mean_noise),
                    "std_noise": float(std_noise),
                    "cv_noise": float(cv_noise),
                    "outlier_ratio": float(outlier_ratio),
                    "block_size": self.block_size,
                },
                artifacts=[
                    Artifact(
                        name="noise_map",
                        artifact_type="heatmap",
                        data=noise_vis,
                        description="Noise level map - inconsistent regions may indicate manipulation",
                    ),
                ],
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Noise analysis failed: {e}")
            return self._create_error_result(str(e))

    def _create_noise_map_visual(self, noise_map: np.ndarray) -> Image.Image:
        """Create visualization of noise map."""
        import matplotlib.cm as cm

        # Normalize for visualization
        min_val, max_val = noise_map.min(), noise_map.max()
        if max_val > min_val:
            normalized = (noise_map - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(noise_map)

        # Apply colormap
        colored = cm.viridis(normalized)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)

        # Upscale to reasonable size
        h, w = colored.shape[:2]
        img = Image.fromarray(colored)
        img = img.resize((w * self.block_size, h * self.block_size), Image.NEAREST)

        return img


@register_plugin
class JPEGArtifactsDetector(ForensicsPlugin):
    """
    JPEG compression artifacts detector.

    Analyzes:
    - Block boundary discontinuities
    - Quantization patterns
    - Double JPEG compression signatures

    Strong signal for: recompression, social media processing
    """

    plugin_id = "jpeg_artifacts"
    plugin_name = "JPEG Artifacts Analysis"
    category = PluginCategory.PIXEL
    description = "Analyzes JPEG compression artifacts and detects double compression"
    version = "1.0.0"
    requires_jpeg = False  # Can work on any format (checks if was JPEG)

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Analyze JPEG artifacts."""
        start_time = time.perf_counter()

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            img_array = np.array(image, dtype=np.float32)
            gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

            h, w = gray.shape
            block_size = 8  # JPEG uses 8x8 blocks

            # 1. Analyze block boundary discontinuities
            h_boundaries = []
            v_boundaries = []

            # Horizontal boundaries (vertical lines at x=8n)
            for x in range(block_size, w - block_size, block_size):
                diff = np.mean(np.abs(gray[:, x] - gray[:, x - 1]))
                h_boundaries.append(diff)

            # Vertical boundaries (horizontal lines at y=8n)
            for y in range(block_size, h - block_size, block_size):
                diff = np.mean(np.abs(gray[y, :] - gray[y - 1, :]))
                v_boundaries.append(diff)

            h_boundaries = np.array(h_boundaries) if h_boundaries else np.array([0])
            v_boundaries = np.array(v_boundaries) if v_boundaries else np.array([0])

            # Calculate blocking artifact strength
            block_artifact_h = np.mean(h_boundaries)
            block_artifact_v = np.mean(v_boundaries)
            blocking_strength = (block_artifact_h + block_artifact_v) / 2

            # 2. Check for double JPEG (DCT histogram analysis)
            # In double-compressed images, DCT coefficients show periodic patterns
            double_jpeg_score = self._check_double_jpeg(gray)

            # 3. Analyze block variance distribution
            block_variances = []
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y + block_size, x:x + block_size]
                    block_variances.append(np.var(block))

            block_variances = np.array(block_variances)
            var_mean = np.mean(block_variances)
            var_std = np.std(block_variances)

            # Detection: significant blocking artifacts indicate JPEG processing
            jpeg_detected = blocking_strength > 2.0
            double_detected = double_jpeg_score > 0.5

            if double_detected:
                detected = True
                score = double_jpeg_score
                confidence = Confidence.HIGH if double_jpeg_score > 0.7 else Confidence.MEDIUM
                explanation = (
                    f"Double JPEG compression detected (score={double_jpeg_score:.2f}), "
                    f"suggesting the image was saved as JPEG multiple times"
                )
            elif jpeg_detected:
                detected = True
                score = min(1.0, blocking_strength / 5.0)
                confidence = Confidence.MEDIUM
                explanation = (
                    f"JPEG compression artifacts detected (blocking={blocking_strength:.2f}), "
                    f"image has been JPEG compressed"
                )
            else:
                detected = False
                score = 0.0
                confidence = Confidence.HIGH
                explanation = (
                    f"No significant JPEG artifacts detected (blocking={blocking_strength:.2f}), "
                    f"image may be original camera output, PNG, or losslessly processed"
                )

            limitations = [
                "High-quality JPEG (Q>95) may not show detectable artifacts",
                "Some cameras apply in-camera JPEG compression",
                "Double JPEG detection requires sufficient compression",
            ]

            # Create blocking artifacts map
            artifacts_map = self._create_blocking_map(gray, block_size)

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=score,
                confidence=confidence,
                detected=detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "blocking_strength": float(blocking_strength),
                    "block_artifact_h": float(block_artifact_h),
                    "block_artifact_v": float(block_artifact_v),
                    "double_jpeg_score": float(double_jpeg_score),
                    "block_variance_mean": float(var_mean),
                    "block_variance_std": float(var_std),
                },
                artifacts=[
                    Artifact(
                        name="jpeg_blocking_map",
                        artifact_type="heatmap",
                        data=artifacts_map,
                        description="JPEG blocking artifact map",
                    ),
                ],
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"JPEG artifacts analysis failed: {e}")
            return self._create_error_result(str(e))

    def _check_double_jpeg(self, gray: np.ndarray) -> float:
        """
        Check for double JPEG compression.

        Uses DCT coefficient histogram analysis - double compressed
        images show periodic patterns in DCT coefficients.
        """
        try:
            from scipy.fftpack import dct

            h, w = gray.shape
            block_size = 8

            # Collect DCT coefficients from blocks
            dct_coeffs = []

            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y + block_size, x:x + block_size]
                    # 2D DCT
                    block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                    # Take AC coefficients (skip DC at [0,0])
                    dct_coeffs.extend(block_dct.flatten()[1:])

            dct_coeffs = np.array(dct_coeffs)

            # Analyze histogram for periodicity
            hist, bin_edges = np.histogram(dct_coeffs, bins=256, range=(-128, 128))

            # Double compression creates periodic dips in histogram
            # Check for periodic patterns using autocorrelation
            hist_centered = hist - np.mean(hist)
            autocorr = np.correlate(hist_centered, hist_centered, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)

            # Look for peaks at regular intervals (suggesting quantization grid)
            peaks = []
            for i in range(5, min(50, len(autocorr))):
                if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                    if autocorr[i] > 0.1:  # Significant peak
                        peaks.append((i, autocorr[i]))

            # Score based on peak strength
            if peaks:
                max_peak = max(p[1] for p in peaks)
                return min(1.0, max_peak * 2)

            return 0.0

        except Exception:
            return 0.0

    def _create_blocking_map(self, gray: np.ndarray, block_size: int) -> Image.Image:
        """Create visualization of blocking artifacts."""
        h, w = gray.shape
        blocking_map = np.zeros_like(gray)

        # Compute local blocking at each boundary
        for x in range(block_size, w - 1, block_size):
            diff = np.abs(gray[:, x] - gray[:, x - 1])
            blocking_map[:, x - 1:x + 1] = np.maximum(
                blocking_map[:, x - 1:x + 1],
                diff[:, np.newaxis]
            )

        for y in range(block_size, h - 1, block_size):
            diff = np.abs(gray[y, :] - gray[y - 1, :])
            blocking_map[y - 1:y + 1, :] = np.maximum(
                blocking_map[y - 1:y + 1, :],
                diff[np.newaxis, :]
            )

        # Normalize and colorize
        blocking_map = np.clip(blocking_map * 5, 0, 255).astype(np.uint8)

        import matplotlib.cm as cm
        colored = cm.hot(blocking_map / 255.0)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)

        return Image.fromarray(colored)


@register_plugin
class ResamplingDetector(ForensicsPlugin):
    """
    Resampling/resize detection.

    Detects traces of image resizing/interpolation through
    frequency analysis of interpolation artifacts.

    Strong signal for: resized images, upscaled images
    """

    plugin_id = "resampling_detector"
    plugin_name = "Resampling/Resize Detection"
    category = PluginCategory.PIXEL
    description = "Detects traces of image resizing through interpolation artifact analysis"
    version = "1.0.0"

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Detect resampling artifacts."""
        start_time = time.perf_counter()

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            img_array = np.array(image, dtype=np.float32)
            gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

            # 1. Radon transform-based detection (simplified)
            # Resampling creates periodic patterns in second derivative
            laplacian = ndimage.laplace(gray)

            # 2D FFT of Laplacian
            fft = np.fft.fft2(laplacian)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            log_magnitude = np.log1p(magnitude)

            h, w = gray.shape
            cy, cx = h // 2, w // 2

            # 2. Look for periodic peaks in spectrum (resampling creates these)
            # Examine horizontal and vertical axes
            horizontal_slice = log_magnitude[cy, :]
            vertical_slice = log_magnitude[:, cx]

            # Autocorrelation to find periodicity
            h_autocorr = np.correlate(horizontal_slice - np.mean(horizontal_slice),
                                       horizontal_slice - np.mean(horizontal_slice), mode='same')
            v_autocorr = np.correlate(vertical_slice - np.mean(vertical_slice),
                                       vertical_slice - np.mean(vertical_slice), mode='same')

            # Normalize
            h_autocorr = h_autocorr / (h_autocorr[len(h_autocorr) // 2] + 1e-10)
            v_autocorr = v_autocorr / (v_autocorr[len(v_autocorr) // 2] + 1e-10)

            # Find secondary peaks (indicating periodicity)
            h_peaks = self._find_secondary_peaks(h_autocorr)
            v_peaks = self._find_secondary_peaks(v_autocorr)

            # 3. Check for interpolation signature in pixel differences
            # Resized images have smoother transitions
            dx = np.diff(gray, axis=1)
            dy = np.diff(gray, axis=0)

            # Variance of differences (resampled images tend to have more uniform gradients)
            dx_var = np.var(dx)
            dy_var = np.var(dy)
            gradient_uniformity = 1.0 / (1.0 + (dx_var + dy_var) / 1000)

            # Combine scores
            periodicity_score = (len(h_peaks) + len(v_peaks)) / 10.0
            periodicity_score = min(1.0, periodicity_score)

            # Detection
            detected = periodicity_score > 0.3 or gradient_uniformity > 0.7
            score = max(periodicity_score, gradient_uniformity * 0.5)

            if periodicity_score > 0.5:
                confidence = Confidence.HIGH
            elif periodicity_score > 0.3 or gradient_uniformity > 0.6:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            if detected:
                explanation = (
                    f"Resampling artifacts detected: periodicity score={periodicity_score:.2f}, "
                    f"gradient uniformity={gradient_uniformity:.2f}. Image appears to have been resized."
                )
            else:
                explanation = (
                    f"No significant resampling artifacts detected "
                    f"(periodicity={periodicity_score:.2f}). Image appears to be at native resolution."
                )

            limitations = [
                "Some cameras internally resize/process images",
                "Very small resize factors may not be detectable",
                "Anti-aliasing can mask resampling artifacts",
            ]

            # Create frequency domain visualization
            freq_vis = self._create_frequency_visual(log_magnitude)

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=score,
                confidence=confidence,
                detected=detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "periodicity_score": float(periodicity_score),
                    "gradient_uniformity": float(gradient_uniformity),
                    "h_peaks": len(h_peaks),
                    "v_peaks": len(v_peaks),
                    "dx_variance": float(dx_var),
                    "dy_variance": float(dy_var),
                },
                artifacts=[
                    Artifact(
                        name="frequency_spectrum",
                        artifact_type="heatmap",
                        data=freq_vis,
                        description="Frequency spectrum - peaks may indicate resampling",
                    ),
                ],
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Resampling detection failed: {e}")
            return self._create_error_result(str(e))

    def _find_secondary_peaks(self, autocorr: np.ndarray) -> List[int]:
        """Find secondary peaks in autocorrelation."""
        peaks = []
        center = len(autocorr) // 2

        for i in range(center + 5, min(center + 100, len(autocorr) - 1)):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                if autocorr[i] > 0.2:  # Significant peak
                    peaks.append(i - center)

        return peaks

    def _create_frequency_visual(self, log_magnitude: np.ndarray) -> Image.Image:
        """Create frequency domain visualization."""
        # Normalize
        min_val, max_val = log_magnitude.min(), log_magnitude.max()
        normalized = (log_magnitude - min_val) / (max_val - min_val + 1e-10)

        # Convert to uint8
        img_array = (normalized * 255).astype(np.uint8)

        return Image.fromarray(img_array)


@register_plugin
class EdgeHaloDetector(ForensicsPlugin):
    """
    Edge/halo artifacts detector.

    Detects unnatural edges and halos that may indicate:
    - Over-sharpening
    - AI upscaling
    - Cloning/splicing boundaries
    """

    plugin_id = "edge_halo_detector"
    plugin_name = "Edge/Halo Artifacts Detection"
    category = PluginCategory.PIXEL
    description = "Detects unnatural edges, halos, and sharpening artifacts"
    version = "1.0.0"

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Detect edge and halo artifacts."""
        start_time = time.perf_counter()

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            img_array = np.array(image, dtype=np.float32)
            gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

            # 1. Detect edges using Sobel
            sobel_x = ndimage.sobel(gray, axis=1)
            sobel_y = ndimage.sobel(gray, axis=0)
            edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            # 2. Detect halos using Laplacian (over-sharpening creates ringing)
            laplacian = ndimage.laplace(gray)

            # 3. Look for unnatural edge patterns
            # Strong edges should have gradual falloff, not abrupt
            edge_threshold = np.percentile(edge_magnitude, 95)
            strong_edges = edge_magnitude > edge_threshold

            # Check for ringing/halos near strong edges
            dilated_edges = ndimage.binary_dilation(strong_edges, iterations=3)
            halo_region = dilated_edges & ~strong_edges

            if np.sum(halo_region) > 0:
                halo_strength = np.mean(np.abs(laplacian[halo_region]))
            else:
                halo_strength = 0

            # 4. Edge sharpness analysis
            # Natural images have varied edge sharpness
            edge_sharpness = np.std(edge_magnitude[strong_edges]) if np.sum(strong_edges) > 0 else 0

            # 5. Calculate uniformity of strong edges (AI/processed tend to be more uniform)
            edge_cv = edge_sharpness / (np.mean(edge_magnitude[strong_edges]) + 1e-10) if np.sum(strong_edges) > 0 else 1.0

            # Detection logic
            halo_detected = halo_strength > 5.0
            oversharp_detected = edge_cv < 0.3 and np.mean(edge_magnitude) > 20

            detected = halo_detected or oversharp_detected
            score = min(1.0, halo_strength / 10.0 + (1 - edge_cv) * 0.5)

            if halo_strength > 10 or edge_cv < 0.2:
                confidence = Confidence.HIGH
            elif halo_strength > 5 or edge_cv < 0.4:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            if detected:
                reasons = []
                if halo_detected:
                    reasons.append(f"halo artifacts (strength={halo_strength:.2f})")
                if oversharp_detected:
                    reasons.append(f"over-sharpening (edge CV={edge_cv:.2f})")
                explanation = f"Detected: {', '.join(reasons)}. May indicate AI processing or heavy editing."
            else:
                explanation = f"Edge patterns appear natural (halo={halo_strength:.2f}, edge CV={edge_cv:.2f})."

            limitations = [
                "Some cameras apply in-camera sharpening",
                "HDR processing can create similar artifacts",
                "Compressed images may lose subtle halo information",
            ]

            # Create edge visualization
            edge_vis = self._create_edge_visual(edge_magnitude, laplacian)

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=score,
                confidence=confidence,
                detected=detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "halo_strength": float(halo_strength),
                    "edge_sharpness": float(edge_sharpness),
                    "edge_cv": float(edge_cv),
                    "mean_edge_magnitude": float(np.mean(edge_magnitude)),
                    "strong_edge_ratio": float(np.mean(strong_edges)),
                },
                artifacts=[
                    Artifact(
                        name="edge_analysis",
                        artifact_type="heatmap",
                        data=edge_vis,
                        description="Edge magnitude map with halo indicators",
                    ),
                ],
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Edge/halo detection failed: {e}")
            return self._create_error_result(str(e))

    def _create_edge_visual(self, edge_magnitude: np.ndarray, laplacian: np.ndarray) -> Image.Image:
        """Create edge visualization."""
        # Normalize edge magnitude (avoid division by zero for uniform images)
        percentile_99 = np.percentile(edge_magnitude, 99)
        if percentile_99 > 0:
            edge_norm = np.clip(edge_magnitude / percentile_99 * 255, 0, 255).astype(np.uint8)
        else:
            edge_norm = np.zeros_like(edge_magnitude, dtype=np.uint8)

        # Create RGB: edges in green, halos (laplacian) in red
        h, w = edge_magnitude.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[:, :, 1] = edge_norm  # Green = edges
        vis[:, :, 0] = np.clip(np.abs(laplacian) * 2, 0, 255).astype(np.uint8)  # Red = halos

        return Image.fromarray(vis)
