"""
Test-Time Adaptive Image Restoration for Robust AI Detection.

Instead of augmenting training data (which can destroy forensic artifacts),
this module restores degraded images at inference time BEFORE classification.

Strategy:
1. Detect degradation type and severity (JPEG artifacts, blur, noise)
2. Apply targeted restoration (denoise, sharpen, dejpeg)
3. Feed the restored image to the classifier

This avoids the professor's concern: naive augmentation destroys the very
artifacts the model needs to detect. By restoring at test-time, we
"normalize" inputs so the classifier sees cleaner signal.
"""

import io
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


class DegradationType(Enum):
    """Types of image degradation detected."""
    NONE = "none"
    JPEG_ARTIFACTS = "jpeg_artifacts"
    BLUR = "blur"
    NOISE = "noise"
    LOW_RESOLUTION = "low_resolution"
    MULTIPLE = "multiple"


@dataclass
class DegradationProfile:
    """Detected degradation profile of an image."""
    primary_type: DegradationType
    severity: float  # 0-1 (0=no degradation, 1=severe)
    jpeg_quality_est: Optional[int]  # Estimated JPEG quality (if applicable)
    blur_level: float  # Estimated blur sigma
    noise_level: float  # Estimated noise std
    is_low_res: bool  # True if image is very small
    details: Dict[str, Any]


@dataclass
class RestorationResult:
    """Result of adaptive restoration."""
    restored_image: Image.Image
    original_image: Image.Image
    degradation: DegradationProfile
    was_restored: bool  # False if image was clean enough
    restoration_applied: List[str]  # List of restoration steps


def detect_degradation(image: Image.Image) -> DegradationProfile:
    """
    Detect the type and severity of image degradation.

    Analyzes JPEG blocking artifacts, blur level, and noise level
    to determine what restoration is needed.

    Args:
        image: PIL Image to analyze.

    Returns:
        DegradationProfile describing the detected degradation.
    """
    img_array = np.array(image.convert("RGB"), dtype=np.float64)
    h, w = img_array.shape[:2]

    details: Dict[str, Any] = {}

    # 1. Estimate JPEG quality via blocking artifacts
    jpeg_quality, jpeg_detail = _estimate_jpeg_quality(img_array)
    details["jpeg"] = jpeg_detail

    # 2. Estimate blur level
    blur_level, blur_detail = _estimate_blur(img_array)
    details["blur"] = blur_detail

    # 3. Estimate noise level
    noise_level, noise_detail = _estimate_noise(img_array)
    details["noise"] = noise_detail

    # 4. Check resolution
    is_low_res = min(h, w) < 256
    details["resolution"] = {"width": w, "height": h, "is_low_res": is_low_res}

    # Determine primary degradation and severity
    severities = {
        DegradationType.JPEG_ARTIFACTS: 0.0,
        DegradationType.BLUR: 0.0,
        DegradationType.NOISE: 0.0,
        DegradationType.LOW_RESOLUTION: 0.0,
    }

    # JPEG severity (low estimated quality = high severity)
    if jpeg_quality is not None and jpeg_quality < 70:
        severities[DegradationType.JPEG_ARTIFACTS] = max(
            0, (70 - jpeg_quality) / 45
        )

    # Blur severity
    if blur_level > 1.5:
        severities[DegradationType.BLUR] = min(1.0, (blur_level - 1.5) / 3.0)

    # Noise severity
    if noise_level > 8.0:
        severities[DegradationType.NOISE] = min(1.0, (noise_level - 8.0) / 20.0)

    # Low resolution
    if is_low_res:
        severities[DegradationType.LOW_RESOLUTION] = 0.5

    # Find primary degradation
    max_severity = max(severities.values())
    if max_severity < 0.1:
        primary = DegradationType.NONE
        severity = 0.0
    elif sum(1 for s in severities.values() if s > 0.2) > 1:
        primary = DegradationType.MULTIPLE
        severity = max_severity
    else:
        primary = max(severities, key=severities.get)
        severity = max_severity

    return DegradationProfile(
        primary_type=primary,
        severity=float(np.clip(severity, 0, 1)),
        jpeg_quality_est=jpeg_quality,
        blur_level=float(blur_level),
        noise_level=float(noise_level),
        is_low_res=is_low_res,
        details=details,
    )


def adaptive_restore(
    image: Image.Image,
    degradation: Optional[DegradationProfile] = None,
    force: bool = False,
) -> RestorationResult:
    """
    Apply adaptive restoration based on detected degradation.

    Only restores if degradation is significant. Light restoration
    preserves forensic artifacts while removing degradation noise.

    Args:
        image: PIL Image to restore.
        degradation: Pre-computed degradation profile (computed if None).
        force: Force restoration even if degradation is minimal.

    Returns:
        RestorationResult with restored image and metadata.
    """
    if degradation is None:
        degradation = detect_degradation(image)

    restored = image.copy()
    steps: List[str] = []

    # Skip restoration if image is clean
    if not force and degradation.severity < 0.15:
        return RestorationResult(
            restored_image=restored,
            original_image=image,
            degradation=degradation,
            was_restored=False,
            restoration_applied=[],
        )

    # 1. Denoise (if noisy)
    if degradation.noise_level > 10.0:
        strength = min(2.0, degradation.noise_level / 15.0)
        restored = _denoise(restored, strength=strength)
        steps.append(f"denoise(strength={strength:.1f})")

    # 2. De-JPEG (if heavy JPEG compression — conservative threshold)
    if (degradation.jpeg_quality_est is not None
            and degradation.jpeg_quality_est < 50):
        restored = _dejpeg(restored, quality=degradation.jpeg_quality_est)
        steps.append(f"dejpeg(q_est={degradation.jpeg_quality_est})")

    # 3. Deblur / sharpen (if clearly blurry — conservative threshold)
    if degradation.blur_level > 3.5:
        strength = min(1.5, (degradation.blur_level - 3.5) / 2.0)
        restored = _sharpen(restored, strength=strength)
        steps.append(f"sharpen(strength={strength:.1f})")

    was_restored = len(steps) > 0

    return RestorationResult(
        restored_image=restored,
        original_image=image,
        degradation=degradation,
        was_restored=was_restored,
        restoration_applied=steps,
    )


def _estimate_jpeg_quality(img: np.ndarray) -> Tuple[Optional[int], Dict]:
    """
    Estimate JPEG compression quality from blocking artifacts.

    Uses a periodicity-based approach: computes the average gradient
    at every row/column offset within the 8-pixel JPEG block period.
    JPEG compression creates a periodic spike at block boundaries
    (offset 0 mod 8). The strength of this spike relative to the
    average gradient at other offsets gives a content-adaptive
    blockiness measure.
    """
    gray = np.mean(img, axis=2)
    h, w = gray.shape

    if h < 24 or w < 24:
        return None, {"method": "too_small"}

    # Compute absolute gradients
    row_grad = np.abs(np.diff(gray, axis=0))  # (h-1, w)
    col_grad = np.abs(np.diff(gray, axis=1))  # (h, w-1)

    # For each offset 0..7 within the 8-pixel period, compute average gradient
    row_period = np.zeros(8)
    row_counts = np.zeros(8)
    for r in range(row_grad.shape[0]):
        offset = r % 8
        row_period[offset] += np.mean(row_grad[r, :])
        row_counts[offset] += 1

    col_period = np.zeros(8)
    col_counts = np.zeros(8)
    for c in range(col_grad.shape[1]):
        offset = c % 8
        col_period[offset] += np.mean(col_grad[:, c])
        col_counts[offset] += 1

    row_period /= np.maximum(row_counts, 1)
    col_period /= np.maximum(col_counts, 1)

    # JPEG block boundaries are at offsets 7 (row 7->8, 15->16, etc.)
    # Blockiness = gradient at boundary offset vs average of non-boundary
    boundary_offset = 7

    row_non_boundary = np.mean(
        [row_period[i] for i in range(8) if i != boundary_offset]
    )
    col_non_boundary = np.mean(
        [col_period[i] for i in range(8) if i != boundary_offset]
    )

    row_blockiness = row_period[boundary_offset] / (row_non_boundary + 1e-10)
    col_blockiness = col_period[boundary_offset] / (col_non_boundary + 1e-10)
    blockiness = (row_blockiness + col_blockiness) / 2

    # Map blockiness to estimated quality.
    # Conservative thresholds — only flag clearly degraded images.
    # Smooth images inflate blockiness; textured images mask it.
    # We prefer false negatives (missing mild JPEG) over false
    # positives (incorrectly restoring clean images).
    if blockiness < 1.05:
        quality_est = 95
    elif blockiness < 1.25:
        quality_est = 80
    elif blockiness < 1.60:
        quality_est = 60
    elif blockiness < 2.20:
        quality_est = 40
    else:
        quality_est = 25

    return quality_est, {
        "row_blockiness": float(row_blockiness),
        "col_blockiness": float(col_blockiness),
        "blockiness": float(blockiness),
        "quality_est": quality_est,
    }


def _estimate_blur(img: np.ndarray) -> Tuple[float, Dict]:
    """
    Estimate blur level using Laplacian variance method.

    Low variance of Laplacian = blurry image.
    """
    gray = np.mean(img, axis=2)

    # Laplacian kernel
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="reflect")
    lap = np.zeros_like(gray)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            lap += padded[di:di + h, dj:dj + w] * kernel[di, dj]

    lap_var = np.var(lap)
    lap_mean = np.mean(np.abs(lap))

    # Map variance to blur sigma estimate
    # High variance → sharp, low variance → blurry
    # Note: smooth images can have low variance without being blurry,
    # so we use conservative thresholds to avoid false positives.
    if lap_var > 80:
        blur_sigma = 0.5  # Sharp enough
    elif lap_var > 30:
        blur_sigma = 1.5  # Slightly soft
    elif lap_var > 10:
        blur_sigma = 3.0  # Moderately blurry
    elif lap_var > 3:
        blur_sigma = 4.5  # Blurry
    else:
        blur_sigma = 6.0  # Very blurry

    return blur_sigma, {
        "laplacian_variance": float(lap_var),
        "laplacian_mean": float(lap_mean),
        "estimated_sigma": float(blur_sigma),
    }


def _estimate_noise(img: np.ndarray) -> Tuple[float, Dict]:
    """
    Estimate noise level using median absolute deviation.

    Uses the high-frequency component of the image as noise estimate.
    """
    gray = np.mean(img, axis=2)

    # High-pass filter (difference of adjacent pixels)
    diff_h = gray[:, 1:] - gray[:, :-1]
    diff_v = gray[1:, :] - gray[:-1, :]

    # Robust noise estimation via MAD
    mad_h = np.median(np.abs(diff_h - np.median(diff_h)))
    mad_v = np.median(np.abs(diff_v - np.median(diff_v)))

    # Convert MAD to sigma (assuming Gaussian)
    sigma_h = mad_h * 1.4826
    sigma_v = mad_v * 1.4826
    noise_sigma = (sigma_h + sigma_v) / 2

    return float(noise_sigma), {
        "sigma_h": float(sigma_h),
        "sigma_v": float(sigma_v),
        "noise_sigma": float(noise_sigma),
    }


def _denoise(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Light denoising using bilateral-like filtering.

    Preserves edges while smoothing noise.
    Uses PIL's built-in smoothing with controlled strength.
    """
    img_array = np.array(image, dtype=np.float32)

    # Simple bilateral approximation: smooth, then blend
    smoothed = image.filter(ImageFilter.GaussianBlur(radius=strength))
    smooth_array = np.array(smoothed, dtype=np.float32)

    # Edge-aware blending: where gradients are small, use smoothed version
    diff = np.abs(img_array - smooth_array)
    edge_mask = np.clip(diff / 30.0, 0, 1)  # Edges have high diff

    # Keep edges from original, smooth flat areas
    alpha = 1.0 - edge_mask * 0.8  # 80% preservation at edges
    result = img_array * (1 - alpha) + smooth_array * alpha
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


def _dejpeg(image: Image.Image, quality: int = 50) -> Image.Image:
    """
    Reduce JPEG blocking artifacts.

    Applies targeted smoothing at 8x8 block boundaries.
    """
    img_array = np.array(image, dtype=np.float32)
    h, w, c = img_array.shape
    result = img_array.copy()

    # Smooth along 8-pixel boundaries
    smoothing_strength = max(0.3, (80 - quality) / 100.0)

    for ch in range(c):
        # Horizontal boundaries
        for y in range(8, h - 1, 8):
            # Average the boundary row with its neighbors
            result[y, :, ch] = (
                img_array[y - 1, :, ch] * 0.25
                + img_array[y, :, ch] * 0.5
                + img_array[y + 1, :, ch] * 0.25
            ) * smoothing_strength + img_array[y, :, ch] * (1 - smoothing_strength)

        # Vertical boundaries
        for x in range(8, w - 1, 8):
            result[:, x, ch] = (
                img_array[:, x - 1, ch] * 0.25
                + img_array[:, x, ch] * 0.5
                + img_array[:, x + 1, ch] * 0.25
            ) * smoothing_strength + img_array[:, x, ch] * (1 - smoothing_strength)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def _sharpen(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Light unsharp masking for deblurring.

    Conservative sharpening that doesn't amplify noise.
    """
    # Unsharp mask: original + strength * (original - blurred)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=2.0))

    orig = np.array(image, dtype=np.float32)
    blur = np.array(blurred, dtype=np.float32)

    # Unsharp mask
    detail = orig - blur
    sharpened = orig + strength * detail
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return Image.fromarray(sharpened)
