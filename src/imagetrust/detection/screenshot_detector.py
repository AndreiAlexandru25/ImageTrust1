"""
Screenshot / Screen Capture / Game Render Detection Module.

Detects images that are neither photographs nor AI-generated, but rather:
- Screenshots (desktop, mobile)
- Game renders / game captures
- Screen recordings (single frames)
- UI/application captures

These images should NOT be classified as "real" (no camera) or "AI-generated"
(no generative model). They get the "screenshot" verdict.

Detection is heuristic-based, analyzing:
1. Noise profile (screenshots have very uniform/low sensor noise)
2. Edge characteristics (sharp UI edges, aliasing patterns)
3. Color quantization (limited color palettes, flat regions)
4. Resolution patterns (standard screen resolutions)
5. Metadata absence (no EXIF camera data, PNG format)
6. Frequency domain (unnatural frequency distribution)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Standard screen resolutions (width, height)
SCREEN_RESOLUTIONS = {
    (1920, 1080), (2560, 1440), (3840, 2160),  # 16:9
    (1366, 768), (1536, 864), (1600, 900),      # Laptop
    (2560, 1600), (2880, 1800), (3024, 1964),   # Mac
    (1080, 1920), (1170, 2532), (1284, 2778),   # Mobile portrait
    (1440, 3200), (1080, 2400), (1440, 3120),   # Android
    (750, 1334), (1125, 2436), (828, 1792),     # iPhone
    (2048, 1536), (2732, 2048),                  # iPad
    (1280, 720), (1280, 800), (1440, 900),      # Misc
}


@dataclass
class ScreenshotScore:
    """Screenshot detection result."""
    is_screenshot: bool
    probability: float  # 0-1
    confidence: float   # 0-1
    indicators: list     # List of detected indicators
    details: Dict[str, Any]


def detect_screenshot(
    image: Image.Image,
    exif_data: Optional[Dict] = None,
    file_format: Optional[str] = None,
    filename: Optional[str] = None,
) -> ScreenshotScore:
    """
    Detect if an image is a screenshot or screen capture.

    Args:
        image: PIL Image to analyze.
        exif_data: Optional EXIF metadata dict.
        file_format: Original file format (e.g., "PNG", "JPEG").
        filename: Original filename (for keyword detection).

    Returns:
        ScreenshotScore with detection results.
    """
    scores = []
    indicators = []
    details = {}

    img_array = np.array(image.convert("RGB"), dtype=np.float64)
    h, w = img_array.shape[:2]

    # 0. Filename analysis — strongest signal when present
    filename_score, filename_detail = _analyze_filename(filename)
    scores.append(("filename", filename_score, 0.25))
    details["filename"] = filename_detail
    if filename_score > 0.6:
        indicators.append(
            f"Filename contains screenshot keyword: {filename_detail.get('match', '')}"
        )

    # 1. Noise profile analysis
    noise_score, noise_detail = _analyze_noise_profile(img_array)
    scores.append(("noise", noise_score, 0.15))
    details["noise"] = noise_detail
    if noise_score > 0.6:
        indicators.append("Very low/uniform sensor noise (typical of renders)")

    # 2. Edge sharpness analysis
    edge_score, edge_detail = _analyze_edge_sharpness(img_array)
    scores.append(("edges", edge_score, 0.10))
    details["edges"] = edge_detail
    if edge_score > 0.6:
        indicators.append("Unnaturally sharp edges (typical of UI/game renders)")

    # 3. Color quantization / flat regions
    flat_score, flat_detail = _analyze_flat_regions(img_array)
    scores.append(("flat_regions", flat_score, 0.10))
    details["flat_regions"] = flat_detail
    if flat_score > 0.5:
        indicators.append("Large flat color regions (typical of UI elements)")

    # 4. Resolution pattern
    res_score = _check_resolution_pattern(w, h)
    scores.append(("resolution", res_score, 0.10))
    details["resolution"] = {"width": w, "height": h, "matches_screen": res_score > 0.5}
    if res_score > 0.5:
        indicators.append(f"Resolution {w}x{h} matches standard screen resolution")

    # 5. Metadata analysis
    meta_score = _analyze_metadata(exif_data, file_format)
    scores.append(("metadata", meta_score, 0.15))
    details["metadata"] = {
        "has_exif": bool(exif_data),
        "format": file_format,
        "no_camera": meta_score > 0.5,
    }
    if meta_score > 0.5:
        indicators.append("No camera EXIF data (not from a camera sensor)")

    # 6. Frequency analysis
    freq_score, freq_detail = _analyze_frequency_profile(img_array)
    scores.append(("frequency", freq_score, 0.15))
    details["frequency"] = freq_detail
    if freq_score > 0.6:
        indicators.append("Frequency profile inconsistent with camera capture")

    # Weighted combination
    total_weight = sum(w for _, _, w in scores)
    weighted_prob = sum(s * w for _, s, w in scores) / total_weight

    # Confidence based on agreement
    score_values = [s for _, s, _ in scores]
    agreement = 1.0 - np.std(score_values) * 2
    confidence = max(0.3, min(1.0, agreement))

    is_screenshot = weighted_prob > 0.42

    return ScreenshotScore(
        is_screenshot=is_screenshot,
        probability=float(np.clip(weighted_prob, 0, 1)),
        confidence=float(np.clip(confidence, 0, 1)),
        indicators=indicators,
        details=details,
    )


def _analyze_filename(
    filename: Optional[str],
) -> Tuple[float, Dict[str, Any]]:
    """
    Analyze filename for screenshot keywords.
    This is the strongest single signal — 'Screenshot' in the filename
    is almost always definitive.
    """
    if not filename:
        return 0.3, {"match": None, "filename": None}

    name_lower = filename.lower()
    # Strong keywords (definitive screenshot indicators)
    strong_keywords = [
        "screenshot", "screen shot", "screen_shot", "screen-shot",
        "capture", "snip", "printscreen", "game_capture",
        "obs_", "sharex", "gyazo", "greenshot", "lightshot",
    ]
    # Moderate keywords (likely but not definitive)
    moderate_keywords = [
        "screen", "desktop", "display", "monitor", "window",
        "gameplay", "game_", "clip_",
    ]

    for kw in strong_keywords:
        if kw in name_lower:
            return 0.95, {"match": kw, "filename": filename, "strength": "strong"}

    for kw in moderate_keywords:
        if kw in name_lower:
            return 0.65, {"match": kw, "filename": filename, "strength": "moderate"}

    return 0.15, {"match": None, "filename": filename}


def _analyze_noise_profile(
    img: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    """
    Analyze noise characteristics.
    Screenshots/renders have very low and uniform noise.
    Camera photos have higher, spatially varying sensor noise.
    """
    # Estimate noise via high-pass filter (Laplacian)
    gray = np.mean(img, axis=2)

    # Laplacian for noise estimation
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    h, w = gray.shape

    # Apply Laplacian via convolution (simple)
    padded = np.pad(gray, 1, mode="reflect")
    lap = np.zeros_like(gray)
    for di in range(3):
        for dj in range(3):
            lap += padded[di:di + h, dj:dj + w] * laplacian[di, dj]

    noise_std = np.std(lap) / (np.mean(np.abs(lap)) + 1e-10)

    # Low noise → likely screenshot/render
    # Typical camera noise_std: 5-20
    # Typical screenshot noise_std: 1-5
    local_noise_std = np.std(lap)

    # Check noise uniformity across quadrants
    qh, qw = h // 2, w // 2
    quad_stds = [
        np.std(lap[:qh, :qw]),
        np.std(lap[:qh, qw:]),
        np.std(lap[qh:, :qw]),
        np.std(lap[qh:, qw:]),
    ]
    noise_uniformity = 1.0 - (np.std(quad_stds) / (np.mean(quad_stds) + 1e-10))

    # Score: low noise + uniform → screenshot
    if local_noise_std < 3.0:
        noise_score = 0.8
    elif local_noise_std < 6.0:
        noise_score = 0.5
    elif local_noise_std < 12.0:
        noise_score = 0.3
    else:
        noise_score = 0.1

    # Boost if very uniform
    if noise_uniformity > 0.85:
        noise_score = min(1.0, noise_score + 0.15)

    return noise_score, {
        "noise_std": float(local_noise_std),
        "noise_uniformity": float(noise_uniformity),
        "quadrant_stds": [float(q) for q in quad_stds],
    }


def _analyze_edge_sharpness(
    img: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    """
    Analyze edge characteristics.
    Screenshots have many perfectly sharp, axis-aligned edges.
    Camera photos have softer, more organic edges.
    """
    gray = np.mean(img, axis=2)
    h, w = gray.shape

    # Horizontal and vertical gradients
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))

    # Count very sharp transitions (>30 intensity jump in 1 pixel)
    sharp_h = np.sum(dx > 30)
    sharp_v = np.sum(dy > 30)
    total_pixels = h * w

    sharp_ratio = (sharp_h + sharp_v) / (2 * total_pixels)

    # Check for axis-aligned dominance (UI elements are axis-aligned)
    h_energy = np.sum(dx)
    v_energy = np.sum(dy)
    total_energy = h_energy + v_energy + 1e-10
    axis_dominance = max(h_energy, v_energy) / total_energy

    # Score
    if sharp_ratio > 0.05:
        edge_score = 0.8
    elif sharp_ratio > 0.02:
        edge_score = 0.5
    elif sharp_ratio > 0.01:
        edge_score = 0.3
    else:
        edge_score = 0.1

    # Boost if axis-aligned
    if axis_dominance > 0.6:
        edge_score = min(1.0, edge_score + 0.1)

    return edge_score, {
        "sharp_ratio": float(sharp_ratio),
        "axis_dominance": float(axis_dominance),
        "h_energy": float(h_energy),
        "v_energy": float(v_energy),
    }


def _analyze_flat_regions(
    img: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    """
    Analyze flat color regions.
    Screenshots/UI have large areas of exactly the same color.
    Camera photos almost never have perfectly flat regions.
    """
    # Downsample for speed
    h, w = img.shape[:2]
    step = max(1, min(h, w) // 256)
    small = img[::step, ::step]

    # Count unique colors in 8x8 blocks
    bh, bw = small.shape[0] // 8, small.shape[1] // 8
    flat_blocks = 0
    total_blocks = 0

    for i in range(8):
        for j in range(8):
            block = small[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            if block.size == 0:
                continue
            total_blocks += 1
            # Check color variance in block
            variance = np.mean(np.var(block.reshape(-1, 3), axis=0))
            if variance < 10.0:  # Very flat
                flat_blocks += 1

    flat_ratio = flat_blocks / max(total_blocks, 1)

    # Count exact duplicate pixels (quantized)
    quantized = (small // 4) * 4  # Quantize to reduce noise
    flat_q = quantized.reshape(-1, 3)
    unique_colors = len(np.unique(flat_q, axis=0))
    total_pixels = flat_q.shape[0]
    color_diversity = unique_colors / max(total_pixels, 1)

    # Score: many flat blocks + low color diversity → screenshot
    if flat_ratio > 0.4:
        flat_score = 0.85
    elif flat_ratio > 0.25:
        flat_score = 0.6
    elif flat_ratio > 0.15:
        flat_score = 0.4
    else:
        flat_score = 0.15

    # Low color diversity boosts score
    if color_diversity < 0.1:
        flat_score = min(1.0, flat_score + 0.15)

    return flat_score, {
        "flat_ratio": float(flat_ratio),
        "flat_blocks": flat_blocks,
        "total_blocks": total_blocks,
        "unique_colors": unique_colors,
        "color_diversity": float(color_diversity),
    }


def _check_resolution_pattern(w: int, h: int) -> float:
    """
    Check if resolution matches common DESKTOP screen resolutions.
    Excludes common phone camera resolutions to avoid false positives.
    """
    # Desktop/monitor exact matches (landscape orientation)
    desktop_resolutions = {
        (1920, 1080), (2560, 1440), (3840, 2160),  # 16:9
        (1366, 768), (1536, 864), (1600, 900),      # Laptop
        (2560, 1600), (2880, 1800), (3024, 1964),   # Mac
        (1280, 720), (1280, 800), (1280, 1024),     # Standard
        (1440, 900), (1680, 1050), (1920, 1200),    # WUXGA
        (2048, 1536), (2732, 2048),                  # iPad
    }

    # Exact match (either orientation)
    if (w, h) in desktop_resolutions or (h, w) in desktop_resolutions:
        return 0.85

    # Desktop-only widths (NOT phone camera widths)
    desktop_widths = {1280, 1366, 1440, 1536, 1600, 1680, 1920, 2048,
                      2560, 2880, 3024, 3440, 3840}
    desktop_heights = {720, 768, 800, 864, 900, 1024, 1050, 1080,
                       1200, 1440, 1536, 1600, 1800, 2160}

    # Both dimensions match desktop standards
    if w in desktop_widths and h in desktop_heights:
        return 0.65
    if h in desktop_widths and w in desktop_heights:
        return 0.65

    # Single dimension match — weak signal (phones also hit these)
    if w in desktop_widths or h in desktop_widths:
        return 0.30

    return 0.10


def _analyze_metadata(
    exif_data: Optional[Dict],
    file_format: Optional[str],
) -> float:
    """
    Analyze metadata for screenshot indicators.
    Screenshots typically: PNG format, no EXIF, no camera info.
    """
    score = 0.3  # Base

    # PNG without EXIF is very common for screenshots
    if file_format and file_format.upper() == "PNG":
        score += 0.2
    elif file_format and file_format.upper() == "BMP":
        score += 0.3

    # No EXIF at all
    if not exif_data:
        score += 0.25
    else:
        # Has EXIF but no camera info
        has_camera = (
            exif_data.get("make") or exif_data.get("model")
            or exif_data.get("exposure_time") or exif_data.get("f_number")
        )
        if not has_camera:
            score += 0.15

        # Check for screenshot-related software
        software = str(exif_data.get("software", "")).lower()
        screenshot_sw = [
            "snipping", "screenshot", "greenshot", "sharex",
            "lightshot", "gyazo", "snagit", "obs", "nvidia",
            "game bar", "xbox", "steam", "geforce",
        ]
        if any(sw in software for sw in screenshot_sw):
            score += 0.3

    return min(1.0, score)


def _analyze_frequency_profile(
    img: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    """
    Analyze frequency domain characteristics.
    Screenshots have different frequency profiles than camera photos.
    Renders lack natural high-frequency noise from camera sensors.
    """
    gray = np.mean(img, axis=2)

    # 2D FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Radial frequency analysis
    # Low freq (center), mid freq, high freq (edges)
    r_max = min(cy, cx)
    r_low = r_max // 4
    r_mid = r_max // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    low_energy = np.mean(magnitude[r < r_low])
    mid_energy = np.mean(magnitude[(r >= r_low) & (r < r_mid)])
    high_energy = np.mean(magnitude[r >= r_mid])
    total_energy = low_energy + mid_energy + high_energy + 1e-10

    # Screenshots: high low/mid ratio, low high-frequency
    high_freq_ratio = high_energy / total_energy
    low_freq_ratio = low_energy / total_energy

    # Camera photos have more high-frequency content (sensor noise)
    # Screenshots/renders are "cleaner" in high frequencies
    if high_freq_ratio < 0.15:
        freq_score = 0.75
    elif high_freq_ratio < 0.22:
        freq_score = 0.5
    elif high_freq_ratio < 0.30:
        freq_score = 0.3
    else:
        freq_score = 0.1

    return freq_score, {
        "low_energy": float(low_energy),
        "mid_energy": float(mid_energy),
        "high_energy": float(high_energy),
        "high_freq_ratio": float(high_freq_ratio),
        "low_freq_ratio": float(low_freq_ratio),
    }
