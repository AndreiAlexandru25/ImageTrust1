"""
Source/Platform Detection Pack.

Detects image source and processing:
- Screenshot detection
- Social media platform hints (WhatsApp, Instagram, Facebook, Telegram)
- Platform-specific processing signatures
"""

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

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


# Common screen resolutions and aspect ratios
SCREEN_RESOLUTIONS = {
    # Mobile (portrait)
    (1080, 1920): "1080p Mobile",
    (1080, 2340): "FHD+ Mobile (19.5:9)",
    (1080, 2400): "FHD+ Mobile (20:9)",
    (1170, 2532): "iPhone 12/13/14",
    (1284, 2778): "iPhone 12/13/14 Pro Max",
    (1290, 2796): "iPhone 14/15 Pro Max",
    (1440, 3200): "QHD+ Mobile",
    (1440, 3088): "Samsung Galaxy S series",
    (750, 1334): "iPhone 6/7/8",
    (828, 1792): "iPhone XR/11",
    (1125, 2436): "iPhone X/XS/11 Pro",
    # Desktop
    (1920, 1080): "1080p Desktop",
    (2560, 1440): "1440p Desktop",
    (3840, 2160): "4K Desktop",
    (1366, 768): "HD Laptop",
    (1536, 864): "HD+ Laptop",
    (2560, 1600): "MacBook Air/Pro",
    (2880, 1800): "MacBook Pro Retina",
    # Tablet
    (2048, 2732): "iPad Pro 12.9",
    (1668, 2388): "iPad Pro 11",
    (2160, 1620): "iPad 10.2",
}

# Platform-specific characteristics
PLATFORM_SIGNATURES = {
    "whatsapp": {
        "max_dimension": 1600,
        "quality_range": (75, 85),
        "strips_exif": True,
        "typical_aspect_ratios": [(4, 3), (16, 9), (1, 1)],
    },
    "instagram": {
        "max_dimension": 1080,
        "quality_range": (70, 85),
        "strips_exif": True,
        "typical_aspect_ratios": [(1, 1), (4, 5), (16, 9)],
    },
    "facebook": {
        "max_dimension": 2048,
        "quality_range": (75, 90),
        "strips_exif": True,
        "typical_aspect_ratios": [(16, 9), (4, 3), (1, 1)],
    },
    "telegram": {
        "max_dimension": 1280,
        "quality_range": (80, 90),
        "strips_exif": False,  # Telegram preserves more metadata
        "typical_aspect_ratios": [(4, 3), (16, 9)],
    },
    "twitter": {
        "max_dimension": 4096,
        "quality_range": (80, 90),
        "strips_exif": True,
        "typical_aspect_ratios": [(16, 9), (4, 3)],
    },
}


@register_plugin
class ScreenshotDetector(ForensicsPlugin):
    """
    Screenshot detection.

    Detects if an image is likely a screenshot based on:
    - Resolution matching common screens
    - PNG format with no EXIF
    - UI element patterns (optional)
    - Color depth and compression characteristics
    """

    plugin_id = "screenshot_detector"
    plugin_name = "Screenshot Detection"
    category = PluginCategory.SOURCE
    description = "Detects if image is likely a screenshot"
    version = "1.0.0"

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Detect screenshot characteristics."""
        start_time = time.perf_counter()

        try:
            width, height = image.size
            indicators = []
            scores = []

            # 1. Check resolution against known screen sizes
            resolution_match = self._check_resolution((width, height))
            if resolution_match:
                indicators.append(f"Resolution matches: {resolution_match}")
                scores.append(0.4)

            # 2. Check for exact screen aspect ratios
            aspect_ratio = width / height
            common_ratios = {
                16 / 9: "16:9 (widescreen)",
                9 / 16: "9:16 (mobile portrait)",
                4 / 3: "4:3 (tablet)",
                3 / 4: "3:4 (tablet portrait)",
                19.5 / 9: "19.5:9 (modern mobile)",
                9 / 19.5: "9:19.5 (modern mobile portrait)",
                20 / 9: "20:9 (mobile)",
                9 / 20: "9:20 (mobile portrait)",
            }

            for ratio, name in common_ratios.items():
                if abs(aspect_ratio - ratio) < 0.02:
                    indicators.append(f"Aspect ratio: {name}")
                    scores.append(0.2)
                    break

            # 3. Check format - PNG screenshots are common
            if image.format == "PNG":
                # PNG without transparency might be screenshot
                if image.mode in ["RGB", "L"]:
                    indicators.append("PNG format without transparency")
                    scores.append(0.15)

            # 4. Check for lack of EXIF (screenshots typically have none)
            exif = image.getexif() if hasattr(image, "getexif") else {}
            if not exif or len(exif) == 0:
                indicators.append("No EXIF metadata")
                scores.append(0.15)

            # 5. Analyze for UI-like patterns (status bar regions)
            ui_hints = self._detect_ui_patterns(image)
            if ui_hints:
                indicators.extend(ui_hints)
                scores.append(0.3)

            # 6. Check color depth
            if image.mode == "RGB" and self._check_limited_colors(image):
                indicators.append("Limited color palette (typical of UI)")
                scores.append(0.1)

            # Calculate overall score
            total_score = min(1.0, sum(scores))
            detected = total_score > 0.4

            if total_score > 0.7:
                confidence = Confidence.HIGH
            elif total_score > 0.4:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            if detected:
                explanation = (
                    f"Screenshot likely: {', '.join(indicators[:3])}. "
                    f"Score: {total_score:.2f}"
                )
            else:
                explanation = (
                    f"Not clearly a screenshot (score={total_score:.2f}). "
                    f"Image dimensions and characteristics don't match typical screenshots."
                )

            limitations = [
                "Screenshots can be cropped, changing dimensions",
                "Some cameras produce screen-like resolutions",
                "Resized screenshots lose detection signals",
            ]

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=total_score,
                confidence=confidence,
                detected=detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "indicators": indicators,
                    "resolution": (width, height),
                    "aspect_ratio": float(aspect_ratio),
                    "format": image.format,
                    "mode": image.mode,
                    "resolution_match": resolution_match,
                    "has_exif": bool(exif),
                },
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Screenshot detection failed: {e}")
            return self._create_error_result(str(e))

    def _check_resolution(self, size: Tuple[int, int]) -> Optional[str]:
        """Check if resolution matches known screen sizes."""
        width, height = size

        # Check exact match
        if size in SCREEN_RESOLUTIONS:
            return SCREEN_RESOLUTIONS[size]

        # Check swapped (portrait vs landscape)
        swapped = (height, width)
        if swapped in SCREEN_RESOLUTIONS:
            return f"{SCREEN_RESOLUTIONS[swapped]} (portrait)"

        # Check with small tolerance (± 10 pixels for UI cropping)
        for res, name in SCREEN_RESOLUTIONS.items():
            if abs(width - res[0]) <= 10 and abs(height - res[1]) <= 10:
                return f"~{name}"
            if abs(width - res[1]) <= 10 and abs(height - res[0]) <= 10:
                return f"~{name} (portrait)"

        return None

    def _detect_ui_patterns(self, image: Image.Image) -> List[str]:
        """Detect UI-like patterns in image regions."""
        hints = []

        try:
            img_array = np.array(image.convert("RGB"))
            h, w = img_array.shape[:2]

            # Check top region for status bar (typically uniform color)
            if h > 100:
                top_region = img_array[:50, :, :]
                top_variance = np.var(top_region)

                if top_variance < 500:  # Very uniform = possible status bar
                    hints.append("Uniform top region (possible status bar)")

            # Check bottom region for navigation bar
            if h > 100:
                bottom_region = img_array[-50:, :, :]
                bottom_variance = np.var(bottom_region)

                if bottom_variance < 500:
                    hints.append("Uniform bottom region (possible nav bar)")

        except Exception:
            pass

        return hints

    def _check_limited_colors(self, image: Image.Image, threshold: int = 1000) -> bool:
        """Check if image has limited color palette (UI-like)."""
        try:
            # Sample colors
            img_small = image.resize((100, 100), Image.NEAREST)
            colors = img_small.getcolors(maxcolors=threshold)

            if colors and len(colors) < 500:
                return True

        except Exception:
            pass

        return False


@register_plugin
class PlatformDetector(ForensicsPlugin):
    """
    Social media platform detection.

    Detects if image was processed by specific platforms:
    - WhatsApp (aggressive compression, EXIF stripping)
    - Instagram (square crops, specific dimensions)
    - Facebook (moderate compression)
    - Telegram (preserves more quality)
    """

    plugin_id = "platform_detector"
    plugin_name = "Platform Detection (Social Media)"
    category = PluginCategory.SOURCE
    description = "Detects social media platform processing signatures"
    version = "1.0.0"

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Detect platform processing signatures."""
        start_time = time.perf_counter()

        try:
            width, height = image.size
            platform_scores = {}

            # Check EXIF
            exif = image.getexif() if hasattr(image, "getexif") else {}
            has_exif = bool(exif) and len(exif) > 0

            # Estimate JPEG quality if possible
            jpeg_quality = self._estimate_jpeg_quality(image, raw_bytes)

            # Analyze for each platform
            for platform, sig in PLATFORM_SIGNATURES.items():
                score = 0.0
                reasons = []

                # 1. Check max dimension
                max_dim = max(width, height)
                expected_max = sig["max_dimension"]

                if max_dim == expected_max:
                    score += 0.4
                    reasons.append(f"Max dimension matches ({max_dim}px)")
                elif abs(max_dim - expected_max) < 50:
                    score += 0.2
                    reasons.append(f"Max dimension close to {expected_max}px")

                # 2. Check EXIF stripping
                if sig["strips_exif"] and not has_exif:
                    score += 0.2
                    reasons.append("EXIF stripped (consistent with platform)")
                elif not sig["strips_exif"] and has_exif:
                    score += 0.1
                    reasons.append("EXIF preserved (consistent with platform)")

                # 3. Check quality range
                if jpeg_quality:
                    q_min, q_max = sig["quality_range"]
                    if q_min <= jpeg_quality <= q_max:
                        score += 0.2
                        reasons.append(f"JPEG quality in range ({jpeg_quality}%)")

                # 4. Check aspect ratio
                aspect = width / height
                for ar in sig["typical_aspect_ratios"]:
                    expected_ar = ar[0] / ar[1]
                    if abs(aspect - expected_ar) < 0.05:
                        score += 0.2
                        reasons.append(f"Aspect ratio matches {ar[0]}:{ar[1]}")
                        break

                platform_scores[platform] = {
                    "score": min(1.0, score),
                    "reasons": reasons,
                }

            # Find best match
            best_platform = max(platform_scores.keys(), key=lambda p: platform_scores[p]["score"])
            best_score = platform_scores[best_platform]["score"]
            best_reasons = platform_scores[best_platform]["reasons"]

            # Detection threshold
            detected = best_score > 0.5

            if best_score > 0.7:
                confidence = Confidence.HIGH
            elif best_score > 0.5:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            if detected:
                explanation = (
                    f"Image characteristics match {best_platform.title()}: "
                    f"{', '.join(best_reasons[:2])} (score={best_score:.2f})"
                )
            else:
                explanation = (
                    f"No strong platform match. Best guess: {best_platform.title()} "
                    f"(score={best_score:.2f})"
                )

            limitations = [
                "Platform signatures overlap significantly",
                "Resaving images changes compression signatures",
                "This is a probabilistic hint, not definitive identification",
            ]

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=best_score,
                confidence=confidence,
                detected=detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "platform_scores": platform_scores,
                    "best_platform": best_platform if detected else None,
                    "dimensions": (width, height),
                    "has_exif": has_exif,
                    "estimated_jpeg_quality": jpeg_quality,
                },
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Platform detection failed: {e}")
            return self._create_error_result(str(e))

    def _estimate_jpeg_quality(self, image: Image.Image, raw_bytes: Optional[bytes]) -> Optional[int]:
        """Estimate JPEG quality level."""
        try:
            # Method 1: Check quantization tables if available
            if hasattr(image, "quantization") and image.quantization:
                # Simplified quality estimation from quantization tables
                q_tables = image.quantization
                if 0 in q_tables:
                    avg_q = np.mean(list(q_tables[0].values()))
                    # Lower average = higher quality
                    quality = max(1, min(100, int(100 - avg_q)))
                    return quality

            # Method 2: Recompress and compare file sizes
            if raw_bytes and image.format == "JPEG":
                original_size = len(raw_bytes)

                for quality in [95, 85, 75, 65, 55]:
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=quality)
                    test_size = len(buffer.getvalue())

                    if test_size > original_size * 0.95:
                        return quality

                return 50  # Very low quality

        except Exception:
            pass

        return None


@register_plugin
class CompressionHistoryDetector(ForensicsPlugin):
    """
    Compression history analyzer.

    Analyzes compression artifacts to determine if image
    has been recompressed multiple times.
    """

    plugin_id = "compression_history"
    plugin_name = "Compression History Analysis"
    category = PluginCategory.SOURCE
    description = "Analyzes image compression history and quality"
    version = "1.0.0"

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Analyze compression history."""
        start_time = time.perf_counter()

        try:
            findings = []

            # Get format
            img_format = image.format or "Unknown"

            if img_format == "PNG":
                # PNG is lossless
                findings.append("PNG format (lossless compression)")
                score = 0.0
                detected = False
                explanation = "Image is in lossless PNG format - no JPEG compression artifacts"

            elif img_format in ["JPEG", "JPG", None]:
                # Analyze JPEG compression
                compression_analysis = self._analyze_jpeg_compression(image, raw_bytes)
                findings.extend(compression_analysis["findings"])

                score = compression_analysis["recompression_score"]
                detected = score > 0.3

                if detected:
                    explanation = (
                        f"Compression analysis suggests reprocessing: "
                        f"{', '.join(compression_analysis['findings'][:2])}"
                    )
                else:
                    explanation = "Compression appears consistent with single JPEG save"

            else:
                findings.append(f"Format: {img_format}")
                score = 0.0
                detected = False
                explanation = f"Image format is {img_format}"

            confidence = Confidence.MEDIUM  # Compression analysis is inherently uncertain

            limitations = [
                "High-quality JPEG is hard to distinguish from original",
                "Some cameras produce heavily compressed output",
                "Multiple factors affect compression artifacts",
            ]

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=score,
                confidence=confidence,
                detected=detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "format": img_format,
                    "findings": findings,
                },
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Compression history analysis failed: {e}")
            return self._create_error_result(str(e))

    def _analyze_jpeg_compression(self, image: Image.Image, raw_bytes: Optional[bytes]) -> Dict[str, Any]:
        """Analyze JPEG-specific compression."""
        findings = []
        recompression_score = 0.0

        try:
            img_array = np.array(image.convert("RGB"), dtype=np.float32)
            gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

            # Check blocking artifacts
            h, w = gray.shape
            block_boundaries_h = []
            block_boundaries_v = []

            for x in range(8, w - 8, 8):
                diff = np.mean(np.abs(gray[:, x] - gray[:, x - 1]))
                block_boundaries_h.append(diff)

            for y in range(8, h - 8, 8):
                diff = np.mean(np.abs(gray[y, :] - gray[y - 1, :]))
                block_boundaries_v.append(diff)

            avg_blocking = (np.mean(block_boundaries_h) + np.mean(block_boundaries_v)) / 2

            if avg_blocking > 3.0:
                findings.append(f"Significant blocking artifacts ({avg_blocking:.1f})")
                recompression_score += 0.4
            elif avg_blocking > 1.5:
                findings.append(f"Moderate blocking artifacts ({avg_blocking:.1f})")
                recompression_score += 0.2

            # Estimate quality from artifacts
            if avg_blocking > 5.0:
                findings.append("Quality estimate: <70%")
            elif avg_blocking > 2.0:
                findings.append("Quality estimate: 70-85%")
            else:
                findings.append("Quality estimate: >85%")

        except Exception as e:
            findings.append(f"Analysis error: {e}")

        return {
            "findings": findings,
            "recompression_score": min(1.0, recompression_score),
        }
