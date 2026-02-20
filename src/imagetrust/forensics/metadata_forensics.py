"""
Metadata Forensics Pack.

Comprehensive metadata analysis for image forensics:
- EXIF/IPTC/XMP extraction and analysis
- Metadata stripping detection
- Software traces (Photoshop, Lightroom, etc.)
- Thumbnail mismatch detection
- Consistency checks
"""

import hashlib
import io
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# Software signatures for editing detection
EDITING_SOFTWARE = {
    # Adobe products
    "adobe photoshop": {"category": "professional_editor", "ai_capable": True},
    "photoshop": {"category": "professional_editor", "ai_capable": True},
    "adobe lightroom": {"category": "professional_editor", "ai_capable": True},
    "lightroom": {"category": "professional_editor", "ai_capable": True},
    "adobe illustrator": {"category": "vector_editor", "ai_capable": False},
    # Mobile apps
    "snapseed": {"category": "mobile_editor", "ai_capable": True},
    "vsco": {"category": "mobile_editor", "ai_capable": False},
    "instagram": {"category": "social_media", "ai_capable": False},
    "snapchat": {"category": "social_media", "ai_capable": True},
    "facetune": {"category": "retouching", "ai_capable": True},
    "picsart": {"category": "mobile_editor", "ai_capable": True},
    "meitu": {"category": "beautification", "ai_capable": True},
    "beauty plus": {"category": "beautification", "ai_capable": True},
    # Desktop tools
    "gimp": {"category": "open_source_editor", "ai_capable": False},
    "paint.net": {"category": "basic_editor", "ai_capable": False},
    "affinity photo": {"category": "professional_editor", "ai_capable": True},
    "corel": {"category": "professional_editor", "ai_capable": True},
    "capture one": {"category": "raw_processor", "ai_capable": False},
    "dxo": {"category": "raw_processor", "ai_capable": True},
    "topaz": {"category": "ai_enhancer", "ai_capable": True},
    "luminar": {"category": "ai_editor", "ai_capable": True},
    # AI generators
    "midjourney": {"category": "ai_generator", "ai_capable": True},
    "dall-e": {"category": "ai_generator", "ai_capable": True},
    "dalle": {"category": "ai_generator", "ai_capable": True},
    "stable diffusion": {"category": "ai_generator", "ai_capable": True},
    "leonardo": {"category": "ai_generator", "ai_capable": True},
    "firefly": {"category": "ai_generator", "ai_capable": True},
    "runway": {"category": "ai_generator", "ai_capable": True},
    # Screenshot tools
    "screenshot": {"category": "screenshot", "ai_capable": False},
    "snipping tool": {"category": "screenshot", "ai_capable": False},
    "greenshot": {"category": "screenshot", "ai_capable": False},
    "lightshot": {"category": "screenshot", "ai_capable": False},
}

# Camera manufacturer patterns
CAMERA_MANUFACTURERS = [
    "canon", "nikon", "sony", "fujifilm", "panasonic", "olympus",
    "leica", "pentax", "hasselblad", "phase one", "samsung", "lg",
    "apple", "google", "huawei", "xiaomi", "oppo", "vivo", "oneplus",
]


@register_plugin
class MetadataAnalyzer(ForensicsPlugin):
    """
    Comprehensive metadata analyzer.

    Extracts and analyzes EXIF, IPTC, XMP metadata to determine
    image provenance and detect manipulation signs.
    """

    plugin_id = "metadata_analyzer"
    plugin_name = "Metadata Analysis"
    category = PluginCategory.METADATA
    description = "Comprehensive EXIF/IPTC/XMP metadata extraction and analysis"
    version = "1.0.0"

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Analyze image metadata."""
        start_time = time.perf_counter()

        try:
            metadata = {
                "exif": {},
                "iptc": {},
                "xmp": {},
                "important_tags": [],
                "warnings": [],
            }

            # Extract EXIF
            exif_data = self._extract_exif(image)
            metadata["exif"] = exif_data

            # Extract XMP if possible
            xmp_data = self._extract_xmp(raw_bytes)
            metadata["xmp"] = xmp_data

            # Analyze metadata completeness
            analysis = self._analyze_metadata(exif_data, xmp_data)

            # Determine if metadata suggests authentic camera image
            has_camera_info = bool(exif_data.get("make") or exif_data.get("model"))
            has_exposure = bool(exif_data.get("exposure_time") or exif_data.get("f_number"))
            has_datetime = bool(exif_data.get("datetime_original"))

            # Score calculation
            authenticity_indicators = sum([
                has_camera_info,
                has_exposure,
                has_datetime,
                bool(exif_data.get("iso")),
                bool(exif_data.get("focal_length")),
                bool(exif_data.get("gps")),
            ])

            # Check for suspicious patterns
            suspicious_patterns = analysis.get("suspicious_patterns", [])
            software_detected = analysis.get("software_detected")

            # Determine overall score
            # High score = looks like authentic camera image
            score = authenticity_indicators / 6.0

            if software_detected:
                sw_info = EDITING_SOFTWARE.get(software_detected.lower(), {})
                if sw_info.get("category") == "ai_generator":
                    score = 0.1  # Very low - likely AI generated
                elif sw_info.get("category") in ["professional_editor", "mobile_editor"]:
                    score *= 0.7  # Reduce but don't eliminate

            # Detection: we detect "metadata issues" not "fake"
            metadata_stripped = authenticity_indicators == 0 and image.format not in ["PNG", "BMP"]
            detected = metadata_stripped or bool(suspicious_patterns) or bool(software_detected)

            if authenticity_indicators >= 4:
                confidence = Confidence.HIGH
            elif authenticity_indicators >= 2:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            # Build explanation
            if metadata_stripped:
                explanation = "Metadata appears stripped - no camera information, dates, or settings found"
            elif software_detected:
                explanation = f"Software trace detected: {software_detected}"
            elif suspicious_patterns:
                explanation = f"Suspicious patterns: {', '.join(suspicious_patterns)}"
            elif has_camera_info and has_exposure:
                explanation = f"Metadata consistent with camera capture ({exif_data.get('make', 'Unknown')} {exif_data.get('model', '')})"
            else:
                explanation = "Limited metadata available - authenticity uncertain"

            # Compile important tags
            important_tags = []
            if exif_data.get("make"):
                important_tags.append(f"Camera: {exif_data['make']} {exif_data.get('model', '')}")
            if exif_data.get("datetime_original"):
                important_tags.append(f"Date: {exif_data['datetime_original']}")
            if exif_data.get("software"):
                important_tags.append(f"Software: {exif_data['software']}")
            if exif_data.get("gps"):
                important_tags.append("GPS coordinates present")

            limitations = [
                "Metadata can be stripped or forged",
                "Some cameras/phones strip metadata by default",
                "Screenshot and web downloads typically lack metadata",
            ]

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=score,
                confidence=confidence,
                detected=detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "exif": exif_data,
                    "important_tags": important_tags,
                    "authenticity_indicators": authenticity_indicators,
                    "has_camera_info": has_camera_info,
                    "has_exposure": has_exposure,
                    "has_datetime": has_datetime,
                    "software_detected": software_detected,
                    "suspicious_patterns": suspicious_patterns,
                    "metadata_stripped": metadata_stripped,
                },
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Metadata analysis failed: {e}")
            return self._create_error_result(str(e))

    def _extract_exif(self, image: Image.Image) -> Dict[str, Any]:
        """Extract EXIF data from image."""
        exif_data = {}

        try:
            exif = image.getexif() if hasattr(image, "getexif") else {}

            if exif:
                from PIL.ExifTags import TAGS, GPSTAGS

                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))

                    # Handle special cases
                    if tag_name == "GPSInfo":
                        gps_data = {}
                        for gps_tag_id, gps_value in value.items():
                            gps_tag_name = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                            gps_data[gps_tag_name] = str(gps_value)
                        exif_data["gps"] = gps_data
                    else:
                        exif_data[tag_name.lower().replace(" ", "_")] = str(value)

        except Exception as e:
            logger.debug(f"EXIF extraction error: {e}")

        return exif_data

    def _extract_xmp(self, raw_bytes: Optional[bytes]) -> Dict[str, Any]:
        """Extract XMP data from raw bytes."""
        if not raw_bytes:
            return {}

        xmp_data = {}
        try:
            # Simple XMP extraction - look for XML packet
            xmp_start = raw_bytes.find(b"<x:xmpmeta")
            xmp_end = raw_bytes.find(b"</x:xmpmeta>")

            if xmp_start != -1 and xmp_end != -1:
                xmp_packet = raw_bytes[xmp_start:xmp_end + 12].decode("utf-8", errors="ignore")
                xmp_data["raw_xmp_found"] = True
                xmp_data["xmp_length"] = len(xmp_packet)

                # Extract key fields
                if "photoshop" in xmp_packet.lower():
                    xmp_data["photoshop_traces"] = True
                if "lightroom" in xmp_packet.lower():
                    xmp_data["lightroom_traces"] = True
                if "CreatorTool" in xmp_packet:
                    match = re.search(r'CreatorTool["\s>]+([^<"]+)', xmp_packet)
                    if match:
                        xmp_data["creator_tool"] = match.group(1)

        except Exception as e:
            logger.debug(f"XMP extraction error: {e}")

        return xmp_data

    def _analyze_metadata(self, exif: Dict, xmp: Dict) -> Dict[str, Any]:
        """Analyze metadata for suspicious patterns."""
        analysis = {
            "suspicious_patterns": [],
            "software_detected": None,
        }

        # Check software field
        software = exif.get("software", "").lower()
        for sw_name in EDITING_SOFTWARE:
            if sw_name in software:
                analysis["software_detected"] = sw_name
                break

        # Check for inconsistencies
        if exif.get("make") and not exif.get("model"):
            analysis["suspicious_patterns"].append("Camera make without model")

        if exif.get("datetime_original"):
            try:
                dt = datetime.strptime(str(exif["datetime_original"]), "%Y:%m:%d %H:%M:%S")
                if dt.year < 1990 or dt.year > datetime.now().year + 1:
                    analysis["suspicious_patterns"].append(f"Implausible date: {dt.year}")
            except ValueError:
                pass

        return analysis


@register_plugin
class ThumbnailMismatchDetector(ForensicsPlugin):
    """
    Thumbnail mismatch detector.

    Compares embedded EXIF thumbnail with main image to detect
    if the main image was edited after thumbnail generation.
    """

    plugin_id = "thumbnail_mismatch"
    plugin_name = "Thumbnail Mismatch Detection"
    category = PluginCategory.METADATA
    description = "Detects if embedded thumbnail doesn't match main image"
    version = "1.0.0"

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Compare thumbnail with main image."""
        start_time = time.perf_counter()

        try:
            # Try to extract thumbnail
            thumbnail = self._extract_thumbnail(image, raw_bytes)

            if thumbnail is None:
                return self._create_result(
                    score=0.0,
                    confidence=Confidence.VERY_LOW,
                    detected=False,
                    explanation="No embedded thumbnail found - cannot perform mismatch detection",
                    limitations=["Many image formats don't include thumbnails"],
                )

            # Resize main image to thumbnail size for comparison
            main_resized = image.convert("RGB").resize(thumbnail.size, Image.LANCZOS)
            thumb_rgb = thumbnail.convert("RGB")

            # Calculate similarity metrics
            main_arr = np.array(main_resized, dtype=np.float32)
            thumb_arr = np.array(thumb_rgb, dtype=np.float32)

            # MSE and structural similarity
            mse = np.mean((main_arr - thumb_arr) ** 2)

            # Histogram comparison
            hist_diff = self._compare_histograms(main_resized, thumb_rgb)

            # Detection threshold
            mismatch_detected = mse > 500 or hist_diff > 0.3

            score = min(1.0, mse / 1000 + hist_diff)

            if mse > 1000 or hist_diff > 0.5:
                confidence = Confidence.HIGH
            elif mse > 500 or hist_diff > 0.3:
                confidence = Confidence.MEDIUM
            else:
                confidence = Confidence.LOW

            if mismatch_detected:
                explanation = (
                    f"Thumbnail mismatch detected (MSE={mse:.1f}, histogram diff={hist_diff:.2f}). "
                    f"Main image may have been edited after thumbnail generation."
                )
            else:
                explanation = (
                    f"Thumbnail matches main image (MSE={mse:.1f}, histogram diff={hist_diff:.2f}). "
                    f"No signs of post-thumbnail editing."
                )

            limitations = [
                "Some editors regenerate thumbnails after edits",
                "JPEG recompression can cause small differences",
                "Not all images contain thumbnails",
            ]

            import numpy as np

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=score,
                confidence=confidence,
                detected=mismatch_detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "mse": float(mse),
                    "histogram_diff": float(hist_diff),
                    "thumbnail_size": thumbnail.size,
                    "main_size": image.size,
                },
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Thumbnail mismatch detection failed: {e}")
            return self._create_error_result(str(e))

    def _extract_thumbnail(self, image: Image.Image, raw_bytes: Optional[bytes]) -> Optional[Image.Image]:
        """Extract embedded thumbnail."""
        try:
            exif = image.getexif()
            if exif:
                # EXIF thumbnail is often at tag 0x0201 (JpegThumbnail) or in IFD1
                for tag_id in [0x0201, 0x00FE]:
                    if tag_id in exif:
                        thumb_data = exif[tag_id]
                        if isinstance(thumb_data, bytes):
                            return Image.open(io.BytesIO(thumb_data))
        except Exception:
            pass

        return None

    def _compare_histograms(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compare color histograms of two images."""
        import numpy as np

        hist1 = img1.histogram()
        hist2 = img2.histogram()

        # Normalize
        hist1 = np.array(hist1, dtype=np.float32)
        hist2 = np.array(hist2, dtype=np.float32)
        hist1 = hist1 / (np.sum(hist1) + 1e-10)
        hist2 = hist2 / (np.sum(hist2) + 1e-10)

        # Chi-square distance
        diff = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))

        return float(diff)


@register_plugin
class SoftwareTraceDetector(ForensicsPlugin):
    """
    Software trace detector.

    Specifically looks for traces of editing software in metadata
    and file structure.
    """

    plugin_id = "software_traces"
    plugin_name = "Software Trace Detection"
    category = PluginCategory.METADATA
    description = "Detects editing software traces in metadata"
    version = "1.0.0"

    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """Detect software traces."""
        start_time = time.perf_counter()

        try:
            traces_found = []
            software_info = {}

            # Check EXIF software tag
            exif = image.getexif() if hasattr(image, "getexif") else {}
            software_tag = None

            if exif:
                from PIL.ExifTags import TAGS
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, "")
                    if tag_name.lower() == "software":
                        software_tag = str(value)
                        break

            if software_tag:
                traces_found.append(f"EXIF Software: {software_tag}")
                sw_lower = software_tag.lower()

                for sw_name, sw_data in EDITING_SOFTWARE.items():
                    if sw_name in sw_lower:
                        software_info = {
                            "name": sw_name,
                            "category": sw_data["category"],
                            "ai_capable": sw_data["ai_capable"],
                            "full_tag": software_tag,
                        }
                        break

            # Check raw bytes for software signatures
            if raw_bytes:
                bytes_lower = raw_bytes[:10000].lower()  # Check first 10KB

                # Look for common signatures
                signatures = [
                    (b"photoshop", "Adobe Photoshop"),
                    (b"lightroom", "Adobe Lightroom"),
                    (b"gimp", "GIMP"),
                    (b"paint.net", "Paint.NET"),
                    (b"snapseed", "Snapseed"),
                    (b"midjourney", "Midjourney"),
                    (b"dall-e", "DALL-E"),
                    (b"stable diffusion", "Stable Diffusion"),
                ]

                for sig, name in signatures:
                    if sig in bytes_lower:
                        if name not in [t.split(":")[0] for t in traces_found]:
                            traces_found.append(f"Binary signature: {name}")

            # Detection
            detected = len(traces_found) > 0

            if software_info.get("category") == "ai_generator":
                score = 1.0
                confidence = Confidence.VERY_HIGH
                explanation = f"AI generator detected: {software_info['name']}. Image is likely AI-generated."
            elif software_info.get("category") in ["professional_editor", "mobile_editor"]:
                score = 0.7
                confidence = Confidence.HIGH
                explanation = f"Editing software detected: {software_info.get('name', traces_found[0])}. Image has been processed."
            elif traces_found:
                score = 0.5
                confidence = Confidence.MEDIUM
                explanation = f"Software traces found: {', '.join(traces_found)}"
            else:
                score = 0.0
                confidence = Confidence.HIGH
                explanation = "No editing software traces detected in metadata"

            limitations = [
                "Software traces can be removed or spoofed",
                "Camera firmware may include its name in software field",
                "Absence of traces doesn't guarantee authenticity",
            ]

            processing_time = (time.perf_counter() - start_time) * 1000

            return self._create_result(
                score=score,
                confidence=confidence,
                detected=detected,
                explanation=explanation,
                limitations=limitations,
                details={
                    "traces_found": traces_found,
                    "software_info": software_info,
                    "software_tag": software_tag,
                },
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Software trace detection failed: {e}")
            return self._create_error_result(str(e))


# Import numpy for modules that need it
import numpy as np
