"""
EXIF metadata parser.

Extracts EXIF data from image files.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from imagetrust.core.types import EXIFData
from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)

# AI-related software indicators
AI_SOFTWARE_INDICATORS = [
    "midjourney", "dall-e", "dalle", "stable diffusion", "sd",
    "leonardo", "firefly", "ideogram", "flux", "playground",
    "nightcafe", "artbreeder", "deepai", "jasper", "wombo",
    "starryai", "craiyon", "runway", "pika", "gen-1", "gen-2",
]


class EXIFParser:
    """
    Parses EXIF metadata from images.
    
    Example:
        >>> parser = EXIFParser()
        >>> exif = parser.parse("photo.jpg")
        >>> print(exif.make, exif.model)
    """

    def parse(
        self,
        source: Union[Path, str, bytes, Image.Image],
    ) -> EXIFData:
        """
        Parse EXIF data from an image.
        
        Args:
            source: Image path, bytes, or PIL Image
            
        Returns:
            EXIFData object with extracted metadata
        """
        try:
            # Open image
            if isinstance(source, Image.Image):
                img = source
            elif isinstance(source, bytes):
                img = Image.open(BytesIO(source))
            else:
                img = Image.open(source)
            
            # Get EXIF data
            exif_dict = self._extract_exif(img)
            
            return self._map_to_model(exif_dict)
            
        except Exception as e:
            logger.warning(f"Failed to parse EXIF: {e}")
            return EXIFData()

    def _extract_exif(self, img: Image.Image) -> Dict[str, Any]:
        """Extract raw EXIF dictionary from PIL Image."""
        exif_dict = {}
        
        # Try to get EXIF info
        exif = img.getexif() if hasattr(img, "getexif") else {}
        
        if exif:
            # Map EXIF tag IDs to names
            from PIL.ExifTags import TAGS, GPSTAGS
            
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_dict[tag_name] = value
            
            # Handle GPS info specially
            if "GPSInfo" in exif_dict:
                gps_info = {}
                for key, val in exif_dict["GPSInfo"].items():
                    gps_name = GPSTAGS.get(key, key)
                    gps_info[gps_name] = val
                exif_dict["GPSInfo"] = gps_info
        
        return exif_dict

    def _map_to_model(self, exif_dict: Dict[str, Any]) -> EXIFData:
        """Map raw EXIF dict to EXIFData model."""
        exif = EXIFData(raw_data=exif_dict)
        
        # Basic info
        exif.make = str(exif_dict.get("Make", "")) or None
        exif.model = str(exif_dict.get("Model", "")) or None
        exif.software = str(exif_dict.get("Software", "")) or None
        
        # Dates
        for date_field in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
            if date_field in exif_dict:
                try:
                    date_str = str(exif_dict[date_field])
                    parsed = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                    if date_field == "DateTimeOriginal":
                        exif.datetime_original = parsed
                    elif date_field == "DateTimeDigitized":
                        exif.datetime_digitized = parsed
                except ValueError:
                    pass
        
        # Camera settings
        if "ExposureTime" in exif_dict:
            exif.exposure_time = str(exif_dict["ExposureTime"])
        if "FNumber" in exif_dict:
            try:
                exif.f_number = float(exif_dict["FNumber"])
            except (ValueError, TypeError):
                pass
        if "ISOSpeedRatings" in exif_dict:
            try:
                exif.iso = int(exif_dict["ISOSpeedRatings"])
            except (ValueError, TypeError):
                pass
        if "FocalLength" in exif_dict:
            try:
                exif.focal_length = float(exif_dict["FocalLength"])
            except (ValueError, TypeError):
                pass
        
        # GPS
        if "GPSInfo" in exif_dict:
            gps = exif_dict["GPSInfo"]
            exif.gps_latitude = self._parse_gps_coord(
                gps.get("GPSLatitude"), gps.get("GPSLatitudeRef")
            )
            exif.gps_longitude = self._parse_gps_coord(
                gps.get("GPSLongitude"), gps.get("GPSLongitudeRef")
            )
        
        return exif

    def _parse_gps_coord(
        self,
        coord: Any,
        ref: Any,
    ) -> Optional[float]:
        """Parse GPS coordinates to decimal degrees."""
        if coord is None:
            return None
        
        try:
            # coord is typically a tuple of (degrees, minutes, seconds)
            if isinstance(coord, (list, tuple)) and len(coord) >= 3:
                d = float(coord[0])
                m = float(coord[1])
                s = float(coord[2])
                decimal = d + m / 60 + s / 3600
                
                if ref in ["S", "W"]:
                    decimal = -decimal
                
                return decimal
        except (ValueError, TypeError, IndexError):
            pass
        
        return None

    def detect_ai_indicators(self, exif: EXIFData) -> List[str]:
        """
        Detect indicators that suggest AI generation.
        
        Args:
            exif: Parsed EXIF data
            
        Returns:
            List of detected AI indicators
        """
        indicators = []
        
        # Check software field
        if exif.software:
            software_lower = exif.software.lower()
            for ai_tool in AI_SOFTWARE_INDICATORS:
                if ai_tool in software_lower:
                    indicators.append(f"AI software detected: {exif.software}")
                    break
        
        # Missing camera info with other data present
        if exif.raw_data and not exif.has_camera_info:
            indicators.append("EXIF present but no camera information")
        
        # Suspicious software
        if exif.software:
            suspicious = ["ai", "generative", "synthetic"]
            for term in suspicious:
                if term in exif.software.lower():
                    indicators.append(f"Suspicious software: {exif.software}")
                    break
        
        return indicators
