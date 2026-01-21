"""
XMP metadata parser.

Extracts XMP data from image files.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from imagetrust.core.types import XMPData
from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


class XMPParser:
    """
    Parses XMP metadata from images.
    
    Example:
        >>> parser = XMPParser()
        >>> xmp = parser.parse("photo.jpg")
        >>> print(xmp.creator_tool)
    """

    def parse(
        self,
        source: Union[Path, str, bytes, Image.Image],
    ) -> XMPData:
        """
        Parse XMP data from an image.
        
        Args:
            source: Image path, bytes, or PIL Image
            
        Returns:
            XMPData object with extracted metadata
        """
        try:
            # Get raw bytes
            if isinstance(source, Image.Image):
                buffer = BytesIO()
                source.save(buffer, format=source.format or "PNG")
                data = buffer.getvalue()
            elif isinstance(source, bytes):
                data = source
            else:
                with open(source, "rb") as f:
                    data = f.read()
            
            # Extract XMP
            xmp_dict = self._extract_xmp(data)
            
            return self._map_to_model(xmp_dict)
            
        except Exception as e:
            logger.warning(f"Failed to parse XMP: {e}")
            return XMPData()

    def _extract_xmp(self, data: bytes) -> Dict[str, Any]:
        """Extract XMP data from raw bytes."""
        xmp_dict = {}
        
        # Look for XMP packet markers
        xmp_start = data.find(b"<x:xmpmeta")
        if xmp_start == -1:
            xmp_start = data.find(b"<?xpacket begin")
        
        xmp_end = data.find(b"</x:xmpmeta>")
        if xmp_end == -1:
            xmp_end = data.find(b"<?xpacket end")
        
        if xmp_start != -1 and xmp_end != -1:
            xmp_bytes = data[xmp_start:xmp_end + 12]
            xmp_dict = self._parse_xmp_xml(xmp_bytes.decode("utf-8", errors="ignore"))
        
        return xmp_dict

    def _parse_xmp_xml(self, xmp_str: str) -> Dict[str, Any]:
        """Parse XMP XML string."""
        result = {}
        
        try:
            import xml.etree.ElementTree as ET
            
            # Clean up the string
            xmp_str = xmp_str.strip()
            if not xmp_str.startswith("<"):
                xmp_str = "<" + xmp_str.split("<", 1)[1]
            
            root = ET.fromstring(xmp_str)
            
            # Common XMP namespaces
            namespaces = {
                "x": "adobe:ns:meta/",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "xmp": "http://ns.adobe.com/xap/1.0/",
                "dc": "http://purl.org/dc/elements/1.1/",
                "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
                "stEvt": "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#",
            }
            
            # Extract common fields
            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                text = elem.text.strip() if elem.text else None
                
                if text:
                    result[tag] = text
            
        except ET.ParseError as e:
            logger.debug(f"XMP XML parse error: {e}")
        except Exception as e:
            logger.debug(f"XMP extraction error: {e}")
        
        return result

    def _map_to_model(self, xmp_dict: Dict[str, Any]) -> XMPData:
        """Map raw XMP dict to XMPData model."""
        xmp = XMPData(raw_data=xmp_dict)
        
        # Creator
        xmp.creator = xmp_dict.get("creator") or xmp_dict.get("Creator")
        xmp.creator_tool = xmp_dict.get("CreatorTool")
        
        # Dates
        for key in ["CreateDate", "CreateDate", "DateCreated"]:
            if key in xmp_dict:
                try:
                    xmp.create_date = self._parse_date(xmp_dict[key])
                    break
                except ValueError:
                    pass
        
        for key in ["ModifyDate", "ModifyDate", "DateModified"]:
            if key in xmp_dict:
                try:
                    xmp.modify_date = self._parse_date(xmp_dict[key])
                    break
                except ValueError:
                    pass
        
        return xmp

    def _parse_date(self, date_str: str) -> datetime:
        """Parse XMP date string."""
        # Try common formats
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
            "%Y:%m:%d %H:%M:%S",
        ]
        
        # Remove timezone suffix variations
        date_str = date_str.replace("Z", "").split("+")[0].split("-0")[0]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str[:len(fmt.replace("%", ""))], fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Could not parse date: {date_str}")

    def detect_ai_indicators(self, xmp: XMPData) -> List[str]:
        """
        Detect indicators that suggest AI generation.
        
        Args:
            xmp: Parsed XMP data
            
        Returns:
            List of detected AI indicators
        """
        indicators = []
        
        # Check creator tool
        if xmp.creator_tool:
            tool_lower = xmp.creator_tool.lower()
            ai_tools = [
                "midjourney", "dall-e", "dalle", "stable diffusion",
                "leonardo", "firefly", "ideogram", "flux",
            ]
            for ai_tool in ai_tools:
                if ai_tool in tool_lower:
                    indicators.append(f"AI creator tool: {xmp.creator_tool}")
                    break
        
        return indicators
