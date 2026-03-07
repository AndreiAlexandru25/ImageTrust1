"""
Provenance analysis module.

Combines metadata analysis to assess image authenticity.
"""

import hashlib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from imagetrust.core.types import (
    MetadataAnalysis,
    ProvenanceAnalysis,
    ProvenanceStatus,
    C2PAStatus,
)
from imagetrust.metadata.exif_parser import EXIFParser
from imagetrust.metadata.xmp_parser import XMPParser
from imagetrust.metadata.c2pa_validator import C2PAValidator
from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import calculate_file_hash

logger = get_logger(__name__)


class ProvenanceAnalyzer:
    """
    Analyzes image provenance by combining metadata sources.
    
    Examines EXIF, XMP, and C2PA to assess the authenticity
    and origin of an image.
    
    Example:
        >>> analyzer = ProvenanceAnalyzer()
        >>> metadata, provenance = analyzer.analyze("photo.jpg")
        >>> print(provenance.status)
    """

    def __init__(self) -> None:
        self.exif_parser = EXIFParser()
        self.xmp_parser = XMPParser()
        self.c2pa_validator = C2PAValidator()

    def analyze(
        self,
        source: Union[Path, str, bytes],
        filename: Optional[str] = None,
    ) -> Tuple[MetadataAnalysis, ProvenanceAnalysis]:
        """
        Analyze image provenance.
        
        Args:
            source: Image path or bytes
            filename: Optional filename if source is bytes
            
        Returns:
            Tuple of (MetadataAnalysis, ProvenanceAnalysis)
        """
        # Initialize results
        metadata = MetadataAnalysis()
        provenance = ProvenanceAnalysis()
        
        # Load image and get basic info
        try:
            if isinstance(source, bytes):
                img = Image.open(BytesIO(source))
                image_data = source
                metadata.file_name = filename
                metadata.file_size = len(source)
                metadata.file_hash = calculate_file_hash(data=source)
            else:
                source = Path(source)
                img = Image.open(source)
                image_data = source.read_bytes()
                metadata.file_name = source.name
                metadata.file_path = source
                metadata.file_size = source.stat().st_size
                metadata.file_hash = calculate_file_hash(file_path=source)
            
            # Image properties
            metadata.width = img.width
            metadata.height = img.height
            metadata.format = img.format
            metadata.mode = img.mode
            metadata.has_alpha = img.mode in ("RGBA", "LA", "PA")
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            provenance.status = ProvenanceStatus.UNKNOWN
            return metadata, provenance
        
        # Parse EXIF
        metadata.exif = self.exif_parser.parse(img)
        exif_indicators = self.exif_parser.detect_ai_indicators(metadata.exif)
        
        # Parse XMP
        metadata.xmp = self.xmp_parser.parse(image_data)
        xmp_indicators = self.xmp_parser.detect_ai_indicators(metadata.xmp)
        
        # Validate C2PA
        provenance.c2pa = self.c2pa_validator.validate(image_data)
        if hasattr(self.c2pa_validator, 'get_trust_indicators'):
            c2pa_indicators = self.c2pa_validator.get_trust_indicators(provenance.c2pa)
        else:
            c2pa_indicators = []
        
        # Aggregate metadata presence
        metadata.has_metadata = bool(
            metadata.exif.raw_data or
            metadata.xmp.raw_data or
            getattr(provenance.c2pa, 'has_c2pa', False)
        )
        
        # Collect AI indicators
        metadata.ai_indicators = exif_indicators + xmp_indicators
        
        # Determine provenance status
        provenance = self._assess_provenance(metadata, provenance, c2pa_indicators)
        
        return metadata, provenance

    def _assess_provenance(
        self,
        metadata: MetadataAnalysis,
        provenance: ProvenanceAnalysis,
        c2pa_indicators: List[str],
    ) -> ProvenanceAnalysis:
        """Assess overall provenance status."""
        
        # Start with confidence at 0.5 (uncertain)
        confidence = 0.5
        
        # C2PA is the strongest indicator
        c2pa = provenance.c2pa
        c2pa_valid = getattr(c2pa, 'has_c2pa', False) and getattr(c2pa, 'trust_score', 0) > 50
        c2pa_present = getattr(c2pa, 'has_c2pa', False)

        if c2pa_valid:
            provenance.status = ProvenanceStatus.VERIFIED
            provenance.trust_indicators.extend(c2pa_indicators)
            provenance.claimed_creator = getattr(c2pa, 'ai_generator', None)
            creation = getattr(c2pa, 'creation_info', {})
            provenance.creation_date = creation.get('date') if creation else None
            confidence += 0.4

        elif c2pa_present:
            provenance.status = ProvenanceStatus.TAMPERED
            provenance.warning_indicators.append("Invalid C2PA signature")
            confidence -= 0.3

        else:
            # No C2PA, rely on other metadata
            if metadata.has_metadata:
                if metadata.exif.has_camera_info:
                    provenance.status = ProvenanceStatus.PARTIAL
                    provenance.trust_indicators.append(
                        f"Camera: {metadata.exif.make} {metadata.exif.model}"
                    )
                    confidence += 0.1
                else:
                    provenance.status = ProvenanceStatus.PARTIAL
            else:
                provenance.status = ProvenanceStatus.MISSING
                provenance.warning_indicators.append("No metadata found")
                confidence -= 0.1
        
        # Check for AI indicators
        if metadata.ai_indicators:
            for indicator in metadata.ai_indicators:
                provenance.warning_indicators.append(indicator)
            confidence -= 0.2 * len(metadata.ai_indicators)
        
        # Set claimed source from metadata
        if metadata.exif.software:
            provenance.claimed_source = metadata.exif.software
        elif metadata.xmp.creator_tool:
            provenance.claimed_source = metadata.xmp.creator_tool
        
        # Set creation date from metadata
        if provenance.creation_date is None:
            if metadata.exif.datetime_original:
                provenance.creation_date = metadata.exif.datetime_original
            elif metadata.xmp.create_date:
                provenance.creation_date = metadata.xmp.create_date
        
        # Bound confidence
        provenance.confidence_score = max(0.0, min(1.0, confidence))
        
        return provenance

    def get_summary(
        self,
        metadata: MetadataAnalysis,
        provenance: ProvenanceAnalysis,
    ) -> str:
        """Generate a summary of the provenance analysis."""
        parts = [f"Provenance: {provenance.status.value.title()}"]
        
        if provenance.claimed_creator:
            parts.append(f"Creator: {provenance.claimed_creator}")
        
        if provenance.creation_date:
            parts.append(f"Created: {provenance.creation_date.strftime('%Y-%m-%d')}")
        
        if provenance.trust_indicators:
            parts.append(f"Trust: {len(provenance.trust_indicators)} indicators")
        
        if provenance.warning_indicators:
            parts.append(f"Warnings: {len(provenance.warning_indicators)}")
        
        return " | ".join(parts)
