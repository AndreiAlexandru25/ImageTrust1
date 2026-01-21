"""
Custom data types and Pydantic models for ImageTrust.

This module defines the structure of data used across the application,
ensuring consistency and validation.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, computed_field


# =============================================================================
# Enums
# =============================================================================

class DetectionVerdict(str, Enum):
    """Possible verdicts for image authenticity."""
    REAL = "real"
    AI_GENERATED = "ai_generated"
    MANIPULATED = "manipulated"
    UNCERTAIN = "uncertain"


class Confidence(str, Enum):
    """Confidence levels for detection results."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

    @classmethod
    def from_probability(cls, prob: float) -> "Confidence":
        """Determine confidence from probability distance from 0.5."""
        distance = abs(prob - 0.5)
        if distance >= 0.45:  # prob <= 0.05 or prob >= 0.95
            return cls.VERY_HIGH
        elif distance >= 0.35:  # prob <= 0.15 or prob >= 0.85
            return cls.HIGH
        elif distance >= 0.25:  # prob <= 0.25 or prob >= 0.75
            return cls.MEDIUM
        elif distance >= 0.15:  # prob <= 0.35 or prob >= 0.65
            return cls.LOW
        else:
            return cls.VERY_LOW


class ProvenanceStatus(str, Enum):
    """Status of provenance validation."""
    VERIFIED = "verified"
    PARTIAL = "partial"
    MISSING = "missing"
    TAMPERED = "tampered"
    UNKNOWN = "unknown"


class C2PAStatus(str, Enum):
    """Status of C2PA validation."""
    NOT_PRESENT = "not_present"
    PRESENT_VALID = "present_valid"
    PRESENT_INVALID = "present_invalid"
    UNKNOWN = "unknown"


# =============================================================================
# Core Data Models
# =============================================================================

class ImageInfo(BaseModel):
    """Basic information about an image file."""
    file_name: Optional[str] = None
    file_path: Optional[Path] = None
    file_size: Optional[int] = None  # bytes
    file_hash: Optional[str] = None  # SHA-256
    width: int = 0
    height: int = 0
    format: Optional[str] = None  # JPEG, PNG, etc.
    mode: Optional[str] = None  # RGB, RGBA, L, etc.
    has_alpha: bool = False

    @computed_field
    @property
    def megapixels(self) -> float:
        """Calculate megapixels."""
        return (self.width * self.height) / 1_000_000

    @computed_field
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        if self.height == 0:
            return 0
        return self.width / self.height


class DetectionScore(BaseModel):
    """Result from a single AI detection model."""
    detector_name: str
    detector_version: str = "1.0"
    ai_probability: float = Field(..., ge=0.0, le=1.0)
    real_probability: float = Field(..., ge=0.0, le=1.0)
    raw_score: Optional[float] = None  # Raw logit before softmax
    calibrated: bool = False
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    def get_verdict(self) -> DetectionVerdict:
        """Determine verdict from probabilities."""
        if self.ai_probability > 0.6:
            return DetectionVerdict.AI_GENERATED
        elif self.ai_probability < 0.4:
            return DetectionVerdict.REAL
        else:
            return DetectionVerdict.UNCERTAIN

    def get_confidence(self) -> Confidence:
        """Determine confidence level."""
        return Confidence.from_probability(self.ai_probability)


# =============================================================================
# Metadata Models
# =============================================================================

class EXIFData(BaseModel):
    """Extracted EXIF metadata."""
    make: Optional[str] = None
    model: Optional[str] = None
    software: Optional[str] = None
    datetime_original: Optional[datetime] = None
    datetime_digitized: Optional[datetime] = None
    exposure_time: Optional[str] = None
    f_number: Optional[float] = None
    iso: Optional[int] = None
    focal_length: Optional[float] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    raw_data: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_camera_info(self) -> bool:
        """Check if camera information is present."""
        return bool(self.make or self.model)


class XMPData(BaseModel):
    """Extracted XMP metadata."""
    creator: Optional[str] = None
    creator_tool: Optional[str] = None
    create_date: Optional[datetime] = None
    modify_date: Optional[datetime] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    raw_data: Dict[str, Any] = Field(default_factory=dict)


class C2PAManifest(BaseModel):
    """C2PA manifest data."""
    is_present: bool = False
    is_valid: bool = False
    claim_generator: Optional[str] = None
    creation_time: Optional[datetime] = None
    assertions: List[Dict[str, Any]] = Field(default_factory=list)
    ingredients: List[Dict[str, Any]] = Field(default_factory=list)
    signature_info: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = Field(default_factory=list)


class MetadataAnalysis(ImageInfo):
    """Comprehensive metadata analysis result."""
    has_metadata: bool = False
    exif: Optional[EXIFData] = None
    xmp: Optional[XMPData] = None
    ai_indicators: List[str] = Field(default_factory=list)
    anomalies: List[str] = Field(default_factory=list)


class ProvenanceAnalysis(BaseModel):
    """Analysis of image provenance."""
    status: ProvenanceStatus = ProvenanceStatus.UNKNOWN
    c2pa: Optional[C2PAManifest] = None
    trust_indicators: List[str] = Field(default_factory=list)
    warning_indicators: List[str] = Field(default_factory=list)
    claimed_source: Optional[str] = None
    claimed_creator: Optional[str] = None
    creation_date: Optional[datetime] = None
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)


# =============================================================================
# Explainability Models
# =============================================================================

class PatchScore(BaseModel):
    """Score for a specific image patch."""
    x: int
    y: int
    width: int
    height: int
    score: float = Field(..., ge=0.0, le=1.0)

    @property
    def center(self) -> Tuple[int, int]:
        """Get patch center coordinates."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class ExplainabilityAnalysis(BaseModel):
    """Results from explainability methods."""
    gradcam_heatmap: Optional[str] = None  # Base64 encoded
    gradcam_overlay: Optional[str] = None  # Base64 encoded
    attention_map: Optional[str] = None  # Base64 encoded (for ViT)
    patch_scores: List[PatchScore] = Field(default_factory=list)
    top_regions: List[Dict[str, Any]] = Field(default_factory=list)
    frequency_analysis: Optional[Dict[str, Any]] = None
    explanation_text: Optional[str] = None


# =============================================================================
# Overall Analysis Result
# =============================================================================

class AnalysisResult(BaseModel):
    """Comprehensive result of an image analysis."""
    # Identification
    analysis_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Image info
    image_hash: Optional[str] = None
    image_dimensions: Tuple[int, int] = (0, 0)
    
    # Detection
    ai_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    detection_scores: List[DetectionScore] = Field(default_factory=list)
    
    # Analysis components
    metadata: Optional[MetadataAnalysis] = None
    provenance: Optional[ProvenanceAnalysis] = None
    explainability: Optional[ExplainabilityAnalysis] = None
    
    # Processing info
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    models_used: List[str] = Field(default_factory=list)
    ensemble_method: Optional[str] = None
    
    # Notes
    warnings: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @computed_field
    @property
    def verdict(self) -> DetectionVerdict:
        """Overall verdict based on AI probability."""
        if self.ai_probability >= 0.7:
            return DetectionVerdict.AI_GENERATED
        elif self.ai_probability <= 0.3:
            return DetectionVerdict.REAL
        else:
            return DetectionVerdict.UNCERTAIN

    @computed_field
    @property
    def confidence(self) -> Confidence:
        """Overall confidence level."""
        return Confidence.from_probability(self.ai_probability)

    def get_summary(self) -> str:
        """Generate a concise summary of the analysis."""
        parts = [
            f"Verdict: {self.verdict.value.replace('_', ' ').title()}",
            f"AI Probability: {self.ai_probability:.1%}",
            f"Confidence: {self.confidence.value.replace('_', ' ').title()}",
        ]
        
        if self.metadata:
            parts.append(f"Metadata: {'Present' if self.metadata.has_metadata else 'Missing'}")
        
        if self.provenance:
            parts.append(f"Provenance: {self.provenance.status.value.title()}")
        
        if self.warnings:
            parts.append(f"Warnings: {len(self.warnings)}")
        
        return " | ".join(parts)

    def to_report_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for report generation."""
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "verdict": self.verdict.value,
            "ai_probability": self.ai_probability,
            "confidence": self.confidence.value,
            "image_hash": self.image_hash,
            "image_dimensions": self.image_dimensions,
            "models_used": self.models_used,
            "processing_time_ms": self.processing_time_ms,
            "summary": self.get_summary(),
        }
