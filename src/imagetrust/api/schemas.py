"""
Pydantic schemas for API requests and responses.
"""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Request Schemas
# =============================================================================

class AnalysisOptions(BaseModel):
    """Options for image analysis."""
    
    include_metadata: bool = Field(
        default=True,
        description="Include metadata extraction"
    )
    include_explainability: bool = Field(
        default=False,
        description="Include Grad-CAM heatmap"
    )
    include_frequency: bool = Field(
        default=False,
        description="Include frequency analysis"
    )
    include_patches: bool = Field(
        default=False,
        description="Include patch analysis"
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection threshold"
    )


# =============================================================================
# Response Schemas
# =============================================================================

class DetectionResult(BaseModel):
    """Detection result."""
    
    ai_probability: float = Field(
        description="Probability of AI generation (0-1)"
    )
    real_probability: float = Field(
        description="Probability of real image (0-1)"
    )
    verdict: str = Field(
        description="Classification verdict (real, ai_generated, uncertain)"
    )
    confidence: str = Field(
        description="Confidence level (low, medium, high)"
    )
    calibrated: bool = Field(
        default=False,
        description="Whether probability is calibrated"
    )


class MetadataResult(BaseModel):
    """Metadata analysis result."""
    
    has_metadata: bool = Field(description="Whether image has metadata")
    file_hash: Optional[str] = Field(default=None, description="SHA-256 hash")
    width: int = Field(description="Image width")
    height: int = Field(description="Image height")
    format: Optional[str] = Field(default=None, description="Image format")
    
    # EXIF
    exif_present: bool = Field(default=False, description="EXIF data present")
    camera: Optional[str] = Field(default=None, description="Camera make/model")
    software: Optional[str] = Field(default=None, description="Software used")
    creation_date: Optional[datetime] = Field(default=None, description="Creation date")
    
    # XMP
    xmp_present: bool = Field(default=False, description="XMP data present")
    
    # C2PA
    c2pa_present: bool = Field(default=False, description="C2PA manifest present")
    c2pa_valid: bool = Field(default=False, description="C2PA validation status")
    
    # Anomalies
    anomalies: List[str] = Field(
        default_factory=list,
        description="Detected metadata anomalies"
    )


class ExplainabilityResult(BaseModel):
    """Explainability analysis result."""
    
    gradcam_heatmap: Optional[str] = Field(
        default=None,
        description="Base64-encoded Grad-CAM heatmap image"
    )
    attention_map: Optional[str] = Field(
        default=None,
        description="Base64-encoded attention map"
    )
    patch_scores: Optional[List[float]] = Field(
        default=None,
        description="Per-patch AI scores"
    )
    top_regions: Optional[List[dict]] = Field(
        default=None,
        description="Top suspicious regions"
    )
    frequency_score: Optional[float] = Field(
        default=None,
        description="Frequency analysis AI score"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Human-readable explanation"
    )


class AnalysisResponse(BaseModel):
    """Complete analysis response."""
    
    analysis_id: str = Field(description="Unique analysis ID")
    timestamp: datetime = Field(description="Analysis timestamp")
    filename: Optional[str] = Field(default=None, description="Original filename")
    file_size: int = Field(description="File size in bytes")
    
    # Detection
    detection: DetectionResult = Field(description="Detection results")
    
    # Optional analyses
    metadata: Optional[MetadataResult] = Field(
        default=None,
        description="Metadata analysis"
    )
    explainability: Optional[ExplainabilityResult] = Field(
        default=None,
        description="Explainability results"
    )
    
    # Processing info
    processing_time_ms: float = Field(description="Processing time in ms")
    model_used: str = Field(description="Detection model used")
    warnings: List[str] = Field(
        default_factory=list,
        description="Processing warnings"
    )


class BatchAnalysisResponse(BaseModel):
    """Batch analysis response."""
    
    batch_id: str = Field(description="Batch ID")
    total_images: int = Field(description="Total images submitted")
    successful: int = Field(description="Successfully processed")
    failed: int = Field(description="Failed to process")
    results: List[AnalysisResponse] = Field(description="Individual results")
    errors: List[dict] = Field(
        default_factory=list,
        description="Processing errors"
    )
    total_processing_time_ms: float = Field(description="Total processing time")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    timestamp: datetime = Field(description="Current timestamp")
    model_loaded: bool = Field(description="Detection model loaded")
    cuda_available: Optional[bool] = Field(
        default=None,
        description="CUDA GPU available"
    )


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    device: str = Field(description="Compute device")
    input_size: int = Field(description="Expected input size")
    total_parameters: int = Field(description="Total parameters")
    calibrator_loaded: bool = Field(description="Calibrator loaded")


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(description="Error code")
    message: str = Field(description="Error message")
    details: dict = Field(default_factory=dict, description="Additional details")


class ReportRequest(BaseModel):
    """Report generation request."""
    
    analysis_id: str = Field(description="Analysis ID to generate report for")
    format: str = Field(
        default="json",
        description="Report format (json, pdf, html)"
    )
    include_images: bool = Field(
        default=True,
        description="Include visualizations in report"
    )


class ReportResponse(BaseModel):
    """Report generation response."""
    
    report_id: str = Field(description="Report ID")
    format: str = Field(description="Report format")
    download_url: Optional[str] = Field(
        default=None,
        description="Download URL for report"
    )
    content: Optional[str] = Field(
        default=None,
        description="Report content (for JSON format)"
    )
