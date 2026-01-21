"""
ImageTrust Core Module
======================
Contains fundamental configurations, data types, and exception handling.
"""

from imagetrust.core.config import get_settings, Settings
from imagetrust.core.exceptions import (
    ImageTrustError,
    InvalidImageError,
    ModelLoadingError,
    ConfigurationError,
    AnalysisError,
)
from imagetrust.core.types import (
    AnalysisResult,
    DetectionScore,
    DetectionVerdict,
    Confidence,
    MetadataAnalysis,
    ProvenanceAnalysis,
    ProvenanceStatus,
    ExplainabilityAnalysis,
    ImageInfo,
    EXIFData,
    XMPData,
    C2PAManifest,
    C2PAStatus,
    PatchScore,
)

__all__ = [
    "get_settings",
    "Settings",
    "ImageTrustError",
    "InvalidImageError",
    "ModelLoadingError",
    "ConfigurationError",
    "AnalysisError",
    "AnalysisResult",
    "DetectionScore",
    "DetectionVerdict",
    "Confidence",
    "MetadataAnalysis",
    "ProvenanceAnalysis",
    "ProvenanceStatus",
    "ExplainabilityAnalysis",
    "ImageInfo",
    "EXIFData",
    "XMPData",
    "C2PAManifest",
    "C2PAStatus",
    "PatchScore",
]
