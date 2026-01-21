"""
ImageTrust: A Forensic Application for Identifying AI-Generated and Digitally Manipulated Images.

This package provides tools for:
- Detecting AI-generated and digitally manipulated images.
- Analyzing image metadata and provenance (C2PA).
- Providing explainability for detection results.
- Generating comprehensive forensic reports.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
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
    ExplainabilityAnalysis,
    ImageInfo,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "get_settings",
    "Settings",
    # Exceptions
    "ImageTrustError",
    "InvalidImageError",
    "ModelLoadingError",
    "ConfigurationError",
    "AnalysisError",
    # Types
    "AnalysisResult",
    "DetectionScore",
    "DetectionVerdict",
    "Confidence",
    "MetadataAnalysis",
    "ProvenanceAnalysis",
    "ExplainabilityAnalysis",
    "ImageInfo",
]
