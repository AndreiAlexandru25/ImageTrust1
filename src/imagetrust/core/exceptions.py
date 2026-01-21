"""
Custom exception classes for ImageTrust.

Provides structured error handling for various scenarios,
making it easier to catch and respond to specific issues.
"""

from typing import Any, Dict, Optional


class ImageTrustError(Exception):
    """Base exception for all ImageTrust-related errors."""

    def __init__(
        self,
        message: str,
        code: str = "IMAGETRUST_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:
        detail_str = f" (Details: {self.details})" if self.details else ""
        return f"[{self.code}] {self.message}{detail_str}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message='{self.message}', code='{self.code}')"


class InvalidImageError(ImageTrustError):
    """Exception raised for invalid or unreadable image files."""

    def __init__(
        self,
        message: str = "Invalid or unreadable image",
        file_path: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        details = {}
        if file_path:
            details["file_path"] = file_path
        if reason:
            details["reason"] = reason
        super().__init__(message, code="INVALID_IMAGE", details=details)


class ModelLoadingError(ImageTrustError):
    """Exception raised when a detection model fails to load."""

    def __init__(
        self,
        message: str = "Failed to load model",
        model_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        details = {}
        if model_name:
            details["model_name"] = model_name
        if checkpoint_path:
            details["checkpoint_path"] = checkpoint_path
        super().__init__(message, code="MODEL_LOADING_ERROR", details=details)


class ConfigurationError(ImageTrustError):
    """Exception raised for invalid or missing configuration."""

    def __init__(
        self,
        message: str = "Configuration error",
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
    ) -> None:
        details = {}
        if config_key:
            details["config_key"] = config_key
        if expected_type:
            details["expected_type"] = expected_type
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)


class AnalysisError(ImageTrustError):
    """Exception raised when an analysis step fails."""

    def __init__(
        self,
        message: str = "Analysis failed",
        step: Optional[str] = None,
        image_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        details = {}
        if step:
            details["step"] = step
        if image_info:
            details["image_info"] = image_info
        super().__init__(message, code="ANALYSIS_ERROR", details=details)


class CalibrationError(ImageTrustError):
    """Exception raised when probability calibration fails."""

    def __init__(
        self,
        message: str = "Calibration failed",
        method: Optional[str] = None,
    ) -> None:
        details = {"method": method} if method else {}
        super().__init__(message, code="CALIBRATION_ERROR", details=details)


class ReportGenerationError(ImageTrustError):
    """Exception raised when report generation fails."""

    def __init__(
        self,
        message: str = "Report generation failed",
        report_format: Optional[str] = None,
    ) -> None:
        details = {"format": report_format} if report_format else {}
        super().__init__(message, code="REPORT_GENERATION_ERROR", details=details)


class ProvenanceError(ImageTrustError):
    """Exception raised when provenance analysis fails."""

    def __init__(
        self,
        message: str = "Provenance analysis failed",
        component: Optional[str] = None,  # "c2pa", "exif", "xmp"
    ) -> None:
        details = {"component": component} if component else {}
        super().__init__(message, code="PROVENANCE_ERROR", details=details)
