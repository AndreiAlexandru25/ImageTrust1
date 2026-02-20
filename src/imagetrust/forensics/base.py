"""
Base classes for Forensics Plugin Architecture.

Each detector is a plugin with standardized interface:
- id, name, category
- analyze() -> ForensicsResult
- generate_artifacts() -> visual outputs
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


class PluginCategory(Enum):
    """Categories of forensics plugins."""
    PIXEL = "pixel_forensics"
    METADATA = "metadata_forensics"
    SOURCE = "source_platform"
    AI_DETECTION = "ai_detection"
    TAMPERING = "tampering_detection"


class Confidence(Enum):
    """Confidence levels with numeric values."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95

    @classmethod
    def from_score(cls, score: float) -> "Confidence":
        """Convert numeric score to confidence level."""
        if score >= 0.9:
            return cls.VERY_HIGH
        elif score >= 0.7:
            return cls.HIGH
        elif score >= 0.5:
            return cls.MEDIUM
        elif score >= 0.3:
            return cls.LOW
        else:
            return cls.VERY_LOW


@dataclass
class Artifact:
    """Visual or data artifact from analysis."""
    name: str
    artifact_type: str  # "heatmap", "plot", "image", "data"
    data: Any  # numpy array, PIL Image, dict, etc.
    description: str = ""

    def save(self, output_dir: Path, prefix: str = "") -> Path:
        """Save artifact to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{prefix}_{self.name}" if prefix else self.name

        if self.artifact_type in ["heatmap", "image"]:
            path = output_dir / f"{filename}.png"
            if isinstance(self.data, np.ndarray):
                Image.fromarray(self.data).save(path)
            elif isinstance(self.data, Image.Image):
                self.data.save(path)
            return path
        elif self.artifact_type == "data":
            import json
            path = output_dir / f"{filename}.json"
            with open(path, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
            return path

        return output_dir / filename


@dataclass
class ForensicsResult:
    """
    Standardized result from any forensics plugin.

    Designed for truthfulness:
    - score: 0-1 probability of the detected condition
    - confidence: how certain the detector is
    - explanation: human-readable "because..." string
    - limitations: when/why this might be wrong
    """
    # Plugin identification
    plugin_id: str
    plugin_name: str
    category: PluginCategory

    # Core results
    score: float  # 0-1, meaning depends on plugin
    confidence: Confidence
    detected: bool  # Was the condition detected?

    # Truthfulness fields
    explanation: str  # "because ..."
    limitations: List[str] = field(default_factory=list)

    # Detailed data
    details: Dict[str, Any] = field(default_factory=dict)

    # Visual artifacts (heatmaps, etc.)
    artifacts: List[Artifact] = field(default_factory=list)

    # Processing info
    processing_time_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "plugin_id": self.plugin_id,
            "plugin_name": self.plugin_name,
            "category": self.category.value,
            "score": self.score,
            "confidence": self.confidence.name,
            "confidence_value": self.confidence.value,
            "detected": self.detected,
            "explanation": self.explanation,
            "limitations": self.limitations,
            "details": self.details,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
            "artifacts": [
                {"name": a.name, "type": a.artifact_type, "description": a.description}
                for a in self.artifacts
            ],
        }

    @property
    def summary(self) -> str:
        """One-line summary for reports."""
        status = "DETECTED" if self.detected else "not detected"
        return f"{self.plugin_name}: {status} (score={self.score:.2f}, {self.confidence.name})"


class ForensicsPlugin(ABC):
    """
    Abstract base class for all forensics plugins.

    Each plugin must implement:
    - analyze(): core detection logic
    - Provide metadata (id, name, category, description)

    Design principles:
    1. CPU-first (GPU optional)
    2. Deterministic (seed support)
    3. Truthful outputs (never claim certainty without evidence)
    """

    # Plugin metadata (override in subclasses)
    plugin_id: str = "base_plugin"
    plugin_name: str = "Base Plugin"
    category: PluginCategory = PluginCategory.PIXEL
    description: str = "Base forensics plugin"
    version: str = "1.0.0"

    # What this plugin needs
    requires_jpeg: bool = False  # Some plugins only work on JPEG
    requires_exif: bool = False
    min_image_size: Tuple[int, int] = (64, 64)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin with optional configuration.

        Args:
            config: Plugin-specific configuration dict
        """
        self.config = config or {}
        self._initialized = True

    @abstractmethod
    def analyze(
        self,
        image: Image.Image,
        image_path: Optional[Path] = None,
        raw_bytes: Optional[bytes] = None,
    ) -> ForensicsResult:
        """
        Analyze an image and return forensics result.

        Args:
            image: PIL Image object
            image_path: Optional path (for file-based analysis like JPEG structure)
            raw_bytes: Optional raw file bytes

        Returns:
            ForensicsResult with score, confidence, explanation, etc.
        """
        pass

    def can_analyze(self, image: Image.Image, image_path: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Check if this plugin can analyze the given image.

        Returns:
            (can_analyze, reason)
        """
        # Check minimum size
        if image.size[0] < self.min_image_size[0] or image.size[1] < self.min_image_size[1]:
            return False, f"Image too small (min: {self.min_image_size})"

        # Check JPEG requirement
        if self.requires_jpeg:
            if image_path is None:
                return False, "Requires JPEG file path"
            if not str(image_path).lower().endswith(('.jpg', '.jpeg')):
                return False, "Requires JPEG format"

        return True, "OK"

    def _create_result(
        self,
        score: float,
        confidence: Confidence,
        detected: bool,
        explanation: str,
        limitations: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[Artifact]] = None,
        processing_time_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> ForensicsResult:
        """Helper to create standardized result."""
        return ForensicsResult(
            plugin_id=self.plugin_id,
            plugin_name=self.plugin_name,
            category=self.category,
            score=score,
            confidence=confidence,
            detected=detected,
            explanation=explanation,
            limitations=limitations or [],
            details=details or {},
            artifacts=artifacts or [],
            processing_time_ms=processing_time_ms,
            error=error,
        )

    def _create_error_result(self, error_msg: str) -> ForensicsResult:
        """Create result for error cases."""
        return self._create_result(
            score=0.0,
            confidence=Confidence.VERY_LOW,
            detected=False,
            explanation=f"Analysis failed: {error_msg}",
            limitations=["Analysis could not be completed"],
            error=error_msg,
        )


# Plugin registry
_PLUGIN_REGISTRY: Dict[str, type] = {}


def register_plugin(plugin_class: type) -> type:
    """Decorator to register a forensics plugin."""
    if not issubclass(plugin_class, ForensicsPlugin):
        raise TypeError(f"{plugin_class} must be a ForensicsPlugin subclass")

    _PLUGIN_REGISTRY[plugin_class.plugin_id] = plugin_class
    return plugin_class


def get_plugin(plugin_id: str) -> Optional[type]:
    """Get a plugin class by ID."""
    return _PLUGIN_REGISTRY.get(plugin_id)


def list_plugins() -> List[str]:
    """List all registered plugin IDs."""
    return list(_PLUGIN_REGISTRY.keys())


def get_plugins_by_category(category: PluginCategory) -> List[type]:
    """Get all plugins in a category."""
    return [
        cls for cls in _PLUGIN_REGISTRY.values()
        if cls.category == category
    ]
