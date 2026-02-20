"""
ImageTrust Forensics Engine.

A modular, plugin-based forensics analysis system for detecting
image manipulations, recompressions, and authenticity issues.

Designed for "truthfulness" - every conclusion has:
- Score (0-1) + confidence level
- Explanation ("because...")
- Limitations (when uncertain)
"""

from imagetrust.forensics.engine import ForensicsEngine
from imagetrust.forensics.base import (
    ForensicsPlugin,
    ForensicsResult,
    PluginCategory,
    Confidence,
)
from imagetrust.forensics.fusion import FusionLayer, ForensicsVerdict

__all__ = [
    "ForensicsEngine",
    "ForensicsPlugin",
    "ForensicsResult",
    "PluginCategory",
    "Confidence",
    "FusionLayer",
    "ForensicsVerdict",
]
