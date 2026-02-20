"""
ImageTrust Explainability Module
================================
Provides tools for explaining AI detection decisions.
"""

from imagetrust.explainability.gradcam import GradCAMAnalyzer
from imagetrust.explainability.patch_analysis import PatchAnalyzer
from imagetrust.explainability.frequency import FrequencyAnalyzer
from imagetrust.explainability.visualizations import ExplainabilityVisualizer

__all__ = [
    "GradCAMAnalyzer",
    "PatchAnalyzer",
    "FrequencyAnalyzer",
    "ExplainabilityVisualizer",
]
