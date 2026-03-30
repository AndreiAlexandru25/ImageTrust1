"""
Detection Models Module
=======================
Contains various model architectures for AI detection.
"""

from imagetrust.detection.models.base import BaseDetector
from imagetrust.detection.models.cnn_detector import CNNDetector
from imagetrust.detection.models.vit_detector import ViTDetector
from imagetrust.detection.models.ensemble import EnsembleDetector
from imagetrust.detection.models.kaggle_detector import KaggleDeepfakeDetector

__all__ = [
    "BaseDetector",
    "CNNDetector",
    "ViTDetector",
    "EnsembleDetector",
    "KaggleDeepfakeDetector",
]
