"""
ImageTrust Detection Module
===========================
Contains the core logic for AI-generated image detection.
"""

from imagetrust.detection.detector import AIDetector
from imagetrust.detection.preprocessing import ImagePreprocessor
from imagetrust.detection.calibration import (
    TemperatureScaling,
    PlattScaling,
    IsotonicCalibration,
    CalibrationWrapper,
    ExpectedCalibrationError,
)
from imagetrust.detection.models.base import BaseDetector
from imagetrust.detection.models.cnn_detector import CNNDetector
from imagetrust.detection.models.vit_detector import ViTDetector
from imagetrust.detection.models.ensemble import EnsembleDetector

__all__ = [
    "AIDetector",
    "ImagePreprocessor",
    "TemperatureScaling",
    "PlattScaling",
    "IsotonicCalibration",
    "CalibrationWrapper",
    "ExpectedCalibrationError",
    "BaseDetector",
    "CNNDetector",
    "ViTDetector",
    "EnsembleDetector",
]
