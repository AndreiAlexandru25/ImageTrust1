"""
Baseline detectors for comparison in the thesis.

This module provides three baseline implementations + ImageTrust wrapper:
- B1: Classical (LogReg/XGBoost on forensic features)
- B2: CNN (ResNet-50 or EfficientNet-B0)
- B3: Modern (ViT-B/16 or CLIP linear probe)
- ImageTrust: Our method (4 HF models + signal analysis ensemble)

All baselines share a common interface via BaselineDetector.

Additional utilities:
- Calibration: Post-hoc probability calibration
- Uncertainty: Selective prediction (abstain) support
"""

from imagetrust.baselines.base import BaselineDetector, BaselineConfig, BaselineResult
from imagetrust.baselines.classical_baseline import ClassicalBaseline
from imagetrust.baselines.cnn_baseline import CNNBaseline
from imagetrust.baselines.vit_baseline import ViTBaseline
from imagetrust.baselines.imagetrust_wrapper import ImageTrustWrapper
from imagetrust.baselines.registry import (
    get_baseline,
    list_baselines,
    register_baseline,
    BASELINE_REGISTRY,
)
from imagetrust.baselines.calibration import (
    BaselineCalibrator,
    CalibrationResult,
    calibrate_baseline,
    compare_calibration_methods,
)
from imagetrust.baselines.uncertainty import (
    UncertaintyEstimator,
    UncertaintyResult,
    SelectivePredictor,
    SelectivePredictionResult,
    compute_risk_coverage_auc,
)

__all__ = [
    # Base classes
    "BaselineDetector",
    "BaselineConfig",
    "BaselineResult",
    # Baseline implementations
    "ClassicalBaseline",
    "CNNBaseline",
    "ViTBaseline",
    "ImageTrustWrapper",
    # Registry
    "get_baseline",
    "list_baselines",
    "register_baseline",
    "BASELINE_REGISTRY",
    # Calibration
    "BaselineCalibrator",
    "CalibrationResult",
    "calibrate_baseline",
    "compare_calibration_methods",
    # Uncertainty
    "UncertaintyEstimator",
    "UncertaintyResult",
    "SelectivePredictor",
    "SelectivePredictionResult",
    "compute_risk_coverage_auc",
]
