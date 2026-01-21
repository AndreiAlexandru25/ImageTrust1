"""
ImageTrust Evaluation Module
============================
Provides tools for benchmarking and evaluating AI detection models.
"""

from imagetrust.evaluation.metrics import (
    compute_metrics,
    compute_roc_auc,
    compute_calibration_metrics,
)
from imagetrust.evaluation.benchmark import Benchmark
from imagetrust.evaluation.cross_generator import CrossGeneratorEvaluator
from imagetrust.evaluation.degradation import DegradationEvaluator
from imagetrust.evaluation.ablation import AblationStudy

__all__ = [
    "compute_metrics",
    "compute_roc_auc",
    "compute_calibration_metrics",
    "Benchmark",
    "CrossGeneratorEvaluator",
    "DegradationEvaluator",
    "AblationStudy",
]
