#!/usr/bin/env python3
"""
Threshold Calibration Script - Academic Validation

This script calibrates thresholds for EACH forensic method
and generates documentation for the thesis (tables, plots, justifications).

Professor requirement: "Don't use arbitrary values, demonstrate them experimentally!"

Usage:
    python scripts/calibrate_thresholds.py --dataset data/calibration --output outputs/calibration
    python scripts/calibrate_thresholds.py --method ela --dataset data/casia
    python scripts/calibrate_thresholds.py --all --generate-latex
"""

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class CalibrationResult:
    """Result of threshold calibration for a single method."""

    method_name: str
    threshold_optimal: float
    threshold_method: str  # "youden", "f1_max", "precision_target", etc.
    auc_roc: float
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    n_samples: int
    n_positive: int
    n_negative: int
    dataset_name: str
    calibration_date: str
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    literature_threshold: Optional[float] = None
    literature_source: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "threshold_optimal": round(self.threshold_optimal, 4),
            "threshold_method": self.threshold_method,
            "auc_roc": round(self.auc_roc, 4),
            "f1_score": round(self.f1_score, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "accuracy": round(self.accuracy, 4),
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "dataset_name": self.dataset_name,
            "calibration_date": self.calibration_date,
            "confidence_interval": [round(x, 4) for x in self.confidence_interval],
            "literature_threshold": self.literature_threshold,
            "literature_source": self.literature_source,
            "notes": self.notes,
        }

    def to_latex_row(self) -> str:
        """Generate LaTeX table row for thesis."""
        lit_thresh = (
            f"{self.literature_threshold:.2f}" if self.literature_threshold else "N/A"
        )
        return (
            f"{self.method_name} & {self.threshold_optimal:.2f} & "
            f"{self.auc_roc:.3f} & {self.f1_score:.3f} & "
            f"{self.precision:.3f} & {self.recall:.3f} & "
            f"{lit_thresh} & {self.dataset_name} \\\\"
        )


@dataclass
class MethodConfig:
    """Configuration for a forensic method."""

    name: str
    description: str
    compute_score: Callable  # Function to compute score for an image
    literature_refs: List[Dict[str, str]] = field(default_factory=list)
    literature_threshold: Optional[float] = None
    default_threshold: float = 0.5
    higher_is_suspicious: bool = True  # True = high score means manipulation


# =============================================================================
# THRESHOLD CALCULATION METHODS
# =============================================================================


def find_threshold_youden(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    """
    Find optimal threshold using Youden's J statistic.
    J = Sensitivity + Specificity - 1 = TPR - FPR

    This maximizes the sum of sensitivity and specificity.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Youden's J
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)

    return thresholds[optimal_idx], {
        "method": "youden",
        "j_score": j_scores[optimal_idx],
        "tpr_at_threshold": tpr[optimal_idx],
        "fpr_at_threshold": fpr[optimal_idx],
    }


def find_threshold_f1_max(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict]:
    """
    Find threshold that maximizes F1 score.
    Good for imbalanced datasets.
    """
    from sklearn.metrics import f1_score

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.05, 0.95, 0.01):
        preds = (scores >= threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, {"method": "f1_max", "f1_at_threshold": best_f1}


def find_threshold_precision_target(
    labels: np.ndarray, scores: np.ndarray, target_precision: float = 0.95
) -> Tuple[float, Dict]:
    """
    Find threshold that achieves target precision (e.g., 95%).
    Useful when false positives are costly (don't want to falsely accuse).
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # Find threshold where precision >= target
    valid_indices = np.where(precision >= target_precision)[0]

    if len(valid_indices) == 0:
        # Can't achieve target, return highest precision threshold
        idx = np.argmax(precision)
    else:
        # Among valid, pick one with best recall
        best_recall_idx = valid_indices[np.argmax(recall[valid_indices])]
        idx = best_recall_idx

    # Handle edge case
    if idx >= len(thresholds):
        idx = len(thresholds) - 1

    return thresholds[idx], {
        "method": f"precision_target_{target_precision}",
        "achieved_precision": precision[idx],
        "achieved_recall": recall[idx],
    }


def find_threshold_recall_target(
    labels: np.ndarray, scores: np.ndarray, target_recall: float = 0.90
) -> Tuple[float, Dict]:
    """
    Find threshold that achieves target recall (e.g., 90%).
    Useful when false negatives are costly (don't want to miss manipulations).
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # Find threshold where recall >= target
    valid_indices = np.where(recall >= target_recall)[0]

    if len(valid_indices) == 0:
        idx = np.argmax(recall)
    else:
        # Among valid, pick one with best precision
        best_precision_idx = valid_indices[np.argmax(precision[valid_indices])]
        idx = best_precision_idx

    if idx >= len(thresholds):
        idx = len(thresholds) - 1

    return thresholds[idx], {
        "method": f"recall_target_{target_recall}",
        "achieved_precision": precision[idx],
        "achieved_recall": recall[idx],
    }


def bootstrap_confidence_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold_func: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate confidence interval for threshold using bootstrap.
    """
    n_samples = len(labels)
    thresholds = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_labels = labels[indices]
        boot_scores = scores[indices]

        try:
            thresh, _ = threshold_func(boot_labels, boot_scores)
            thresholds.append(thresh)
        except Exception:
            continue

    if len(thresholds) < 100:
        return (0.0, 1.0)  # Not enough samples

    alpha = 1 - confidence
    lower = np.percentile(thresholds, 100 * alpha / 2)
    upper = np.percentile(thresholds, 100 * (1 - alpha / 2))

    return (lower, upper)


# =============================================================================
# FORENSIC METHOD SCORE CALCULATORS
# =============================================================================


def compute_ela_score(image_path: Path, quality: int = 95) -> float:
    """
    Compute ELA (Error Level Analysis) score.

    ELA re-saves image at known quality and measures difference.
    Manipulated regions show different error levels.

    References:
    - Krawetz (2007): "A Picture's Worth"
    - Gunawan et al. (2017): IJECE

    Returns: 0.0 (authentic) to 1.0 (suspicious)
    """
    try:
        from PIL import Image
        import io

        img = Image.open(image_path).convert("RGB")
        original = np.array(img, dtype=np.float32)

        # Re-compress at known quality
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed = np.array(Image.open(buffer), dtype=np.float32)

        # Calculate error
        error = np.abs(original - recompressed)

        # Normalize to 0-1 (max possible error is 255)
        error_normalized = error / 255.0

        # Statistics
        mean_error = np.mean(error_normalized)
        std_error = np.std(error_normalized)
        max_error = np.max(error_normalized)

        # Score: combination of mean and variance
        # High variance = inconsistent error levels = suspicious
        score = 0.4 * mean_error + 0.4 * std_error + 0.2 * (max_error / 3)

        # Clamp to 0-1
        return float(np.clip(score * 3, 0.0, 1.0))  # Scale factor based on typical values

    except Exception as e:
        logger.warning(f"ELA failed for {image_path}: {e}")
        return 0.5  # Neutral on error


def compute_noise_score(image_path: Path, block_size: int = 32) -> float:
    """
    Compute noise inconsistency score.

    Different camera sensors produce different noise patterns.
    Spliced regions have mismatched noise.

    References:
    - Mahdian & Saic (2009): "Using noise inconsistencies"
    - Pan et al. (2011): IEEE TIFS

    Returns: 0.0 (consistent noise) to 1.0 (inconsistent = suspicious)
    """
    try:
        from PIL import Image
        from scipy import ndimage

        img = Image.open(image_path).convert("L")  # Grayscale
        arr = np.array(img, dtype=np.float32)

        # Estimate noise using Laplacian
        laplacian = ndimage.laplace(arr)

        h, w = arr.shape
        variances = []

        # Calculate variance per block
        for i in range(0, h - block_size, block_size // 2):
            for j in range(0, w - block_size, block_size // 2):
                block = laplacian[i : i + block_size, j : j + block_size]
                variances.append(np.var(block))

        if len(variances) < 4:
            return 0.5

        variances = np.array(variances)

        # Coefficient of variation of variances
        # High CV = inconsistent noise = suspicious
        mean_var = np.mean(variances)
        std_var = np.std(variances)

        if mean_var < 1e-6:
            return 0.0

        cv = std_var / mean_var

        # Normalize (typical CV range is 0.1 to 2.0)
        score = np.clip(cv / 1.5, 0.0, 1.0)

        return float(score)

    except Exception as e:
        logger.warning(f"Noise analysis failed for {image_path}: {e}")
        return 0.5


def compute_jpeg_score(image_path: Path) -> float:
    """
    Compute double-JPEG compression score.

    First JPEG compression leaves "ghosts" in DCT histogram.
    Re-compression creates detectable periodic patterns.

    References:
    - Farid (2009): "Exposing Digital Forgeries from JPEG Ghosts"
    - Bianchi & Piva (2012): IEEE TIFS

    Returns: 0.0 (single compression) to 1.0 (multiple compressions = suspicious)
    """
    try:
        from PIL import Image

        # Check if actually JPEG
        img = Image.open(image_path)
        if img.format != "JPEG":
            return 0.0  # Not JPEG, can't have double compression

        arr = np.array(img.convert("L"), dtype=np.float32)

        # Simple DCT-based analysis
        # In double JPEG, DCT coefficients show periodic patterns

        # Calculate 8x8 block DCT statistics
        from scipy.fftpack import dct

        h, w = arr.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        arr = arr[:h8, :w8]

        dct_coeffs = []
        for i in range(0, h8, 8):
            for j in range(0, w8, 8):
                block = arr[i : i + 8, j : j + 8]
                dct_block = dct(dct(block.T, norm="ortho").T, norm="ortho")
                dct_coeffs.extend(dct_block.flatten()[1:])  # Skip DC

        dct_coeffs = np.array(dct_coeffs)

        # Check for periodicity in histogram
        hist, _ = np.histogram(dct_coeffs, bins=256, range=(-128, 128))
        hist = hist.astype(np.float32)
        hist = hist / (np.sum(hist) + 1e-6)

        # FFT of histogram to detect periodicity
        fft_hist = np.abs(np.fft.fft(hist))
        fft_hist = fft_hist[1 : len(fft_hist) // 2]  # Remove DC and mirror

        # Periodic patterns show peaks in FFT
        mean_fft = np.mean(fft_hist)
        max_fft = np.max(fft_hist)

        if mean_fft < 1e-6:
            return 0.0

        periodicity_score = (max_fft / mean_fft - 1) / 10  # Normalize

        return float(np.clip(periodicity_score, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"JPEG analysis failed for {image_path}: {e}")
        return 0.5


def compute_screenshot_score(image_path: Path) -> float:
    """
    Compute screenshot likelihood score.

    Screenshots have characteristic:
    - Standard display resolutions
    - Specific aspect ratios
    - Often PNG format
    - No camera EXIF

    Returns: 0.0 (not screenshot) to 1.0 (likely screenshot)
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(image_path)
        width, height = img.size

        score = 0.0

        # Common screen resolutions
        screen_resolutions = [
            (1920, 1080),
            (2560, 1440),
            (3840, 2160),
            (1366, 768),
            (1536, 864),
            (1440, 900),
            (1280, 720),
            (2560, 1600),
            (1680, 1050),
            # Mobile
            (1170, 2532),
            (1284, 2778),
            (1080, 2400),
            (1080, 2340),
            (1440, 3200),
            # Rotated mobile
            (2532, 1170),
            (2778, 1284),
            (2400, 1080),
        ]

        # Check resolution match
        for sw, sh in screen_resolutions:
            if (width == sw and height == sh) or (width == sh and height == sw):
                score += 0.4
                break

        # Check common aspect ratios
        aspect = width / height if height > 0 else 0
        screen_aspects = [16 / 9, 16 / 10, 4 / 3, 21 / 9, 9 / 16, 9 / 19.5, 9 / 20]

        for sa in screen_aspects:
            if abs(aspect - sa) < 0.02:
                score += 0.2
                break

        # Check for no camera EXIF
        try:
            exif = img._getexif()
            if exif is None:
                score += 0.2
            else:
                # Check if has camera make/model
                has_camera = False
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag in ["Make", "Model", "LensModel"]:
                        has_camera = True
                        break
                if not has_camera:
                    score += 0.1
        except Exception:
            score += 0.1

        # PNG format often used for screenshots
        if img.format == "PNG":
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"Screenshot analysis failed for {image_path}: {e}")
        return 0.5


def compute_platform_score(image_path: Path) -> Dict[str, float]:
    """
    Compute platform likelihood scores (WhatsApp, Instagram, etc.).

    Platforms apply specific compression and resize patterns.
    This is HEURISTIC - not definitive proof!

    Returns: Dict with platform probabilities
    """
    try:
        from PIL import Image

        img = Image.open(image_path)
        width, height = img.size

        scores = {
            "whatsapp": 0.0,
            "instagram": 0.0,
            "facebook": 0.0,
            "twitter": 0.0,
            "telegram": 0.0,
            "unknown": 0.5,
        }

        # WhatsApp patterns (as of 2024)
        # - Max dimension ~1600px
        # - JPEG quality ~70-80
        # - Strips most EXIF
        if max(width, height) <= 1600 and min(width, height) >= 100:
            if img.format == "JPEG":
                # Check approximate quality via file size heuristic
                file_size = image_path.stat().st_size
                pixels = width * height
                bpp = file_size * 8 / pixels if pixels > 0 else 0

                # WhatsApp typically 0.5-1.5 bpp
                if 0.3 < bpp < 2.0:
                    scores["whatsapp"] = 0.3

        # Instagram patterns
        # - Square crops common (1:1)
        # - 1080px width typical
        # - 4:5 ratio for portraits
        aspect = width / height if height > 0 else 0
        if abs(aspect - 1.0) < 0.05:  # Square
            scores["instagram"] += 0.2
        if abs(aspect - 0.8) < 0.05:  # 4:5
            scores["instagram"] += 0.15
        if width == 1080:
            scores["instagram"] += 0.2

        # Facebook patterns
        # - 2048px max
        # - Heavy compression
        if max(width, height) == 2048:
            scores["facebook"] += 0.25

        # Twitter patterns
        # - 4096px max
        # - Converts PNG to JPEG sometimes
        if max(width, height) <= 4096 and max(width, height) >= 1200:
            scores["twitter"] += 0.1

        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    except Exception as e:
        logger.warning(f"Platform analysis failed for {image_path}: {e}")
        return {"unknown": 1.0}


def compute_ai_score_simple(image_path: Path) -> float:
    """
    Simple AI detection score without loading heavy models.
    Uses frequency and texture analysis.

    For full AI detection, use HuggingFace models separately.

    Returns: 0.0 (likely real) to 1.0 (likely AI)
    """
    try:
        from PIL import Image
        from scipy.fftpack import fft2, fftshift

        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=np.float32)

        # Frequency analysis
        # AI images often lack high-frequency detail
        f_transform = fft2(arr)
        f_shift = fftshift(f_transform)
        magnitude = np.abs(f_shift)

        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        # Low frequency (center)
        low_freq = magnitude[
            center_h - h // 8 : center_h + h // 8,
            center_w - w // 8 : center_w + w // 8,
        ]

        # High frequency (edges)
        high_freq_mask = np.ones_like(magnitude, dtype=bool)
        high_freq_mask[
            center_h - h // 4 : center_h + h // 4,
            center_w - w // 4 : center_w + w // 4,
        ] = False
        high_freq = magnitude[high_freq_mask]

        low_energy = np.sum(low_freq)
        high_energy = np.sum(high_freq)

        if high_energy < 1e-6:
            freq_ratio = 1.0
        else:
            freq_ratio = low_energy / (low_energy + high_energy)

        # AI images tend to have higher low-freq ratio (smoother)
        # Real photos have more high-freq detail
        # Typical: real ~0.3-0.5, AI ~0.5-0.8
        ai_score = (freq_ratio - 0.3) / 0.5

        return float(np.clip(ai_score, 0.0, 1.0))

    except Exception as e:
        logger.warning(f"AI analysis failed for {image_path}: {e}")
        return 0.5


# =============================================================================
# CALIBRATION ENGINE
# =============================================================================


class CalibrationEngine:
    """Main calibration engine for all forensic methods."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, CalibrationResult] = {}

        # Define methods with literature references
        self.methods = {
            "ela": MethodConfig(
                name="ELA (Error Level Analysis)",
                description="Detects manipulation by analyzing JPEG re-compression error levels",
                compute_score=compute_ela_score,
                literature_refs=[
                    {
                        "author": "Krawetz, N.",
                        "year": "2007",
                        "title": "A Picture's Worth... Digital Image Analysis and Forensics",
                        "venue": "Hack in the Box Conference",
                    },
                    {
                        "author": "Gunawan et al.",
                        "year": "2017",
                        "title": "Development of Photo Forensic using ELA",
                        "venue": "IJECE",
                    },
                ],
                literature_threshold=None,  # No universal threshold in literature
                default_threshold=0.5,
            ),
            "noise": MethodConfig(
                name="Noise Inconsistency",
                description="Detects splicing by analyzing noise pattern variations",
                compute_score=compute_noise_score,
                literature_refs=[
                    {
                        "author": "Mahdian & Saic",
                        "year": "2009",
                        "title": "Using noise inconsistencies for blind image forensics",
                        "venue": "Image and Vision Computing",
                    },
                ],
                literature_threshold=None,
                default_threshold=0.5,
            ),
            "jpeg": MethodConfig(
                name="JPEG Double Compression",
                description="Detects multiple JPEG compressions via DCT analysis",
                compute_score=compute_jpeg_score,
                literature_refs=[
                    {
                        "author": "Farid, H.",
                        "year": "2009",
                        "title": "Exposing Digital Forgeries from JPEG Ghosts",
                        "venue": "IEEE TIFS",
                    },
                    {
                        "author": "Bianchi & Piva",
                        "year": "2012",
                        "title": "Image Forgery Localization via Block-Grained Analysis",
                        "venue": "IEEE TIFS",
                    },
                ],
                literature_threshold=None,
                default_threshold=0.5,
            ),
            "screenshot": MethodConfig(
                name="Screenshot Detection",
                description="Detects screenshots via resolution and metadata analysis",
                compute_score=compute_screenshot_score,
                literature_refs=[],  # Heuristic method
                literature_threshold=None,
                default_threshold=0.5,
            ),
            "ai_simple": MethodConfig(
                name="AI Detection (Frequency)",
                description="Detects AI-generated images via frequency analysis",
                compute_score=compute_ai_score_simple,
                literature_refs=[
                    {
                        "author": "Wang et al.",
                        "year": "2020",
                        "title": "CNN-generated images are surprisingly easy to spot",
                        "venue": "CVPR",
                    },
                ],
                literature_threshold=None,
                default_threshold=0.5,
            ),
        }

    def load_dataset(
        self, dataset_path: Path, labels_file: Optional[Path] = None
    ) -> Tuple[List[Path], np.ndarray]:
        """
        Load dataset with labels.

        Expected structure:
        dataset/
        ├── authentic/     # Label 0
        │   ├── img1.jpg
        │   └── ...
        └── manipulated/   # Label 1 (or 'fake', 'ai', 'tampered')
            ├── img1.jpg
            └── ...

        Or with labels.json:
        {"img1.jpg": 0, "img2.jpg": 1, ...}
        """
        images = []
        labels = []

        if labels_file and labels_file.exists():
            # Load from JSON
            with open(labels_file) as f:
                label_map = json.load(f)
            for img_name, label in label_map.items():
                img_path = dataset_path / img_name
                if img_path.exists():
                    images.append(img_path)
                    labels.append(label)
        else:
            # Infer from folder structure
            authentic_names = ["authentic", "real", "original", "genuine", "0"]
            manipulated_names = [
                "manipulated",
                "fake",
                "tampered",
                "forged",
                "ai",
                "synthetic",
                "1",
            ]

            for subdir in dataset_path.iterdir():
                if not subdir.is_dir():
                    continue

                name_lower = subdir.name.lower()

                if any(n in name_lower for n in authentic_names):
                    label = 0
                elif any(n in name_lower for n in manipulated_names):
                    label = 1
                else:
                    continue

                for img_path in subdir.glob("*"):
                    if img_path.suffix.lower() in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".webp",
                        ".bmp",
                    ]:
                        images.append(img_path)
                        labels.append(label)

        logger.info(
            f"Loaded {len(images)} images: {sum(1 for l in labels if l == 0)} authentic, "
            f"{sum(1 for l in labels if l == 1)} manipulated"
        )

        return images, np.array(labels)

    def compute_scores(
        self, method_name: str, images: List[Path], show_progress: bool = True
    ) -> np.ndarray:
        """Compute scores for all images using specified method."""
        method = self.methods[method_name]
        scores = []

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(images, desc=f"Computing {method_name}")
            except ImportError:
                iterator = images
                logger.info(f"Computing {method_name} for {len(images)} images...")
        else:
            iterator = images

        for img_path in iterator:
            score = method.compute_score(img_path)
            scores.append(score)

        return np.array(scores)

    def calibrate_method(
        self,
        method_name: str,
        images: List[Path],
        labels: np.ndarray,
        dataset_name: str = "unknown",
    ) -> CalibrationResult:
        """
        Calibrate threshold for a single method.

        Tries multiple threshold selection methods and picks the best.
        """
        from sklearn.metrics import (
            accuracy_score,
            auc,
            f1_score,
            precision_score,
            recall_score,
            roc_curve,
        )

        method = self.methods[method_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Calibrating: {method.name}")
        logger.info(f"{'='*60}")

        # Compute scores
        scores = self.compute_scores(method_name, images)

        # Try different threshold methods
        threshold_methods = [
            ("youden", find_threshold_youden),
            ("f1_max", find_threshold_f1_max),
            ("precision_95", lambda l, s: find_threshold_precision_target(l, s, 0.95)),
            ("recall_90", lambda l, s: find_threshold_recall_target(l, s, 0.90)),
        ]

        best_result = None
        best_f1 = 0

        for thresh_name, thresh_func in threshold_methods:
            try:
                threshold, details = thresh_func(labels, scores)

                # Calculate metrics at this threshold
                preds = (scores >= threshold).astype(int)

                f1 = f1_score(labels, preds, zero_division=0)
                precision = precision_score(labels, preds, zero_division=0)
                recall = recall_score(labels, preds, zero_division=0)
                accuracy = accuracy_score(labels, preds)

                # AUC
                fpr, tpr, _ = roc_curve(labels, scores)
                auc_score = auc(fpr, tpr)

                logger.info(
                    f"  {thresh_name}: threshold={threshold:.3f}, F1={f1:.3f}, AUC={auc_score:.3f}"
                )

                if f1 > best_f1:
                    best_f1 = f1
                    best_result = {
                        "threshold": threshold,
                        "method": thresh_name,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                        "accuracy": accuracy,
                        "auc": auc_score,
                    }

            except Exception as e:
                logger.warning(f"  {thresh_name} failed: {e}")

        if best_result is None:
            logger.error(f"All threshold methods failed for {method_name}")
            return None

        # Bootstrap confidence interval
        logger.info("  Computing confidence interval (bootstrap)...")
        ci = bootstrap_confidence_interval(
            labels, scores, find_threshold_f1_max, n_bootstrap=500
        )

        # Create result
        result = CalibrationResult(
            method_name=method.name,
            threshold_optimal=best_result["threshold"],
            threshold_method=best_result["method"],
            auc_roc=best_result["auc"],
            f1_score=best_result["f1"],
            precision=best_result["precision"],
            recall=best_result["recall"],
            accuracy=best_result["accuracy"],
            n_samples=len(images),
            n_positive=int(np.sum(labels)),
            n_negative=int(np.sum(1 - labels)),
            dataset_name=dataset_name,
            calibration_date=datetime.now().isoformat(),
            confidence_interval=ci,
            literature_threshold=method.literature_threshold,
            literature_source=(
                method.literature_refs[0]["author"] if method.literature_refs else None
            ),
        )

        logger.info(f"\n  BEST: threshold={result.threshold_optimal:.3f} ({result.threshold_method})")
        logger.info(f"        AUC={result.auc_roc:.3f}, F1={result.f1_score:.3f}")
        logger.info(f"        95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

        self.results[method_name] = result
        return result

    def calibrate_all(
        self, images: List[Path], labels: np.ndarray, dataset_name: str = "unknown"
    ):
        """Calibrate all methods."""
        for method_name in self.methods:
            self.calibrate_method(method_name, images, labels, dataset_name)

    def generate_report(self) -> str:
        """Generate markdown report with all results."""
        report = []
        report.append("# Threshold Calibration Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        report.append("## Summary Table\n")
        report.append(
            "| Method | Threshold | AUC | F1 | Precision | Recall | Dataset |"
        )
        report.append(
            "|--------|-----------|-----|----|-----------| -------|---------|"
        )

        for name, result in self.results.items():
            report.append(
                f"| {result.method_name} | {result.threshold_optimal:.3f} | "
                f"{result.auc_roc:.3f} | {result.f1_score:.3f} | "
                f"{result.precision:.3f} | {result.recall:.3f} | {result.dataset_name} |"
            )

        report.append("\n## Detailed Results\n")

        for name, result in self.results.items():
            method = self.methods[name]
            report.append(f"### {result.method_name}\n")
            report.append(f"**Description:** {method.description}\n")
            report.append(f"**Optimal Threshold:** {result.threshold_optimal:.4f}")
            report.append(f"**Selection Method:** {result.threshold_method}")
            report.append(
                f"**95% Confidence Interval:** [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]"
            )
            report.append(f"\n**Metrics at Threshold:**")
            report.append(f"- AUC-ROC: {result.auc_roc:.4f}")
            report.append(f"- F1 Score: {result.f1_score:.4f}")
            report.append(f"- Precision: {result.precision:.4f}")
            report.append(f"- Recall: {result.recall:.4f}")
            report.append(f"- Accuracy: {result.accuracy:.4f}")
            report.append(f"\n**Dataset:** {result.dataset_name}")
            report.append(
                f"- Total samples: {result.n_samples}"
            )
            report.append(
                f"- Positive (manipulated): {result.n_positive}"
            )
            report.append(
                f"- Negative (authentic): {result.n_negative}"
            )

            if method.literature_refs:
                report.append("\n**Literature References:**")
                for ref in method.literature_refs:
                    report.append(
                        f"- {ref['author']} ({ref['year']}): \"{ref['title']}\", {ref['venue']}"
                    )

            report.append("")

        return "\n".join(report)

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for thesis."""
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Calibrated Thresholds for Forensic Methods}")
        latex.append("\\label{tab:thresholds}")
        latex.append("\\begin{tabular}{lccccccl}")
        latex.append("\\toprule")
        latex.append(
            "Method & Threshold & AUC & F1 & Precision & Recall & Lit. & Dataset \\\\"
        )
        latex.append("\\midrule")

        for result in self.results.values():
            latex.append(result.to_latex_row())

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        return "\n".join(latex)

    def save_results(self):
        """Save all results to files."""
        # JSON
        json_path = self.output_dir / "calibration_results.json"
        with open(json_path, "w") as f:
            json.dump({k: v.to_dict() for k, v in self.results.items()}, f, indent=2)
        logger.info(f"Saved JSON: {json_path}")

        # Markdown report
        md_path = self.output_dir / "calibration_report.md"
        with open(md_path, "w") as f:
            f.write(self.generate_report())
        logger.info(f"Saved Markdown: {md_path}")

        # LaTeX table
        latex_path = self.output_dir / "calibration_table.tex"
        with open(latex_path, "w") as f:
            f.write(self.generate_latex_table())
        logger.info(f"Saved LaTeX: {latex_path}")

        # Python config file (for use in code)
        config_path = self.output_dir / "calibrated_thresholds.py"
        with open(config_path, "w") as f:
            f.write('"""Auto-generated calibrated thresholds."""\n\n')
            f.write("CALIBRATED_THRESHOLDS = {\n")
            for name, result in self.results.items():
                f.write(f'    "{name}": {result.threshold_optimal:.4f},\n')
            f.write("}\n\n")
            f.write("CALIBRATION_METADATA = {\n")
            for name, result in self.results.items():
                f.write(f'    "{name}": {{\n')
                f.write(f'        "auc": {result.auc_roc:.4f},\n')
                f.write(f'        "f1": {result.f1_score:.4f},\n')
                f.write(f'        "dataset": "{result.dataset_name}",\n')
                f.write(f'        "ci_lower": {result.confidence_interval[0]:.4f},\n')
                f.write(f'        "ci_upper": {result.confidence_interval[1]:.4f},\n')
                f.write(f"    }},\n")
            f.write("}\n")
        logger.info(f"Saved Python config: {config_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate forensic method thresholds with academic rigor"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset with authentic/ and manipulated/ subfolders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/calibration"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--method",
        choices=["ela", "noise", "jpeg", "screenshot", "ai_simple", "all"],
        default="all",
        help="Which method to calibrate",
    )
    parser.add_argument(
        "--labels", type=Path, help="Optional JSON file with image labels"
    )
    parser.add_argument(
        "--dataset-name", default="custom", help="Name of dataset for documentation"
    )
    parser.add_argument(
        "--generate-latex", action="store_true", help="Generate LaTeX table"
    )

    args = parser.parse_args()

    # Initialize engine
    engine = CalibrationEngine(args.output)

    # Load dataset
    images, labels = engine.load_dataset(args.dataset, args.labels)

    if len(images) == 0:
        logger.error("No images found in dataset!")
        sys.exit(1)

    # Calibrate
    if args.method == "all":
        engine.calibrate_all(images, labels, args.dataset_name)
    else:
        engine.calibrate_method(args.method, images, labels, args.dataset_name)

    # Save results
    engine.save_results()

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {args.output}")
    print("\nGenerated files:")
    print(f"  - calibration_results.json (raw data)")
    print(f"  - calibration_report.md (documentation)")
    print(f"  - calibration_table.tex (LaTeX for thesis)")
    print(f"  - calibrated_thresholds.py (use in code)")


if __name__ == "__main__":
    main()
