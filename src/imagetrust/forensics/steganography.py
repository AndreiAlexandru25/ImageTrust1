"""
Steganography Detection Module

Detection of hidden information (steganography) in images.

Implemented techniques:
1. LSB Analysis (Least Significant Bit) - detects modifications in LSB bits
2. Chi-Square Attack - statistical analysis for LSB embedding
3. RS Analysis (Regular-Singular) - detects LSB with high precision
4. Sample Pair Analysis (SPA) - advanced method for LSB
5. Histogram Analysis - detects anomalies in the histogram
6. DCT Analysis - for JPEG steganography (F5, OutGuess)
7. Visual Attack - visualization of bit planes
8. Noise Residual Analysis - detects unnatural patterns

References:
- Fridrich et al. (2001): "Detecting LSB Steganography in Color and Gray-Scale Images"
- Dumitrescu et al. (2003): "Detection of LSB Steganography via Sample Pair Analysis"
- Westfeld & Pfitzmann (1999): "Attacks on Steganographic Systems"
- Fridrich et al. (2001): "Reliable Detection of LSB Steganography in Grayscale and Color Images"

Author: ImageTrust Project
"""

import io
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES & ENUMS
# =============================================================================


class StegMethod(Enum):
    """Known steganography methods."""
    LSB_REPLACEMENT = "lsb_replacement"
    LSB_MATCHING = "lsb_matching"
    JSTEG = "jsteg"
    F5 = "f5"
    OUTGUESS = "outguess"
    STEGHIDE = "steghide"
    OPENSTEGO = "openstego"
    UNKNOWN = "unknown"


class StegConfidence(Enum):
    """Confidence levels for steganography detection."""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class StegAnalysisResult:
    """Result of a single steganography analysis method."""
    method_name: str
    score: float  # 0.0 = clean, 1.0 = definitely contains hidden data
    confidence: StegConfidence
    detected: bool
    estimated_payload: float  # Estimated % of capacity used (0-100)
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Tuple[str, np.ndarray]] = field(default_factory=list)


@dataclass
class StegDetectionReport:
    """Complete steganography detection report."""
    image_path: Optional[str]
    image_size: Tuple[int, int]
    image_format: str
    overall_score: float
    overall_confidence: StegConfidence
    steg_detected: bool
    likely_method: StegMethod
    estimated_payload_percent: float
    analysis_results: List[StegAnalysisResult]
    recommendations: List[str]
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "image_size": self.image_size,
            "image_format": self.image_format,
            "overall_score": round(self.overall_score, 4),
            "overall_confidence": self.overall_confidence.name,
            "steg_detected": self.steg_detected,
            "likely_method": self.likely_method.value,
            "estimated_payload_percent": round(self.estimated_payload_percent, 2),
            "analysis_results": [
                {
                    "method": r.method_name,
                    "score": round(r.score, 4),
                    "confidence": r.confidence.name,
                    "detected": r.detected,
                    "payload": round(r.estimated_payload, 2),
                    "explanation": r.explanation,
                }
                for r in self.analysis_results
            ],
            "recommendations": self.recommendations,
            "limitations": self.limitations,
        }


# =============================================================================
# LSB ANALYSIS (Least Significant Bit)
# =============================================================================


class LSBAnalyzer:
    """
    LSB (Least Significant Bit) analysis for steganography detection.

    LSB is the most common steganography method:
    - Hides data in the least significant bit of each pixel
    - Modifications are imperceptible to the human eye
    - But leaves detectable statistical traces

    References:
    - Westfeld & Pfitzmann (1999): "Attacks on Steganographic Systems"
    """

    def __init__(self):
        self.name = "LSB Analysis"

    def analyze(self, image: Image.Image) -> StegAnalysisResult:
        """Perform LSB analysis on image."""
        try:
            arr = np.array(image)

            if len(arr.shape) == 2:
                # Grayscale
                results = self._analyze_channel(arr)
                avg_score = results["score"]
                details = {"grayscale": results}
            else:
                # Color image - analyze each channel
                channel_results = {}
                scores = []

                for i, channel_name in enumerate(["red", "green", "blue"]):
                    if i < arr.shape[2]:
                        channel_data = arr[:, :, i]
                        result = self._analyze_channel(channel_data)
                        channel_results[channel_name] = result
                        scores.append(result["score"])

                avg_score = np.mean(scores)
                details = channel_results

            # Determine confidence based on score consistency
            if len(details) > 1:
                score_std = np.std(scores)
                if score_std < 0.1:
                    confidence = StegConfidence.HIGH
                elif score_std < 0.2:
                    confidence = StegConfidence.MEDIUM
                else:
                    confidence = StegConfidence.LOW
            else:
                confidence = StegConfidence.MEDIUM

            # Estimate payload
            payload = self._estimate_payload(details)

            # Determine if steganography detected
            detected = avg_score > 0.5

            explanation = self._generate_explanation(avg_score, details)

            return StegAnalysisResult(
                method_name=self.name,
                score=float(avg_score),
                confidence=confidence,
                detected=detected,
                estimated_payload=payload,
                explanation=explanation,
                details=details,
            )

        except Exception as e:
            logger.error(f"LSB analysis failed: {e}")
            return StegAnalysisResult(
                method_name=self.name,
                score=0.5,
                confidence=StegConfidence.VERY_LOW,
                detected=False,
                estimated_payload=0.0,
                explanation=f"Analysis failed: {str(e)}",
            )

    def _analyze_channel(self, channel: np.ndarray) -> Dict[str, Any]:
        """Analyze a single color channel for LSB anomalies."""
        # Extract LSB plane
        lsb = channel & 1

        # 1. Bit frequency analysis
        ones_ratio = np.mean(lsb)
        # In natural images, LSB should be ~50% ones
        # In LSB stego, it tends to be very close to 50%
        freq_deviation = abs(ones_ratio - 0.5)

        # 2. LSB flip detection
        # Count pairs where adjacent pixels differ only in LSB
        h_diff = np.abs(channel[:, :-1].astype(int) - channel[:, 1:].astype(int))
        v_diff = np.abs(channel[:-1, :].astype(int) - channel[1:, :].astype(int))

        # LSB differences (should be 0 or 1)
        h_lsb_diff = h_diff == 1
        v_lsb_diff = v_diff == 1

        lsb_pair_ratio = (np.sum(h_lsb_diff) + np.sum(v_lsb_diff)) / (
            h_lsb_diff.size + v_lsb_diff.size + 1e-10
        )

        # 3. Histogram analysis of LSB
        hist_0 = np.sum(lsb == 0)
        hist_1 = np.sum(lsb == 1)
        hist_balance = min(hist_0, hist_1) / (max(hist_0, hist_1) + 1e-10)

        # 4. Blockiness in LSB plane
        block_size = 8
        h, w = lsb.shape
        blocks_h = h // block_size
        blocks_w = w // block_size

        block_variances = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = lsb[
                    i * block_size : (i + 1) * block_size,
                    j * block_size : (j + 1) * block_size,
                ]
                block_variances.append(np.var(block))

        avg_block_var = np.mean(block_variances) if block_variances else 0.25

        # Combine metrics into score
        # Low freq deviation + high hist balance + moderate lsb pairs = suspicious
        score = 0.0

        # Very balanced histogram is suspicious (natural images have slight imbalance)
        if hist_balance > 0.98:
            score += 0.3
        elif hist_balance > 0.95:
            score += 0.2

        # Very close to 50% ones is suspicious
        if freq_deviation < 0.01:
            score += 0.3
        elif freq_deviation < 0.02:
            score += 0.2

        # High LSB pair ratio can indicate embedding
        if lsb_pair_ratio > 0.4:
            score += 0.2
        elif lsb_pair_ratio > 0.3:
            score += 0.1

        # Low block variance in LSB plane is suspicious
        if avg_block_var < 0.2:
            score += 0.2
        elif avg_block_var < 0.23:
            score += 0.1

        return {
            "score": min(score, 1.0),
            "ones_ratio": float(ones_ratio),
            "freq_deviation": float(freq_deviation),
            "lsb_pair_ratio": float(lsb_pair_ratio),
            "hist_balance": float(hist_balance),
            "avg_block_variance": float(avg_block_var),
        }

    def _estimate_payload(self, details: Dict[str, Any]) -> float:
        """Estimate the payload size based on analysis results."""
        # This is a rough estimation
        scores = []
        for key, value in details.items():
            if isinstance(value, dict) and "score" in value:
                scores.append(value["score"])

        if not scores:
            return 0.0

        avg_score = np.mean(scores)

        # Higher score suggests more data embedded
        # Estimate as percentage of max capacity
        return float(avg_score * 50)  # Max 50% estimate

    def _generate_explanation(self, score: float, details: Dict) -> str:
        """Generate human-readable explanation."""
        if score < 0.3:
            return "No significant LSB anomalies detected. Image appears clean."
        elif score < 0.5:
            return "Minor LSB statistical anomalies present. Could be natural variation or light embedding."
        elif score < 0.7:
            return "Moderate LSB anomalies detected. Possible steganography present."
        else:
            return "Strong LSB anomalies detected. High probability of hidden data."


# =============================================================================
# CHI-SQUARE ANALYSIS
# =============================================================================


class ChiSquareAnalyzer:
    """
    Chi-Square Attack for LSB steganography detection.

    Principle:
    - In natural images, pixel values 2n and 2n+1 have different frequencies
    - After LSB embedding, these pairs tend to have equal frequencies
    - Chi-square test detects this "flattening"

    References:
    - Westfeld & Pfitzmann (1999): "Attacks on Steganographic Systems"
    """

    def __init__(self):
        self.name = "Chi-Square Attack"

    def analyze(self, image: Image.Image) -> StegAnalysisResult:
        """Perform chi-square analysis."""
        try:
            arr = np.array(image)

            if len(arr.shape) == 2:
                p_value, chi_sq = self._chi_square_test(arr.flatten())
                results = {"grayscale": {"p_value": p_value, "chi_square": chi_sq}}
            else:
                results = {}
                p_values = []

                for i, name in enumerate(["red", "green", "blue"]):
                    if i < arr.shape[2]:
                        p_val, chi_sq = self._chi_square_test(arr[:, :, i].flatten())
                        results[name] = {"p_value": p_val, "chi_square": chi_sq}
                        p_values.append(p_val)

            # Combine p-values (Fisher's method)
            all_p_values = [
                r["p_value"] for r in results.values() if "p_value" in r
            ]

            if all_p_values:
                # Low p-value = evidence of steganography
                avg_p = np.mean(all_p_values)

                # Convert p-value to score (low p = high score)
                if avg_p < 0.001:
                    score = 0.95
                elif avg_p < 0.01:
                    score = 0.8
                elif avg_p < 0.05:
                    score = 0.6
                elif avg_p < 0.1:
                    score = 0.4
                else:
                    score = 0.2
            else:
                score = 0.5
                avg_p = 0.5

            detected = score > 0.5
            confidence = self._determine_confidence(score, all_p_values)

            if detected:
                explanation = f"Chi-square test indicates LSB manipulation (p={avg_p:.4f})"
            else:
                explanation = f"Chi-square test shows normal LSB distribution (p={avg_p:.4f})"

            return StegAnalysisResult(
                method_name=self.name,
                score=score,
                confidence=confidence,
                detected=detected,
                estimated_payload=score * 40,  # Rough estimate
                explanation=explanation,
                details=results,
            )

        except Exception as e:
            logger.error(f"Chi-square analysis failed: {e}")
            return StegAnalysisResult(
                method_name=self.name,
                score=0.5,
                confidence=StegConfidence.VERY_LOW,
                detected=False,
                estimated_payload=0.0,
                explanation=f"Analysis failed: {str(e)}",
            )

    def _chi_square_test(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Perform chi-square test on pixel values.

        Tests if pairs (2k, 2k+1) have equal frequencies,
        which would indicate LSB replacement.
        """
        from scipy import stats

        # Create histogram with 256 bins
        hist, _ = np.histogram(data, bins=256, range=(0, 256))

        # Group into pairs (0,1), (2,3), (4,5), ...
        pairs = hist.reshape(-1, 2)

        # Expected: average of each pair
        expected = np.mean(pairs, axis=1, keepdims=True)
        expected = np.tile(expected, (1, 2)).flatten()

        # Observed
        observed = hist.astype(float)

        # Remove zero expected values to avoid division by zero
        mask = expected > 0
        if np.sum(mask) < 10:
            return 0.5, 0.0

        observed = observed[mask]
        expected = expected[mask]

        # Chi-square statistic
        chi_sq = np.sum((observed - expected) ** 2 / expected)
        df = len(observed) - 1

        # P-value
        p_value = 1 - stats.chi2.cdf(chi_sq, df)

        return float(p_value), float(chi_sq)

    def _determine_confidence(
        self, score: float, p_values: List[float]
    ) -> StegConfidence:
        """Determine confidence based on consistency of p-values."""
        if not p_values:
            return StegConfidence.VERY_LOW

        std_p = np.std(p_values)

        if score > 0.8 and std_p < 0.1:
            return StegConfidence.VERY_HIGH
        elif score > 0.6 and std_p < 0.2:
            return StegConfidence.HIGH
        elif score > 0.4:
            return StegConfidence.MEDIUM
        else:
            return StegConfidence.LOW


# =============================================================================
# RS ANALYSIS (Regular-Singular)
# =============================================================================


class RSAnalyzer:
    """
    RS Analysis (Regular-Singular) for LSB steganography detection.

    Principle:
    - Divides the image into pixel groups
    - Applies the "flipping" function and measures "smoothness"
    - Groups are classified as Regular, Singular, or Unusable
    - The relationship between R, S, R*, S* indicates embedding

    Advantages:
    - Can estimate message length
    - Works even for partial embedding
    - More robust than chi-square

    References:
    - Fridrich et al. (2001): "Reliable Detection of LSB Steganography"
    """

    def __init__(self, mask: Optional[np.ndarray] = None):
        self.name = "RS Analysis"
        # Default mask for pixel groups
        self.mask = mask if mask is not None else np.array([0, 1, 1, 0])

    def analyze(self, image: Image.Image) -> StegAnalysisResult:
        """Perform RS analysis."""
        try:
            arr = np.array(image)

            if len(arr.shape) == 2:
                result = self._rs_analysis_channel(arr)
                results = {"grayscale": result}
                scores = [result["estimated_rate"]]
            else:
                results = {}
                scores = []

                for i, name in enumerate(["red", "green", "blue"]):
                    if i < arr.shape[2]:
                        result = self._rs_analysis_channel(arr[:, :, i])
                        results[name] = result
                        scores.append(result["estimated_rate"])

            avg_rate = np.mean(scores)

            # Convert embedding rate to detection score
            if avg_rate > 0.5:
                score = 0.95
            elif avg_rate > 0.3:
                score = 0.8
            elif avg_rate > 0.1:
                score = 0.6
            elif avg_rate > 0.05:
                score = 0.4
            else:
                score = 0.2

            detected = avg_rate > 0.1
            confidence = StegConfidence.HIGH if detected else StegConfidence.MEDIUM

            if detected:
                explanation = (
                    f"RS analysis estimates ~{avg_rate*100:.1f}% embedding rate. "
                    "LSB steganography likely present."
                )
            else:
                explanation = (
                    f"RS analysis estimates ~{avg_rate*100:.1f}% embedding rate. "
                    "No significant steganography detected."
                )

            return StegAnalysisResult(
                method_name=self.name,
                score=score,
                confidence=confidence,
                detected=detected,
                estimated_payload=avg_rate * 100,
                explanation=explanation,
                details=results,
            )

        except Exception as e:
            logger.error(f"RS analysis failed: {e}")
            return StegAnalysisResult(
                method_name=self.name,
                score=0.5,
                confidence=StegConfidence.VERY_LOW,
                detected=False,
                estimated_payload=0.0,
                explanation=f"Analysis failed: {str(e)}",
            )

    def _rs_analysis_channel(self, channel: np.ndarray) -> Dict[str, float]:
        """Perform RS analysis on a single channel."""
        h, w = channel.shape
        mask_len = len(self.mask)

        # Pad image to be divisible by mask length
        pad_w = (mask_len - (w % mask_len)) % mask_len
        if pad_w > 0:
            channel = np.pad(channel, ((0, 0), (0, pad_w)), mode="edge")

        h, w = channel.shape
        groups_per_row = w // mask_len

        # Initialize counters
        R_m = 0  # Regular groups with mask M
        S_m = 0  # Singular groups with mask M
        R_neg_m = 0  # Regular groups with mask -M
        S_neg_m = 0  # Singular groups with mask -M

        # Negative mask
        neg_mask = -self.mask

        for row in range(h):
            for g in range(groups_per_row):
                start = g * mask_len
                group = channel[row, start : start + mask_len].astype(np.float64)

                # Original smoothness
                f_orig = self._discrimination_function(group)

                # Apply mask M (flip LSB where mask is 1)
                group_m = self._apply_mask(group.copy(), self.mask)
                f_m = self._discrimination_function(group_m)

                # Apply mask -M
                group_neg_m = self._apply_mask(group.copy(), neg_mask)
                f_neg_m = self._discrimination_function(group_neg_m)

                # Classify for mask M
                if f_m > f_orig:
                    R_m += 1
                elif f_m < f_orig:
                    S_m += 1

                # Classify for mask -M
                if f_neg_m > f_orig:
                    R_neg_m += 1
                elif f_neg_m < f_orig:
                    S_neg_m += 1

        total_groups = h * groups_per_row

        # Normalize
        R_m /= total_groups
        S_m /= total_groups
        R_neg_m /= total_groups
        S_neg_m /= total_groups

        # Estimate embedding rate using RS equations
        # The formula is derived from the relationship between R, S, R*, S*
        d0 = R_m - S_m
        d1 = R_neg_m - S_neg_m

        # Quadratic equation coefficients
        # p = estimated message length / max capacity
        if abs(d0 - d1) < 1e-10:
            p = 0.0
        else:
            # Simplified estimation
            a = 2 * (d1 + d0)
            b = d0 - d1
            c = d1 - d0

            if abs(a) < 1e-10:
                p = 0.0
            else:
                discriminant = b * b - 4 * a * c
                if discriminant < 0:
                    p = 0.0
                else:
                    p1 = (-b + np.sqrt(discriminant)) / (2 * a)
                    p2 = (-b - np.sqrt(discriminant)) / (2 * a)

                    # Choose the root in [0, 1]
                    if 0 <= p1 <= 1:
                        p = p1
                    elif 0 <= p2 <= 1:
                        p = p2
                    else:
                        p = max(0, min(1, (p1 + p2) / 2))

        return {
            "R_m": R_m,
            "S_m": S_m,
            "R_neg_m": R_neg_m,
            "S_neg_m": S_neg_m,
            "d0": d0,
            "d1": d1,
            "estimated_rate": float(max(0, min(1, p))),
        }

    def _discrimination_function(self, group: np.ndarray) -> float:
        """
        Calculate discrimination function (smoothness measure).
        Higher value = smoother (more predictable) pixel values.
        """
        # Sum of absolute differences between adjacent pixels
        return float(np.sum(np.abs(np.diff(group))))

    def _apply_mask(self, group: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply flipping mask to pixel group."""
        for i in range(len(mask)):
            if mask[i] == 1:
                # F1: flip LSB (0->1, 1->0, 2->3, 3->2, ...)
                group[i] = group[i] ^ 1
            elif mask[i] == -1:
                # F-1: flip and shift (0->255, 1->0, 2->1, ...)
                group[i] = (group[i] ^ 1) - 1
                if group[i] < 0:
                    group[i] = 255
        return group


# =============================================================================
# SAMPLE PAIR ANALYSIS (SPA)
# =============================================================================


class SPAAnalyzer:
    """
    Sample Pair Analysis for LSB steganography detection.

    Principle:
    - Analyzes pairs of adjacent pixels
    - Classifies pairs based on their difference
    - Estimates embedding rate from pair distribution

    Advantages:
    - Very accurate for small embedding rates
    - Works for images of any size

    References:
    - Dumitrescu et al. (2003): "Detection of LSB Steganography via Sample Pair Analysis"
    """

    def __init__(self):
        self.name = "Sample Pair Analysis (SPA)"

    def analyze(self, image: Image.Image) -> StegAnalysisResult:
        """Perform SPA analysis."""
        try:
            arr = np.array(image)

            if len(arr.shape) == 2:
                result = self._spa_channel(arr)
                results = {"grayscale": result}
                rates = [result["estimated_rate"]]
            else:
                results = {}
                rates = []

                for i, name in enumerate(["red", "green", "blue"]):
                    if i < arr.shape[2]:
                        result = self._spa_channel(arr[:, :, i])
                        results[name] = result
                        rates.append(result["estimated_rate"])

            avg_rate = np.mean(rates)

            # Convert to score
            if avg_rate > 0.4:
                score = 0.95
            elif avg_rate > 0.2:
                score = 0.8
            elif avg_rate > 0.1:
                score = 0.6
            elif avg_rate > 0.05:
                score = 0.4
            else:
                score = 0.2

            detected = avg_rate > 0.08
            confidence = StegConfidence.HIGH if avg_rate > 0.15 else StegConfidence.MEDIUM

            explanation = (
                f"SPA estimates {avg_rate*100:.1f}% message embedding. "
                f"{'Steganography likely detected.' if detected else 'No significant steganography.'}"
            )

            return StegAnalysisResult(
                method_name=self.name,
                score=score,
                confidence=confidence,
                detected=detected,
                estimated_payload=avg_rate * 100,
                explanation=explanation,
                details=results,
            )

        except Exception as e:
            logger.error(f"SPA analysis failed: {e}")
            return StegAnalysisResult(
                method_name=self.name,
                score=0.5,
                confidence=StegConfidence.VERY_LOW,
                detected=False,
                estimated_payload=0.0,
                explanation=f"Analysis failed: {str(e)}",
            )

    def _spa_channel(self, channel: np.ndarray) -> Dict[str, float]:
        """Perform SPA on a single channel."""
        h, w = channel.shape

        # Analyze horizontal pairs
        pairs_h = channel[:, :-1].flatten(), channel[:, 1:].flatten()

        # Analyze vertical pairs
        pairs_v = channel[:-1, :].flatten(), channel[1:, :].flatten()

        # Combine
        x = np.concatenate([pairs_h[0], pairs_v[0]])
        y = np.concatenate([pairs_h[1], pairs_v[1]])

        # Categorize pairs
        # Type X: x < y and both even, or x > y and both odd
        # Type Y: x < y and both odd, or x > y and both even
        # Type Z: x == y
        # Type W: |x - y| == 1

        x_even = (x % 2) == 0
        y_even = (y % 2) == 0

        type_X = np.sum(((x < y) & x_even & y_even) | ((x > y) & ~x_even & ~y_even))
        type_Y = np.sum(((x < y) & ~x_even & ~y_even) | ((x > y) & x_even & y_even))
        type_Z = np.sum(x == y)
        type_W = np.sum(np.abs(x.astype(int) - y.astype(int)) == 1)

        total = len(x)

        # Estimate embedding rate
        # In cover image: X ≈ Y
        # After embedding: X shifts towards Y
        if type_X + type_Y > 0:
            # Simplified estimation
            diff = abs(type_X - type_Y) / (type_X + type_Y)
            # Lower diff after embedding
            estimated_rate = 1 - diff
        else:
            estimated_rate = 0.0

        # Clamp to reasonable range
        estimated_rate = max(0, min(1, estimated_rate * 0.5))  # Scale down

        return {
            "type_X": int(type_X),
            "type_Y": int(type_Y),
            "type_Z": int(type_Z),
            "type_W": int(type_W),
            "total_pairs": total,
            "X_Y_ratio": type_X / (type_Y + 1e-10),
            "estimated_rate": float(estimated_rate),
        }


# =============================================================================
# HISTOGRAM ANALYSIS
# =============================================================================


class HistogramAnalyzer:
    """
    Histogram analysis for steganography detection.

    Principle:
    - LSB embedding "smooths" the histogram
    - Consecutive value pairs become more equal
    - Unnatural patterns in the distribution

    Detects:
    - Pairs of Values (PoV) attack
    - Histogram shifting
    """

    def __init__(self):
        self.name = "Histogram Analysis"

    def analyze(self, image: Image.Image) -> StegAnalysisResult:
        """Perform histogram analysis."""
        try:
            arr = np.array(image)

            if len(arr.shape) == 2:
                result = self._analyze_histogram(arr)
                results = {"grayscale": result}
                scores = [result["anomaly_score"]]
            else:
                results = {}
                scores = []

                for i, name in enumerate(["red", "green", "blue"]):
                    if i < arr.shape[2]:
                        result = self._analyze_histogram(arr[:, :, i])
                        results[name] = result
                        scores.append(result["anomaly_score"])

            avg_score = np.mean(scores)
            detected = avg_score > 0.5

            # Determine confidence
            score_std = np.std(scores) if len(scores) > 1 else 0
            if score_std < 0.1:
                confidence = StegConfidence.HIGH
            elif score_std < 0.2:
                confidence = StegConfidence.MEDIUM
            else:
                confidence = StegConfidence.LOW

            if detected:
                explanation = "Histogram shows anomalies consistent with LSB steganography."
            else:
                explanation = "Histogram appears natural, no significant steganography indicators."

            return StegAnalysisResult(
                method_name=self.name,
                score=avg_score,
                confidence=confidence,
                detected=detected,
                estimated_payload=avg_score * 40,
                explanation=explanation,
                details=results,
            )

        except Exception as e:
            logger.error(f"Histogram analysis failed: {e}")
            return StegAnalysisResult(
                method_name=self.name,
                score=0.5,
                confidence=StegConfidence.VERY_LOW,
                detected=False,
                estimated_payload=0.0,
                explanation=f"Analysis failed: {str(e)}",
            )

    def _analyze_histogram(self, channel: np.ndarray) -> Dict[str, float]:
        """Analyze histogram of a single channel."""
        hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)

        # 1. Pairs of Values analysis
        # Group histogram into pairs (0,1), (2,3), ...
        pairs = hist.reshape(-1, 2)
        pair_ratios = pairs[:, 0] / (pairs[:, 1] + 1e-10)

        # In natural images, pairs have varying ratios
        # After LSB embedding, ratios tend towards 1
        ratio_variance = np.var(pair_ratios)
        ratio_close_to_one = np.mean(np.abs(pair_ratios - 1) < 0.1)

        # 2. Smoothness of histogram
        hist_diff = np.abs(np.diff(hist))
        hist_smoothness = np.mean(hist_diff) / (np.std(hist) + 1e-10)

        # 3. Entropy
        hist_norm = hist / (np.sum(hist) + 1e-10)
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        max_entropy = np.log2(256)
        normalized_entropy = entropy / max_entropy

        # Combine into anomaly score
        anomaly_score = 0.0

        # Low variance in pair ratios is suspicious
        if ratio_variance < 0.5:
            anomaly_score += 0.3
        elif ratio_variance < 1.0:
            anomaly_score += 0.2

        # Many ratios close to 1 is suspicious
        if ratio_close_to_one > 0.5:
            anomaly_score += 0.3
        elif ratio_close_to_one > 0.3:
            anomaly_score += 0.2

        # Very high entropy can indicate embedding
        if normalized_entropy > 0.95:
            anomaly_score += 0.2
        elif normalized_entropy > 0.9:
            anomaly_score += 0.1

        # Smooth histogram can indicate embedding
        if hist_smoothness < 0.5:
            anomaly_score += 0.2

        return {
            "anomaly_score": min(1.0, anomaly_score),
            "ratio_variance": float(ratio_variance),
            "ratio_close_to_one": float(ratio_close_to_one),
            "normalized_entropy": float(normalized_entropy),
            "hist_smoothness": float(hist_smoothness),
        }


# =============================================================================
# DCT ANALYSIS (for JPEG steganography)
# =============================================================================


class DCTAnalyzer:
    """
    DCT analysis for JPEG steganography detection.

    Detects:
    - JSteg: hides data in DCT coefficient LSBs
    - F5: uses matrix embedding
    - OutGuess: avoids detectable modifications

    Principle:
    - Analyzes the distribution of DCT coefficients
    - Detects anomalies in AC coefficients
    """

    def __init__(self):
        self.name = "DCT Analysis (JPEG)"

    def analyze(self, image: Image.Image, raw_bytes: Optional[bytes] = None) -> StegAnalysisResult:
        """Perform DCT analysis for JPEG steganography."""
        try:
            # Check if JPEG
            if image.format != "JPEG" and not self._is_jpeg(raw_bytes):
                return StegAnalysisResult(
                    method_name=self.name,
                    score=0.0,
                    confidence=StegConfidence.HIGH,
                    detected=False,
                    estimated_payload=0.0,
                    explanation="Not a JPEG image. DCT analysis not applicable.",
                    details={"skipped": True, "reason": "not_jpeg"},
                )

            arr = np.array(image.convert("L"))

            # Perform block-based DCT analysis
            result = self._analyze_dct_blocks(arr)

            score = result["anomaly_score"]
            detected = score > 0.5

            if detected:
                explanation = (
                    f"DCT coefficient analysis shows anomalies. "
                    f"Possible JPEG steganography (JSteg/F5/OutGuess)."
                )
            else:
                explanation = "DCT coefficients appear normal. No JPEG steganography detected."

            return StegAnalysisResult(
                method_name=self.name,
                score=score,
                confidence=StegConfidence.MEDIUM,
                detected=detected,
                estimated_payload=score * 30,
                explanation=explanation,
                details=result,
            )

        except Exception as e:
            logger.error(f"DCT analysis failed: {e}")
            return StegAnalysisResult(
                method_name=self.name,
                score=0.5,
                confidence=StegConfidence.VERY_LOW,
                detected=False,
                estimated_payload=0.0,
                explanation=f"Analysis failed: {str(e)}",
            )

    def _is_jpeg(self, raw_bytes: Optional[bytes]) -> bool:
        """Check if data is JPEG."""
        if raw_bytes is None:
            return False
        return raw_bytes[:2] == b"\xff\xd8"

    def _analyze_dct_blocks(self, arr: np.ndarray) -> Dict[str, float]:
        """Analyze DCT coefficients in 8x8 blocks."""
        from scipy.fftpack import dct

        h, w = arr.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        arr = arr[:h8, :w8].astype(np.float64)

        # Collect all AC coefficients (excluding DC)
        ac_coeffs = []

        for i in range(0, h8, 8):
            for j in range(0, w8, 8):
                block = arr[i : i + 8, j : j + 8]
                dct_block = dct(dct(block.T, norm="ortho").T, norm="ortho")

                # Flatten and exclude DC (top-left)
                flat = dct_block.flatten()[1:]  # Skip DC
                ac_coeffs.extend(flat)

        ac_coeffs = np.array(ac_coeffs)

        # 1. Analyze distribution of AC coefficients
        # JSteg modifies LSBs of AC coefficients ≠ 0, 1
        non_zero_one = ac_coeffs[(np.abs(ac_coeffs) > 1)]

        if len(non_zero_one) > 0:
            # Check LSB distribution of quantized coefficients
            quantized = np.round(non_zero_one).astype(int)
            lsb = quantized & 1

            ones_ratio = np.mean(lsb)
            lsb_balance = min(ones_ratio, 1 - ones_ratio) * 2  # 1 = perfectly balanced
        else:
            lsb_balance = 0.5

        # 2. Histogram of AC coefficients
        hist, bins = np.histogram(ac_coeffs, bins=100, range=(-50, 50))
        hist = hist.astype(np.float64)

        # Check for "holes" in histogram (F5 characteristic)
        # F5 creates histogram with reduced values at certain positions
        holes = np.sum(hist[1:-1] == 0)
        hole_ratio = holes / len(hist)

        # 3. Check coefficient pairs
        # JSteg creates pairs with equal frequency
        pairs = hist.reshape(-1, 2)
        pair_balance = np.mean(np.abs(pairs[:, 0] - pairs[:, 1]) / (pairs.sum(axis=1) + 1e-10))

        # Combine into anomaly score
        anomaly_score = 0.0

        # Very balanced LSB is suspicious
        if lsb_balance > 0.95:
            anomaly_score += 0.4
        elif lsb_balance > 0.9:
            anomaly_score += 0.2

        # Holes in histogram (F5)
        if hole_ratio > 0.1:
            anomaly_score += 0.3
        elif hole_ratio > 0.05:
            anomaly_score += 0.15

        # Balanced pairs (JSteg)
        if pair_balance < 0.1:
            anomaly_score += 0.3
        elif pair_balance < 0.2:
            anomaly_score += 0.15

        return {
            "anomaly_score": min(1.0, anomaly_score),
            "lsb_balance": float(lsb_balance),
            "hole_ratio": float(hole_ratio),
            "pair_balance": float(pair_balance),
            "n_ac_coefficients": len(ac_coeffs),
        }


# =============================================================================
# VISUAL ATTACK (Bit Plane Visualization)
# =============================================================================


class VisualAttackAnalyzer:
    """
    Visual Attack - visualization of bit planes.

    This is not an automatic detection method, but generates images
    of bit planes for visual inspection.

    The LSB plane of an image with steganography shows regular
    patterns or "noise" different from natural noise.
    """

    def __init__(self):
        self.name = "Visual Attack (Bit Planes)"

    def analyze(self, image: Image.Image) -> StegAnalysisResult:
        """Generate bit plane visualizations."""
        try:
            arr = np.array(image.convert("L"))

            # Extract each bit plane
            bit_planes = []
            for bit in range(8):
                plane = ((arr >> bit) & 1) * 255
                bit_planes.append((f"bit_{bit}", plane))

            # Analyze LSB plane for patterns
            lsb_plane = bit_planes[0][1]
            pattern_score = self._analyze_lsb_patterns(lsb_plane)

            detected = pattern_score > 0.5

            if detected:
                explanation = "LSB plane shows regular patterns that may indicate steganography."
            else:
                explanation = "LSB plane appears to contain natural noise patterns."

            return StegAnalysisResult(
                method_name=self.name,
                score=pattern_score,
                confidence=StegConfidence.LOW,  # Visual analysis is subjective
                detected=detected,
                estimated_payload=pattern_score * 30,
                explanation=explanation,
                details={"pattern_score": pattern_score},
                artifacts=bit_planes,
            )

        except Exception as e:
            logger.error(f"Visual attack analysis failed: {e}")
            return StegAnalysisResult(
                method_name=self.name,
                score=0.5,
                confidence=StegConfidence.VERY_LOW,
                detected=False,
                estimated_payload=0.0,
                explanation=f"Analysis failed: {str(e)}",
            )

    def _analyze_lsb_patterns(self, lsb_plane: np.ndarray) -> float:
        """Analyze LSB plane for unnatural patterns."""
        # Convert to binary
        binary = (lsb_plane > 127).astype(np.float64)

        # 1. Check for horizontal runs
        h_runs = self._count_runs(binary, axis=1)

        # 2. Check for vertical runs
        v_runs = self._count_runs(binary, axis=0)

        # 3. Block-based variance
        block_size = 16
        h, w = binary.shape
        variances = []

        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = binary[i : i + block_size, j : j + block_size]
                variances.append(np.var(block))

        avg_variance = np.mean(variances) if variances else 0.25

        # In natural images, LSB has high variance (~0.25)
        # In steganography, variance may be different
        variance_deviation = abs(avg_variance - 0.25)

        # Combine metrics
        score = 0.0

        if variance_deviation < 0.02:
            score += 0.3  # Very close to random
        elif variance_deviation > 0.1:
            score += 0.3  # Too structured

        # Run analysis
        if h_runs < 0.4 or v_runs < 0.4:
            score += 0.2

        return min(1.0, score)

    def _count_runs(self, binary: np.ndarray, axis: int) -> float:
        """Count average run length along axis."""
        if axis == 1:
            data = binary
        else:
            data = binary.T

        total_runs = 0
        total_elements = 0

        for row in data:
            diff = np.diff(row)
            runs = np.sum(diff != 0) + 1
            total_runs += runs
            total_elements += len(row)

        avg_run_length = total_elements / (total_runs + 1)
        # Normalize: expected run length in random data is ~2
        return avg_run_length / 2


# =============================================================================
# MAIN STEGANOGRAPHY DETECTOR
# =============================================================================


class SteganographyDetector:
    """
    Main steganography detection class.

    Combines multiple analysis methods for comprehensive detection.
    """

    def __init__(
        self,
        enable_lsb: bool = True,
        enable_chi_square: bool = True,
        enable_rs: bool = True,
        enable_spa: bool = True,
        enable_histogram: bool = True,
        enable_dct: bool = True,
        enable_visual: bool = False,  # Disabled by default (slow)
    ):
        self.analyzers = []

        if enable_lsb:
            self.analyzers.append(LSBAnalyzer())
        if enable_chi_square:
            self.analyzers.append(ChiSquareAnalyzer())
        if enable_rs:
            self.analyzers.append(RSAnalyzer())
        if enable_spa:
            self.analyzers.append(SPAAnalyzer())
        if enable_histogram:
            self.analyzers.append(HistogramAnalyzer())
        if enable_dct:
            self.analyzers.append(DCTAnalyzer())
        if enable_visual:
            self.analyzers.append(VisualAttackAnalyzer())

    def analyze(
        self,
        image: Union[Image.Image, Path, str],
        raw_bytes: Optional[bytes] = None,
    ) -> StegDetectionReport:
        """
        Perform comprehensive steganography analysis.

        Args:
            image: PIL Image or path to image file
            raw_bytes: Optional raw file bytes for format-specific analysis

        Returns:
            StegDetectionReport with all analysis results
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = str(image)
            with open(image_path, "rb") as f:
                raw_bytes = f.read()
            image = Image.open(io.BytesIO(raw_bytes))
        else:
            image_path = None

        # Get image info
        image_size = image.size
        image_format = image.format or "UNKNOWN"

        # Run all analyzers
        results: List[StegAnalysisResult] = []

        for analyzer in self.analyzers:
            logger.info(f"Running {analyzer.name}...")

            if isinstance(analyzer, DCTAnalyzer):
                result = analyzer.analyze(image, raw_bytes)
            else:
                result = analyzer.analyze(image)

            results.append(result)

        # Combine results
        overall_score, overall_confidence = self._combine_results(results)
        steg_detected = overall_score > 0.5

        # Determine likely method
        likely_method = self._determine_likely_method(results, image_format)

        # Calculate estimated payload
        payloads = [r.estimated_payload for r in results if r.estimated_payload > 0]
        estimated_payload = np.mean(payloads) if payloads else 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(steg_detected, results)

        # List limitations
        limitations = [
            "Detection is probabilistic, not definitive proof",
            "Advanced steganography methods may evade detection",
            "Results depend on image quality and format",
            "False positives possible with certain image types (e.g., synthetic images)",
        ]

        return StegDetectionReport(
            image_path=image_path,
            image_size=image_size,
            image_format=image_format,
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            steg_detected=steg_detected,
            likely_method=likely_method,
            estimated_payload_percent=estimated_payload,
            analysis_results=results,
            recommendations=recommendations,
            limitations=limitations,
        )

    def _combine_results(
        self, results: List[StegAnalysisResult]
    ) -> Tuple[float, StegConfidence]:
        """Combine results from multiple analyzers."""
        if not results:
            return 0.0, StegConfidence.VERY_LOW

        # Weighted average based on confidence
        weights = {
            StegConfidence.VERY_LOW: 0.2,
            StegConfidence.LOW: 0.4,
            StegConfidence.MEDIUM: 0.6,
            StegConfidence.HIGH: 0.8,
            StegConfidence.VERY_HIGH: 1.0,
        }

        weighted_sum = 0.0
        weight_total = 0.0

        for result in results:
            w = weights.get(result.confidence, 0.5)
            weighted_sum += result.score * w
            weight_total += w

        overall_score = weighted_sum / (weight_total + 1e-10)

        # Determine overall confidence
        detection_count = sum(1 for r in results if r.detected)
        detection_ratio = detection_count / len(results)

        if detection_ratio > 0.7 and overall_score > 0.7:
            overall_confidence = StegConfidence.HIGH
        elif detection_ratio > 0.5 and overall_score > 0.5:
            overall_confidence = StegConfidence.MEDIUM
        elif detection_ratio > 0.3:
            overall_confidence = StegConfidence.LOW
        else:
            overall_confidence = StegConfidence.VERY_LOW

        return overall_score, overall_confidence

    def _determine_likely_method(
        self, results: List[StegAnalysisResult], image_format: str
    ) -> StegMethod:
        """Try to determine the likely steganography method used."""
        # Check for JPEG-specific methods
        dct_result = next((r for r in results if "DCT" in r.method_name), None)
        if dct_result and dct_result.detected:
            details = dct_result.details
            if details.get("lsb_balance", 0) > 0.9:
                return StegMethod.JSTEG
            elif details.get("hole_ratio", 0) > 0.05:
                return StegMethod.F5

        # Check LSB methods
        lsb_result = next((r for r in results if "LSB" in r.method_name), None)
        rs_result = next((r for r in results if "RS" in r.method_name), None)

        if lsb_result and lsb_result.detected:
            if rs_result and rs_result.details.get("estimated_rate", 0) > 0.2:
                return StegMethod.LSB_REPLACEMENT
            else:
                return StegMethod.LSB_MATCHING

        return StegMethod.UNKNOWN

    def _generate_recommendations(
        self, detected: bool, results: List[StegAnalysisResult]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if detected:
            recommendations.append(
                "Steganography indicators detected. Treat image as potentially containing hidden data."
            )
            recommendations.append(
                "Consider using specialized steganalysis tools for deeper investigation."
            )
            recommendations.append(
                "If forensic evidence is needed, preserve the original file without modification."
            )

            # Specific recommendations based on results
            high_confidence = [r for r in results if r.confidence in [StegConfidence.HIGH, StegConfidence.VERY_HIGH]]
            if high_confidence:
                methods = ", ".join(r.method_name for r in high_confidence)
                recommendations.append(
                    f"High-confidence detection from: {methods}"
                )
        else:
            recommendations.append(
                "No strong steganography indicators found."
            )
            recommendations.append(
                "Note: Advanced steganography methods may not be detected."
            )
            recommendations.append(
                "If suspicion remains, consider manual inspection of bit planes."
            )

        return recommendations


# =============================================================================
# PLUGIN INTEGRATION
# =============================================================================


# For integration with the ForensicsPlugin system
def create_steganography_plugin():
    """Create a steganography forensics plugin."""
    from .base import ForensicsPlugin, ForensicsResult, PluginCategory, Confidence, Artifact

    class SteganographyPlugin(ForensicsPlugin):
        plugin_id = "steganography_detector"
        plugin_name = "Steganography Detection"
        category = PluginCategory.TAMPERING
        version = "1.0.0"

        def __init__(self):
            self.detector = SteganographyDetector()

        def analyze(
            self,
            image: Image.Image,
            image_path: Optional[Path] = None,
            raw_bytes: Optional[bytes] = None,
        ) -> ForensicsResult:
            """Run steganography analysis."""
            report = self.detector.analyze(image, raw_bytes)

            # Convert to ForensicsResult
            if report.overall_confidence == StegConfidence.VERY_HIGH:
                confidence = Confidence.VERY_HIGH
            elif report.overall_confidence == StegConfidence.HIGH:
                confidence = Confidence.HIGH
            elif report.overall_confidence == StegConfidence.MEDIUM:
                confidence = Confidence.MEDIUM
            elif report.overall_confidence == StegConfidence.LOW:
                confidence = Confidence.LOW
            else:
                confidence = Confidence.VERY_LOW

            # Create artifacts from visual analysis
            artifacts = []
            for result in report.analysis_results:
                for name, data in result.artifacts:
                    artifacts.append(
                        Artifact(
                            name=name,
                            artifact_type="heatmap",
                            data=data,
                            description=f"Bit plane visualization: {name}",
                        )
                    )

            return ForensicsResult(
                plugin_id=self.plugin_id,
                score=report.overall_score,
                confidence=confidence,
                detected=report.steg_detected,
                explanation=(
                    f"Steganography {'detected' if report.steg_detected else 'not detected'}. "
                    f"Likely method: {report.likely_method.value}. "
                    f"Estimated payload: {report.estimated_payload_percent:.1f}%"
                ),
                details=report.to_dict(),
                limitations=report.limitations,
                artifacts=artifacts,
            )

    return SteganographyPlugin


# =============================================================================
# CLI / STANDALONE USAGE
# =============================================================================


def main():
    """Command-line interface for steganography detection."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Detect steganography in images"
    )
    parser.add_argument("image", type=Path, help="Path to image file")
    parser.add_argument(
        "--output", "-o", type=Path, help="Output JSON file"
    )
    parser.add_argument(
        "--visual", "-v", action="store_true", help="Enable visual attack analysis"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Create detector
    detector = SteganographyDetector(enable_visual=args.visual)

    # Analyze
    print(f"Analyzing: {args.image}")
    report = detector.analyze(args.image)

    # Print results
    print("\n" + "=" * 60)
    print("STEGANOGRAPHY ANALYSIS REPORT")
    print("=" * 60)
    print(f"Image: {report.image_path}")
    print(f"Size: {report.image_size}")
    print(f"Format: {report.image_format}")
    print()
    print(f"Overall Score: {report.overall_score:.3f}")
    print(f"Confidence: {report.overall_confidence.name}")
    print(f"Steganography Detected: {'YES' if report.steg_detected else 'NO'}")
    print(f"Likely Method: {report.likely_method.value}")
    print(f"Estimated Payload: {report.estimated_payload_percent:.1f}%")
    print()
    print("Analysis Results:")
    print("-" * 40)
    for result in report.analysis_results:
        status = "✓" if result.detected else "✗"
        print(
            f"  {status} {result.method_name}: score={result.score:.3f}, "
            f"conf={result.confidence.name}"
        )
    print()
    print("Recommendations:")
    for rec in report.recommendations:
        print(f"  • {rec}")

    # Save JSON if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
