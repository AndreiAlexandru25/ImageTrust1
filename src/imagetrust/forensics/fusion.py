"""
Fusion/Decision Layer.

Combines results from all forensics detectors into:
- Multi-label verdicts
- Overall authenticity assessment
- Top evidence summary
- Confidence-calibrated outputs

Designed for truthfulness - never claims certainty without evidence.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from imagetrust.forensics.base import (
    Confidence,
    ForensicsResult,
    PluginCategory,
)
from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


class VerdictLabel(Enum):
    """Multi-label verdict categories."""
    CAMERA_ORIGINAL = "camera_original_likely"
    RECOMPRESSED = "recompressed_likely"
    SCREENSHOT = "screenshot_likely"
    SOCIAL_MEDIA = "social_media_likely"
    EDITED = "edited_likely"
    LOCAL_TAMPER = "local_tamper_suspected"
    AI_GENERATED = "ai_generated_suspected"
    METADATA_STRIPPED = "metadata_stripped"
    UNKNOWN = "unknown"


@dataclass
class LabelScore:
    """Score for a single verdict label."""
    label: VerdictLabel
    probability: float  # 0-1
    confidence: Confidence
    evidence: List[str]
    against_evidence: List[str] = field(default_factory=list)


@dataclass
class ForensicsVerdict:
    """
    Final forensics verdict combining all analysis results.

    Designed for truthfulness:
    - Multi-label (image can be both screenshot AND edited)
    - Calibrated probabilities
    - Clear evidence for each claim
    - Explicit uncertainty when appropriate
    """

    # Multi-label results
    labels: List[LabelScore]

    # Primary verdict (highest confidence label)
    primary_verdict: VerdictLabel
    primary_confidence: Confidence
    primary_probability: float

    # Overall authenticity score (0=definitely manipulated, 1=likely authentic)
    authenticity_score: float

    # Evidence summary
    top_evidence: List[str]  # Top 5 strongest signals
    contradictions: List[str]  # Conflicting signals
    what_would_help: List[str]  # What could increase confidence

    # Detailed breakdown
    category_summaries: Dict[str, str]

    # Metadata
    total_detectors_run: int
    total_processing_time_ms: float
    inconclusive: bool  # True if no strong signals

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "labels": [
                {
                    "label": ls.label.value,
                    "probability": ls.probability,
                    "confidence": ls.confidence.name,
                    "evidence": ls.evidence,
                    "against_evidence": ls.against_evidence,
                }
                for ls in self.labels
            ],
            "primary_verdict": self.primary_verdict.value,
            "primary_confidence": self.primary_confidence.name,
            "primary_probability": self.primary_probability,
            "authenticity_score": self.authenticity_score,
            "top_evidence": self.top_evidence,
            "contradictions": self.contradictions,
            "what_would_help": self.what_would_help,
            "category_summaries": self.category_summaries,
            "total_detectors_run": self.total_detectors_run,
            "total_processing_time_ms": self.total_processing_time_ms,
            "inconclusive": self.inconclusive,
        }

    @property
    def summary(self) -> str:
        """One-line summary."""
        if self.inconclusive:
            return f"Inconclusive - insufficient evidence (authenticity: {self.authenticity_score:.2f})"
        return (
            f"{self.primary_verdict.value} ({self.primary_confidence.name}, "
            f"p={self.primary_probability:.2f}, authenticity={self.authenticity_score:.2f})"
        )


class FusionLayer:
    """
    Combines forensics results into coherent verdicts.

    Uses weighted evidence aggregation with:
    - Category-specific weights
    - Confidence calibration
    - Contradiction detection
    - Uncertainty propagation
    """

    # Default weights for each category
    DEFAULT_WEIGHTS = {
        PluginCategory.PIXEL: 0.25,
        PluginCategory.METADATA: 0.25,
        PluginCategory.SOURCE: 0.20,
        PluginCategory.AI_DETECTION: 0.25,
        PluginCategory.TAMPERING: 0.05,
    }

    # Thresholds for label assignment
    DETECTION_THRESHOLD = 0.4
    HIGH_CONFIDENCE_THRESHOLD = 0.7

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fusion layer.

        Args:
            config: Optional configuration with custom weights/thresholds
        """
        self.config = config or {}
        self.weights = self.config.get("weights", self.DEFAULT_WEIGHTS)

    def fuse(self, results: List[ForensicsResult]) -> ForensicsVerdict:
        """
        Fuse multiple detector results into final verdict.

        Args:
            results: List of ForensicsResult from various detectors

        Returns:
            ForensicsVerdict with multi-label assessment
        """
        if not results:
            return self._create_empty_verdict()

        # Group results by category
        by_category = self._group_by_category(results)

        # Calculate label scores
        label_scores = self._calculate_label_scores(results, by_category)

        # Determine primary verdict
        primary = self._determine_primary_verdict(label_scores)

        # Calculate authenticity score
        authenticity = self._calculate_authenticity_score(label_scores, results)

        # Extract top evidence
        top_evidence = self._extract_top_evidence(results)

        # Detect contradictions
        contradictions = self._detect_contradictions(results, label_scores)

        # Generate suggestions
        what_would_help = self._generate_suggestions(label_scores, results)

        # Category summaries
        category_summaries = self._generate_category_summaries(by_category)

        # Check if inconclusive
        inconclusive = self._is_inconclusive(label_scores)

        # Calculate total processing time
        total_time = sum(r.processing_time_ms for r in results)

        return ForensicsVerdict(
            labels=label_scores,
            primary_verdict=primary[0],
            primary_confidence=primary[1],
            primary_probability=primary[2],
            authenticity_score=authenticity,
            top_evidence=top_evidence,
            contradictions=contradictions,
            what_would_help=what_would_help,
            category_summaries=category_summaries,
            total_detectors_run=len(results),
            total_processing_time_ms=total_time,
            inconclusive=inconclusive,
        )

    def _group_by_category(self, results: List[ForensicsResult]) -> Dict[PluginCategory, List[ForensicsResult]]:
        """Group results by plugin category."""
        groups = {}
        for r in results:
            if r.category not in groups:
                groups[r.category] = []
            groups[r.category].append(r)
        return groups

    def _calculate_label_scores(
        self,
        results: List[ForensicsResult],
        by_category: Dict[PluginCategory, List[ForensicsResult]],
    ) -> List[LabelScore]:
        """Calculate probability for each verdict label."""
        label_scores = []

        # 1. Camera Original
        camera_evidence = []
        camera_against = []
        camera_prob = 0.5  # Neutral prior

        # Check metadata for camera info
        for r in by_category.get(PluginCategory.METADATA, []):
            if r.details.get("has_camera_info"):
                camera_prob += 0.2
                camera_evidence.append("Camera info present in metadata")
            if r.details.get("has_exposure"):
                camera_prob += 0.1
                camera_evidence.append("Exposure settings present")
            if r.details.get("metadata_stripped"):
                camera_prob -= 0.3
                camera_against.append("Metadata appears stripped")

        # Check for signs against
        for r in by_category.get(PluginCategory.PIXEL, []):
            if r.detected and r.plugin_id == "ela_detector":
                camera_prob -= 0.2
                camera_against.append("ELA shows editing artifacts")

        camera_prob = max(0, min(1, camera_prob))
        camera_conf = Confidence.from_score(abs(camera_prob - 0.5) * 2)

        label_scores.append(LabelScore(
            label=VerdictLabel.CAMERA_ORIGINAL,
            probability=camera_prob,
            confidence=camera_conf,
            evidence=camera_evidence,
            against_evidence=camera_against,
        ))

        # 2. Recompressed
        recomp_evidence = []
        recomp_against = []
        recomp_prob = 0.0

        for r in by_category.get(PluginCategory.PIXEL, []):
            if r.plugin_id == "jpeg_artifacts":
                if r.details.get("double_jpeg_score", 0) > 0.5:
                    recomp_prob += 0.5
                    recomp_evidence.append("Double JPEG compression detected")
                elif r.details.get("blocking_strength", 0) > 2:
                    recomp_prob += 0.2
                    recomp_evidence.append("JPEG blocking artifacts present")

        for r in by_category.get(PluginCategory.SOURCE, []):
            if r.plugin_id == "compression_history" and r.detected:
                recomp_prob += 0.3
                recomp_evidence.append(r.explanation)

        recomp_prob = min(1, recomp_prob)
        label_scores.append(LabelScore(
            label=VerdictLabel.RECOMPRESSED,
            probability=recomp_prob,
            confidence=Confidence.from_score(recomp_prob) if recomp_prob > 0.3 else Confidence.LOW,
            evidence=recomp_evidence,
            against_evidence=recomp_against,
        ))

        # 3. Screenshot
        screenshot_evidence = []
        screenshot_prob = 0.0

        for r in by_category.get(PluginCategory.SOURCE, []):
            if r.plugin_id == "screenshot_detector" and r.detected:
                screenshot_prob = r.score
                screenshot_evidence.extend(r.details.get("indicators", []))

        label_scores.append(LabelScore(
            label=VerdictLabel.SCREENSHOT,
            probability=screenshot_prob,
            confidence=Confidence.from_score(screenshot_prob) if screenshot_prob > 0.3 else Confidence.LOW,
            evidence=screenshot_evidence,
            against_evidence=[],
        ))

        # 4. Social Media
        social_evidence = []
        social_prob = 0.0

        for r in by_category.get(PluginCategory.SOURCE, []):
            if r.plugin_id == "platform_detector" and r.detected:
                social_prob = r.score
                platform = r.details.get("best_platform", "unknown")
                social_evidence.append(f"Platform match: {platform}")

        label_scores.append(LabelScore(
            label=VerdictLabel.SOCIAL_MEDIA,
            probability=social_prob,
            confidence=Confidence.from_score(social_prob) if social_prob > 0.3 else Confidence.LOW,
            evidence=social_evidence,
            against_evidence=[],
        ))

        # 5. Edited
        edited_evidence = []
        edited_against = []
        edited_prob = 0.0

        for r in by_category.get(PluginCategory.METADATA, []):
            if r.plugin_id == "software_traces" and r.detected:
                sw_info = r.details.get("software_info", {})
                if sw_info.get("category") in ["professional_editor", "mobile_editor"]:
                    edited_prob += 0.4
                    edited_evidence.append(f"Editing software: {sw_info.get('name', 'unknown')}")

        for r in by_category.get(PluginCategory.PIXEL, []):
            if r.plugin_id == "ela_detector" and r.detected:
                edited_prob += 0.3
                edited_evidence.append("ELA shows editing artifacts")

        edited_prob = min(1, edited_prob)
        label_scores.append(LabelScore(
            label=VerdictLabel.EDITED,
            probability=edited_prob,
            confidence=Confidence.from_score(edited_prob) if edited_prob > 0.3 else Confidence.LOW,
            evidence=edited_evidence,
            against_evidence=edited_against,
        ))

        # 6. Local Tamper
        tamper_evidence = []
        tamper_prob = 0.0

        for r in by_category.get(PluginCategory.PIXEL, []):
            if r.plugin_id == "noise_inconsistency" and r.detected:
                tamper_prob += 0.4
                tamper_evidence.append("Noise inconsistency detected")
            if r.plugin_id == "ela_detector" and r.score > 0.5:
                tamper_prob += 0.3
                tamper_evidence.append("ELA shows localized anomalies")

        tamper_prob = min(1, tamper_prob)
        label_scores.append(LabelScore(
            label=VerdictLabel.LOCAL_TAMPER,
            probability=tamper_prob,
            confidence=Confidence.from_score(tamper_prob) if tamper_prob > 0.3 else Confidence.LOW,
            evidence=tamper_evidence,
            against_evidence=[],
        ))

        # 7. AI Generated
        ai_evidence = []
        ai_prob = 0.0

        for r in by_category.get(PluginCategory.AI_DETECTION, []):
            if r.detected:
                ai_prob = max(ai_prob, r.score)
                ai_evidence.append(r.explanation)

        for r in by_category.get(PluginCategory.METADATA, []):
            if r.plugin_id == "software_traces":
                sw_info = r.details.get("software_info", {})
                if sw_info.get("category") == "ai_generator":
                    ai_prob = max(ai_prob, 0.9)
                    ai_evidence.append(f"AI generator detected: {sw_info.get('name')}")

        label_scores.append(LabelScore(
            label=VerdictLabel.AI_GENERATED,
            probability=ai_prob,
            confidence=Confidence.from_score(ai_prob) if ai_prob > 0.3 else Confidence.LOW,
            evidence=ai_evidence,
            against_evidence=[],
        ))

        # 8. Metadata Stripped
        stripped_evidence = []
        stripped_prob = 0.0

        for r in by_category.get(PluginCategory.METADATA, []):
            if r.details.get("metadata_stripped"):
                stripped_prob = 0.8
                stripped_evidence.append("No metadata found")

        label_scores.append(LabelScore(
            label=VerdictLabel.METADATA_STRIPPED,
            probability=stripped_prob,
            confidence=Confidence.HIGH if stripped_prob > 0.5 else Confidence.LOW,
            evidence=stripped_evidence,
            against_evidence=[],
        ))

        return label_scores

    def _determine_primary_verdict(
        self,
        label_scores: List[LabelScore],
    ) -> Tuple[VerdictLabel, Confidence, float]:
        """Determine the primary (most likely) verdict."""
        # Filter to detected labels
        detected = [ls for ls in label_scores if ls.probability > self.DETECTION_THRESHOLD]

        if not detected:
            return VerdictLabel.UNKNOWN, Confidence.LOW, 0.5

        # Sort by probability
        detected.sort(key=lambda x: x.probability, reverse=True)
        primary = detected[0]

        return primary.label, primary.confidence, primary.probability

    def _calculate_authenticity_score(
        self,
        label_scores: List[LabelScore],
        results: List[ForensicsResult],
    ) -> float:
        """
        Calculate overall authenticity score.

        1.0 = likely authentic camera image
        0.0 = definitely manipulated/synthetic
        """
        # Start neutral
        score = 0.5

        for ls in label_scores:
            if ls.label == VerdictLabel.CAMERA_ORIGINAL:
                score += (ls.probability - 0.5) * 0.3
            elif ls.label == VerdictLabel.AI_GENERATED:
                score -= ls.probability * 0.4
            elif ls.label == VerdictLabel.LOCAL_TAMPER:
                score -= ls.probability * 0.3
            elif ls.label == VerdictLabel.EDITED:
                score -= ls.probability * 0.2
            elif ls.label in [VerdictLabel.SCREENSHOT, VerdictLabel.SOCIAL_MEDIA]:
                score -= ls.probability * 0.1  # Not manipulation, but processing

        return max(0, min(1, score))

    def _extract_top_evidence(self, results: List[ForensicsResult], top_n: int = 5) -> List[str]:
        """Extract the strongest evidence points."""
        evidence_items = []

        for r in results:
            if r.detected and r.confidence.value >= Confidence.MEDIUM.value:
                evidence_items.append((
                    r.confidence.value * r.score,
                    r.explanation
                ))

        # Sort by strength
        evidence_items.sort(key=lambda x: x[0], reverse=True)

        return [e[1] for e in evidence_items[:top_n]]

    def _detect_contradictions(
        self,
        results: List[ForensicsResult],
        label_scores: List[LabelScore],
    ) -> List[str]:
        """Detect contradictory signals."""
        contradictions = []

        # Check for contradictory labels
        camera_score = next((ls.probability for ls in label_scores if ls.label == VerdictLabel.CAMERA_ORIGINAL), 0)
        ai_score = next((ls.probability for ls in label_scores if ls.label == VerdictLabel.AI_GENERATED), 0)

        if camera_score > 0.5 and ai_score > 0.5:
            contradictions.append("Conflicting signals: metadata suggests camera, but AI indicators present")

        # Check for screenshot + camera original
        screenshot_score = next((ls.probability for ls in label_scores if ls.label == VerdictLabel.SCREENSHOT), 0)
        if screenshot_score > 0.5 and camera_score > 0.5:
            contradictions.append("Conflicting: screenshot dimensions but camera metadata")

        return contradictions

    def _generate_suggestions(
        self,
        label_scores: List[LabelScore],
        results: List[ForensicsResult],
    ) -> List[str]:
        """Generate suggestions for increasing confidence."""
        suggestions = []

        # Check what's missing
        categories_present = {r.category for r in results}

        if PluginCategory.METADATA not in categories_present:
            suggestions.append("Analyze original file with metadata (not screenshot/export)")

        # Check for low confidence verdicts
        primary_score = max(ls.probability for ls in label_scores)
        if primary_score < 0.6:
            suggestions.append("Higher resolution original would improve detection")

        # General suggestions
        any_detected = any(r.detected for r in results)
        if not any_detected:
            suggestions.append("No strong signals - original camera file would help confirm authenticity")

        return suggestions

    def _generate_category_summaries(
        self,
        by_category: Dict[PluginCategory, List[ForensicsResult]],
    ) -> Dict[str, str]:
        """Generate summary for each category."""
        summaries = {}

        for category, results in by_category.items():
            detected_count = sum(1 for r in results if r.detected)

            if detected_count == 0:
                summaries[category.value] = "No issues detected"
            else:
                explanations = [r.explanation for r in results if r.detected]
                summaries[category.value] = "; ".join(explanations[:2])

        return summaries

    def _is_inconclusive(self, label_scores: List[LabelScore]) -> bool:
        """Check if verdict is inconclusive."""
        # Inconclusive if no label has high probability
        max_prob = max(ls.probability for ls in label_scores)
        return max_prob < self.DETECTION_THRESHOLD

    def _create_empty_verdict(self) -> ForensicsVerdict:
        """Create verdict when no results available."""
        return ForensicsVerdict(
            labels=[],
            primary_verdict=VerdictLabel.UNKNOWN,
            primary_confidence=Confidence.VERY_LOW,
            primary_probability=0.0,
            authenticity_score=0.5,
            top_evidence=[],
            contradictions=[],
            what_would_help=["Run forensics analysis first"],
            category_summaries={},
            total_detectors_run=0,
            total_processing_time_ms=0.0,
            inconclusive=True,
        )
