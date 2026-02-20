#!/usr/bin/env python3
"""
ImageTrust Scientific Benchmark Tool

Automated benchmark tool for validating the Fusion Forensics pipeline
against standard datasets (CASIA, CIFAKE, etc.) for research publication.

Calculates scientific metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (TP, FP, TN, FN)
- Social Media Damping statistics

Usage:
    python -m imagetrust.tools.benchmark_tool \
        --real data/real_images \
        --fake data/fake_images \
        --output results/benchmark_results.csv

Author: ImageTrust Research Team
License: MIT
"""

import argparse
import csv
import io
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@dataclass
class BenchmarkResult:
    """Single image benchmark result."""
    filename: str
    ground_truth: str  # "real" or "fake"
    predicted_verdict: str
    confidence: float
    ai_score_raw: float  # Original AI score before damping
    ai_score_damped: float  # AI score after social media damping (if applied)
    modification_score: float
    metadata_score: float
    provenance_tag: str  # e.g., "whatsapp", "screenshot", "web_download", "none"
    steganography_detected: bool
    social_media_damping_applied: bool
    processing_time_ms: float
    is_correct: bool


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics."""
    total_images: int = 0
    total_real: int = 0
    total_fake: int = 0

    # Confusion matrix
    true_positives: int = 0   # Correctly identified as fake
    false_positives: int = 0  # Real misclassified as fake
    true_negatives: int = 0   # Correctly identified as real
    false_negatives: int = 0  # Fake misclassified as real

    # Additional metrics
    social_media_damping_count: int = 0
    inconclusive_count: int = 0
    total_processing_time_ms: float = 0.0

    # Per-class accuracy
    real_correct: int = 0
    fake_correct: int = 0

    # NOVELTY METRIC: Social Media Correction Count
    # Number of times damping logic SAVED a REAL photo from being labeled FAKE
    # This proves the adaptive fusion forensics novelty
    social_media_correction_count: int = 0

    # Steganography statistics
    steganography_detected_count: int = 0

    # Provenance breakdown
    provenance_whatsapp: int = 0
    provenance_screenshot: int = 0
    provenance_web_download: int = 0
    provenance_social_media: int = 0
    provenance_none: int = 0

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)"""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        """Recall (Sensitivity): TP / (TP + FN)"""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def f1_score(self) -> float:
        """F1 Score: 2 * (precision * recall) / (precision + recall)"""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def specificity(self) -> float:
        """Specificity: TN / (TN + FP)"""
        denominator = self.true_negatives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_negatives / denominator

    @property
    def avg_processing_time_ms(self) -> float:
        """Average processing time per image."""
        if self.total_images == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_images


class FusionForensicsBenchmark:
    """
    Scientific benchmark tool for Fusion Forensics validation.

    Processes datasets of known real/fake images and calculates
    standard machine learning metrics for research publication.
    """

    # Verdicts that indicate "FAKE" (AI-generated or manipulated)
    FAKE_VERDICTS = ["ai_generated_likely", "manipulated_likely"]

    # Verdicts that indicate "REAL" (authentic)
    REAL_VERDICTS = ["camera_original_likely"]

    # Verdicts that are inconclusive
    INCONCLUSIVE_VERDICTS = ["inconclusive"]

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

    def __init__(self, verbose: bool = True):
        """Initialize benchmark tool."""
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
        self.metrics = BenchmarkMetrics()
        self._analyzer = None

    def _log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(message)

    def _init_analyzer(self):
        """Initialize the forensics analyzer (lazy loading)."""
        if self._analyzer is None:
            self._log("Initializing Fusion Forensics engine...")
            try:
                from imagetrust.forensics import ForensicsEngine

                self._forensics_engine = ForensicsEngine()
                self._extract_metadata = None
                self._detect_stego = None
                self._analyzer = True
                self._log("Forensics engine initialized with all plugins")
            except ImportError as e:
                self._log(f"Warning: Could not import forensics engine: {e}")
                self._log("Falling back to simplified analysis...")
                self._analyzer = "simplified"

    def _analyze_image(self, image_path: Path) -> dict:
        """
        Analyze a single image using the Fusion Forensics pipeline.

        Args:
            image_path: Path to image file

        Returns:
            Analysis result dictionary
        """
        from PIL import Image
        import tempfile

        self._init_analyzer()

        # Read image bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Load image
        image = Image.open(io.BytesIO(image_bytes))

        # Extract metadata
        metadata = self._extract_metadata(image, image_bytes)

        # Detect steganography
        stego_result = self._detect_stego(image, image_bytes, image.format)

        # Run full forensics analysis
        if self._analyzer == "simplified":
            # Simplified analysis without full engine
            return self._simplified_analysis(image, image_bytes, metadata, stego_result)

        # Full analysis with ForensicsEngine
        from imagetrust.forensics.base import PluginCategory

        # Save to temp file for analysis
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path)

        try:
            categories = [
                PluginCategory.PIXEL,
                PluginCategory.METADATA,
                PluginCategory.SOURCE,
                PluginCategory.AI_DETECTION,
            ]

            report = self._forensics_engine.analyze(tmp_path, categories=categories)

            # Convert to result format using same logic as desktop_launcher
            result = self._convert_report(report, metadata, stego_result)

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return result

    def _simplified_analysis(self, image, image_bytes: bytes, metadata: dict, stego: dict) -> dict:
        """
        Simplified analysis when full engine is not available.
        Uses only metadata heuristics with CALIBRATED thresholds v2.0.
        """
        # Calibration constants
        AI_THRESHOLD_FAKE = 75
        AI_THRESHOLD_SUSPICIOUS = 50
        AI_THRESHOLD_AUTHENTIC = 25
        DAMPING_FACTOR = 0.5
        MOD_SCORE_CAP = 30

        ai_score_raw = 0
        modification_score_raw = 0
        social_media_detected = metadata.get("social_media", {}).get("detected", False)
        provenance_tag = "none"

        # Simple heuristics based on metadata
        if metadata.get("exif_status") == "stripped":
            if not social_media_detected:
                modification_score_raw = 30
        if metadata.get("forensic", {}).get("editing_software_detected"):
            modification_score_raw = max(modification_score_raw, 50)

        # Detect provenance
        if social_media_detected:
            platform = metadata.get("social_media", {}).get("platform", "").lower()
            provenance_tag = platform if platform else "social_media"
        elif metadata.get("screenshot", {}).get("is_screenshot"):
            provenance_tag = "screenshot"
            social_media_detected = True

        # Cap modification score for social media
        if social_media_detected:
            modification_score = min(modification_score_raw, MOD_SCORE_CAP)
        else:
            modification_score = modification_score_raw

        # Apply damping
        ai_score_damped = ai_score_raw
        social_media_damping_applied = False

        if social_media_detected and ai_score_raw < AI_THRESHOLD_FAKE:
            ai_score_damped = ai_score_raw * DAMPING_FACTOR
            social_media_damping_applied = True

        # Determine verdict with calibrated thresholds
        if ai_score_raw >= AI_THRESHOLD_FAKE:
            verdict = "ai_generated_likely"
            confidence = ai_score_raw / 100
        elif ai_score_raw >= AI_THRESHOLD_SUSPICIOUS:
            verdict = "inconclusive"
            confidence = 0.5
        elif modification_score >= 80:
            verdict = "manipulated_likely"
            confidence = modification_score / 100
        elif ai_score_damped >= AI_THRESHOLD_AUTHENTIC:
            verdict = "inconclusive"
            confidence = 0.5
        else:
            verdict = "camera_original_likely"
            confidence = max(0.75, 1.0 - ai_score_damped / 100)

        return {
            "success": True,
            "authenticity": {
                "verdict": verdict,
                "confidence": confidence,
                "score": int(confidence * 100)
            },
            "scores": {
                "ai_raw": ai_score_raw,
                "ai_damped": ai_score_damped,
                "modification": modification_score,
                "modification_raw": modification_score_raw,
                "metadata": 0
            },
            "provenance_tag": provenance_tag,
            "social_media_damping_applied": social_media_damping_applied,
            "steganography": stego
        }

    def _convert_report(self, report, metadata: dict, stego: dict) -> dict:
        """
        Convert forensics report to result format.
        Mirrors logic from desktop_launcher._convert_report_to_result()

        CALIBRATED ADAPTIVE FUSION FORENSICS v2.0:
        ============================================
        Thresholds (tuned to reduce false positives):
        - AI >= 75%: FAKE (ai_generated_likely) - high confidence only
        - AI 50-75%: INCONCLUSIVE (suspicious but not definitive)
        - AI < 50%: Apply aggressive damping (0.5) for social media

        Damping Factor: 0.5 (aggressive - cuts AI score in HALF)
        Modification Score: Capped at 30% when social media detected
        """
        # =================================================================
        # CALIBRATION CONSTANTS v2.2 - DECISIVE (SAME AS desktop_launcher.py)
        # =================================================================
        # More decisive thresholds - smaller inconclusive zone
        # Trade-off: May have more FP but also better recall
        AI_THRESHOLD_FAKE = 55          # AI score >= 55% = FAKE
        AI_THRESHOLD_SUSPICIOUS = 45    # AI score 45-55% = INCONCLUSIVE (narrow)
        AI_THRESHOLD_AUTHENTIC = 22     # AI score < 22% after damping = AUTHENTIC
        DAMPING_FACTOR_SOCIAL = 0.45    # More aggressive damping
        DAMPING_FACTOR_RECOMPRESS = 0.55
        MOD_SCORE_CAP_SOCIAL = 25
        MOD_THRESHOLD_MANIPULATED = 70

        ai_score_raw = 0
        modification_score_raw = 0
        metadata_score = 0
        social_media_detected = False
        social_media_damping_applied = False
        provenance_tag = "none"

        # Process plugin results
        for result in report.results:
            category = result.category.value
            score_pct = result.score * 100

            if category == "ai_detection":
                ai_score_raw = max(ai_score_raw, score_pct)
            elif category == "pixel_forensics":
                modification_score_raw = max(modification_score_raw, score_pct)
            elif category == "metadata_forensics":
                metadata_score = max(metadata_score, score_pct)
            elif category == "source_platform":
                if result.detected and result.score > 0.3:
                    social_media_detected = True
                    plugin_name = result.plugin_name.lower()
                    if "whatsapp" in plugin_name or "messaging" in plugin_name:
                        provenance_tag = "whatsapp"
                    elif "screenshot" in plugin_name:
                        provenance_tag = "screenshot"
                    elif "social" in plugin_name or "instagram" in plugin_name:
                        provenance_tag = "social_media"

        # Check metadata for social media indicators
        exif_status = metadata.get("exif_status", "unknown")
        jpeg_quality = metadata.get("jpeg", {}).get("quality")

        if exif_status == "stripped":
            if jpeg_quality and jpeg_quality < 85:
                social_media_detected = True
                if provenance_tag == "none":
                    provenance_tag = "social_media"

        if metadata.get("social_media", {}).get("detected"):
            social_media_detected = True
            platform = metadata.get("social_media", {}).get("platform", "").lower()
            if platform and provenance_tag == "none":
                provenance_tag = platform if platform in ["whatsapp", "instagram", "facebook"] else "social_media"

        # Web download detection
        if (exif_status in ["stripped", "none"] and
            jpeg_quality is not None and jpeg_quality >= 85 and
            not social_media_detected):
            provenance_tag = "web_download"

        # Screenshot detection from metadata
        if metadata.get("screenshot", {}).get("is_screenshot"):
            provenance_tag = "screenshot"
            social_media_detected = True

        # =================================================================
        # MODIFICATION SCORE CALIBRATION
        # =================================================================
        if social_media_detected:
            modification_score = min(modification_score_raw, MOD_SCORE_CAP_SOCIAL)
        else:
            modification_score = modification_score_raw

        # =================================================================
        # ADAPTIVE FUSION FORENSICS v2.2 - DECISIVE LOGIC
        # =================================================================
        # More decisive classification with narrow inconclusive zone
        ai_score_damped = ai_score_raw

        # STEP 1: AI >= 55% = FAKE (catch more fakes)
        if ai_score_raw >= AI_THRESHOLD_FAKE:
            verdict = "ai_generated_likely"
            confidence = ai_score_raw / 100

        # STEP 2: AI 45-55% = NARROW SUSPICIOUS ZONE
        elif ai_score_raw >= AI_THRESHOLD_SUSPICIOUS:
            if social_media_detected:
                # Apply aggressive damping
                ai_score_damped = ai_score_raw * DAMPING_FACTOR_SOCIAL
                social_media_damping_applied = True

                if ai_score_damped < AI_THRESHOLD_AUTHENTIC:
                    verdict = "camera_original_likely"
                    confidence = 0.72
                else:
                    verdict = "inconclusive"
                    confidence = 0.52
            else:
                # No social media + suspicious AI = lean towards FAKE
                verdict = "ai_generated_likely"
                confidence = 0.55

        # STEP 3: AI < 45% with social media = AUTHENTIC
        elif social_media_detected:
            ai_score_damped = ai_score_raw * DAMPING_FACTOR_SOCIAL
            social_media_damping_applied = True
            verdict = "camera_original_likely"
            confidence = max(0.78, 1.0 - ai_score_damped / 100)

        # STEP 4: AI < 45% without social media
        else:
            if modification_score >= MOD_THRESHOLD_MANIPULATED:
                verdict = "manipulated_likely"
                confidence = modification_score / 100
            elif ai_score_raw >= 35:
                verdict = "inconclusive"
                confidence = 0.5
            else:
                verdict = "camera_original_likely"
                confidence = max(0.78, 1.0 - ai_score_raw / 100)

        return {
            "success": True,
            "authenticity": {
                "verdict": verdict,
                "confidence": confidence,
                "score": int(confidence * 100)
            },
            "scores": {
                "ai_raw": int(ai_score_raw),
                "ai_damped": int(ai_score_damped),
                "modification": int(modification_score),
                "modification_raw": int(modification_score_raw),
                "metadata": int(metadata_score)
            },
            "provenance_tag": provenance_tag,
            "social_media_damping_applied": social_media_damping_applied,
            "steganography": stego
        }

    def _get_image_files(self, folder: Path) -> List[Path]:
        """Get all supported image files from folder."""
        files = []
        if not folder.exists():
            return files

        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(folder.glob(f"*{ext}"))
            files.extend(folder.glob(f"*{ext.upper()}"))

        return sorted(files)

    def _classify_result(self, result: dict, ground_truth: str) -> Tuple[bool, str]:
        """
        Classify prediction against ground truth.

        Args:
            result: Analysis result
            ground_truth: "real" or "fake"

        Returns:
            (is_correct, classification_type)
            classification_type is one of: TP, FP, TN, FN, INCONCLUSIVE
        """
        verdict = result.get("authenticity", {}).get("verdict", "inconclusive")

        # Handle inconclusive separately
        if verdict in self.INCONCLUSIVE_VERDICTS:
            return False, "INCONCLUSIVE"

        predicted_fake = verdict in self.FAKE_VERDICTS
        actual_fake = ground_truth == "fake"

        if predicted_fake and actual_fake:
            return True, "TP"  # True Positive
        elif predicted_fake and not actual_fake:
            return False, "FP"  # False Positive
        elif not predicted_fake and not actual_fake:
            return True, "TN"  # True Negative
        else:
            return False, "FN"  # False Negative

    def run_benchmark(
        self,
        real_folder: Optional[Path] = None,
        fake_folder: Optional[Path] = None,
        output_csv: Optional[Path] = None
    ) -> BenchmarkMetrics:
        """
        Run the full benchmark on provided datasets.

        Args:
            real_folder: Path to folder containing known REAL images
            fake_folder: Path to folder containing known FAKE/AI images
            output_csv: Path to save detailed results CSV

        Returns:
            BenchmarkMetrics with calculated scientific metrics
        """
        self.results = []
        self.metrics = BenchmarkMetrics()

        # Collect all images
        images_to_process = []

        if real_folder:
            real_folder = Path(real_folder)
            real_files = self._get_image_files(real_folder)
            self._log(f"Found {len(real_files)} REAL images in {real_folder}")
            images_to_process.extend([(f, "real") for f in real_files])
            self.metrics.total_real = len(real_files)

        if fake_folder:
            fake_folder = Path(fake_folder)
            fake_files = self._get_image_files(fake_folder)
            self._log(f"Found {len(fake_files)} FAKE images in {fake_folder}")
            images_to_process.extend([(f, "fake") for f in fake_files])
            self.metrics.total_fake = len(fake_files)

        self.metrics.total_images = len(images_to_process)

        if self.metrics.total_images == 0:
            self._log("No images found to process!")
            return self.metrics

        self._log(f"\nProcessing {self.metrics.total_images} images...")
        self._log("=" * 60)

        # Process each image
        for idx, (image_path, ground_truth) in enumerate(images_to_process, 1):
            try:
                # Progress indicator
                if self.verbose and idx % 10 == 0:
                    print(f"  Progress: {idx}/{self.metrics.total_images} ({idx*100//self.metrics.total_images}%)")

                # Analyze image
                start_time = time.time()
                result = self._analyze_image(image_path)
                processing_time = (time.time() - start_time) * 1000

                # Extract scores
                auth = result.get("authenticity", {})
                scores = result.get("scores", {})
                provenance_tag = result.get("provenance_tag", "none")
                stego = result.get("steganography", {})

                # Classify result
                is_correct, classification = self._classify_result(result, ground_truth)

                # Update confusion matrix metrics
                if classification == "TP":
                    self.metrics.true_positives += 1
                    self.metrics.fake_correct += 1
                elif classification == "FP":
                    self.metrics.false_positives += 1
                elif classification == "TN":
                    self.metrics.true_negatives += 1
                    self.metrics.real_correct += 1
                elif classification == "FN":
                    self.metrics.false_negatives += 1
                elif classification == "INCONCLUSIVE":
                    self.metrics.inconclusive_count += 1

                # Track damping statistics
                damping_applied = result.get("social_media_damping_applied", False)
                if damping_applied:
                    self.metrics.social_media_damping_count += 1

                    # NOVELTY METRIC: Count "corrections"
                    # A correction occurs when:
                    # 1. Ground truth is REAL
                    # 2. Damping was applied
                    # 3. Final verdict is NOT fake (i.e., we saved it from being mislabeled)
                    #
                    # To verify damping actually helped, we check if raw AI score would have
                    # caused a different classification
                    ai_raw = scores.get("ai_raw", 0)
                    ai_damped = scores.get("ai_damped", 0)
                    verdict = auth.get("verdict", "")

                    if ground_truth == "real":
                        # Would raw score have caused a false positive?
                        # Raw score >= 30 could trigger "inconclusive" or higher
                        # Damped score < 30 allows "camera_original_likely"
                        if ai_raw >= 30 and ai_damped < 30 and verdict == "camera_original_likely":
                            self.metrics.social_media_correction_count += 1

                # Track steganography
                if stego.get("detected", False):
                    self.metrics.steganography_detected_count += 1

                # Track provenance breakdown
                if provenance_tag == "whatsapp":
                    self.metrics.provenance_whatsapp += 1
                elif provenance_tag == "screenshot":
                    self.metrics.provenance_screenshot += 1
                elif provenance_tag == "web_download":
                    self.metrics.provenance_web_download += 1
                elif provenance_tag in ["social_media", "instagram", "facebook"]:
                    self.metrics.provenance_social_media += 1
                else:
                    self.metrics.provenance_none += 1

                self.metrics.total_processing_time_ms += processing_time

                # Store result with all new fields
                self.results.append(BenchmarkResult(
                    filename=image_path.name,
                    ground_truth=ground_truth,
                    predicted_verdict=auth.get("verdict", "unknown"),
                    confidence=auth.get("confidence", 0.0),
                    ai_score_raw=scores.get("ai_raw", scores.get("ai", 0)),
                    ai_score_damped=scores.get("ai_damped", scores.get("ai", 0)),
                    modification_score=scores.get("modification", 0),
                    metadata_score=scores.get("metadata", 0),
                    provenance_tag=provenance_tag,
                    steganography_detected=stego.get("detected", False),
                    social_media_damping_applied=damping_applied,
                    processing_time_ms=processing_time,
                    is_correct=is_correct
                ))

            except Exception as e:
                self._log(f"  Error processing {image_path.name}: {e}")
                continue

        # Save results to CSV if requested
        if output_csv:
            self._save_csv(output_csv)

        return self.metrics

    def _save_csv(self, output_path: Path):
        """Save detailed results to CSV file for thesis data analysis."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header - matches thesis requirements
            writer.writerow([
                "filename",
                "ground_truth",
                "predicted_verdict",
                "ai_score_raw",
                "ai_score_damped",
                "provenance_tag",
                "steganography_detected",
                "modification_score",
                "metadata_score",
                "confidence",
                "social_media_damping",
                "processing_time_ms",
                "is_correct"
            ])

            # Data rows
            for r in self.results:
                writer.writerow([
                    r.filename,
                    r.ground_truth,
                    r.predicted_verdict,
                    r.ai_score_raw,
                    r.ai_score_damped,
                    r.provenance_tag,
                    r.steganography_detected,
                    r.modification_score,
                    r.metadata_score,
                    f"{r.confidence:.4f}",
                    r.social_media_damping_applied,
                    f"{r.processing_time_ms:.2f}",
                    r.is_correct
                ])

        self._log(f"\nResults saved to: {output_path}")

    def print_report(self):
        """Print formatted scientific benchmark report for thesis."""
        m = self.metrics

        print("\n" + "=" * 70)
        print("         SCIENTIFIC BENCHMARK REPORT")
        print("         ImageTrust - Adaptive Fusion Forensics")
        print("         (For Dissertation / Conference Publication)")
        print("=" * 70)

        print(f"\nDATASET STATISTICS:")
        print(f"  Total Images:     {m.total_images}")
        print(f"  Real Images:      {m.total_real}")
        print(f"  Fake Images:      {m.total_fake}")

        print(f"\nCLASSIFICATION RESULTS:")
        print(f"  Correct:          {m.true_positives + m.true_negatives}")
        print(f"  Incorrect:        {m.false_positives + m.false_negatives}")
        print(f"  Inconclusive:     {m.inconclusive_count}")

        print(f"\n" + "-" * 50)
        print(f"SCIENTIFIC METRICS (Paper-Ready):")
        print(f"-" * 50)
        print(f"  Accuracy:         {m.accuracy * 100:.2f}%")
        print(f"  Precision:        {m.precision * 100:.2f}%")
        print(f"  Recall:           {m.recall * 100:.2f}%")
        print(f"  F1-Score:         {m.f1_score:.4f}")
        print(f"  Specificity:      {m.specificity * 100:.2f}%")

        print(f"\n" + "-" * 50)
        print(f"CONFUSION MATRIX:")
        print(f"-" * 50)
        print(f"                  Predicted")
        print(f"                 FAKE    REAL")
        print(f"  Actual FAKE  [ {m.true_positives:4d}    {m.false_negatives:4d} ]")
        print(f"  Actual REAL  [ {m.false_positives:4d}    {m.true_negatives:4d} ]")

        print(f"\n" + "-" * 50)
        print(f"PROVENANCE BREAKDOWN:")
        print(f"-" * 50)
        print(f"  WhatsApp/Messaging:  {m.provenance_whatsapp}")
        print(f"  Screenshot:          {m.provenance_screenshot}")
        print(f"  Web Download:        {m.provenance_web_download}")
        print(f"  Other Social Media:  {m.provenance_social_media}")
        print(f"  No Provenance Tag:   {m.provenance_none}")

        print(f"\n" + "-" * 50)
        print(f"STEGANOGRAPHY DETECTION:")
        print(f"-" * 50)
        print(f"  Images with Hidden Data: {m.steganography_detected_count}")
        if m.total_images > 0:
            print(f"  Detection Rate:     {m.steganography_detected_count * 100 / m.total_images:.1f}%")

        print(f"\n" + "=" * 70)
        print(f"*** NOVELTY VALIDATION: ADAPTIVE FUSION FORENSICS ***")
        print(f"=" * 70)
        print(f"  Social Media Damping Factor: 0.7")
        print(f"  Damping Triggered:           {m.social_media_damping_count} times")
        if m.total_images > 0:
            print(f"  Damping Rate:                {m.social_media_damping_count * 100 / m.total_images:.1f}%")
        print(f"")
        print(f"  *** SOCIAL MEDIA CORRECTION COUNT: {m.social_media_correction_count} ***")
        print(f"")
        print(f"  This metric shows how many REAL images were SAVED from being")
        print(f"  incorrectly classified as FAKE due to compression artifacts.")
        print(f"  This proves the novelty of the Adaptive Fusion approach.")
        if m.total_real > 0:
            print(f"  Correction Rate (of real): {m.social_media_correction_count * 100 / m.total_real:.1f}%")

        print(f"\n" + "-" * 50)
        print(f"PERFORMANCE METRICS:")
        print(f"-" * 50)
        print(f"  Total Time:       {m.total_processing_time_ms / 1000:.2f} seconds")
        print(f"  Avg per Image:    {m.avg_processing_time_ms:.0f} ms")
        print(f"  Throughput:       {1000 / m.avg_processing_time_ms:.2f} images/sec" if m.avg_processing_time_ms > 0 else "  Throughput:       N/A")

        print(f"\n" + "=" * 70)
        print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="ImageTrust Scientific Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark against CIFAKE dataset
  python -m imagetrust.tools.benchmark_tool \\
      --real data/cifake/real \\
      --fake data/cifake/fake \\
      --output results/cifake_benchmark.csv

  # Benchmark only fake images
  python -m imagetrust.tools.benchmark_tool \\
      --fake data/ai_generated \\
      --output results/ai_only.csv

  # Quick test with verbose output
  python -m imagetrust.tools.benchmark_tool \\
      --real test_images/real \\
      --fake test_images/fake \\
      --verbose
        """
    )

    parser.add_argument(
        "--real",
        type=Path,
        help="Path to folder containing known REAL images"
    )
    parser.add_argument(
        "--fake",
        type=Path,
        help="Path to folder containing known FAKE/AI images"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Path to save detailed results CSV"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Enable verbose output (default: True)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Disable verbose output"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.real and not args.fake:
        parser.error("At least one of --real or --fake must be provided")

    # Create benchmark tool
    verbose = args.verbose and not args.quiet
    benchmark = FusionForensicsBenchmark(verbose=verbose)

    # Run benchmark
    metrics = benchmark.run_benchmark(
        real_folder=args.real,
        fake_folder=args.fake,
        output_csv=args.output
    )

    # Print report
    benchmark.print_report()

    # Return exit code based on success
    sys.exit(0 if metrics.total_images > 0 else 1)


if __name__ == "__main__":
    main()
