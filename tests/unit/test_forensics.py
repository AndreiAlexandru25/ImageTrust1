"""
Unit tests for ImageTrust Forensics Engine.
"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


class TestForensicsBase:
    """Tests for forensics base classes."""

    def test_plugin_registration(self):
        """Test plugin registry."""
        from imagetrust.forensics.base import (
            ForensicsPlugin,
            list_plugins,
            register_plugin,
        )

        # Import modules to trigger registration
        from imagetrust.forensics import pixel_forensics  # noqa
        from imagetrust.forensics import metadata_forensics  # noqa
        from imagetrust.forensics import source_detection  # noqa

        plugins = list_plugins()
        assert len(plugins) > 0
        assert "ela_detector" in plugins

    def test_confidence_from_score(self):
        """Test confidence level calculation."""
        from imagetrust.forensics.base import Confidence

        assert Confidence.from_score(0.95) == Confidence.VERY_HIGH
        assert Confidence.from_score(0.75) == Confidence.HIGH
        assert Confidence.from_score(0.55) == Confidence.MEDIUM
        assert Confidence.from_score(0.35) == Confidence.LOW
        assert Confidence.from_score(0.15) == Confidence.VERY_LOW


class TestPixelForensics:
    """Tests for pixel-level forensics detectors."""

    @pytest.fixture
    def sample_image(self):
        """Create a simple test image."""
        return Image.new("RGB", (256, 256), color=(128, 128, 128))

    @pytest.fixture
    def noisy_image(self):
        """Create a noisy test image."""
        arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_ela_detector(self, sample_image):
        """Test ELA detector."""
        from imagetrust.forensics.pixel_forensics import ELADetector

        detector = ELADetector()
        result = detector.analyze(sample_image)

        assert result.plugin_id == "ela_detector"
        assert 0 <= result.score <= 1
        assert result.explanation  # Has explanation
        assert len(result.limitations) > 0  # Has limitations

    def test_noise_inconsistency_detector(self, noisy_image):
        """Test noise inconsistency detector."""
        from imagetrust.forensics.pixel_forensics import NoiseInconsistencyDetector

        detector = NoiseInconsistencyDetector()
        result = detector.analyze(noisy_image)

        assert result.plugin_id == "noise_inconsistency"
        assert "cv_noise" in result.details
        assert result.explanation

    def test_jpeg_artifacts_detector(self, sample_image):
        """Test JPEG artifacts detector."""
        from imagetrust.forensics.pixel_forensics import JPEGArtifactsDetector

        detector = JPEGArtifactsDetector()
        result = detector.analyze(sample_image)

        assert result.plugin_id == "jpeg_artifacts"
        assert "blocking_strength" in result.details

    def test_resampling_detector(self, sample_image):
        """Test resampling detector."""
        from imagetrust.forensics.pixel_forensics import ResamplingDetector

        detector = ResamplingDetector()
        result = detector.analyze(sample_image)

        assert result.plugin_id == "resampling_detector"
        assert "periodicity_score" in result.details

    def test_edge_halo_detector(self, sample_image):
        """Test edge/halo detector."""
        from imagetrust.forensics.pixel_forensics import EdgeHaloDetector

        detector = EdgeHaloDetector()
        result = detector.analyze(sample_image)

        assert result.plugin_id == "edge_halo_detector"
        assert "halo_strength" in result.details


class TestMetadataForensics:
    """Tests for metadata forensics detectors."""

    @pytest.fixture
    def jpeg_image(self, tmp_path):
        """Create a JPEG image file."""
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        path = tmp_path / "test.jpg"
        img.save(path, format="JPEG", quality=90)

        with open(path, "rb") as f:
            raw_bytes = f.read()

        return Image.open(path), path, raw_bytes

    def test_metadata_analyzer(self, jpeg_image):
        """Test metadata analyzer."""
        from imagetrust.forensics.metadata_forensics import MetadataAnalyzer

        img, path, raw_bytes = jpeg_image
        detector = MetadataAnalyzer()
        result = detector.analyze(img, path, raw_bytes)

        assert result.plugin_id == "metadata_analyzer"
        assert "important_tags" in result.details

    def test_software_trace_detector(self, jpeg_image):
        """Test software trace detector."""
        from imagetrust.forensics.metadata_forensics import SoftwareTraceDetector

        img, path, raw_bytes = jpeg_image
        detector = SoftwareTraceDetector()
        result = detector.analyze(img, path, raw_bytes)

        assert result.plugin_id == "software_traces"
        assert "traces_found" in result.details


class TestSourceDetection:
    """Tests for source/platform detection."""

    @pytest.fixture
    def mobile_screenshot(self):
        """Create image with mobile screenshot dimensions."""
        return Image.new("RGB", (1080, 1920), color=(50, 50, 50))

    @pytest.fixture
    def desktop_screenshot(self):
        """Create image with desktop screenshot dimensions."""
        return Image.new("RGB", (1920, 1080), color=(50, 50, 50))

    def test_screenshot_detector_mobile(self, mobile_screenshot):
        """Test screenshot detector with mobile dimensions."""
        from imagetrust.forensics.source_detection import ScreenshotDetector

        detector = ScreenshotDetector()
        result = detector.analyze(mobile_screenshot)

        assert result.plugin_id == "screenshot_detector"
        # Mobile resolution should be detected
        assert result.details.get("resolution_match") is not None or result.score > 0

    def test_screenshot_detector_desktop(self, desktop_screenshot):
        """Test screenshot detector with desktop dimensions."""
        from imagetrust.forensics.source_detection import ScreenshotDetector

        detector = ScreenshotDetector()
        result = detector.analyze(desktop_screenshot)

        assert result.plugin_id == "screenshot_detector"
        assert result.details["resolution"] == (1920, 1080)

    def test_platform_detector(self):
        """Test platform detector."""
        from imagetrust.forensics.source_detection import PlatformDetector

        # Create WhatsApp-like image (1600px max)
        img = Image.new("RGB", (1600, 1200), color=(128, 128, 128))

        detector = PlatformDetector()
        result = detector.analyze(img)

        assert result.plugin_id == "platform_detector"
        assert "platform_scores" in result.details


class TestFusionLayer:
    """Tests for the fusion/decision layer."""

    def test_fusion_empty_results(self):
        """Test fusion with no results."""
        from imagetrust.forensics.fusion import FusionLayer

        fusion = FusionLayer()
        verdict = fusion.fuse([])

        assert verdict.inconclusive
        assert verdict.total_detectors_run == 0

    def test_fusion_verdict_structure(self):
        """Test verdict structure."""
        from imagetrust.forensics.base import Confidence, ForensicsResult, PluginCategory
        from imagetrust.forensics.fusion import FusionLayer

        results = [
            ForensicsResult(
                plugin_id="test_plugin",
                plugin_name="Test Plugin",
                category=PluginCategory.PIXEL,
                score=0.7,
                confidence=Confidence.HIGH,
                detected=True,
                explanation="Test detection",
                limitations=["Test limitation"],
            ),
        ]

        fusion = FusionLayer()
        verdict = fusion.fuse(results)

        assert verdict.total_detectors_run == 1
        assert len(verdict.labels) > 0
        assert verdict.authenticity_score >= 0 and verdict.authenticity_score <= 1


class TestForensicsEngine:
    """Tests for the main forensics engine."""

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample image file."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        path = tmp_path / "test_image.jpg"
        img.save(path, format="JPEG", quality=95)
        return path

    def test_engine_initialization(self):
        """Test engine initialization."""
        from imagetrust.forensics import ForensicsEngine

        engine = ForensicsEngine()
        plugins = engine.get_available_plugins()

        assert len(plugins) > 0

    def test_engine_analyze(self, sample_image):
        """Test full analysis."""
        from imagetrust.forensics import ForensicsEngine

        engine = ForensicsEngine()
        report = engine.analyze(sample_image)

        assert report.run_id
        assert report.image_path == str(sample_image)
        assert len(report.results) > 0
        assert report.verdict is not None

    def test_report_serialization(self, sample_image):
        """Test report JSON serialization."""
        from imagetrust.forensics import ForensicsEngine

        engine = ForensicsEngine()
        report = engine.analyze(sample_image)

        # Test to_dict
        data = report.to_dict()
        assert "run_id" in data
        assert "verdict" in data
        assert "results" in data

        # Test to_json
        json_str = report.to_json()
        assert len(json_str) > 0

    def test_report_markdown(self, sample_image):
        """Test report Markdown generation."""
        from imagetrust.forensics import ForensicsEngine

        engine = ForensicsEngine()
        report = engine.analyze(sample_image)

        md = report.to_markdown()
        assert "# Forensics Analysis Report" in md
        assert "Primary Verdict" in md
