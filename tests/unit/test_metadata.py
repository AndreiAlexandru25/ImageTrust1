"""
Unit tests for metadata module.
"""

import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestEXIFParser:
    """Tests for EXIF parser."""

    def test_parse_image_without_exif(self):
        """Test parsing image without EXIF data."""
        from imagetrust.metadata.exif_parser import EXIFParser

        parser = EXIFParser()

        # Create simple image without EXIF
        image = Image.new("RGB", (100, 100), color="red")

        exif = parser.parse(image)

        # Should return empty EXIFData
        assert exif.make is None
        assert exif.model is None

    def test_parse_bytes(self):
        """Test parsing from bytes."""
        from imagetrust.metadata.exif_parser import EXIFParser

        parser = EXIFParser()

        # Create image and save to bytes
        image = Image.new("RGB", (100, 100), color="red")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        exif = parser.parse(image_bytes)

        assert exif is not None

    def test_detect_ai_indicators_software(self):
        """Test AI indicator detection from software field."""
        from imagetrust.core.types import EXIFData
        from imagetrust.metadata.exif_parser import EXIFParser

        parser = EXIFParser()

        # Test with AI software
        exif = EXIFData(software="Midjourney v5")
        indicators = parser.detect_ai_indicators(exif)

        assert len(indicators) > 0
        assert any("AI" in ind or "software" in ind.lower() for ind in indicators)

    def test_detect_ai_indicators_missing_camera(self):
        """Test AI indicator for missing camera info."""
        from imagetrust.core.types import EXIFData
        from imagetrust.metadata.exif_parser import EXIFParser

        parser = EXIFParser()

        # EXIF with data but no camera
        exif = EXIFData(raw_data={"something": "value"})
        indicators = parser.detect_ai_indicators(exif)

        assert any("camera" in ind.lower() for ind in indicators)


class TestXMPParser:
    """Tests for XMP parser."""

    def test_parse_image_without_xmp(self):
        """Test parsing image without XMP data."""
        from imagetrust.metadata.xmp_parser import XMPParser

        parser = XMPParser()

        # Create simple image
        image = Image.new("RGB", (100, 100), color="blue")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")

        xmp = parser.parse(buffer.getvalue())

        # Should return empty XMPData
        assert xmp.creator is None
        assert xmp.creator_tool is None

    def test_detect_ai_indicators(self):
        """Test AI indicator detection."""
        from imagetrust.core.types import XMPData
        from imagetrust.metadata.xmp_parser import XMPParser

        parser = XMPParser()

        # Test with AI tool
        xmp = XMPData(creator_tool="DALL-E 3")
        indicators = parser.detect_ai_indicators(xmp)

        assert len(indicators) > 0


class TestC2PAValidator:
    """Tests for C2PA validator."""

    def test_validate_image_without_c2pa(self):
        """Test validating image without C2PA."""
        from imagetrust.metadata.c2pa_validator import C2PAValidator

        validator = C2PAValidator()

        # Create simple image
        image = Image.new("RGB", (100, 100), color="green")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")

        manifest = validator.validate(buffer.getvalue())

        # Should indicate no C2PA
        assert manifest.is_present is False

    def test_get_trust_indicators_valid(self):
        """Test trust indicators for valid manifest."""
        from imagetrust.core.types import C2PAManifest
        from imagetrust.metadata.c2pa_validator import C2PAValidator

        validator = C2PAValidator()

        manifest = C2PAManifest(
            is_present=True,
            is_valid=True,
            claim_generator="Adobe Photoshop",
        )

        indicators = validator.get_trust_indicators(manifest)

        assert len(indicators) > 0
        assert any("valid" in ind.lower() or "trusted" in ind.lower() for ind in indicators)


class TestProvenanceAnalyzer:
    """Tests for provenance analyzer."""

    def test_analyze_simple_image(self):
        """Test analyzing a simple image."""
        from imagetrust.metadata.provenance import ProvenanceAnalyzer

        analyzer = ProvenanceAnalyzer()

        # Create simple image
        image = Image.new("RGB", (100, 100), color="yellow")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")

        metadata, provenance = analyzer.analyze(buffer.getvalue())

        assert metadata is not None
        assert provenance is not None
        assert metadata.width == 100
        assert metadata.height == 100

    def test_file_hash_computation(self):
        """Test file hash is computed correctly."""
        from imagetrust.metadata.provenance import ProvenanceAnalyzer

        analyzer = ProvenanceAnalyzer()

        # Create image
        image = Image.new("RGB", (50, 50), color="purple")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        data = buffer.getvalue()

        metadata, _ = analyzer.analyze(data)

        assert metadata.file_hash is not None
        assert len(metadata.file_hash) == 64  # SHA-256 hex

    def test_provenance_status(self):
        """Test provenance status determination."""
        from imagetrust.core.types import ProvenanceStatus
        from imagetrust.metadata.provenance import ProvenanceAnalyzer

        analyzer = ProvenanceAnalyzer()

        # Simple image should have MISSING status
        image = Image.new("RGB", (50, 50), color="cyan")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")

        _, provenance = analyzer.analyze(buffer.getvalue())

        assert provenance.status == ProvenanceStatus.MISSING

    def test_get_summary(self):
        """Test summary generation."""
        from imagetrust.metadata.provenance import ProvenanceAnalyzer

        analyzer = ProvenanceAnalyzer()

        # Create image
        image = Image.new("RGB", (50, 50), color="pink")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")

        metadata, provenance = analyzer.analyze(buffer.getvalue())

        summary = analyzer.get_summary(metadata, provenance)

        assert isinstance(summary, str)
        assert len(summary) > 0


class TestTypes:
    """Tests for type definitions."""

    def test_detection_score(self):
        """Test DetectionScore creation."""
        from imagetrust.core.types import DetectionScore, DetectionVerdict

        score = DetectionScore(
            detector_name="test",
            detector_version="1.0",
            ai_probability=0.85,
            real_probability=0.15,
            raw_score=0.85,
        )

        assert score.ai_probability == 0.85
        assert score.get_verdict() == DetectionVerdict.AI_GENERATED

    def test_confidence_from_probability(self):
        """Test confidence level determination."""
        from imagetrust.core.types import Confidence

        # High AI probability = high confidence
        assert Confidence.from_probability(0.95) == Confidence.VERY_HIGH

        # Near 0.5 = low confidence
        assert Confidence.from_probability(0.52) == Confidence.VERY_LOW

        # Low AI probability = high confidence in REAL
        assert Confidence.from_probability(0.05) == Confidence.VERY_HIGH

    def test_analysis_result_summary(self):
        """Test analysis result summary generation."""
        from imagetrust.core.types import AnalysisResult, Confidence, DetectionVerdict

        result = AnalysisResult(
            analysis_id="test-123",
            image_hash="abc123",
            image_dimensions=(100, 100),
            ai_probability=0.9,
            verdict=DetectionVerdict.AI_GENERATED,
            confidence=Confidence.HIGH,
        )

        summary = result.get_summary()

        assert "AI-generated" in summary or "ai" in summary.lower()
        assert "90" in summary or "0.9" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
