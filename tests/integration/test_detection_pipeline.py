"""
Integration tests for the detection pipeline.

Tests the full detection workflow from image input to result output.
"""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    return img


@pytest.fixture
def sample_image_path(sample_image, tmp_path):
    """Save sample image to a temporary path."""
    path = tmp_path / "test_image.jpg"
    sample_image.save(path, "JPEG", quality=95)
    return path


@pytest.fixture
def sample_image_bytes(sample_image):
    """Get sample image as bytes."""
    buffer = io.BytesIO()
    sample_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def batch_images(tmp_path):
    """Create multiple test images."""
    paths = []
    for i in range(5):
        img = Image.new("RGB", (224, 224), color=(i * 50, i * 50, i * 50))
        path = tmp_path / f"test_image_{i}.jpg"
        img.save(path, "JPEG")
        paths.append(path)
    return paths


# =============================================================================
# AIDetector Integration Tests
# =============================================================================


class TestAIDetectorIntegration:
    """Integration tests for AIDetector."""

    @pytest.mark.integration
    def test_detect_from_path(self, sample_image_path):
        """Test detection from file path."""
        try:
            from imagetrust.detection import AIDetector

            detector = AIDetector()
            result = detector.detect(str(sample_image_path))

            assert "ai_probability" in result
            assert "real_probability" in result
            assert "verdict" in result
            assert "confidence" in result
            assert 0 <= result["ai_probability"] <= 1
            assert 0 <= result["real_probability"] <= 1
            assert abs(result["ai_probability"] + result["real_probability"] - 1.0) < 0.01

        except ImportError:
            pytest.skip("AIDetector not available")

    @pytest.mark.integration
    def test_detect_from_pil_image(self, sample_image):
        """Test detection from PIL Image object."""
        try:
            from imagetrust.detection import AIDetector

            detector = AIDetector()
            result = detector.detect(sample_image)

            assert "ai_probability" in result
            assert result["verdict"] is not None

        except ImportError:
            pytest.skip("AIDetector not available")

    @pytest.mark.integration
    def test_detect_from_bytes(self, sample_image_bytes):
        """Test detection from image bytes."""
        try:
            from imagetrust.detection import AIDetector

            detector = AIDetector()
            result = detector.detect(sample_image_bytes)

            assert "ai_probability" in result

        except ImportError:
            pytest.skip("AIDetector not available")

    @pytest.mark.integration
    def test_batch_detection(self, batch_images):
        """Test batch detection."""
        try:
            from imagetrust.detection import AIDetector

            detector = AIDetector()
            results = detector.detect_batch([str(p) for p in batch_images])

            assert len(results) == len(batch_images)
            for result in results:
                assert "ai_probability" in result

        except ImportError:
            pytest.skip("AIDetector not available")

    @pytest.mark.integration
    def test_result_consistency(self, sample_image_path):
        """Test that repeated detection gives consistent results."""
        try:
            from imagetrust.detection import AIDetector

            detector = AIDetector()

            result1 = detector.detect(str(sample_image_path))
            result2 = detector.detect(str(sample_image_path))

            # Results should be identical for same image
            assert abs(result1["ai_probability"] - result2["ai_probability"]) < 0.001

        except ImportError:
            pytest.skip("AIDetector not available")


# =============================================================================
# Baseline Integration Tests
# =============================================================================


class TestBaselineIntegration:
    """Integration tests for baseline framework."""

    @pytest.mark.integration
    def test_list_baselines(self):
        """Test listing available baselines."""
        try:
            from imagetrust.baselines import list_baselines

            baselines = list_baselines()
            assert isinstance(baselines, list)
            assert len(baselines) > 0

        except ImportError:
            pytest.skip("Baselines not available")

    @pytest.mark.integration
    def test_get_baseline(self):
        """Test getting a baseline instance."""
        try:
            from imagetrust.baselines import get_baseline

            baseline = get_baseline("imagetrust")
            assert baseline is not None
            assert hasattr(baseline, "predict_proba")

        except ImportError:
            pytest.skip("Baselines not available")

    @pytest.mark.integration
    def test_baseline_prediction(self, sample_image_path):
        """Test baseline prediction."""
        try:
            from imagetrust.baselines import get_baseline

            baseline = get_baseline("imagetrust")
            result = baseline.predict_proba(str(sample_image_path))

            assert hasattr(result, "ai_probability")
            assert hasattr(result, "real_probability")
            assert 0 <= result.ai_probability <= 1

        except ImportError:
            pytest.skip("Baselines not available")

    @pytest.mark.integration
    def test_baseline_batch_prediction(self, batch_images):
        """Test baseline batch prediction."""
        try:
            from imagetrust.baselines import get_baseline

            baseline = get_baseline("imagetrust")
            results = baseline.predict_proba_batch([str(p) for p in batch_images])

            assert len(results) == len(batch_images)

        except ImportError:
            pytest.skip("Baselines not available")


# =============================================================================
# Calibration Integration Tests
# =============================================================================


class TestCalibrationIntegration:
    """Integration tests for calibration."""

    @pytest.mark.integration
    def test_calibrator_creation(self):
        """Test calibrator creation."""
        try:
            from imagetrust.baselines import BaselineCalibrator

            calibrator = BaselineCalibrator(method="temperature")
            assert calibrator is not None

        except ImportError:
            pytest.skip("Calibration not available")

    @pytest.mark.integration
    def test_calibration_fit(self):
        """Test calibrator fitting."""
        try:
            from imagetrust.baselines import BaselineCalibrator

            calibrator = BaselineCalibrator(method="temperature")

            # Synthetic data
            probs = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
            labels = np.array([0, 0, 1, 1, 1])

            calibrator.fit(probs, labels)
            assert calibrator.is_fitted

        except ImportError:
            pytest.skip("Calibration not available")

    @pytest.mark.integration
    def test_calibration_transform(self):
        """Test calibrator transform."""
        try:
            from imagetrust.baselines import BaselineCalibrator

            calibrator = BaselineCalibrator(method="temperature")

            probs = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
            labels = np.array([0, 0, 1, 1, 1])

            calibrator.fit(probs, labels)
            calibrated = calibrator.calibrate(0.85)

            assert 0 <= calibrated <= 1

        except ImportError:
            pytest.skip("Calibration not available")


# =============================================================================
# Data Pipeline Integration Tests
# =============================================================================


class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""

    @pytest.mark.integration
    def test_split_creation(self, tmp_path):
        """Test dataset split creation."""
        try:
            from imagetrust.data.splits import create_split, load_split, save_split

            # Create mock dataset structure
            (tmp_path / "real").mkdir()
            (tmp_path / "ai").mkdir()

            for i in range(10):
                img = Image.new("RGB", (100, 100))
                img.save(tmp_path / "real" / f"real_{i}.jpg")
                img.save(tmp_path / "ai" / f"ai_{i}.jpg")

            # Create split
            split = create_split(
                data_root=tmp_path,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
                seed=42,
            )

            assert len(split.train) > 0
            assert len(split.val) > 0
            assert len(split.test) > 0

            # Save and reload
            split_path = tmp_path / "split.json"
            save_split(split, split_path)

            loaded = load_split(split_path)
            assert len(loaded.train) == len(split.train)

        except ImportError:
            pytest.skip("Data pipeline not available")


# =============================================================================
# CLI Integration Tests
# =============================================================================


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    @pytest.mark.integration
    def test_cli_analyze(self, sample_image_path):
        """Test CLI analyze command."""
        try:
            from click.testing import CliRunner

            from imagetrust.cli import main

            runner = CliRunner()
            result = runner.invoke(main, ["analyze", str(sample_image_path)])

            # Should not crash
            assert result.exit_code in [0, 1]  # 1 if models not loaded

        except ImportError:
            pytest.skip("CLI not available")

    @pytest.mark.integration
    def test_cli_info(self):
        """Test CLI info command."""
        try:
            from click.testing import CliRunner

            from imagetrust.cli import main

            runner = CliRunner()
            result = runner.invoke(main, ["info"])

            assert result.exit_code == 0

        except ImportError:
            pytest.skip("CLI not available")


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_detection_workflow(self, batch_images, tmp_path):
        """Test complete detection workflow."""
        try:
            from imagetrust.detection import AIDetector

            detector = AIDetector()

            # Process all images
            results = []
            for path in batch_images:
                result = detector.detect(str(path))
                results.append(
                    {
                        "path": str(path),
                        "ai_probability": result["ai_probability"],
                        "verdict": result["verdict"].value,
                    }
                )

            # Save results
            output_path = tmp_path / "results.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            # Verify results were saved
            assert output_path.exists()
            with open(output_path) as f:
                loaded = json.load(f)
            assert len(loaded) == len(batch_images)

        except ImportError:
            pytest.skip("Detection pipeline not available")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_evaluation_workflow(self, tmp_path):
        """Test evaluation workflow with synthetic data."""
        try:
            from imagetrust.baselines import get_baseline
            from imagetrust.evaluation.metrics import compute_metrics

            # Create labeled test data
            (tmp_path / "real").mkdir()
            (tmp_path / "ai").mkdir()

            paths = []
            labels = []

            for i in range(5):
                img = Image.new("RGB", (224, 224), color=(100, 100, 100))
                path = tmp_path / "real" / f"real_{i}.jpg"
                img.save(path)
                paths.append(path)
                labels.append(0)

            for i in range(5):
                img = Image.new("RGB", (224, 224), color=(200, 200, 200))
                path = tmp_path / "ai" / f"ai_{i}.jpg"
                img.save(path)
                paths.append(path)
                labels.append(1)

            # Get predictions
            baseline = get_baseline("imagetrust")
            probs = []
            for path in paths:
                result = baseline.predict_proba(str(path))
                probs.append(result.ai_probability)

            # Compute metrics
            preds = [1 if p > 0.5 else 0 for p in probs]
            metrics = compute_metrics(
                np.array(labels),
                np.array(preds),
                np.array(probs),
            )

            assert "accuracy" in metrics
            assert "roc_auc" in metrics

        except ImportError:
            pytest.skip("Evaluation pipeline not available")


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance-related integration tests."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_inference_speed(self, sample_image_path):
        """Test that inference completes in reasonable time."""
        import time

        try:
            from imagetrust.detection import AIDetector

            detector = AIDetector()

            # Warm-up
            detector.detect(str(sample_image_path))

            # Measure inference time
            start = time.time()
            for _ in range(5):
                detector.detect(str(sample_image_path))
            elapsed = time.time() - start

            avg_time = elapsed / 5
            # Should complete in under 5 seconds per image on CPU
            assert avg_time < 5.0, f"Inference too slow: {avg_time:.2f}s per image"

        except ImportError:
            pytest.skip("Detection not available")

    @pytest.mark.integration
    def test_memory_stability(self, batch_images):
        """Test that memory doesn't grow unbounded."""
        try:
            import gc

            from imagetrust.detection import AIDetector

            detector = AIDetector()

            # Process images multiple times
            for _ in range(3):
                for path in batch_images:
                    detector.detect(str(path))
                gc.collect()

            # If we get here without OOM, test passes

        except ImportError:
            pytest.skip("Detection not available")
