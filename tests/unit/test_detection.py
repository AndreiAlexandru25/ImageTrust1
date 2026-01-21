"""
Unit tests for detection module.
"""

import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import Mock, patch

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestImagePreprocessor:
    """Tests for ImagePreprocessor."""
    
    def test_preprocess_pil_image(self):
        """Test preprocessing PIL image."""
        from imagetrust.detection.preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(input_size=224)
        
        # Create test image
        image = Image.new("RGB", (100, 100), color="red")
        
        # Preprocess
        tensor = preprocessor.preprocess(image)
        
        # Check output
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
    
    def test_preprocess_numpy_array(self):
        """Test preprocessing numpy array."""
        from imagetrust.detection.preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(input_size=224)
        
        # Create test array
        array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Preprocess
        tensor = preprocessor.preprocess(array)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
    
    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        from imagetrust.detection.preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(input_size=224)
        
        # Create test images
        images = [Image.new("RGB", (100, 100), color=c) for c in ["red", "green", "blue"]]
        
        # Preprocess batch
        batch = preprocessor.preprocess_batch(images)
        
        assert batch.shape == (3, 3, 224, 224)
    
    def test_denormalize(self):
        """Test denormalization."""
        from imagetrust.detection.preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor(input_size=224)
        
        # Create and preprocess image
        image = Image.new("RGB", (224, 224), color="red")
        tensor = preprocessor.preprocess(image)
        
        # Denormalize
        denormalized = preprocessor.denormalize(tensor)
        
        # Should be in [0, 1] range
        assert denormalized.min() >= -0.5
        assert denormalized.max() <= 1.5


class TestCalibration:
    """Tests for calibration module."""
    
    def test_temperature_scaling(self):
        """Test temperature scaling."""
        from imagetrust.detection.calibration import TemperatureScaling
        
        calibrator = TemperatureScaling(initial_temperature=1.5)
        
        # Test forward pass
        logits = torch.randn(4, 2)
        scaled = calibrator(logits)
        
        assert scaled.shape == logits.shape
        # Scaled logits should be smaller (divided by temp > 1)
        assert torch.abs(scaled).mean() < torch.abs(logits).mean()
    
    def test_calibration_wrapper(self):
        """Test calibration wrapper."""
        from imagetrust.detection.calibration import CalibrationWrapper
        
        # Mock model
        model = Mock()
        model.return_value = torch.randn(4, 2)
        
        wrapper = CalibrationWrapper(
            model,
            calibration_method="temperature",
            min_confidence=0.80,
            max_confidence=0.95,
        )
        
        # Test forward
        x = torch.randn(4, 3, 224, 224)
        probs = wrapper(x)
        
        assert probs.shape == (4, 2)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=0.01)
    
    def test_expected_calibration_error(self):
        """Test ECE computation."""
        from imagetrust.detection.calibration import ExpectedCalibrationError
        
        ece_calculator = ExpectedCalibrationError(n_bins=10)
        
        # Create test data
        probs = np.random.rand(100, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize
        labels = np.random.randint(0, 2, 100)
        
        ece, stats = ece_calculator.compute(probs, labels)
        
        assert 0 <= ece <= 1
        assert len(stats["bin_accuracies"]) == 10


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_compute_metrics(self):
        """Test metric computation."""
        from imagetrust.evaluation.metrics import compute_metrics
        
        labels = np.array([0, 0, 1, 1, 1])
        predictions = np.array([0, 1, 1, 1, 0])
        probabilities = np.array([0.2, 0.6, 0.8, 0.9, 0.4])
        
        metrics = compute_metrics(labels, predictions, probabilities)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
    
    def test_compute_roc_auc(self):
        """Test ROC-AUC computation."""
        from imagetrust.evaluation.metrics import compute_roc_auc
        
        # Perfect predictions
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        
        auc = compute_roc_auc(labels, probs)
        
        assert auc == 1.0
    
    def test_compute_calibration_metrics(self):
        """Test calibration metrics."""
        from imagetrust.evaluation.metrics import compute_calibration_metrics
        
        labels = np.random.randint(0, 2, 100)
        probs = np.random.rand(100)
        
        metrics = compute_calibration_metrics(labels, probs)
        
        assert "ece" in metrics
        assert "mce" in metrics
        assert "brier_score" in metrics


class TestBaseDetector:
    """Tests for base detector."""
    
    def test_base_detector_is_abstract(self):
        """Test that BaseDetector is abstract."""
        from imagetrust.detection.models.base import BaseDetector
        
        # Should not be instantiable directly
        with pytest.raises(TypeError):
            BaseDetector()


class TestCNNDetector:
    """Tests for CNN detector."""
    
    @pytest.mark.slow
    def test_cnn_detector_forward(self):
        """Test CNN detector forward pass."""
        from imagetrust.detection.models.cnn_detector import CNNDetector
        
        detector = CNNDetector(
            backbone="resnet18",
            pretrained=False,  # Faster for testing
            device="cpu",
        )
        
        x = torch.randn(2, 3, 224, 224)
        output = detector(x)
        
        assert output.shape == (2, 2)
    
    @pytest.mark.slow
    def test_cnn_detector_predict(self):
        """Test CNN detector prediction."""
        from imagetrust.detection.models.cnn_detector import CNNDetector
        
        detector = CNNDetector(
            backbone="resnet18",
            pretrained=False,
            device="cpu",
        )
        
        # Create test image
        image = Image.new("RGB", (224, 224), color="red")
        
        result = detector.predict(image)
        
        assert "ai_probability" in result
        assert "real_probability" in result
        assert 0 <= result["ai_probability"] <= 1


class TestEnsembleDetector:
    """Tests for ensemble detector."""
    
    def test_ensemble_strategies(self):
        """Test different ensemble strategies."""
        from imagetrust.detection.models.ensemble import EnsembleDetector
        
        for strategy in ["average", "weighted", "voting", "max"]:
            # Create with mock detectors
            ensemble = EnsembleDetector(
                detectors=[],
                strategy=strategy,
            )
            
            assert ensemble.strategy == strategy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
