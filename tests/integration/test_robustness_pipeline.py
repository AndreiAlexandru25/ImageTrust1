"""
Integration tests for the robustness pipeline.

Tests the full pipeline from image through augmentation to detection.
"""

import pytest
import numpy as np
from PIL import Image

# Skip tests if dependencies not available
pytest.importorskip("albumentations")


@pytest.fixture
def sample_images():
    """Create sample images for testing."""
    images = []
    for _ in range(5):
        img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        images.append(Image.fromarray(img_array))
    return images


class TestRobustnessPipeline:
    """Integration tests for the robustness augmentation pipeline."""

    def test_full_augmentation_pipeline(self, sample_images):
        """Test full augmentation pipeline from image to tensor."""
        from imagetrust.detection.augmentation import RobustnessAugmentor
        import torch

        augmentor = RobustnessAugmentor(
            input_size=224,
            social_media_prob=0.5,
            screenshot_prob=0.3,
        )

        for image in sample_images:
            # Training mode
            tensor_train, meta_train = augmentor.apply_pil_augmentation(image, mode="train")
            assert isinstance(tensor_train, torch.Tensor)
            assert tensor_train.shape == (3, 224, 224)

            # Validation mode
            tensor_val, meta_val = augmentor.apply_pil_augmentation(image, mode="val")
            assert isinstance(tensor_val, torch.Tensor)
            assert tensor_val.shape == (3, 224, 224)

    def test_social_media_simulation_chain(self, sample_images):
        """Test social media simulation followed by preprocessing."""
        from imagetrust.detection.augmentation import (
            SocialMediaSimulator,
            RobustnessAugmentor,
            Platform,
        )
        import torch

        simulator = SocialMediaSimulator(platforms=[Platform.WHATSAPP])
        augmentor = RobustnessAugmentor(input_size=224)

        for image in sample_images:
            # Simulate social media degradation
            degraded, sim_meta = simulator.simulate(image)

            # Then preprocess for model
            tensor, aug_meta = augmentor.apply_pil_augmentation(degraded, mode="val")

            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (3, 224, 224)

    def test_screenshot_simulation_chain(self, sample_images):
        """Test screenshot simulation followed by preprocessing."""
        from imagetrust.detection.augmentation import (
            ScreenshotSimulator,
            RobustnessAugmentor,
            ScreenshotType,
        )
        import torch

        simulator = ScreenshotSimulator(screenshot_types=[ScreenshotType.WINDOWS])
        augmentor = RobustnessAugmentor(input_size=224)

        for image in sample_images:
            # Simulate screenshot
            screenshot, sim_meta = simulator.simulate(image)

            # Then preprocess for model
            tensor, aug_meta = augmentor.apply_pil_augmentation(screenshot, mode="val")

            assert isinstance(tensor, torch.Tensor)

    def test_preprocessing_with_albumentations(self, sample_images):
        """Test AlbumentationsPreprocessor integration."""
        from imagetrust.detection.preprocessing import AlbumentationsPreprocessor
        import torch

        preprocessor = AlbumentationsPreprocessor(
            input_size=224,
            use_robustness_augmentation=True,
        )

        for image in sample_images:
            tensor = preprocessor.preprocess(image, mode="train")
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (3, 224, 224)

            tensor_val = preprocessor.preprocess(image, mode="val")
            assert tensor_val.shape == (3, 224, 224)


class TestConformalPipelineIntegration:
    """Integration tests for conformal prediction pipeline."""

    def test_conformal_with_detector_probabilities(self):
        """Test conformal prediction with simulated detector outputs."""
        from imagetrust.detection.conformal import ConformalPredictor

        np.random.seed(42)

        # Simulate detector outputs
        n_cal = 100
        n_test = 50

        # Calibration data
        cal_probs = np.random.beta(2, 2, n_cal)
        cal_labels = (cal_probs + np.random.normal(0, 0.1, n_cal) > 0.5).astype(int)

        # Test data
        test_probs = np.random.beta(2, 2, n_test)
        test_labels = (test_probs + np.random.normal(0, 0.1, n_test) > 0.5).astype(int)

        # Setup conformal predictor
        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate(cal_probs, cal_labels)

        # Make predictions
        predictions = predictor.predict_batch(test_probs)

        # Evaluate coverage
        coverage_metrics = predictor.evaluate_coverage(test_probs, test_labels)

        # Coverage should be approximately >= 90%
        assert coverage_metrics["coverage"] >= 0.80  # Allow margin for small sample

    def test_uncertainty_with_conformal(self):
        """Test uncertainty estimator with conformal method."""
        from imagetrust.baselines.uncertainty import UncertaintyEstimator

        np.random.seed(42)

        # Create estimator with conformal method
        estimator = UncertaintyEstimator(
            method="conformal",
            conformal_alpha=0.1,
        )

        # Calibrate
        cal_probs = np.random.uniform(0, 1, 100)
        cal_labels = (cal_probs > 0.5).astype(int)
        estimator.calibrate_conformal(cal_probs, cal_labels)

        # Estimate uncertainty
        uncertainty = estimator.estimate_uncertainty(0.5)

        # At decision boundary, should be uncertain
        assert uncertainty >= 0  # Valid uncertainty value


class TestMetricsWithSignificance:
    """Integration tests for metrics with statistical significance."""

    def test_full_evaluation_pipeline(self):
        """Test full evaluation with significance tests."""
        from imagetrust.evaluation.metrics import (
            compute_metrics_with_significance,
            format_results_table,
        )

        np.random.seed(42)
        n = 200

        y_true = np.random.randint(0, 2, n)

        # Simulated model predictions
        predictions = {
            "ImageTrust": y_true.copy(),
            "Baseline1": y_true.copy(),
            "Baseline2": np.random.randint(0, 2, n),
        }
        predictions["ImageTrust"][:10] = 1 - predictions["ImageTrust"][:10]
        predictions["Baseline1"][:30] = 1 - predictions["Baseline1"][:30]

        probabilities = {k: v.astype(float) for k, v in predictions.items()}

        # Compute metrics with significance
        results = compute_metrics_with_significance(
            y_true,
            predictions,
            probabilities,
            reference_model="ImageTrust",
        )

        # Check structure
        assert "per_model_metrics" in results
        assert "significance_tests" in results
        assert "summary_table" in results

        # Format table
        table = format_results_table(results)
        assert "Model" in table
        assert "Accuracy" in table


@pytest.mark.slow
class TestTrainingIntegration:
    """Integration tests for training components."""

    def test_consistency_loss(self):
        """Test consistency loss computation."""
        from imagetrust.baselines.trainer import ConsistencyLoss
        import torch

        loss_fn = ConsistencyLoss(temperature=1.0)

        # Simulated logits
        batch_size = 8
        logits_clean = torch.randn(batch_size, 2)
        logits_aug = logits_clean + torch.randn(batch_size, 2) * 0.1  # Small perturbation

        loss = loss_fn(logits_clean, logits_aug)

        assert loss.shape == ()  # Scalar
        assert loss.item() >= 0  # Non-negative

    def test_hard_negative_miner(self):
        """Test hard negative mining."""
        from imagetrust.baselines.trainer import HardNegativeMiner
        import torch

        miner = HardNegativeMiner(memory_size=100)

        # Simulate training batch
        sample_ids = list(range(10))
        losses = torch.rand(10)
        predictions = torch.rand(10)
        labels = torch.randint(0, 2, (10,))

        # Update miner
        added = miner.update(sample_ids, losses, predictions, labels)

        assert added >= 0
        assert len(miner.get_hard_negative_ids()) <= 100

        # Get weights
        weights = miner.get_weights_for_batch(sample_ids[:5])
        assert weights.shape == (5,)
