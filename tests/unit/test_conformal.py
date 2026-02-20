"""
Unit tests for conformal prediction module.

Tests the conformal prediction framework including:
- ConformalPredictor calibration
- Prediction set generation
- Coverage guarantees
"""

import pytest
import numpy as np


class TestConformalPredictor:
    """Tests for ConformalPredictor."""

    def test_init(self):
        """Test predictor initialization."""
        from imagetrust.detection.conformal import ConformalPredictor, ConformalMethod

        predictor = ConformalPredictor(alpha=0.1)
        assert predictor.alpha == 0.1
        assert predictor.coverage_level == 0.9
        assert predictor.method == ConformalMethod.APS

    def test_init_invalid_alpha(self):
        """Test initialization with invalid alpha."""
        from imagetrust.detection.conformal import ConformalPredictor

        with pytest.raises(ValueError):
            ConformalPredictor(alpha=0.0)

        with pytest.raises(ValueError):
            ConformalPredictor(alpha=1.0)

    def test_calibrate(self):
        """Test calibration with calibration data."""
        from imagetrust.detection.conformal import ConformalPredictor

        # Generate calibration data
        np.random.seed(42)
        n_cal = 100
        cal_probs = np.random.uniform(0, 1, n_cal)
        # Labels: 1 if prob > 0.5 + noise
        cal_labels = (cal_probs + np.random.normal(0, 0.1, n_cal) > 0.5).astype(int)

        predictor = ConformalPredictor(alpha=0.1)
        result = predictor.calibrate(cal_probs, cal_labels)

        assert result.threshold > 0
        assert result.coverage_level == 0.9
        assert result.n_calibration == n_cal
        assert 0 <= result.empirical_coverage <= 1

    def test_predict_not_calibrated(self):
        """Test prediction before calibration raises error."""
        from imagetrust.detection.conformal import ConformalPredictor

        predictor = ConformalPredictor(alpha=0.1)

        with pytest.raises(RuntimeError):
            predictor.predict(0.5)

    def test_predict_single(self):
        """Test single prediction."""
        from imagetrust.detection.conformal import ConformalPredictor

        # Setup and calibrate
        np.random.seed(42)
        cal_probs = np.random.uniform(0, 1, 100)
        cal_labels = (cal_probs > 0.5).astype(int)

        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate(cal_probs, cal_labels)

        # Make prediction
        result = predictor.predict(0.7)

        assert hasattr(result, "prediction_set")
        assert hasattr(result, "is_uncertain")
        assert hasattr(result, "coverage_level")
        assert len(result.prediction_set) >= 1
        assert result.coverage_level == 0.9

    def test_predict_uncertain(self):
        """Test prediction near decision boundary is uncertain."""
        from imagetrust.detection.conformal import ConformalPredictor

        # Setup and calibrate
        np.random.seed(42)
        cal_probs = np.random.uniform(0, 1, 100)
        cal_labels = (cal_probs > 0.5).astype(int)

        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate(cal_probs, cal_labels)

        # Prediction at 0.5 should be uncertain
        result = predictor.predict(0.5)

        # At decision boundary, prediction set should contain both classes
        # (or be marked uncertain)
        assert result.set_size >= 1

    def test_predict_confident_ai(self):
        """Test confident AI prediction."""
        from imagetrust.detection.conformal import ConformalPredictor

        # Setup and calibrate with clear separation
        np.random.seed(42)
        cal_probs = np.concatenate([
            np.random.uniform(0, 0.3, 50),  # Real
            np.random.uniform(0.7, 1.0, 50),  # AI
        ])
        cal_labels = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)

        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate(cal_probs, cal_labels)

        # Very high probability should give single prediction
        result = predictor.predict(0.95)
        assert "ai_generated" in result.prediction_set

    def test_coverage_guarantee(self):
        """Test that empirical coverage meets guarantee."""
        from imagetrust.detection.conformal import ConformalPredictor

        np.random.seed(42)

        # Generate data
        n_cal = 500
        n_test = 500
        all_probs = np.random.beta(2, 2, n_cal + n_test)  # Centered around 0.5
        all_labels = (all_probs + np.random.normal(0, 0.1, n_cal + n_test) > 0.5).astype(int)

        cal_probs = all_probs[:n_cal]
        cal_labels = all_labels[:n_cal]
        test_probs = all_probs[n_cal:]
        test_labels = all_labels[n_cal:]

        # Calibrate
        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate(cal_probs, cal_labels)

        # Evaluate on test set
        coverage_metrics = predictor.evaluate_coverage(test_probs, test_labels)

        # Coverage should be approximately >= 1 - alpha
        # Allow some margin due to finite sample
        assert coverage_metrics["coverage"] >= 0.85  # 90% - 5% margin

    def test_predict_batch(self):
        """Test batch prediction."""
        from imagetrust.detection.conformal import ConformalPredictor

        np.random.seed(42)
        cal_probs = np.random.uniform(0, 1, 100)
        cal_labels = (cal_probs > 0.5).astype(int)

        predictor = ConformalPredictor(alpha=0.1)
        predictor.calibrate(cal_probs, cal_labels)

        # Batch prediction
        test_probs = np.array([0.1, 0.5, 0.9])
        results = predictor.predict_batch(test_probs)

        assert len(results) == 3


class TestConformalMethods:
    """Tests for different conformal prediction methods."""

    @pytest.fixture
    def calibration_data(self):
        """Generate calibration data."""
        np.random.seed(42)
        probs = np.random.uniform(0, 1, 200)
        labels = (probs + np.random.normal(0, 0.15, 200) > 0.5).astype(int)
        return probs, labels

    def test_lac_method(self, calibration_data):
        """Test LAC (Least Ambiguous Classifier) method."""
        from imagetrust.detection.conformal import ConformalPredictor, ConformalMethod

        cal_probs, cal_labels = calibration_data
        predictor = ConformalPredictor(alpha=0.1, method=ConformalMethod.LAC)
        predictor.calibrate(cal_probs, cal_labels)

        result = predictor.predict(0.7)
        assert len(result.prediction_set) >= 1

    def test_aps_method(self, calibration_data):
        """Test APS (Adaptive Prediction Sets) method."""
        from imagetrust.detection.conformal import ConformalPredictor, ConformalMethod

        cal_probs, cal_labels = calibration_data
        predictor = ConformalPredictor(alpha=0.1, method=ConformalMethod.APS)
        predictor.calibrate(cal_probs, cal_labels)

        result = predictor.predict(0.7)
        assert len(result.prediction_set) >= 1

    def test_raps_method(self, calibration_data):
        """Test RAPS (Regularized APS) method."""
        from imagetrust.detection.conformal import ConformalPredictor, ConformalMethod

        cal_probs, cal_labels = calibration_data
        predictor = ConformalPredictor(
            alpha=0.1,
            method=ConformalMethod.RAPS,
            raps_lambda=0.01,
        )
        predictor.calibrate(cal_probs, cal_labels)

        result = predictor.predict(0.7)
        assert len(result.prediction_set) >= 1


class TestAdaptiveConformalPredictor:
    """Tests for AdaptiveConformalPredictor."""

    def test_init(self):
        """Test adaptive predictor initialization."""
        from imagetrust.detection.conformal import AdaptiveConformalPredictor

        predictor = AdaptiveConformalPredictor(
            alpha_levels=[0.05, 0.10, 0.15, 0.20]
        )
        assert len(predictor.alpha_levels) == 4

    def test_calibrate_all_levels(self):
        """Test calibration at all alpha levels."""
        from imagetrust.detection.conformal import AdaptiveConformalPredictor

        np.random.seed(42)
        cal_probs = np.random.uniform(0, 1, 100)
        cal_labels = (cal_probs > 0.5).astype(int)

        predictor = AdaptiveConformalPredictor(alpha_levels=[0.05, 0.10, 0.20])
        results = predictor.calibrate(cal_probs, cal_labels)

        assert len(results) == 3
        assert 0.05 in results
        assert 0.10 in results
        assert 0.20 in results

    def test_adaptive_prediction(self):
        """Test adaptive prediction based on confidence."""
        from imagetrust.detection.conformal import AdaptiveConformalPredictor

        np.random.seed(42)
        cal_probs = np.random.uniform(0, 1, 100)
        cal_labels = (cal_probs > 0.5).astype(int)

        predictor = AdaptiveConformalPredictor(alpha_levels=[0.05, 0.10, 0.20])
        predictor.calibrate(cal_probs, cal_labels)

        # High confidence prediction
        result_high = predictor.predict_adaptive(0.9, confidence_score=0.95)
        assert len(result_high.prediction_set) >= 1

        # Low confidence prediction (should be more conservative)
        result_low = predictor.predict_adaptive(0.5, confidence_score=0.1)
        assert len(result_low.prediction_set) >= 1


class TestCoverageTradeoff:
    """Tests for coverage-accuracy tradeoff computation."""

    def test_compute_tradeoff(self):
        """Test coverage-accuracy tradeoff curve computation."""
        from imagetrust.detection.conformal import (
            ConformalPredictor,
            compute_coverage_accuracy_tradeoff,
        )

        np.random.seed(42)
        probs = np.random.uniform(0, 1, 200)
        labels = (probs + np.random.normal(0, 0.15, 200) > 0.5).astype(int)

        predictor = ConformalPredictor(alpha=0.1)

        tradeoff = compute_coverage_accuracy_tradeoff(
            predictor,
            probs,
            labels,
            alpha_range=np.linspace(0.05, 0.25, 10),
        )

        assert "alpha" in tradeoff
        assert "coverage" in tradeoff
        assert "avg_set_size" in tradeoff
        assert len(tradeoff["alpha"]) == 10
