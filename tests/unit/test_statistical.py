"""
Unit tests for statistical tests module.

Tests for publication-required significance testing:
- McNemar's test
- DeLong's test
- Bootstrap confidence intervals
- Permutation tests
"""

import pytest
import numpy as np


class TestMcNemarTest:
    """Tests for McNemar's test."""

    def test_identical_predictions(self):
        """Test with identical predictions (no difference)."""
        from imagetrust.evaluation.statistical_tests import mcnemar_test

        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        pred_a = np.array([0, 0, 1, 1, 0, 1, 0, 1])  # Same as pred_b
        pred_b = np.array([0, 0, 1, 1, 0, 1, 0, 1])

        result = mcnemar_test(y_true, pred_a, pred_b)

        assert result.chi2_statistic == 0.0
        assert result.p_value == 1.0
        assert result.significant is False
        assert result.n_disagreements == 0

    def test_clear_difference(self):
        """Test with clear difference between classifiers."""
        from imagetrust.evaluation.statistical_tests import mcnemar_test

        np.random.seed(42)
        n = 100

        y_true = np.random.randint(0, 2, n)

        # Model A: 80% accuracy
        pred_a = y_true.copy()
        pred_a[:20] = 1 - pred_a[:20]

        # Model B: 60% accuracy
        pred_b = y_true.copy()
        pred_b[:40] = 1 - pred_b[:40]

        result = mcnemar_test(y_true, pred_a, pred_b)

        # A should be better than B
        assert result.n_10 > result.n_01  # A correct when B wrong

    def test_disagreement_counts(self):
        """Test that disagreement counts are correct."""
        from imagetrust.evaluation.statistical_tests import mcnemar_test

        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        pred_a = np.array([1, 1, 0, 0, 0, 0, 1, 1])  # 4 correct
        pred_b = np.array([1, 0, 1, 0, 0, 1, 0, 1])  # 4 correct

        result = mcnemar_test(y_true, pred_a, pred_b)

        # Check that all disagreements are counted
        assert result.n_01 + result.n_10 == result.n_disagreements

    def test_continuity_correction(self):
        """Test with and without continuity correction."""
        from imagetrust.evaluation.statistical_tests import mcnemar_test

        # Use data where n_01 != n_10 so correction reduces chi2
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        pred_a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        pred_b = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0])

        result_with = mcnemar_test(y_true, pred_a, pred_b, continuity_correction=True)
        result_without = mcnemar_test(y_true, pred_a, pred_b, continuity_correction=False)

        # With correction should have smaller or equal chi2
        assert result_with.chi2_statistic <= result_without.chi2_statistic


class TestDeLongTest:
    """Tests for DeLong's test."""

    def test_identical_models(self):
        """Test with identical models."""
        from imagetrust.evaluation.statistical_tests import delong_test

        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        result = delong_test(y_true, probs, probs)

        assert result.auc_difference == 0.0
        assert result.significant == False

    def test_different_aucs(self):
        """Test with clearly different AUCs."""
        from imagetrust.evaluation.statistical_tests import delong_test

        np.random.seed(42)
        n = 100
        y_true = np.concatenate([np.zeros(50), np.ones(50)])

        # Model A: Good separation (high AUC)
        prob_a = np.concatenate([
            np.random.uniform(0.1, 0.4, 50),
            np.random.uniform(0.6, 0.9, 50),
        ])

        # Model B: Poor separation (lower AUC)
        prob_b = np.concatenate([
            np.random.uniform(0.3, 0.6, 50),
            np.random.uniform(0.4, 0.7, 50),
        ])

        result = delong_test(y_true, prob_a, prob_b)

        assert result.auc_a > result.auc_b
        assert result.auc_difference > 0

    def test_confidence_interval(self):
        """Test confidence interval computation."""
        from imagetrust.evaluation.statistical_tests import delong_test

        np.random.seed(42)
        y_true = np.concatenate([np.zeros(30), np.ones(30)])
        prob_a = np.concatenate([np.random.uniform(0, 0.5, 30), np.random.uniform(0.5, 1, 30)])
        prob_b = np.concatenate([np.random.uniform(0, 0.6, 30), np.random.uniform(0.4, 1, 30)])

        result = delong_test(y_true, prob_a, prob_b)

        # CI should contain the point estimate
        assert result.ci_lower <= result.auc_difference <= result.ci_upper


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_basic_ci(self):
        """Test basic confidence interval computation."""
        from imagetrust.evaluation.statistical_tests import bootstrap_ci
        from sklearn.metrics import accuracy_score

        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_pred = y_true.copy()
        y_pred[:10] = 1 - y_pred[:10]  # 90% accuracy

        def accuracy_fn(yt, yp):
            return accuracy_score(yt, (yp > 0.5).astype(int))

        result = bootstrap_ci(
            accuracy_fn,
            y_true,
            y_pred.astype(float),
            n_bootstrap=500,
        )

        # Point estimate should be around 0.9
        assert 0.85 <= result.estimate <= 0.95

        # CI should contain point estimate
        assert result.ci_lower <= result.estimate <= result.ci_upper

        # CI should be reasonable width
        ci_width = result.ci_upper - result.ci_lower
        assert 0 < ci_width < 0.3

    def test_confidence_level(self):
        """Test different confidence levels."""
        from imagetrust.evaluation.statistical_tests import bootstrap_ci

        np.random.seed(42)
        y_true = np.random.randint(0, 2, 50)
        y_pred = np.random.uniform(0, 1, 50)

        def mean_fn(yt, yp):
            return np.mean(yp)

        result_95 = bootstrap_ci(mean_fn, y_true, y_pred, confidence_level=0.95, n_bootstrap=500)
        result_80 = bootstrap_ci(mean_fn, y_true, y_pred, confidence_level=0.80, n_bootstrap=500)

        # 95% CI should be wider than 80% CI
        width_95 = result_95.ci_upper - result_95.ci_lower
        width_80 = result_80.ci_upper - result_80.ci_lower
        assert width_95 >= width_80


class TestPermutationTest:
    """Tests for permutation test."""

    def test_identical_models(self):
        """Test with identical predictions."""
        from imagetrust.evaluation.statistical_tests import permutation_test
        from sklearn.metrics import accuracy_score

        y_true = np.array([0, 0, 1, 1, 0, 1])
        pred = np.array([0, 0, 1, 1, 0, 1])

        def accuracy_fn(yt, yp):
            return accuracy_score(yt, (yp > 0.5).astype(int))

        result = permutation_test(
            accuracy_fn,
            y_true,
            pred.astype(float),
            pred.astype(float),
            n_permutations=500,
        )

        assert result.observed_difference == 0.0
        assert result.significant == False

    def test_different_models(self):
        """Test with clearly different predictions."""
        from imagetrust.evaluation.statistical_tests import permutation_test
        from sklearn.metrics import accuracy_score

        np.random.seed(42)
        n = 50
        y_true = np.random.randint(0, 2, n)

        # Model A: Good
        pred_a = y_true.astype(float)

        # Model B: Random
        pred_b = np.random.uniform(0, 1, n)

        def accuracy_fn(yt, yp):
            return accuracy_score(yt, (yp > 0.5).astype(int))

        result = permutation_test(
            accuracy_fn,
            y_true,
            pred_a,
            pred_b,
            n_permutations=500,
        )

        assert result.observed_difference > 0  # A should be better


class TestPairwiseSignificance:
    """Tests for pairwise significance computation."""

    def test_multiple_models(self):
        """Test with multiple models."""
        from imagetrust.evaluation.statistical_tests import compute_pairwise_significance

        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)

        predictions = {
            "ModelA": y_true.copy(),
            "ModelB": y_true.copy(),
            "ModelC": np.random.randint(0, 2, n),
        }
        predictions["ModelA"][:5] = 1 - predictions["ModelA"][:5]
        predictions["ModelB"][:15] = 1 - predictions["ModelB"][:15]

        probabilities = {k: v.astype(float) for k, v in predictions.items()}

        results = compute_pairwise_significance(
            y_true,
            predictions,
            probabilities,
        )

        assert "mcnemar" in results
        assert "delong" in results
        assert "summary" in results
        assert results["summary"]["total_comparisons"] == 3  # 3 pairs

    def test_reference_model(self):
        """Test comparison against reference model."""
        from imagetrust.evaluation.statistical_tests import compute_pairwise_significance

        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)

        predictions = {
            "Reference": y_true.copy(),
            "ModelA": y_true.copy(),
            "ModelB": np.random.randint(0, 2, n),
        }

        probabilities = {k: v.astype(float) for k, v in predictions.items()}

        results = compute_pairwise_significance(
            y_true,
            predictions,
            probabilities,
            reference_model="Reference",
        )

        # Only comparisons against reference
        assert results["summary"]["total_comparisons"] == 2


class TestFormatSignificanceTable:
    """Tests for table formatting."""

    def test_markdown_format(self):
        """Test markdown table generation."""
        from imagetrust.evaluation.statistical_tests import (
            compute_pairwise_significance,
            format_significance_table,
        )

        np.random.seed(42)
        n = 50
        y_true = np.random.randint(0, 2, n)
        predictions = {
            "A": y_true.copy(),
            "B": np.random.randint(0, 2, n),
        }
        probabilities = {k: v.astype(float) for k, v in predictions.items()}

        results = compute_pairwise_significance(y_true, predictions, probabilities)
        table = format_significance_table(results)

        assert "| Comparison |" in table
        assert "McNemar" in table
