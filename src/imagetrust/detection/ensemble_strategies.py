"""
Ensemble strategies for combining multiple detector outputs.

Provides configurable ensemble methods for ablation studies and
optimal model combination in AI image detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EnsembleMethod(Enum):
    """Available ensemble methods."""

    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted"
    MAJORITY_VOTING = "voting"
    MAX_PROBABILITY = "max"
    MEDIAN = "median"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    SOFTMAX_WEIGHTED = "softmax_weighted"


@dataclass
class EnsembleResult:
    """Result from ensemble combination."""

    combined_probability: float
    method: str
    individual_probs: List[float]
    weights_used: List[float]
    agreement_score: float  # How much models agree (0-1)
    details: Dict[str, Any]


class EnsembleStrategy(ABC):
    """Abstract base class for ensemble strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @abstractmethod
    def combine(
        self,
        probabilities: List[float],
        weights: Optional[List[float]] = None,
        confidences: Optional[List[float]] = None,
    ) -> EnsembleResult:
        """
        Combine multiple probabilities into a single prediction.

        Args:
            probabilities: List of P(AI) from each model
            weights: Optional per-model weights
            confidences: Optional per-model confidence scores

        Returns:
            EnsembleResult with combined prediction
        """
        pass

    def _compute_agreement(self, probs: List[float], threshold: float = 0.5) -> float:
        """Compute agreement score (fraction of models that agree)."""
        if not probs:
            return 0.0
        votes = [1 if p > threshold else 0 for p in probs]
        majority = max(sum(votes), len(votes) - sum(votes))
        return majority / len(probs)


class AverageStrategy(EnsembleStrategy):
    """Simple averaging of probabilities."""

    @property
    def name(self) -> str:
        return "average"

    def combine(
        self,
        probabilities: List[float],
        weights: Optional[List[float]] = None,
        confidences: Optional[List[float]] = None,
    ) -> EnsembleResult:
        if not probabilities:
            return EnsembleResult(
                combined_probability=0.5,
                method=self.name,
                individual_probs=[],
                weights_used=[],
                agreement_score=0.0,
                details={},
            )

        combined = float(np.mean(probabilities))
        uniform_weights = [1.0 / len(probabilities)] * len(probabilities)

        return EnsembleResult(
            combined_probability=combined,
            method=self.name,
            individual_probs=probabilities,
            weights_used=uniform_weights,
            agreement_score=self._compute_agreement(probabilities),
            details={"std": float(np.std(probabilities))},
        )


class WeightedAverageStrategy(EnsembleStrategy):
    """Weighted averaging of probabilities."""

    @property
    def name(self) -> str:
        return "weighted"

    def combine(
        self,
        probabilities: List[float],
        weights: Optional[List[float]] = None,
        confidences: Optional[List[float]] = None,
    ) -> EnsembleResult:
        if not probabilities:
            return EnsembleResult(
                combined_probability=0.5,
                method=self.name,
                individual_probs=[],
                weights_used=[],
                agreement_score=0.0,
                details={},
            )

        # Default to uniform weights if not provided
        if weights is None:
            weights = [1.0] * len(probabilities)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)

        combined = float(np.average(probabilities, weights=normalized_weights))

        return EnsembleResult(
            combined_probability=combined,
            method=self.name,
            individual_probs=probabilities,
            weights_used=normalized_weights,
            agreement_score=self._compute_agreement(probabilities),
            details={
                "weighted_std": float(
                    np.sqrt(
                        np.average(
                            (np.array(probabilities) - combined) ** 2,
                            weights=normalized_weights,
                        )
                    )
                )
            },
        )


class MajorityVotingStrategy(EnsembleStrategy):
    """Majority voting with probability based on vote fraction."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "voting"

    def combine(
        self,
        probabilities: List[float],
        weights: Optional[List[float]] = None,
        confidences: Optional[List[float]] = None,
    ) -> EnsembleResult:
        if not probabilities:
            return EnsembleResult(
                combined_probability=0.5,
                method=self.name,
                individual_probs=[],
                weights_used=[],
                agreement_score=0.0,
                details={},
            )

        # Count votes
        ai_votes = sum(1 for p in probabilities if p > self.threshold)
        total = len(probabilities)

        # Combined probability is the vote fraction
        combined = ai_votes / total

        # Apply weights to votes if provided
        if weights is not None:
            weighted_ai_votes = sum(
                w for p, w in zip(probabilities, weights) if p > self.threshold
            )
            total_weight = sum(weights)
            combined = weighted_ai_votes / total_weight if total_weight > 0 else 0.5

        return EnsembleResult(
            combined_probability=combined,
            method=self.name,
            individual_probs=probabilities,
            weights_used=weights or [1.0] * total,
            agreement_score=self._compute_agreement(probabilities, self.threshold),
            details={
                "ai_votes": ai_votes,
                "real_votes": total - ai_votes,
                "total_models": total,
                "threshold": self.threshold,
            },
        )


class MaxProbabilityStrategy(EnsembleStrategy):
    """Use the maximum probability (most confident AI prediction)."""

    @property
    def name(self) -> str:
        return "max"

    def combine(
        self,
        probabilities: List[float],
        weights: Optional[List[float]] = None,
        confidences: Optional[List[float]] = None,
    ) -> EnsembleResult:
        if not probabilities:
            return EnsembleResult(
                combined_probability=0.5,
                method=self.name,
                individual_probs=[],
                weights_used=[],
                agreement_score=0.0,
                details={},
            )

        max_prob = float(np.max(probabilities))
        max_idx = int(np.argmax(probabilities))

        return EnsembleResult(
            combined_probability=max_prob,
            method=self.name,
            individual_probs=probabilities,
            weights_used=[1.0 if i == max_idx else 0.0 for i in range(len(probabilities))],
            agreement_score=self._compute_agreement(probabilities),
            details={
                "max_model_index": max_idx,
                "min_prob": float(np.min(probabilities)),
                "range": float(np.max(probabilities) - np.min(probabilities)),
            },
        )


class MedianStrategy(EnsembleStrategy):
    """Use the median probability (robust to outliers)."""

    @property
    def name(self) -> str:
        return "median"

    def combine(
        self,
        probabilities: List[float],
        weights: Optional[List[float]] = None,
        confidences: Optional[List[float]] = None,
    ) -> EnsembleResult:
        if not probabilities:
            return EnsembleResult(
                combined_probability=0.5,
                method=self.name,
                individual_probs=[],
                weights_used=[],
                agreement_score=0.0,
                details={},
            )

        combined = float(np.median(probabilities))

        return EnsembleResult(
            combined_probability=combined,
            method=self.name,
            individual_probs=probabilities,
            weights_used=[1.0 / len(probabilities)] * len(probabilities),
            agreement_score=self._compute_agreement(probabilities),
            details={
                "q1": float(np.percentile(probabilities, 25)),
                "q3": float(np.percentile(probabilities, 75)),
                "iqr": float(np.percentile(probabilities, 75) - np.percentile(probabilities, 25)),
            },
        )


class ConfidenceWeightedStrategy(EnsembleStrategy):
    """Weight probabilities by model confidence scores."""

    @property
    def name(self) -> str:
        return "confidence_weighted"

    def combine(
        self,
        probabilities: List[float],
        weights: Optional[List[float]] = None,
        confidences: Optional[List[float]] = None,
    ) -> EnsembleResult:
        if not probabilities:
            return EnsembleResult(
                combined_probability=0.5,
                method=self.name,
                individual_probs=[],
                weights_used=[],
                agreement_score=0.0,
                details={},
            )

        # Use confidences as weights if provided, else use uniform
        if confidences is not None:
            effective_weights = confidences
        elif weights is not None:
            effective_weights = weights
        else:
            effective_weights = [1.0] * len(probabilities)

        # Normalize
        total = sum(effective_weights)
        if total > 0:
            normalized = [w / total for w in effective_weights]
        else:
            normalized = [1.0 / len(effective_weights)] * len(effective_weights)

        combined = float(np.average(probabilities, weights=normalized))

        return EnsembleResult(
            combined_probability=combined,
            method=self.name,
            individual_probs=probabilities,
            weights_used=normalized,
            agreement_score=self._compute_agreement(probabilities),
            details={
                "confidence_weights": effective_weights,
                "highest_confidence_idx": int(np.argmax(effective_weights)),
            },
        )


class SoftmaxWeightedStrategy(EnsembleStrategy):
    """Apply softmax to confidences for sharper weighting."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "softmax_weighted"

    def combine(
        self,
        probabilities: List[float],
        weights: Optional[List[float]] = None,
        confidences: Optional[List[float]] = None,
    ) -> EnsembleResult:
        if not probabilities:
            return EnsembleResult(
                combined_probability=0.5,
                method=self.name,
                individual_probs=[],
                weights_used=[],
                agreement_score=0.0,
                details={},
            )

        # Use confidences or weights for softmax
        if confidences is not None:
            scores = np.array(confidences)
        elif weights is not None:
            scores = np.array(weights)
        else:
            scores = np.ones(len(probabilities))

        # Apply softmax with temperature
        exp_scores = np.exp(scores / self.temperature)
        softmax_weights = exp_scores / exp_scores.sum()

        combined = float(np.average(probabilities, weights=softmax_weights))

        return EnsembleResult(
            combined_probability=combined,
            method=self.name,
            individual_probs=probabilities,
            weights_used=softmax_weights.tolist(),
            agreement_score=self._compute_agreement(probabilities),
            details={
                "temperature": self.temperature,
                "softmax_weights": softmax_weights.tolist(),
            },
        )


# Factory function
def create_ensemble_strategy(
    method: str,
    **kwargs,
) -> EnsembleStrategy:
    """
    Create an ensemble strategy by name.

    Args:
        method: Strategy name (average, weighted, voting, max, median,
                confidence_weighted, softmax_weighted)
        **kwargs: Additional arguments for specific strategies

    Returns:
        EnsembleStrategy instance
    """
    strategies = {
        "average": AverageStrategy,
        "weighted": WeightedAverageStrategy,
        "voting": MajorityVotingStrategy,
        "max": MaxProbabilityStrategy,
        "median": MedianStrategy,
        "confidence_weighted": ConfidenceWeightedStrategy,
        "softmax_weighted": SoftmaxWeightedStrategy,
    }

    if method not in strategies:
        raise ValueError(f"Unknown ensemble method: {method}. Available: {list(strategies.keys())}")

    strategy_class = strategies[method]

    # Handle special kwargs
    if method == "voting" and "threshold" in kwargs:
        return strategy_class(threshold=kwargs["threshold"])
    elif method == "softmax_weighted" and "temperature" in kwargs:
        return strategy_class(temperature=kwargs["temperature"])

    return strategy_class()


def get_available_strategies() -> List[str]:
    """Get list of available ensemble strategy names."""
    return [
        "average",
        "weighted",
        "voting",
        "max",
        "median",
        "confidence_weighted",
        "softmax_weighted",
    ]


def compare_strategies(
    probabilities: List[float],
    weights: Optional[List[float]] = None,
    confidences: Optional[List[float]] = None,
) -> Dict[str, EnsembleResult]:
    """
    Compare all ensemble strategies on the same input.

    Useful for ablation studies to see how strategy choice affects results.

    Args:
        probabilities: List of P(AI) from each model
        weights: Optional per-model weights
        confidences: Optional per-model confidence scores

    Returns:
        Dictionary mapping strategy name to EnsembleResult
    """
    results = {}
    for strategy_name in get_available_strategies():
        strategy = create_ensemble_strategy(strategy_name)
        results[strategy_name] = strategy.combine(probabilities, weights, confidences)
    return results
