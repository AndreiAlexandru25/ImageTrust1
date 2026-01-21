"""
Cross-generator evaluation module.

Evaluates model generalization across different AI generators.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from imagetrust.evaluation.benchmark import Benchmark
from imagetrust.evaluation.metrics import compute_metrics
from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)


class CrossGeneratorEvaluator:
    """
    Evaluates AI detection across different generators.
    
    Tests how well a model generalizes to AI generators
    not seen during training.
    
    Example:
        >>> evaluator = CrossGeneratorEvaluator(output_dir="results")
        >>> evaluator.add_generator_from_directory("midjourney", "data/midjourney")
        >>> evaluator.add_generator_from_directory("dalle3", "data/dalle3")
        >>> results = evaluator.evaluate(detector)
    """

    def __init__(
        self,
        output_dir: Optional[Union[Path, str]] = None,
        verbose: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("results/cross_generator")
        self.verbose = verbose
        
        self.generators: Dict[str, Dict[str, Any]] = {}
        self.results: Optional[Dict[str, Any]] = None
        
        # Underlying benchmark
        self.benchmark = Benchmark(output_dir=self.output_dir, verbose=verbose)

    def add_generator_from_directory(
        self,
        name: str,
        directory: Union[Path, str],
        label: int = 1,
    ) -> None:
        """
        Add a generator dataset from directory.
        
        Args:
            name: Generator name (e.g., "midjourney_v5")
            directory: Path to images
            label: Label (typically 1 for AI, 0 for real)
        """
        self.generators[name] = {
            "directory": Path(directory),
            "label": label,
        }
        
        category = "real" if label == 0 else f"ai_{name}"
        self.benchmark.add_dataset_from_directory(
            name=name,
            directory=directory,
            label=label,
            category=category,
        )

    def evaluate(
        self,
        detector,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """
        Run cross-generator evaluation.
        
        Args:
            detector: AIDetector instance
            batch_size: Batch size
            
        Returns:
            Evaluation results
        """
        logger.info(f"Running cross-generator evaluation on {len(self.generators)} generators")
        
        # Run benchmark
        results = self.benchmark.run(detector, batch_size=batch_size)
        
        # Add cross-generator specific analysis
        generator_scores = {}
        for gen_name in self.generators:
            if gen_name in results.get("datasets", {}):
                gen_results = results["datasets"][gen_name]
                metrics = gen_results.get("metrics", {})
                generator_scores[gen_name] = {
                    "accuracy": metrics.get("accuracy", 0),
                    "f1_score": metrics.get("f1_score", 0),
                    "roc_auc": metrics.get("roc_auc", 0),
                    "num_samples": gen_results.get("num_samples", 0),
                }
        
        results["generator_analysis"] = generator_scores
        results["generalization_score"] = self._compute_generalization_score(generator_scores)
        
        self.results = results
        return results

    def _compute_generalization_score(
        self,
        generator_scores: Dict[str, Dict[str, float]],
    ) -> float:
        """
        Compute overall generalization score.
        
        Average F1 across all generators, weighted by sample count.
        """
        if not generator_scores:
            return 0.0
        
        total_samples = sum(g["num_samples"] for g in generator_scores.values())
        if total_samples == 0:
            return 0.0
        
        weighted_f1 = sum(
            g["f1_score"] * g["num_samples"]
            for g in generator_scores.values()
        )
        
        return weighted_f1 / total_samples

    def get_generalization_score(self) -> float:
        """Get the generalization score from results."""
        if self.results is None:
            return 0.0
        return self.results.get("generalization_score", 0.0)

    def save_results(self, filename: str = "cross_generator_results.json") -> Path:
        """Save results to file."""
        return self.benchmark.save_results(self.results, filename)

    def print_summary(self) -> None:
        """Print evaluation summary."""
        if self.results is None:
            print("No results available. Run evaluation first.")
            return
        
        print("\n" + "=" * 60)
        print("CROSS-GENERATOR EVALUATION SUMMARY")
        print("=" * 60)
        
        gen_analysis = self.results.get("generator_analysis", {})
        
        print("\nPer-Generator Results")
        print("-" * 50)
        print(f"{'Generator':<20} {'Accuracy':>10} {'F1':>10} {'Samples':>10}")
        print("-" * 50)
        
        for gen_name, scores in gen_analysis.items():
            print(f"{gen_name:<20} {scores['accuracy']:>10.2%} "
                  f"{scores['f1_score']:>10.2%} {scores['num_samples']:>10}")
        
        print("-" * 50)
        print(f"\nGeneralization Score: {self.get_generalization_score():.2%}")
        print("=" * 60 + "\n")
