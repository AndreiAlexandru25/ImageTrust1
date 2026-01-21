"""
Benchmarking framework for AI detection models.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from imagetrust.core.config import get_settings
from imagetrust.evaluation.metrics import compute_metrics, compute_calibration_metrics
from imagetrust.utils.logging import get_logger
from imagetrust.utils.helpers import ensure_dir, load_image

logger = get_logger(__name__)


class Benchmark:
    """
    Benchmarking framework for evaluating AI detection models.
    
    Example:
        >>> benchmark = Benchmark(output_dir="results")
        >>> benchmark.add_dataset_from_directory("real", "data/real", label=0)
        >>> benchmark.add_dataset_from_directory("ai", "data/ai", label=1)
        >>> results = benchmark.run(detector)
        >>> benchmark.print_summary()
    """

    def __init__(
        self,
        output_dir: Optional[Union[Path, str]] = None,
        verbose: bool = True,
    ) -> None:
        self.settings = get_settings()
        self.output_dir = Path(output_dir) if output_dir else self.settings.outputs_dir
        self.verbose = verbose
        
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.results: Optional[Dict[str, Any]] = None

    def add_dataset(
        self,
        name: str,
        images: List[Union[Path, str]],
        labels: Union[int, List[int]],
        category: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a dataset to the benchmark.
        
        Args:
            name: Dataset name
            images: List of image paths
            labels: Single label or list of labels
            category: Dataset category (e.g., "real", "ai_generated")
            metadata: Optional metadata
        """
        if isinstance(labels, int):
            labels = [labels] * len(images)
        
        self.datasets[name] = {
            "images": [Path(p) for p in images],
            "labels": labels,
            "category": category,
            "metadata": metadata or {},
        }
        
        logger.info(f"Added dataset '{name}': {len(images)} images")

    def add_dataset_from_directory(
        self,
        name: str,
        directory: Union[Path, str],
        label: int,
        category: Optional[str] = None,
        extensions: List[str] = [".jpg", ".jpeg", ".png", ".webp"],
    ) -> None:
        """
        Add a dataset from a directory.
        
        Args:
            name: Dataset name
            directory: Path to directory
            label: Label for all images (0=real, 1=AI)
            category: Dataset category
            extensions: Valid image extensions
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return
        
        images = []
        for ext in extensions:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        
        if not images:
            logger.warning(f"No images found in {directory}")
            return
        
        self.add_dataset(
            name=name,
            images=images,
            labels=label,
            category=category or ("real" if label == 0 else "ai_generated"),
        )

    def run(
        self,
        detector,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """
        Run the benchmark.
        
        Args:
            detector: AIDetector instance
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Running benchmark on {len(self.datasets)} datasets")
        
        all_labels = []
        all_preds = []
        all_probs = []
        dataset_results = {}
        
        for dataset_name, dataset in self.datasets.items():
            logger.info(f"Evaluating dataset: {dataset_name}")
            
            labels = []
            preds = []
            probs = []
            
            images = dataset["images"]
            iterator = tqdm(images, desc=dataset_name, disable=not self.verbose)
            
            for img_path in iterator:
                try:
                    result = detector.detect(img_path)
                    ai_prob = result["ai_probability"]
                    
                    labels.append(dataset["labels"][images.index(img_path)])
                    probs.append(ai_prob)
                    preds.append(1 if ai_prob > 0.5 else 0)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
            
            # Compute dataset metrics
            if labels:
                metrics = compute_metrics(
                    np.array(labels),
                    np.array(preds),
                    np.array(probs),
                )
                dataset_results[dataset_name] = {
                    "metrics": metrics,
                    "num_samples": len(labels),
                    "category": dataset["category"],
                }
                
                all_labels.extend(labels)
                all_preds.extend(preds)
                all_probs.extend(probs)
        
        # Compute overall metrics
        overall_metrics = {}
        if all_labels:
            overall_metrics = compute_metrics(
                np.array(all_labels),
                np.array(all_preds),
                np.array(all_probs),
            )
            overall_metrics["calibration"] = compute_calibration_metrics(
                np.array(all_labels),
                np.array(all_probs),
            )
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model": detector.get_model_info() if hasattr(detector, "get_model_info") else {},
            "datasets": dataset_results,
            "overall": overall_metrics,
            "total_samples": len(all_labels),
        }
        
        return self.results

    def save_results(
        self,
        results: Optional[Dict[str, Any]] = None,
        filename: str = "benchmark_results.json",
    ) -> Path:
        """Save results to JSON file."""
        results = results or self.results
        if results is None:
            raise ValueError("No results to save. Run benchmark first.")
        
        ensure_dir(self.output_dir)
        output_path = self.output_dir / filename
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        return output_path

    def print_summary(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Print a summary of benchmark results."""
        results = results or self.results
        if results is None:
            print("No results available. Run benchmark first.")
            return
        
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        overall = results.get("overall", {})
        print(f"\nOverall Results ({results.get('total_samples', 0)} samples)")
        print("-" * 40)
        print(f"  Accuracy:    {overall.get('accuracy', 0):.2%}")
        print(f"  Precision:   {overall.get('precision', 0):.2%}")
        print(f"  Recall:      {overall.get('recall', 0):.2%}")
        print(f"  F1 Score:    {overall.get('f1_score', 0):.2%}")
        print(f"  ROC-AUC:     {overall.get('roc_auc', 0):.3f}")
        
        if "calibration" in overall:
            cal = overall["calibration"]
            print(f"  ECE:         {cal.get('ece', 0):.4f}")
            print(f"  Brier Score: {cal.get('brier_score', 0):.4f}")
        
        print("\nPer-Dataset Results")
        print("-" * 40)
        for name, data in results.get("datasets", {}).items():
            metrics = data.get("metrics", {})
            print(f"  {name}: Acc={metrics.get('accuracy', 0):.2%}, "
                  f"F1={metrics.get('f1_score', 0):.2%}, "
                  f"N={data.get('num_samples', 0)}")
        
        print("=" * 60 + "\n")
