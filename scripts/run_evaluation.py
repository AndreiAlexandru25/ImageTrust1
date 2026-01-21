#!/usr/bin/env python3
"""
Run comprehensive evaluation for ImageTrust.

Includes:
- Cross-generator evaluation
- Degradation robustness testing
- Ablation study
- Calibration analysis

Usage:
    python scripts/run_evaluation.py --dataset ./data/test
    python scripts/run_evaluation.py --cross-generator
    python scripts/run_evaluation.py --ablation
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imagetrust.utils.logging import get_logger, setup_logging
from imagetrust.utils.helpers import ensure_dir

logger = get_logger(__name__)


def run_benchmark(args):
    """Run standard benchmark evaluation."""
    from imagetrust.detection import AIDetector
    from imagetrust.evaluation import Benchmark
    
    logger.info("Running benchmark evaluation...")
    
    # Initialize detector
    detector = AIDetector(model=args.model, device=args.device)
    
    # Initialize benchmark
    benchmark = Benchmark(output_dir=args.output_dir)
    
    # Add datasets from directory structure
    dataset_path = Path(args.dataset)
    
    # Expected structure: dataset/real/, dataset/ai/
    real_dir = dataset_path / "real"
    ai_dir = dataset_path / "ai"
    
    if real_dir.exists():
        benchmark.add_dataset_from_directory("real", real_dir, label=0, category="real")
    
    if ai_dir.exists():
        benchmark.add_dataset_from_directory("ai", ai_dir, label=1, category="ai")
    
    # Run benchmark
    results = benchmark.run(detector, batch_size=args.batch_size)
    
    # Save and print results
    output_path = benchmark.save_results(results)
    benchmark.print_summary()
    
    logger.info(f"Results saved to {output_path}")
    return results


def run_cross_generator(args):
    """Run cross-generator evaluation."""
    from imagetrust.detection import AIDetector
    from imagetrust.evaluation import CrossGeneratorEvaluator
    
    logger.info("Running cross-generator evaluation...")
    
    # Initialize
    detector = AIDetector(model=args.model, device=args.device)
    evaluator = CrossGeneratorEvaluator(output_dir=args.output_dir)
    
    # Add generators from directory structure
    # Expected: dataset/generator_name/
    dataset_path = Path(args.dataset)
    
    for subdir in dataset_path.iterdir():
        if subdir.is_dir():
            name = subdir.name.lower()
            label = 0 if name == "real" else 1
            evaluator.add_generator_from_directory(name, subdir, label=label)
    
    # Run evaluation
    results = evaluator.evaluate(detector, batch_size=args.batch_size)
    
    # Save and print results
    output_path = evaluator.save_results()
    evaluator.print_summary()
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Generalization score: {evaluator.get_generalization_score():.2%}")
    
    return results


def run_degradation(args):
    """Run degradation robustness evaluation."""
    from imagetrust.detection import AIDetector
    from imagetrust.evaluation import DegradationEvaluator
    
    logger.info("Running degradation robustness evaluation...")
    
    # Initialize
    detector = AIDetector(model=args.model, device=args.device)
    evaluator = DegradationEvaluator(output_dir=args.output_dir)
    
    # Add images
    dataset_path = Path(args.dataset)
    
    for label, label_name in [(0, "real"), (1, "ai")]:
        label_dir = dataset_path / label_name
        if label_dir.exists():
            images = list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.png"))
            labels = [label] * len(images)
            evaluator.add_images(images, labels)
    
    # Run evaluation
    results = evaluator.evaluate(detector)
    
    # Save and print results
    output_path = evaluator.save_results()
    evaluator.print_summary()
    
    logger.info(f"Results saved to {output_path}")
    return results


def run_ablation(args):
    """Run ablation study."""
    from imagetrust.detection import AIDetector
    from imagetrust.evaluation import AblationStudy
    
    logger.info("Running ablation study (MLE-STAR inspired)...")
    
    # Initialize detector
    detector = AIDetector(model=args.model, device=args.device)
    
    # Create simple validation dataset
    dataset_path = Path(args.dataset)
    
    class SimpleDataset:
        def __init__(self, path):
            self.items = []
            for label, name in [(0, "real"), (1, "ai")]:
                label_dir = path / name
                if label_dir.exists():
                    for img in list(label_dir.glob("*.jpg"))[:50]:
                        self.items.append((img, label))
        
        def __iter__(self):
            return iter(self.items)
        
        def __len__(self):
            return len(self.items)
    
    val_dataset = SimpleDataset(dataset_path)
    
    if len(val_dataset) == 0:
        logger.error("No validation images found")
        return None
    
    # Run ablation study
    ablation = AblationStudy(
        detector=detector,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
    )
    
    results = ablation.run_full_study()
    
    # Save and print results
    output_path = ablation.save_results(results)
    ablation.print_summary()
    
    logger.info(f"Results saved to {output_path}")
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run ImageTrust evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_evaluation.py --dataset ./data/test --benchmark
    python run_evaluation.py --dataset ./data/test --cross-generator
    python run_evaluation.py --dataset ./data/test --degradation
    python run_evaluation.py --dataset ./data/test --ablation
    python run_evaluation.py --dataset ./data/test --all
        """
    )
    
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="ensemble",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size for processing",
    )
    
    # Evaluation types
    parser.add_argument("--benchmark", action="store_true", help="Run standard benchmark")
    parser.add_argument("--cross-generator", action="store_true", help="Run cross-generator evaluation")
    parser.add_argument("--degradation", action="store_true", help="Run degradation evaluation")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    
    args = parser.parse_args()
    
    setup_logging(level="INFO")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = Path(args.output_dir) / timestamp
    ensure_dir(args.output_dir)
    
    logger.info(f"Output directory: {args.output_dir}")
    
    # Run selected evaluations
    results = {}
    
    if args.all or args.benchmark:
        results["benchmark"] = run_benchmark(args)
    
    if args.all or args.cross_generator:
        results["cross_generator"] = run_cross_generator(args)
    
    if args.all or args.degradation:
        results["degradation"] = run_degradation(args)
    
    if args.all or args.ablation:
        results["ablation"] = run_ablation(args)
    
    # If no specific evaluation selected, run benchmark
    if not any([args.benchmark, args.cross_generator, args.degradation, args.ablation, args.all]):
        results["benchmark"] = run_benchmark(args)
    
    # Save combined results
    combined_path = args.output_dir / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump({"timestamp": timestamp, "evaluations": list(results.keys())}, f, indent=2)
    
    logger.info(f"\n✅ All evaluations complete. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
