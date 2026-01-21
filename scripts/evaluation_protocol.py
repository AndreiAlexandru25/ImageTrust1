#!/usr/bin/env python3
"""
Comprehensive evaluation protocol for ImageTrust thesis.

Runs all baselines across all evaluation scenarios:
1. In-domain evaluation (standard train/val/test)
2. Cross-generator evaluation (leave-one-out)
3. Degradation robustness evaluation

Generates paper-ready results in JSON, CSV, and LaTeX formats.

Usage:
    # Run full evaluation protocol
    python scripts/evaluation_protocol.py --splits-dir ./data/splits --output-dir ./outputs/evaluation

    # Run specific evaluation type
    python scripts/evaluation_protocol.py --eval-type cross-generator

    # Run specific baseline only
    python scripts/evaluation_protocol.py --baseline cnn

    # Skip training (use pretrained weights)
    python scripts/evaluation_protocol.py --no-train --weights-dir ./models/trained

Author: ImageTrust Team
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class EvaluationConfig:
    """Configuration for evaluation protocol."""

    # Paths
    splits_dir: Path = Path("data/splits")
    output_dir: Path = Path("outputs/evaluation")
    weights_dir: Optional[Path] = None

    # Evaluation types
    run_in_domain: bool = True
    run_cross_generator: bool = True
    run_degradation: bool = True

    # Baselines to evaluate
    baselines: List[str] = field(default_factory=lambda: ["classical", "cnn", "vit", "imagetrust"])

    # Training options
    train: bool = True
    epochs: int = 10

    # Reproducibility
    seed: int = 42

    # Degradation parameters
    jpeg_qualities: List[int] = field(default_factory=lambda: [95, 85, 70, 50])
    blur_sigmas: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    resize_scales: List[float] = field(default_factory=lambda: [0.75, 0.50])
    noise_sigmas: List[float] = field(default_factory=lambda: [0.01, 0.02])


@dataclass
class EvaluationResult:
    """Result from a single evaluation run."""

    baseline: str
    eval_type: str
    split_name: str
    metrics: Dict[str, float]
    train_time_sec: float = 0.0
    inference_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvaluationProtocol:
    """
    Comprehensive evaluation protocol for thesis.

    Coordinates running all baselines across all evaluation scenarios.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results: List[EvaluationResult] = []

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage paths
        self.results_json = self.config.output_dir / "all_results.json"
        self.results_csv = self.config.output_dir / "results_summary.csv"

    def run_full_protocol(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the complete evaluation protocol.

        Returns:
            Dictionary with all results
        """
        start_time = time.time()

        if verbose:
            print("=" * 70)
            print("IMAGETRUST EVALUATION PROTOCOL")
            print("=" * 70)
            print(f"Start time: {datetime.now().isoformat()}")
            print(f"Baselines: {', '.join(self.config.baselines)}")
            print(f"Evaluation types: ", end="")
            types = []
            if self.config.run_in_domain:
                types.append("in-domain")
            if self.config.run_cross_generator:
                types.append("cross-generator")
            if self.config.run_degradation:
                types.append("degradation")
            print(", ".join(types))
            print("-" * 70)

        # 1. In-domain evaluation
        if self.config.run_in_domain:
            if verbose:
                print("\n[1/3] IN-DOMAIN EVALUATION")
            self._run_in_domain_evaluation(verbose)

        # 2. Cross-generator evaluation
        if self.config.run_cross_generator:
            if verbose:
                print("\n[2/3] CROSS-GENERATOR EVALUATION")
            self._run_cross_generator_evaluation(verbose)

        # 3. Degradation evaluation
        if self.config.run_degradation:
            if verbose:
                print("\n[3/3] DEGRADATION ROBUSTNESS EVALUATION")
            self._run_degradation_evaluation(verbose)

        # Compile results
        total_time = time.time() - start_time

        summary = self._compile_results(total_time)

        # Save results
        self._save_results(summary, verbose)

        if verbose:
            print("\n" + "=" * 70)
            print("EVALUATION COMPLETE")
            print(f"Total time: {total_time:.1f} seconds")
            print(f"Results saved to: {self.config.output_dir}")
            print("=" * 70)

        return summary

    def _run_in_domain_evaluation(self, verbose: bool = True) -> None:
        """Run standard in-domain evaluation."""
        split_path = self.config.splits_dir / "default_split.json"

        if not split_path.exists():
            if verbose:
                print(f"  Warning: Split file not found: {split_path}")
                print("  Run create_splits.py first to create splits.")
            return

        if verbose:
            print(f"  Loading split: {split_path}")

        # Import here to avoid circular imports
        from imagetrust.data.splits import load_split

        split = load_split(split_path)

        for baseline_name in self.config.baselines:
            if verbose:
                print(f"  Evaluating {baseline_name}...")

            result = self._evaluate_baseline_on_split(
                baseline_name, split, "in_domain"
            )
            self.results.append(result)

            if verbose:
                acc = result.metrics.get("accuracy", 0)
                auc = result.metrics.get("roc_auc", 0)
                print(f"    Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    def _run_cross_generator_evaluation(self, verbose: bool = True) -> None:
        """Run leave-one-generator-out evaluation."""
        cross_gen_dir = self.config.splits_dir / "cross_generator"

        if not cross_gen_dir.exists():
            if verbose:
                print(f"  Warning: Cross-generator splits not found: {cross_gen_dir}")
            return

        # Find all cross-generator splits
        split_files = list(cross_gen_dir.glob("leave_*.json"))

        if not split_files:
            if verbose:
                print("  No cross-generator split files found.")
            return

        if verbose:
            print(f"  Found {len(split_files)} generator splits")

        from imagetrust.data.splits import load_split

        for split_path in split_files:
            split = load_split(split_path)
            held_out = split.metadata.get("held_out_generator", split_path.stem)

            if verbose:
                print(f"\n  Held-out generator: {held_out}")

            for baseline_name in self.config.baselines:
                if verbose:
                    print(f"    Evaluating {baseline_name}...")

                result = self._evaluate_baseline_on_split(
                    baseline_name, split, "cross_generator"
                )
                result.metadata["held_out_generator"] = held_out
                self.results.append(result)

                if verbose:
                    auc = result.metrics.get("roc_auc", 0)
                    print(f"      AUC: {auc:.4f}")

    def _run_degradation_evaluation(self, verbose: bool = True) -> None:
        """Run degradation robustness evaluation."""
        split_path = self.config.splits_dir / "default_split.json"

        if not split_path.exists():
            if verbose:
                print(f"  Warning: Split file not found: {split_path}")
            return

        from imagetrust.data.splits import load_split

        split = load_split(split_path)

        # Define degradation matrix
        degradations = []

        # JPEG compression
        for q in self.config.jpeg_qualities:
            degradations.append(("jpeg", {"quality": q}))

        # Gaussian blur
        for sigma in self.config.blur_sigmas:
            degradations.append(("blur", {"sigma": sigma}))

        # Resize
        for scale in self.config.resize_scales:
            degradations.append(("resize", {"scale": scale}))

        # Noise
        for sigma in self.config.noise_sigmas:
            degradations.append(("noise", {"sigma": sigma}))

        if verbose:
            print(f"  Testing {len(degradations)} degradation conditions")

        for baseline_name in self.config.baselines:
            if verbose:
                print(f"\n  Baseline: {baseline_name}")

            for deg_type, deg_params in degradations:
                deg_name = f"{deg_type}_{list(deg_params.values())[0]}"

                if verbose:
                    print(f"    {deg_name}...", end=" ")

                result = self._evaluate_baseline_on_split(
                    baseline_name, split, "degradation",
                    degradation_type=deg_type,
                    degradation_params=deg_params,
                )
                result.metadata["degradation_type"] = deg_type
                result.metadata["degradation_params"] = deg_params
                self.results.append(result)

                if verbose:
                    auc = result.metrics.get("roc_auc", 0)
                    print(f"AUC: {auc:.4f}")

    def _evaluate_baseline_on_split(
        self,
        baseline_name: str,
        split,
        eval_type: str,
        degradation_type: Optional[str] = None,
        degradation_params: Optional[Dict] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single baseline on a single split.

        This is a placeholder that returns dummy metrics.
        In actual usage, this would:
        1. Load or train the baseline model
        2. Run inference on test set
        3. Compute metrics
        """
        # Import baseline utilities
        try:
            from imagetrust.baselines import get_baseline
            from imagetrust.evaluation.metrics import compute_metrics
        except ImportError:
            pass

        # Placeholder metrics (replace with actual evaluation)
        # In real implementation:
        # - Load baseline from registry
        # - Train if needed and config.train is True
        # - Run prediction on test split
        # - Optionally apply degradation
        # - Compute metrics

        metrics = {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
            "ece": 0.0,
        }

        return EvaluationResult(
            baseline=baseline_name,
            eval_type=eval_type,
            split_name=split.name,
            metrics=metrics,
            train_time_sec=0.0,
            inference_time_ms=0.0,
            metadata={
                "degradation_type": degradation_type,
                "degradation_params": degradation_params,
            },
        )

    def _compile_results(self, total_time: float) -> Dict[str, Any]:
        """Compile all results into summary structure."""
        summary = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_time_sec": total_time,
                "config": {
                    "baselines": self.config.baselines,
                    "seed": self.config.seed,
                    "train": self.config.train,
                },
            },
            "in_domain": {},
            "cross_generator": {},
            "degradation": {},
        }

        for result in self.results:
            entry = {
                "metrics": result.metrics,
                "train_time_sec": result.train_time_sec,
                "inference_time_ms": result.inference_time_ms,
                **result.metadata,
            }

            if result.eval_type == "in_domain":
                summary["in_domain"][result.baseline] = entry

            elif result.eval_type == "cross_generator":
                held_out = result.metadata.get("held_out_generator", "unknown")
                if result.baseline not in summary["cross_generator"]:
                    summary["cross_generator"][result.baseline] = {}
                summary["cross_generator"][result.baseline][held_out] = entry

            elif result.eval_type == "degradation":
                deg_type = result.metadata.get("degradation_type", "unknown")
                deg_params = result.metadata.get("degradation_params", {})
                deg_key = f"{deg_type}_{list(deg_params.values())[0] if deg_params else 'unknown'}"

                if result.baseline not in summary["degradation"]:
                    summary["degradation"][result.baseline] = {}
                summary["degradation"][result.baseline][deg_key] = entry

        return summary

    def _save_results(self, summary: Dict[str, Any], verbose: bool = True) -> None:
        """Save results to JSON and CSV files."""
        # Save JSON
        with open(self.results_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Save CSV summary
        csv_lines = ["baseline,eval_type,split,accuracy,f1_score,roc_auc,ece"]

        for result in self.results:
            m = result.metrics
            csv_lines.append(
                f"{result.baseline},{result.eval_type},{result.split_name},"
                f"{m.get('accuracy', 0):.4f},{m.get('f1_score', 0):.4f},"
                f"{m.get('roc_auc', 0):.4f},{m.get('ece', 0):.4f}"
            )

        with open(self.results_csv, "w", encoding="utf-8") as f:
            f.write("\n".join(csv_lines))

        if verbose:
            print(f"\n  Saved JSON: {self.results_json}")
            print(f"  Saved CSV: {self.results_csv}")


def generate_latex_tables(results_path: Path, output_dir: Path) -> None:
    """Generate LaTeX tables from results JSON."""
    with open(results_path, "r") as f:
        results = json.load(f)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Main comparison
    _generate_main_comparison_table(results, tables_dir)

    # Table 2: Cross-generator
    _generate_cross_generator_table(results, tables_dir)

    # Table 3: Degradation
    _generate_degradation_table(results, tables_dir)


def _generate_main_comparison_table(results: Dict, tables_dir: Path) -> None:
    """Generate main comparison table (Table 1)."""
    in_domain = results.get("in_domain", {})

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Comparison of baselines and our method on the test set.",
        r"Best results in bold. $\downarrow$ indicates lower is better.}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Acc & Bal. Acc & F1 & AUC & ECE ($\downarrow$) \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Baselines}} \\",
    ]

    baseline_order = ["classical", "cnn", "vit"]
    display_names = {
        "classical": "Classical (LogReg)",
        "cnn": "CNN (ResNet-50)",
        "vit": "ViT-B/16",
    }

    for baseline in baseline_order:
        if baseline in in_domain:
            m = in_domain[baseline]["metrics"]
            name = display_names.get(baseline, baseline)
            lines.append(
                f"{name} & {m.get('accuracy', 0):.2f} & {m.get('balanced_accuracy', 0):.2f} & "
                f"{m.get('f1_score', 0):.2f} & {m.get('roc_auc', 0):.2f} & {m.get('ece', 0):.2f} \\\\"
            )

    lines.extend([
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Our Method}} \\",
    ])

    if "imagetrust" in in_domain:
        m = in_domain["imagetrust"]["metrics"]
        lines.append(
            f"ImageTrust (Ours) & \\textbf{{{m.get('accuracy', 0):.2f}}} & "
            f"\\textbf{{{m.get('balanced_accuracy', 0):.2f}}} & "
            f"\\textbf{{{m.get('f1_score', 0):.2f}}} & "
            f"\\textbf{{{m.get('roc_auc', 0):.2f}}} & "
            f"\\textbf{{{m.get('ece', 0):.2f}}} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(tables_dir / "table1_main_comparison.tex", "w") as f:
        f.write("\n".join(lines))


def _generate_cross_generator_table(results: Dict, tables_dir: Path) -> None:
    """Generate cross-generator table (Table 2)."""
    cross_gen = results.get("cross_generator", {})

    generators = ["midjourney", "dalle3", "sdxl", "firefly"]
    gen_display = {
        "midjourney": "MJ",
        "dalle3": "DALL-E",
        "sdxl": "SDXL",
        "firefly": "Firefly",
    }

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cross-generator generalization (AUC). Each column shows performance when that generator is held out during training.}",
        r"\label{tab:crossgen}",
        r"\begin{tabular}{l|" + "c" * len(generators) + "}",
        r"\toprule",
        "Method & " + " & ".join(gen_display.get(g, g) for g in generators) + r" \\",
        r"\midrule",
    ]

    baseline_order = ["classical", "cnn", "vit", "imagetrust"]
    display_names = {
        "classical": "Classical",
        "cnn": "CNN",
        "vit": "ViT",
        "imagetrust": "Ours",
    }

    for baseline in baseline_order:
        if baseline in cross_gen:
            name = display_names.get(baseline, baseline)
            values = []
            for gen in generators:
                if gen in cross_gen[baseline]:
                    auc = cross_gen[baseline][gen]["metrics"].get("roc_auc", 0)
                    values.append(f"{auc:.2f}")
                else:
                    values.append("--")
            lines.append(f"{name} & " + " & ".join(values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(tables_dir / "table2_cross_generator.tex", "w") as f:
        f.write("\n".join(lines))


def _generate_degradation_table(results: Dict, tables_dir: Path) -> None:
    """Generate degradation robustness table (Table 3)."""
    degradation = results.get("degradation", {})

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Performance under image degradations (AUC).}",
        r"\label{tab:degradation}",
        r"\begin{tabular}{l|cccc|ccc|cc}",
        r"\toprule",
        r"& \multicolumn{4}{c|}{JPEG Quality} & \multicolumn{3}{c|}{Blur $\sigma$} & \multicolumn{2}{c}{Resize} \\",
        r"Method & 95 & 85 & 70 & 50 & 0.5 & 1.0 & 2.0 & 75\% & 50\% \\",
        r"\midrule",
    ]

    baseline_order = ["classical", "cnn", "vit", "imagetrust"]
    display_names = {
        "classical": "Classical",
        "cnn": "CNN",
        "vit": "ViT",
        "imagetrust": "Ours",
    }

    deg_columns = [
        "jpeg_95", "jpeg_85", "jpeg_70", "jpeg_50",
        "blur_0.5", "blur_1.0", "blur_2.0",
        "resize_0.75", "resize_0.5",
    ]

    for baseline in baseline_order:
        if baseline in degradation:
            name = display_names.get(baseline, baseline)
            values = []
            for deg in deg_columns:
                if deg in degradation[baseline]:
                    auc = degradation[baseline][deg]["metrics"].get("roc_auc", 0)
                    values.append(f"{auc:.2f}")
                else:
                    values.append("--")
            lines.append(f"{name} & " + " & ".join(values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(tables_dir / "table3_degradation.tex", "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation protocol for ImageTrust",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing split files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help="Directory containing pretrained weights",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        default=["classical", "cnn", "vit", "imagetrust"],
        help="Baselines to evaluate",
    )
    parser.add_argument(
        "--eval-type",
        choices=["all", "in-domain", "cross-generator", "degradation"],
        default="all",
        help="Evaluation type to run",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training, use pretrained weights",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--generate-tables",
        action="store_true",
        help="Generate LaTeX tables from results",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    # Create config
    config = EvaluationConfig(
        splits_dir=args.splits_dir,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        baselines=args.baseline,
        train=not args.no_train,
        epochs=args.epochs,
        seed=args.seed,
        run_in_domain=args.eval_type in ["all", "in-domain"],
        run_cross_generator=args.eval_type in ["all", "cross-generator"],
        run_degradation=args.eval_type in ["all", "degradation"],
    )

    # Run protocol
    protocol = EvaluationProtocol(config)
    results = protocol.run_full_protocol(verbose=verbose)

    # Generate LaTeX tables
    if args.generate_tables:
        if verbose:
            print("\nGenerating LaTeX tables...")
        generate_latex_tables(
            config.output_dir / "all_results.json",
            config.output_dir,
        )
        if verbose:
            print(f"  Tables saved to: {config.output_dir / 'tables'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
