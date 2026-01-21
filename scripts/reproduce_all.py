#!/usr/bin/env python3
"""
Master reproducibility script for ImageTrust thesis.

Runs all experiments required for the thesis in sequence:
1. Dataset preparation and splits
2. Baseline training and evaluation
3. Cross-generator evaluation
4. Degradation robustness evaluation
5. Calibration analysis
6. Ablation study
7. Figure and table generation

This script ensures complete reproducibility of all thesis results.

Usage:
    # Run complete reproducibility pipeline
    python scripts/reproduce_all.py

    # Run specific stages
    python scripts/reproduce_all.py --stage baselines

    # Dry run (show what would be executed)
    python scripts/reproduce_all.py --dry-run

    # Skip data preparation (if splits already exist)
    python scripts/reproduce_all.py --skip-data-prep

Requirements:
    - Dataset must be prepared in data/raw/ directory
    - GPU recommended for baseline training
    - ~2-4 hours for complete pipeline

Author: ImageTrust Team
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    name: str
    success: bool
    duration_sec: float
    output_files: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class PipelineConfig:
    """Configuration for reproducibility pipeline."""

    # Directories
    data_dir: Path = Path("data/raw")
    splits_dir: Path = Path("data/splits")
    output_dir: Path = Path("outputs")
    paper_dir: Path = Path("outputs/paper")

    # Execution options
    seed: int = 42
    skip_data_prep: bool = False
    skip_training: bool = False
    generate_figures: bool = True
    generate_tables: bool = True

    # Baseline options
    baselines: List[str] = field(default_factory=lambda: ["classical", "cnn", "vit", "imagetrust"])
    epochs: int = 10

    # Evaluation options
    cross_generator: bool = True
    degradation: bool = True
    calibration: bool = True
    ablation: bool = True


class ReproducibilityPipeline:
    """
    Master pipeline for reproducing all thesis experiments.
    """

    STAGES = [
        "data_prep",
        "baselines",
        "cross_generator",
        "degradation",
        "calibration",
        "ablation",
        "figures",
        "tables",
        "summary",
    ]

    def __init__(self, config: PipelineConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.results: List[StageResult] = []
        self.start_time: Optional[float] = None

    def run(self, stages: Optional[List[str]] = None) -> bool:
        """
        Run the reproducibility pipeline.

        Args:
            stages: Specific stages to run (None = all)

        Returns:
            True if all stages succeeded
        """
        self.start_time = time.time()

        stages_to_run = stages or self.STAGES

        print("=" * 70)
        print("IMAGETRUST REPRODUCIBILITY PIPELINE")
        print("=" * 70)
        print(f"Start time: {datetime.now().isoformat()}")
        print(f"Dry run: {self.dry_run}")
        print(f"Stages: {', '.join(stages_to_run)}")
        print(f"Output directory: {self.config.output_dir}")
        print("-" * 70)

        # Ensure output directories exist
        if not self.dry_run:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            self.config.paper_dir.mkdir(parents=True, exist_ok=True)

        # Run stages
        all_success = True
        for stage in stages_to_run:
            if stage not in self.STAGES:
                print(f"Warning: Unknown stage '{stage}', skipping")
                continue

            result = self._run_stage(stage)
            self.results.append(result)

            if not result.success:
                all_success = False
                print(f"\n[FAILED] Stage '{stage}' failed: {result.error_message}")
                if stage in ["data_prep", "baselines"]:
                    print("Critical stage failed, stopping pipeline.")
                    break

        # Print summary
        self._print_summary()

        # Save pipeline results
        if not self.dry_run:
            self._save_results()

        return all_success

    def _run_stage(self, stage: str) -> StageResult:
        """Run a single pipeline stage."""
        print(f"\n{'=' * 50}")
        print(f"STAGE: {stage.upper()}")
        print("=" * 50)

        start = time.time()

        try:
            if stage == "data_prep":
                result = self._stage_data_prep()
            elif stage == "baselines":
                result = self._stage_baselines()
            elif stage == "cross_generator":
                result = self._stage_cross_generator()
            elif stage == "degradation":
                result = self._stage_degradation()
            elif stage == "calibration":
                result = self._stage_calibration()
            elif stage == "ablation":
                result = self._stage_ablation()
            elif stage == "figures":
                result = self._stage_figures()
            elif stage == "tables":
                result = self._stage_tables()
            elif stage == "summary":
                result = self._stage_summary()
            else:
                result = StageResult(
                    name=stage,
                    success=False,
                    duration_sec=0,
                    error_message=f"Unknown stage: {stage}"
                )

            result.duration_sec = time.time() - start
            status = "OK" if result.success else "FAILED"
            print(f"\n[{status}] {stage} completed in {result.duration_sec:.1f}s")

            return result

        except Exception as e:
            return StageResult(
                name=stage,
                success=False,
                duration_sec=time.time() - start,
                error_message=str(e)
            )

    def _stage_data_prep(self) -> StageResult:
        """Stage 1: Data preparation and splits."""
        if self.config.skip_data_prep:
            print("Skipping data preparation (--skip-data-prep)")
            return StageResult(name="data_prep", success=True, duration_sec=0)

        cmd = [
            sys.executable, "scripts/create_splits.py",
            "--data-dir", str(self.config.data_dir),
            "--output-dir", str(self.config.splits_dir),
            "--seed", str(self.config.seed),
        ]

        return self._run_command(cmd, "data_prep")

    def _stage_baselines(self) -> StageResult:
        """Stage 2: Baseline training and evaluation."""
        output_files = []

        for baseline in self.config.baselines:
            print(f"\n  Training {baseline}...")

            cmd = [
                sys.executable, "scripts/run_baselines.py",
                "--dataset", str(self.config.splits_dir),
                "--baseline", baseline,
                "--output-dir", str(self.config.output_dir / "baselines"),
            ]

            if not self.config.skip_training and baseline != "imagetrust":
                cmd.append("--train")
                cmd.extend(["--epochs", str(self.config.epochs)])

            result = self._run_command(cmd, f"baseline_{baseline}")
            if not result.success:
                return result

            output_files.extend(result.output_files)

        return StageResult(
            name="baselines",
            success=True,
            duration_sec=0,
            output_files=output_files
        )

    def _stage_cross_generator(self) -> StageResult:
        """Stage 3: Cross-generator evaluation."""
        if not self.config.cross_generator:
            print("Skipping cross-generator evaluation")
            return StageResult(name="cross_generator", success=True, duration_sec=0)

        cmd = [
            sys.executable, "scripts/evaluation_protocol.py",
            "--splits-dir", str(self.config.splits_dir),
            "--output-dir", str(self.config.output_dir / "evaluation"),
            "--eval-type", "cross-generator",
        ]

        return self._run_command(cmd, "cross_generator")

    def _stage_degradation(self) -> StageResult:
        """Stage 4: Degradation robustness evaluation."""
        if not self.config.degradation:
            print("Skipping degradation evaluation")
            return StageResult(name="degradation", success=True, duration_sec=0)

        cmd = [
            sys.executable, "scripts/evaluation_protocol.py",
            "--splits-dir", str(self.config.splits_dir),
            "--output-dir", str(self.config.output_dir / "evaluation"),
            "--eval-type", "degradation",
        ]

        return self._run_command(cmd, "degradation")

    def _stage_calibration(self) -> StageResult:
        """Stage 5: Calibration analysis."""
        if not self.config.calibration:
            print("Skipping calibration analysis")
            return StageResult(name="calibration", success=True, duration_sec=0)

        cmd = [
            sys.executable, "scripts/evaluate_calibration.py",
            "--splits-dir", str(self.config.splits_dir),
            "--output-dir", str(self.config.output_dir / "calibration"),
            "--compare-methods",
            "--selective-prediction",
        ]

        return self._run_command(cmd, "calibration")

    def _stage_ablation(self) -> StageResult:
        """Stage 6: Ablation study."""
        if not self.config.ablation:
            print("Skipping ablation study")
            return StageResult(name="ablation", success=True, duration_sec=0)

        cmd = [
            sys.executable, "scripts/run_ablation.py",
            "--splits-dir", str(self.config.splits_dir),
            "--output-dir", str(self.config.output_dir / "ablation"),
            "--ablation-type", "all",
        ]

        return self._run_command(cmd, "ablation")

    def _stage_figures(self) -> StageResult:
        """Stage 7: Generate all figures."""
        if not self.config.generate_figures:
            print("Skipping figure generation")
            return StageResult(name="figures", success=True, duration_sec=0)

        cmd = [
            sys.executable, "scripts/generate_figures.py",
            "--results-dir", str(self.config.output_dir),
            "--output-dir", str(self.config.paper_dir / "figures"),
        ]

        return self._run_command(cmd, "figures")

    def _stage_tables(self) -> StageResult:
        """Stage 8: Generate all LaTeX tables."""
        if not self.config.generate_tables:
            print("Skipping table generation")
            return StageResult(name="tables", success=True, duration_sec=0)

        # Run table generation from multiple sources
        stages = [
            ("evaluation_protocol.py", "--generate-tables"),
            ("evaluate_calibration.py", "--generate-figures"),
            ("run_ablation.py", "--generate-tables"),
        ]

        for script, flag in stages:
            cmd = [
                sys.executable, f"scripts/{script}",
                "--results-file", str(self.config.output_dir / script.replace(".py", "") / "results.json"),
                flag,
            ]
            # Don't fail on individual table generation
            self._run_command(cmd, f"tables_{script}", allow_failure=True)

        return StageResult(name="tables", success=True, duration_sec=0)

    def _stage_summary(self) -> StageResult:
        """Stage 9: Generate final summary."""
        summary = self._generate_summary()

        summary_path = self.config.output_dir / "reproducibility_summary.json"
        if not self.dry_run:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Summary saved to: {summary_path}")

        return StageResult(
            name="summary",
            success=True,
            duration_sec=0,
            output_files=[str(summary_path)]
        )

    def _run_command(
        self,
        cmd: List[str],
        name: str,
        allow_failure: bool = False,
    ) -> StageResult:
        """Run a subprocess command."""
        print(f"  Command: {' '.join(cmd)}")

        if self.dry_run:
            print("  [DRY RUN] Would execute above command")
            return StageResult(name=name, success=True, duration_sec=0)

        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0 and not allow_failure:
                return StageResult(
                    name=name,
                    success=False,
                    duration_sec=0,
                    error_message=result.stderr or f"Exit code: {result.returncode}"
                )

            return StageResult(name=name, success=True, duration_sec=0)

        except subprocess.TimeoutExpired:
            return StageResult(
                name=name,
                success=False,
                duration_sec=3600,
                error_message="Command timed out (1 hour limit)"
            )
        except Exception as e:
            return StageResult(
                name=name,
                success=False,
                duration_sec=0,
                error_message=str(e)
            )

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate pipeline summary."""
        total_duration = time.time() - self.start_time if self.start_time else 0

        return {
            "pipeline": "ImageTrust Reproducibility",
            "timestamp": datetime.now().isoformat(),
            "total_duration_sec": total_duration,
            "config": {
                "seed": self.config.seed,
                "baselines": self.config.baselines,
                "epochs": self.config.epochs,
            },
            "stages": [
                {
                    "name": r.name,
                    "success": r.success,
                    "duration_sec": r.duration_sec,
                    "output_files": r.output_files,
                    "error": r.error_message,
                }
                for r in self.results
            ],
            "success": all(r.success for r in self.results),
            "output_directory": str(self.config.output_dir),
        }

    def _print_summary(self):
        """Print pipeline summary."""
        total_duration = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)

        for result in self.results:
            status = "[OK]" if result.success else "[FAILED]"
            print(f"  {status} {result.name}: {result.duration_sec:.1f}s")
            if result.error_message:
                print(f"       Error: {result.error_message}")

        print("-" * 70)
        print(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")

        success_count = sum(1 for r in self.results if r.success)
        print(f"Stages: {success_count}/{len(self.results)} successful")

        if all(r.success for r in self.results):
            print("\n[SUCCESS] All stages completed successfully!")
            print(f"Results available in: {self.config.output_dir}")
        else:
            print("\n[FAILURE] Some stages failed. Check errors above.")

    def _save_results(self):
        """Save pipeline results to file."""
        results_path = self.config.output_dir / "pipeline_results.json"
        summary = self._generate_summary()

        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nPipeline results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Master reproducibility script for ImageTrust thesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  data_prep       Create dataset splits
  baselines       Train and evaluate baselines
  cross_generator Cross-generator evaluation
  degradation     Degradation robustness
  calibration     Calibration analysis
  ablation        Ablation study
  figures         Generate all figures
  tables          Generate LaTeX tables
  summary         Generate final summary

Examples:
  # Run complete pipeline
  python scripts/reproduce_all.py

  # Run specific stages
  python scripts/reproduce_all.py --stage baselines calibration

  # Dry run
  python scripts/reproduce_all.py --dry-run

  # Skip data prep and training
  python scripts/reproduce_all.py --skip-data-prep --skip-training
        """,
    )

    parser.add_argument(
        "--stage",
        nargs="+",
        choices=ReproducibilityPipeline.STAGES + ["all"],
        default=["all"],
        help="Stages to run",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs for baselines",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation stage",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip baseline training",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip table generation",
    )

    args = parser.parse_args()

    # Build config
    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        skip_data_prep=args.skip_data_prep,
        skip_training=args.skip_training,
        generate_figures=not args.no_figures,
        generate_tables=not args.no_tables,
    )

    # Determine stages
    stages = None if "all" in args.stage else args.stage

    # Run pipeline
    pipeline = ReproducibilityPipeline(config, dry_run=args.dry_run)
    success = pipeline.run(stages)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
