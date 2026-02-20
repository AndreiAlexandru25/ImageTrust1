"""
Efficiency profiler for thesis Table 6.

Measures:
- Inference time (ms/image, images/sec)
- Memory usage (GPU VRAM, CPU RAM)
- Throughput at different batch sizes
- Per-component timing breakdown
"""

import gc
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EfficiencyMetrics:
    """Efficiency metrics for a single configuration."""

    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_imgs_sec: float
    peak_gpu_vram_mb: float
    peak_cpu_ram_mb: float
    batch_size: int
    device: str
    num_samples: int
    warmup_iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "avg_time_ms": self.avg_time_ms,
            "std_time_ms": self.std_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "throughput_imgs_sec": self.throughput_imgs_sec,
            "peak_gpu_vram_mb": self.peak_gpu_vram_mb,
            "peak_cpu_ram_mb": self.peak_cpu_ram_mb,
            "batch_size": self.batch_size,
            "device": self.device,
            "num_samples": self.num_samples,
            "warmup_iterations": self.warmup_iterations,
        }


@dataclass
class ComponentTimingResult:
    """Per-component timing breakdown."""

    component_times: Dict[str, float]  # Component name -> avg time in ms
    total_time_ms: float
    component_percentages: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_times_ms": self.component_times,
            "total_time_ms": self.total_time_ms,
            "component_percentages": self.component_percentages,
        }


class EfficiencyProfiler:
    """
    Profiler for thesis efficiency table (Table 6).

    Measures runtime, memory, and throughput for AI detection models.
    """

    def __init__(
        self,
        warmup: int = 5,
        num_runs: int = 100,
        verbose: bool = True,
    ):
        """
        Initialize profiler.

        Args:
            warmup: Number of warmup iterations (not measured)
            num_runs: Number of measured runs
            verbose: Print progress
        """
        self.warmup = warmup
        self.num_runs = num_runs
        self.verbose = verbose

    def profile_detector(
        self,
        detector,
        images: List[Image.Image],
        batch_sizes: List[int] = [1],
    ) -> Dict[int, EfficiencyMetrics]:
        """
        Profile detector at different batch sizes.

        Note: Most detectors process single images, so batch_size=1 is typical.

        Args:
            detector: Detector with analyze() method
            images: List of test images
            batch_sizes: Batch sizes to test

        Returns:
            Dictionary mapping batch_size to EfficiencyMetrics
        """
        results = {}
        device = getattr(detector, "device", "cpu")

        for batch_size in batch_sizes:
            logger.info(f"Profiling batch_size={batch_size}")

            # Reset GPU stats if available
            peak_gpu_mb = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass

            # Get initial RAM
            try:
                import psutil
                process = psutil.Process()
                initial_ram = process.memory_info().rss / 1024 / 1024  # MB
            except ImportError:
                initial_ram = 0
                process = None

            # Warmup
            if self.verbose:
                logger.info(f"  Warmup ({self.warmup} iterations)...")
            for i in range(self.warmup):
                img = images[i % len(images)]
                _ = detector.analyze(img, return_uncertainty=False, profile=False)

            # Clear cache
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            # Timed runs
            times = []
            if self.verbose:
                logger.info(f"  Measuring ({self.num_runs} iterations)...")

            for i in range(self.num_runs):
                img = images[i % len(images)]

                start = time.perf_counter()
                _ = detector.analyze(img, return_uncertainty=False, profile=False)
                elapsed = (time.perf_counter() - start) * 1000  # ms

                times.append(elapsed)

            # Get peak GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            except ImportError:
                pass

            # Get peak RAM
            peak_ram = 0.0
            if process is not None:
                peak_ram = process.memory_info().rss / 1024 / 1024 - initial_ram

            # Compute metrics
            times = np.array(times)
            avg_time = float(np.mean(times))
            std_time = float(np.std(times))
            min_time = float(np.min(times))
            max_time = float(np.max(times))
            throughput = 1000.0 / avg_time  # images per second

            results[batch_size] = EfficiencyMetrics(
                avg_time_ms=avg_time,
                std_time_ms=std_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                throughput_imgs_sec=throughput,
                peak_gpu_vram_mb=peak_gpu_mb,
                peak_cpu_ram_mb=peak_ram,
                batch_size=batch_size,
                device=device,
                num_samples=self.num_runs,
                warmup_iterations=self.warmup,
            )

            if self.verbose:
                logger.info(
                    f"  Results: {avg_time:.1f}ms ± {std_time:.1f}ms, "
                    f"{throughput:.1f} img/s, GPU: {peak_gpu_mb:.0f}MB"
                )

        return results

    def profile_components(
        self,
        detector,
        images: List[Image.Image],
    ) -> ComponentTimingResult:
        """
        Profile individual components of the detector.

        Uses the detector's profile mode to get per-component timing.

        Args:
            detector: Detector with profile=True support
            images: Test images

        Returns:
            ComponentTimingResult with per-component breakdown
        """
        logger.info("Profiling individual components...")

        # Warmup
        for i in range(self.warmup):
            img = images[i % len(images)]
            _ = detector.analyze(img, return_uncertainty=False, profile=True)

        # Collect timing data
        all_timings: Dict[str, List[float]] = {}

        for i in range(self.num_runs):
            img = images[i % len(images)]
            result = detector.analyze(img, return_uncertainty=False, profile=True)

            timing = result.get("timing_breakdown", {})
            for component, time_ms in timing.items():
                if component not in all_timings:
                    all_timings[component] = []
                all_timings[component].append(time_ms)

        # Average timings
        avg_timings = {k: float(np.mean(v)) for k, v in all_timings.items()}
        total_time = sum(avg_timings.values())

        # Calculate percentages
        percentages = {k: (v / total_time * 100) if total_time > 0 else 0 for k, v in avg_timings.items()}

        result = ComponentTimingResult(
            component_times=avg_timings,
            total_time_ms=total_time,
            component_percentages=percentages,
        )

        if self.verbose:
            logger.info("Component breakdown:")
            for comp, time_ms in sorted(avg_timings.items(), key=lambda x: -x[1]):
                pct = percentages[comp]
                logger.info(f"  {comp}: {time_ms:.1f}ms ({pct:.1f}%)")

        return result

    def compare_detectors(
        self,
        detectors: Dict[str, Any],
        images: List[Image.Image],
    ) -> Dict[str, EfficiencyMetrics]:
        """
        Compare efficiency of multiple detectors.

        Args:
            detectors: Dictionary of {name: detector}
            images: Test images

        Returns:
            Dictionary of {name: EfficiencyMetrics}
        """
        results = {}

        for name, detector in detectors.items():
            logger.info(f"Profiling {name}...")
            try:
                metrics = self.profile_detector(detector, images, batch_sizes=[1])
                results[name] = metrics[1]  # batch_size=1
            except Exception as e:
                logger.warning(f"Failed to profile {name}: {e}")

        return results


def generate_efficiency_table(
    results: Dict[str, EfficiencyMetrics],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate Table 6 (Efficiency metrics) in LaTeX format.

    Args:
        results: Dictionary of {method_name: EfficiencyMetrics}
        output_path: Optional path to save the table

    Returns:
        LaTeX table string
    """
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Efficiency Comparison}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & ms/img & img/sec & VRAM (MB) & RAM (MB) \\",
        r"\midrule",
    ]

    for name, metrics in sorted(results.items()):
        lines.append(
            f"{name} & {metrics.avg_time_ms:.1f} $\\pm$ {metrics.std_time_ms:.1f} & "
            f"{metrics.throughput_imgs_sec:.1f} & {metrics.peak_gpu_vram_mb:.0f} & "
            f"{metrics.peak_cpu_ram_mb:.0f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(table)
        logger.info(f"Table saved to {output_path}")

    return table


def profile_hardware_info() -> Dict[str, Any]:
    """Get hardware information for reproducibility."""
    import platform

    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # CPU info
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count()
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = psutil.virtual_memory().total / 1024 / 1024 / 1024
    except ImportError:
        pass

    # GPU info
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["cuda_vram_gb"] = props.total_memory / 1024 / 1024 / 1024
    except ImportError:
        info["cuda_available"] = False

    return info


def create_efficiency_report(
    detector_results: Dict[str, EfficiencyMetrics],
    component_results: Optional[ComponentTimingResult] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Create comprehensive efficiency report.

    Args:
        detector_results: Results from profile_detector
        component_results: Results from profile_components
        output_dir: Optional directory to save outputs

    Returns:
        Complete efficiency report dictionary
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "hardware": profile_hardware_info(),
        "detector_metrics": {k: v.to_dict() for k, v in detector_results.items()},
    }

    if component_results:
        report["component_breakdown"] = component_results.to_dict()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        import json
        with open(output_dir / "efficiency_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Save LaTeX table
        generate_efficiency_table(
            detector_results,
            output_dir / "table_efficiency.tex",
        )

    return report
