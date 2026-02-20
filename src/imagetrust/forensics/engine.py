"""
Forensics Engine - Main orchestrator.

Runs all forensics plugins and produces comprehensive reports.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from imagetrust.forensics.base import (
    ForensicsPlugin,
    ForensicsResult,
    PluginCategory,
    get_plugin,
    get_plugins_by_category,
    list_plugins,
)
from imagetrust.forensics.fusion import ForensicsVerdict, FusionLayer
from imagetrust.utils.helpers import ensure_dir, generate_id
from imagetrust.utils.logging import get_logger

logger = get_logger(__name__)


class ForensicsEngine:
    """
    Main forensics analysis engine.

    Orchestrates all forensics plugins and produces comprehensive reports.

    Example:
        >>> engine = ForensicsEngine()
        >>> report = engine.analyze("image.jpg")
        >>> print(report.verdict.summary)
        >>> report.save("./outputs/")
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        plugins: Optional[List[str]] = None,
    ):
        """
        Initialize forensics engine.

        Args:
            config: Engine configuration
            plugins: List of plugin IDs to use (None = all)
        """
        self.config = config or {}
        self.fusion = FusionLayer(config)

        # Initialize plugins
        self.plugins: Dict[str, ForensicsPlugin] = {}
        self._load_plugins(plugins)

        logger.info(f"ForensicsEngine initialized with {len(self.plugins)} plugins")

    def _load_plugins(self, plugin_ids: Optional[List[str]] = None):
        """Load and initialize plugins."""
        # Import plugin modules to trigger registration
        from imagetrust.forensics import pixel_forensics  # noqa
        from imagetrust.forensics import metadata_forensics  # noqa
        from imagetrust.forensics import source_detection  # noqa

        # Try to import optional AI detection plugins
        try:
            from imagetrust.forensics import ai_detection  # noqa
        except ImportError:
            logger.debug("AI detection module not available")

        # Get plugin classes
        if plugin_ids:
            plugin_classes = [get_plugin(pid) for pid in plugin_ids if get_plugin(pid)]
        else:
            # Load all registered plugins
            plugin_classes = [get_plugin(pid) for pid in list_plugins()]

        # Initialize plugins
        for plugin_cls in plugin_classes:
            if plugin_cls:
                try:
                    plugin = plugin_cls(self.config.get(plugin_cls.plugin_id))
                    self.plugins[plugin.plugin_id] = plugin
                    logger.debug(f"Loaded plugin: {plugin.plugin_name}")
                except Exception as e:
                    logger.warning(f"Failed to load plugin {plugin_cls.plugin_id}: {e}")

    def analyze(
        self,
        source: Union[str, Path, bytes, Image.Image],
        run_all: bool = True,
        categories: Optional[List[PluginCategory]] = None,
    ) -> "ForensicsReport":
        """
        Perform comprehensive forensics analysis.

        Args:
            source: Image path, bytes, or PIL Image
            run_all: Run all available plugins
            categories: Specific categories to run (if not run_all)

        Returns:
            ForensicsReport with all results
        """
        start_time = time.perf_counter()
        run_id = generate_id("forensics")

        logger.info(f"Starting forensics analysis (run_id={run_id})")

        # Load image
        image, image_path, raw_bytes = self._load_image(source)

        if image is None:
            return ForensicsReport.create_error(run_id, "Failed to load image")

        # Collect results
        results: List[ForensicsResult] = []

        # Determine which plugins to run
        plugins_to_run = list(self.plugins.values())
        if categories and not run_all:
            plugins_to_run = [p for p in plugins_to_run if p.category in categories]

        # Run each plugin
        for plugin in plugins_to_run:
            try:
                # Check if plugin can analyze this image
                can_analyze, reason = plugin.can_analyze(image, image_path)
                if not can_analyze:
                    logger.debug(f"Skipping {plugin.plugin_name}: {reason}")
                    continue

                logger.debug(f"Running: {plugin.plugin_name}")
                result = plugin.analyze(image, image_path, raw_bytes)
                results.append(result)

            except Exception as e:
                logger.error(f"Plugin {plugin.plugin_name} failed: {e}")
                results.append(ForensicsResult(
                    plugin_id=plugin.plugin_id,
                    plugin_name=plugin.plugin_name,
                    category=plugin.category,
                    score=0.0,
                    confidence=plugin.category,
                    detected=False,
                    explanation=f"Analysis failed: {e}",
                    error=str(e),
                ))

        # Fuse results
        verdict = self.fusion.fuse(results)

        total_time = (time.perf_counter() - start_time) * 1000

        logger.info(f"Analysis complete: {verdict.summary} ({total_time:.0f}ms)")

        return ForensicsReport(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            image_path=str(image_path) if image_path else None,
            image_size=image.size,
            image_format=image.format or "Unknown",
            results=results,
            verdict=verdict,
            total_processing_time_ms=total_time,
        )

    def analyze_batch(
        self,
        sources: List[Union[str, Path]],
        output_dir: Optional[Path] = None,
    ) -> List["ForensicsReport"]:
        """
        Analyze multiple images.

        Args:
            sources: List of image paths
            output_dir: Directory to save reports

        Returns:
            List of ForensicsReport
        """
        reports = []

        for source in sources:
            try:
                report = self.analyze(source)
                reports.append(report)

                if output_dir:
                    report.save(output_dir)

            except Exception as e:
                logger.error(f"Failed to analyze {source}: {e}")

        return reports

    def _load_image(
        self,
        source: Union[str, Path, bytes, Image.Image],
    ) -> tuple:
        """Load image from various sources."""
        image = None
        image_path = None
        raw_bytes = None

        try:
            if isinstance(source, Image.Image):
                image = source
            elif isinstance(source, bytes):
                import io
                raw_bytes = source
                image = Image.open(io.BytesIO(source))
            else:
                image_path = Path(source)
                with open(image_path, "rb") as f:
                    raw_bytes = f.read()
                image = Image.open(image_path)

            # Convert to RGB if needed (but preserve format info)
            if image.mode not in ["RGB", "RGBA", "L"]:
                image = image.convert("RGB")

        except Exception as e:
            logger.error(f"Failed to load image: {e}")

        return image, image_path, raw_bytes

    def get_available_plugins(self) -> List[Dict[str, Any]]:
        """Get info about available plugins."""
        return [
            {
                "id": p.plugin_id,
                "name": p.plugin_name,
                "category": p.category.value,
                "description": p.description,
                "version": p.version,
            }
            for p in self.plugins.values()
        ]


class ForensicsReport:
    """
    Complete forensics analysis report.

    Contains all results, verdict, and can generate
    various output formats (JSON, Markdown, HTML).
    """

    def __init__(
        self,
        run_id: str,
        timestamp: str,
        image_path: Optional[str],
        image_size: tuple,
        image_format: str,
        results: List[ForensicsResult],
        verdict: ForensicsVerdict,
        total_processing_time_ms: float,
    ):
        self.run_id = run_id
        self.timestamp = timestamp
        self.image_path = image_path
        self.image_size = image_size
        self.image_format = image_format
        self.results = results
        self.verdict = verdict
        self.total_processing_time_ms = total_processing_time_ms

    @classmethod
    def create_error(cls, run_id: str, error: str) -> "ForensicsReport":
        """Create error report."""
        return cls(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            image_path=None,
            image_size=(0, 0),
            image_format="Unknown",
            results=[],
            verdict=FusionLayer()._create_empty_verdict(),
            total_processing_time_ms=0,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "image": {
                "path": self.image_path,
                "size": self.image_size,
                "format": self.image_format,
            },
            "verdict": self.verdict.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "processing_time_ms": self.total_processing_time_ms,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        """Generate Markdown report."""
        lines = [
            f"# Forensics Analysis Report",
            f"",
            f"**Run ID:** {self.run_id}",
            f"**Timestamp:** {self.timestamp}",
            f"**Image:** {self.image_path or 'N/A'}",
            f"**Dimensions:** {self.image_size[0]}x{self.image_size[1]}",
            f"**Format:** {self.image_format}",
            f"",
            f"---",
            f"",
            f"## Summary",
            f"",
            f"**Primary Verdict:** {self.verdict.primary_verdict.value}",
            f"**Confidence:** {self.verdict.primary_confidence.name}",
            f"**Authenticity Score:** {self.verdict.authenticity_score:.2f}",
            f"",
        ]

        if self.verdict.inconclusive:
            lines.append("**Status:** INCONCLUSIVE - insufficient evidence")
            lines.append("")

        # Top Evidence
        if self.verdict.top_evidence:
            lines.append("### Top Evidence")
            lines.append("")
            for i, evidence in enumerate(self.verdict.top_evidence, 1):
                lines.append(f"{i}. {evidence}")
            lines.append("")

        # Contradictions
        if self.verdict.contradictions:
            lines.append("### Contradictions")
            lines.append("")
            for contradiction in self.verdict.contradictions:
                lines.append(f"- {contradiction}")
            lines.append("")

        # Label Breakdown
        lines.append("## Label Analysis")
        lines.append("")
        lines.append("| Label | Probability | Confidence | Evidence |")
        lines.append("|-------|-------------|------------|----------|")

        for ls in sorted(self.verdict.labels, key=lambda x: x.probability, reverse=True):
            if ls.probability > 0.1:
                evidence_str = "; ".join(ls.evidence[:2]) if ls.evidence else "-"
                lines.append(
                    f"| {ls.label.value} | {ls.probability:.2f} | {ls.confidence.name} | {evidence_str} |"
                )

        lines.append("")

        # Detailed Results
        lines.append("## Detailed Results")
        lines.append("")

        for result in self.results:
            status = "DETECTED" if result.detected else "not detected"
            lines.append(f"### {result.plugin_name}")
            lines.append("")
            lines.append(f"- **Status:** {status}")
            lines.append(f"- **Score:** {result.score:.2f}")
            lines.append(f"- **Confidence:** {result.confidence.name}")
            lines.append(f"- **Explanation:** {result.explanation}")
            lines.append("")

            if result.limitations:
                lines.append("**Limitations:**")
                for lim in result.limitations:
                    lines.append(f"- {lim}")
                lines.append("")

        # Suggestions
        if self.verdict.what_would_help:
            lines.append("## Suggestions for Improved Analysis")
            lines.append("")
            for suggestion in self.verdict.what_would_help:
                lines.append(f"- {suggestion}")
            lines.append("")

        lines.append("---")
        lines.append(f"*Analysis completed in {self.total_processing_time_ms:.0f}ms*")

        return "\n".join(lines)

    def save(self, output_dir: Union[str, Path], save_artifacts: bool = True) -> Dict[str, Path]:
        """
        Save report and artifacts to directory.

        Args:
            output_dir: Output directory
            save_artifacts: Whether to save visual artifacts

        Returns:
            Dict of saved file paths
        """
        output_dir = Path(output_dir)
        report_dir = output_dir / self.run_id
        ensure_dir(report_dir)

        saved_files = {}

        # Save JSON
        json_path = report_dir / "report.json"
        with open(json_path, "w") as f:
            f.write(self.to_json())
        saved_files["json"] = json_path

        # Save Markdown
        md_path = report_dir / "report.md"
        with open(md_path, "w") as f:
            f.write(self.to_markdown())
        saved_files["markdown"] = md_path

        # Save artifacts
        if save_artifacts:
            artifacts_dir = report_dir / "artifacts"
            ensure_dir(artifacts_dir)

            for result in self.results:
                for artifact in result.artifacts:
                    try:
                        artifact_path = artifact.save(
                            artifacts_dir,
                            prefix=result.plugin_id
                        )
                        saved_files[f"{result.plugin_id}_{artifact.name}"] = artifact_path
                    except Exception as e:
                        logger.warning(f"Failed to save artifact {artifact.name}: {e}")

        logger.info(f"Report saved to {report_dir}")
        return saved_files

    def print_summary(self):
        """Print summary to console."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        # Header
        console.print(Panel(
            f"[bold]Forensics Analysis[/bold]\n"
            f"Image: {self.image_path or 'N/A'}\n"
            f"Run ID: {self.run_id}",
            title="ImageTrust Forensics",
        ))

        # Verdict
        verdict_color = {
            "camera_original_likely": "green",
            "ai_generated_suspected": "red",
            "edited_likely": "yellow",
            "screenshot_likely": "blue",
            "social_media_likely": "cyan",
            "unknown": "white",
        }.get(self.verdict.primary_verdict.value, "white")

        console.print(f"\n[bold]Primary Verdict:[/bold] [{verdict_color}]{self.verdict.primary_verdict.value}[/]")
        console.print(f"[bold]Confidence:[/bold] {self.verdict.primary_confidence.name}")
        console.print(f"[bold]Authenticity:[/bold] {self.verdict.authenticity_score:.2f}")

        if self.verdict.inconclusive:
            console.print("\n[yellow]Status: INCONCLUSIVE - insufficient evidence[/yellow]")

        # Top Evidence
        if self.verdict.top_evidence:
            console.print("\n[bold]Top Evidence:[/bold]")
            for evidence in self.verdict.top_evidence[:3]:
                console.print(f"  - {evidence}")

        # Results table
        table = Table(title="Detection Results")
        table.add_column("Detector", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Score")
        table.add_column("Confidence")

        for r in self.results:
            status = "[green]DETECTED[/green]" if r.detected else "[dim]not detected[/dim]"
            table.add_row(
                r.plugin_name,
                status,
                f"{r.score:.2f}",
                r.confidence.name,
            )

        console.print("\n")
        console.print(table)

        console.print(f"\n[dim]Analysis completed in {self.total_processing_time_ms:.0f}ms[/dim]")
