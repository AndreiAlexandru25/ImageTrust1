"""
Command-line interface for ImageTrust.
"""

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from imagetrust import __version__

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="imagetrust")
def main():
    """ImageTrust: AI-Generated Image Detection CLI"""
    pass


@main.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
@click.option("--model", "-m", default="ensemble", help="Model to use")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def analyze(image_path: str, output: Optional[str], model: str, verbose: bool):
    """Analyze a single image for AI-generated content."""
    from imagetrust.detection import AIDetector
    from imagetrust.utils.helpers import timer
    
    console.print(f"\n[bold]Analyzing:[/bold] {image_path}")
    
    try:
        with console.status("Loading model..."):
            detector = AIDetector(model=model)
        
        with console.status("Analyzing image..."):
            with timer() as t:
                result = detector.detect(image_path)
        
        # Display results
        ai_prob = result["ai_probability"]
        verdict = result["verdict"].value
        
        # Color based on verdict
        if verdict == "ai_generated":
            color = "red"
            emoji = "🤖"
        elif verdict == "real":
            color = "green"
            emoji = "📷"
        else:
            color = "yellow"
            emoji = "❓"
        
        console.print(f"\n{emoji} [bold {color}]{verdict.upper().replace('_', ' ')}[/bold {color}]")
        console.print(f"   AI Probability: [bold]{ai_prob:.1%}[/bold]")
        console.print(f"   Confidence: {result['confidence'].value.replace('_', ' ').title()}")
        console.print(f"   Processing Time: {t['elapsed_ms']:.0f}ms")
        
        if verbose:
            console.print(f"\n[dim]Model: {result['model_name']}[/dim]")
            console.print(f"[dim]Calibrated: {result['calibrated']}[/dim]")
        
        # Save output if requested
        if output:
            output_data = {
                "image_path": str(image_path),
                "ai_probability": ai_prob,
                "real_probability": result["real_probability"],
                "verdict": verdict,
                "confidence": result["confidence"].value,
                "processing_time_ms": t["elapsed_ms"],
                "model": result["model_name"],
            }
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"\n[dim]Results saved to: {output}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@main.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
@click.option("--model", "-m", default="ensemble", help="Model to use")
def batch(directory: str, output: Optional[str], model: str):
    """Analyze all images in a directory."""
    from imagetrust.detection import AIDetector
    
    dir_path = Path(directory)
    images = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
    
    if not images:
        console.print("[yellow]No images found in directory[/yellow]")
        return
    
    console.print(f"\n[bold]Found {len(images)} images[/bold]")
    
    try:
        with console.status("Loading model..."):
            detector = AIDetector(model=model)
        
        results = []
        
        with Progress() as progress:
            task = progress.add_task("Analyzing...", total=len(images))
            
            for img_path in images:
                result = detector.detect(img_path)
                results.append({
                    "image": img_path.name,
                    "ai_probability": result["ai_probability"],
                    "verdict": result["verdict"].value,
                })
                progress.update(task, advance=1)
        
        # Summary table
        table = Table(title="Results")
        table.add_column("Image", style="cyan")
        table.add_column("AI Probability", justify="right")
        table.add_column("Verdict")
        
        for r in results:
            verdict = r["verdict"]
            color = "red" if verdict == "ai_generated" else "green" if verdict == "real" else "yellow"
            table.add_row(
                r["image"],
                f"{r['ai_probability']:.1%}",
                f"[{color}]{verdict}[/{color}]",
            )
        
        console.print(table)
        
        # Summary
        ai_count = sum(1 for r in results if r["verdict"] == "ai_generated")
        console.print(f"\n[bold]Summary:[/bold] {ai_count}/{len(results)} detected as AI-generated")
        
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"[dim]Results saved to: {output}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", "-p", default=8000, help="Port to bind")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start the API server."""
    import uvicorn
    
    console.print(f"\n[bold]Starting ImageTrust API[/bold]")
    console.print(f"   URL: http://{host}:{port}")
    console.print(f"   Docs: http://{host}:{port}/docs")
    console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")
    
    uvicorn.run(
        "imagetrust.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@main.command()
@click.option("--port", "-p", default=8501, help="Port to bind")
def ui(port: int):
    """Launch the Streamlit web UI."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "frontend" / "app.py"

    console.print(f"\n[bold]Starting ImageTrust UI[/bold]")
    console.print(f"   URL: http://localhost:{port}")
    console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
    ])


@main.command()
def info():
    """Show system information."""
    import torch
    from imagetrust.core.config import get_settings

    settings = get_settings()

    console.print("\n[bold]ImageTrust System Information[/bold]")
    console.print(f"   Version: {settings.project_version}")
    console.print(f"   Environment: {settings.environment}")
    console.print(f"\n[bold]PyTorch[/bold]")
    console.print(f"   Version: {torch.__version__}")
    console.print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
    console.print(f"\n[bold]Paths[/bold]")
    console.print(f"   Data: {settings.data_dir}")
    console.print(f"   Models: {settings.models_dir}")
    console.print(f"   Outputs: {settings.outputs_dir}")


@main.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="./outputs", help="Output directory")
@click.option("--format", "-f", type=click.Choice(["all", "json", "md"]), default="all", help="Output format")
@click.option("--no-ai", is_flag=True, help="Skip AI detection (faster)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def forensics(image_path: str, output: str, format: str, no_ai: bool, verbose: bool):
    """
    Run comprehensive forensics analysis.

    Analyzes image for manipulation, recompression, screenshots,
    social media processing, and optionally AI generation.
    """
    from imagetrust.forensics import ForensicsEngine
    from imagetrust.forensics.base import PluginCategory

    console.print(f"\n[bold cyan]ImageTrust Forensics Analysis[/bold cyan]")
    console.print(f"Image: {image_path}")
    console.print("")

    try:
        # Initialize engine
        with console.status("Initializing forensics engine..."):
            categories = [
                PluginCategory.PIXEL,
                PluginCategory.METADATA,
                PluginCategory.SOURCE,
            ]
            if not no_ai:
                categories.append(PluginCategory.AI_DETECTION)

            engine = ForensicsEngine()

        # Run analysis
        with console.status("Running forensics analysis..."):
            report = engine.analyze(image_path, categories=categories)

        # Print summary
        report.print_summary()

        # Save outputs
        output_path = Path(output)
        saved = report.save(output_path)

        console.print(f"\n[dim]Report saved to: {output_path / report.run_id}[/dim]")

        if verbose:
            console.print(f"\n[dim]Files saved:[/dim]")
            for name, path in saved.items():
                console.print(f"  - {name}: {path}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise click.Abort()


@main.command("forensics-batch")
@click.argument("directory", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="./outputs", help="Output directory")
@click.option("--no-ai", is_flag=True, help="Skip AI detection (faster)")
def forensics_batch(directory: str, output: str, no_ai: bool):
    """Run forensics analysis on all images in a directory."""
    from imagetrust.forensics import ForensicsEngine
    from imagetrust.forensics.base import PluginCategory
    from rich.progress import Progress

    dir_path = Path(directory)
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        images.extend(dir_path.glob(ext))
        images.extend(dir_path.glob(ext.upper()))

    if not images:
        console.print("[yellow]No images found in directory[/yellow]")
        return

    console.print(f"\n[bold]Forensics Batch Analysis[/bold]")
    console.print(f"Found {len(images)} images")
    console.print("")

    try:
        categories = [
            PluginCategory.PIXEL,
            PluginCategory.METADATA,
            PluginCategory.SOURCE,
        ]
        if not no_ai:
            categories.append(PluginCategory.AI_DETECTION)

        engine = ForensicsEngine()
        output_path = Path(output)

        with Progress() as progress:
            task = progress.add_task("Analyzing...", total=len(images))

            for img_path in images:
                report = engine.analyze(img_path, categories=categories)
                report.save(output_path)
                progress.update(task, advance=1)

        console.print(f"\n[green]Analysis complete![/green]")
        console.print(f"Results saved to: {output_path}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@main.command("forensics-plugins")
def forensics_plugins():
    """List available forensics plugins."""
    from imagetrust.forensics import ForensicsEngine

    engine = ForensicsEngine()
    plugins = engine.get_available_plugins()

    from rich.table import Table

    table = Table(title="Available Forensics Plugins")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Category")
    table.add_column("Description")

    for p in plugins:
        table.add_row(
            p["id"],
            p["name"],
            p["category"],
            p["description"][:50] + "..." if len(p["description"]) > 50 else p["description"],
        )

    console.print(table)


@main.command()
def desktop():
    """Launch ImageTrust desktop application (PySide6 / Qt6).

    Professional offline forensics UI with calibrated thresholds,
    drag-and-drop, and JSON report export.
    """
    console.print("\n[bold]Starting ImageTrust Desktop[/bold]")
    try:
        from imagetrust.frontend.pyside_app import main as qt_main
        qt_main()
    except ImportError as e:
        console.print(f"[red]Error: PySide6 not installed[/red]")
        console.print("[dim]Install with: pip install PySide6[/dim]")
        console.print(f"\n[dim]Details: {e}[/dim]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


if __name__ == "__main__":
    main()
