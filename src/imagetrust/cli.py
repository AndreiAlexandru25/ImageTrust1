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

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="imagetrust")
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
@click.option("--legacy", is_flag=True, help="Use legacy Tkinter version")
def desktop(legacy: bool):
    """Launch the desktop GUI application."""
    if legacy:
        # Use legacy Tkinter version
        console.print("\n[bold]Starting ImageTrust Desktop (Tkinter)[/bold]")
        try:
            from imagetrust.desktop_app import main as tkinter_main
            tkinter_main()
        except ImportError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[dim]Make sure tkinter is installed[/dim]")
            raise click.Abort()
    else:
        # Use modern PySide6 version
        console.print("\n[bold]Starting ImageTrust Desktop (Qt)[/bold]")
        try:
            from imagetrust.desktop import main as qt_main
            qt_main()
        except ImportError as e:
            console.print(f"[red]Error: PySide6 not installed[/red]")
            console.print("[dim]Install with: pip install 'imagetrust[desktop]'[/dim]")
            console.print(f"\n[dim]Details: {e}[/dim]")

            # Fallback to Tkinter
            console.print("\n[yellow]Falling back to Tkinter version...[/yellow]")
            try:
                from imagetrust.desktop_app import main as tkinter_main
                tkinter_main()
            except ImportError:
                console.print("[red]Neither PySide6 nor Tkinter available[/red]")
                raise click.Abort()


if __name__ == "__main__":
    main()
