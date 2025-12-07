import typer
from pathlib import Path
import pandas as pd
import pyfiglet
from rich.console import Console
from rich.align import Align
from rich.table import Table
from rich.text import Text
from analyzer import AdReactionAnalyzer


app = typer.Typer()
console = Console()


def print_banner():
    banner = pyfiglet.figlet_format("AdReaction", font="big")
    banner_text = Text(banner, style="bold cyan")
    console.print(Align.center(banner_text))
    console.print(Align.center("[magenta]Facial Emotion Analytics[/magenta]"))
    console.print(Align.center("[dim]Powered by MediaPipe[/dim]\n"))
    console.print() 

@app.command()
def analyze(
    video: str = typer.Argument(..., help="Path to input video (.mp4)"),
    model: str = typer.Option("/home/daniil/code/project_cv/face_landmarker.task", "--model", "-m", help="Model path"),
    baseline_metrics: str = typer.Option("/home/daniil/code/project_cv/data/baseline_metrics.csv", "--baseline_metrics", "-b_m", help="Baseline metrics CSV"),
    fps: int = typer.Option(5, "--fps", help="Target FPS for frame sampling")
    ):
    print_banner()

    analyzer = AdReactionAnalyzer(model_path=model, baseline_results=baseline_metrics)

    try:
        analyzer.fit(video_path=video, target_fps=fps)
        agi, egi, vsi, mgi = analyzer.predict()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    
    console.print(Align.center("[bold cyan]Analysis Results[/bold cyan]"))
    console.print(Align.center("[dim]Relative change vs baseline[/dim]\n"))

    table = Table(show_header=True, header_style="bold cyan", show_lines=True,)

    table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
    table.add_column("Value", justify="center", style="bold")

    table.add_row("AGI", f"{agi * 100:.2f} %")
    table.add_row("EGI", f"{egi * 100:.2f} %")
    table.add_row("VSI", f"{vsi * 100:.2f} %")
    table.add_row("MGI", f"{mgi * 100:.2f} %")

    console.print(Align.center(table))
    console.print()  # финальный отступ

    console.print(Align.center("[dim]Done. Use --help for more options.[/dim]"))


if __name__ == "__main__":
    app()


