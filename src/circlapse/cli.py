"""Command line interface: ``circlapse synth | annotate | eval``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def synth(
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("data/synth"),
    n: Annotated[int, typer.Option(help="Number of images")] = 1000,
    size: Annotated[int, typer.Option(help="Image side length in pixels")] = 512,
    seed: Annotated[int, typer.Option()] = 0,
    clean: Annotated[bool, typer.Option(help="No noise and no perspective warp")] = False,
) -> None:
    """Generate synthetic images with labeled ellipses."""
    from circlapse.synth import SynthConfig, generate_dataset

    cfg = SynthConfig(size=size, noise=not clean, perspective=0.0 if clean else 0.08)
    generate_dataset(out, n, cfg, seed)
    typer.echo(f"wrote {n} images to {out}")


@app.command()
def annotate(
    data: Annotated[Path, typer.Argument(help="Image directory")] = Path("data/real"),
    pred: Annotated[Path | None, typer.Option(help="Predictions file to overlay")] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
) -> None:
    """Serve the annotation tool."""
    import threading
    import webbrowser

    import uvicorn

    from circlapse.annotate.app import create_app

    if not data.is_dir():
        raise typer.BadParameter(f"{data} is not a directory")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, [f"http://{host}:{port}"]).start()
    uvicorn.run(create_app(data, pred), host=host, port=port)


@app.command("eval")
def evaluate(
    gt: Annotated[Path, typer.Option(help="Ground truth labels.json or dataset directory")],
    pred: Annotated[Path, typer.Option(help="Predictions json")],
) -> None:
    """Compare predicted prominent ellipses with ground truth."""
    from circlapse.labels import LABELS_FILE, load_labels
    from circlapse.metrics import evaluate as run_eval

    gt_path = gt / LABELS_FILE if gt.is_dir() else gt
    typer.echo(run_eval(load_labels(gt_path), load_labels(pred)).format())


if __name__ == "__main__":
    app()
