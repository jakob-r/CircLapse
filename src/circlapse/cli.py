"""Command line interface: ``circlapse synth | annotate | train | detect | eval``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
train_app = typer.Typer(no_args_is_help=True, help="Train the neural network detector.")
app.add_typer(train_app, name="train")

CHECKPOINTS = Path("checkpoints")


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


@train_app.command("pretrain")
def train_pretrain(
    synth: Annotated[Path, typer.Option(help="Synthetic training set")] = Path("data/synth"),
    val: Annotated[Path | None, typer.Option(help="Synthetic validation set")] = Path(
        "data/synth_val"
    ),
    out: Annotated[Path, typer.Option(help="Checkpoint")] = CHECKPOINTS / "pretrain.pt",
    backbone: Annotated[str, typer.Option(help="timm backbone")] = "resnet34",
    size: Annotated[int, typer.Option(help="Network input size")] = 384,
    epochs: Annotated[int, typer.Option()] = 6,
    lr: Annotated[float, typer.Option()] = 1e-3,
    batch_size: Annotated[int, typer.Option()] = 32,
    workers: Annotated[int, typer.Option()] = 8,
) -> None:
    """Pretrain on synthetic images."""
    from circlapse.nn.model import ModelConfig
    from circlapse.nn.train import pretrain

    val = val if val is not None and val.is_dir() else None
    cfg = ModelConfig(backbone=backbone, size=size)
    pretrain(synth, val, out, cfg, epochs, lr, batch_size, workers, log=_log)


@train_app.command("finetune")
def train_finetune(
    init: Annotated[Path, typer.Option(help="Pretrained checkpoint")] = CHECKPOINTS / "pretrain.pt",
    real: Annotated[Path, typer.Option(help="Dataset with human labels")] = Path("data/real"),
    synth: Annotated[Path | None, typer.Option(help="Synthetic images mixed in")] = Path(
        "data/synth"
    ),
    out: Annotated[Path, typer.Option(help="Checkpoint")] = CHECKPOINTS / "finetune.pt",
    folds: Annotated[int, typer.Option(help="k-fold CV before the final fit (<2: skip)")] = 5,
    cv_out: Annotated[Path | None, typer.Option(help="Held-out CV predictions")] = Path(
        "data/predictions/cv.json"
    ),
    epochs: Annotated[int, typer.Option()] = 8,
    lr: Annotated[float, typer.Option()] = 3e-4,
    batch_size: Annotated[int, typer.Option()] = 16,
    repeats: Annotated[int, typer.Option(help="Augmented copies per real image and epoch")] = 64,
    workers: Annotated[int, typer.Option()] = 8,
) -> None:
    """Fine-tune on human labels, with k-fold evaluation."""
    from circlapse.nn.train import finetune

    synth = synth if synth is not None and synth.is_dir() else None
    finetune(
        init, real, out, synth, folds, epochs, lr, batch_size, repeats, workers, cv_out, log=_log
    )


@app.command()
def detect(
    data: Annotated[Path, typer.Argument(help="Image directory")] = Path("data/real"),
    out: Annotated[Path, typer.Option(help="Predictions json")] = Path("data/predictions/nn.json"),
    checkpoint: Annotated[Path, typer.Option()] = CHECKPOINTS / "finetune.pt",
    head: Annotated[str, typer.Option(help="mask (fit to predicted mask) or params")] = "mask",
    min_confidence: Annotated[float, typer.Option(help="Drop images below this")] = 0.0,
    tta: Annotated[bool, typer.Option(help="Horizontal flip test-time augmentation")] = True,
) -> None:
    """Detect the prominent ellipse in every image of DATA."""
    from circlapse.detect import run_detector
    from circlapse.detect.nn import NNDetector
    from circlapse.labels import Dataset

    if head not in ("mask", "params"):
        raise typer.BadParameter("head must be mask or params")
    detector = NNDetector(checkpoint, head, min_confidence, tta)  # type: ignore[arg-type]
    preds = run_detector(detector, Dataset(data), out)
    runtimes = sorted(p.runtime_s or 0 for p in preds.values())
    dropped = sum(not p.ellipses for p in preds.values())
    typer.echo(
        f"wrote {len(preds)} predictions to {out} ({dropped} dropped,"
        f" median {runtimes[len(runtimes) // 2] if runtimes else 0:.2f}s,"
        f" max {max(runtimes, default=0):.2f}s per image)"
    )


def _log(msg: str) -> None:
    typer.echo(msg)


if __name__ == "__main__":
    app()
