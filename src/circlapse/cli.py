"""Command line interface: ``circlapse synth | annotate | train | detect | eval | organize``."""

from __future__ import annotations

from datetime import datetime
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

    cfg = SynthConfig(
        size=size, noise=not clean, perspective=0.0 if clean else SynthConfig.perspective
    )
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


@app.command()
def organize(
    data: Annotated[Path, typer.Argument(help="Directory with a dump of photos")],
    from_apple_photos: Annotated[
        bool, typer.Option(help="First export --from-date..--to-date from Apple Photos into DATA")
    ] = False,
    from_date: Annotated[
        datetime | None, typer.Option(formats=["%Y-%m-%d"], help="First day (inclusive)")
    ] = None,
    to_date: Annotated[
        datetime | None, typer.Option(formats=["%Y-%m-%d"], help="Last day (inclusive)")
    ] = None,
    checkpoint: Annotated[Path, typer.Option()] = CHECKPOINTS / "finetune.pt",
    min_confidence: Annotated[
        float, typer.Option(help="Photos below this detector confidence go to rejected/")
    ] = 0.5,
    max_gap_hours: Annotated[
        float, typer.Option(help="Start a new cluster after a longer pause between photos")
    ] = 6.0,
    max_km: Annotated[
        float, typer.Option(help="Start a new cluster after a longer jump between GPS positions")
    ] = 20.0,
    geocode: Annotated[
        bool, typer.Option(help="Name clusters after the place (offline, needs the geo extra)")
    ] = True,
    dry_run: Annotated[
        bool, typer.Option(help="Only print what would be moved (the export still runs)")
    ] = False,
) -> None:
    """Reject photos without a good ellipse and sort the rest into trip folders.

    Clusters are named YYYY-MM-DD_YYYY-MM-DD_Place from EXIF time and GPS.
    Photos without a date go to undated/.
    --from-apple-photos needs the photos extra and Full Disk Access for the terminal.
    """
    from datetime import timedelta

    from circlapse.detect.nn import NNDetector
    from circlapse.organize import Photo, apply_moves, detect_all, offline_geocoder, plan_moves

    fallback: dict[str, Photo] | None = None
    if from_apple_photos:
        if from_date is None or to_date is None:
            raise typer.BadParameter("--from-apple-photos needs --from-date and --to-date")
        try:
            from circlapse.apple_photos import export_range
        except ImportError as e:
            raise typer.BadParameter("install the photos extra: uv sync --extra photos") from e
        fallback = export_range(data, from_date.date(), to_date.date(), log=_log)
        typer.echo(f"exported {len(fallback)} photos from Apple Photos to {data}")
    if not data.is_dir():
        raise typer.BadParameter(f"{data} is not a directory")
    geocoder = offline_geocoder() if geocode else None
    if geocode and geocoder is None:
        typer.echo("reverse_geocode not installed (uv sync --extra geo), clusters get no place")
    detector = NNDetector(checkpoint, "mask", min_confidence)
    photos = detect_all(data, detector, log=_log, fallback=fallback)
    accepted = [p for p in photos if p.label.ellipses]
    rejected = [p for p in photos if not p.label.ellipses]
    moves, folders = plan_moves(
        accepted, rejected, timedelta(hours=max_gap_hours), max_km, geocoder
    )
    for folder in sorted(folders):
        typer.echo(f"{folder}/  ({len(folders[folder])} photos)")
    typer.echo(f"rejected/  ({len(rejected)} photos)")
    if dry_run:
        for move in moves:
            typer.echo(f"  {move.src} -> {move.dst}")
        return
    apply_moves(data, moves, folders)
    typer.echo(f"moved {len(moves)} photos in {data}")


def _log(msg: str) -> None:
    typer.echo(msg)


if __name__ == "__main__":
    app()
