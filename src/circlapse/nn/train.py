"""Training: pretraining on synthetic images, fine-tuning on human labels with k-fold CV."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from circlapse.labels import Dataset, ImageLabel, ScoredEllipse, load_image, save_labels
from circlapse.metrics import Summary, evaluate
from circlapse.nn.data import (
    Augment,
    Item,
    ItemDataset,
    SynthDataset,
    letterbox,
    load_items,
    to_tensor,
    transform_ellipse,
)
from circlapse.nn.infer import Head, run_batch
from circlapse.nn.model import EllipseNet, ModelConfig, device, load_checkpoint, save_checkpoint

Log = Callable[[str], None]

SYNTH_AUG = Augment(scale=(0.85, 1.15), rotate=180, shift=0.08, perspective=0.03)
REAL_AUG = Augment(scale=(0.7, 1.3), rotate=180, shift=0.15, perspective=0.08)


def loss_fn(
    logits: torch.Tensor, params: torch.Tensor, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    bce = F.binary_cross_entropy_with_logits(logits, mask)
    p = torch.sigmoid(logits)
    inter = (p * mask).sum(dim=(1, 2, 3))
    dice = 1 - ((2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) + 1)).mean()
    reg = F.smooth_l1_loss(params, batch["params"], beta=0.01)
    total = bce + dice + 10 * reg
    return total, {"bce": bce.item(), "dice": dice.item(), "reg": reg.item()}


def _set_epoch(ds: object, epoch: int) -> None:
    if isinstance(ds, ConcatDataset):
        for d in ds.datasets:
            _set_epoch(d, epoch)
    elif isinstance(ds, Subset):
        _set_epoch(ds.dataset, epoch)
    elif hasattr(ds, "epoch"):
        ds.epoch = epoch


def fit(
    model: EllipseNet,
    train_ds: torch.utils.data.Dataset[dict[str, torch.Tensor]],
    epochs: int,
    lr: float,
    batch_size: int,
    workers: int = 8,
    on_epoch: Callable[[int], None] | None = None,
    log: Log = print,
) -> None:
    dev = device()
    model.to(dev).train()
    steps_per_epoch = math.ceil(len(train_ds) / batch_size)  # type: ignore[arg-type]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.05
    )
    step = 0
    for epoch in range(epochs):
        _set_epoch(train_ds, epoch)
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
        model.train()
        start, running = time.time(), {"loss": 0.0, "bce": 0.0, "dice": 0.0, "reg": 0.0}
        n = 0
        for batch in loader:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            logits, params = model(batch["image"])
            loss, parts = loss_fn(logits, params, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            n += 1
            running["loss"] += loss.item()
            for k, v in parts.items():
                running[k] += v
            if step % 100 == 0:
                msg = " ".join(f"{k} {v / n:.4f}" for k, v in running.items())
                log(f"  step {step}/{epochs * steps_per_epoch} {msg} ({time.time() - start:.0f}s)")
        msg = " ".join(f"{k} {v / max(n, 1):.4f}" for k, v in running.items())
        log(f"epoch {epoch + 1}/{epochs}: {msg} ({time.time() - start:.0f}s)")
        if on_epoch:
            on_epoch(epoch)


# ---------------------------------------------------------------------------- evaluation


def predict_images(
    model: EllipseNet,
    images: Sequence[np.ndarray],
    names: Sequence[str],
    batch_size: int = 16,
    tta: bool = True,
) -> dict[Head, dict[str, ImageLabel]]:
    """Predictions of both heads for full-size images, in image coordinates."""
    dev = device()
    model.to(dev).eval()
    out: dict[Head, dict[str, ImageLabel]] = {"mask": {}, "params": {}}
    size = model.cfg.size
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        canvases, invs = [], []
        for img in chunk:
            c, h = letterbox(img, size)
            canvases.append(to_tensor(c))
            invs.append(np.linalg.inv(h))
        preds = run_batch(model, torch.stack(canvases).to(dev), tta)
        for name, pred, inv in zip(names[i : i + batch_size], preds, invs, strict=True):
            heads: tuple[Head, ...] = ("mask", "params")
            for head in heads:
                e = pred.ellipse(head)
                if e is None:
                    out[head][name] = ImageLabel(status="no_ellipse", source="detector")
                    continue
                e = transform_ellipse(e, inv)
                out[head][name] = ImageLabel(
                    ellipses=[ScoredEllipse(**e.model_dump(), score=pred.confidence)],
                    prominent=0,
                    status="labeled",
                    source="detector",
                )
    return out


def evaluate_dir(
    model: EllipseNet, directory: Path, limit: int | None = None
) -> dict[Head, Summary]:
    ds = Dataset(directory)
    names = [n for n, lab in sorted(ds.labels.items()) if lab.status == "labeled"][:limit]
    images = [load_image(ds.image_path(n)) for n in names]
    preds = predict_images(model, images, names)
    gt = {n: ds.labels[n] for n in names}
    return {head: evaluate(gt, p) for head, p in preds.items()}


def _short(summaries: dict[Head, Summary]) -> str:
    return "  ".join(
        f"[{h}] good {s.good_rate:.1%} hit {s.hit_rate:.1%} strict {s.strict_hit_rate:.1%}"
        f" medIoU {s._median('iou') or 0:.3f}"
        for h, s in summaries.items()
    )


# ---------------------------------------------------------------------------- stages


def pretrain(
    synth: Path,
    val: Path | None,
    out: Path,
    cfg: ModelConfig,
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 32,
    workers: int = 8,
    log: Log = print,
) -> EllipseNet:
    model = EllipseNet(cfg)
    train_ds = SynthDataset(synth, cfg.size, SYNTH_AUG)
    log(f"pretraining {cfg.backbone}@{cfg.size} on {len(train_ds)} synthetic images")

    def on_epoch(epoch: int) -> None:
        save_checkpoint(out, model, {"stage": "pretrain", "epoch": epoch + 1})
        if val is not None:
            log(f"  synth val: {_short(evaluate_dir(model, val, limit=300))}")

    fit(model, train_ds, epochs, lr, batch_size, workers, on_epoch, log)
    return model


def _finetune(
    init: Path,
    items: list[Item],
    synth: Path | None,
    epochs: int,
    lr: float,
    batch_size: int,
    repeats: int,
    workers: int,
    log: Log,
) -> EllipseNet:
    model = load_checkpoint(init)
    size = model.cfg.size
    parts: list[torch.utils.data.Dataset[dict[str, torch.Tensor]]] = [
        ItemDataset(items, size, REAL_AUG, repeats=repeats)
    ]
    n_real = len(parts[0])  # type: ignore[arg-type]
    if synth is not None:  # keep seeing synthetic data so the model does not forget
        synth_ds = SynthDataset(synth, size, SYNTH_AUG, seed=1)
        idx = np.random.default_rng(0).choice(len(synth_ds), min(n_real, len(synth_ds)), False)
        parts.append(Subset(synth_ds, idx.tolist()))
    fit(model, ConcatDataset(parts), epochs, lr, batch_size, workers, None, log)
    return model


def finetune(
    init: Path,
    real: Path,
    out: Path,
    synth: Path | None,
    folds: int,
    epochs: int = 8,
    lr: float = 3e-4,
    batch_size: int = 16,
    repeats: int = 64,
    workers: int = 8,
    cv_out: Path | None = None,
    log: Log = print,
) -> None:
    """K-fold CV on the labeled photos (held-out predictions to ``cv_out``), then train on all."""
    items = load_items(real)
    names = [n for n, lab in sorted(Dataset(real).labels.items()) if lab.status == "labeled"]
    names = [n for n in names if Dataset(real).labels[n].prominent_ellipse is not None]
    assert len(names) == len(items)
    ds = Dataset(real)
    gt = {n: ds.labels[n] for n in names}
    full_images = [load_image(ds.image_path(n)) for n in names]

    base = predict_images(load_checkpoint(init), full_images, names)
    for head, p in base.items():
        log(f"pretrained only, all {len(names)} real images [{head}]:")
        log(evaluate(gt, p).format(per_image=False))

    if folds > 1:
        order = np.random.default_rng(0).permutation(len(items))
        cv: dict[Head, dict[str, ImageLabel]] = {"mask": {}, "params": {}}
        for k in range(folds):
            held = sorted(order[k::folds].tolist())
            train_items = [it for i, it in enumerate(items) if i not in held]
            log(f"fold {k + 1}/{folds}: holding out {[names[i] for i in held]}")
            model = _finetune(
                init, train_items, synth, epochs, lr, batch_size, repeats, workers, log
            )
            preds = predict_images(model, [full_images[i] for i in held], [names[i] for i in held])
            for head, p in preds.items():
                cv[head].update(p)
        for head, p in cv.items():
            log(f"fine-tuned, held-out folds [{head}]:")
            log(evaluate(gt, p).format())
            if cv_out is not None:
                path = cv_out.with_name(f"{cv_out.stem}_{head}{cv_out.suffix}")
                save_labels(path, p)
                log(f"wrote {path}")

    log(f"fine-tuning on all {len(items)} labeled images")
    model = _finetune(init, items, synth, epochs, lr, batch_size, repeats, workers, log)
    save_checkpoint(out, model, {"stage": "finetune", "init": str(init), "n_real": len(items)})
    log(f"saved {out}")
