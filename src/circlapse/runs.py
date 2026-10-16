"""Training runs: one folder per run under ``runs/`` and a results table in ``RESULTS.md``.

A run folder holds the checkpoints, logs and held-out CV predictions of one
``make experiment`` plus ``meta.json`` (git commit, settings, scores). ``RESULTS.md`` is
committed and lists every run, so a score can be traced back to the code that produced it:
check out the commit and use the run's checkpoint (``make use RUN=...``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from circlapse.detect.base import DEFAULT_ROUNDNESS
from circlapse.labels import ImageLabel, load_labels
from circlapse.metrics import Summary, evaluate

RUNS = Path("runs")
RESULTS = Path("RESULTS.md")
META = "meta.json"

HEADER = (
    "| run | date | commit | photos | mask good / hit / strict"
    " | mask rounded good / hit / strict | params good / hit / strict | note |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def git_version() -> dict[str, str | bool]:
    """Current commit and whether tracked files have uncommitted changes."""
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no").strip())
    return {"commit": _git("rev-parse", "--short", "HEAD").strip(), "dirty": dirty}


def start(run: Path, note: str) -> None:
    """Create the run folder and record the code version and training settings."""
    from circlapse.nn.train import REAL_AUG, SYNTH_AUG
    from circlapse.synth import SynthConfig

    run.mkdir(parents=True, exist_ok=False)
    meta = {
        "run": run.name,
        "date": datetime.now().isoformat(timespec="minutes"),
        **git_version(),
        "note": note,
        "settings": {
            "synth_aug": asdict(SYNTH_AUG),
            "real_aug": asdict(REAL_AUG),
            "synth_config": asdict(SynthConfig()),
        },
    }
    (run / META).write_text(json.dumps(meta, indent=2) + "\n")


def _rounded(preds: dict[str, ImageLabel], strength: float) -> dict[str, ImageLabel]:
    out = {}
    for name, label in preds.items():
        label = label.model_copy(deep=True)
        label.ellipses = [
            type(e)(**{**e.rounded(strength).model_dump(), "score": e.score})
            for e in label.ellipses
        ]
        out[name] = label
    return out


def _scores(s: Summary) -> dict[str, float]:
    return {"good": s.good_rate, "hit": s.hit_rate, "strict": s.strict_hit_rate}


def score(run: Path, gt_path: Path, roundness: float = DEFAULT_ROUNDNESS) -> dict[str, object]:
    """CV scores of a run, on the photos its CV covered (labels may have grown since)."""
    gt_all = load_labels(gt_path)
    out: dict[str, object] = {"roundness": roundness}
    for head in ("mask", "params"):
        path = run / f"cv_{head}.json"
        if not path.exists():
            continue
        preds = load_labels(path)
        gt = {n: gt_all[n] for n in preds if n in gt_all}
        out[head] = _scores(evaluate(gt, preds))
        if head == "mask":
            out["mask_rounded"] = _scores(evaluate(gt, _rounded(preds, roundness)))
        out["photos"] = len(gt)
    return out


def _fmt(s: object) -> str:
    if not isinstance(s, dict):
        return "-"
    return " / ".join(f"{100 * s[k]:.1f}" for k in ("good", "hit", "strict"))


def record(run: Path, gt_path: Path) -> str:
    """Score ``run``, store the scores in its meta.json and add its row to RESULTS.md."""
    meta_path = run / META
    meta: dict[str, object] = (
        json.loads(meta_path.read_text()) if meta_path.exists() else {"run": run.name}
    )
    s = score(run, gt_path)
    meta["scores"] = s
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    commit = f"{meta.get('commit', '?')}{'+dirty' if meta.get('dirty') else ''}"
    row = (
        f"| {meta['run']} | {str(meta.get('date', '?'))[:10]} | {commit} | {s.get('photos', '-')}"
        f" | {_fmt(s.get('mask'))} | {_fmt(s.get('mask_rounded'))} | {_fmt(s.get('params'))}"
        f" | {meta.get('note', '')} |\n"
    )
    text = RESULTS.read_text() if RESULTS.exists() else ""
    if HEADER not in text:
        text += ("\n" if text else "") + HEADER
    lines = [ln for ln in text.splitlines(keepends=True) if not ln.startswith(f"| {meta['run']} |")]
    RESULTS.write_text("".join(lines) + row)
    return row


def use(run: Path, checkpoints: Path = Path("checkpoints")) -> None:
    """Make ``run``'s checkpoints the current ones (``checkpoints/*.pt``)."""
    checkpoints.mkdir(exist_ok=True)
    copied = 0
    for name in ("pretrain.pt", "finetune.pt"):
        if (run / name).exists():
            shutil.copy2(run / name, checkpoints / name)
            copied += 1
    if not copied:
        raise FileNotFoundError(f"no checkpoints in {run}")
