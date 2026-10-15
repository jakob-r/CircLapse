"""Dataset and label IO.

A dataset is a directory of images plus a ``labels.json`` mapping image file name to
:class:`ImageLabel`. Detector predictions use the same file format.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, RootModel

from circlapse.ellipse import Ellipse

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
LABELS_FILE = "labels.json"

Status = Literal["unlabeled", "labeled", "no_ellipse", "skip"]
Source = Literal["human", "synth", "detector"]


class ScoredEllipse(Ellipse):
    score: float | None = None


class ImageLabel(BaseModel):
    ellipses: list[ScoredEllipse] = Field(default_factory=list)
    prominent: int | None = None
    status: Status = "unlabeled"
    source: Source = "human"
    runtime_s: float | None = None

    @property
    def prominent_ellipse(self) -> ScoredEllipse | None:
        if self.prominent is None or not 0 <= self.prominent < len(self.ellipses):
            return None
        return self.ellipses[self.prominent]


class LabelFile(RootModel[dict[str, ImageLabel]]):
    root: dict[str, ImageLabel] = Field(default_factory=dict)


def load_labels(path: Path) -> dict[str, ImageLabel]:
    if not path.exists():
        return {}
    return LabelFile.model_validate_json(path.read_text()).root


def save_labels(path: Path, labels: dict[str, ImageLabel]) -> None:
    """Atomically write ``labels`` to ``path`` (sorted by image name)."""
    data = LabelFile(dict(sorted(labels.items()))).model_dump_json(indent=2, exclude_none=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(data)
    os.replace(tmp, path)


def list_images(directory: Path) -> list[str]:
    return sorted(
        p.name
        for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_SUFFIXES
    )


def open_image(path: Path) -> Image.Image:
    """Open an image with EXIF orientation applied, as RGB."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def load_image(path: Path) -> np.ndarray:
    """EXIF-corrected RGB uint8 array of shape (h, w, 3)."""
    return np.asarray(open_image(path))


class Dataset:
    """A directory of images with a labels file."""

    def __init__(self, directory: Path, labels_path: Path | None = None) -> None:
        self.dir = Path(directory)
        self.labels_path = labels_path or self.dir / LABELS_FILE
        self.labels = load_labels(self.labels_path)

    def images(self) -> list[str]:
        return list_images(self.dir)

    def image_path(self, name: str) -> Path:
        path = (self.dir / name).resolve()
        if path.parent != self.dir.resolve() or not path.is_file():
            raise FileNotFoundError(name)
        return path

    def label(self, name: str) -> ImageLabel:
        return self.labels.get(name, ImageLabel())

    def set_label(self, name: str, label: ImageLabel) -> None:
        self.labels[name] = label
        self.save()

    def save(self) -> None:
        save_labels(self.labels_path, self.labels)
