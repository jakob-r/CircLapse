"""A studio project: the candidate photos of one hyperlapse with their order and export settings.

A project is a folder with copies of every source photo in which the detector found an ellipse
(named ``YYYYMMDD-HHMMSS_<original>``), their ``labels.json`` and a ``project.json`` that
records the frame order, the photos the user excluded and the export settings. The source
folder is never changed; photos without an ellipse are not copied.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from circlapse.detect.base import Detector
from circlapse.labels import Dataset, ImageLabel, list_images, save_labels
from circlapse.organize import Photo, dated_name, detect_all, read_photo
from circlapse.render import RenderConfig

PROJECT_FILE = "project.json"
EXPORT_DIR = "export"

Format = Literal["mp4", "webm", "gif"]


def _default_formats() -> list[Format]:
    return ["mp4", "gif"]


class ExportSettings(BaseModel):
    formats: list[Format] = Field(default_factory=_default_formats)
    size: int = Field(RenderConfig.size, ge=64, le=4096)
    fps: float = Field(RenderConfig.fps, gt=0, le=60)
    keep: float = Field(RenderConfig.keep, gt=0, le=1)
    max_extent: float = Field(RenderConfig.max_extent, ge=1, le=10)
    grade: bool = RenderConfig.grade

    def render_config(self) -> RenderConfig:
        return RenderConfig(
            size=self.size,
            fps=self.fps,
            keep=self.keep,
            max_extent=self.max_extent,
            grade=self.grade,
        )


class Project(BaseModel):
    name: str = "circlapse"
    source: str = ""
    order: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    times: dict[str, datetime | None] = Field(default_factory=dict)
    settings: ExportSettings = Field(default_factory=ExportSettings)

    @property
    def included(self) -> list[str]:
        excluded = set(self.excluded)
        return [n for n in self.order if n not in excluded]


def load_project(directory: Path) -> Project:
    return Project.model_validate_json((directory / PROJECT_FILE).read_text())


def save_project(directory: Path, project: Project) -> None:
    """Atomically write ``project.json``."""
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{PROJECT_FILE}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(project.model_dump_json(indent=2))
    os.replace(tmp, directory / PROJECT_FILE)


def is_project(directory: Path) -> bool:
    return (directory / PROJECT_FILE).is_file()


def _by_time(names: list[str], times: dict[str, datetime | None]) -> list[str]:
    """Capture order; undated photos last, in name order."""
    return sorted(names, key=lambda n: (times.get(n) is None, times.get(n) or datetime.min, n))


def sync(project: Project, directory: Path) -> bool:
    """Match the project to the image files: forget deleted ones, append new ones by date.

    Returns whether the project changed.
    """
    files = list_images(directory)
    present = set(files)
    order = [n for n in project.order if n in present]
    known = set(order)
    new = [n for n in files if n not in known]
    for name in new:
        project.times[name] = read_photo(directory / name).time
    changed = len(order) != len(project.order) or bool(new)
    project.order = order + _by_time(new, project.times)
    project.excluded = [n for n in project.excluded if n in present]
    project.times = {n: t for n, t in project.times.items() if n in present}
    return changed


def create_project(
    source: Path,
    directory: Path,
    detector: Detector,
    fallback: dict[str, Photo] | None = None,
    log: Callable[[str], None] = print,
) -> Project:
    """Detect ellipses in the photos directly in ``source`` and copy those with one into
    ``directory``, which becomes a project sorted by capture time."""
    photos = detect_all(source, detector, log=log, fallback=fallback)
    directory.mkdir(parents=True, exist_ok=True)
    labels: dict[str, ImageLabel] = {}
    times: dict[str, datetime | None] = {}
    skipped = []
    for photo in photos:
        if not photo.label.ellipses:
            skipped.append(photo.name)
            continue
        name = dated_name(photo)
        stem, suffix, i = Path(name).stem, Path(name).suffix, 2
        while name in labels or (directory / name).exists():
            name = f"{stem}_{i}{suffix}"
            i += 1
        shutil.copy2(source / photo.name, directory / name)
        labels[name] = photo.label
        times[name] = photo.time
    if skipped:
        log(f"no ellipse found in {len(skipped)} photos (not copied): {', '.join(skipped)}")
    save_labels(directory / "labels.json", labels)
    project = Project(
        name=source.resolve().name,
        source=str(source.resolve()),
        order=_by_time(list(labels), times),
        times=times,
    )
    sync(project, directory)
    save_project(directory, project)
    return project


def usable_labels(project: Project, dataset: Dataset) -> dict[str, ImageLabel]:
    """Labels of the included photos that have a prominent ellipse."""
    out = {}
    for name in project.included:
        label = dataset.label(name)
        if label.status == "labeled" and label.prominent_ellipse is not None:
            out[name] = label
    return out
