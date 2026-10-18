"""FastAPI backend of the studio: the annotator routes plus project order, aligned previews
and video export for one project folder (see :mod:`circlapse.studio.project`)."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from circlapse.annotate.app import ImageStore, annotation_router, mount_frontend
from circlapse.ellipse import Ellipse
from circlapse.labels import Dataset, load_image
from circlapse.render import Frame, grade, plan_frames, render, warp
from circlapse.studio.project import (
    EXPORT_DIR,
    ExportSettings,
    Project,
    adopt_label_skips,
    load_project,
    save_project,
    sync,
    usable_labels,
)
from circlapse.studio.similarity import SortBy, features, similarity_path

PREVIEW_SIZE = 320


class PhotoInfo(BaseModel):
    name: str
    time: datetime | None
    # Has a prominent ellipse, so it can be a frame
    usable: bool
    # Included and usable, but its circle is too close to the border for the common crop
    dropped: bool
    # Aligned preview URL, versioned by ellipse and crop; None when not usable
    preview: str | None


class ProjectInfo(BaseModel):
    name: str
    path: str
    order: list[str]
    excluded: list[str]
    settings: ExportSettings
    # Common crop as frame half side in circle radii; None when no photo can be a frame
    extent: float | None
    frames: int
    photos: list[PhotoInfo]


class ProjectUpdate(BaseModel):
    order: list[str]
    excluded: list[str]
    settings: ExportSettings


class SortedOrder(BaseModel):
    order: list[str]


class ExportStatus(BaseModel):
    state: Literal["idle", "running", "done", "error"] = "idle"
    log: list[str] = []
    outputs: list[str] = []
    dropped: list[str] = []
    error: str | None = None
    directory: str = ""


def _preview_key(e: Ellipse, extent: float, size: int, graded: bool) -> str:
    text = f"{e.model_dump_json()}|{extent:.4f}|{size}|{graded}"
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def create_studio_app(project_dir: Path) -> FastAPI:
    project_dir = project_dir.resolve()
    project = load_project(project_dir)
    dataset = Dataset(project_dir)
    if sync(project, project_dir) | adopt_label_skips(project, dataset):
        save_project(project_dir, project)
    store = ImageStore(dataset)
    export_dir = project_dir / EXPORT_DIR
    status = ExportStatus(directory=str(export_dir))
    lock = threading.Lock()
    # Similarity descriptors by photo, valid while its ellipse and crop are unchanged
    descriptors: dict[str, tuple[str, dict[SortBy, np.ndarray]]] = {}

    app = FastAPI(title="circlapse studio")
    app.include_router(annotation_router(store, names=lambda: project.order))

    def shapes() -> dict[str, tuple[int, int]]:
        out = {}
        for name in project.order:
            w, h = store.size_of(name)
            out[name] = (h, w)
        return out

    def plan(p: Project) -> tuple[list[Frame], list[str], float | None]:
        try:
            cfg = p.settings.render_config()
            return plan_frames(usable_labels(p, dataset), shapes(), cfg, p.included)
        except ValueError:
            return [], p.included, None

    def preview_extent(name: str, e: Ellipse, extent: float | None) -> float:
        """The common crop, or the photo's own widest crop when there is none."""
        if extent is not None:
            return extent
        own = Frame(name, e, shapes()[name]).max_extent
        return max(1.0, min(own, project.settings.max_extent))

    def info() -> ProjectInfo:
        frames, dropped, extent = plan(project)
        dropped_set, excluded = set(dropped), set(project.excluded)
        photos = []
        for name in project.order:
            label = dataset.label(name)
            e = label.prominent_ellipse
            usable = label.status == "labeled" and e is not None
            preview = None
            if usable and e is not None:
                s = preview_extent(name, e, extent)
                key = _preview_key(e, s, PREVIEW_SIZE, project.settings.grade)
                preview = f"/api/preview/{name}?v={key}"
            photos.append(
                PhotoInfo(
                    name=name,
                    time=project.times.get(name),
                    usable=usable,
                    dropped=usable and name in dropped_set and name not in excluded,
                    preview=preview,
                )
            )
        return ProjectInfo(
            name=project.name,
            path=str(project_dir),
            order=project.order,
            excluded=project.excluded,
            settings=project.settings,
            extent=extent,
            frames=len(frames),
            photos=photos,
        )

    @app.get("/api/project")
    def get_project() -> ProjectInfo:
        if sync(project, project_dir) | adopt_label_skips(project, dataset):
            save_project(project_dir, project)
        return info()

    @app.put("/api/project")
    def put_project(update: ProjectUpdate) -> ProjectInfo:
        if sorted(update.order) != sorted(project.order):
            raise HTTPException(422, "order must contain every photo of the project exactly once")
        if not set(update.excluded) <= set(project.order):
            raise HTTPException(422, "excluded photos must be part of the project")
        project.order = update.order
        project.excluded = [n for n in update.order if n in set(update.excluded)]
        project.settings = update.settings
        save_project(project_dir, project)
        return info()

    @app.get("/api/preview/{name}")
    def preview(name: str, size: int = PREVIEW_SIZE) -> FileResponse:
        store.path_of(name)
        e = dataset.label(name).prominent_ellipse
        if e is None:
            raise HTTPException(404, f"{name} has no prominent ellipse")
        size = max(32, min(size, 1080))
        extent = preview_extent(name, e, plan(project)[2])
        graded = project.settings.grade
        key = _preview_key(e, extent, size, graded)
        dst = store.cache_dir / "preview" / f"{name}.{key}.jpg"
        if not dst.exists():
            src = store.display(name)
            image = load_image(src)
            scale = image.shape[1] / store.size_of(name)[0]
            frame = warp(image, e.scaled(scale), extent, size)
            if graded:
                frame = grade(frame)
            dst.parent.mkdir(parents=True, exist_ok=True)
            for old in dst.parent.glob(f"{name}.*.jpg"):
                old.unlink(missing_ok=True)
            cv2.imwrite(str(dst), cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
        # The URL carries the version, so the browser may keep it
        return FileResponse(
            dst, media_type="image/jpeg", headers={"Cache-Control": "max-age=31536000"}
        )

    def descriptor(name: str, e: Ellipse, extent: float) -> dict[SortBy, np.ndarray]:
        key = f"{e.model_dump_json()}|{extent:.4f}"
        cached = descriptors.get(name)
        if cached is None or cached[0] != key:
            width = store.size_of(name)[0]
            cached = (key, features(store.display(name), e, width, extent))
            descriptors[name] = cached
        return cached[1]

    @app.get("/api/sorted")
    def sorted_order(by: SortBy, loop: bool = False) -> SortedOrder:
        """The project order with the photos that have a circle arranged by similarity;
        the others keep their relative order at the end. ``loop`` makes the last photo
        similar to the first, for videos that play in a loop. Photos are compared as the
        video shows them: aligned and cropped like the aligned previews."""
        extent = plan(project)[2]
        ellipses = {}
        for name in project.order:
            label = dataset.label(name)
            if label.status == "labeled" and label.prominent_ellipse is not None:
                ellipses[name] = label.prominent_ellipse
        names = list(ellipses)
        with ThreadPoolExecutor() as pool:
            descs = list(
                pool.map(
                    lambda n: descriptor(n, ellipses[n], preview_extent(n, ellipses[n], extent)),
                    names,
                )
            )
        x = np.array([d[by] for d in descs]).reshape(len(names), -1)
        ordered = [names[i] for i in similarity_path(x, closed=loop)]
        rest = [n for n in project.order if n not in ellipses]
        return SortedOrder(order=ordered + rest)

    def run_export(p: Project) -> None:
        def log(msg: str) -> None:
            status.log.append(msg.strip())

        try:
            result = render(
                project_dir,
                usable_labels(p, dataset),
                export_dir / p.name,
                p.settings.render_config(),
                p.settings.formats,
                log=log,
                order=p.included,
            )
            status.outputs = [o.name for o in result.outputs]
            status.dropped = result.dropped
            status.state = "done"
        except Exception as e:  # reported to the UI
            status.error = str(e) or type(e).__name__
            status.state = "error"

    @app.post("/api/export")
    def start_export(settings: ExportSettings) -> ExportStatus:
        nonlocal status
        with lock:
            if status.state == "running":
                raise HTTPException(409, "an export is already running")
            if not settings.formats:
                raise HTTPException(422, "choose at least one format")
            project.settings = settings
            save_project(project_dir, project)
            status = ExportStatus(state="running", directory=str(export_dir))
            snapshot = project.model_copy(deep=True)
        threading.Thread(target=run_export, args=(snapshot,), daemon=True).start()
        return status

    @app.get("/api/export")
    def export_status() -> ExportStatus:
        return status

    @app.get("/api/export/file/{filename}")
    def export_file(filename: str) -> FileResponse:
        path = (export_dir / filename).resolve()
        if path.parent != export_dir or not path.is_file():
            raise HTTPException(404, f"unknown export {filename}")
        return FileResponse(path, filename=filename)

    mount_frontend(app)
    return app
