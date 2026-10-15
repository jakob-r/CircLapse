"""FastAPI backend for the annotation tool.

All ellipse coordinates exchanged with the frontend are in full-resolution pixels of the
EXIF-corrected image; the served display image may be downscaled, and ``/api/images``
reports the full size so the frontend can map between both.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from circlapse.ellipse import Ellipse
from circlapse.labels import Dataset, ImageLabel, Status, load_labels, open_image

THUMB_SIZE = 256
DISPLAY_SIZE = 2048
FRONTEND_DIST = Path(
    os.environ.get("CIRCLAPSE_FRONTEND", Path(__file__).parents[3] / "frontend" / "dist")
)


class ImageInfo(BaseModel):
    name: str
    width: int
    height: int
    status: Status
    n_ellipses: int
    prominent: Ellipse | None
    predicted: Ellipse | None


class FitRequest(BaseModel):
    points: list[tuple[float, float]]


def _jpeg(im: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def create_app(data_dir: Path, predictions: Path | None = None) -> FastAPI:
    dataset = Dataset(data_dir)
    preds = load_labels(predictions) if predictions else {}
    cache_dir = data_dir / ".thumbs"
    sizes: dict[str, tuple[int, int]] = {}

    app = FastAPI(title="circlapse annotator")

    def path_of(name: str) -> Path:
        try:
            return dataset.image_path(name)
        except FileNotFoundError:
            raise HTTPException(404, f"unknown image {name}") from None

    def size_of(name: str) -> tuple[int, int]:
        if name not in sizes:
            with Image.open(path_of(name)) as im:
                w, h = im.size
                # EXIF orientations 5-8 swap width and height
                if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):
                    w, h = h, w
            sizes[name] = (w, h)
        return sizes[name]

    def cached(name: str, kind: str, max_side: int, quality: int) -> Path:
        src = path_of(name)
        dst = cache_dir / kind / f"{name}.jpg"
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            im = open_image(src)
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(_jpeg(im, quality))
        return dst

    @app.get("/api/images")
    def list_images() -> list[ImageInfo]:
        out = []
        for name in dataset.images():
            label = dataset.label(name)
            w, h = size_of(name)
            pred = preds.get(name)
            out.append(
                ImageInfo(
                    name=name,
                    width=w,
                    height=h,
                    status=label.status,
                    n_ellipses=len(label.ellipses),
                    prominent=label.prominent_ellipse,
                    predicted=pred.prominent_ellipse if pred else None,
                )
            )
        return out

    @app.get("/api/thumb/{name}")
    def thumb(name: str) -> FileResponse:
        return FileResponse(cached(name, "thumb", THUMB_SIZE, 80), media_type="image/jpeg")

    @app.get("/api/image/{name}")
    def image(name: str) -> FileResponse:
        return FileResponse(cached(name, "display", DISPLAY_SIZE, 92), media_type="image/jpeg")

    @app.get("/api/labels/{name}")
    def get_label(name: str) -> ImageLabel:
        path_of(name)
        return dataset.label(name)

    @app.put("/api/labels/{name}")
    def put_label(name: str, label: ImageLabel) -> ImageLabel:
        path_of(name)
        if label.prominent is not None and not 0 <= label.prominent < len(label.ellipses):
            raise HTTPException(422, "prominent index out of range")
        dataset.set_label(name, label)
        return label

    @app.get("/api/predictions/{name}")
    def get_prediction(name: str) -> ImageLabel | None:
        path_of(name)
        return preds.get(name)

    @app.post("/api/fit")
    def fit(req: FitRequest) -> Ellipse:
        try:
            return Ellipse.from_points(req.points)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None

    if (FRONTEND_DIST / "index.html").exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    else:

        @app.get("/")
        def no_frontend() -> Response:
            return Response(
                "Frontend not built. Run `make build`, or `make dev` for development.",
                media_type="text/plain",
            )

    return app


def app_from_env() -> FastAPI:
    """Factory for ``uvicorn --factory`` (dev server with reload)."""
    pred = os.environ.get("CIRCLAPSE_PRED")
    return create_app(
        Path(os.environ.get("CIRCLAPSE_DATA", "data/real")), Path(pred) if pred else None
    )
