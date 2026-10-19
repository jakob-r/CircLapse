"""Render the circle hyperlapse: rectify, align, crop, grade and encode the photos.

Every photo is mapped by an affine transform that turns its prominent ellipse into a circle
(stretching along the minor axis, so the photo keeps its orientation) and puts the circle in
the center of a square frame. All frames share one crop extent ``s``: the frame's half side
measured in circle radii. ``s`` is the largest value for which the frame stays inside every
kept photo, so no frame shows empty borders. Photos that would force a much tighter crop can
be dropped with ``keep``, and ``zoom`` crops tighter than that.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from circlapse.ellipse import Ellipse
from circlapse.labels import ImageLabel, list_images, load_image

FORMATS = ("mp4", "webm", "gif")
_CORNERS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)


@dataclass(frozen=True)
class RenderConfig:
    size: int = 1080
    fps: float = 10.0
    # Fraction of photos that must survive the common crop; the rest (those whose circle is
    # closest to the photo border) are dropped to allow a wider crop
    keep: float = 1.0
    # Extra zoom on top of the widest crop that works for all kept photos: the crop extent is
    # divided by it (never below ``min_extent``), so 1 keeps the minimal alignment crop
    zoom: float = 1.0
    # Photos whose circle does not fit this many radii into the photo are always dropped
    min_extent: float = 1.0
    grade: bool = True
    gif_size: int | None = None


@dataclass(frozen=True)
class Frame:
    name: str
    ellipse: Ellipse
    shape: tuple[int, int]  # (h, w) of the EXIF-corrected photo

    @property
    def rectify(self) -> np.ndarray:
        """2x2 matrix mapping photo offsets from the center to circle units (radius 1)."""
        return rectify_matrix(self.ellipse)

    @property
    def max_extent(self) -> float:
        return max_extent(self.ellipse, self.shape)


def rectify_matrix(e: Ellipse) -> np.ndarray:
    th = math.radians(e.angle_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return rot @ np.diag([1 / e.a, 1 / e.b]) @ rot.T


def max_extent(e: Ellipse, shape: tuple[int, ...]) -> float:
    """Largest frame half side, in circle radii, whose frame lies inside the photo."""
    h, w = shape[:2]
    # Frame corners at extent 1, as offsets in the photo; they scale linearly with the extent
    offsets = _CORNERS @ np.linalg.inv(rectify_matrix(e)).T
    bounds = []
    for d, c, hi in ((offsets[:, 0], e.cx, w), (offsets[:, 1], e.cy, h)):
        bounds += [(hi - c) / x for x in d if x > 0] + [-c / x for x in d if x < 0]
    return max(0.0, min(bounds))


def choose_extent(extents: Iterable[float], cfg: RenderConfig) -> float:
    """Common crop extent: the widest that fits a fraction ``cfg.keep`` of the photos, zoomed in
    by ``cfg.zoom`` but never tighter than ``cfg.min_extent``."""
    ok = sorted((s for s in extents if s >= cfg.min_extent), reverse=True)
    if not ok:
        raise ValueError(f"no photo leaves room for a crop of {cfg.min_extent} circle radii")
    n = max(1, math.ceil(cfg.keep * len(ok)))
    return max(cfg.min_extent, ok[n - 1] / cfg.zoom)


def warp_matrix(e: Ellipse, extent: float, size: int) -> np.ndarray:
    """2x3 affine matrix from photo pixels to frame pixels."""
    k = size / 2 / extent
    lin = k * rectify_matrix(e)
    center = np.array([size / 2, size / 2])
    offset = center - lin @ np.array([e.cx, e.cy])
    return np.hstack([lin, offset[:, None]])


def warp(image: np.ndarray, e: Ellipse, extent: float, size: int) -> np.ndarray:
    """The aligned square frame; large photos are first shrunk with area averaging."""
    m = warp_matrix(e, extent, size)
    zoom = math.sqrt(abs(np.linalg.det(m[:, :2])))
    if zoom < 0.5:
        f = 2 * zoom
        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        m = np.hstack([m[:, :2] / f, m[:, 2:]])
    return cv2.warpAffine(
        image, m, (size, size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
    )


# ---------------------------------------------------------------------- grading


def lightness(frame: np.ndarray) -> float:
    """Mean L* (0..255) of the central part of an RGB frame."""
    h, w = frame.shape[:2]
    center = frame[h // 8 : h - h // 8, w // 8 : w - w // 8]
    return float(cv2.cvtColor(center, cv2.COLOR_RGB2LAB)[..., 0].mean())


def grade(frame: np.ndarray, target_lightness: float | None = None) -> np.ndarray:
    """Mild automatic grade: contrast stretch, local contrast, partial white balance,
    a little saturation, and exposure pulled halfway towards ``target_lightness``
    (the sequence median) to reduce flicker between frames."""
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    lum, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    lo, hi = np.percentile(lum, (0.5, 99.5))
    if hi - lo > 1:
        stretched = (lum - lo) * (255 / (hi - lo))
        lum = 0.5 * lum + 0.5 * stretched
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    local = clahe.apply(np.clip(lum, 0, 255).astype(np.uint8)).astype(np.float32)
    lum = 0.7 * lum + 0.3 * local
    if target_lightness is not None:
        lum = lum + 0.5 * (target_lightness - float(lum.mean()))

    # Remove a quarter of the color cast, then boost chroma by 8%
    a = (a - 0.25 * (a.mean() - 128) - 128) * 1.08 + 128
    b = (b - 0.25 * (b.mean() - 128) - 128) * 1.08 + 128

    out = np.stack([lum, a, b], axis=-1)
    rgb = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.2)
    return cv2.addWeighted(rgb, 1.25, blur, -0.25, 0)


# ---------------------------------------------------------------------- encoding


def encode(frames_dir: Path, out: Path, fmt: str, cfg: RenderConfig) -> Path:
    """Encode ``frames_dir/%05d.png`` with ffmpeg to ``out`` with suffix ``fmt``."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    path = out.with_suffix(f".{fmt}")
    args = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(cfg.fps)]
    args += ["-i", str(frames_dir / "%05d.png")]
    if fmt == "mp4":
        args += ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"]
        args += ["-movflags", "+faststart"]
    elif fmt == "webm":
        args += ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-row-mt", "1"]
        args += ["-pix_fmt", "yuv420p"]
    elif fmt == "gif":
        gif_size = cfg.gif_size or cfg.size
        args += [
            "-vf",
            f"scale={gif_size}:{gif_size}:flags=lanczos,split[a][b];"
            "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a",
            "-loop",
            "0",
        ]
    else:
        raise ValueError(f"unknown format {fmt!r}, expected one of {FORMATS}")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([*args, str(path)], check=True)
    return path


# ---------------------------------------------------------------------- pipeline


@dataclass
class RenderResult:
    outputs: list[Path]
    used: list[str]
    dropped: list[str]
    extent: float


def plan_frames(
    labels: dict[str, ImageLabel],
    shapes: dict[str, tuple[int, int]],
    cfg: RenderConfig,
    order: list[str] | None = None,
) -> tuple[list[Frame], list[str], float]:
    """Frames to render, dropped photo names and the common crop extent.

    Frames follow ``order`` (names missing there are dropped), by default name order.
    """
    names = [n for n in order if n in shapes] if order is not None else sorted(shapes)
    candidates = [
        Frame(name, e, shapes[name])
        for name in names
        if (e := labels[name].prominent_ellipse if name in labels else None) is not None
    ]
    extent = choose_extent((f.max_extent for f in candidates), cfg)
    # A tiny tolerance keeps the photo that defines the extent despite float rounding
    used = [f for f in candidates if f.max_extent >= extent * (1 - 1e-9)]
    used_names = {f.name for f in used}
    dropped = [n for n in names if n not in used_names]
    return used, dropped, extent


def image_shape(path: Path) -> tuple[int, int]:
    """(h, w) of the EXIF-corrected image without decoding the pixels."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
        if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):
            w, h = h, w
    return h, w


def render(
    directory: Path,
    labels: dict[str, ImageLabel],
    out: Path,
    cfg: RenderConfig = RenderConfig(),  # noqa: B008
    formats: Iterable[str] = FORMATS,
    frames_out: Path | None = None,
    log: Callable[[str], None] = lambda _: None,
    order: list[str] | None = None,
) -> RenderResult:
    """Render the photos of ``directory`` with their prominent ellipses in ``labels``.

    Writes ``out.<fmt>`` for every format; ``frames_out`` keeps the graded frames as PNGs.
    ``order`` sets the frame order (see :func:`plan_frames`).
    """
    shapes = {name: image_shape(directory / name) for name in list_images(directory)}
    frames, dropped, extent = plan_frames(labels, shapes, cfg, order)
    log(
        f"{len(frames)} frames, {len(dropped)} dropped, crop extent {extent:.2f} circle radii"
        f" (circle diameter {100 / extent:.0f}% of the frame)"
    )
    with tempfile.TemporaryDirectory(prefix="circlapse-") as tmp:
        raw_dir, final_dir = Path(tmp) / "raw", Path(tmp) / "final"
        raw_dir.mkdir()
        final_dir.mkdir()
        lightnesses = []
        for i, f in enumerate(frames):
            frame = warp(load_image(directory / f.name), f.ellipse, extent, cfg.size)
            lightnesses.append(lightness(frame))
            cv2.imwrite(str(raw_dir / f"{i:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if (i + 1) % 20 == 0:
                log(f"  aligned {i + 1}/{len(frames)}")
        target = float(np.median(lightnesses)) if lightnesses else None
        for i in range(len(frames)):
            src = raw_dir / f"{i:05d}.png"
            if cfg.grade:
                rgb = load_image(src)
                bgr = cv2.cvtColor(grade(rgb, target), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(final_dir / src.name), bgr)
            else:
                shutil.copy(src, final_dir / src.name)
        if frames_out is not None:
            frames_out.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(frames):
                dst = frames_out / f"{i:05d}_{Path(f.name).stem}.png"
                shutil.copy(final_dir / f"{i:05d}.png", dst)
        outputs = []
        for fmt in formats:
            outputs.append(encode(final_dir, out, fmt, cfg))
            log(f"  wrote {outputs[-1]}")
    return RenderResult(outputs, [f.name for f in frames], dropped, extent)
