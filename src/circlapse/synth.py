"""Synthetic training images with one prominent ellipse plus distractors.

Every image contains a prominent ellipse (major axis >= 1/3 of the image side) rendered as a
filled disk, an outline, or concentric rings (wall / frame / glass, as with a round window,
where the outer edge of the frame is the prominent ellipse). Smaller distractor ellipses,
clutter, a mild perspective warp and photometric noise are added on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from circlapse.ellipse import Ellipse
from circlapse.labels import ImageLabel, ScoredEllipse, save_labels

Color = tuple[float, float, float]


@dataclass
class SynthConfig:
    size: int = 512
    min_major: float = 0.35  # relative to image side
    max_major: float = 0.8
    padding: float = 0.1  # the prominent ellipse stays this far (relative) from every border
    min_aspect: float = 0.4
    max_distractors: int = 5
    style_weights: tuple[float, float, float] = (0.3, 0.35, 0.35)  # filled, outline, rings
    perspective: float = 0.08  # max corner jitter relative to image side
    noise: bool = True


# ---------------------------------------------------------------------------- drawing


def _color(rng: np.random.Generator, near: Color | None = None, min_dist: float = 60) -> Color:
    """Random RGB color, at least ``min_dist`` away from ``near`` when given."""
    for _ in range(20):
        c = rng.uniform(0, 255, 3)
        if near is None or np.linalg.norm(c - np.asarray(near)) >= min_dist:
            break
    return (float(c[0]), float(c[1]), float(c[2]))


def _texture(rng: np.random.Generator, size: int, octaves: int = 4) -> np.ndarray:
    """Smooth multi-octave value noise in [0, 1], shape (size, size)."""
    out = np.zeros((size, size), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = 2 ** (o + 2)
        small = rng.random((cells, cells)).astype(np.float32)
        out += amp * cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    return np.clip(out / total, 0, 1)


def _background(rng: np.random.Generator, size: int) -> np.ndarray:
    c0, c1 = np.asarray(_color(rng)), np.asarray(_color(rng))
    kind = rng.choice(["flat", "gradient", "texture"])
    if kind == "flat":
        t = np.zeros((size, size), np.float32)
    elif kind == "gradient":
        angle = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
        t = np.clip(xx * np.cos(angle) + yy * np.sin(angle), 0, 1)
    else:
        t = _texture(rng, size)
    img = c0 + t[..., None] * (c1 - c0)

    # clutter: rectangles, lines and polygons
    for _ in range(rng.integers(0, 8)):
        color = _color(rng)
        shape = rng.choice(["rect", "line", "poly"])
        if shape == "rect":
            p0 = rng.integers(0, size, 2)
            p1 = p0 + rng.integers(-size // 2, size // 2, 2)
            cv2.rectangle(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), color, -1)
        elif shape == "line":
            p0, p1 = rng.integers(-size // 4, size + size // 4, (2, 2))
            thickness = int(rng.integers(1, max(2, size // 40)))
            cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), color, thickness)
        else:
            pts = rng.integers(0, size, (int(rng.integers(3, 7)), 2)).astype(np.int32)
            cv2.fillPoly(img, [pts], color)
    return img.astype(np.float32)


def _draw(img: np.ndarray, e: Ellipse, color: Color, thickness: int = -1) -> None:
    cv2.ellipse(img, e.to_cv_box(), color, thickness, cv2.LINE_AA)


def _random_ellipse(
    rng: np.random.Generator, size: int, min_major: float, max_major: float, min_aspect: float
) -> Ellipse:
    major = rng.uniform(min_major, max_major) * size
    a = major / 2
    b = a * rng.uniform(min_aspect, 1.0)
    # keep the center inside the image; distractors may be partially cut off
    margin = min(a * 0.6, size * 0.45)
    cx = rng.uniform(margin, size - margin)
    cy = rng.uniform(margin, size - margin)
    return Ellipse(cx=cx, cy=cy, a=a, b=b, angle_deg=rng.uniform(0, 180))


def _inside(e: Ellipse, size: int, padding: float) -> bool:
    x0, y0, x1, y1 = e.bbox()
    lo, hi = padding * size, (1 - padding) * size
    return x0 >= lo and y0 >= lo and x1 <= hi and y1 <= hi


def _prominent_ellipse(rng: np.random.Generator, cfg: SynthConfig) -> Ellipse:
    """Random ellipse whose bounding box lies completely inside the padded image."""
    size, pad = cfg.size, cfg.padding * cfg.size
    a = rng.uniform(cfg.min_major, cfg.max_major) * size / 2
    shape = Ellipse(
        cx=0, cy=0, a=a, b=a * rng.uniform(cfg.min_aspect, 1.0), angle_deg=rng.uniform(0, 180)
    )
    x0, y0, x1, y1 = shape.bbox()
    # shrink if the rotated bounding box does not fit into the padded area
    fit = min(1.0, (size - 2 * pad) / max(x1 - x0, y1 - y0))
    hw, hh = (x1 - x0) / 2 * fit, (y1 - y0) / 2 * fit
    return Ellipse(
        cx=rng.uniform(pad + hw, size - pad - hw),
        cy=rng.uniform(pad + hh, size - pad - hh),
        a=shape.a * fit,
        b=shape.b * fit,
        angle_deg=shape.angle_deg,
    )


def _scaled_about_center(e: Ellipse, factor: float) -> Ellipse:
    return Ellipse(cx=e.cx, cy=e.cy, a=e.a * factor, b=e.b * factor, angle_deg=e.angle_deg)


def _draw_prominent(
    rng: np.random.Generator, img: np.ndarray, e: Ellipse, style: str
) -> list[Ellipse]:
    """Draw the prominent ellipse; return all visible ellipses with the prominent one first."""
    bg = tuple(
        float(v)
        for v in img[
            int(np.clip(e.cy, 0, img.shape[0] - 1)), int(np.clip(e.cx, 0, img.shape[1] - 1))
        ]
    )
    color = _color(rng, near=bg)  # type: ignore[arg-type]
    if style == "filled":
        _draw(img, e, color)
        return [e]
    if style == "outline":
        thickness = max(1, int(e.b * rng.uniform(0.01, 0.12)))
        # the drawn stroke is centered on the rim; shift so the outer edge matches the label
        h = thickness / 2
        stroke = Ellipse(cx=e.cx, cy=e.cy, a=e.a - h, b=max(e.b - h, 1.0), angle_deg=e.angle_deg)
        _draw(img, stroke, color, thickness)
        return [e]
    # rings: low-contrast wall, contrasting frame (outer edge = label), glass inside
    wall = _scaled_about_center(e, rng.uniform(1.1, 1.4))
    glass = _scaled_about_center(e, rng.uniform(0.7, 0.92))
    wall_color = tuple(float(v) for v in np.clip(np.asarray(bg) + rng.normal(0, 25, 3), 0, 255))
    _draw(img, wall, wall_color)  # type: ignore[arg-type]
    _draw(img, e, _color(rng, near=wall_color, min_dist=90))  # type: ignore[arg-type]
    _draw(img, glass, _color(rng, near=color))
    if rng.random() < 0.5:  # reflections / mullions inside the glass
        for _ in range(rng.integers(1, 4)):
            p0, p1 = glass.sample_points(32)[rng.choice(32, 2, replace=False)]
            cv2.line(
                img,
                tuple(p0.astype(int)),
                tuple(p1.astype(int)),
                _color(rng),
                max(1, int(glass.b * 0.03)),
                cv2.LINE_AA,
            )
    return [e, wall, glass]


def _perspective(
    rng: np.random.Generator,
    img: np.ndarray,
    ellipses: list[Ellipse],
    strength: float,
    padding: float,
    attempts: int = 20,
) -> tuple[np.ndarray, list[Ellipse]]:
    """Random homography that keeps the prominent ellipse (index 0) inside the padding."""
    size = img.shape[0]
    src = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    for _ in range(attempts):
        dst = src + rng.uniform(-strength, strength, (4, 2)).astype(np.float32) * size
        h = cv2.getPerspectiveTransform(src, dst)
        out = []
        for e in ellipses:
            pts = cv2.perspectiveTransform(e.sample_points(64)[None].astype(np.float32), h)[0]
            out.append(Ellipse.from_points(pts))  # projective image of an ellipse is an ellipse
        if _inside(out[0], size, padding):
            warped = cv2.warpPerspective(img, h, (size, size), borderMode=cv2.BORDER_REFLECT)
            return warped, out
    return img, ellipses


def _photometric(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-30, 30)
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 2.0))
    if rng.random() < 0.7:
        img = img + rng.normal(0, rng.uniform(2, 20), img.shape)
    if rng.random() < 0.2:
        mask = rng.random(img.shape[:2]) < rng.uniform(0.001, 0.02)
        img[mask] = rng.choice([0, 255], size=(int(mask.sum()), 1))
    return img


# ---------------------------------------------------------------------------- public API


def generate(rng: np.random.Generator, cfg: SynthConfig) -> tuple[np.ndarray, ImageLabel]:
    """One synthetic RGB uint8 image and its label (prominent ellipse at index 0)."""
    size = cfg.size
    img = _background(rng, size)

    distractors = [
        _random_ellipse(rng, size, 0.05, 0.3, 0.3)
        for _ in range(rng.integers(0, cfg.max_distractors + 1))
    ]
    below = [d for d in distractors if rng.random() < 0.5]
    above = [d for d in distractors if d not in below]
    for d in below:
        _draw(img, d, _color(rng), -1 if rng.random() < 0.5 else int(rng.integers(1, 6)))

    prominent = _prominent_ellipse(rng, cfg)
    style = rng.choice(
        ["filled", "outline", "rings"], p=np.asarray(cfg.style_weights) / sum(cfg.style_weights)
    )
    ellipses = _draw_prominent(rng, img, prominent, str(style))

    for d in above:
        _draw(img, d, _color(rng), -1 if rng.random() < 0.5 else int(rng.integers(1, 6)))
    ellipses += distractors

    if cfg.perspective > 0:
        img, ellipses = _perspective(rng, img, ellipses, cfg.perspective, cfg.padding)
    if cfg.noise:
        img = _photometric(rng, img)

    label = ImageLabel(
        ellipses=[ScoredEllipse(**e.model_dump()) for e in ellipses],
        prominent=0,
        status="labeled",
        source="synth",
    )
    return np.clip(img, 0, 255).astype(np.uint8), label


def generate_dataset(out: Path, n: int, cfg: SynthConfig, seed: int = 0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    labels: dict[str, ImageLabel] = {}
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        img, label = generate(rng, cfg)
        name = f"synth_{i:06d}.jpg"
        quality = int(rng.integers(40, 96)) if cfg.noise else 95
        cv2.imwrite(
            str(out / name),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        labels[name] = label
    save_labels(out / "labels.json", labels)
