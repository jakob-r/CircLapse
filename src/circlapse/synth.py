"""Synthetic training images with one prominent ellipse plus distractors.

Every image contains a prominent ellipse (major axis >= 1/3 of the image side) rendered as a
filled disk, an outline, or 1-4 concentric sister rings of one object (wall, frame, moldings,
glass of a round window), which are annotated as well, like in the real labels. Smaller
distractor ellipses, clutter, a mild perspective warp and photometric noise are added on top.
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
    # real photos are roughly frontal: annotated ellipses reach at most ~0.48, see CONCEPT.md
    max_eccentricity: float = 0.6
    # prominent eccentricity ~ |N(0, scale)|, near-circular like the real labels (median 0.18)
    eccentricity_scale: float = 0.18
    max_distractors: int = 5
    style_weights: tuple[float, float, float] = (0.3, 0.35, 0.35)  # filled, outline, rings
    # max corner jitter relative to image side; small, since it makes circles eccentric
    perspective: float = 0.01
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


def _min_aspect(max_eccentricity: float) -> float:
    """Smallest minor/major axis ratio allowed by ``max_eccentricity``."""
    return float(np.sqrt(1.0 - max_eccentricity**2))


def _prominent_aspect(rng: np.random.Generator, cfg: SynthConfig) -> float:
    """Minor/major axis ratio of the prominent ellipse, skewed towards circles."""
    ecc = min(abs(rng.normal(0.0, cfg.eccentricity_scale)), cfg.max_eccentricity)
    return float(np.sqrt(1.0 - ecc**2))


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
        cx=0,
        cy=0,
        a=a,
        b=a * _prominent_aspect(rng, cfg),
        angle_deg=rng.uniform(0, 180),
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


def _sister_scales(rng: np.random.Generator) -> list[float]:
    """Scales of 1-4 sister ellipses relative to the prominent one.

    Like the annotated photos: often close rings (moldings, rims) within 0.9-1.1x, sometimes
    far ones (wall, glass) up to 0.7x and 1.4x, and sometimes only smaller or only larger ones.
    """
    side = rng.choice(["both", "inner", "outer"], p=[0.5, 0.35, 0.15])
    scales = []
    for _ in range(rng.integers(1, 5)):
        close = rng.random() < 0.5
        inner = side == "inner" or (side == "both" and rng.random() < 0.6)
        if inner:
            scales.append(rng.uniform(0.88, 0.97) if close else rng.uniform(0.68, 0.88))
        else:
            scales.append(rng.uniform(1.03, 1.12) if close else rng.uniform(1.12, 1.42))
    return scales


def _shading(rng: np.random.Generator, size: int, color: Color) -> np.ndarray:
    """Color field of a band in relief: lit from one side, darker on the other."""
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size - 0.5
    ramp = xx * np.cos(angle) + yy * np.sin(angle)
    field = 1 + rng.uniform(0, 0.8) * ramp
    return np.asarray(color, np.float32) * field[..., None]


def _fill(img: np.ndarray, e: Ellipse, field: np.ndarray) -> None:
    """Paint the anti-aliased filled ellipse ``e`` with the color field."""
    m = np.zeros(img.shape[:2], np.float32)
    _draw(m, e, (1.0, 1.0, 1.0))
    img += m[..., None] * (field - img)


def _draw_rings(rng: np.random.Generator, img: np.ndarray, e: Ellipse, bg: Color) -> list[Ellipse]:
    """Concentric rings of one object (a round window: wall, frame, moldings, glass).

    Sister centers drift by up to ~0.08 radii in one direction, growing with the scale
    difference, as from depth parallax. The prominent rim gets a strong edge, the other rims
    random contrast, so sometimes a sister edge is just as strong.
    """
    direction = rng.uniform(0, 2 * np.pi)
    drift = rng.uniform(0, 0.25) * e.mean_radius
    sisters = []
    for f in _sister_scales(rng):
        d = drift * (f - 1)
        sisters.append(
            Ellipse(
                cx=e.cx + d * np.cos(direction),
                cy=e.cy + d * np.sin(direction),
                a=e.a * f,
                b=e.b * f,
                angle_deg=e.angle_deg,
            )
        )
    rings = sorted([e, *sisters], key=lambda r: -r.a)  # paint outside in
    color = bg
    for r in rings:
        strong = r is e or rng.random() < 0.25
        color = _color(rng, near=color, min_dist=90 if strong else rng.uniform(15, 70))
        _fill(img, r, _shading(rng, img.shape[0], color))
    if rng.random() < 0.5:  # reflections / mullions inside the innermost ring
        glass = rings[-1]
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
    return [e, *sisters]


def _draw_prominent(
    rng: np.random.Generator, img: np.ndarray, e: Ellipse, style: str
) -> list[Ellipse]:
    """Draw the prominent ellipse; return all its annotated ellipses, the prominent one first."""
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
        if thickness < 0.04 * e.b:
            return [e]
        # a thick stroke has a second rim, a sister an annotator might pick as well
        t = thickness
        return [e, Ellipse(cx=e.cx, cy=e.cy, a=e.a - t, b=e.b - t, angle_deg=e.angle_deg)]
    return _draw_rings(rng, img, e, bg)  # type: ignore[arg-type]


def _perspective(
    rng: np.random.Generator,
    img: np.ndarray,
    ellipses: list[Ellipse],
    strength: float,
    padding: float,
    max_eccentricity: float,
    attempts: int = 20,
) -> tuple[np.ndarray, list[Ellipse]]:
    """Random homography that keeps the prominent ellipse (index 0) inside the padding and
    every ellipse within ``max_eccentricity``."""
    size = img.shape[0]
    src = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    for _ in range(attempts):
        dst = src + rng.uniform(-strength, strength, (4, 2)).astype(np.float32) * size
        h = cv2.getPerspectiveTransform(src, dst)
        out = []
        for e in ellipses:
            pts = cv2.perspectiveTransform(e.sample_points(64)[None].astype(np.float32), h)[0]
            out.append(Ellipse.from_points(pts))  # projective image of an ellipse is an ellipse
        if _inside(out[0], size, padding) and all(e.eccentricity <= max_eccentricity for e in out):
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
        _random_ellipse(rng, size, 0.05, 0.3, _min_aspect(cfg.max_eccentricity))
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
        img, ellipses = _perspective(
            rng, img, ellipses, cfg.perspective, cfg.padding, cfg.max_eccentricity
        )
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
