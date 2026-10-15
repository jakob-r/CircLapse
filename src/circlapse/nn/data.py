"""Training samples: letterboxing, geometric and photometric augmentation, targets.

Every image is mapped onto a square ``size`` x ``size`` canvas by a homography (letterbox plus
random scale, rotation, flip, translation and perspective). The same homography is applied to
the labeled ellipses by transforming rim points and refitting, so labels stay consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from circlapse.ellipse import Ellipse
from circlapse.labels import Dataset, ImageLabel, load_image

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
MASK_STRIDE = 4
CACHE_SIDE = 1024  # real photos are downscaled to this longer side once


# ---------------------------------------------------------------------------- geometry


def letterbox_matrix(h: int, w: int, size: int) -> np.ndarray:
    """Homography that fits an (h, w) image centered into a size x size canvas."""
    s = size / max(h, w)
    return np.array(
        [[s, 0, (size - s * w) / 2], [0, s, (size - s * h) / 2], [0, 0, 1]], dtype=np.float64
    )


def transform_ellipse(e: Ellipse, h: np.ndarray) -> Ellipse:
    pts = cv2.perspectiveTransform(e.sample_points(64)[None].astype(np.float64), h)[0]
    return Ellipse.from_points(pts)


def encode(e: Ellipse, size: int) -> np.ndarray:
    """Rotation-safe parameters: center, mean axis and the axis difference as a 2θ vector.

    ``(cx, cy, (a+b)/2, (a-b)/2·cos 2θ, (a-b)/2·sin 2θ) / size`` is continuous everywhere,
    including near-circles where θ is undefined.
    """
    th = 2 * math.radians(e.angle_deg)
    d = (e.a - e.b) / 2
    return np.array(
        [e.cx, e.cy, (e.a + e.b) / 2, d * math.cos(th), d * math.sin(th)], np.float32
    ) / np.float32(size)


def decode(p: np.ndarray, size: int) -> Ellipse:
    cx, cy, r, dc, ds = (float(v) * size for v in p)
    d = math.hypot(dc, ds)
    r = max(r, 1.0)
    d = min(d, r - 0.5)
    return Ellipse(cx=cx, cy=cy, a=r + d, b=r - d, angle_deg=math.degrees(math.atan2(ds, dc)) / 2)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """RGB uint8 (h, w, 3) -> normalized float tensor (3, h, w)."""
    return torch.from_numpy(((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Canvas image and the homography from original pixels to canvas pixels."""
    h = letterbox_matrix(img.shape[0], img.shape[1], size)
    out = cv2.warpAffine(img, h[:2], (size, size), flags=cv2.INTER_AREA)
    return out, h


# ---------------------------------------------------------------------------- augmentation


@dataclass
class Augment:
    scale: tuple[float, float] = (0.75, 1.25)  # zoom relative to the letterboxed image
    rotate: float = 180.0  # max degrees
    shift: float = 0.15  # max translation relative to canvas
    perspective: float = 0.06  # max corner jitter relative to canvas
    flip: bool = True
    photometric: bool = True

    @classmethod
    def none(cls) -> Augment:
        return cls((1.0, 1.0), 0.0, 0.0, 0.0, False, False)


def _random_homography(rng: np.random.Generator, size: int, aug: Augment) -> np.ndarray:
    c = size / 2
    to_center = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]], dtype=np.float64)
    s = rng.uniform(*aug.scale)
    th = math.radians(rng.uniform(-aug.rotate, aug.rotate))
    fx = -1.0 if aug.flip and rng.random() < 0.5 else 1.0
    lin = s * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    lin = lin @ np.diag([fx, 1.0])
    t = rng.uniform(-aug.shift, aug.shift, 2) * size + c
    affine = np.eye(3)
    affine[:2, :2], affine[:2, 2] = lin, t
    h = affine @ to_center
    if aug.perspective > 0:
        src = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
        dst = src + rng.uniform(-aug.perspective, aug.perspective, (4, 2)).astype(np.float32) * (
            size
        )
        h = cv2.getPerspectiveTransform(src, dst).astype(np.float64) @ h
    return h


def _visible(e: Ellipse, size: int, margin: float = 0.02) -> bool:
    x0, y0, x1, y1 = e.bbox()
    lo, hi = -margin * size, (1 + margin) * size
    return x0 >= lo and y0 >= lo and x1 <= hi and y1 <= hi and e.b > 0.05 * size


def _photometric(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)
    if rng.random() < 0.8:  # brightness / contrast / per-channel gain
        out = (out - 128) * rng.uniform(0.6, 1.4) + 128 + rng.uniform(-40, 40)
        out *= rng.uniform(0.85, 1.15, 3).astype(np.float32)
    if rng.random() < 0.5:  # saturation
        gray = out.mean(axis=2, keepdims=True)
        out = gray + (out - gray) * rng.uniform(0.0, 1.6)
    if rng.random() < 0.3:
        out = np.asarray(cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.5)), np.float32)
    if rng.random() < 0.3:
        out = out + rng.normal(0, rng.uniform(2, 12), out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def augment(
    rng: np.random.Generator,
    img: np.ndarray,
    ellipses: list[Ellipse],
    size: int,
    aug: Augment,
    attempts: int = 20,
) -> tuple[np.ndarray, list[Ellipse]]:
    """Warp ``img`` onto the canvas; ``ellipses[0]`` (the target) stays inside it."""
    base = letterbox_matrix(img.shape[0], img.shape[1], size)
    h, out = base, [transform_ellipse(e, base) for e in ellipses]
    for _ in range(attempts):
        cand_h = _random_homography(rng, size, aug) @ base
        cand = [transform_ellipse(e, cand_h) for e in ellipses]
        if _visible(cand[0], size):
            h, out = cand_h, cand
            break
    warped = cv2.warpPerspective(img, h, (size, size), flags=cv2.INTER_LINEAR)
    if aug.photometric:
        warped = _photometric(rng, warped)
    return warped, out


def target_mask(e: Ellipse, size: int) -> np.ndarray:
    """Soft filled-ellipse mask at 1/MASK_STRIDE resolution."""
    m = np.zeros((size, size), np.float32)
    shift = 4  # sub-pixel precision for cv2.ellipse
    f = 1 << shift
    cv2.ellipse(
        m,
        (round(e.cx * f), round(e.cy * f)),
        (round(e.a * f), round(e.b * f)),
        e.angle_deg,
        0,
        360,
        1.0,
        -1,
        cv2.LINE_AA,
        shift,
    )
    n = size // MASK_STRIDE
    return cv2.resize(m, (n, n), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------- datasets


@dataclass
class Item:
    image: np.ndarray  # RGB uint8
    ellipses: list[Ellipse]  # target first


def load_items(directory: Path, names: list[str] | None = None) -> list[Item]:
    """Labeled images of a dataset, downscaled to CACHE_SIDE, target ellipse first."""
    ds = Dataset(directory)
    items = []
    for name, label in sorted(ds.labels.items()):
        if names is not None and name not in names:
            continue
        target = label.prominent_ellipse
        if label.status != "labeled" or target is None:
            continue
        img = load_image(ds.image_path(name))
        s = min(1.0, CACHE_SIDE / max(img.shape[:2]))
        if s < 1:
            img = cv2.resize(
                img,
                (round(img.shape[1] * s), round(img.shape[0] * s)),
                interpolation=cv2.INTER_AREA,
            )
        others = [e for e in label.ellipses if e is not target]
        items.append(
            Item(
                img,
                [Ellipse(**e.model_dump(exclude={"score"})).scaled(s) for e in [target, *others]],
            )
        )
    return items


def labeled_names(directory: Path) -> list[str]:
    ds = Dataset(directory)
    return sorted(
        n for n, lab in ds.labels.items() if lab.status == "labeled" and lab.prominent_ellipse
    )


def make_sample(
    rng: np.random.Generator, img: np.ndarray, ellipses: list[Ellipse], size: int, aug: Augment
) -> dict[str, torch.Tensor]:
    canvas, out = augment(rng, img, ellipses[:1], size, aug)
    target = out[0]
    return {
        "image": to_tensor(canvas),
        "mask": torch.from_numpy(target_mask(target, size))[None],
        "params": torch.from_numpy(encode(target, size)),
    }


class SynthDataset(TorchDataset[dict[str, torch.Tensor]]):
    """Synthetic images on disk (``circlapse synth``); target = labeled prominent ellipse."""

    def __init__(self, directory: Path, size: int, aug: Augment, seed: int = 0) -> None:
        self.dir = Path(directory)
        ds = Dataset(self.dir)
        self.entries: list[tuple[str, Ellipse]] = []
        for name, label in sorted(ds.labels.items()):
            e = label.prominent_ellipse
            if e is not None:
                self.entries.append((name, Ellipse(**e.model_dump(exclude={"score"}))))
        self.size, self.aug, self.seed = size, aug, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        name, e = self.entries[i]
        rng = np.random.default_rng([self.seed, self.epoch, i])
        bgr = cv2.imread(str(self.dir / name))
        if bgr is None:
            raise FileNotFoundError(self.dir / name)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return make_sample(rng, img, [e], self.size, self.aug)


class ItemDataset(TorchDataset[dict[str, torch.Tensor]]):
    """In-memory labeled photos, repeated ``repeats`` times per epoch, freshly augmented."""

    def __init__(
        self, items: list[Item], size: int, aug: Augment, repeats: int = 1, seed: int = 0
    ) -> None:
        self.items, self.size, self.aug, self.repeats, self.seed = items, size, aug, repeats, seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items) * self.repeats

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = self.items[i % len(self.items)]
        rng = np.random.default_rng([self.seed, self.epoch, i, 1])
        return make_sample(rng, item.image, item.ellipses, self.size, self.aug)


def to_label(e: Ellipse, score: float) -> ImageLabel:
    from circlapse.labels import ScoredEllipse

    return ImageLabel(
        ellipses=[ScoredEllipse(**e.model_dump(), score=score)],
        prominent=0,
        status="labeled",
        source="detector",
    )
