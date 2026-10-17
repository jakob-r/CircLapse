"""Order photos so that similar ones follow each other.

Every photo is first rectified so that its circle becomes a centered disk (the same warp as the
video frames), which makes the descriptors independent of viewpoint and circle size:

- **color**: mean L*a*b* of four rings — center, middle, rim and the surroundings just outside
  the circle;
- **geometry**: the structure inside the circle, as the gradient magnitude in polar coordinates
  whose Fourier magnitudes along the angle are kept. That ignores rotation and mirroring but keeps
  e.g. the number of spokes and where the rings are.

The order is a short path through the descriptors: a greedy nearest-neighbour chain started at
the most unusual photo, then improved by 2-opt. Neither is exact, both are fast for a few
hundred photos.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from circlapse.ellipse import Ellipse
from circlapse.render import warp

SortBy = Literal["color", "geometry"]

SIZE = 96
# Frame half side in circle radii, so that a ring of surroundings is part of the color
EXTENT = 1.4
RADII, ANGLES, HARMONICS = 24, 64, 12


def _frame(path: Path, e: Ellipse, width: int) -> np.ndarray:
    """The rectified RGB frame of the photo at ``path`` whose full width is ``width`` pixels."""
    # Decoding at a quarter of the size is plenty for a 96 px frame and much faster
    bgr = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if bgr is None:
        raise ValueError(f"cannot read {path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return warp(image, e.scaled(image.shape[1] / width), EXTENT, SIZE)


def _radius_map() -> np.ndarray:
    """Distance of every frame pixel from the center, in circle radii."""
    c = (SIZE - 1) / 2
    y, x = np.mgrid[:SIZE, :SIZE]
    return np.hypot(x - c, y - c) / (SIZE / 2 / EXTENT)


def color_features(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    r = _radius_map().ravel()
    zones = [(0, 0.35), (0.35, 0.7), (0.7, 1.0), (1.05, EXTENT)]
    weights = [1.0, 1.0, 1.0, 0.6]
    out = []
    for (lo, hi), w in zip(zones, weights, strict=True):
        mean = lab[(r >= lo) & (r < hi)].mean(axis=0)
        # Lightness counts half: the same cover in sun and shade is still the same color
        out.append(w * mean * np.array([0.5, 1.0, 1.0]))
    return np.concatenate(out)


def geometry_features(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    c = (SIZE - 1) / 2
    # Rows are angles, columns radii from the center to just beyond the rim
    polar = cv2.warpPolar(
        mag, (RADII, ANGLES), (c, c), SIZE / 2 / EXTENT * 1.05, cv2.WARP_POLAR_LINEAR
    )
    spectrum = np.abs(np.fft.rfft(polar, axis=0))[: HARMONICS + 1]
    # Compress the dynamic range so that a few strong edges do not dominate
    f = np.sqrt(spectrum).ravel()
    norm = np.linalg.norm(f)
    return f / norm if norm > 0 else f


FEATURES: dict[SortBy, Callable[[np.ndarray], np.ndarray]] = {
    "color": color_features,
    "geometry": geometry_features,
}


def features(path: Path, e: Ellipse, width: int) -> dict[SortBy, np.ndarray]:
    frame = _frame(path, e, width)
    return {by: fn(frame) for by, fn in FEATURES.items()}


def similarity_path(x: np.ndarray) -> list[int]:
    """Indices of the rows of ``x`` in an order in which neighbours are close."""
    n = len(x)
    if n <= 2:
        return list(range(n))
    # Plain Euclidean distances: each descriptor is built so that its dimensions are comparable
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))

    # Start at the photo farthest from all others, so the chain does not begin in the middle
    path = [int(d.sum(1).argmax())]
    free = np.ones(n, dtype=bool)
    free[path[0]] = False
    for _ in range(n - 1):
        row = np.where(free, d[path[-1]], np.inf)
        nxt = int(row.argmin())
        path.append(nxt)
        free[nxt] = False

    # 2-opt on an open path: reverse path[i:j+1] when that shortens it
    p = np.array(path)
    for _ in range(50):
        improved = False
        for i in range(1, n - 1):
            a, b = p[i - 1], p[i]
            cs, ds = p[i + 1 :], np.append(p[i + 2 :], -1)
            old = d[a, b] + np.where(ds >= 0, d[cs, ds], 0)
            new = d[a, cs] + np.where(ds >= 0, d[b, ds], 0)
            gain = old - new
            j = int(gain.argmax())
            if gain[j] > 1e-9:
                p[i : i + 2 + j] = p[i : i + 2 + j][::-1]
                improved = True
        if not improved:
            break
    return [int(i) for i in p]
