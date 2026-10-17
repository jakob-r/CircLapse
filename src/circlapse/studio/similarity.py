"""Order photos so that similar ones follow each other.

Every photo is first turned into its video frame: rectified so that its circle becomes a
centered disk and cropped like the video. That makes the descriptors independent of viewpoint
and circle size, and only what the video shows counts:

- **color**: mean L*a*b* of the center, middle and rim of the circle and of the surroundings
  in the frame;
- **geometry**: the structure of the frame, as the gradient magnitude in polar coordinates
  whose Fourier magnitudes along the angle are kept. That ignores rotation and mirroring but keeps
  e.g. the number of spokes and where the rings are. The frame corners are left out.

The order is a short path through the descriptors: a greedy nearest-neighbour chain started at
the most unusual photo, then improved by 2-opt. For looping videos the path can be closed, so
that the last photo is also similar to the first. Neither is exact, both are fast for a few
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
RADII, ANGLES, HARMONICS = 32, 64, 12


def _frame(path: Path, e: Ellipse, width: int, extent: float) -> np.ndarray:
    """The video frame of the photo at ``path``, whose full width is ``width`` pixels, with a
    half side of ``extent`` circle radii."""
    # Decoding at a quarter of the size is plenty for a 96 px frame and much faster
    bgr = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if bgr is None:
        raise ValueError(f"cannot read {path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return warp(image, e.scaled(image.shape[1] / width), extent, SIZE)


def _radius_map(extent: float) -> np.ndarray:
    """Distance of every frame pixel from the center, in circle radii."""
    c = (SIZE - 1) / 2
    y, x = np.mgrid[:SIZE, :SIZE]
    return np.hypot(x - c, y - c) / (SIZE / 2 / extent)


def color_features(frame: np.ndarray, extent: float) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    r = _radius_map(extent).ravel()
    out = []
    # Center, middle and rim of the circle, then everything around it (at least the corners)
    for lo, hi in [(0, 0.35), (0.35, 0.7), (0.7, 1.0), (1.05, np.inf)]:
        mean = lab[(r >= lo) & (r < hi)].mean(axis=0)
        # Lightness counts half: the same cover in sun and shade is still the same color
        out.append(mean * np.array([0.5, 1.0, 1.0]))
    return np.concatenate(out)


def geometry_features(frame: np.ndarray, extent: float) -> np.ndarray:
    del extent  # the polar grid covers the frame, whatever the crop
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    c = (SIZE - 1) / 2
    # Rows are angles, columns radii from the center to the middle of the frame edges
    polar = cv2.warpPolar(mag, (RADII, ANGLES), (c, c), SIZE / 2, cv2.WARP_POLAR_LINEAR)
    spectrum = np.abs(np.fft.rfft(polar, axis=0))[: HARMONICS + 1]
    # Compress the dynamic range so that a few strong edges do not dominate
    f = np.sqrt(spectrum).ravel()
    norm = np.linalg.norm(f)
    return f / norm if norm > 0 else f


FEATURES: dict[SortBy, Callable[[np.ndarray, float], np.ndarray]] = {
    "color": color_features,
    "geometry": geometry_features,
}


def features(path: Path, e: Ellipse, width: int, extent: float) -> dict[SortBy, np.ndarray]:
    """Both descriptors of the video frame with a half side of ``extent`` circle radii."""
    frame = _frame(path, e, width, extent)
    return {by: fn(frame, extent) for by, fn in FEATURES.items()}


def similarity_path(x: np.ndarray, closed: bool = False) -> list[int]:
    """Indices of the rows of ``x`` in an order in which neighbours are close; ``closed``
    also counts the step from the last back to the first."""
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

    # 2-opt: reverse a stretch of the path when that shortens it. The first photo stays put;
    # a closed path also counts the step from its last photo back to it
    p = np.array(path)
    end = p[0] if closed else -1
    for _ in range(50):
        improved = False
        for i in range(1, n - 1):
            a, b = p[i - 1], p[i]
            # Reversing p[i..k] replaces the steps a→b and c→d by a→c and b→d
            cs, ds = p[i + 1 :], np.append(p[i + 2 :], end)
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
