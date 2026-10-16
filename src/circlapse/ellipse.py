"""Ellipse data model and geometry helpers.

Convention: pixel coordinates of the EXIF-corrected image, x to the right, y down.
``a`` is the semi-major axis, ``b`` the semi-minor axis and ``angle_deg`` the angle of the
major axis measured from the x-axis towards the y-axis (clockwise on screen), in [0, 180).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class Ellipse(BaseModel):
    model_config = ConfigDict(frozen=True)

    cx: float
    cy: float
    a: float
    b: float
    angle_deg: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        a, b = abs(float(d["a"])), abs(float(d["b"]))
        angle = float(d.get("angle_deg", 0.0))
        if b > a:
            a, b = b, a
            angle += 90.0
        d["a"], d["b"], d["angle_deg"] = a, b, angle % 180.0
        return d

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_points(cls, points: np.ndarray | Sequence[Sequence[float]]) -> Ellipse:
        """Least-squares fit through at least 5 rim points."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 5:
            raise ValueError(f"need at least 5 points to fit an ellipse, got {len(pts)}")
        (cx, cy), (w, h), angle = cv2.fitEllipse(pts)
        return cls(cx=cx, cy=cy, a=w / 2, b=h / 2, angle_deg=angle)

    @classmethod
    def from_cv_box(cls, box: tuple[tuple[float, float], tuple[float, float], float]) -> Ellipse:
        (cx, cy), (w, h), angle = box
        return cls(cx=cx, cy=cy, a=w / 2, b=h / 2, angle_deg=angle)

    # ------------------------------------------------------------------ conversion

    def to_cv_box(self) -> tuple[tuple[float, float], tuple[float, float], float]:
        return (self.cx, self.cy), (2 * self.a, 2 * self.b), self.angle_deg

    def scaled(self, factor: float) -> Ellipse:
        return Ellipse(
            cx=self.cx * factor,
            cy=self.cy * factor,
            a=self.a * factor,
            b=self.b * factor,
            angle_deg=self.angle_deg,
        )

    # ------------------------------------------------------------------ geometry

    @property
    def mean_radius(self) -> float:
        return math.sqrt(self.a * self.b)

    @property
    def aspect(self) -> float:
        return self.b / self.a if self.a > 0 else 0.0

    @property
    def eccentricity(self) -> float:
        return math.sqrt(max(0.0, 1.0 - self.aspect**2))

    def sample_points(self, n: int = 64) -> np.ndarray:
        """``n`` points on the rim, shape (n, 2)."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        th = math.radians(self.angle_deg)
        c, s = math.cos(th), math.sin(th)
        x = self.a * np.cos(t)
        y = self.b * np.sin(t)
        return np.stack([self.cx + c * x - s * y, self.cy + s * x + c * y], axis=1)

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        th = math.radians(self.angle_deg)
        c, s = math.cos(th), math.sin(th)
        dx, dy = x - self.cx, y - self.cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        return (u / self.a) ** 2 + (v / max(self.b, 1e-9)) ** 2 <= 1.0

    def bbox(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box (x0, y0, x1, y1)."""
        th = math.radians(self.angle_deg)
        hw = math.hypot(self.a * math.cos(th), self.b * math.sin(th))
        hh = math.hypot(self.a * math.sin(th), self.b * math.cos(th))
        return self.cx - hw, self.cy - hh, self.cx + hw, self.cy + hh

    def mask(self, shape: tuple[int, ...]) -> np.ndarray:
        """Boolean filled mask of the given (h, w[, c]) image shape."""
        m = np.zeros(shape[:2], dtype=np.uint8)
        cv2.ellipse(m, self.to_cv_box(), 1, thickness=-1)
        return m.astype(bool)

    def iou(self, other: Ellipse, resolution: int = 512) -> float:
        """Intersection over union, estimated on a grid over the joint bounding box."""
        ax0, ay0, ax1, ay1 = self.bbox()
        bx0, by0, bx1, by1 = other.bbox()
        x0, y0, x1, y1 = min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)
        xs = np.linspace(x0, x1, resolution)
        ys = np.linspace(y0, y1, resolution)
        gx, gy = np.meshgrid(xs, ys)
        ma, mb = self.contains(gx, gy), other.contains(gx, gy)
        union = np.count_nonzero(ma | mb)
        return float(np.count_nonzero(ma & mb) / union) if union else 0.0

    def center_error(self, other: Ellipse) -> float:
        """Distance between centers, relative to this ellipse's mean radius."""
        return math.hypot(self.cx - other.cx, self.cy - other.cy) / self.mean_radius

    def rel_size(self, shape: tuple[int, ...]) -> float:
        """Major axis length relative to the shorter image side."""
        return 2 * self.a / min(shape[:2])
