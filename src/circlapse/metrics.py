"""Evaluate predicted prominent ellipses against ground truth labels.

Which ellipse is the "most prominent" is subjective (see CONCEPT.md): objects often consist of
several concentric ellipses and a slightly scaled ellipse rectifies the image just as well. So a
prediction is compared with *every* annotated ellipse of the image:

- ``hit``: IoU >= :data:`HIT_IOU` with any annotated ellipse.
- ``good``: no hit, but center, rectification and scale match some annotated ellipse within the
  tolerances below (scale-tolerant).
- ``miss``: neither.

Strict IoU against the ellipse marked as prominent is kept as a secondary number.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np

from circlapse.ellipse import Ellipse
from circlapse.labels import ImageLabel

HIT_IOU = 0.9
CENTER_TOL = 0.1  # center distance relative to the annotated ellipse's mean radius
RECT_TOL = 0.05  # residual distortion after rectification, see rectification_error
SCALE_TOL = 1.5  # allowed size ratio (either way) between prediction and annotation


def _shape_matrix(e: Ellipse) -> np.ndarray:
    """Symmetric 2x2 matrix R diag(a², b²) Rᵀ, normalized to determinant 1."""
    th = math.radians(e.angle_deg)
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    m = r @ np.diag([e.a**2, e.b**2]) @ r.T
    return m / math.sqrt(max(np.linalg.det(m), 1e-12))


def rectification_error(gt: Ellipse, pred: Ellipse) -> float:
    """``1 - b/a`` of ``gt`` after warping the image so that ``pred`` becomes a circle.

    0 means the predicted rectification turns the annotated ellipse into a perfect circle.
    Independent of position and scale, and well defined for near-circles where the rotation
    angle is meaningless. Combines aspect ratio and rotation error.
    """
    w, v = np.linalg.eigh(_shape_matrix(pred))
    inv_sqrt = v @ np.diag(w**-0.5) @ v.T
    lam = np.linalg.eigvalsh(inv_sqrt @ _shape_matrix(gt) @ inv_sqrt.T)
    return float(1 - math.sqrt(max(lam[0], 0.0) / lam[1]))


def scale_ratio(gt: Ellipse, pred: Ellipse) -> float:
    return pred.mean_radius / gt.mean_radius


def tolerance_score(gt: Ellipse, pred: Ellipse) -> float:
    """Worst tolerance violation of center, rectification and scale; <= 1 means good."""
    return max(
        gt.center_error(pred) / CENTER_TOL,
        rectification_error(gt, pred) / RECT_TOL,
        abs(math.log(scale_ratio(gt, pred))) / math.log(SCALE_TOL),
    )


@dataclass
class ImageResult:
    name: str
    outcome: str  # "hit" | "good" | "miss" | "dropped" | "false_positive" | "true_negative"
    iou: float | None = None  # strict: against the prominent ellipse
    best_iou: float | None = None  # best IoU against any annotated ellipse
    matched: int | None = None  # annotated ellipse with the best tolerance score
    center_error: float | None = None  # these three against the matched ellipse
    rect_error: float | None = None
    scale: float | None = None
    runtime_s: float | None = None

    @property
    def strict_hit(self) -> bool:
        return self.iou is not None and self.iou >= HIT_IOU


@dataclass
class Summary:
    results: list[ImageResult]

    def count(self, outcome: str) -> int:
        return sum(r.outcome == outcome for r in self.results)

    @property
    def n_positive(self) -> int:
        return sum(r.outcome in ("hit", "good", "miss", "dropped") for r in self.results)

    def _rate(self, n: int) -> float:
        return n / self.n_positive if self.n_positive else 0.0

    @property
    def good_rate(self) -> float:
        """Primary metric: share of images whose prediction is a hit or good."""
        return self._rate(self.count("hit") + self.count("good"))

    @property
    def hit_rate(self) -> float:
        return self._rate(self.count("hit"))

    @property
    def strict_hit_rate(self) -> float:
        return self._rate(sum(r.strict_hit for r in self.results))

    def _median(self, attr: str) -> float | None:
        values = [getattr(r, attr) for r in self.results if getattr(r, attr) is not None]
        return statistics.median(values) if values else None

    def format(self, per_image: bool = True) -> str:
        def fmt(v: float | None) -> str:
            return "-" if v is None else f"{v:.3f}"

        header = (
            f"{'image':<40} {'outcome':<15} {'iou':>6} {'iou_any':>7} {'match':>5} "
            f"{'c_err':>6} {'r_err':>6} {'scale':>6} {'time':>6}"
        )
        rows = [
            f"{r.name[:40]:<40} {r.outcome:<15} {fmt(r.iou):>6} {fmt(r.best_iou):>7} "
            f"{'-' if r.matched is None else r.matched:>5} {fmt(r.center_error):>6} "
            f"{fmt(r.rect_error):>6} {fmt(r.scale):>6} {fmt(r.runtime_s):>6}"
            for r in self.results
        ]
        lines = [header, *rows, ""] if per_image else []
        lines += [
            f"images with ellipse: {self.n_positive}  hit (IoU>={HIT_IOU} any): {self.count('hit')}"
            f"  good (c_err<={CENTER_TOL}, r_err<={RECT_TOL}, scale within {SCALE_TOL}x):"
            f" {self.count('good')}  miss: {self.count('miss')}"
            f"  dropped: {self.count('dropped')}",
            f"images without ellipse: false positives {self.count('false_positive')},"
            f" true negatives {self.count('true_negative')}",
            f"good rate (hit+good): {self.good_rate:.1%}  hit rate (any): {self.hit_rate:.1%}"
            f"  strict hit rate (prominent): {self.strict_hit_rate:.1%}",
            f"median IoU (prominent): {fmt(self._median('iou'))}"
            f"  median center error: {fmt(self._median('center_error'))}"
            f"  median rect error: {fmt(self._median('rect_error'))}"
            f"  median runtime: {fmt(self._median('runtime_s'))}s",
        ]
        return "\n".join(lines)


def compare(name: str, gt: ImageLabel, pred: Ellipse, runtime_s: float | None) -> ImageResult:
    """Compare one predicted ellipse with all annotated ellipses of a labeled image."""
    ious = [g.iou(pred) for g in gt.ellipses]
    scores = [tolerance_score(g, pred) for g in gt.ellipses]
    m = int(np.argmin(scores))
    g = gt.ellipses[m]
    prominent = gt.prominent_ellipse
    if max(ious) >= HIT_IOU:
        outcome = "hit"
    elif scores[m] <= 1:
        outcome = "good"
    else:
        outcome = "miss"
    return ImageResult(
        name,
        outcome,
        iou=prominent.iou(pred) if prominent is not None else None,
        best_iou=max(ious),
        matched=m,
        center_error=g.center_error(pred),
        rect_error=rectification_error(g, pred),
        scale=scale_ratio(g, pred),
        runtime_s=runtime_s,
    )


def evaluate(gt: dict[str, ImageLabel], pred: dict[str, ImageLabel]) -> Summary:
    """Compare predicted prominent ellipses. Images not labeled in ``gt`` are ignored."""
    results = []
    for name, g in sorted(gt.items()):
        if g.status not in ("labeled", "no_ellipse"):
            continue
        p = pred.get(name)
        p_ell = p.prominent_ellipse if p else None
        runtime = p.runtime_s if p else None
        if g.status == "no_ellipse" or not g.ellipses:
            outcome = "false_positive" if p_ell else "true_negative"
            results.append(ImageResult(name, outcome, runtime_s=runtime))
        elif p_ell is None:
            results.append(ImageResult(name, "dropped", runtime_s=runtime))
        else:
            results.append(compare(name, g, p_ell, runtime))
    return Summary(results)
