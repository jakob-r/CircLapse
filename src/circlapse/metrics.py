"""Evaluate predicted prominent ellipses against ground truth labels."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from circlapse.labels import ImageLabel

HIT_IOU = 0.9


@dataclass
class ImageResult:
    name: str
    outcome: str  # "hit" | "miss" | "dropped" | "false_positive" | "true_negative"
    iou: float | None = None
    center_error: float | None = None
    runtime_s: float | None = None


@dataclass
class Summary:
    results: list[ImageResult]

    def count(self, outcome: str) -> int:
        return sum(r.outcome == outcome for r in self.results)

    @property
    def n_positive(self) -> int:
        return sum(r.outcome in ("hit", "miss", "dropped") for r in self.results)

    @property
    def hit_rate(self) -> float:
        return self.count("hit") / self.n_positive if self.n_positive else 0.0

    def _median(self, attr: str) -> float | None:
        values = [getattr(r, attr) for r in self.results if getattr(r, attr) is not None]
        return statistics.median(values) if values else None

    def format(self) -> str:
        def fmt(v: float | None) -> str:
            return "-" if v is None else f"{v:.3f}"

        lines = [
            f"{'image':<40} {'outcome':<15} {'iou':>6} {'c_err':>6} {'time':>6}",
            *(
                f"{r.name[:40]:<40} {r.outcome:<15} {fmt(r.iou):>6} "
                f"{fmt(r.center_error):>6} {fmt(r.runtime_s):>6}"
                for r in self.results
            ),
            "",
            f"images with ellipse: {self.n_positive}  hit (IoU>={HIT_IOU}): {self.count('hit')}"
            f"  miss: {self.count('miss')}  dropped: {self.count('dropped')}",
            f"images without ellipse: false positives {self.count('false_positive')},"
            f" true negatives {self.count('true_negative')}",
            f"hit rate: {self.hit_rate:.1%}  median IoU: {fmt(self._median('iou'))}"
            f"  median center error: {fmt(self._median('center_error'))}"
            f"  median runtime: {fmt(self._median('runtime_s'))}s",
        ]
        return "\n".join(lines)


def evaluate(gt: dict[str, ImageLabel], pred: dict[str, ImageLabel]) -> Summary:
    """Compare prominent ellipses. Images not labeled in ``gt`` are ignored."""
    results = []
    for name, g in sorted(gt.items()):
        if g.status not in ("labeled", "no_ellipse"):
            continue
        p = pred.get(name)
        p_ell = p.prominent_ellipse if p else None
        runtime = p.runtime_s if p else None
        g_ell = g.prominent_ellipse if g.status == "labeled" else None
        if g_ell is None:
            outcome = "false_positive" if p_ell else "true_negative"
            results.append(ImageResult(name, outcome, runtime_s=runtime))
        elif p_ell is None:
            results.append(ImageResult(name, "dropped", runtime_s=runtime))
        else:
            iou = g_ell.iou(p_ell)
            results.append(
                ImageResult(
                    name,
                    "hit" if iou >= HIT_IOU else "miss",
                    iou=iou,
                    center_error=g_ell.center_error(p_ell),
                    runtime_s=runtime,
                )
            )
    return Summary(results)
