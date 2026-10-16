"""Neural network detector (needs the ``ml`` extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from circlapse.detect.base import DEFAULT_ROUNDNESS
from circlapse.labels import ScoredEllipse
from circlapse.nn.infer import Head, Predictor
from circlapse.nn.model import device, load_checkpoint


class NNDetector:
    """EllipseNet prediction of the prominent ellipse; drops images below ``min_confidence``.

    The ellipse is pulled towards a circle by ``roundness`` (see :meth:`Ellipse.rounded`).
    """

    def __init__(
        self,
        checkpoint: Path,
        head: Head = "mask",
        min_confidence: float = 0.0,
        tta: bool = True,
        roundness: float = DEFAULT_ROUNDNESS,
    ) -> None:
        self.name = f"nn-{checkpoint.stem}-{head}"
        dev = device()
        self.predictor = Predictor(load_checkpoint(checkpoint, dev), dev, tta=tta)
        self.head: Head = head
        self.min_confidence = min_confidence
        self.roundness = roundness

    def detect(self, image: np.ndarray) -> list[ScoredEllipse]:
        e, confidence = self.predictor.detect(image, self.head)
        if e is None or confidence < self.min_confidence:
            return []
        return [ScoredEllipse(**e.rounded(self.roundness).model_dump(), score=confidence)]
