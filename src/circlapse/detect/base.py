"""Detector interface. Concrete detectors implement :class:`Detector`."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np

from circlapse.labels import Dataset, ImageLabel, ScoredEllipse, load_image, save_labels

# Strength of Ellipse.rounded for predictions; chosen on held-out CV predictions, see RESULTS.md
DEFAULT_ROUNDNESS = 0.75


class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[ScoredEllipse]:
        """Detect ellipses in an RGB uint8 image, most prominent first. Empty = drop."""
        ...


def run_detector(detector: Detector, dataset: Dataset, out: Path) -> dict[str, ImageLabel]:
    """Run ``detector`` on every image of ``dataset`` and write predictions to ``out``."""
    predictions: dict[str, ImageLabel] = {}
    for name in dataset.images():
        image = load_image(dataset.image_path(name))
        start = time.perf_counter()
        ellipses = detector.detect(image)
        runtime = time.perf_counter() - start
        predictions[name] = ImageLabel(
            ellipses=ellipses,
            prominent=0 if ellipses else None,
            status="labeled" if ellipses else "no_ellipse",
            source="detector",
            runtime_s=runtime,
        )
    save_labels(out, predictions)
    return predictions
