"""Embed video frames by what they show, for sorting photos by meaning.

DINOv2 ViT-S/14 (self-supervised, about 22M parameters) maps a frame to a vector in which
photos of similar things lie close together, whatever their exact colors or layout. Its weights
are downloaded once into the Hugging Face cache on first use. Needs the ``ml`` extra.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

MODEL = "vit_small_patch14_dinov2.lvd142m"
SIZE = 224
BATCH = 32

_lock = threading.Lock()
_loaded: tuple[Any, Any, np.ndarray, np.ndarray] | None = None


def available() -> bool:
    try:
        import timm  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _model() -> tuple[Any, Any, np.ndarray, np.ndarray]:
    """The model, its device and the input mean and std, loaded on first use."""
    global _loaded
    with _lock:
        if _loaded is None:
            import timm
            from timm.data import resolve_data_config

            from circlapse.nn.model import device

            model = timm.create_model(MODEL, pretrained=True, num_classes=0, img_size=SIZE)
            dev = device()
            model.eval().to(dev)
            cfg = resolve_data_config({}, model=model)
            mean = np.array(cfg["mean"], dtype=np.float32)
            std = np.array(cfg["std"], dtype=np.float32)
            _loaded = (model, dev, mean, std)
        return _loaded


def embed(frames: list[np.ndarray]) -> np.ndarray:
    """One unit-length row per ``SIZE`` px RGB frame, so that Euclidean distances rank like
    cosine distances."""
    if not frames:
        return np.zeros((0, 0), dtype=np.float32)
    import torch

    model, dev, mean, std = _model()
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), BATCH):
            x = (np.stack(frames[i : i + BATCH]).astype(np.float32) / 255 - mean) / std
            t = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(dev)
            out.append(model(t).float().cpu().numpy())
    e = np.concatenate(out)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
