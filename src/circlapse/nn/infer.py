"""Inference: network outputs to ellipses in original image coordinates plus a confidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from circlapse.detect.base import combine_heads
from circlapse.ellipse import Ellipse
from circlapse.nn.data import decode, letterbox, to_tensor, transform_ellipse
from circlapse.nn.model import EllipseNet

Head = Literal["mask", "params", "combined"]


@dataclass
class Prediction:
    mask_ellipse: Ellipse | None  # fitted to the predicted mask, canvas coordinates
    param_ellipse: Ellipse  # regressed directly, canvas coordinates
    fit_iou: float  # how ellipse-shaped the predicted mask is
    agreement: float  # IoU between the two ellipses

    @property
    def confidence(self) -> float:
        return self.fit_iou * self.agreement

    def ellipse(self, head: Head) -> Ellipse | None:
        if head == "combined":
            return combine_heads(self.mask_ellipse, self.param_ellipse)
        return self.mask_ellipse if head == "mask" else self.param_ellipse


def fit_mask(prob: np.ndarray, threshold: float = 0.5) -> tuple[Ellipse | None, float]:
    """Fit an ellipse to the most confident blob of a probability map (canvas resolution)."""
    binary = (prob > threshold).astype(np.uint8)
    n, cc = cv2.connectedComponents(binary)
    if n < 2:
        return None, 0.0
    mass = np.bincount(np.asarray(cc, np.int64).ravel(), weights=prob.ravel(), minlength=n)
    mass[0] = 0
    blob = (cc == int(np.argmax(mass))).astype(np.uint8)
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(contours, key=len).reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return None, 0.0
    e = Ellipse.from_points(pts)
    # contour pixels lie half a pixel inside the true boundary
    e = Ellipse(cx=e.cx, cy=e.cy, a=e.a + 0.5, b=e.b + 0.5, angle_deg=e.angle_deg)
    fitted = e.mask(blob.shape)
    union = np.count_nonzero(fitted | blob.astype(bool))
    return e, float(np.count_nonzero(fitted & blob.astype(bool)) / union) if union else 0.0


@torch.no_grad()
def run_batch(model: EllipseNet, x: torch.Tensor, tta: bool = True) -> list[Prediction]:
    """Predictions for a batch of normalized canvases (B, 3, S, S), canvas coordinates."""
    size = x.shape[-1]
    logits, params = model(x)
    prob = torch.sigmoid(logits)
    if tta:
        logits_f, params_f = model(torch.flip(x, dims=[3]))
        prob = (prob + torch.flip(torch.sigmoid(logits_f), dims=[3])) / 2
        # mirror x: cx -> 1 - cx, angle -> -angle (sin 2θ flips sign)
        flipped = params_f * params_f.new_tensor([-1, 1, 1, 1, -1])
        flipped[:, 0] += 1
        params = (params + flipped) / 2
    prob = F.interpolate(prob, size=(size, size), mode="bilinear", align_corners=False)
    probs = prob[:, 0].float().cpu().numpy()
    params_np = params.float().cpu().numpy()
    out = []
    for p, q in zip(probs, params_np, strict=True):
        pe = decode(q, size)
        me, fit = fit_mask(p)
        out.append(Prediction(me, pe, fit, me.iou(pe) if me is not None else 0.0))
    return out


class Predictor:
    """Run an :class:`EllipseNet` on full-size RGB images."""

    def __init__(self, model: EllipseNet, device: torch.device, tta: bool = True) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.size = model.cfg.size
        self.tta = tta

    def predict(self, image: np.ndarray) -> tuple[Prediction, np.ndarray]:
        """Prediction in canvas coordinates and the canvas-to-image homography."""
        canvas, h = letterbox(image, self.size)
        x = to_tensor(canvas)[None].to(self.device)
        return run_batch(self.model, x, self.tta)[0], np.linalg.inv(h)

    def detect(self, image: np.ndarray, head: Head = "combined") -> tuple[Ellipse | None, float]:
        pred, inv = self.predict(image)
        e = pred.ellipse(head)
        if e is None:
            return None, 0.0
        return transform_ellipse(e, inv), pred.confidence
