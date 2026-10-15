"""EllipseNet: pretrained timm backbone with a mask head and a parameter head.

- Mask head: a light FPN decoder predicting the filled prominent ellipse at stride 4. At
  inference an ellipse is fitted to the mask contour.
- Parameter head: regresses the rotation-safe ellipse encoding of :func:`nn.data.encode`
  directly from the stride-32 feature map (flattened, so it keeps positional information).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    backbone: str = "resnet34"
    size: int = 384
    fpn_channels: int = 128


class EllipseNet(nn.Module):
    def __init__(self, cfg: ModelConfig, pretrained: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4)
        )
        chans: list[int] = self.backbone.feature_info.channels()  # type: ignore[union-attr,operator]
        c = cfg.fpn_channels
        self.lateral = nn.ModuleList(nn.Conv2d(ch, c, 1) for ch in chans)
        self.smooth = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 1),
        )
        n = cfg.size // 32
        self.param_head = nn.Sequential(
            nn.Conv2d(chans[-1], 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * n * n, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 5),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask logits (B, 1, H/4, W/4) and encoded ellipse parameters (B, 5)."""
        feats = self.backbone(x)
        y = self.lateral[-1](feats[-1])
        for lat, f in zip(reversed(self.lateral[:-1]), reversed(feats[:-1]), strict=True):
            y = lat(f) + F.interpolate(y, size=f.shape[-2:], mode="nearest")
        return self.smooth(y), self.param_head(feats[-1])


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path: Path, model: EllipseNet, meta: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": asdict(model.cfg), "state_dict": model.state_dict(), "meta": meta or {}}, path
    )


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> EllipseNet:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = EllipseNet(ModelConfig(**ckpt["config"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model
