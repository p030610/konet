from __future__ import annotations

import torch
import torch.nn as nn

from lower_face_generator.geometry import LOWER_FACE_POINT_COUNT


class ProbeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__()
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class LowerFaceGeometryProbeV2(nn.Module):
    architecture_version = "lower_face_geometry_probe_v2"
    architecture_id = architecture_version

    def __init__(self, base_channels: int = 24) -> None:
        super().__init__()
        base = int(base_channels)
        self.base_channels = base
        self.encoder = nn.Sequential(
            ProbeBlock(3, base, 1),
            ProbeBlock(base, base * 2),
            ProbeBlock(base * 2, base * 4),
            ProbeBlock(base * 4, base * 8),
            ProbeBlock(base * 8, base * 8),
            ProbeBlock(base * 8, base * 8),
            nn.AdaptiveAvgPool2d((4, 5)),
        )
        hidden = base * 8
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden * 20, hidden * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.points_head = nn.Linear(hidden * 4, LOWER_FACE_POINT_COUNT * 2)
        self.teeth_head = nn.Linear(hidden * 4, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.head(self.encoder(image))
        points = torch.sigmoid(self.points_head(hidden)).view(-1, LOWER_FACE_POINT_COUNT, 2)
        teeth = torch.sigmoid(self.teeth_head(hidden))
        return {"points_normalized": points, "teeth_score": teeth}


__all__ = ["LowerFaceGeometryProbeV2"]
