from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rasterizer import LowerFaceRasterizerV2


class StructureBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


@dataclass(frozen=True)
class LowerFaceStructureV4Config:
    image_height: int = 256
    image_width: int = 320
    base_channels: int = 32


class LowerFaceStructureV4(nn.Module):
    """Geometry-only lower-face spatial representation predictor."""

    architecture_version = "lower_face_structure_v4"

    def __init__(self, config: LowerFaceStructureV4Config | None = None) -> None:
        super().__init__()
        self.config = config or LowerFaceStructureV4Config()
        cfg = self.config
        self.rasterizer = LowerFaceRasterizerV2((cfg.image_height, cfg.image_width))
        # target/all, lip, jaw, flesh, x, y, and teeth map.
        self.enc0 = StructureBlock(7, cfg.base_channels)
        self.enc1 = StructureBlock(cfg.base_channels, cfg.base_channels * 2, 2)
        self.enc2 = StructureBlock(cfg.base_channels * 2, cfg.base_channels * 4, 2)
        self.enc3 = StructureBlock(cfg.base_channels * 4, cfg.base_channels * 8, 2)
        self.bottleneck = StructureBlock(cfg.base_channels * 8, cfg.base_channels * 8)
        self.dec2 = StructureBlock(cfg.base_channels * 12, cfg.base_channels * 4)
        self.dec1 = StructureBlock(cfg.base_channels * 6, cfg.base_channels * 2)
        self.dec0 = StructureBlock(cfg.base_channels * 3, cfg.base_channels)
        # lip/oral, motion, edit, and structure confidence.
        self.head = nn.Conv2d(cfg.base_channels, 4, 1)

    def geometry_fields(
        self,
        target_points_px: torch.Tensor,
        teeth_score: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = len(target_points_px)
        ones = target_points_px.new_ones(
            batch, 1, self.config.image_height, self.config.image_width
        )
        maps = self.rasterizer(target_points_px, target_points_px, ones)
        y = torch.linspace(-1, 1, self.config.image_height, device=target_points_px.device)
        x = torch.linspace(-1, 1, self.config.image_width, device=target_points_px.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([xx, yy])[None].expand(batch, -1, -1, -1)
        teeth = teeth_score.float().view(batch, 1, 1, 1).expand(
            -1, -1, self.config.image_height, self.config.image_width
        )
        fields = torch.cat(
            [maps["target_heat"], maps["lip_heat"], maps["jaw_heat"], maps["flesh_heat"], coordinates, teeth],
            dim=1,
        )
        return fields, maps

    def forward(
        self,
        *,
        target_points_px: torch.Tensor,
        geometry81: torch.Tensor,
        target_geometry_xyz: torch.Tensor | None = None,
        projection19: torch.Tensor | None = None,
        pose6: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fields, maps = self.geometry_fields(target_points_px.float(), geometry81[:, 80:81])
        e0 = self.enc0(fields)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        value = self.bottleneck(e3)
        value = F.interpolate(value, e2.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec2(torch.cat([value, e2], dim=1))
        value = F.interpolate(value, e1.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec1(torch.cat([value, e1], dim=1))
        value = F.interpolate(value, e0.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec0(torch.cat([value, e0], dim=1))
        logits = self.head(value)
        lip = torch.sigmoid(logits[:, 0:1])
        motion = torch.sigmoid(logits[:, 1:2])
        edit = torch.sigmoid(logits[:, 2:3])
        confidence = torch.sigmoid(logits[:, 3:4])
        return {
            "lip_oral": lip,
            "motion": motion,
            "edit": edit,
            "confidence": confidence,
            "logits": logits,
            "geometry_fields": fields,
            "structure_features": value,
            **maps,
        }


def _dice_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (prediction * target).sum((1, 2, 3))
    denominator = prediction.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1.0 - (2 * intersection + 1) / (denominator + 1)).mean()


def lower_face_structure_v4_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    targets = {
        "lip_oral": batch["lip_oral_mask"],
        "motion": batch["motion_mask"],
        "edit": batch["edit_mask"],
    }
    losses = {}
    channel_lookup = {"lip_oral": 0, "motion": 1, "edit": 2}
    for name, target in targets.items():
        value = prediction[name]
        logit = prediction["logits"][:, channel_lookup[name]:channel_lookup[name] + 1]
        positive = target.mean().clamp_min(1e-4)
        weight = ((1 - positive) / positive).clamp(1.0, 20.0)
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
        weighted = (bce * (1 + target * (weight - 1))).mean()
        losses[name] = weighted + _dice_loss(value, target)
    hierarchy = (
        F.relu(prediction["lip_oral"] - prediction["edit"]).mean()
        + F.relu(prediction["motion"] - prediction["edit"]).mean()
    )
    confidence_target = torch.maximum(targets["lip_oral"], targets["motion"])
    confidence = F.binary_cross_entropy_with_logits(
        prediction["logits"][:, 3:4], confidence_target
    )
    total = losses["lip_oral"] * 2.0 + losses["motion"] + losses["edit"] + 0.2 * hierarchy + 0.1 * confidence
    metrics = {key: value.detach() for key, value in losses.items()}
    metrics.update({"hierarchy": hierarchy.detach(), "confidence": confidence.detach(), "total": total.detach()})
    return total, metrics


__all__ = [
    "LowerFaceStructureV4",
    "LowerFaceStructureV4Config",
    "lower_face_structure_v4_loss",
]
