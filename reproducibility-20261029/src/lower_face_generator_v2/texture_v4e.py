from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .texture_v4 import LowerFaceTextureV4, TextureBlockV4


@dataclass(frozen=True)
class LowerFaceIdentityDetailV4EConfig:
    base_channels: int = 24
    residual_scale: float = 0.12


class LowerFaceIdentityDetailV4E(nn.Module):
    """Context/reference-aware bounded residual over an accepted V4B decoder."""

    architecture_version = "lower_face_identity_detail_v4e"

    def __init__(
        self,
        texture: LowerFaceTextureV4,
        config: LowerFaceIdentityDetailV4EConfig | None = None,
    ) -> None:
        super().__init__()
        self.texture = texture.eval().requires_grad_(False)
        self.config = config or LowerFaceIdentityDetailV4EConfig()
        base = self.config.base_channels
        # masked source + completion + aligned reference + direct branch +
        # lip/motion/edit/confidence maps + x/y coordinates.
        self.enc0 = TextureBlockV4(18, base)
        self.enc1 = TextureBlockV4(base, base * 2, 2)
        self.enc2 = TextureBlockV4(base * 2, base * 4, 2)
        self.bottleneck = TextureBlockV4(base * 4, base * 8, 2)
        self.dec2 = TextureBlockV4(base * 12, base * 4)
        self.dec1 = TextureBlockV4(base * 6, base * 2)
        self.dec0 = TextureBlockV4(base * 3, base)
        self.output_head = nn.Conv2d(base, 4, 3, 1, 1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.texture.eval()
        return self

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        texture_trainable = any(parameter.requires_grad for parameter in self.texture.parameters())
        with torch.enable_grad() if texture_trainable else torch.no_grad():
            base_prediction = self.texture(**inputs)
        source = inputs["source_rgb"]
        structure = base_prediction["structure"]
        maps = torch.cat(
            [structure[name].detach() for name in ("lip_oral", "motion", "edit", "confidence")],
            dim=1,
        ).to(source.dtype)
        height, width = source.shape[-2:]
        y = torch.linspace(-1, 1, height, device=source.device, dtype=source.dtype)
        x = torch.linspace(-1, 1, width, device=source.device, dtype=source.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([xx, yy])[None].expand(len(source), -1, -1, -1)
        value = torch.cat(
            [
                inputs["source_masked"],
                base_prediction["completion"] if texture_trainable else base_prediction["completion"].detach(),
                base_prediction["aligned_reference_rgb"] if texture_trainable else base_prediction["aligned_reference_rgb"].detach(),
                base_prediction["direct_completion"] if texture_trainable else base_prediction["direct_completion"].detach(),
                maps,
                coordinates,
            ],
            dim=1,
        )
        e0 = self.enc0(value)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        hidden = self.bottleneck(e2)
        hidden = F.interpolate(hidden, e2.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec2(torch.cat([hidden, e2], dim=1))
        hidden = F.interpolate(hidden, e1.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec1(torch.cat([hidden, e1], dim=1))
        hidden = F.interpolate(hidden, e0.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec0(torch.cat([hidden, e0], dim=1))
        residual_logits, gate_logits = self.output_head(hidden).split([3, 1], dim=1)
        residual = torch.tanh(residual_logits) * self.config.residual_scale
        gate = torch.sigmoid(gate_logits) * structure["edit"].detach().to(source.dtype)
        base_completion = (
            base_prediction["completion"]
            if texture_trainable
            else base_prediction["completion"].detach()
        )
        completion = (base_completion + residual * gate).clamp(0, 1)
        alpha = structure["edit"].detach().to(source.dtype)
        output = source * (1.0 - alpha) + completion * alpha
        return {
            **base_prediction,
            "base_output": base_prediction["output"].detach(),
            "base_completion": base_prediction["completion"].detach(),
            "completion": completion,
            "output": output,
            "detail_residual": residual,
            "detail_gate": gate,
        }


def _laplacian(value: torch.Tensor) -> torch.Tensor:
    channels = value.shape[1]
    kernel = value.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel = kernel.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    return F.conv2d(value, kernel, padding=1, groups=channels)


def lower_face_identity_detail_v4e_loss(
    prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output, target = prediction["output"], batch["target_rgb"]
    edit = batch["edit_mask"]
    lip = batch["lip_oral_mask"]
    motion = batch["motion_mask"]
    identity_skin = edit * (1.0 - lip)
    weight = edit * (1.0 + 3.0 * lip + 1.0 * motion)
    rgb = (torch.sqrt((output - target).square() + 1e-6) * weight).sum()
    rgb = rgb / (weight.sum() * 3).clamp_min(1.0)
    skin = ((output - target).abs() * identity_skin).sum()
    skin = skin / (identity_skin.sum() * 3).clamp_min(1.0)
    detail = output.new_zeros(())
    current_output, current_target, current_mask = output, target, edit
    for level_weight in (1.0, 0.6, 0.3):
        detail = detail + level_weight * (
            (_laplacian(current_output) - _laplacian(current_target)).abs() * current_mask
        ).sum() / (current_mask.sum() * 3).clamp_min(1.0)
        current_output = F.avg_pool2d(current_output, 2)
        current_target = F.avg_pool2d(current_target, 2)
        current_mask = F.avg_pool2d(current_mask, 2)
    exterior = ((output - batch["source_rgb"]).abs() * (1.0 - edit)).mean()
    residual = (prediction["detail_residual"].abs() * prediction["detail_gate"]).sum()
    residual = residual / (prediction["detail_gate"].sum() * 3).clamp_min(1.0)
    total = rgb + 0.35 * detail + 0.30 * skin + 10.0 * exterior + 0.02 * residual
    parts = {
        "total": total.detach(), "rgb": rgb.detach(), "detail": detail.detach(),
        "identity_skin": skin.detach(), "exterior": exterior.detach(),
        "residual": residual.detach(),
    }
    return total, parts


__all__ = [
    "LowerFaceIdentityDetailV4E",
    "LowerFaceIdentityDetailV4EConfig",
    "lower_face_identity_detail_v4e_loss",
]
