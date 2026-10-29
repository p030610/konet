from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .texture_v4 import TextureBlockV4


@dataclass(frozen=True)
class LowerFaceTemporalAuthorityV4KConfig:
    audio_code_dim: int = 40
    audio_map_channels: int = 16
    base_channels: int = 16
    residual_scale: float = 0.05


class TemporalBlockV4K(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        groups = min(16, channels)
        while channels % groups:
            groups -= 1
        self.body = nn.Sequential(
            nn.Conv3d(
                channels, channels, 3,
                padding=(dilation, 1, 1), dilation=(dilation, 1, 1),
            ),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.body(value), inplace=True)


class LowerFaceTemporalAuthorityV4K(nn.Module):
    """Frozen R7C renderer plus an explicit 16-frame spatiotemporal adapter."""

    architecture_version = "lower_face_temporal_authority_v4k"

    def __init__(
        self,
        base_model: nn.Module,
        config: LowerFaceTemporalAuthorityV4KConfig | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model.eval().requires_grad_(False)
        self.config = config or LowerFaceTemporalAuthorityV4KConfig()
        base = int(self.config.base_channels)
        self.audio_projection = nn.Sequential(
            nn.Linear(self.config.audio_code_dim, self.config.audio_map_channels),
            nn.LayerNorm(self.config.audio_map_channels),
            nn.SiLU(inplace=True),
        )
        # base output, source RGB, base audio residual, edit/lip/motion masks,
        # and a spatial broadcast of the per-frame R7C audio code.
        input_channels = 3 + 3 + 3 + 3 + int(self.config.audio_map_channels)
        self.enc0 = TextureBlockV4(input_channels, base, 2)
        self.enc1 = TextureBlockV4(base, base * 2, 2)
        self.temporal = nn.Sequential(
            TemporalBlockV4K(base * 2, 1),
            TemporalBlockV4K(base * 2, 2),
            TemporalBlockV4K(base * 2, 4),
        )
        self.dec0 = TextureBlockV4(base * 3, base)
        self.dec_full = TextureBlockV4(base + input_channels, base)
        self.head = nn.Conv2d(base, 4, 3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Single-frame path remains exactly the accepted frozen R7C model."""
        with torch.no_grad():
            return self.base_model(**inputs)

    def refine_sequence(
        self,
        baseline: dict[str, torch.Tensor],
        source_rgb: torch.Tensor,
        edit_mask: torch.Tensor,
        lip_mask: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if source_rgb.ndim != 5:
            raise ValueError("source_rgb must have shape [B,T,3,H,W]")
        batch, frames, _, height, width = source_rgb.shape
        required = ("output", "audio_residual", "audio_code")
        if any(name not in baseline for name in required):
            raise ValueError(f"baseline must contain {required}")
        audio_code = baseline["audio_code"]
        if audio_code.shape[:2] != (batch, frames):
            raise ValueError("baseline audio_code must have shape [B,T,C]")
        audio_map = self.audio_projection(audio_code.float()).to(source_rgb.dtype)
        audio_map = audio_map[..., None, None].expand(-1, -1, -1, height, width)
        value = torch.cat(
            [
                baseline["output"].detach(), source_rgb,
                baseline["audio_residual"].detach(),
                edit_mask, lip_mask, motion_mask, audio_map,
            ],
            dim=2,
        )
        flat = value.flatten(0, 1)
        e0 = self.enc0(flat)
        e1 = self.enc1(e0)
        temporal = e1.reshape(batch, frames, *e1.shape[1:]).permute(0, 2, 1, 3, 4)
        temporal = self.temporal(temporal)
        temporal = temporal.permute(0, 2, 1, 3, 4).flatten(0, 1)
        hidden = F.interpolate(temporal, e0.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.dec0(torch.cat([hidden, e0], dim=1))
        hidden = F.interpolate(hidden, (height, width), mode="bilinear", align_corners=False)
        hidden = self.dec_full(torch.cat([hidden, flat], dim=1))
        residual_logits, gate_logits = self.head(hidden).split([3, 1], dim=1)
        residual = torch.tanh(residual_logits) * float(self.config.residual_scale)
        gate = torch.sigmoid(gate_logits) * edit_mask.flatten(0, 1)
        refined = (
            baseline["output"].detach().flatten(0, 1) + residual * gate
        ).clamp(0, 1)
        return {
            **baseline,
            "base_sequence_output": baseline["output"].detach(),
            "output": refined.reshape(batch, frames, 3, height, width),
            "temporal_residual": residual.reshape(batch, frames, 3, height, width),
            "temporal_gate": gate.reshape(batch, frames, 1, height, width),
        }


__all__ = ["LowerFaceTemporalAuthorityV4K", "LowerFaceTemporalAuthorityV4KConfig"]
