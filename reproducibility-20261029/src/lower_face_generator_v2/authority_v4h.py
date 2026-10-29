from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .texture_v4 import TextureBlockV4


@dataclass(frozen=True)
class LowerFaceAuthorityV4HConfig:
    base_channels: int = 20
    residual_scale: float = 0.10
    teeth_residual_scale: float = 0.16


class LowerFaceAuthorityV4H(nn.Module):
    """Zero-initialized jaw/teeth authority adapter over an accepted renderer."""

    architecture_version = "lower_face_authority_v4h"

    def __init__(self, base_model: nn.Module, config: LowerFaceAuthorityV4HConfig | None = None) -> None:
        super().__init__()
        self.base_model = base_model.eval().requires_grad_(False)
        self.config = config or LowerFaceAuthorityV4HConfig()
        base = int(self.config.base_channels)
        # Base completion, aligned reference, seven explicit geometry fields,
        # and four accepted structure maps.
        self.enc0 = TextureBlockV4(17, base)
        self.enc1 = TextureBlockV4(base, base * 2, 2)
        self.enc2 = TextureBlockV4(base * 2, base * 4, 2)
        self.bottleneck = TextureBlockV4(base * 4, base * 8, 2)
        self.dec2 = TextureBlockV4(base * 12, base * 4)
        self.dec1 = TextureBlockV4(base * 6, base * 2)
        self.dec0 = TextureBlockV4(base * 3, base)
        self.head = nn.Conv2d(base, 4, 3, 1, 1)
        self.teeth_head = nn.Sequential(
            TextureBlockV4(8, base),
            TextureBlockV4(base, base),
            nn.Conv2d(base, 4, 3, 1, 1),
        )
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.teeth_head[-1].weight)
        nn.init.zeros_(self.teeth_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            baseline = self.base_model(**inputs)
        structure = baseline["structure"]
        structure_maps = torch.cat(
            [structure[name] for name in ("lip_oral", "motion", "edit", "confidence")], dim=1
        )
        value = torch.cat(
            [
                baseline["completion"].detach(),
                baseline["aligned_reference_rgb"].detach(),
                structure["geometry_fields"].detach(),
                structure_maps.detach(),
            ],
            dim=1,
        ).to(inputs["source_rgb"].dtype)
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
        residual_logits, gate_logits = self.head(hidden).split([3, 1], dim=1)
        residual = torch.tanh(residual_logits) * float(self.config.residual_scale)
        gate = torch.sigmoid(gate_logits) * structure["edit"].detach().to(residual.dtype)
        teeth_input = torch.cat(
            [
                baseline["completion"].detach(),
                baseline["aligned_reference_rgb"].detach(),
                structure["lip_oral"].detach(),
                structure["geometry_fields"][:, 6:7].detach(),
            ],
            dim=1,
        ).to(residual.dtype)
        teeth_logits, teeth_gate_logits = self.teeth_head(teeth_input).split([3, 1], dim=1)
        teeth_residual = torch.tanh(teeth_logits) * float(self.config.teeth_residual_scale)
        teeth_gate = (
            torch.sigmoid(teeth_gate_logits)
            * structure["lip_oral"].detach().to(residual.dtype)
            * structure["geometry_fields"][:, 6:7].detach().to(residual.dtype)
        )
        completion = (
            baseline["completion"].detach() + residual * gate + teeth_residual * teeth_gate
        ).clamp(0, 1)
        source = inputs["source_rgb"]
        alpha = structure["edit"].detach().to(source.dtype)
        output = source * (1.0 - alpha) + completion * alpha
        return {
            **baseline,
            "base_output": baseline["output"].detach(),
            "base_completion": baseline["completion"].detach(),
            "completion": completion,
            "output": output.clamp(0, 1),
            "authority_residual": residual,
            "authority_gate": gate,
            "teeth_residual": teeth_residual,
            "teeth_gate": teeth_gate,
        }


__all__ = ["LowerFaceAuthorityV4H", "LowerFaceAuthorityV4HConfig"]
