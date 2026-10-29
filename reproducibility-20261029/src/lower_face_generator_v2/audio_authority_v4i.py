from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .texture_v4 import TextureBlockV4


@dataclass(frozen=True)
class LowerFaceAudioAuthorityV4IConfig:
    audio_dim: int = 768
    audio_channels: int = 128
    base_channels: int = 20
    residual_scale: float = 0.08


class TemporalAudioEncoderV4I(nn.Module):
    def __init__(self, config: LowerFaceAudioAuthorityV4IConfig) -> None:
        super().__init__()
        channels = int(config.audio_channels)
        self.input = nn.Sequential(
            nn.Linear(config.audio_dim, channels), nn.LayerNorm(channels), nn.SiLU(inplace=True)
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2), nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=4, dilation=4), nn.SiLU(inplace=True),
        )
        self.attention = nn.Linear(channels, 1)

    def forward(self, audio_window: torch.Tensor) -> torch.Tensor:
        if audio_window.ndim != 3 or audio_window.shape[-1] != 768:
            raise ValueError("Expected audio_window [B,T,768]")
        hidden = self.input(audio_window)
        hidden = hidden + self.temporal(hidden.transpose(1, 2)).transpose(1, 2)
        weights = torch.softmax(self.attention(hidden), dim=1)
        return torch.sum(hidden * weights, dim=1)


class LowerFaceAudioAuthorityV4I(nn.Module):
    """Temporal wav2vec residual adapter over the accepted R7B renderer."""

    architecture_version = "lower_face_audio_authority_v4i"

    def __init__(self, base_model: nn.Module, config: LowerFaceAudioAuthorityV4IConfig | None = None) -> None:
        super().__init__()
        self.base_model = base_model.eval().requires_grad_(False)
        self.config = config or LowerFaceAudioAuthorityV4IConfig()
        base = int(self.config.base_channels)
        self.audio_encoder = TemporalAudioEncoderV4I(self.config)
        self.audio_map = nn.Sequential(
            nn.Linear(self.config.audio_channels, base * 2), nn.SiLU(inplace=True)
        )
        # Base completion, aligned reference, lip/motion/edit/confidence, teeth,
        # and spatially broadcast temporal audio code.
        self.enc0 = TextureBlockV4(3 + 3 + 4 + 1 + base * 2, base)
        self.enc1 = TextureBlockV4(base, base * 2, 2)
        self.enc2 = TextureBlockV4(base * 2, base * 4, 2)
        self.bottleneck = TextureBlockV4(base * 4, base * 8, 2)
        self.dec2 = TextureBlockV4(base * 12, base * 4)
        self.dec1 = TextureBlockV4(base * 6, base * 2)
        self.dec0 = TextureBlockV4(base * 3, base)
        self.head = nn.Conv2d(base, 4, 3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, *, audio_window: torch.Tensor | None = None, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            baseline = self.base_model(**inputs)
        source = inputs["source_rgb"]
        if audio_window is None:
            zeros = source.new_zeros(source.shape)
            zero_gate = source.new_zeros(len(source), 1, *source.shape[-2:])
            return {
                **baseline,
                "base_output": baseline["output"].detach(),
                "base_completion": baseline["completion"].detach(),
                "audio_residual": zeros,
                "audio_gate": zero_gate,
                "audio_code": source.new_zeros(len(source), self.config.base_channels * 2),
            }
        audio_code = self.audio_map(self.audio_encoder(audio_window.float())).to(source.dtype)
        height, width = source.shape[-2:]
        audio_map = audio_code[..., None, None].expand(-1, -1, height, width)
        structure = baseline["structure"]
        maps = torch.cat(
            [structure[name].detach() for name in ("lip_oral", "motion", "edit", "confidence")],
            dim=1,
        ).to(source.dtype)
        teeth = structure["geometry_fields"][:, 6:7].detach().to(source.dtype)
        value = torch.cat(
            [baseline["completion"].detach(), baseline["aligned_reference_rgb"].detach(), maps, teeth, audio_map],
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
        residual_logits, gate_logits = self.head(hidden).split([3, 1], dim=1)
        residual = torch.tanh(residual_logits) * float(self.config.residual_scale)
        gate = (
            torch.sigmoid(gate_logits)
            * torch.maximum(structure["lip_oral"], structure["motion"]).detach().to(residual.dtype)
        )
        completion = (baseline["completion"].detach() + residual * gate).clamp(0, 1)
        alpha = structure["edit"].detach().to(source.dtype)
        output = source * (1 - alpha) + completion * alpha
        return {
            **baseline,
            "base_output": baseline["output"].detach(),
            "base_completion": baseline["completion"].detach(),
            "completion": completion,
            "output": output.clamp(0, 1),
            "audio_residual": residual,
            "audio_gate": gate,
            "audio_code": audio_code,
        }


__all__ = [
    "LowerFaceAudioAuthorityV4I", "LowerFaceAudioAuthorityV4IConfig",
    "TemporalAudioEncoderV4I",
]
