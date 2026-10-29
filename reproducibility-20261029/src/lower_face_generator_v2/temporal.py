from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LowerFaceGeneratorV2


FRAME_MODEL_KEYS = (
    "source_rgb", "source_masked", "references_rgb", "source_points_px",
    "target_points_px", "source_geometry_xyz", "target_geometry_xyz",
    "reference_geometry_xyz", "geometry81", "reference_geometry81",
    "projection19", "pose6", "edit_mask", "inpaint_core_mask", "motion_mask",
    "ring_mask", "nose_exclusion_mask", "alpha_core_mask", "upper_overlap_mask",
    "upper_alpha_target_mask", "lip_oral_mask",
)


class CausalTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=0)
        self.norm = nn.GroupNorm(groups, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.pad(value, (1, 1, 1, 1, 2, 0))
        return F.silu(self.norm(self.conv(value)), inplace=True)


class TemporalResidualAdapter(nn.Module):
    """Causal low-resolution flow/color/alpha correction for frame outputs."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            CausalTemporalBlock(13, channels),
            CausalTemporalBlock(channels, channels),
            CausalTemporalBlock(channels, channels),
        )
        self.head = nn.Conv3d(channels, 6, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(value))


@dataclass(frozen=True)
class TemporalLowerFaceConfig:
    sequence_length: int = 16
    temporal_channels: int = 16
    spatial_downsample: int = 4
    max_flow_px: float = 2.0
    max_rgb_residual: float = 0.08
    max_alpha_residual: float = 0.08
    frame_chunk: int = 2


class TemporalLowerFaceGeneratorV2(nn.Module):
    architecture_version = "lower_face_generator_v2_temporal_v1"

    def __init__(
        self,
        frame_renderer: LowerFaceGeneratorV2,
        config: TemporalLowerFaceConfig | None = None,
        freeze_frame_renderer: bool = True,
    ) -> None:
        super().__init__()
        self.frame_renderer = frame_renderer
        self.config = config or TemporalLowerFaceConfig()
        self.temporal_adapter = TemporalResidualAdapter(self.config.temporal_channels)
        if freeze_frame_renderer:
            self.frame_renderer.requires_grad_(False)
            self.frame_renderer.eval()

    @staticmethod
    def _flatten(value: torch.Tensor) -> torch.Tensor:
        return value.flatten(0, 1)

    @staticmethod
    def _sequence(value: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        return value.view(batch, frames, *value.shape[1:])

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.frame_renderer.parameters()):
            self.frame_renderer.eval()
        return self

    def forward(self, **batch: torch.Tensor) -> dict[str, torch.Tensor]:
        source = batch["source_rgb"]
        if source.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(source.shape)}")
        batch_size, frames, _, height, width = source.shape
        if frames != self.config.sequence_length:
            raise ValueError(f"Expected {self.config.sequence_length} frames, got {frames}")
        chunks: dict[str, list[torch.Tensor]] = {}
        chunk_size = max(1, int(self.config.frame_chunk))
        for start in range(0, frames, chunk_size):
            end = min(start + chunk_size, frames)
            flat_batch = {
                key: self._flatten(batch[key][:, start:end])
                for key in FRAME_MODEL_KEYS
                if key in batch
            }
            result = self.frame_renderer(**flat_batch)
            for key, value in result.items():
                if torch.is_tensor(value) and value.shape[0] == batch_size * (end - start):
                    chunks.setdefault(key, []).append(
                        value.view(batch_size, end - start, *value.shape[1:])
                    )
        sequence = {key: torch.cat(values, dim=1) for key, values in chunks.items()}
        completion = sequence["completion"]
        alpha = sequence["alpha"]
        features = torch.cat(
            [
                completion,
                sequence["output"],
                source,
                alpha,
                batch["edit_mask"],
                batch["ring_mask"],
                batch["lip_oral_mask"],
            ],
            dim=2,
        )
        low_height = max(1, height // self.config.spatial_downsample)
        low_width = max(1, width // self.config.spatial_downsample)
        low = F.interpolate(
            features.flatten(0, 1),
            (low_height, low_width),
            mode="bilinear",
            align_corners=False,
        ).view(batch_size, frames, 13, low_height, low_width).permute(0, 2, 1, 3, 4)
        residual = self.temporal_adapter(low).permute(0, 2, 1, 3, 4)
        residual = F.interpolate(
            residual.flatten(0, 1),
            (height, width),
            mode="bilinear",
            align_corners=False,
        ).view(batch_size, frames, 6, height, width)
        allowed = batch["edit_mask"] * (1.0 - batch["nose_exclusion_mask"])
        temporal_flow = (
            torch.tanh(residual[:, :, :2]) * self.config.max_flow_px * allowed
        )
        warped_completion = self.frame_renderer._warp(
            completion.flatten(0, 1), temporal_flow.flatten(0, 1)
        ).view_as(completion)
        rgb_correction = (
            torch.tanh(residual[:, :, 2:5]) * self.config.max_rgb_residual * allowed
        )
        temporal_completion = (warped_completion + rgb_correction).clamp(0.0, 1.0)
        upper_overlap = batch.get("upper_overlap_mask")
        if upper_overlap is None:
            upper_overlap = torch.zeros_like(batch["ring_mask"])
        alpha_support = (
            batch["ring_mask"]
            * (1.0 - upper_overlap)
            * (1.0 - batch["nose_exclusion_mask"])
        )
        alpha_correction = (
            torch.tanh(residual[:, :, 5:6])
            * self.config.max_alpha_residual
            * alpha_support
        )
        temporal_alpha = (alpha + alpha_correction).clamp(0.0, 1.0)
        temporal_alpha = torch.maximum(temporal_alpha, batch["alpha_core_mask"])
        temporal_alpha = temporal_alpha * (1.0 - batch["nose_exclusion_mask"])
        output = source * (1.0 - temporal_alpha) + temporal_completion * temporal_alpha
        # Preserve the frame renderer's bit-exact exterior contract under AMP.
        output = torch.where(allowed.expand_as(output) > 0, output, source)
        return {
            **sequence,
            "base_output": sequence["output"],
            "base_completion": completion,
            "base_alpha": alpha,
            "output": output.clamp(0.0, 1.0),
            "completion": temporal_completion,
            "alpha": temporal_alpha,
            "temporal_flow": temporal_flow,
            "temporal_rgb_correction": rgb_correction,
            "temporal_alpha_correction": alpha_correction,
        }


__all__ = [
    "FRAME_MODEL_KEYS",
    "TemporalLowerFaceConfig",
    "TemporalLowerFaceGeneratorV2",
    "TemporalResidualAdapter",
]
