from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import LowerFaceGeneratorV2, LowerFaceGeneratorV2Config


def _group_count(channels: int) -> int:
    groups = min(16, channels)
    while channels % groups:
        groups -= 1
    return groups


class GeometryModulatedResidualBlock(nn.Module):
    """Residual block whose spatial affine parameters come from geometry maps."""

    def __init__(self, channels: int, condition_channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.modulation1 = nn.Conv2d(condition_channels, channels * 2, 1)
        self.modulation2 = nn.Conv2d(condition_channels, channels * 2, 1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    @staticmethod
    def _modulate(
        value: torch.Tensor,
        condition: torch.Tensor,
        norm: nn.GroupNorm,
        modulation: nn.Conv2d,
    ) -> torch.Tensor:
        scale, bias = modulation(condition).chunk(2, dim=1)
        return norm(value) * (1.0 + 0.2 * torch.tanh(scale)) + 0.2 * torch.tanh(bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self._modulate(value, condition, self.norm1, self.modulation1)
        residual = self.conv1(F.silu(residual, inplace=True))
        residual = self._modulate(residual, condition, self.norm2, self.modulation2)
        residual = self.conv2(F.silu(residual, inplace=True))
        return value + residual


class GeometryRefinementTower(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        condition_channels: int,
        blocks: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            [GeometryModulatedResidualBlock(channels, condition_channels) for _ in range(blocks)]
        )
        self.head = nn.Conv2d(channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        value = F.silu(self.stem(value), inplace=True)
        for block in self.blocks:
            value = block(value, condition)
        return self.head(value)


@dataclass(frozen=True)
class LowerFaceGeneratorV3Config:
    frame: LowerFaceGeneratorV2Config = field(
        default_factory=lambda: LowerFaceGeneratorV2Config(
            base_channels=16,
            alpha_feather_radius=3,
            boundary_low_frequency_correction=True,
        )
    )
    structure_channels: int = 192
    structure_blocks: int = 6
    appearance_channels: int = 192
    appearance_blocks: int = 6
    max_structure_correction_px: float = 1.5
    dense_geometry_sigma_px: float = 32.0
    max_structure_rgb: float = 0.15
    max_low_frequency_rgb: float = 0.12
    max_detail_rgb: float = 0.10
    detail_mean_kernel: int = 9


class LowerFaceGeneratorV3(nn.Module):
    """Capacity-expanded renderer with an explicit target-geometry authority path."""

    architecture_version = "lower_face_generator_v3"

    def __init__(self, config: LowerFaceGeneratorV3Config | None = None) -> None:
        super().__init__()
        self.config = config or LowerFaceGeneratorV3Config()
        cfg = self.config
        self.frame_renderer = LowerFaceGeneratorV2(cfg.frame)
        condition_channels = self.frame_renderer.rasterizer.condition_channels

        # Four RGB candidates, eight geometry maps, and six contractual masks.
        refinement_inputs = 12 + condition_channels + 6
        self.structure = GeometryRefinementTower(
            refinement_inputs,
            cfg.structure_channels,
            condition_channels,
            cfg.structure_blocks,
            6,  # residual flow, geometry mix, RGB correction
        )
        self.appearance = GeometryRefinementTower(
            refinement_inputs,
            cfg.appearance_channels,
            condition_channels,
            cfg.appearance_blocks,
            7,  # low RGB, detail RGB, detail confidence
        )

    @staticmethod
    def _resize(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _mask_inputs(
        edit: torch.Tensor,
        core: torch.Tensor,
        motion: torch.Tensor,
        ring: torch.Tensor,
        nose: torch.Tensor,
        lip: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([edit, core, motion, ring, nose, lip], dim=1)

    def _dense_geometry_flow(
        self,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate target-to-source landmark displacement into a dense field."""

        height, width = allowed.shape[-2:]
        half_height, half_width = max(height // 2, 1), max(width // 2, 1)
        scale = target_points.new_tensor(
            [half_width / max(width, 1), half_height / max(height, 1)]
        )
        source = source_points.float() * scale
        target = target_points.float() * scale
        y, x = torch.meshgrid(
            torch.arange(half_height, device=target.device, dtype=target.dtype),
            torch.arange(half_width, device=target.device, dtype=target.dtype),
            indexing="ij",
        )
        grid = torch.stack([x, y], dim=-1)[None, None]
        distance2 = (grid - target[:, :, None, None]).square().sum(dim=-1)
        sigma = self.config.dense_geometry_sigma_px * 0.5
        weights = torch.exp(-0.5 * distance2 / max(sigma * sigma, 1e-6))
        # grid_sample uses output-to-input flow, hence source minus target.
        displacement = source - target
        flow = torch.einsum("bnhw,bnc->bchw", weights, displacement)
        flow = flow / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        confidence = weights.max(dim=1, keepdim=True).values
        flow = flow * confidence
        flow = F.interpolate(flow, (height, width), mode="bilinear", align_corners=False)
        flow[:, 0] *= width / max(half_width, 1)
        flow[:, 1] *= height / max(half_height, 1)
        return flow.to(allowed.dtype) * allowed

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.frame_renderer(**inputs)
        source = inputs["source_rgb"]
        edit = inputs["edit_mask"]
        core = inputs["inpaint_core_mask"]
        motion = inputs["motion_mask"]
        ring = inputs["ring_mask"]
        nose = inputs["nose_exclusion_mask"]
        lip = inputs.get("lip_oral_mask")
        if lip is None:
            lip = torch.zeros_like(edit)
        allowed = edit * (1.0 - nose)
        masks = self._mask_inputs(edit, core, motion, ring, nose, lip)
        condition = base["condition"].to(source.dtype)

        geometry_flow = self._dense_geometry_flow(
            inputs["source_points_px"], inputs["target_points_px"], allowed
        )
        half_size = (max(source.shape[-2] // 2, 1), max(source.shape[-1] // 2, 1))
        structure_input = torch.cat(
            [
                base["completion"],
                base["warped_source"],
                base["warped_reference"],
                base["synthesized"],
                condition,
                masks,
            ],
            dim=1,
        )
        structure_logits = self.structure(
            self._resize(structure_input, half_size),
            self._resize(condition, half_size),
        )
        structure_logits = self._resize(structure_logits, source.shape[-2:])
        residual_flow = (
            torch.tanh(structure_logits[:, :2])
            * self.config.max_structure_correction_px
            * allowed
        )
        geometry_warped_source = self.frame_renderer._warp(
            source, geometry_flow + residual_flow
        )
        mix_logits = structure_logits[:, 2:3]
        # Forward is nonnegative and exactly zero at initialization. The
        # straight-through derivative lets the zero-initialized head learn.
        geometry_mix = mix_logits + (F.relu(mix_logits) - mix_logits).detach()
        geometry_mix = geometry_mix.clamp(max=1.0) * allowed
        structure_rgb = (
            torch.tanh(structure_logits[:, 3:6])
            * self.config.max_structure_rgb
            * allowed
        )
        structure_completion = (
            base["completion"]
            + geometry_mix * (geometry_warped_source - base["completion"])
            + structure_rgb
        ).clamp(0.0, 1.0)

        appearance_input = torch.cat(
            [
                structure_completion,
                base["completion"],
                base["warped_reference"],
                source,
                condition,
                masks,
            ],
            dim=1,
        )
        appearance_logits = self.appearance(appearance_input, condition)
        inner = core * allowed
        low = (
            torch.tanh(appearance_logits[:, :3])
            * self.config.max_low_frequency_rgb
            * inner
        )
        raw_detail = torch.tanh(appearance_logits[:, 3:6])
        kernel = self.config.detail_mean_kernel
        if kernel % 2 != 1:
            raise ValueError("detail_mean_kernel must be odd")
        high_detail = raw_detail - F.avg_pool2d(
            raw_detail, kernel, stride=1, padding=kernel // 2
        )
        detail_support = torch.maximum(lip, motion * core) * (1.0 - ring) * allowed
        detail_confidence = torch.sigmoid(appearance_logits[:, 6:7]) * detail_support
        detail = high_detail * self.config.max_detail_rgb * detail_confidence
        completion = (structure_completion + low + detail).clamp(0.0, 1.0)
        alpha = base["alpha"]
        output = source * (1.0 - alpha) + completion * alpha

        return {
            **base,
            "output": output.clamp(0.0, 1.0),
            "completion": completion,
            "base_output": base["output"],
            "base_completion": base["completion"],
            "geometry_flow": geometry_flow,
            "structure_residual_flow": residual_flow,
            "geometry_warped_source": geometry_warped_source,
            "geometry_mix": geometry_mix,
            "structure_rgb_correction": structure_rgb,
            "low_frequency_correction": low,
            "detail_correction": detail,
            "detail_confidence": detail_confidence,
            "detail_support": detail_support,
        }


def initialize_v3_from_v2_checkpoint(
    model: LowerFaceGeneratorV3,
    checkpoint: str | Path,
) -> dict:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    architecture = payload.get("architecture")
    if architecture != LowerFaceGeneratorV2.architecture_version:
        raise ValueError(f"Expected a V2 checkpoint, got {architecture!r}")
    model.frame_renderer.load_state_dict(payload["model"], strict=True)
    return payload


__all__ = [
    "GeometryModulatedResidualBlock",
    "LowerFaceGeneratorV3",
    "LowerFaceGeneratorV3Config",
    "initialize_v3_from_v2_checkpoint",
]
