from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import boundary_continuity_loss
from .texture_v4 import LowerFaceTextureV4


def _masked_moments(value: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = weight.sum((2, 3), keepdim=True).clamp_min(32.0)
    mean = (value * weight).sum((2, 3), keepdim=True) / denominator
    variance = ((value - mean).square() * weight).sum((2, 3), keepdim=True) / denominator
    return mean, variance.clamp_min(1e-5).sqrt()


def robust_boundary_color_transform(
    generated: torch.Tensor,
    source: torch.Tensor,
    support: torch.Tensor,
    previous_scale: torch.Tensor | None = None,
    previous_bias: torch.Tensor | None = None,
    ema: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate a clipped affine transform from paired, skin-only low frequencies."""
    generated_low = F.avg_pool2d(F.pad(generated.float(), (7, 7, 7, 7), mode="replicate"), 15, stride=1)
    source_low = F.avg_pool2d(F.pad(source.float(), (7, 7, 7, 7), mode="replicate"), 15, stride=1)
    residual = (generated_low - source_low).abs().mean(1, keepdim=True)
    residual_center = _masked_moments(residual, support.float())[0]
    robust = support.float() / (1.0 + (residual / (residual_center + 0.015)).square())
    generated_mean, generated_std = _masked_moments(generated_low, robust)
    source_mean, source_std = _masked_moments(source_low, robust)
    scale = (source_std / generated_std).clamp(0.85, 1.15)
    bias = (source_mean - generated_mean * scale).clamp(-0.12, 0.12)
    valid = (support.sum((2, 3), keepdim=True) >= 32.0).to(scale.dtype)
    scale = scale * valid + (1.0 - valid)
    bias = bias * valid
    if previous_scale is not None and previous_bias is not None and ema > 0.0:
        scale = ema * previous_scale.float() + (1.0 - ema) * scale
        bias = ema * previous_bias.float() + (1.0 - ema) * bias
    matched_low = generated_low * scale + bias
    correction = (matched_low - generated_low).clamp(-0.12, 0.12)
    return scale.to(generated.dtype), bias.to(generated.dtype), correction.to(generated.dtype)


def _gaussian(value: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(value, 5, stride=1, padding=2)


def multiband_blend(source: torch.Tensor, generated: torch.Tensor, alpha: torch.Tensor, levels: int = 4) -> torch.Tensor:
    source_gaussian = [source]
    generated_gaussian = [generated]
    alpha_gaussian = [alpha]
    for _ in range(max(levels - 1, 0)):
        source_gaussian.append(F.avg_pool2d(_gaussian(source_gaussian[-1]), 2, 2))
        generated_gaussian.append(F.avg_pool2d(_gaussian(generated_gaussian[-1]), 2, 2))
        alpha_gaussian.append(F.avg_pool2d(_gaussian(alpha_gaussian[-1]), 2, 2))
    blended = source_gaussian[-1] * (1.0 - alpha_gaussian[-1]) + generated_gaussian[-1] * alpha_gaussian[-1]
    for level in range(len(source_gaussian) - 2, -1, -1):
        size = source_gaussian[level].shape[-2:]
        source_up = F.interpolate(source_gaussian[level + 1], size, mode="bilinear", align_corners=False)
        generated_up = F.interpolate(generated_gaussian[level + 1], size, mode="bilinear", align_corners=False)
        source_laplacian = source_gaussian[level] - source_up
        generated_laplacian = generated_gaussian[level] - generated_up
        blended = F.interpolate(blended, size, mode="bilinear", align_corners=False)
        blended = blended + source_laplacian * (1.0 - alpha_gaussian[level]) + generated_laplacian * alpha_gaussian[level]
    return blended


class BoundaryResidualV4C(nn.Module):
    def __init__(self, channels: int = 24, residual_scale: float = 0.06) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.network = nn.Sequential(
            nn.Conv2d(13, channels, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1), nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 2, dilation=2), nn.SiLU(inplace=True),
            nn.Conv2d(channels, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, source: torch.Tensor, generated: torch.Tensor, ring: torch.Tensor,
        lip: torch.Tensor, edit: torch.Tensor, nose: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.cat([source, generated, generated - source, ring, lip, edit, nose], dim=1)
        return torch.tanh(self.network(value)) * self.residual_scale * ring * (1.0 - nose)


@dataclass(frozen=True)
class LowerFaceCompositorV4CConfig:
    boundary_channels: int = 24
    residual_scale: float = 0.06
    color_strength: float = 0.90
    lip_color_strength: float = 0.15
    pyramid_levels: int = 4
    temporal_ema: float = 0.90


class LowerFaceCompositorV4C(nn.Module):
    """Frozen V4B plus skin-calibrated, geometry-preserving boundary compositing."""

    architecture_version = "lower_face_compositor_v4c"

    def __init__(self, texture: LowerFaceTextureV4, config: LowerFaceCompositorV4CConfig | None = None) -> None:
        super().__init__()
        self.config = config or LowerFaceCompositorV4CConfig()
        self.texture = texture.eval().requires_grad_(False)
        self.boundary_residual = BoundaryResidualV4C(
            self.config.boundary_channels, self.config.residual_scale
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.texture.eval()
        return self

    def forward(
        self, *, source_rgb: torch.Tensor, source_masked: torch.Tensor,
        references_rgb: torch.Tensor, target_points_px: torch.Tensor,
        reference_points_px: torch.Tensor, geometry81: torch.Tensor,
        edit_mask: torch.Tensor, lip_oral_mask: torch.Tensor, motion_mask: torch.Tensor,
        ring_mask: torch.Tensor, nose_exclusion_mask: torch.Tensor,
        alpha_core_mask: torch.Tensor, upper_overlap_mask: torch.Tensor | None = None,
        upper_alpha_target_mask: torch.Tensor | None = None,
        previous_color_scale: torch.Tensor | None = None,
        previous_color_bias: torch.Tensor | None = None,
        temporal_ema: float | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            base = self.texture(
                source_rgb=source_rgb, source_masked=source_masked,
                references_rgb=references_rgb, target_points_px=target_points_px,
                reference_points_px=reference_points_px, geometry81=geometry81,
            )
        generated = base["completion"]
        valid_skin = (1.0 - nose_exclusion_mask) * (1.0 - lip_oral_mask)
        calibration_support = ring_mask * edit_mask * valid_skin
        ema = self.config.temporal_ema if temporal_ema is None else float(temporal_ema)
        scale, bias, low_correction = robust_boundary_color_transform(
            generated, source_rgb, calibration_support,
            previous_color_scale, previous_color_bias, ema,
        )
        color_envelope = (
            self.config.color_strength * ring_mask * (1.0 - motion_mask)
            + self.config.lip_color_strength * lip_oral_mask
        ) * edit_mask * (1.0 - nose_exclusion_mask)
        calibrated = (generated + low_correction * color_envelope).clamp(0.0, 1.0)
        residual = self.boundary_residual(
            source_rgb, calibrated, ring_mask, lip_oral_mask, edit_mask, nose_exclusion_mask
        )
        corrected = (calibrated + residual).clamp(0.0, 1.0)
        soft_alpha = F.avg_pool2d(edit_mask, 9, stride=1, padding=4) * edit_mask
        alpha = torch.maximum(soft_alpha, alpha_core_mask)
        if upper_overlap_mask is not None and upper_alpha_target_mask is not None:
            alpha = alpha * (1.0 - upper_overlap_mask) + upper_alpha_target_mask * upper_overlap_mask
        alpha = torch.maximum(alpha, lip_oral_mask)
        alpha = alpha * edit_mask * (1.0 - nose_exclusion_mask)
        output = multiband_blend(source_rgb, corrected, alpha, self.config.pyramid_levels)
        # Multiband reconstruction intentionally softens high frequencies. The
        # generated lip/oral core remains an opaque V4B authority region.
        eroded_motion = -F.max_pool2d(-motion_mask, 9, stride=1, padding=4)
        authority = torch.maximum(lip_oral_mask, eroded_motion) * (1.0 - nose_exclusion_mask)
        output = output * (1.0 - authority) + base["output"] * authority
        output = output * edit_mask + source_rgb * (1.0 - edit_mask)
        return {
            "output": output.clamp(0.0, 1.0), "completion": corrected,
            "calibrated_completion": calibrated, "alpha": alpha,
            "boundary_correction": residual, "low_frequency_correction": low_correction,
            "color_scale": scale, "color_bias": bias,
            "calibration_support": calibration_support, "authority_mask": authority, "base": base,
        }


@dataclass(frozen=True)
class LowerFaceCompositorV4CLossConfig:
    reconstruction_weight: float = 0.30
    boundary_color_weight: float = 2.0
    boundary_gradient_weight: float = 1.0
    upper_color_weight: float = 1.0
    upper_gradient_weight: float = 0.5
    lip_distillation_weight: float = 2.0
    residual_regularization_weight: float = 0.20


def lower_face_compositor_v4c_loss(
    prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
    config: LowerFaceCompositorV4CLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config or LowerFaceCompositorV4CLossConfig()
    output, target = prediction["output"], batch["target_rgb"]
    ring = batch["ring_mask"] * (1.0 - batch["nose_exclusion_mask"])
    reconstruction = ((output - target).abs() * ring).sum() / (ring.sum() * 3).clamp_min(1.0)
    boundary = boundary_continuity_loss(
        output, batch["source_rgb"], prediction["alpha"], batch["edit_mask"],
        batch["alpha_core_mask"], batch["ring_mask"], batch["nose_exclusion_mask"],
        batch.get("upper_overlap_mask"), batch.get("upper_alpha_target_mask"),
    )
    lip = batch["lip_oral_mask"]
    base = prediction["base"]["output"].detach()
    lip_distillation = ((output - base).abs() * lip).sum() / (lip.sum() * 3).clamp_min(1.0)
    residual_regularization = prediction["boundary_correction"].abs().sum() / (ring.sum() * 3).clamp_min(1.0)
    total = (
        cfg.reconstruction_weight * reconstruction
        + cfg.boundary_color_weight * boundary["boundary_color"]
        + cfg.boundary_gradient_weight * boundary["boundary_gradient"]
        + cfg.upper_color_weight * boundary["upper_boundary_color"]
        + cfg.upper_gradient_weight * boundary["upper_boundary_gradient"]
        + cfg.lip_distillation_weight * lip_distillation
        + cfg.residual_regularization_weight * residual_regularization
    )
    metrics = {
        "total": total.detach(), "reconstruction": reconstruction.detach(),
        "lip_distillation": lip_distillation.detach(),
        "residual_regularization": residual_regularization.detach(),
        **{name: value.detach() for name, value in boundary.items()},
    }
    return total, metrics


__all__ = [
    "LowerFaceCompositorV4C", "LowerFaceCompositorV4CConfig",
    "LowerFaceCompositorV4CLossConfig", "lower_face_compositor_v4c_loss",
    "multiband_blend", "robust_boundary_color_transform",
]
