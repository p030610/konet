from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compositor_v4 import multiband_blend, robust_boundary_color_transform
from .losses import boundary_continuity_loss
from .texture_v4 import LowerFaceTextureV4


class ContextBlockV4D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.GroupNorm(groups, out_channels), nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(groups, out_channels), nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


class ContextIlluminationDecoderV4D(nn.Module):
    """Wide-context U-Net that predicts bounded illumination and alpha fields."""

    def __init__(self, input_channels: int = 25, base_channels: int = 24) -> None:
        super().__init__()
        base = int(base_channels)
        self.enc0 = ContextBlockV4D(input_channels, base)
        self.enc1 = ContextBlockV4D(base, base * 2, 2)
        self.enc2 = ContextBlockV4D(base * 2, base * 4, 2)
        self.enc3 = ContextBlockV4D(base * 4, base * 8, 2)
        self.enc4 = ContextBlockV4D(base * 8, base * 8, 2)
        self.mid = ContextBlockV4D(base * 8, base * 8)
        self.dec3 = ContextBlockV4D(base * 16, base * 8)
        self.dec2 = ContextBlockV4D(base * 12, base * 4)
        self.dec1 = ContextBlockV4D(base * 6, base * 2)
        self.dec0 = ContextBlockV4D(base * 3, base)
        self.head = nn.Conv2d(base, 7, 3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _up(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(value)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        value = self.mid(e4)
        value = self.dec3(torch.cat([self._up(value, e3), e3], 1))
        value = self.dec2(torch.cat([self._up(value, e2), e2], 1))
        value = self.dec1(torch.cat([self._up(value, e1), e1], 1))
        value = self.dec0(torch.cat([self._up(value, e0), e0], 1))
        return self.head(value)


@dataclass(frozen=True)
class LowerFaceCompositorV4DConfig:
    context_channels: int = 24
    context_dilate_px: int = 21
    gain_scale: float = 0.10
    bias_scale: float = 0.08
    alpha_delta_scale: float = 0.20
    analytic_color_strength: float = 0.75
    pyramid_levels: int = 4
    temporal_ema: float = 0.90


class LowerFaceCompositorV4D(nn.Module):
    """Frozen V4B with spatial illumination and alpha inferred from wide context."""

    architecture_version = "lower_face_compositor_v4d"

    def __init__(self, texture: LowerFaceTextureV4, config: LowerFaceCompositorV4DConfig | None = None) -> None:
        super().__init__()
        self.config = config or LowerFaceCompositorV4DConfig()
        self.texture = texture.eval().requires_grad_(False)
        self.context_decoder = ContextIlluminationDecoderV4D(25, self.config.context_channels)

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
        cfg = self.config
        with torch.no_grad():
            base = self.texture(
                source_rgb=source_rgb, source_masked=source_masked,
                references_rgb=references_rgb, target_points_px=target_points_px,
                reference_points_px=reference_points_px, geometry81=geometry81,
            )
        generated = base["completion"]
        valid_skin = (1.0 - nose_exclusion_mask) * (1.0 - lip_oral_mask)
        calibration_support = ring_mask * edit_mask * valid_skin
        ema = cfg.temporal_ema if temporal_ema is None else float(temporal_ema)
        scale, bias, low_correction = robust_boundary_color_transform(
            generated, source_rgb, calibration_support,
            previous_color_scale, previous_color_bias, ema,
        )
        kernel = max(3, int(cfg.context_dilate_px) | 1)
        context_band = F.max_pool2d(ring_mask, kernel, stride=1, padding=kernel // 2)
        context_band = context_band * edit_mask * (1.0 - nose_exclusion_mask)
        eroded_motion = -F.max_pool2d(-motion_mask, 9, stride=1, padding=4)
        transition_guard = F.max_pool2d(ring_mask, 7, stride=1, padding=3)
        motion_core = eroded_motion * (1.0 - transition_guard)
        authority = torch.maximum(lip_oral_mask, motion_core) * (1.0 - nose_exclusion_mask)
        editable_context = context_band * (1.0 - authority)
        analytic = (generated + cfg.analytic_color_strength * low_correction * editable_context).clamp(0, 1)

        source_low = F.avg_pool2d(F.pad(source_rgb, (7, 7, 7, 7), mode="replicate"), 15, 1)
        generated_low = F.avg_pool2d(F.pad(analytic, (7, 7, 7, 7), mode="replicate"), 15, 1)
        batch, _, height, width = source_rgb.shape
        y = torch.linspace(-1, 1, height, device=source_rgb.device, dtype=source_rgb.dtype)
        x = torch.linspace(-1, 1, width, device=source_rgb.device, dtype=source_rgb.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([xx, yy])[None].expand(batch, -1, -1, -1)
        if upper_overlap_mask is None:
            upper_overlap_mask = torch.zeros_like(edit_mask)
        if upper_alpha_target_mask is None:
            upper_alpha_target_mask = torch.zeros_like(edit_mask)
        context = torch.cat([
            source_rgb, analytic, source_low, generated_low, generated_low - source_low,
            edit_mask, lip_oral_mask, motion_mask, ring_mask, nose_exclusion_mask,
            alpha_core_mask, upper_overlap_mask, upper_alpha_target_mask, coordinates,
        ], 1)
        field = self.context_decoder(context)
        gain_logits, bias_logits, alpha_logits = field.split([3, 3, 1], 1)
        gain_field = torch.tanh(gain_logits) * cfg.gain_scale * editable_context
        bias_field = torch.tanh(bias_logits) * cfg.bias_scale * editable_context
        corrected = (analytic * (1.0 + gain_field) + bias_field).clamp(0, 1)

        soft_alpha = F.avg_pool2d(edit_mask, 9, stride=1, padding=4) * edit_mask
        alpha = torch.maximum(soft_alpha, alpha_core_mask)
        alpha = alpha * (1.0 - upper_overlap_mask) + upper_alpha_target_mask * upper_overlap_mask
        alpha_delta = torch.tanh(alpha_logits) * cfg.alpha_delta_scale * ring_mask * (1.0 - authority)
        alpha = (alpha + alpha_delta).clamp(0, 1)
        alpha = alpha * edit_mask * (1.0 - nose_exclusion_mask)
        output = multiband_blend(source_rgb, corrected, alpha, cfg.pyramid_levels)
        output = output * (1.0 - authority) + base["output"] * authority
        output = output * edit_mask + source_rgb * (1.0 - edit_mask)
        return {
            "output": output.clamp(0, 1), "completion": corrected,
            "calibrated_completion": analytic, "alpha": alpha,
            "boundary_correction": bias_field + analytic * gain_field,
            "low_frequency_correction": low_correction,
            "gain_field": gain_field, "bias_field": bias_field, "alpha_delta": alpha_delta,
            "color_scale": scale, "color_bias": bias,
            "calibration_support": calibration_support, "context_band": context_band,
            "authority_mask": authority, "base": base,
        }


@dataclass(frozen=True)
class LowerFaceCompositorV4DLossConfig:
    context_reconstruction_weight: float = 0.50
    boundary_color_weight: float = 2.0
    boundary_gradient_weight: float = 1.0
    upper_color_weight: float = 1.0
    upper_gradient_weight: float = 0.5
    field_regularization_weight: float = 0.10
    field_smoothness_weight: float = 0.05
    alpha_regularization_weight: float = 0.10
    ring_alpha_weight: float = 0.50
    upper_alpha_weight: float = 1.00


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / (mask.sum() * value.shape[1]).clamp_min(1.0)


def lower_face_compositor_v4d_loss(
    prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
    config: LowerFaceCompositorV4DLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config or LowerFaceCompositorV4DLossConfig()
    output = prediction["output"]
    context = prediction["context_band"] * (1.0 - prediction["authority_mask"])
    reconstruction = _masked_mean((output - batch["target_rgb"]).abs(), context)
    boundary = boundary_continuity_loss(
        output, batch["source_rgb"], prediction["alpha"], batch["edit_mask"],
        batch["alpha_core_mask"], batch["ring_mask"], batch["nose_exclusion_mask"],
        batch.get("upper_overlap_mask"), batch.get("upper_alpha_target_mask"),
    )
    gain, bias, alpha_delta = prediction["gain_field"], prediction["bias_field"], prediction["alpha_delta"]
    field_regularization = _masked_mean(gain.abs() + bias.abs(), context)
    dx = (gain[..., :, 1:] - gain[..., :, :-1]).abs() + (bias[..., :, 1:] - bias[..., :, :-1]).abs()
    dy = (gain[..., 1:, :] - gain[..., :-1, :]).abs() + (bias[..., 1:, :] - bias[..., :-1, :]).abs()
    field_smoothness = _masked_mean(dx, context[..., :, 1:]) + _masked_mean(dy, context[..., 1:, :])
    alpha_regularization = _masked_mean(alpha_delta.abs(), batch["ring_mask"])
    total = (
        cfg.context_reconstruction_weight * reconstruction
        + cfg.boundary_color_weight * boundary["boundary_color"]
        + cfg.boundary_gradient_weight * boundary["boundary_gradient"]
        + cfg.upper_color_weight * boundary["upper_boundary_color"]
        + cfg.upper_gradient_weight * boundary["upper_boundary_gradient"]
        + cfg.field_regularization_weight * field_regularization
        + cfg.field_smoothness_weight * field_smoothness
        + cfg.alpha_regularization_weight * alpha_regularization
        + cfg.ring_alpha_weight * boundary["ring_alpha"]
        + cfg.upper_alpha_weight * boundary["upper_alpha"]
    )
    metrics = {
        "total": total.detach(), "context_reconstruction": reconstruction.detach(),
        "field_regularization": field_regularization.detach(),
        "field_smoothness": field_smoothness.detach(),
        "alpha_regularization": alpha_regularization.detach(),
        **{name: value.detach() for name, value in boundary.items()},
    }
    return total, metrics


__all__ = [
    "ContextIlluminationDecoderV4D", "LowerFaceCompositorV4D",
    "LowerFaceCompositorV4DConfig", "LowerFaceCompositorV4DLossConfig",
    "lower_face_compositor_v4d_loss",
]
