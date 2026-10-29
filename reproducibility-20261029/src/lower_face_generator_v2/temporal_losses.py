from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TemporalLossV2Config:
    frame_rgb_weight: float = 1.0
    temporal_rgb_weight: float = 0.5
    landmark_velocity_weight: float = 0.1
    flow_acceleration_weight: float = 0.01
    alpha_temporal_weight: float = 0.05
    correction_regularization_weight: float = 0.01
    lip_laplacian_weight: float = 0.0
    base_detail_distillation_weight: float = 0.0
    epsilon: float = 1e-3


def _charbonnier(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.sqrt(value.square() + epsilon * epsilon)


def _laplacian(value: torch.Tensor) -> torch.Tensor:
    channels = value.shape[-3]
    kernel = value.new_tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    ).view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    shape = value.shape
    filtered = F.conv2d(value.flatten(0, 1), kernel, padding=1, groups=channels)
    return filtered.view(*shape)


def _masked_channel_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / (mask.sum() * value.shape[-3]).clamp_min(1.0)


def _similarity_current_to_previous(
    previous_points: torch.Tensor,
    current_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    previous_center = previous_points.mean(dim=-2)
    current_center = current_points.mean(dim=-2)
    previous = previous_points - previous_center[..., None, :]
    current = current_points - current_center[..., None, :]
    denominator = current.square().sum(dim=(-2, -1)).clamp_min(1e-6)
    a = (current * previous).sum(dim=(-2, -1)) / denominator
    b = (
        current[..., 0] * previous[..., 1]
        - current[..., 1] * previous[..., 0]
    ).sum(dim=-1) / denominator
    translation_x = previous_center[..., 0] - (
        a * current_center[..., 0] - b * current_center[..., 1]
    )
    translation_y = previous_center[..., 1] - (
        b * current_center[..., 0] + a * current_center[..., 1]
    )
    return a, b, translation_x, translation_y


def warp_previous_to_current(
    previous: torch.Tensor,
    previous_points: torch.Tensor,
    current_points: torch.Tensor,
) -> torch.Tensor:
    """Rigidly align previous-frame tensors to the current face anchors."""
    batch, pairs, channels, height, width = previous.shape
    a, b, tx, ty = _similarity_current_to_previous(previous_points, current_points)
    y, x = torch.meshgrid(
        torch.arange(height, device=previous.device, dtype=previous.dtype),
        torch.arange(width, device=previous.device, dtype=previous.dtype),
        indexing="ij",
    )
    x = x.view(1, 1, height, width)
    y = y.view(1, 1, height, width)
    source_x = a[..., None, None] * x - b[..., None, None] * y + tx[..., None, None]
    source_y = b[..., None, None] * x + a[..., None, None] * y + ty[..., None, None]
    grid = torch.stack(
        [
            2.0 * source_x / max(width - 1, 1) - 1.0,
            2.0 * source_y / max(height - 1, 1) - 1.0,
        ],
        dim=-1,
    )
    warped = F.grid_sample(
        previous.flatten(0, 1),
        grid.flatten(0, 1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return warped.view(batch, pairs, channels, height, width)


def temporal_lower_face_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    geometry_probe: torch.nn.Module,
    config: TemporalLossV2Config | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config or TemporalLossV2Config()
    output = prediction["output"]
    target = batch["target_rgb"]
    edit = batch["edit_mask"]
    weight = edit * (
        1.0
        + 3.0 * batch["inpaint_core_mask"]
        + 4.0 * batch["lip_oral_mask"]
        + 1.5 * batch["motion_mask"]
    )
    frame_rgb = (_charbonnier(output - target, cfg.epsilon) * weight).sum()
    frame_rgb = frame_rgb / (weight.sum() * output.shape[2]).clamp_min(1.0)

    previous_points = batch["stability_points_px"][:, :-1].float()
    current_points = batch["stability_points_px"][:, 1:].float()
    warped_output = warp_previous_to_current(output[:, :-1], previous_points, current_points)
    warped_target = warp_previous_to_current(target[:, :-1], previous_points, current_points)
    warped_edit = warp_previous_to_current(edit[:, :-1], previous_points, current_points)
    support = torch.maximum(edit[:, 1:], warped_edit).clamp(0.0, 1.0)
    output_velocity = output[:, 1:] - warped_output
    target_velocity = target[:, 1:] - warped_target
    temporal_rgb = (_charbonnier(output_velocity - target_velocity, cfg.epsilon) * support).sum()
    temporal_rgb = temporal_rgb / (support.sum() * output.shape[2]).clamp_min(1.0)

    batch_size, frames = output.shape[:2]
    probe = geometry_probe(output.flatten(0, 1))
    scale = output.new_tensor([output.shape[-1] - 1.0, output.shape[-2] - 1.0])
    predicted_points = (probe["points_normalized"] * scale).view(batch_size, frames, -1, 2)
    target_points = batch["target_points_px"].float()
    predicted_velocity = predicted_points[:, 1:] - predicted_points[:, :-1]
    target_point_velocity = target_points[:, 1:] - target_points[:, :-1]
    landmark_velocity = F.smooth_l1_loss(
        predicted_velocity / 16.0,
        target_point_velocity / 16.0,
        beta=0.05,
    )

    flow = prediction["temporal_flow"]
    if frames > 2:
        flow_acceleration = (flow[:, 2:] - 2 * flow[:, 1:-1] + flow[:, :-2]).abs().mean()
    else:
        flow_acceleration = flow.new_zeros(())
    alpha_correction = prediction["temporal_alpha_correction"]
    alpha_velocity = alpha_correction[:, 1:] - alpha_correction[:, :-1]
    alpha_support = torch.maximum(batch["ring_mask"][:, 1:], batch["ring_mask"][:, :-1])
    alpha_temporal = (alpha_velocity.abs() * alpha_support).sum() / alpha_support.sum().clamp_min(1.0)
    correction_regularization = (
        prediction["temporal_rgb_correction"].abs().mean()
        + flow.abs().mean() / 2.0
        + alpha_correction.abs().mean()
    )
    output_laplacian = _laplacian(output)
    target_laplacian = _laplacian(target)
    lip = batch["lip_oral_mask"]
    lip_laplacian = _masked_channel_mean(
        _charbonnier(output_laplacian - target_laplacian, cfg.epsilon), lip
    )
    base_laplacian = _laplacian(prediction.get("base_output", output).detach())
    base_detail_distillation = _masked_channel_mean(
        (output_laplacian - base_laplacian).abs(), lip
    )

    target_motion = target_point_velocity.square().sum(-1).sqrt().mean()
    predicted_motion = predicted_velocity.square().sum(-1).sqrt().mean()
    motion_amplitude_ratio = predicted_motion / target_motion.clamp_min(1e-4)
    total = (
        cfg.frame_rgb_weight * frame_rgb
        + cfg.temporal_rgb_weight * temporal_rgb
        + cfg.landmark_velocity_weight * landmark_velocity
        + cfg.flow_acceleration_weight * flow_acceleration
        + cfg.alpha_temporal_weight * alpha_temporal
        + cfg.correction_regularization_weight * correction_regularization
        + cfg.lip_laplacian_weight * lip_laplacian
        + cfg.base_detail_distillation_weight * base_detail_distillation
    )
    metrics = {
        "total": total.detach(),
        "frame_rgb": frame_rgb.detach(),
        "temporal_rgb": temporal_rgb.detach(),
        "landmark_velocity": landmark_velocity.detach(),
        "flow_acceleration": flow_acceleration.detach(),
        "alpha_temporal": alpha_temporal.detach(),
        "correction_regularization": correction_regularization.detach(),
        "motion_amplitude_ratio": motion_amplitude_ratio.detach(),
        "predicted_motion_px": predicted_motion.detach(),
        "target_motion_px": target_motion.detach(),
        "lip_laplacian": lip_laplacian.detach(),
        "base_detail_distillation": base_detail_distillation.detach(),
    }
    return total, metrics


__all__ = [
    "TemporalLossV2Config",
    "temporal_lower_face_loss",
    "warp_previous_to_current",
]
