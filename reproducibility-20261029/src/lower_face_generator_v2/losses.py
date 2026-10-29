from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LowerFaceLossV2Config:
    rgb_weight: float = 1.0
    edge_weight: float = 0.15
    flow_smooth_weight: float = 0.02
    flow_landmark_weight: float = 0.005
    alpha_weight: float = 0.10
    exterior_weight: float = 10.0
    boundary_color_weight: float = 0.0
    boundary_gradient_weight: float = 0.0
    ring_alpha_weight: float = 0.0
    ring_alpha_smooth_weight: float = 0.0
    upper_boundary_color_weight: float = 0.0
    upper_boundary_gradient_weight: float = 0.0
    upper_alpha_weight: float = 0.0
    upper_paired_color_weight: float = 0.0
    boundary_correction_regularization_weight: float = 0.0
    epsilon: float = 1e-3


def _charbonnier(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.sqrt(value.square() + epsilon * epsilon)


def _gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return value[..., :, 1:] - value[..., :, :-1], value[..., 1:, :] - value[..., :-1, :]


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    channels = value.shape[1]
    return (value * mask).sum() / (mask.sum() * channels).clamp_min(1.0)


def paired_upper_boundary_color_loss(
    output: torch.Tensor,
    source: torch.Tensor,
    edit: torch.Tensor,
    upper_overlap: torch.Tensor,
    upper_alpha_target: torch.Tensor,
    nose: torch.Tensor,
) -> torch.Tensor:
    """Match inner transition color to local source context outside the seam."""
    valid = 1.0 - nose
    outer = F.max_pool2d(upper_overlap, 15, stride=1, padding=7) * (1.0 - edit) * valid
    source_low = F.avg_pool2d(source, 15, stride=1, padding=7)
    numerator = F.avg_pool2d(source_low * outer, 31, stride=1, padding=15)
    denominator = F.avg_pool2d(outer, 31, stride=1, padding=15)
    context = numerator / denominator.clamp_min(1e-4)
    seam = 4.0 * upper_alpha_target * (1.0 - upper_alpha_target)
    support = upper_overlap * seam * valid * (denominator > 1e-4).to(output.dtype)
    output_low = F.avg_pool2d(output, 15, stride=1, padding=7)
    return _masked_mean((output_low - context).abs(), support)


def boundary_continuity_loss(
    output: torch.Tensor,
    source: torch.Tensor,
    alpha: torch.Tensor,
    edit: torch.Tensor,
    core: torch.Tensor,
    ring: torch.Tensor,
    nose: torch.Tensor,
    upper_overlap: torch.Tensor | None = None,
    upper_alpha_target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    valid = 1.0 - nose
    transition = edit * (1.0 - core) * valid
    ring_support = ring * valid
    output_low = F.avg_pool2d(output, 11, stride=1, padding=5)
    source_low = F.avg_pool2d(source, 11, stride=1, padding=5)
    color = _masked_mean((output_low - source_low).abs(), transition)

    output_dx, output_dy = _gradient(output)
    source_dx, source_dy = _gradient(source)
    gradient = _masked_mean((output_dx - source_dx).abs(), ring_support[..., :, 1:])
    gradient = gradient + _masked_mean((output_dy - source_dy).abs(), ring_support[..., 1:, :])

    # The model is contractually zero outside edit. Inside the two-pixel
    # transition, a blurred edit support supplies a monotonic fractional target.
    soft_target = F.avg_pool2d(edit, 11, stride=1, padding=5) * edit * valid
    if upper_overlap is None:
        upper_overlap = torch.zeros_like(edit)
    else:
        upper_overlap = upper_overlap * valid
    if upper_alpha_target is not None:
        soft_target = (
            soft_target * (1.0 - upper_overlap)
            + upper_alpha_target * upper_overlap
        )
    alpha_target = (alpha - soft_target).abs()
    ring_alpha = (alpha_target * ring_support).sum() / ring_support.sum().clamp_min(1.0)
    alpha_dx, alpha_dy = _gradient(alpha)
    alpha_smooth = (alpha_dx.abs() * ring_support[..., :, 1:]).sum() / ring_support[..., :, 1:].sum().clamp_min(1.0)
    alpha_smooth = alpha_smooth + (alpha_dy.abs() * ring_support[..., 1:, :]).sum() / ring_support[..., 1:, :].sum().clamp_min(1.0)
    upper_color = _masked_mean((output_low - source_low).abs(), upper_overlap)
    upper_gradient = _masked_mean(
        (output_dx - source_dx).abs(), upper_overlap[..., :, 1:]
    )
    upper_gradient = upper_gradient + _masked_mean(
        (output_dy - source_dy).abs(), upper_overlap[..., 1:, :]
    )
    upper_alpha = (
        (alpha - soft_target).abs() * upper_overlap
    ).sum() / upper_overlap.sum().clamp_min(1.0)
    upper_paired_color = paired_upper_boundary_color_loss(
        output,
        source,
        edit,
        upper_overlap,
        soft_target,
        nose,
    )
    return {
        "boundary_color": color,
        "boundary_gradient": gradient,
        "ring_alpha": ring_alpha,
        "ring_alpha_smooth": alpha_smooth,
        "upper_boundary_color": upper_color,
        "upper_boundary_gradient": upper_gradient,
        "upper_alpha": upper_alpha,
        "upper_paired_color": upper_paired_color,
    }


def lower_face_generator_v2_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: LowerFaceLossV2Config | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config or LowerFaceLossV2Config()
    output = prediction["output"]
    target = batch["target_rgb"]
    edit = batch["edit_mask"]
    core = batch["inpaint_core_mask"]
    lip = batch["lip_oral_mask"]
    motion = batch["motion_mask"]
    ring = batch["ring_mask"]
    nose = batch["nose_exclusion_mask"]
    region_weight = edit * (1.0 + 3.0 * core + 4.0 * lip + 1.5 * motion + ring)
    rgb_error = _charbonnier(output - target, cfg.epsilon) * region_weight
    rgb = rgb_error.sum() / (region_weight.sum() * output.shape[1]).clamp_min(1.0)

    output_dx, output_dy = _gradient(output)
    target_dx, target_dy = _gradient(target)
    edge = (
        _charbonnier(output_dx - target_dx, cfg.epsilon) * lip[..., :, 1:]
    ).mean() + (
        _charbonnier(output_dy - target_dy, cfg.epsilon) * lip[..., 1:, :]
    ).mean()

    source_dx, source_dy = _gradient(prediction["source_flow"])
    reference_dx, reference_dy = _gradient(prediction["reference_flow"])
    flow_smooth = sum(value.abs().mean() for value in (source_dx, source_dy, reference_dx, reference_dy))
    expected_dx = prediction["dx"] * output.shape[-1]
    expected_dy = prediction["dy"] * output.shape[-2]
    support = prediction["target_heat"].detach().clamp(0.0, 1.0)
    flow_landmark = (
        (prediction["source_flow"][:, :1] - expected_dx).abs()
        + (prediction["source_flow"][:, 1:2] - expected_dy).abs()
    )
    flow_landmark = (flow_landmark * support).sum() / support.sum().clamp_min(1.0)

    alpha = prediction["alpha"]
    alpha_core = prediction.get("opaque_core", core)
    alpha_loss = ((1.0 - alpha) * alpha_core).mean() + (alpha * (1.0 - edit)).mean() + (alpha * nose).mean()
    boundary = boundary_continuity_loss(
        output,
        batch["source_rgb"],
        alpha,
        edit,
        alpha_core,
        ring,
        nose,
        batch.get("upper_overlap_mask"),
        batch.get("upper_alpha_target_mask"),
    )
    exterior = ((output - batch["source_rgb"]).abs() * (1.0 - edit)).mean()
    correction = prediction.get("boundary_correction")
    correction_regularization = (
        correction.abs().mean() if correction is not None else output.new_zeros(())
    )
    total = (
        cfg.rgb_weight * rgb
        + cfg.edge_weight * edge
        + cfg.flow_smooth_weight * flow_smooth
        + cfg.flow_landmark_weight * flow_landmark
        + cfg.alpha_weight * alpha_loss
        + cfg.exterior_weight * exterior
        + cfg.boundary_color_weight * boundary["boundary_color"]
        + cfg.boundary_gradient_weight * boundary["boundary_gradient"]
        + cfg.ring_alpha_weight * boundary["ring_alpha"]
        + cfg.ring_alpha_smooth_weight * boundary["ring_alpha_smooth"]
        + cfg.upper_boundary_color_weight * boundary["upper_boundary_color"]
        + cfg.upper_boundary_gradient_weight * boundary["upper_boundary_gradient"]
        + cfg.upper_alpha_weight * boundary["upper_alpha"]
        + cfg.upper_paired_color_weight * boundary["upper_paired_color"]
        + cfg.boundary_correction_regularization_weight * correction_regularization
    )
    metrics = {
        "total": total.detach(),
        "rgb": rgb.detach(),
        "edge": edge.detach(),
        "flow_smooth": flow_smooth.detach(),
        "flow_landmark": flow_landmark.detach(),
        "alpha": alpha_loss.detach(),
        "exterior": exterior.detach(),
        "boundary_correction_regularization": correction_regularization.detach(),
        **{key: value.detach() for key, value in boundary.items()},
    }
    return total, metrics


__all__ = [
    "LowerFaceLossV2Config",
    "boundary_continuity_loss",
    "lower_face_generator_v2_loss",
    "paired_upper_boundary_color_loss",
]
