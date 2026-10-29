from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LowerFaceTextureV5LossConfig:
    direct_weight: float = 2.0
    reference_weight: float = 1.0
    transition_weight: float = 1.5
    final_rgb_weight: float = 1.0
    mouth_edge_weight: float = 0.35
    transition_gradient_weight: float = 0.25
    exterior_weight: float = 20.0
    nose_weight: float = 20.0
    epsilon: float = 1e-3


def _charbonnier(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.sqrt(value.square() + epsilon * epsilon)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / (mask.sum() * value.shape[1]).clamp_min(1.0)


def _gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return value[..., :, 1:] - value[..., :, :-1], value[..., 1:, :] - value[..., :-1, :]


def lower_face_texture_v5_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: LowerFaceTextureV5LossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config or LowerFaceTextureV5LossConfig()
    target = batch["target_rgb"]
    source = batch["source_rgb"]
    direct = batch["direct_rgb_ownership_mask"]
    reference = batch["safe_reference_skin_mask"]
    transition = batch["mouth_skin_transition_mask"]
    edit = batch["edit_union_mask"]
    lip = batch["target_lip_support_mask"]
    nose = batch["nose_exclusion_mask"]
    alpha = batch["alpha_target_mask"]
    final = prediction["final_output"]

    direct_rgb = _masked_mean(
        _charbonnier(prediction["direct_mouth_rgb"] - target, cfg.epsilon), direct
    )
    reference_rgb = _masked_mean(
        _charbonnier(prediction["reference_skin_rgb"] - target, cfg.epsilon), reference
    )
    transition_rgb = _masked_mean(
        _charbonnier(prediction["transition_rgb"] - target, cfg.epsilon), transition
    )
    final_rgb = _masked_mean(_charbonnier(final - target, cfg.epsilon), edit)

    final_dx, final_dy = _gradient(final)
    target_dx, target_dy = _gradient(target)
    mouth_edge = _masked_mean(
        _charbonnier(final_dx - target_dx, cfg.epsilon), lip[..., :, 1:]
    ) + _masked_mean(
        _charbonnier(final_dy - target_dy, cfg.epsilon), lip[..., 1:, :]
    )
    transition_gradient = _masked_mean(
        _charbonnier(final_dx - target_dx, cfg.epsilon), transition[..., :, 1:]
    ) + _masked_mean(
        _charbonnier(final_dy - target_dy, cfg.epsilon), transition[..., 1:, :]
    )
    exterior_mask = (alpha == 0).to(final.dtype)
    exterior = _masked_mean((final - source).abs(), exterior_mask)
    nose_error = _masked_mean((final - source).abs(), nose)

    partition = direct + reference + transition
    ownership_overlap = torch.relu(partition - 1.0).mean()
    ownership_unassigned = (edit * torch.relu(1.0 - partition)).mean()
    alpha_contract = (
        (prediction["alpha"] - alpha).abs().mean()
        + (prediction["alpha"] * nose).mean()
        + (prediction["alpha"] * (1.0 - edit)).mean()
    )
    total = (
        cfg.direct_weight * direct_rgb
        + cfg.reference_weight * reference_rgb
        + cfg.transition_weight * transition_rgb
        + cfg.final_rgb_weight * final_rgb
        + cfg.mouth_edge_weight * mouth_edge
        + cfg.transition_gradient_weight * transition_gradient
        + cfg.exterior_weight * exterior
        + cfg.nose_weight * nose_error
    )
    metrics = {
        "total": total.detach(),
        "direct_rgb": direct_rgb.detach(),
        "reference_rgb": reference_rgb.detach(),
        "transition_rgb": transition_rgb.detach(),
        "final_rgb": final_rgb.detach(),
        "mouth_edge": mouth_edge.detach(),
        "transition_gradient": transition_gradient.detach(),
        "exterior": exterior.detach(),
        "nose": nose_error.detach(),
        "ownership_overlap": ownership_overlap.detach(),
        "ownership_unassigned": ownership_unassigned.detach(),
        "alpha_contract": alpha_contract.detach(),
    }
    return total, metrics


__all__ = ["LowerFaceTextureV5LossConfig", "lower_face_texture_v5_loss"]
