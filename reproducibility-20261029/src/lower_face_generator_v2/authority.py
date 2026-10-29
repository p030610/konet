from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GeometryAuthorityLossConfig:
    delta_weight: float = 1.0
    rank_weight: float = 0.5
    source_copy_weight: float = 0.25
    rank_margin_px: float = 1.0
    source_margin_px: float = 0.5


def _point_error(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left - right).square().sum(dim=-1).sqrt().mean(dim=-1)


def geometry_authority_loss(
    output_a_points: torch.Tensor,
    output_b_points: torch.Tensor,
    target_a_points: torch.Tensor,
    target_b_points: torch.Tensor,
    source_points: torch.Tensor,
    config: GeometryAuthorityLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Enforce target-geometry response under a fixed RGB context."""

    cfg = config or GeometryAuthorityLossConfig()
    requested_delta = target_b_points - target_a_points
    output_delta = output_b_points - output_a_points
    delta = F.smooth_l1_loss(output_delta, requested_delta, beta=1.0)

    correct_error = 0.5 * (
        _point_error(output_a_points, target_a_points)
        + _point_error(output_b_points, target_b_points)
    )
    shuffled_error = 0.5 * (
        _point_error(output_a_points, target_b_points)
        + _point_error(output_b_points, target_a_points)
    )
    rank = F.relu(cfg.rank_margin_px + correct_error - shuffled_error).mean()

    target_error = _point_error(output_b_points, target_b_points)
    source_error = _point_error(output_b_points, source_points)
    source_copy = F.relu(cfg.source_margin_px + target_error - source_error).mean()
    total = (
        cfg.delta_weight * delta
        + cfg.rank_weight * rank
        + cfg.source_copy_weight * source_copy
    )

    requested_energy = requested_delta.square().sum(dim=(-1, -2)).clamp_min(1e-6)
    projected_response = (output_delta * requested_delta).sum(dim=(-1, -2)) / requested_energy
    correct_wins = (correct_error < shuffled_error).float()
    metrics = {
        "geometry_authority_total": total.detach(),
        "geometry_delta": delta.detach(),
        "geometry_rank": rank.detach(),
        "source_copy": source_copy.detach(),
        "correct_geometry_accuracy": correct_wins.mean().detach(),
        "requested_response_ratio": projected_response.mean().detach(),
        "correct_geometry_error_px": correct_error.mean().detach(),
        "shuffled_geometry_error_px": shuffled_error.mean().detach(),
    }
    return total, metrics


__all__ = ["GeometryAuthorityLossConfig", "geometry_authority_loss"]
