from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


NEUTRAL_POSE6 = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
MOUTH_PROJECTION63_DIM = 63


def load_geometry81(path: str | Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32).reshape(-1)
    if arr.shape[0] != 81:
        raise ValueError(f"Expected geometry81 at {path}, got shape {arr.shape}")
    return arr


def load_pose6(path: str | Path | None) -> np.ndarray:
    if path is None:
        return NEUTRAL_POSE6.copy()
    arr = np.load(path).astype(np.float32).reshape(-1)
    if arr.shape[0] >= 6:
        return arr[:6].astype(np.float32)
    if arr.shape[0] >= 4:
        # Legacy canonical-alignment params: [cx, cy, scale, angle].
        angle = float(arr[3])
        return np.asarray([0.0, 1.0, 0.0, 1.0, math.sin(angle), math.cos(angle)], dtype=np.float32)
    raise ValueError(f"Expected pose6 or legacy 4D alignment params at {path}, got shape {arr.shape}")


def load_mouth_alignment5(path: str | Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32).reshape(-1)
    if arr.shape[0] != 5:
        raise ValueError(
            f"Expected mouth alignment [center_x, center_y, scale, sin, cos] "
            f"at {path}, got shape {arr.shape}"
        )
    if not np.isfinite(arr).all() or arr[2] <= 0:
        raise ValueError(f"Invalid mouth alignment at {path}: {arr.tolist()}")
    return arr


def load_mouth_projection63(path: str | Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.shape[-1] != MOUTH_PROJECTION63_DIM:
        raise ValueError(
            f"Expected mouth projection [..., {MOUTH_PROJECTION63_DIM}] "
            f"at {path}, got shape {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite mouth projection at {path}")
    return arr


def geometry81_to_points_teeth(geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if geometry.shape[-1] != 81:
        raise ValueError(f"Expected geometry shape (..., 81), got {tuple(geometry.shape)}")
    points = geometry[..., :80].reshape(*geometry.shape[:-1], 40, 2)
    teeth = geometry[..., 80:81]
    return points, teeth


def canonical_points_to_roi_pixels(
    points: torch.Tensor,
    mouth_alignment5: torch.Tensor,
) -> torch.Tensor:
    """Invert canonical mouth alignment into ROI-space pixel coordinates.

    ``mouth_alignment5`` stores ``center_x, center_y, scale, sin(angle),
    cos(angle)`` in the same ROI pixel coordinate system as the target image.
    """

    if points.shape[-1] != 2:
        raise ValueError(
            f"Expected canonical points shape (..., N, 2), got {tuple(points.shape)}"
        )
    if mouth_alignment5.shape[-1] != 5:
        raise ValueError(
            "Expected mouth alignment shape (..., 5), "
            f"got {tuple(mouth_alignment5.shape)}"
        )
    if points.shape[:-2] != mouth_alignment5.shape[:-1]:
        raise ValueError(
            "Canonical point and mouth alignment batch shapes differ: "
            f"{tuple(points.shape[:-2])} vs {tuple(mouth_alignment5.shape[:-1])}"
        )

    alignment = mouth_alignment5.to(device=points.device, dtype=points.dtype)
    center = alignment[..., :2].unsqueeze(-2)
    scale = alignment[..., 2:3].abs().clamp_min(1e-6).unsqueeze(-1)
    sine = alignment[..., 3:4]
    cosine = alignment[..., 4:5]
    norm = torch.sqrt(sine.square() + cosine.square()).clamp_min(1e-6)
    sine = (sine / norm).unsqueeze(-2)
    cosine = (cosine / norm).unsqueeze(-2)

    x = points[..., 0:1] * cosine - points[..., 1:2] * sine
    y = points[..., 0:1] * sine + points[..., 1:2] * cosine
    return center + scale * torch.cat([x, y], dim=-1)


def frontal2d_points_to_roi_pixels(
    points: torch.Tensor,
    mouth_projection63: torch.Tensor,
) -> torch.Tensor:
    if points.shape[-2:] != (40, 2):
        raise ValueError(
            f"Expected frontal points shape (..., 40, 2), got {tuple(points.shape)}"
        )
    if mouth_projection63.shape[-1] != MOUTH_PROJECTION63_DIM:
        raise ValueError(
            f"Expected projection shape (..., {MOUTH_PROJECTION63_DIM}), "
            f"got {tuple(mouth_projection63.shape)}"
        )
    if points.shape[:-2] != mouth_projection63.shape[:-1]:
        raise ValueError(
            "Frontal point and projection batch shapes differ: "
            f"{tuple(points.shape[:-2])} vs "
            f"{tuple(mouth_projection63.shape[:-1])}"
        )

    projection = mouth_projection63.to(
        device=points.device,
        dtype=points.dtype,
    )
    rotation = projection[..., 0:9].reshape(*points.shape[:-2], 3, 3)
    origin = projection[..., 9:12].unsqueeze(-2)
    alignment = projection[..., 12:17]
    depth = projection[..., 17:57].unsqueeze(-1)
    affine = projection[..., 57:63].reshape(*points.shape[:-2], 2, 3)

    center = alignment[..., :2].unsqueeze(-2)
    scale = alignment[..., 2:3].abs().clamp_min(1e-6).unsqueeze(-1)
    sine = alignment[..., 3:4]
    cosine = alignment[..., 4:5]
    norm = torch.sqrt(sine.square() + cosine.square()).clamp_min(1e-6)
    sine = (sine / norm).unsqueeze(-2)
    cosine = (cosine / norm).unsqueeze(-2)
    x = points[..., 0:1] * cosine - points[..., 1:2] * sine
    y = points[..., 0:1] * sine + points[..., 1:2] * cosine
    frontal2d = center + scale * torch.cat([x, y], dim=-1)
    frontal3d = torch.cat(
        [frontal2d[..., 0:1], -frontal2d[..., 1:2], depth],
        dim=-1,
    )
    camera = torch.matmul(
        frontal3d,
        rotation.transpose(-1, -2),
    ) + origin
    original_xy = torch.cat(
        [camera[..., 0:1], -camera[..., 1:2]],
        dim=-1,
    )
    linear = affine[..., :2]
    translation = affine[..., 2].unsqueeze(-2)
    return torch.matmul(
        original_xy,
        linear.transpose(-1, -2),
    ) + translation


def pose6_to_euler(pose6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pose6.shape[-1] < 6:
        raise ValueError(f"Expected pose shape (..., 6), got {tuple(pose6.shape)}")
    yaw = torch.atan2(pose6[..., 0], pose6[..., 1])
    pitch = torch.atan2(pose6[..., 2], pose6[..., 3])
    roll = torch.atan2(pose6[..., 4], pose6[..., 5])
    return yaw, pitch, roll


def adapt_canonical_points_to_view(points: torch.Tensor, pose6: torch.Tensor) -> torch.Tensor:
    """Approximate view adaptation from canonical mouth points and angular pose.

    The operation is intentionally light-weight. It gives the renderer a
    pose-aware structural target without making the audio geometry regressor
    responsible for head pose.
    """

    yaw, pitch, roll = pose6_to_euler(pose6)
    center = points.mean(dim=-2, keepdim=True)
    local = points - center

    scale_x = torch.cos(yaw.abs()).clamp(0.55, 1.0).view(-1, 1)
    scale_y = torch.cos(pitch.abs()).clamp(0.65, 1.0).view(-1, 1)
    yaw_strength = torch.sin(yaw).view(-1, 1)
    pitch_strength = torch.sin(pitch).view(-1, 1)

    lx = local[..., 0]
    ly = local[..., 1]
    mouth_w = (lx.max(dim=-1, keepdim=True).values - lx.min(dim=-1, keepdim=True).values).clamp_min(1e-4)
    mouth_h = (ly.max(dim=-1, keepdim=True).values - ly.min(dim=-1, keepdim=True).values).clamp_min(1e-4)

    x = lx * scale_x + 0.12 * yaw_strength * (ly / mouth_h) * mouth_w
    y = ly * scale_y + 0.08 * pitch_strength * (lx / mouth_w) * mouth_h

    cos_r = torch.cos(roll).view(-1, 1)
    sin_r = torch.sin(roll).view(-1, 1)
    xr = x * cos_r - y * sin_r
    yr = x * sin_r + y * cos_r
    return torch.stack([xr, yr], dim=-1) + center


def points_to_pixels(
    points: torch.Tensor,
    image_size: int,
    geometry_scale: float = 0.92,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> torch.Tensor:
    """Map normalized geometry coordinates to ROI pixel coordinates."""

    scale = float(image_size - 1)
    x = (center_x + points[..., 0] * geometry_scale) * scale
    y = (center_y + points[..., 1] * geometry_scale) * scale
    return torch.stack([x, y], dim=-1)


def neutral_pose_sequence(length: int) -> np.ndarray:
    return np.repeat(NEUTRAL_POSE6[None, :], int(length), axis=0).astype(np.float32)
