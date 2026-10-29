from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lower_face_generator.geometry import (
    FLESH_INDICES,
    JAW_INDICES,
    LIP_INDICES,
    LOWER_FACE_INDICES,
    LOWER_FACE_POINT_COUNT,
    project_lower_face_torch,
)


def _group_positions(indices: tuple[int, ...]) -> torch.Tensor:
    lookup = {landmark: index for index, landmark in enumerate(LOWER_FACE_INDICES)}
    return torch.tensor([lookup[index] for index in indices], dtype=torch.long)


def project_geometry_to_crop(
    geometry_xyz: torch.Tensor,
    projection19: torch.Tensor,
    face_to_crop_affine: torch.Tensor,
) -> torch.Tensor:
    """Replay frontalized geometry into rectangular crop pixels."""

    face_points = project_lower_face_torch(geometry_xyz, projection19)
    affine = face_to_crop_affine.reshape(*face_points.shape[:-2], 2, 3)
    return (
        torch.matmul(face_points, affine[..., :2].transpose(-1, -2))
        + affine[..., 2].unsqueeze(-2)
    )


class LowerFaceRasterizerV2(nn.Module):
    """Rectangular, differentiable lower-face geometry rasterizer."""

    condition_channels = 8

    def __init__(self, image_size: tuple[int, int] = (256, 320), sigma: float = 2.5) -> None:
        super().__init__()
        self.height = int(image_size[0])
        self.width = int(image_size[1])
        self.sigma = float(sigma)
        self.register_buffer("lip_indices", _group_positions(LIP_INDICES), persistent=False)
        self.register_buffer("jaw_indices", _group_positions(JAW_INDICES), persistent=False)
        self.register_buffer("flesh_indices", _group_positions(FLESH_INDICES), persistent=False)
        self.register_buffer(
            "all_indices", torch.arange(LOWER_FACE_POINT_COUNT, dtype=torch.long), persistent=False
        )

    def _splat(
        self,
        points: torch.Tensor,
        indices: torch.Tensor,
        values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selected = points.index_select(1, indices)
        batch, count = selected.shape[:2]
        x = selected[..., 0]
        y = selected[..., 1]
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        fx = x - x0
        fy = y - y0
        base_values = torch.ones_like(x) if values is None else values.index_select(1, indices)
        output = points.new_zeros(batch, self.height * self.width)
        for ox, oy, weight in (
            (0, 0, (1.0 - fx) * (1.0 - fy)),
            (1, 0, fx * (1.0 - fy)),
            (0, 1, (1.0 - fx) * fy),
            (1, 1, fx * fy),
        ):
            xi = (x0 + ox).long()
            yi = (y0 + oy).long()
            valid = (xi >= 0) & (xi < self.width) & (yi >= 0) & (yi < self.height)
            flat = yi.clamp(0, self.height - 1) * self.width + xi.clamp(0, self.width - 1)
            output.scatter_add_(1, flat, base_values * weight * valid.to(points.dtype))
        output = output.view(batch, 1, self.height, self.width)
        if self.sigma > 0:
            radius = max(1, int(round(self.sigma * 3.0)))
            coordinates = torch.arange(-radius, radius + 1, device=points.device, dtype=points.dtype)
            kernel = torch.exp(-0.5 * (coordinates / self.sigma) ** 2)
            kernel = kernel / kernel.sum()
            output = F.conv2d(output, kernel.view(1, 1, 1, -1), padding=(0, radius))
            output = F.conv2d(output, kernel.view(1, 1, -1, 1), padding=(radius, 0))
        return output

    def forward(
        self,
        source_points_px: torch.Tensor,
        target_points_px: torch.Tensor,
        allowed_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected = (LOWER_FACE_POINT_COUNT, 2)
        if source_points_px.shape[-2:] != expected or target_points_px.shape[-2:] != expected:
            raise ValueError(
                f"Expected source/target points [...,{expected[0]},2], got "
                f"{tuple(source_points_px.shape)} and {tuple(target_points_px.shape)}"
            )
        if source_points_px.ndim != 3 or target_points_px.ndim != 3:
            raise ValueError("Rasterizer expects batched point tensors [B,93,2]")
        source = source_points_px.float()
        target = target_points_px.float()
        displacement = target - source
        source_all = self._splat(source, self.all_indices)
        target_all = self._splat(target, self.all_indices)
        lip = self._splat(target, self.lip_indices)
        jaw = self._splat(target, self.jaw_indices)
        flesh = self._splat(target, self.flesh_indices)
        support = target_all.clamp_min(1e-6)
        dx = self._splat(target, self.all_indices, displacement[..., 0] / max(self.width, 1)) / support
        dy = self._splat(target, self.all_indices, displacement[..., 1] / max(self.height, 1)) / support
        allowed = F.interpolate(
            allowed_mask.float(), (self.height, self.width), mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        condition = torch.cat([target_all, lip, jaw, flesh, source_all, dx, dy, allowed], dim=1)
        return {
            "condition": condition,
            "target_heat": target_all,
            "source_heat": source_all,
            "lip_heat": lip,
            "jaw_heat": jaw,
            "flesh_heat": flesh,
            "dx": dx,
            "dy": dy,
            "allowed": allowed,
        }


__all__ = ["LowerFaceRasterizerV2", "project_geometry_to_crop"]
