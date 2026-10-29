from __future__ import annotations

import torch
import torch.nn as nn

from .geometry import (
    adapt_canonical_points_to_view,
    canonical_points_to_roi_pixels,
    frontal2d_points_to_roi_pixels,
    geometry81_to_points_teeth,
    points_to_pixels,
    pose6_to_euler,
)


OUTER_LIP_ORDER = (0, 4, 5, 6, 20, 7, 17, 15, 19, 12, 3, 30, 37, 33, 35, 25, 38, 24, 23, 22)
INNER_LIP_ORDER = (1, 11, 10, 9, 21, 8, 16, 14, 18, 13, 2, 31, 36, 32, 34, 26, 39, 27, 28, 29)
OUTER_LIP_NODES = tuple(sorted(OUTER_LIP_ORDER))
INNER_LIP_NODES = tuple(index for index in range(40) if index not in OUTER_LIP_NODES)


class GeometryRasterizer(nn.Module):
    """Convert canonical geometry into ROI-space spatial conditioning maps."""

    def __init__(
        self,
        image_size: int = 192,
        sigma: float = 2.4,
        geometry_scale: float = 0.92,
        outer_points: int = 20,
        use_mouth_topology: bool = False,
        include_lip_mask_condition: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.sigma = float(sigma)
        self.geometry_scale = float(geometry_scale)
        self.outer_points = int(outer_points)
        self.use_mouth_topology = bool(use_mouth_topology)
        self.include_lip_mask_condition = bool(include_lip_mask_condition)

        y = torch.arange(self.image_size, dtype=torch.float32)
        x = torch.arange(self.image_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("grid_x", xx[None, None, :, :], persistent=False)
        self.register_buffer("grid_y", yy[None, None, :, :], persistent=False)

    @property
    def condition_channels(self) -> int:
        return 9 if self.include_lip_mask_condition else 8

    def _point_heatmap(self, pixels: torch.Tensor) -> torch.Tensor:
        px = pixels[..., 0][:, :, None, None]
        py = pixels[..., 1][:, :, None, None]
        dist2 = (self.grid_x - px).square() + (self.grid_y - py).square()
        heat = torch.exp(-dist2 / (2.0 * self.sigma * self.sigma))
        return heat.amax(dim=1, keepdim=True)

    def _bbox_mask(self, pixels: torch.Tensor) -> torch.Tensor:
        margin = 7.0
        softness = 2.5
        min_xy = pixels.amin(dim=1)
        max_xy = pixels.amax(dim=1)
        left = (min_xy[:, 0] - margin)[:, None, None, None]
        right = (max_xy[:, 0] + margin)[:, None, None, None]
        top = (min_xy[:, 1] - margin)[:, None, None, None]
        bottom = (max_xy[:, 1] + margin)[:, None, None, None]
        inside_x = torch.sigmoid((self.grid_x - left) / softness) * torch.sigmoid((right - self.grid_x) / softness)
        inside_y = torch.sigmoid((self.grid_y - top) / softness) * torch.sigmoid((bottom - self.grid_y) / softness)
        return inside_x * inside_y

    def _topology_mask(self, pixels: torch.Tensor, order: tuple[int, ...]) -> torch.Tensor:
        """Differentiable soft mask for an ordered, approximately convex contour."""
        indices = torch.as_tensor(order, device=pixels.device)
        polygon = pixels.index_select(1, indices)
        next_polygon = torch.roll(polygon, shifts=-1, dims=1)
        edges = next_polygon - polygon
        signed_area = (
            polygon[..., 0] * next_polygon[..., 1]
            - next_polygon[..., 0] * polygon[..., 1]
        ).sum(dim=1)
        orientation = torch.where(signed_area >= 0, 1.0, -1.0)[:, None, None, None]
        px = polygon[..., 0][:, :, None, None]
        py = polygon[..., 1][:, :, None, None]
        ex = edges[..., 0][:, :, None, None]
        ey = edges[..., 1][:, :, None, None]
        cross = ex * (self.grid_y - py) - ey * (self.grid_x - px)
        distance = cross * orientation / torch.sqrt(ex.square() + ey.square() + 1e-6)
        inside_distance = distance.amin(dim=1, keepdim=True)
        return torch.sigmoid((inside_distance + 1.5) / 1.25)

    def forward(
        self,
        geometry81: torch.Tensor,
        pose6: torch.Tensor,
        mouth_alignment5: torch.Tensor | None = None,
        mouth_projection63: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        points, teeth = geometry81_to_points_teeth(geometry81)
        if mouth_projection63 is not None:
            view_points = points
            pixels = frontal2d_points_to_roi_pixels(
                points,
                mouth_projection63,
            )
        elif mouth_alignment5 is None:
            view_points = adapt_canonical_points_to_view(points, pose6)
            pixels = points_to_pixels(
                view_points,
                self.image_size,
                geometry_scale=self.geometry_scale,
            )
        else:
            view_points = points
            pixels = canonical_points_to_roi_pixels(points, mouth_alignment5)

        if self.use_mouth_topology:
            outer_indices = torch.as_tensor(OUTER_LIP_NODES, device=pixels.device)
            inner_indices = torch.as_tensor(INNER_LIP_NODES, device=pixels.device)
            outer_heat = self._point_heatmap(pixels.index_select(1, outer_indices))
            inner_heat = self._point_heatmap(pixels.index_select(1, inner_indices))
        else:
            split = min(max(self.outer_points, 1), pixels.shape[1] - 1)
            outer_heat = self._point_heatmap(pixels[:, :split])
            inner_heat = self._point_heatmap(pixels[:, split:])
        all_heat = self._point_heatmap(pixels)
        mouth_mask = self._bbox_mask(pixels)
        lip_mask = self._topology_mask(pixels, OUTER_LIP_ORDER) if self.use_mouth_topology else mouth_mask
        inner_mouth_mask = (
            self._topology_mask(pixels, INNER_LIP_ORDER) if self.use_mouth_topology else mouth_mask
        )
        inner_mouth_mask = torch.minimum(inner_mouth_mask, lip_mask)

        bsz = geometry81.shape[0]
        h = w = self.image_size
        teeth_map = teeth.clamp(0.0, 1.0).view(bsz, 1, 1, 1).expand(bsz, 1, h, w)

        yaw, pitch, roll = pose6_to_euler(pose6)
        yaw_map = (yaw / torch.pi).view(bsz, 1, 1, 1).expand(bsz, 1, h, w)
        pitch_map = (pitch / torch.pi).view(bsz, 1, 1, 1).expand(bsz, 1, h, w)
        roll_map = (roll / torch.pi).view(bsz, 1, 1, 1).expand(bsz, 1, h, w)

        condition_parts = [outer_heat, inner_heat, all_heat, mouth_mask]
        if self.include_lip_mask_condition:
            condition_parts.append(lip_mask)
        condition_parts.extend([teeth_map, yaw_map, pitch_map, roll_map])
        condition = torch.cat(condition_parts, dim=1)
        return {
            "condition": condition,
            "view_points": view_points,
            "pixel_points": pixels,
            "mouth_mask": mouth_mask,
            "lip_mask": lip_mask,
            "inner_mouth_mask": inner_mouth_mask,
            "teeth_map": teeth_map,
        }
