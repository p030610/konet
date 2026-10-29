from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import (
    FLESH_INDICES,
    JAW_INDICES,
    LIP_INDICES,
    LOWER_FACE_INDICES,
    LOWER_FACE_POINT_COUNT,
)


def _groups() -> tuple[list[int], list[int], list[int]]:
    lookup = {landmark: index for index, landmark in enumerate(LOWER_FACE_INDICES)}
    return (
        [lookup[index] for index in LIP_INDICES],
        [lookup[index] for index in JAW_INDICES],
        [lookup[index] for index in FLESH_INDICES],
    )


class LowerFaceRasterizer(nn.Module):
    """Rasterize target structure and source-to-target lower-face motion."""

    condition_channels = 7

    def __init__(self, image_size: int = 384, blur_radius: int = 5) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.blur_radius = int(blur_radius)
        lips, jaw, flesh = _groups()
        self.register_buffer("lip_indices", torch.tensor(lips, dtype=torch.long), persistent=False)
        self.register_buffer("jaw_indices", torch.tensor(jaw, dtype=torch.long), persistent=False)
        self.register_buffer("flesh_indices", torch.tensor(flesh, dtype=torch.long), persistent=False)

    def _scatter(self, points: torch.Tensor, indices: torch.Tensor, values: torch.Tensor | None = None) -> torch.Tensor:
        batch = points.shape[0]
        selected = points.index_select(1, indices)
        x = selected[..., 0].round().long().clamp(0, self.image_size - 1)
        y = selected[..., 1].round().long().clamp(0, self.image_size - 1)
        flat_index = y * self.image_size + x
        output = points.new_zeros(batch, self.image_size * self.image_size)
        source = torch.ones_like(selected[..., 0]) if values is None else values.index_select(1, indices)
        output.scatter_add_(1, flat_index, source)
        output = output.view(batch, 1, self.image_size, self.image_size)
        radius = max(1, self.blur_radius)
        output = F.avg_pool2d(output, radius * 2 + 1, stride=1, padding=radius)
        return (output * float((radius * 2 + 1) ** 2)).clamp(-1.0, 1.0)

    def forward(self, source_points: torch.Tensor, target_points: torch.Tensor, allowed_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if source_points.shape[-2:] != (LOWER_FACE_POINT_COUNT, 2):
            raise ValueError(f"Invalid source points {tuple(source_points.shape)}")
        displacement = (target_points - source_points) / max(float(self.image_size), 1.0)
        all_indices = torch.arange(LOWER_FACE_POINT_COUNT, device=target_points.device)
        all_heat = self._scatter(target_points, all_indices)
        lip_heat = self._scatter(target_points, self.lip_indices)
        jaw_heat = self._scatter(target_points, self.jaw_indices)
        flesh_heat = self._scatter(target_points, self.flesh_indices)
        weight = self._scatter(target_points, all_indices).abs().clamp_min(1e-4)
        dx = self._scatter(target_points, all_indices, displacement[..., 0]) / weight
        dy = self._scatter(target_points, all_indices, displacement[..., 1]) / weight
        allowed = F.interpolate(allowed_mask.float(), (self.image_size, self.image_size), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        condition = torch.cat([all_heat, lip_heat, jaw_heat, flesh_heat, dx, dy, allowed], dim=1)
        return {"condition": condition, "lip_heat": lip_heat, "jaw_heat": jaw_heat, "allowed": allowed}


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(min(16, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, base: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConvBlock(in_channels, base),
                ConvBlock(base, base * 2, 2),
                ConvBlock(base * 2, base * 4, 2),
                ConvBlock(base * 4, base * 8, 2),
                ConvBlock(base * 8, base * 8, 2),
            ]
        )

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        result = []
        for block in self.blocks:
            value = block(value)
            result.append(value)
        return result


class LowerFaceGenerator(nn.Module):
    """Hybrid deformation and synthesis model for speech-driven lower faces."""

    architecture_version = "lower_face_generator_v1"

    def __init__(
        self,
        image_size: int = 384,
        base_channels: int = 40,
        reference_count: int = 5,
        blendshape_dim: int = 52,
        audio_dim: int = 768,
        max_flow_pixels: float = 14.0,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.reference_count = int(reference_count)
        self.max_flow_pixels = float(max_flow_pixels)
        self.rasterizer = LowerFaceRasterizer(image_size=image_size)
        self.source_encoder = Encoder(5, base_channels)
        self.reference_encoder = Encoder(3, base_channels)
        self.condition_encoder = Encoder(self.rasterizer.condition_channels, base_channels)
        vector_dim = 81 + 6 + blendshape_dim + audio_dim
        bottleneck = base_channels * 8
        self.condition_vector = nn.Sequential(
            nn.Linear(vector_dim, bottleneck * 2),
            nn.SiLU(inplace=True),
            nn.Linear(bottleneck * 2, bottleneck * 2),
        )
        self.decode4 = ConvBlock(bottleneck * 3, bottleneck)
        self.decode3 = ConvBlock(bottleneck + base_channels * 24, base_channels * 8)
        self.decode2 = ConvBlock(base_channels * 8 + base_channels * 12, base_channels * 4)
        self.decode1 = ConvBlock(base_channels * 4 + base_channels * 6, base_channels * 2)
        self.decode0 = ConvBlock(base_channels * 2 + base_channels * 3, base_channels)
        self.output_head = nn.Conv2d(base_channels, 7, 3, 1, 1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    @staticmethod
    def _warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = image.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        grid = torch.stack([x, y], dim=-1)[None].expand(batch, -1, -1, -1)
        normalized = torch.stack(
            [2.0 * flow[:, 0] / max(width - 1, 1), 2.0 * flow[:, 1] / max(height - 1, 1)],
            dim=-1,
        )
        return F.grid_sample(image, grid + normalized, mode="bilinear", padding_mode="border", align_corners=True)

    def forward(
        self,
        source_face: torch.Tensor,
        reference_faces: torch.Tensor,
        allowed_mask: torch.Tensor,
        source_points_px: torch.Tensor,
        target_points_px: torch.Tensor,
        geometry81: torch.Tensor,
        pose6: torch.Tensor,
        target_blendshapes52: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if reference_faces.ndim != 5 or reference_faces.shape[1] != self.reference_count:
            raise ValueError(f"Expected references [B,{self.reference_count},3,H,W], got {tuple(reference_faces.shape)}")
        if audio_features.ndim == 3:
            audio_vector = audio_features.mean(dim=1)
        elif audio_features.ndim == 2:
            audio_vector = audio_features
        else:
            raise ValueError(f"Invalid audio shape {tuple(audio_features.shape)}")
        maps = self.rasterizer(source_points_px.float(), target_points_px.float(), allowed_mask)
        condition = maps["condition"].to(source_face.dtype)
        coordinates = torch.linspace(-1.0, 1.0, self.image_size, device=source_face.device, dtype=source_face.dtype)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        coords = torch.stack([xx, yy])[None].expand(len(source_face), -1, -1, -1)
        source_features = self.source_encoder(torch.cat([source_face, coords], dim=1))
        batch, refs = reference_faces.shape[:2]
        reference_flat = reference_faces.flatten(0, 1)
        reference_encoded = self.reference_encoder(reference_flat)
        reference_features = [feature.view(batch, refs, *feature.shape[1:]).mean(dim=1) for feature in reference_encoded]
        condition_features = self.condition_encoder(condition)

        vector = torch.cat([geometry81, pose6[:, :6], target_blendshapes52, audio_vector], dim=1)
        scale, bias = self.condition_vector(vector.float()).chunk(2, dim=1)
        scale = scale.to(source_face.dtype)[..., None, None]
        bias = bias.to(source_face.dtype)[..., None, None]
        bottleneck = source_features[-1] * (1.0 + 0.25 * torch.tanh(scale)) + 0.25 * torch.tanh(bias)
        value = self.decode4(torch.cat([bottleneck, reference_features[-1], condition_features[-1]], dim=1))
        skips = zip(
            (3, 2, 1, 0),
            (self.decode3, self.decode2, self.decode1, self.decode0),
        )
        for index, decoder in skips:
            value = F.interpolate(value, source_features[index].shape[-2:], mode="bilinear", align_corners=False)
            value = decoder(torch.cat([value, source_features[index], reference_features[index], condition_features[index]], dim=1))

        flow_logits, rgb_logits, confidence_logits, alpha_logits = torch.split(self.output_head(value), [2, 3, 1, 1], dim=1)
        flow = torch.tanh(flow_logits) * self.max_flow_pixels * maps["allowed"]
        warped_source = self._warp(source_face, flow)
        reference_rgb = reference_faces.mean(dim=1)
        synthesized = (reference_rgb + 0.35 * torch.tanh(rgb_logits)).clamp(0.0, 1.0)
        lip_core = (maps["lip_heat"] * 2.0).clamp(0.0, 1.0)
        confidence = torch.sigmoid(confidence_logits) * (1.0 - lip_core)
        completion = confidence * warped_source + (1.0 - confidence) * synthesized
        alpha = torch.maximum(lip_core, torch.sigmoid(alpha_logits) * maps["allowed"]).clamp(0.0, 1.0)
        output = source_face * (1.0 - alpha) + completion * alpha
        return {
            "output": output.clamp(0.0, 1.0),
            "completion": completion,
            "warped_source": warped_source,
            "synthesized": synthesized,
            "flow": flow,
            "confidence": confidence,
            "alpha": alpha,
            **maps,
        }


__all__ = ["LowerFaceGenerator", "LowerFaceRasterizer"]
