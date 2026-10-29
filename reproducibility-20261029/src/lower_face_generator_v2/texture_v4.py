from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .structure_v4 import LowerFaceStructureV4


class TextureBlockV4(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.body(value)


class TexturePyramidV4(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            TextureBlockV4(in_channels, base_channels),
            TextureBlockV4(base_channels, base_channels * 2, 2),
            TextureBlockV4(base_channels * 2, base_channels * 4, 2),
            TextureBlockV4(base_channels * 4, base_channels * 8, 2),
        ])

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        result = []
        for block in self.blocks:
            value = block(value)
            result.append(value)
        return result


class GeometryIndexedReferenceSamplerV4(nn.Module):
    """Gather reference tokens at landmarks and splat them at target landmarks."""

    def __init__(self, sigma_px: float = 14.0, reference_temperature: float = 0.02) -> None:
        super().__init__()
        self.sigma_px = float(sigma_px)
        self.reference_temperature = float(reference_temperature)

    def forward(
        self,
        reference_features: torch.Tensor,
        reference_points_px: torch.Tensor,
        target_points_px: torch.Tensor,
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, references, channels, height, width = reference_features.shape
        image_height, image_width = image_size
        points = reference_points_px.shape[2]
        reference_grid = reference_points_px.float().clone()
        reference_grid[..., 0] = reference_grid[..., 0] / max(image_width - 1, 1) * 2.0 - 1.0
        reference_grid[..., 1] = reference_grid[..., 1] / max(image_height - 1, 1) * 2.0 - 1.0
        sampled = F.grid_sample(
            reference_features.flatten(0, 1),
            reference_grid.flatten(0, 1)[:, :, None].to(reference_features.dtype),
            mode="bilinear", padding_mode="border", align_corners=True,
        )[:, :, :, 0].view(batch, references, channels, points)

        target_normalized = target_points_px.float().clone()
        target_normalized[..., 0] /= max(image_width - 1, 1)
        target_normalized[..., 1] /= max(image_height - 1, 1)
        reference_normalized = reference_points_px.float().clone()
        reference_normalized[..., 0] /= max(image_width - 1, 1)
        reference_normalized[..., 1] /= max(image_height - 1, 1)
        distance2 = (reference_normalized - target_normalized[:, None]).square().sum(-1)
        reference_weights = torch.softmax(
            -distance2 / max(self.reference_temperature, 1e-6), dim=1
        )
        tokens = torch.sum(sampled * reference_weights[:, :, None].to(sampled.dtype), dim=1)

        target_x = target_points_px[..., 0] / max(image_width - 1, 1) * max(width - 1, 1)
        target_y = target_points_px[..., 1] / max(image_height - 1, 1) * max(height - 1, 1)
        y = torch.arange(height, device=tokens.device, dtype=torch.float32)
        x = torch.arange(width, device=tokens.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        sigma = max(self.sigma_px * width / max(image_width, 1), 0.5)
        support = torch.exp(-(
            (xx[None, None] - target_x[..., None, None]).square()
            + (yy[None, None] - target_y[..., None, None]).square()
        ) / (2.0 * sigma ** 2))
        normalizer = support.sum(1, keepdim=True).clamp_min(1e-5)
        aligned = torch.einsum("bcp,bphw->bchw", tokens.float(), support) / normalizer
        confidence = (1.0 - torch.exp(-normalizer)).to(reference_features.dtype)
        return aligned.to(reference_features.dtype), confidence, reference_weights

    def warp_rgb(
        self,
        references_rgb: torch.Tensor,
        reference_points_px: torch.Tensor,
        target_points_px: torch.Tensor,
        flow_size: tuple[int, int] = (64, 80),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dense landmark warp that preserves full-resolution reference detail."""
        batch, references, _, image_height, image_width = references_rgb.shape
        flow_height, flow_width = flow_size
        target_x = target_points_px[..., 0] / max(image_width - 1, 1) * max(flow_width - 1, 1)
        target_y = target_points_px[..., 1] / max(image_height - 1, 1) * max(flow_height - 1, 1)
        y = torch.arange(flow_height, device=references_rgb.device, dtype=torch.float32)
        x = torch.arange(flow_width, device=references_rgb.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        sigma = max(self.sigma_px * flow_width / max(image_width, 1), 0.5)
        support = torch.exp(-(
            (xx[None, None] - target_x[..., None, None]).square()
            + (yy[None, None] - target_y[..., None, None]).square()
        ) / (2.0 * sigma ** 2))
        normalizer = support.sum(1, keepdim=True).clamp_min(1e-5)

        displacement = reference_points_px.float() - target_points_px[:, None].float()
        flow = torch.einsum("brpd,bphw->brdhw", displacement, support) / normalizer[:, None]
        flow = F.interpolate(
            flow.flatten(0, 1), (image_height, image_width), mode="bilinear", align_corners=False
        ).view(batch, references, 2, image_height, image_width)
        full_y = torch.linspace(-1.0, 1.0, image_height, device=references_rgb.device)
        full_x = torch.linspace(-1.0, 1.0, image_width, device=references_rgb.device)
        grid_y, grid_x = torch.meshgrid(full_y, full_x, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)[None, None]
        normalized_flow = torch.stack([
            2.0 * flow[:, :, 0] / max(image_width - 1, 1),
            2.0 * flow[:, :, 1] / max(image_height - 1, 1),
        ], dim=-1)
        sampling_grid = base_grid + normalized_flow
        warped = F.grid_sample(
            references_rgb.flatten(0, 1),
            sampling_grid.flatten(0, 1).to(references_rgb.dtype),
            mode="bilinear", padding_mode="border", align_corners=True,
        ).view(batch, references, 3, image_height, image_width)

        target_normalized = target_points_px.float().clone()
        target_normalized[..., 0] /= max(image_width - 1, 1)
        target_normalized[..., 1] /= max(image_height - 1, 1)
        reference_normalized = reference_points_px.float().clone()
        reference_normalized[..., 0] /= max(image_width - 1, 1)
        reference_normalized[..., 1] /= max(image_height - 1, 1)
        distance2 = (reference_normalized - target_normalized[:, None]).square().sum(-1)
        point_reference_weights = torch.softmax(
            -distance2 / max(self.reference_temperature, 1e-6), dim=1
        )
        spatial_weights = torch.einsum(
            "brp,bphw->brhw", point_reference_weights, support
        ) / normalizer
        spatial_weights = F.interpolate(
            spatial_weights, (image_height, image_width), mode="bilinear", align_corners=False
        )
        spatial_weights = spatial_weights / spatial_weights.sum(1, keepdim=True).clamp_min(1e-5)
        fused = torch.sum(warped * spatial_weights[:, :, None].to(warped.dtype), dim=1)
        confidence = (1.0 - torch.exp(-normalizer)).to(references_rgb.dtype)
        return fused, confidence, point_reference_weights


@dataclass(frozen=True)
class LowerFaceTextureV4Config:
    image_height: int = 256
    image_width: int = 320
    base_channels: int = 32
    reference_count: int = 5
    reference_sigma_px: float = 14.0
    reference_temperature: float = 0.02
    residual_scale: float = 0.25
    reference_skip_init_logit: float = 2.2


class LowerFaceTextureV4(nn.Module):
    """V4B geometry-indexed RGB decoder with frozen V4A spatial authority."""

    architecture_version = "lower_face_texture_v4b"

    def __init__(self, structure: LowerFaceStructureV4, config: LowerFaceTextureV4Config | None = None) -> None:
        super().__init__()
        self.config = config or LowerFaceTextureV4Config()
        self.structure = structure.eval().requires_grad_(False)
        base = self.config.base_channels
        self.source_encoder = TexturePyramidV4(9, base)
        self.condition_encoder = TexturePyramidV4(11, base)
        self.reference_encoder = TexturePyramidV4(3, base)
        self.reference_sampler = GeometryIndexedReferenceSamplerV4(
            self.config.reference_sigma_px, self.config.reference_temperature
        )
        self.fuse3 = TextureBlockV4(base * 24, base * 8)
        self.decode2 = TextureBlockV4(base * 20, base * 4)
        self.decode1 = TextureBlockV4(base * 8, base * 2)
        self.decode0 = TextureBlockV4(base * 4, base)
        self.output_head = nn.Conv2d(base, 4, 3, 1, 1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.structure.eval()
        return self

    def forward(
        self, *, source_rgb: torch.Tensor, source_masked: torch.Tensor,
        references_rgb: torch.Tensor, target_points_px: torch.Tensor,
        reference_points_px: torch.Tensor, geometry81: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        if references_rgb.shape[1] != cfg.reference_count:
            raise ValueError(f"Expected {cfg.reference_count} references, got {references_rgb.shape[1]}")
        with torch.no_grad():
            structure = self.structure(target_points_px=target_points_px, geometry81=geometry81)
        structure_maps = torch.cat([
            structure[name].detach() for name in ("lip_oral", "motion", "edit", "confidence")
        ], dim=1)
        y = torch.linspace(-1, 1, cfg.image_height, device=source_rgb.device, dtype=source_rgb.dtype)
        x = torch.linspace(-1, 1, cfg.image_width, device=source_rgb.device, dtype=source_rgb.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([xx, yy])[None].expand(len(source_rgb), -1, -1, -1)
        source_features = self.source_encoder(torch.cat([
            source_masked, structure_maps.to(source_rgb.dtype), coordinates
        ], dim=1))
        condition_features = self.condition_encoder(torch.cat([
            structure["geometry_fields"].to(source_rgb.dtype), structure_maps.to(source_rgb.dtype)
        ], dim=1))
        batch, references = references_rgb.shape[:2]
        encoded_reference = self.reference_encoder(references_rgb.flatten(0, 1))
        encoded_reference = [value.view(batch, references, *value.shape[1:]) for value in encoded_reference]
        aligned2, confidence2, reference_weights = self.reference_sampler(
            encoded_reference[2], reference_points_px, target_points_px,
            (cfg.image_height, cfg.image_width),
        )
        aligned3, confidence3, _ = self.reference_sampler(
            encoded_reference[3], reference_points_px, target_points_px,
            (cfg.image_height, cfg.image_width),
        )
        aligned_rgb, rgb_confidence, _ = self.reference_sampler.warp_rgb(
            references_rgb, reference_points_px, target_points_px,
            (cfg.image_height // 4, cfg.image_width // 4),
        )
        value = self.fuse3(torch.cat([source_features[3], condition_features[3], aligned3], dim=1))
        value = F.interpolate(value, source_features[2].shape[-2:], mode="bilinear", align_corners=False)
        value = self.decode2(torch.cat([value, source_features[2], condition_features[2], aligned2], dim=1))
        value = F.interpolate(value, source_features[1].shape[-2:], mode="bilinear", align_corners=False)
        value = self.decode1(torch.cat([value, source_features[1], condition_features[1]], dim=1))
        value = F.interpolate(value, source_features[0].shape[-2:], mode="bilinear", align_corners=False)
        value = self.decode0(torch.cat([value, source_features[0], condition_features[0]], dim=1))
        completion_logits, skip_logits = self.output_head(value).split([3, 1], dim=1)
        direct_completion = torch.sigmoid(completion_logits)
        # Preserve sharp identity texture on static lower-face skin, let the
        # direct branch own moving perioral flesh, and re-enable warped detail
        # inside the target lip/oral support.
        semantic_reference_envelope = (
            1.0
            - 0.85 * structure["motion"].detach()
            + 0.70 * structure["lip_oral"].detach()
        ).clamp(0.15, 1.0)
        reference_skip = (
            torch.sigmoid(skip_logits + cfg.reference_skip_init_logit)
            * semantic_reference_envelope.to(skip_logits.dtype)
        )
        completion = (
            aligned_rgb * reference_skip + direct_completion * (1.0 - reference_skip)
        ).clamp(0.0, 1.0)
        alpha = structure["edit"].detach().to(source_rgb.dtype)
        output = source_rgb * (1.0 - alpha) + completion * alpha
        return {
            "output": output.clamp(0.0, 1.0), "completion": completion, "alpha": alpha,
            "aligned_reference_rgb": aligned_rgb, "reference_weights": reference_weights,
            "reference_confidence_quarter": confidence2, "reference_confidence_eighth": confidence3,
            "rgb_reference_confidence": rgb_confidence,
            "direct_completion": direct_completion, "reference_skip": reference_skip,
            "detail_confidence": 1.0 - reference_skip, "structure": structure,
        }


def lower_face_texture_v4_loss(prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output, target = prediction["output"], batch["target_rgb"]
    edit, lip = batch["edit_mask"], batch["lip_oral_mask"]
    weight = edit * (1.0 + 4.0 * lip + 1.5 * batch["motion_mask"])
    rgb = (torch.sqrt((output - target).square() + 1e-6) * weight).sum() / (weight.sum() * 3).clamp_min(1.0)
    output_dx, output_dy = output[..., :, 1:] - output[..., :, :-1], output[..., 1:, :] - output[..., :-1, :]
    target_dx, target_dy = target[..., :, 1:] - target[..., :, :-1], target[..., 1:, :] - target[..., :-1, :]
    edge = ((output_dx - target_dx).abs() * lip[..., :, 1:]).sum() / (lip[..., :, 1:].sum() * 3).clamp_min(1.0)
    edge = edge + ((output_dy - target_dy).abs() * lip[..., 1:, :]).sum() / (lip[..., 1:, :].sum() * 3).clamp_min(1.0)
    exterior = ((output - batch["source_rgb"]).abs() * (1.0 - edit)).mean()
    total = rgb + 0.35 * edge + 10.0 * exterior
    return total, {"total": total.detach(), "rgb": rgb.detach(), "edge": edge.detach(), "exterior": exterior.detach()}


__all__ = ["GeometryIndexedReferenceSamplerV4", "LowerFaceTextureV4", "LowerFaceTextureV4Config", "lower_face_texture_v4_loss"]
