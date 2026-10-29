from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lower_face_generator.geometry import LOWER_FACE_GEOMETRY_DIM

from .rasterizer import LowerFaceRasterizerV2


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(16, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class PyramidEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConvBlock(in_channels, base_channels),
                ConvBlock(base_channels, base_channels * 2, 2),
                ConvBlock(base_channels * 2, base_channels * 4, 2),
                ConvBlock(base_channels * 4, base_channels * 8, 2),
                ConvBlock(base_channels * 8, base_channels * 8, 2),
            ]
        )

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for block in self.blocks:
            value = block(value)
            features.append(value)
        return features


class BoundaryLowFrequencyCorrector(nn.Module):
    """Correct only low-frequency color near the source/generated seam."""

    def __init__(self, channels: int = 16, max_residual: float = 0.12) -> None:
        super().__init__()
        self.max_residual = float(max_residual)
        # source low, completion low, their difference, overlap and alpha target.
        self.body = nn.Sequential(
            nn.Conv2d(11, channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 3, 3, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        # Start with a conservative analytic color transfer while the residual
        # branch learns corrections that cannot be represented by a local mean.
        self.analytic_strength_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        source: torch.Tensor,
        completion: torch.Tensor,
        overlap: torch.Tensor,
        alpha_target: torch.Tensor,
        lip_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_low = F.avg_pool2d(source, 15, stride=1, padding=7)
        completion_low = F.avg_pool2d(completion, 15, stride=1, padding=7)
        difference = source_low - completion_low
        features = torch.cat(
            [source_low, completion_low, difference, overlap, alpha_target], dim=1
        )
        learned = F.avg_pool2d(
            torch.tanh(self.body(features)) * self.max_residual,
            9,
            stride=1,
            padding=4,
        )
        analytic = torch.sigmoid(self.analytic_strength_logit) * difference
        # The final alpha multiplies this correction once more. This envelope
        # therefore peaks within the transition and vanishes in the interior.
        support = overlap * (1.0 - alpha_target).clamp(0.0, 1.0)
        if lip_mask is not None:
            lip_guard = 1.0 - F.max_pool2d(lip_mask, 7, stride=1, padding=3)
            support = support * lip_guard
        correction = (analytic + learned) * support
        return (completion + correction).clamp(0.0, 1.0), correction, support


@dataclass(frozen=True)
class LowerFaceGeneratorV2Config:
    image_height: int = 256
    image_width: int = 320
    base_channels: int = 32
    reference_count: int = 5
    attention_dim: int = 128
    max_source_flow_px: float = 24.0
    max_reference_flow_px: float = 32.0
    alpha_feather_radius: int = 0
    boundary_low_frequency_correction: bool = False
    boundary_correction_channels: int = 16
    boundary_correction_max_residual: float = 0.12


class LowerFaceGeneratorV2(nn.Module):
    """GT-geometry lower-face deformation and completion renderer."""

    architecture_version = "lower_face_generator_v2"

    def __init__(self, config: LowerFaceGeneratorV2Config | None = None) -> None:
        super().__init__()
        self.config = config or LowerFaceGeneratorV2Config()
        cfg = self.config
        self.rasterizer = LowerFaceRasterizerV2((cfg.image_height, cfg.image_width))
        # RGB + edit/core/motion/ring/nose masks + x/y coordinates.
        self.source_encoder = PyramidEncoder(10, cfg.base_channels)
        self.reference_encoder = PyramidEncoder(3, cfg.base_channels)
        self.condition_encoder = PyramidEncoder(self.rasterizer.condition_channels, cfg.base_channels)

        reference_dim = LOWER_FACE_GEOMETRY_DIM + 81
        self.reference_key = nn.Sequential(
            nn.Linear(reference_dim, cfg.attention_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.attention_dim, cfg.attention_dim),
        )
        self.target_query = nn.Sequential(
            nn.Linear(reference_dim, cfg.attention_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cfg.attention_dim, cfg.attention_dim),
        )
        global_dim = LOWER_FACE_GEOMETRY_DIM * 2 + 81 + 19 + 6
        bottleneck = cfg.base_channels * 8
        self.global_condition = nn.Sequential(
            nn.Linear(global_dim, bottleneck * 2),
            nn.SiLU(inplace=True),
            nn.Linear(bottleneck * 2, bottleneck * 2),
        )

        self.fuse4 = ConvBlock(bottleneck * 3, bottleneck)
        self.decode3 = ConvBlock(bottleneck * 4, bottleneck)
        self.decode2 = ConvBlock(cfg.base_channels * 20, cfg.base_channels * 4)
        self.decode1 = ConvBlock(cfg.base_channels * 10, cfg.base_channels * 2)
        self.decode0 = ConvBlock(cfg.base_channels * 5, cfg.base_channels)
        # source flow 2, reference flow 2, RGB residual 3, occlusion 1,
        # confidence 1, alpha 1.
        self.output_head = nn.Conv2d(cfg.base_channels, 10, 3, 1, 1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        self.boundary_corrector = (
            BoundaryLowFrequencyCorrector(
                cfg.boundary_correction_channels,
                cfg.boundary_correction_max_residual,
            )
            if cfg.boundary_low_frequency_correction
            else None
        )

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
        return F.grid_sample(
            image, grid + normalized, mode="bilinear", padding_mode="border", align_corners=True
        )

    def _reference_features(
        self,
        references_rgb: torch.Tensor,
        reference_geometry_xyz: torch.Tensor,
        reference_geometry81: torch.Tensor,
        target_geometry_xyz: torch.Tensor,
        geometry81: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        batch, references = references_rgb.shape[:2]
        if references != self.config.reference_count:
            raise ValueError(f"Expected {self.config.reference_count} references, got {references}")
        reference_vector = torch.cat(
            [reference_geometry_xyz.flatten(2), reference_geometry81], dim=-1
        ).float()
        target_vector = torch.cat([target_geometry_xyz.flatten(1), geometry81], dim=-1).float()
        keys = F.normalize(self.reference_key(reference_vector), dim=-1)
        query = F.normalize(self.target_query(target_vector), dim=-1)
        logits = torch.einsum("brd,bd->br", keys, query) / 0.25
        weights = torch.softmax(logits, dim=1)

        encoded = self.reference_encoder(references_rgb.flatten(0, 1))
        fused = []
        for feature in encoded:
            feature = feature.view(batch, references, *feature.shape[1:])
            fused.append(torch.sum(feature * weights[:, :, None, None, None].to(feature.dtype), dim=1))
        reference_rgb = torch.sum(
            references_rgb * weights[:, :, None, None, None].to(references_rgb.dtype), dim=1
        )
        return fused, reference_rgb, weights

    def forward(
        self,
        *,
        source_rgb: torch.Tensor,
        source_masked: torch.Tensor,
        references_rgb: torch.Tensor,
        source_points_px: torch.Tensor,
        target_points_px: torch.Tensor,
        source_geometry_xyz: torch.Tensor,
        target_geometry_xyz: torch.Tensor,
        reference_geometry_xyz: torch.Tensor,
        geometry81: torch.Tensor,
        reference_geometry81: torch.Tensor,
        projection19: torch.Tensor,
        pose6: torch.Tensor,
        edit_mask: torch.Tensor,
        inpaint_core_mask: torch.Tensor,
        motion_mask: torch.Tensor,
        ring_mask: torch.Tensor,
        nose_exclusion_mask: torch.Tensor,
        alpha_core_mask: torch.Tensor | None = None,
        upper_overlap_mask: torch.Tensor | None = None,
        upper_alpha_target_mask: torch.Tensor | None = None,
        lip_oral_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        expected_image = (cfg.image_height, cfg.image_width)
        if source_rgb.shape[-2:] != expected_image:
            raise ValueError(f"Expected image size {expected_image}, got {tuple(source_rgb.shape[-2:])}")
        maps = self.rasterizer(source_points_px, target_points_px, edit_mask)
        condition = maps["condition"].to(source_rgb.dtype)
        y = torch.linspace(-1.0, 1.0, cfg.image_height, device=source_rgb.device, dtype=source_rgb.dtype)
        x = torch.linspace(-1.0, 1.0, cfg.image_width, device=source_rgb.device, dtype=source_rgb.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([xx, yy])[None].expand(len(source_rgb), -1, -1, -1)
        source_input = torch.cat(
            [
                source_masked,
                edit_mask,
                inpaint_core_mask,
                motion_mask,
                ring_mask,
                nose_exclusion_mask,
                coordinates,
            ],
            dim=1,
        )
        source_features = self.source_encoder(source_input)
        condition_features = self.condition_encoder(condition)
        reference_features, reference_rgb, reference_weights = self._reference_features(
            references_rgb,
            reference_geometry_xyz,
            reference_geometry81,
            target_geometry_xyz,
            geometry81,
        )

        global_vector = torch.cat(
            [
                source_geometry_xyz.flatten(1),
                target_geometry_xyz.flatten(1),
                geometry81,
                projection19,
                pose6,
            ],
            dim=1,
        )
        scale, bias = self.global_condition(global_vector.float()).chunk(2, dim=1)
        scale = scale.to(source_rgb.dtype)[..., None, None]
        bias = bias.to(source_rgb.dtype)[..., None, None]
        value = self.fuse4(
            torch.cat([source_features[-1], reference_features[-1], condition_features[-1]], dim=1)
        )
        value = value * (1.0 + 0.2 * torch.tanh(scale)) + 0.2 * torch.tanh(bias)
        decoders = (self.decode3, self.decode2, self.decode1, self.decode0)
        for feature_index, decoder in zip((3, 2, 1, 0), decoders):
            value = F.interpolate(
                value, source_features[feature_index].shape[-2:], mode="bilinear", align_corners=False
            )
            value = decoder(
                torch.cat(
                    [
                        value,
                        source_features[feature_index],
                        reference_features[feature_index],
                        condition_features[feature_index],
                    ],
                    dim=1,
                )
            )

        outputs = torch.split(self.output_head(value), [2, 2, 3, 1, 1, 1], dim=1)
        source_flow_logits, reference_flow_logits, residual_logits, occlusion_logits, confidence_logits, alpha_logits = outputs
        allowed = edit_mask * (1.0 - nose_exclusion_mask)
        source_flow = torch.tanh(source_flow_logits) * cfg.max_source_flow_px * allowed
        reference_flow = torch.tanh(reference_flow_logits) * cfg.max_reference_flow_px * allowed
        warped_source = self._warp(source_rgb, source_flow)
        warped_reference = self._warp(reference_rgb, reference_flow)
        synthesized = (warped_reference + 0.35 * torch.tanh(residual_logits)).clamp(0.0, 1.0)
        occlusion = torch.maximum(inpaint_core_mask, torch.sigmoid(occlusion_logits) * allowed)
        confidence = torch.sigmoid(confidence_logits) * allowed
        completion = warped_source * (1.0 - occlusion) + synthesized * occlusion
        completion = completion * (0.75 + 0.25 * confidence) + synthesized * (0.25 * (1.0 - confidence))
        alpha_transition = torch.sigmoid(alpha_logits) * allowed
        opaque_core = inpaint_core_mask if alpha_core_mask is None else alpha_core_mask
        if cfg.alpha_feather_radius > 0:
            radius = int(cfg.alpha_feather_radius)
            opaque_core = 1.0 - F.max_pool2d(
                1.0 - opaque_core, radius * 2 + 1, stride=1, padding=radius
            )
        alpha = torch.maximum(opaque_core, alpha_transition * (edit_mask - opaque_core).clamp_min(0.0))
        alpha = alpha * (1.0 - nose_exclusion_mask)
        if upper_overlap_mask is not None and upper_alpha_target_mask is not None:
            deterministic = upper_overlap_mask * (1.0 - nose_exclusion_mask)
            alpha = alpha * (1.0 - deterministic) + upper_alpha_target_mask * deterministic
        boundary_correction = torch.zeros_like(completion)
        boundary_correction_support = torch.zeros_like(alpha)
        if (
            self.boundary_corrector is not None
            and upper_overlap_mask is not None
            and upper_alpha_target_mask is not None
        ):
            completion, boundary_correction, boundary_correction_support = self.boundary_corrector(
                source_rgb,
                completion,
                upper_overlap_mask * (1.0 - nose_exclusion_mask),
                upper_alpha_target_mask,
                lip_oral_mask,
            )
        final = source_rgb * (1.0 - alpha) + completion * alpha
        return {
            "output": final.clamp(0.0, 1.0),
            "completion": completion.clamp(0.0, 1.0),
            "warped_source": warped_source,
            "warped_reference": warped_reference,
            "synthesized": synthesized,
            "source_flow": source_flow,
            "reference_flow": reference_flow,
            "occlusion": occlusion,
            "confidence": confidence,
            "alpha": alpha,
            "opaque_core": opaque_core,
            "boundary_correction": boundary_correction,
            "boundary_correction_support": boundary_correction_support,
            "reference_weights": reference_weights,
            **maps,
        }


__all__ = ["LowerFaceGeneratorV2", "LowerFaceGeneratorV2Config"]
