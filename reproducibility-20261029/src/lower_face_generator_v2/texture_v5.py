from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rasterizer import LowerFaceRasterizerV2
from .texture_v4 import (
    GeometryIndexedReferenceSamplerV4,
    TextureBlockV4,
    TexturePyramidV4,
)


@dataclass(frozen=True)
class LowerFaceTextureV5Config:
    image_height: int = 256
    image_width: int = 320
    base_channels: int = 24
    reference_count: int = 5
    reference_sigma_px: float = 14.0
    reference_temperature: float = 0.02
    reference_residual_scale: float = 0.20


class _DirectDecoderV5(nn.Module):
    def __init__(self, base: int) -> None:
        super().__init__()
        self.fuse3 = TextureBlockV4(base * 16, base * 8)
        self.decode2 = TextureBlockV4(base * 16, base * 4)
        self.decode1 = TextureBlockV4(base * 8, base * 2)
        self.decode0 = TextureBlockV4(base * 4, base)

    def forward(
        self, source: list[torch.Tensor], condition: list[torch.Tensor]
    ) -> torch.Tensor:
        value = self.fuse3(torch.cat((source[3], condition[3]), dim=1))
        value = F.interpolate(value, source[2].shape[-2:], mode="bilinear", align_corners=False)
        value = self.decode2(torch.cat((value, source[2], condition[2]), dim=1))
        value = F.interpolate(value, source[1].shape[-2:], mode="bilinear", align_corners=False)
        value = self.decode1(torch.cat((value, source[1], condition[1]), dim=1))
        value = F.interpolate(value, source[0].shape[-2:], mode="bilinear", align_corners=False)
        return self.decode0(torch.cat((value, source[0], condition[0]), dim=1))


class _ReferenceDecoderV5(nn.Module):
    def __init__(self, base: int) -> None:
        super().__init__()
        self.fuse3 = TextureBlockV4(base * 24, base * 8)
        self.decode2 = TextureBlockV4(base * 20, base * 4)
        self.decode1 = TextureBlockV4(base * 8, base * 2)
        self.decode0 = TextureBlockV4(base * 4, base)

    def forward(
        self,
        source: list[torch.Tensor],
        condition: list[torch.Tensor],
        aligned2: torch.Tensor,
        aligned3: torch.Tensor,
    ) -> torch.Tensor:
        value = self.fuse3(torch.cat((source[3], condition[3], aligned3), dim=1))
        value = F.interpolate(value, source[2].shape[-2:], mode="bilinear", align_corners=False)
        value = self.decode2(torch.cat((value, source[2], condition[2], aligned2), dim=1))
        value = F.interpolate(value, source[1].shape[-2:], mode="bilinear", align_corners=False)
        value = self.decode1(torch.cat((value, source[1], condition[1]), dim=1))
        value = F.interpolate(value, source[0].shape[-2:], mode="bilinear", align_corners=False)
        return self.decode0(torch.cat((value, source[0], condition[0]), dim=1))


class LowerFaceTextureV5(nn.Module):
    """Static V5 RGB renderer with mutually exclusive pixel ownership."""

    architecture_version = "lower_face_texture_v5_exclusive"

    def __init__(self, config: LowerFaceTextureV5Config | None = None) -> None:
        super().__init__()
        self.config = config or LowerFaceTextureV5Config()
        cfg = self.config
        base = cfg.base_channels
        self.rasterizer = LowerFaceRasterizerV2((cfg.image_height, cfg.image_width))
        # masked source RGB + edit + alpha + source coordinates
        self.source_encoder = TexturePyramidV4(7, base)
        # target/lip/jaw/flesh heat, teeth, and seven exact V5 condition masks
        self.condition_encoder = TexturePyramidV4(12, base)
        self.reference_encoder = TexturePyramidV4(3, base)
        self.reference_sampler = GeometryIndexedReferenceSamplerV4(
            cfg.reference_sigma_px, cfg.reference_temperature
        )
        self.direct_decoder = _DirectDecoderV5(base)
        self.reference_decoder = _ReferenceDecoderV5(base)
        self.direct_head = nn.Conv2d(base, 3, 3, 1, 1)
        self.reference_residual_head = nn.Conv2d(base, 3, 3, 1, 1)
        self.transition_fuse = TextureBlockV4(base * 4, base)
        self.transition_head = nn.Conv2d(base, 3, 3, 1, 1)
        nn.init.zeros_(self.reference_residual_head.weight)
        nn.init.zeros_(self.reference_residual_head.bias)

    def _conditions(
        self,
        target_points_px: torch.Tensor,
        geometry81: torch.Tensor,
        masks: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        ones = target_points_px.new_ones(
            len(target_points_px), 1, self.config.image_height, self.config.image_width
        )
        raster = self.rasterizer(target_points_px.float(), target_points_px.float(), ones)
        teeth = geometry81[:, 80:81].float().view(-1, 1, 1, 1).expand_as(ones)
        return torch.cat(
            (
                raster["target_heat"],
                raster["lip_heat"],
                raster["jaw_heat"],
                raster["flesh_heat"],
                teeth,
                *masks,
            ),
            dim=1,
        )

    @staticmethod
    def _check_ownership(
        direct: torch.Tensor,
        reference: torch.Tensor,
        transition: torch.Tensor,
        edit: torch.Tensor,
        alpha: torch.Tensor,
        nose: torch.Tensor,
    ) -> None:
        partition = direct + reference + transition
        if torch.any(partition > 1.001):
            raise ValueError("V5 ownership masks overlap")
        if torch.any((edit > 0.5) & (partition < 0.999)):
            raise ValueError("V5 edit pixels are not fully assigned")
        if torch.any((edit < 0.5) & (partition > 0.001)):
            raise ValueError("V5 ownership escapes the edit region")
        if torch.any((nose > 0.5) & (alpha > 0)):
            raise ValueError("V5 alpha enters the nose exclusion")

    def forward(
        self,
        *,
        source_rgb: torch.Tensor,
        source_masked: torch.Tensor,
        references_skin_rgb: torch.Tensor,
        target_points_px: torch.Tensor,
        reference_points_px: torch.Tensor,
        geometry81: torch.Tensor,
        source_lip_support_mask: torch.Tensor,
        target_lip_support_mask: torch.Tensor,
        direct_rgb_ownership_mask: torch.Tensor,
        safe_reference_skin_mask: torch.Tensor,
        mouth_skin_transition_mask: torch.Tensor,
        edit_union_mask: torch.Tensor,
        alpha_target_mask: torch.Tensor,
        nose_exclusion_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        if references_skin_rgb.shape[1] != cfg.reference_count:
            raise ValueError(
                f"Expected {cfg.reference_count} skin references, got {references_skin_rgb.shape[1]}"
            )
        direct_owner = direct_rgb_ownership_mask.float()
        reference_owner = safe_reference_skin_mask.float()
        transition_owner = mouth_skin_transition_mask.float()
        edit = edit_union_mask.float()
        alpha = alpha_target_mask.float()
        nose = nose_exclusion_mask.float()
        self._check_ownership(
            direct_owner, reference_owner, transition_owner, edit, alpha, nose
        )
        y = torch.linspace(-1, 1, cfg.image_height, device=source_rgb.device, dtype=source_rgb.dtype)
        x = torch.linspace(-1, 1, cfg.image_width, device=source_rgb.device, dtype=source_rgb.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack((xx, yy))[None].expand(len(source_rgb), -1, -1, -1)
        source_features = self.source_encoder(
            torch.cat((source_masked, edit.to(source_rgb.dtype), alpha.to(source_rgb.dtype), coordinates), dim=1)
        )
        exact_masks = (
            source_lip_support_mask.float(),
            target_lip_support_mask.float(),
            direct_owner,
            reference_owner,
            transition_owner,
            edit,
            alpha,
        )
        condition_features = self.condition_encoder(
            self._conditions(target_points_px, geometry81, exact_masks).to(source_rgb.dtype)
        )

        # This branch is computed before, and independently of, all reference tensors.
        direct_features = self.direct_decoder(source_features, condition_features)
        direct_mouth_rgb = torch.sigmoid(self.direct_head(direct_features))

        batch, references = references_skin_rgb.shape[:2]
        encoded = self.reference_encoder(references_skin_rgb.flatten(0, 1))
        encoded = [value.view(batch, references, *value.shape[1:]) for value in encoded]
        aligned2, confidence2, reference_weights = self.reference_sampler(
            encoded[2], reference_points_px, target_points_px,
            (cfg.image_height, cfg.image_width),
        )
        aligned3, confidence3, _ = self.reference_sampler(
            encoded[3], reference_points_px, target_points_px,
            (cfg.image_height, cfg.image_width),
        )
        aligned_rgb, rgb_confidence, _ = self.reference_sampler.warp_rgb(
            references_skin_rgb, reference_points_px, target_points_px,
            (cfg.image_height // 4, cfg.image_width // 4),
        )
        reference_features = self.reference_decoder(
            source_features, condition_features, aligned2, aligned3
        )
        reference_skin_rgb = (
            aligned_rgb
            + torch.tanh(self.reference_residual_head(reference_features))
            * cfg.reference_residual_scale
        ).clamp(0.0, 1.0)
        transition_features = self.transition_fuse(
            torch.cat(
                (direct_features, reference_features, source_features[0], condition_features[0]),
                dim=1,
            )
        )
        transition_rgb = torch.sigmoid(self.transition_head(transition_features))
        completed = (
            direct_mouth_rgb * direct_owner
            + reference_skin_rgb * reference_owner
            + transition_rgb * transition_owner
        )
        final = source_rgb * (1.0 - alpha) + completed * alpha
        final = torch.where(alpha.expand_as(final) == 0, source_rgb, final).clamp(0.0, 1.0)
        return {
            "direct_mouth_rgb": direct_mouth_rgb,
            "reference_skin_rgb": reference_skin_rgb,
            "transition_rgb": transition_rgb,
            "completed_lower_face": completed,
            "final_output": final,
            "output": final,
            "alpha": alpha,
            "aligned_reference_skin_rgb": aligned_rgb,
            "reference_weights": reference_weights,
            "reference_confidence_quarter": confidence2,
            "reference_confidence_eighth": confidence3,
            "rgb_reference_confidence": rgb_confidence,
        }


__all__ = ["LowerFaceTextureV5", "LowerFaceTextureV5Config"]
