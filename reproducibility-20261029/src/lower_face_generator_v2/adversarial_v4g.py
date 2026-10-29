from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class GeometryConditionedPatchDiscriminatorV4G(nn.Module):
    """Patch critic conditioned on frozen source context and geometry fields."""

    def __init__(self, input_channels: int = 17, base_channels: int = 32) -> None:
        super().__init__()
        channels = [input_channels, base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.blocks = nn.ModuleList()
        for index, (source, target) in enumerate(zip(channels[:-1], channels[1:])):
            convolution = spectral_norm(nn.Conv2d(source, target, 4, 2, 1))
            layers: list[nn.Module] = [convolution]
            if index:
                layers.append(nn.GroupNorm(min(16, target), target))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.blocks.append(nn.Sequential(*layers))
        self.head = spectral_norm(nn.Conv2d(channels[-1], 1, 3, 1, 1))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = []
        for block in self.blocks:
            value = block(value)
            features.append(value)
        return self.head(value), features


def build_geometry_conditioned_discriminator_input(
    rgb: torch.Tensor,
    source_masked: torch.Tensor,
    structure: dict[str, torch.Tensor],
) -> torch.Tensor:
    maps = torch.cat(
        [structure[name].detach() for name in ("lip_oral", "motion", "edit", "confidence")],
        dim=1,
    ).to(rgb.dtype)
    return torch.cat(
        [rgb, source_masked.detach(), structure["geometry_fields"].detach().to(rgb.dtype), maps],
        dim=1,
    )


def patch_mask(mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return F.interpolate(mask.float(), logits.shape[-2:], mode="area").to(logits.dtype)


def masked_patch_hinge_discriminator_loss(
    real_logits: torch.Tensor, fake_logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    weight = patch_mask(mask, real_logits)
    denominator = weight.sum().clamp_min(1.0)
    real = (F.relu(1.0 - real_logits) * weight).sum() / denominator
    fake = (F.relu(1.0 + fake_logits) * weight).sum() / denominator
    return real + fake


def masked_patch_generator_loss(fake_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = patch_mask(mask, fake_logits)
    return -(fake_logits * weight).sum() / weight.sum().clamp_min(1.0)


def masked_feature_matching_loss(
    real_features: list[torch.Tensor],
    fake_features: list[torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    total = mask.new_zeros(())
    for real, fake in zip(real_features, fake_features):
        weight = F.interpolate(mask.float(), fake.shape[-2:], mode="area").to(fake.dtype)
        total = total + ((fake - real.detach()).abs() * weight).sum() / (
            weight.sum() * fake.shape[1]
        ).clamp_min(1.0)
    return total / max(len(fake_features), 1)


__all__ = [
    "GeometryConditionedPatchDiscriminatorV4G",
    "build_geometry_conditioned_discriminator_input",
    "masked_feature_matching_loss",
    "masked_patch_generator_loss",
    "masked_patch_hinge_discriminator_loss",
]
