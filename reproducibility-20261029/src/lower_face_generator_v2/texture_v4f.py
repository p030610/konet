from __future__ import annotations

from dataclasses import dataclass

import torch

from .texture_v4e import LowerFaceIdentityDetailV4E


@dataclass(frozen=True)
class LowerFaceJointDetailV4FConfig:
    train_reference_encoder: bool = True
    train_decode1: bool = True
    train_decode0: bool = True
    train_output_head: bool = True


class LowerFaceJointDetailV4F(LowerFaceIdentityDetailV4E):
    """R6B: jointly tune late RGB/reference decoding and the R6A detail head."""

    architecture_version = "lower_face_joint_detail_v4f"

    def __init__(
        self,
        detail: LowerFaceIdentityDetailV4E,
        config: LowerFaceJointDetailV4FConfig | None = None,
    ) -> None:
        super().__init__(detail.texture, detail.config)
        self.load_state_dict(detail.state_dict())
        self.joint_config = config or LowerFaceJointDetailV4FConfig()
        self.requires_grad_(False)

        # The residual pathway remains trainable. R4 spatial authority, source
        # conditioning, coarse fusion, and geometry projection stay frozen.
        for name in ("enc0", "enc1", "enc2", "bottleneck", "dec2", "dec1", "dec0", "output_head"):
            getattr(self, name).requires_grad_(True)
        if self.joint_config.train_reference_encoder:
            self.texture.reference_encoder.requires_grad_(True)
        if self.joint_config.train_decode1:
            self.texture.decode1.requires_grad_(True)
        if self.joint_config.train_decode0:
            self.texture.decode0.requires_grad_(True)
        if self.joint_config.train_output_head:
            self.texture.output_head.requires_grad_(True)

    def parameter_groups(
        self, detail_learning_rate: float, pretrained_learning_rate: float
    ) -> list[dict]:
        detail_parameters = []
        pretrained_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("texture."):
                pretrained_parameters.append(parameter)
            else:
                detail_parameters.append(parameter)
        return [
            {"params": detail_parameters, "lr": detail_learning_rate, "name": "detail"},
            {"params": pretrained_parameters, "lr": pretrained_learning_rate, "name": "late_rgb"},
        ]

    def trainable_module_names(self) -> tuple[str, ...]:
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)


__all__ = ["LowerFaceJointDetailV4F", "LowerFaceJointDetailV4FConfig"]
