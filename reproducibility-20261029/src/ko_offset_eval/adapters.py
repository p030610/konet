from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from konet_training.configuration import ModelConfig
from konet_training.lora import apply_lora
from omegaconf import OmegaConf
from torch import nn

AdapterKind = Literal["full", "lora"]


@dataclass(frozen=True, slots=True)
class AdapterLoadError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class KoNetAdapterConfig:
    model_id: str
    checkpoint: Path
    training_config: Path
    device: torch.device


@dataclass(frozen=True, slots=True)
class AdapterReceipt:
    model_id: str
    kind: AdapterKind
    checkpoint: Path
    training_config: Path
    base_checkpoint: Path
    architecture_config: Path
    strict_load: Literal[True]
    lora_applied: bool


class KoNetEncoderAdapter:
    """Frozen StableSyncNet branches for FrozenScorer-preprocessed tensors."""

    def __init__(self, config: KoNetAdapterConfig) -> None:
        model_config = _read_model_config(config.training_config)
        model = _build_stable_syncnet(model_config)
        base_checkpoint = _required_path(
            model_config.base_checkpoint, "base checkpoint"
        )
        _strict_load(model, base_checkpoint)
        lora_applied = model_config.lora is not None
        if lora_applied:
            apply_lora(model, model_config.lora)
        _strict_load(model, config.checkpoint)
        network = model.to(config.device).requires_grad_(False).eval()
        if network.training or any(
            parameter.requires_grad for parameter in network.parameters()
        ):
            raise AdapterLoadError("loaded KoNet adapter is not frozen eval")
        self.network = network
        self.model_id = config.model_id
        self.receipt = AdapterReceipt(
            model_id=config.model_id,
            kind="lora" if lora_applied else "full",
            checkpoint=config.checkpoint,
            training_config=config.training_config,
            base_checkpoint=base_checkpoint,
            architecture_config=_required_path(
                model_config.architecture_config, "architecture config"
            ),
            strict_load=True,
            lora_applied=lora_applied,
        )

    def visual_embedding(self, visual: torch.Tensor) -> torch.Tensor:
        """Embed FrozenScorer's official 16-frame lower-half visual tensor."""
        value = visual.to(self._device)
        return self.network.visual_encoder(value).reshape(value.shape[0], -1).float()

    def audio_embedding(self, mel: torch.Tensor) -> torch.Tensor:
        """Embed FrozenScorer's official 80x52 mel tensor."""
        value = mel.to(self._device)
        return self.network.audio_encoder(value).reshape(value.shape[0], -1).float()

    @property
    def _device(self) -> torch.device:
        try:
            return next(self.network.parameters()).device
        except StopIteration as error:
            raise AdapterLoadError("loaded KoNet adapter has no parameters") from error


def _read_model_config(path: Path) -> ModelConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_payload = payload["model"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise AdapterLoadError(f"training config is unreadable: {path}") from error
    try:
        return ModelConfig.model_validate(model_payload)
    except ValueError as error:
        raise AdapterLoadError(f"training model config is invalid: {path}") from error


def _required_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise AdapterLoadError(f"KoNet {label} is absent")
    return path


def _build_stable_syncnet(config: ModelConfig) -> nn.Module:
    vendor_root = _required_path(config.vendor_root, "vendor root")
    architecture_config = _required_path(
        config.architecture_config, "architecture config"
    )
    vendor_text = str(vendor_root)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    try:
        from latentsync.models.stable_syncnet import StableSyncNet

        architecture = OmegaConf.to_container(
            OmegaConf.load(architecture_config).model, resolve=True
        )
        return StableSyncNet(architecture, gradient_checkpointing=False)
    except (ImportError, OSError, TypeError, ValueError) as error:
        raise AdapterLoadError(
            "StableSyncNet architecture construction failed"
        ) from error


def _strict_load(network: nn.Module, checkpoint: Path) -> None:
    try:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        if not isinstance(payload, Mapping):
            raise AdapterLoadError("checkpoint payload is not a mapping")
        state = _checkpoint_state(payload)
        network.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise AdapterLoadError(
            f"strict checkpoint load failed: {checkpoint}"
        ) from error


def _checkpoint_state(payload: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model_state", payload.get("state_dict", payload))
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise AdapterLoadError("checkpoint does not contain a tensor model state")
    return state


__all__ = [
    "AdapterLoadError",
    "AdapterReceipt",
    "KoNetAdapterConfig",
    "KoNetEncoderAdapter",
]
