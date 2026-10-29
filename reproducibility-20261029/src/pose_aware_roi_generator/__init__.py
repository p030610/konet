from __future__ import annotations

from importlib import import_module
from typing import Final

from torch import nn


_EXPORTS: Final = {
    "PoseAwareRoiGenerator": (".model", "PoseAwareRoiGenerator"),
    "PoseAwareRoiGeneratorV2": (".model_v2", "PoseAwareRoiGeneratorV2"),
    "GeometryConditionedDINetGenerator": (".model_v5_dinet", "GeometryConditionedDINetGenerator"),
    "AudioFiLMDINetGenerator": (".model_v8_audio_film", "AudioFiLMDINetGenerator"),
    "GeometryRasterizer": (".rasterizer", "GeometryRasterizer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> type[nn.Module]:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    return getattr(import_module(module_name, __name__), attribute_name)
