from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn
from torch.nn.utils import parametrize

from .adapter_config import LoRAConfig
from .errors import TrainingError

_TARGET_SUFFIXES: Final = (".attn.to_q", ".attn.to_k", ".attn.to_v", ".attn.to_out.0")
_BRANCHES: Final = ("audio_encoder", "visual_encoder")


@dataclass(frozen=True, slots=True)
class LoRAReport:
    rank: int
    alpha: float
    targets: tuple[str, ...]
    total_parameters: int
    trainable_parameters: int


class LowRankWeight(nn.Module):
    """Trainable low-rank delta; the original Linear weight stays immutable."""

    def __init__(self, weight: torch.Tensor, config: LoRAConfig) -> None:
        super().__init__()
        self.lora_a = nn.Parameter(weight.new_empty((config.rank, weight.shape[1])))
        self.lora_b = nn.Parameter(weight.new_zeros((weight.shape[0], config.rank)))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = config.alpha / config.rank

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + (self.lora_b @ self.lora_a) * self.scaling


def apply_lora(model: nn.Module, config: LoRAConfig) -> LoRAReport:
    targets = tuple(
        (name, module) for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(_TARGET_SUFFIXES)
        and any(name.startswith(branch + ".") for branch in _BRANCHES)
    )
    for branch in _BRANCHES:
        if not any(name.startswith(branch + ".") for name, _module in targets):
            raise TrainingError(f"LoRA attention targets missing in {branch}")
    for name, module in targets:
        if parametrize.is_parametrized(module, "weight") or config.rank > min(module.weight.shape):
            raise TrainingError(f"LoRA target already adapted or rank exceeds dimensions: {name}")
    model.requires_grad_(False)
    for _name, module in targets:
        parametrize.register_parametrization(module, "weight", LowRankWeight(module.weight, config))
    return LoRAReport(
        config.rank, config.alpha, tuple(name for name, _module in targets),
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )


def require_lora_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    expected = {
        id(parameter) for module in model.modules() if isinstance(module, LowRankWeight)
        for parameter in module.parameters()
    }
    actual = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if not expected or actual != expected or not expected.issubset(optimized):
        raise TrainingError("LoRA optimizer must update all and only adapter parameters")
    for branch in _BRANCHES:
        if not expected.intersection(id(parameter) for parameter in model.get_submodule(branch).parameters()):
            raise TrainingError(f"LoRA branch has no trainable adapter: {branch}")
