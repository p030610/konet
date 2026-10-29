from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoRAConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["attention_lora_v1"] = "attention_lora_v1"
    rank: int = Field(default=8, ge=1, le=64)
    alpha: float = Field(default=16.0, gt=0, le=128)
