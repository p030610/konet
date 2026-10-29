from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field

from .syncnet_utility_models_v1 import InventoryRow


AdapterValue = Literal[
    "local_wrapper_baseline",
    "v5_target_lip_support_mask_crop",
    "mediapipe_landmark_aligned_crop_v1",
]
CandidateAdapter = Literal[
    "v5_target_lip_support_mask_crop",
    "mediapipe_landmark_aligned_crop_v1",
]


class EvalError(RuntimeError):
    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


EvaluationItem = InventoryRow


class ScoreRow(InventoryRow):
    adapter: AdapterValue
    cosine: float = Field(allow_inf_nan=False)
    prediction: Literal[0, 1]
    correct: bool


class AdapterSummary(_FrozenModel):
    examples: int = Field(ge=0)
    pairs: int = Field(ge=0)
    clips: int = Field(ge=0)
    identities: int = Field(ge=0)
    positives: int = Field(ge=0)
    negatives: int = Field(ge=0)
    correct: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1, allow_inf_nan=False)
    identity_cluster_lower_95: float = Field(ge=0, le=1, allow_inf_nan=False)
    identity_cluster_upper_95: float = Field(ge=0, le=1, allow_inf_nan=False)


class EvaluationSummary(_FrozenModel):
    schema_version: Literal["syncnet_utility_summary_v1"]
    threshold: Literal[0.5]
    seed: int
    bootstrap_resamples: int = Field(gt=0)
    selected_adapter: CandidateAdapter = "v5_target_lip_support_mask_crop"
    adapters: dict[AdapterValue, AdapterSummary]
    by_domain: dict[str, dict[AdapterValue, AdapterSummary]]
    exclusions: int = Field(ge=0)
    runtime_failures: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class PairPayload:
    positive_rgb: torch.Tensor
    positive_mask: torch.Tensor
    negative_rgb: torch.Tensor
    negative_mask: torch.Tensor
    anchor_mel: torch.Tensor
    positive_face_landmarks_xy: torch.Tensor | None = None
    negative_face_landmarks_xy: torch.Tensor | None = None
    crop_to_face_affine: torch.Tensor | None = None


class EmbeddingEncoder(Protocol):
    def audio_embedding(self, mel: torch.Tensor) -> torch.Tensor: ...

    def visual_embedding(self, visual: torch.Tensor) -> torch.Tensor: ...

    def state_identity(self) -> str: ...

    def content_identity(self) -> str: ...


class PairMaterializer(Protocol):
    def load(self, positive: EvaluationItem, negative: EvaluationItem) -> PairPayload: ...
