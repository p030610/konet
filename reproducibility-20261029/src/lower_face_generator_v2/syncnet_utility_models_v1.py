from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

Split = Literal["train", "val", "test"]
EligibleSplit = Literal["train", "val"]
Domain = Literal["hdtf", "voxceleb2"]
ClockStatus = Literal["verified_zero", "verified_nonzero"]
Adapter = Literal[
    "local_wrapper_baseline",
    "v5_target_lip_support_mask_crop",
    "mediapipe_landmark_aligned_crop_v1",
]


@dataclass(frozen=True, slots=True)
class ProtocolError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def _invalid(message: str) -> PydanticCustomError:
    return PydanticCustomError("syncnet_protocol", message)


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class FrameRange(FrozenModel):
    start: Annotated[int, Field(ge=0)]
    stop: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def ordered(self) -> FrameRange:
        if self.stop <= self.start:
            raise _invalid("frame range must be ordered")
        return self


class ManifestRow(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)
    split: Split
    domain: Domain
    identity: Annotated[str, Field(min_length=1)]
    clip_id: Annotated[str, Field(min_length=1)]
    fps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    frame_count: Annotated[int, Field(ge=16)]
    root: Annotated[str, Field(min_length=1)]
    lower_face_points: Annotated[str, Field(min_length=1)]
    mask_cache_v5: Annotated[str, Field(min_length=1)]
    mask_cache_format_v5: Annotated[str, Field(min_length=1)]
    version_v5: Annotated[str, Field(min_length=1)]


class CacheDescriptors(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)
    crop_root: Annotated[str, Field(min_length=1)]
    points_path: Annotated[str, Field(min_length=1)]
    mask_root: Annotated[str, Field(min_length=1)]
    mask_format: Annotated[str, Field(min_length=1)]
    mask_stage: Annotated[str, Field(min_length=1)]


class CatalogRow(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)
    split: Split
    domain: Domain
    identity: Annotated[str, Field(min_length=1)]
    clip_id: Annotated[str, Field(min_length=1)]
    fps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    frame_count: Annotated[int, Field(ge=16)]
    valid_sequence16_starts: tuple[FrameRange, ...]
    caches: CacheDescriptors


class CatalogDocument(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)
    clips: tuple[CatalogRow, ...]


class ClockSource(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)
    clip_id: Annotated[str, Field(min_length=1)]
    domain: Domain
    split: EligibleSplit
    video_path: Annotated[str, Field(min_length=1)]
    wav_path: Annotated[str, Field(min_length=1)]
    manifest_path: Annotated[str, Field(min_length=1)]
    catalog_fps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    catalog_frames: Annotated[int, Field(ge=16)]
    feature_length: Annotated[int, Field(gt=0)]


class ClockReceipt(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)
    source: ClockSource
    status: ClockStatus
    video_start: Annotated[float, Field(allow_inf_nan=False)]
    audio_start: Annotated[float, Field(allow_inf_nan=False)]
    source_fps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    wav_duration: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    offset_seconds: Annotated[float, Field(allow_inf_nan=False)]


class InventoryRow(FrozenModel):
    protocol_version: Literal["lnet_syncnet_utility_protocol_v1"]
    protocol_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    inventory_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    row_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    pair_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    label: Literal[0, 1]
    split: EligibleSplit
    domain: Domain
    identity: Annotated[str, Field(min_length=1)]
    clip_id: Annotated[str, Field(min_length=1)]
    anchor_start: Annotated[int, Field(ge=0)]
    visual_start: Annotated[int, Field(ge=0)]
    wrong_start: Annotated[int, Field(ge=0)]
    fps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    clock_status: ClockStatus
    clock_offset_seconds: Annotated[float, Field(allow_inf_nan=False)]
    audio_path: Annotated[str, Field(min_length=1)]
    audio_start_seconds: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    mel_start: Annotated[int, Field(ge=0)]
    mel_stop: Annotated[int, Field(gt=0)]
    mel_columns: Annotated[int, Field(gt=0)]
    crop_root: Annotated[str, Field(min_length=1)]
    points_path: Annotated[str, Field(min_length=1)]
    mask_root: Annotated[str, Field(min_length=1)]
    mask_format: Annotated[str, Field(min_length=1)]
    mask_stage: Annotated[str, Field(min_length=1)]
    adapters: tuple[Adapter, Adapter]


class ExclusionRow(FrozenModel):
    split: Split
    domain: Domain
    identity: str
    clip_id: str
    reason: Literal[
        "split_excluded",
        "clock_receipt_missing",
        "insufficient_contiguous_frames",
        "insufficient_valid_starts",
        "invalid_full_mel_bounds",
    ]


class CountSummary(FrozenModel):
    pre_eligibility: dict[str, int]
    post_eligibility: dict[str, int]
    exclusion_reasons: dict[str, int]


@dataclass(frozen=True, slots=True)
class BuildResult:
    rows: tuple[InventoryRow, ...]
    exclusions: tuple[ExclusionRow, ...]
    counts: CountSummary


@dataclass(frozen=True, slots=True)
class BuildConfig:
    splits: tuple[EligibleSplit, ...] = ("train", "val")
    anchors_per_clip: int = 8
    seed: int = 1701

    def __post_init__(self) -> None:
        if not self.splits or len(self.splits) != len(set(self.splits)):
            raise _invalid("splits must be unique and non-empty")
        if not 1 <= self.anchors_per_clip <= 8:
            raise _invalid("anchors_per_clip must be between one and eight")


@dataclass(frozen=True, slots=True)
class OutputPaths:
    protocol: Path
    inventory: Path
    counts: Path
    exclusions: Path
