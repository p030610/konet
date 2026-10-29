from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from typing_extensions import assert_never

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .timing import DIAGNOSTIC_SCORE_LAGS_MS, plan_support


Language = Literal["ko", "en"]
Dataset = Literal["ko_aihub538", "en_voxceleb2", "en_hdtf"]
LanguageStatus = Literal["verified", "automatic", "unknown", "rejected"]
DiagnosticFold = Literal["A", "B", "none"]
EvaluationMode = Literal["native", "diagnostic_search", "diagnostic_score"]
Protocol = Literal["native", "offset"]
Direction = Literal["A_to_B", "B_to_A", "none"]
ScoreStatus = Literal["valid", "missing_support", "nonfinite", "input_error"]


@dataclass(frozen=True, slots=True)
class ContractError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


class CohortWindow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: Literal[1]
    cohort_id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    dataset: Dataset
    language_status: LanguageStatus
    speaker_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_group_id: str = Field(min_length=1)
    recording_id: str = Field(min_length=1)
    video_path: Path
    audio_path: Path
    window_start_pts_seconds: float = Field(ge=0.0)
    window_duration_seconds: float = Field(gt=0.0)
    frame_start: int = Field(ge=0)
    frame_count: Literal[16]
    video_fps: float = Field(gt=0.0)
    source_video_fps: float = Field(gt=0.0)
    audio_start_sample: int = Field(ge=0)
    audio_sample_rate: int = Field(gt=0)
    diagnostic_selected: bool
    diagnostic_fold: DiagnosticFold

    @property
    def language(self) -> Language:
        match self.dataset:
            case "ko_aihub538":
                return "ko"
            case "en_voxceleb2" | "en_hdtf":
                return "en"
            case unreachable:
                assert_never(unreachable)

    @property
    def source_fps(self) -> float:
        return self.source_video_fps

    @field_validator("video_path", "audio_path")
    @classmethod
    def absolute_paths(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ContractError("cohort media paths must be absolute")
        return value

    @model_validator(mode="after")
    def diagnostic_assignment_is_consistent(self) -> CohortWindow:
        if self.diagnostic_selected != (self.diagnostic_fold != "none"):
            raise ContractError("diagnostic selection and fold disagree")
        return self


class ExpectedObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    config_digest: str
    observation_key: str
    window_id: str
    dataset: Dataset
    language: Language
    language_status: LanguageStatus
    speaker_id: str
    source_id: str
    source_group_id: str
    recording_id: str
    window_start_pts_seconds: float
    frame_start: int
    source_video_fps: float
    protocol: Protocol
    evaluation_mode: EvaluationMode
    fold: DiagnosticFold
    direction: Direction
    lag_ms: int
    support_start_sample_16k: int
    support_end_sample_16k: int


class RawObservation(ExpectedObservation):
    score_status: ScoreStatus
    score: float | None
    reason: str | None

    @model_validator(mode="after")
    def terminal_payload_is_consistent(self) -> RawObservation:
        if self.score_status == "valid":
            if self.score is None or self.reason is not None:
                raise ContractError("valid score requires numeric score and null reason")
        elif self.score is not None or self.reason is None:
            raise ContractError("invalid score requires null score and a reason")
        return self


def load_cohort(path: Path) -> tuple[CohortWindow, ...]:
    try:
        rows = tuple(
            CohortWindow.model_validate_json(line)
            for line in path.read_bytes().splitlines()
            if line.strip()
        )
    except (OSError, ValidationError) as error:
        raise ContractError("cohort is unreadable or malformed") from error
    if not rows:
        raise ContractError("cohort is empty")
    window_ids = tuple(row.window_id for row in rows)
    if len(window_ids) != len(set(window_ids)):
        raise ContractError("duplicate window_id")
    selected_groups = {row.source_group_id for row in rows if row.diagnostic_selected}
    for group in selected_groups:
        group_rows = tuple(row for row in rows if row.source_group_id == group)
        folds = {row.diagnostic_fold for row in group_rows}
        if not {"A", "B"}.issubset(folds):
            raise ContractError("selected source group must contain both diagnostic folds")
        bounds: dict[str, list[int]] = {"A": [], "B": []}
        for row in group_rows:
            if row.diagnostic_fold == "none":
                continue
            origin = round(row.audio_start_sample * 16_000 / row.audio_sample_rate)
            supports = tuple(plan_support(origin, lag) for lag in DIAGNOSTIC_SCORE_LAGS_MS)
            bounds[row.diagnostic_fold].extend(
                (min(item.start_sample_16k for item in supports), max(item.end_sample_16k for item in supports))
            )
        a_start, a_end = min(bounds["A"]), max(bounds["A"])
        b_start, b_end = min(bounds["B"]), max(bounds["B"])
        if not (a_end <= b_start or b_end <= a_start):
            raise ContractError("diagnostic A/B expanded waveform supports overlap")
    return rows


__all__ = [
    "CohortWindow",
    "ContractError",
    "ExpectedObservation",
    "RawObservation",
    "load_cohort",
]
