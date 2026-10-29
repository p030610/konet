from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


Language = Literal["ko", "en"]
Split = Literal["development", "confirmation"]
Block = Literal["A", "B", "other"]
Role = Literal["core", "far", "heldout_support"]
Status = Literal["valid", "invalid"]


@dataclass(frozen=True, slots=True)
class ContractError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


class CohortRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: Literal[1]
    language: Language
    split: Split
    speaker_id: str = Field(min_length=1)
    recording_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    video_path: Path
    audio_path: Path
    language_evidence_path: Path
    language_evidence_record_key: str = Field(min_length=1)
    source_video_start_s: float = Field(ge=0.0)
    source_audio_start_s: float
    duration_s: float = Field(gt=0.0)
    source_fps: float = Field(gt=0.0)
    camera: str | None
    noise: str | None
    timing_authority: str = Field(min_length=1)
    waveform_sample_rate: Literal[16000]

    @field_validator("video_path", "audio_path", "language_evidence_path")
    @classmethod
    def require_absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ContractError("cohort paths must be absolute")
        return value


class AttemptFields(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1]
    run_id: str = Field(min_length=1)
    language: Language
    split: Split
    speaker_id: str = Field(min_length=1)
    recording_id: str = Field(min_length=1)
    utterance_id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    block: Block
    lag_ms: int
    role: Role
    video_start_s: float
    frame_start: int
    frame_phase_mod5: int = Field(ge=0, le=4)
    audio_start_sample: int
    audio_end_sample: int
    mel_start: int
    mel_end: int
    motion_value: float | None


class ExpectedRow(AttemptFields):
    pass


class RawRow(AttemptFields):
    status: Status
    reason: str | None
    score: float | None
    positive_loss: float | None

    @model_validator(mode="after")
    def require_terminal_payload(self) -> RawRow:
        if self.status == "invalid":
            if self.reason is None or self.score is not None or self.positive_loss is not None:
                raise ContractError("invalid rows require a reason and null score/loss")
        elif self.reason is not None or self.score is None or self.positive_loss is None:
            raise ContractError("valid rows require finite score/loss and null reason")
        return self


def load_cohort(path: Path) -> tuple[CohortRow, ...]:
    try:
        rows = tuple(
            CohortRow.model_validate_json(line)
            for line in path.read_bytes().splitlines()
            if line.strip()
        )
    except (OSError, ValidationError) as error:
        raise ContractError("cohort is unreadable or malformed") from error
    if not rows:
        raise ContractError("cohort is empty")
    utterance_ids = tuple(row.utterance_id for row in rows)
    if len(set(utterance_ids)) != len(utterance_ids):
        raise ContractError("duplicate utterance_id")
    return rows


__all__ = [
    "AttemptFields",
    "CohortRow",
    "ContractError",
    "ExpectedRow",
    "RawRow",
    "load_cohort",
]
