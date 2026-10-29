from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from koen_syncnet_eval.contracts import ContractError, ExpectedObservation, RawObservation


DecisionStatus = Literal["included", "construction_exclusion"]
CalibrationStatus = Literal["valid", "flat", "tied", "boundary", "incomplete"]


class OffsetRecording(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    recording_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    decision_status: DecisionStatus
    include_in_all276: Literal[True] = True
    reason: str | None
    a_window_ids: tuple[str, ...]
    b_window_ids: tuple[str, ...]

    @model_validator(mode="after")
    def included_has_both_folds(self) -> OffsetRecording:
        if self.decision_status == "included" and (
            len(self.a_window_ids) < 2 or len(self.b_window_ids) < 2
        ):
            raise ContractError("included recording requires at least two windows per fold")
        return self


class OffsetDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    recording_id: str
    fold: Literal["A"] = "A"
    selected_offset_ms: int | None
    observed_peak_ms: int | None
    status: CalibrationStatus
    window_count: int
    peak_mean_score: float | None
    prominence: float | None
    per_window_peak_agreement: float | None


class OffsetExpectedObservation(ExpectedObservation):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(min_length=1)


class OffsetRawObservation(RawObservation):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class PreparationRequest:
    source_cohort: Path
    output_root: Path
    old_raw: Path
    old_config: Path
    old_score_config: Path
    checkpoint: Path
    native_root: Path
    run_id: str
    model_id: str = "original"
    reuse_old_scores: bool = False
    reference_preparation: Path | None = None


__all__ = [
    "CalibrationStatus",
    "ContractError",
    "OffsetDecision",
    "OffsetExpectedObservation",
    "OffsetRawObservation",
    "OffsetRecording",
    "PreparationRequest",
]
