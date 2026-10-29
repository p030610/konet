from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Recording:
    recording_id: str
    speaker_id: str
    decision_status: str
    b_window_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OffsetSelection:
    recording_id: str
    fold: str
    selected_offset_ms: int | None
    status: str


@dataclass(frozen=True, slots=True)
class RawScore:
    recording_id: str
    window_id: str
    lag_ms: int
    score_status: str
    score: float | None
    model_id: str
    fold: str
    direction: str
    evaluation_mode: str


@dataclass(frozen=True, slots=True)
class WindowCell:
    success: float
    margin: float
    outcome: str
    tie: int
    negative_failure: float
    positive_failure: float


@dataclass(frozen=True, slots=True)
class Estimate:
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    speakers: int


@dataclass(frozen=True, slots=True)
class Support:
    all_recordings: int
    available_records: int
    available_windows: int


@dataclass(frozen=True, slots=True)
class DistanceMetric:
    conditional_success: Estimate
    end_to_end_success: Estimate
    margin: Estimate
    negative_failure_rate: Estimate
    positive_failure_rate: Estimate
    support: Support
    outcomes: dict[str, int]
    tie_comparisons: int
    comparison_count: int
    tie_rate: float | None
    any_tie_windows: int
    any_tie_window_rate: float | None


@dataclass(frozen=True, slots=True)
class ModelReport:
    model_id: str
    per_distance: dict[int, DistanceMetric]
    common_support_records: int
    common_support_per_distance: dict[int, DistanceMetric]


@dataclass(frozen=True, slots=True)
class PairedComparison:
    fine_model_id: str
    common_support_records: int
    original: ModelReport
    fine: ModelReport


@dataclass(frozen=True, slots=True)
class FailureExample:
    recording_id: str
    window_id: str
    selected_offset_ms: int
    center_score: float
    negative_80_score: float
    positive_80_score: float
    minimum_margin: float


@dataclass(frozen=True, slots=True)
class Analysis:
    models: dict[str, ModelReport]
    paired_comparisons: dict[str, PairedComparison]
    calibration_status_counts: dict[str, int]
    failure_examples_80ms: tuple[FailureExample, ...]
    bootstrap_samples: int
    seed: int
