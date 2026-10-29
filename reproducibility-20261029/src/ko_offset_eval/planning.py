from __future__ import annotations

from koen_syncnet_eval.contracts import CohortWindow

from .contracts import OffsetDecision, OffsetExpectedObservation, OffsetRecording
from .timing import CALIBRATION_LAGS_MS, heldout_lags, plan_offset_support


def _expected(
    row: CohortWindow,
    run_id: str,
    config_digest: str,
    model_id: str,
    fold: str,
    lag_ms: int,
) -> OffsetExpectedObservation:
    origin = round(row.audio_start_sample * 16_000 / row.audio_sample_rate)
    support = plan_offset_support(origin, lag_ms)
    mode = "diagnostic_search" if fold == "A" else "diagnostic_score"
    return OffsetExpectedObservation(
        run_id=run_id,
        config_digest=config_digest,
        observation_key=f"{model_id}|{row.window_id}|{fold}|{lag_ms}",
        window_id=row.window_id,
        dataset=row.dataset,
        language=row.language,
        language_status=row.language_status,
        speaker_id=row.speaker_id,
        source_id=row.source_id,
        source_group_id=row.source_group_id,
        recording_id=row.recording_id,
        window_start_pts_seconds=row.window_start_pts_seconds,
        frame_start=row.frame_start,
        source_video_fps=row.source_video_fps,
        protocol="offset",
        evaluation_mode=mode,
        fold=fold,
        direction="A_to_B",
        lag_ms=lag_ms,
        support_start_sample_16k=support.start_sample_16k,
        support_end_sample_16k=support.end_sample_16k,
        model_id=model_id,
    )


def plan_calibration(
    windows: tuple[CohortWindow, ...],
    recordings: tuple[OffsetRecording, ...],
    run_id: str,
    config_digest: str,
    model_id: str,
) -> tuple[OffsetExpectedObservation, ...]:
    a_ids = {window_id for row in recordings for window_id in row.a_window_ids}
    return tuple(
        _expected(row, run_id, config_digest, model_id, "A", lag)
        for row in windows
        if row.window_id in a_ids
        for lag in CALIBRATION_LAGS_MS
    )


def plan_heldout(
    windows: tuple[CohortWindow, ...],
    recordings: tuple[OffsetRecording, ...],
    decisions: tuple[OffsetDecision, ...],
    run_id: str,
    config_digest: str,
    model_id: str,
) -> tuple[OffsetExpectedObservation, ...]:
    offsets = {
        row.recording_id: row.selected_offset_ms
        for row in decisions
        if row.status == "valid" and row.selected_offset_ms is not None
    }
    b_ids = {window_id for row in recordings for window_id in row.b_window_ids}
    return tuple(
        _expected(row, run_id, config_digest, model_id, "B", lag)
        for row in windows
        if row.window_id in b_ids and row.recording_id in offsets
        for lag in heldout_lags(offsets[row.recording_id])
    )


__all__ = ["plan_calibration", "plan_heldout"]
