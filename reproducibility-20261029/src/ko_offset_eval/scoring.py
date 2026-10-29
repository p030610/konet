from __future__ import annotations

from collections import defaultdict
import math

from .contracts import OffsetDecision, OffsetRawObservation
from .timing import CALIBRATION_LAGS_MS


EPSILON = 1e-6


def calibrate_recording(
    recording_id: str,
    a_window_ids: tuple[str, ...],
    rows: tuple[OffsetRawObservation, ...],
) -> OffsetDecision:
    scores: dict[int, list[float]] = defaultdict(list)
    per_window: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        if (
            row.recording_id == recording_id
            and row.window_id in a_window_ids
            and row.fold == "A"
            and row.score_status == "valid"
            and row.score is not None
        ):
            scores[row.lag_ms].append(row.score)
            per_window[row.window_id][row.lag_ms] = row.score
    complete = all(
        len(scores[lag]) == len(a_window_ids) and all(lag in per_window[window] for window in a_window_ids)
        for lag in CALIBRATION_LAGS_MS
    )
    if not complete:
        return _decision(recording_id, len(a_window_ids), "incomplete")
    means = {lag: sum(scores[lag]) / len(scores[lag]) for lag in CALIBRATION_LAGS_MS}
    maximum = max(means.values())
    minimum = min(means.values())
    maxima = tuple(lag for lag, value in means.items() if abs(value - maximum) <= EPSILON)
    observed_peak = maxima[0] if len(maxima) == 1 else None
    if maximum - minimum <= EPSILON:
        return _decision(recording_id, len(a_window_ids), "flat", observed_peak=observed_peak)
    if len(maxima) != 1:
        return _decision(recording_id, len(a_window_ids), "tied")
    peak = maxima[0]
    sorted_values = sorted(means.values(), reverse=True)
    prominence = maximum - sorted_values[1]
    window_peaks = []
    for window_id in a_window_ids:
        window_maximum = max(per_window[window_id].values())
        window_maxima = tuple(
            lag
            for lag, value in per_window[window_id].items()
            if abs(value - window_maximum) <= EPSILON
        )
        window_peaks.append(window_maxima[0] if len(window_maxima) == 1 else None)
    agreement = sum(item == peak for item in window_peaks) / len(window_peaks)
    if peak in (CALIBRATION_LAGS_MS[0], CALIBRATION_LAGS_MS[-1]):
        return OffsetDecision(
            recording_id=recording_id,
            selected_offset_ms=None,
            observed_peak_ms=peak,
            status="boundary",
            window_count=len(a_window_ids),
            peak_mean_score=maximum,
            prominence=prominence,
            per_window_peak_agreement=agreement,
        )
    return OffsetDecision(
        recording_id=recording_id,
        selected_offset_ms=peak,
        observed_peak_ms=peak,
        status="valid",
        window_count=len(a_window_ids),
        peak_mean_score=maximum,
        prominence=prominence,
        per_window_peak_agreement=agreement,
    )


def _decision(
    recording_id: str,
    window_count: int,
    status: str,
    observed_peak: int | None = None,
) -> OffsetDecision:
    return OffsetDecision(
        recording_id=recording_id,
        selected_offset_ms=None,
        observed_peak_ms=observed_peak,
        status=status,
        window_count=window_count,
        peak_mean_score=None,
        prominence=None,
        per_window_peak_agreement=None,
    )


__all__ = ["calibrate_recording"]
