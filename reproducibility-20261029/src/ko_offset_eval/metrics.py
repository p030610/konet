from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .models import Analysis, DistanceMetric, Estimate, FailureExample, ModelReport, OffsetSelection, PairedComparison, RawScore, Recording, Support, WindowCell

DISTANCES_MS = (40, 80, 120, 160, 240, 320)
EPSILON = 1e-6
TEST_SPEAKERS = frozenset(("C282", "C371", "C580", "C613"))


class OffsetError(RuntimeError):
    pass


def _text(row: dict[str, object], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise OffsetError(f"raw row {name} must be a nonempty string")
    return value


def _raw(row: dict[str, object]) -> RawScore:
    score = row.get("score")
    status = _text(row, "score_status")
    valid_score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score)) else None
    if status == "valid" and valid_score is None:
        raise OffsetError("valid raw score must be finite")
    if status != "valid" and score is not None:
        raise OffsetError("invalid raw score must be null")
    lag = row.get("lag_ms")
    if isinstance(lag, bool) or not isinstance(lag, int):
        raise OffsetError("raw lag_ms must be an integer")
    return RawScore(_text(row, "recording_id"), _text(row, "window_id"), lag, status, valid_score, _text(row, "model_id"), _text(row, "fold"), _text(row, "direction"), _text(row, "evaluation_mode"))


def read_raw_jsonl(path: Path) -> tuple[RawScore, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OffsetError(f"cannot read raw ledger: {path}") from error
    rows: list[RawScore] = []
    seen: dict[tuple[str, str, str, int], RawScore] = {}
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise OffsetError("raw ledger contains invalid JSON") from error
        if not isinstance(payload, dict):
            raise OffsetError("raw ledger rows must be JSON objects")
        item = _raw(payload)
        key = (item.model_id, item.window_id, item.direction, item.lag_ms)
        prior = seen.get(key)
        if prior is not None and prior != item:
            raise OffsetError(f"conflicting duplicate raw observation: {key}")
        if prior is None:
            seen[key] = item
            rows.append(item)
    if not rows:
        raise OffsetError("raw ledger is empty")
    return tuple(rows)


def _estimate(values: dict[str, float], samples: int, rng: np.random.Generator) -> Estimate:
    if not values:
        return Estimate(None, None, None, 0)
    ordered = np.asarray([values[key] for key in sorted(values)], dtype=np.float64)
    draws = ordered[rng.integers(0, len(ordered), size=(samples, len(ordered)))].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return Estimate(float(ordered.mean()), float(low), float(high), len(ordered))


def _speaker_equal(records: dict[str, tuple[str, float]]) -> dict[str, float]:
    per_speaker: dict[str, list[float]] = defaultdict(list)
    for speaker, value in records.values():
        per_speaker[speaker].append(value)
    return {speaker: float(np.mean(values)) for speaker, values in per_speaker.items()}


def _cell(rows: dict[tuple[str, str, int], RawScore], recording: Recording, window: str, offset: int, distance: int) -> WindowCell | None:
    points = tuple(rows.get((recording.recording_id, window, offset + lag)) for lag in (0, -distance, distance))
    if any(point is None or point.score_status != "valid" or point.score is None for point in points):
        return None
    center, left, right = (float(point.score) for point in points if point is not None and point.score is not None)
    left_failure, right_failure = center <= left + EPSILON, center <= right + EPSILON
    outcome = ("both_failures" if left_failure and right_failure else "negative_only_failure" if left_failure else "positive_only_failure" if right_failure else "neither_failure")
    tie = int(abs(center - left) <= EPSILON) + int(abs(center - right) <= EPSILON)
    return WindowCell(float(outcome == "neither_failure"), center - max(left, right), outcome, tie, float(left_failure), float(right_failure))


def _metric(rows: tuple[RawScore, ...], recordings: tuple[Recording, ...], selections: dict[str, OffsetSelection], model_id: str, distance: int, samples: int, seed: int) -> DistanceMetric:
    lookup = {(row.recording_id, row.window_id, row.lag_ms): row for row in rows if row.model_id == model_id and row.fold == "B" and row.direction == "A_to_B"}
    conditional: dict[str, tuple[str, float]] = {}
    end_to_end: dict[str, tuple[str, float]] = {}
    margins: dict[str, tuple[str, float]] = {}
    negative_failures: dict[str, tuple[str, float]] = {}
    positive_failures: dict[str, tuple[str, float]] = {}
    outcomes = {key: 0 for key in ("neither_failure", "negative_only_failure", "positive_only_failure", "both_failures")}
    tie_comparisons = any_tie_windows = comparisons = available_windows = 0
    for recording in recordings:
        selection = selections.get(recording.recording_id)
        valid_selection = selection is not None and selection.status == "valid" and selection.selected_offset_ms is not None
        cells = tuple(_cell(lookup, recording, window, int(selection.selected_offset_ms), distance) for window in recording.b_window_ids) if valid_selection and selection is not None else ()
        usable = tuple(cell for cell in cells if cell is not None)
        if usable:
            success = float(np.mean([cell.success for cell in usable]))
            margin = float(np.mean([cell.margin for cell in usable]))
            conditional[recording.recording_id] = (recording.speaker_id, success)
            margins[recording.recording_id] = (recording.speaker_id, margin)
            negative_failures[recording.recording_id] = (recording.speaker_id, float(np.mean([cell.negative_failure for cell in usable])))
            positive_failures[recording.recording_id] = (recording.speaker_id, float(np.mean([cell.positive_failure for cell in usable])))
            for cell in usable:
                outcomes[cell.outcome] += 1
                tie_comparisons += cell.tie
                any_tie_windows += int(cell.tie > 0)
            comparisons += len(usable) * 2
            available_windows += len(usable)
        failures = [0.0 if cell is None else cell.success for cell in cells]
        end_to_end[recording.recording_id] = (recording.speaker_id, float(np.mean(failures)) if failures else 0.0)
    rng = np.random.default_rng(seed + distance)
    return DistanceMetric(_estimate(_speaker_equal(conditional), samples, rng), _estimate(_speaker_equal(end_to_end), samples, rng), _estimate(_speaker_equal(margins), samples, rng), _estimate(_speaker_equal(negative_failures), samples, rng), _estimate(_speaker_equal(positive_failures), samples, rng), Support(len(recordings), len(conditional), available_windows), outcomes, tie_comparisons, comparisons, tie_comparisons / comparisons if comparisons else None, any_tie_windows, any_tie_windows / available_windows if available_windows else None)


def analyze(rows: list[dict[str, object]] | tuple[RawScore, ...], recordings: tuple[Recording, ...], offsets: tuple[OffsetSelection, ...], *, bootstrap_samples: int = 10_000, seed: int = 2701) -> Analysis:
    if bootstrap_samples < 20:
        raise OffsetError("bootstrap_samples must be at least 20")
    raw = tuple(_raw(row) for row in rows) if isinstance(rows, list) else rows
    if len({item.recording_id for item in recordings}) != len(recordings):
        raise OffsetError("recordings metadata contains duplicate recording_id")
    selections = {item.recording_id: item for item in offsets}
    missing_decisions = sorted(item.recording_id for item in recordings if item.decision_status == "included" and item.recording_id not in selections)
    if missing_decisions:
        raise OffsetError(f"included recordings are missing A-fold offset decisions: {', '.join(missing_decisions)}")
    reports: dict[str, ModelReport] = {}
    for model_id in sorted({item.model_id for item in raw}):
        model_recordings = recordings if model_id == "original" else tuple(item for item in recordings if item.speaker_id in TEST_SPEAKERS)
        per_distance = {distance: _metric(raw, model_recordings, selections, model_id, distance, bootstrap_samples, seed) for distance in DISTANCES_MS}
        common_recordings = tuple(
            Recording(recording.recording_id, recording.speaker_id, recording.decision_status, tuple(sorted(set.intersection(*(_available_windows(raw, recording, selections, model_id, distance) for distance in DISTANCES_MS)))))
            for recording in model_recordings
            if all(_available_windows(raw, recording, selections, model_id, distance) for distance in DISTANCES_MS)
            and set.intersection(*(_available_windows(raw, recording, selections, model_id, distance) for distance in DISTANCES_MS))
        )
        common = len(common_recordings)
        common_curve = {distance: _metric(raw, common_recordings, selections, model_id, distance, bootstrap_samples, seed) for distance in DISTANCES_MS}
        reports[model_id] = ModelReport(model_id, per_distance, common, common_curve)
    paired = {
        model_id: _paired_comparison(raw, recordings, selections, model_id, bootstrap_samples, seed)
        for model_id in reports
        if model_id != "original"
    }
    statuses = dict(Counter(item.status for item in offsets))
    return Analysis(reports, paired, statuses, _failure_examples(raw, recordings, selections), bootstrap_samples, seed)


def analysis_json(analysis: Analysis) -> str:
    return json.dumps(asdict(analysis), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _available_windows(rows: tuple[RawScore, ...], recording: Recording, selections: dict[str, OffsetSelection], model_id: str, distance: int) -> set[str]:
    selection = selections.get(recording.recording_id)
    if selection is None or selection.status != "valid" or selection.selected_offset_ms is None:
        return set()
    lookup = {(row.recording_id, row.window_id, row.lag_ms): row for row in rows if row.model_id == model_id and row.fold == "B" and row.direction == "A_to_B"}
    return {window for window in recording.b_window_ids if _cell(lookup, recording, window, selection.selected_offset_ms, distance) is not None}


def _paired_comparison(rows: tuple[RawScore, ...], recordings: tuple[Recording, ...], selections: dict[str, OffsetSelection], fine_model_id: str, samples: int, seed: int) -> PairedComparison:
    paired_records: list[Recording] = []
    for recording in recordings:
        if recording.speaker_id not in TEST_SPEAKERS:
            continue
        windows = set(recording.b_window_ids)
        for model_id in ("original", fine_model_id):
            for distance in DISTANCES_MS:
                windows.intersection_update(_available_windows(rows, recording, selections, model_id, distance))
        if windows:
            paired_records.append(Recording(recording.recording_id, recording.speaker_id, recording.decision_status, tuple(sorted(windows))))
    frozen = tuple(paired_records)
    original_curve = {distance: _metric(rows, frozen, selections, "original", distance, samples, seed) for distance in DISTANCES_MS}
    fine_curve = {distance: _metric(rows, frozen, selections, fine_model_id, distance, samples, seed) for distance in DISTANCES_MS}
    original = ModelReport("original", original_curve, len(frozen), original_curve)
    fine = ModelReport(fine_model_id, fine_curve, len(frozen), fine_curve)
    return PairedComparison(fine_model_id, len(frozen), original, fine)


def _failure_examples(rows: tuple[RawScore, ...], recordings: tuple[Recording, ...], selections: dict[str, OffsetSelection]) -> tuple[FailureExample, ...]:
    lookup = {(row.recording_id, row.window_id, row.lag_ms): row for row in rows if row.model_id == "original" and row.fold == "B" and row.direction == "A_to_B"}
    selected: list[FailureExample] = []
    for recording in recordings:
        choice = selections.get(recording.recording_id)
        if choice is None or choice.status != "valid" or choice.selected_offset_ms is None:
            continue
        examples: list[FailureExample] = []
        for window in recording.b_window_ids:
            values = tuple(lookup.get((recording.recording_id, window, choice.selected_offset_ms + lag)) for lag in (0, -80, 80))
            if any(value is None or value.score_status != "valid" or value.score is None for value in values):
                continue
            center, negative, positive = (float(value.score) for value in values if value is not None and value.score is not None)
            examples.append(FailureExample(recording.recording_id, window, choice.selected_offset_ms, center, negative, positive, center - max(negative, positive)))
        if examples:
            selected.append(min(examples, key=lambda item: (item.minimum_margin, item.window_id)))
    return tuple(sorted(selected, key=lambda item: (item.minimum_margin, item.recording_id, item.window_id))[:10])
