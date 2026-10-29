from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Mapping

from pydantic import ValidationError

from koen_syncnet_eval.contracts import CohortWindow, ContractError

from .contracts import OffsetRecording
from .timing import CALIBRATION_LAGS_MS, MAXIMUM_HELDOUT_LAG_MS, plan_offset_support


def load_korean_windows(path: Path) -> tuple[CohortWindow, ...]:
    try:
        rows = tuple(
            CohortWindow.model_validate_json(line)
            for line in path.read_bytes().splitlines()
            if line.strip()
        )
    except (OSError, ValidationError) as error:
        raise ContractError("source cohort is unreadable or malformed") from error
    korean = tuple(row for row in rows if row.dataset == "ko_aihub538")
    if not korean:
        raise ContractError("source cohort contains no Korean windows")
    keys = tuple(row.window_id for row in korean)
    if len(keys) != len(set(keys)):
        raise ContractError("source cohort contains duplicate Korean window_id")
    return korean


def _bounds(rows: tuple[CohortWindow, ...], lags: Iterable[int]) -> tuple[int, int]:
    supports = []
    for row in rows:
        origin = round(row.audio_start_sample * 16_000 / row.audio_sample_rate)
        supports.extend(plan_offset_support(origin, lag) for lag in lags)
    return (
        min(item.start_sample_16k for item in supports),
        max(item.end_sample_16k for item in supports),
    )


def _visual_disjoint(a_rows: tuple[CohortWindow, ...], b_rows: tuple[CohortWindow, ...]) -> bool:
    return max(row.frame_start + row.frame_count for row in a_rows) <= min(
        row.frame_start for row in b_rows
    )


def build_recordings(
    raw_rows: Iterable[CohortWindow | dict[str, bool | int | float | str]],
    audio_sample_counts: Mapping[Path, int] | None = None,
) -> tuple[OffsetRecording, ...]:
    grouped: dict[str, list[CohortWindow]] = defaultdict(list)
    for raw in raw_rows:
        row = raw if isinstance(raw, CohortWindow) else CohortWindow.model_validate(raw)
        if row.dataset == "ko_aihub538":
            grouped[row.recording_id].append(row)
    decisions: list[OffsetRecording] = []
    for recording_id, candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: (item.frame_start, item.window_id))
        count = 4 if len(candidates) >= 12 else 2
        audio_samples = (
            audio_sample_counts.get(candidates[0].audio_path)
            if audio_sample_counts is not None
            else None
        )
        a_options = []
        b_options = []
        for start in range(0, len(candidates) - count + 1):
            group = tuple(candidates[start : start + count])
            a_bounds = _bounds(group, CALIBRATION_LAGS_MS)
            b_bounds = _bounds(group, (-MAXIMUM_HELDOUT_LAG_MS, MAXIMUM_HELDOUT_LAG_MS))
            if a_bounds[0] >= 0 and (audio_samples is None or a_bounds[1] <= audio_samples):
                a_options.append((group, a_bounds))
            if b_bounds[0] >= 0 and (audio_samples is None or b_bounds[1] <= audio_samples):
                b_options.append((group, b_bounds))
        selected = next(
            (
                (a_group, b_group)
                for a_group, a_bounds in a_options
                for b_group, b_bounds in reversed(b_options)
                if (a_bounds[1] <= b_bounds[0] or b_bounds[1] <= a_bounds[0])
                and _visual_disjoint(a_group, b_group)
            ),
            None,
        )
        usable = selected is not None
        a_rows, b_rows = selected if selected is not None else ((), ())
        decisions.append(
            OffsetRecording(
                recording_id=recording_id,
                speaker_id=candidates[0].speaker_id,
                decision_status="included" if usable else "construction_exclusion",
                reason=None if usable else "expanded_cross_fold_support_overlap_or_insufficient",
                a_window_ids=tuple(row.window_id for row in a_rows) if usable else (),
                b_window_ids=tuple(row.window_id for row in b_rows) if usable else (),
            )
        )
    return tuple(decisions)


def selected_windows(
    windows: tuple[CohortWindow, ...], recordings: tuple[OffsetRecording, ...]
) -> tuple[CohortWindow, ...]:
    selected_ids = {
        window_id
        for recording in recordings
        for window_id in (*recording.a_window_ids, *recording.b_window_ids)
    }
    return tuple(row for row in windows if row.window_id in selected_ids)


__all__ = ["build_recordings", "load_korean_windows", "selected_windows"]
