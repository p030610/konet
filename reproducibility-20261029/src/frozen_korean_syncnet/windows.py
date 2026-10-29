from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .models import ManifestRow


SYNC_FPS: Final = 25.0
MEL_RATE: Final = 80.0
FRAME_COUNT: Final = 16
MEL_COLUMNS: Final = 52


@dataclass(frozen=True, slots=True)
class WindowPlanError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class WindowPlan:
    start_s: float
    duration_s: float
    video_fps: float
    audio_duration_s: float
    shifts_ms: tuple[int, ...]
    maximum_anchors: int

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.duration_s <= 0 or self.video_fps <= 0:
            raise WindowPlanError("media timing must be positive")
        if self.audio_duration_s <= 0 or self.maximum_anchors < 1 or not self.shifts_ms:
            raise WindowPlanError("audio duration, shifts, and anchor count are required")


def mel_bounds(anchor_s: float, shift_ms: int) -> tuple[int, int]:
    """Map a signed injected audio delay to an unpadded native mel window."""
    start = int(MEL_RATE * (anchor_s - shift_ms / 1000.0))
    return start, start + MEL_COLUMNS


def paired_anchor(anchor_s: float, source_start_s: float, paired_start_s: float) -> float:
    return paired_start_s + (anchor_s - source_start_s)


def plan_common_anchors(plan: WindowPlan) -> tuple[float, ...]:
    """Return anchors whose video and every shifted audio window share support."""
    clip_end = plan.start_s + plan.duration_s
    video_last_offset = (FRAME_COUNT - 1) / SYNC_FPS
    lower = max(
        plan.start_s,
        max(plan.start_s + shift_ms / 1000.0 for shift_ms in plan.shifts_ms),
    )
    upper = min(
        clip_end - video_last_offset,
        min(
            min(plan.audio_duration_s, clip_end) - MEL_COLUMNS / MEL_RATE + shift_ms / 1000.0
            for shift_ms in plan.shifts_ms
        ),
    )
    if upper < lower:
        return ()
    count = min(plan.maximum_anchors, max(1, int((upper - lower) * SYNC_FPS) + 1))
    return tuple(float(value) for value in np.linspace(lower, upper, count))


def select_wrong_utterance(
    row: ManifestRow,
    cohort: tuple[ManifestRow, ...],
) -> ManifestRow | None:
    candidates = sorted(
        (
            candidate
            for candidate in cohort
            if candidate.speaker_id == row.speaker_id
            and candidate.utterance_id != row.utterance_id
        ),
        key=lambda candidate: (candidate.utterance_id, candidate.clip_id),
    )
    return candidates[0] if candidates else None


def select_primaries_and_support(
    cohort: tuple[ManifestRow, ...],
    limit_clips: int,
) -> tuple[tuple[ManifestRow, ...], tuple[ManifestRow, ...]]:
    primaries = cohort[: limit_clips or None]
    negatives = tuple(
        negative
        for row in primaries
        if (negative := select_wrong_utterance(row, cohort)) is not None
    )
    support_by_clip = {row.clip_id: row for row in (*primaries, *negatives)}
    return primaries, tuple(support_by_clip.values())


__all__ = [
    "WindowPlan",
    "WindowPlanError",
    "mel_bounds",
    "paired_anchor",
    "plan_common_anchors",
    "select_wrong_utterance",
    "select_primaries_and_support",
]
