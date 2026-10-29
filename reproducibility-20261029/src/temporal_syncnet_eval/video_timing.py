from __future__ import annotations

import bisect
import math
from pathlib import Path
import subprocess

from .contracts import ContractError
from .timing import VIDEO_FPS


def probe_frame_pts(video_path: Path) -> tuple[float, ...]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ContractError("ffprobe_unavailable") from error
    if result.returncode != 0:
        raise ContractError("video_pts_probe_failed")
    try:
        values = tuple(float(line.strip().split(",")[0]) for line in result.stdout.splitlines() if line.strip())
    except ValueError as error:
        raise ContractError("video_pts_malformed") from error
    if not values or not all(math.isfinite(value) for value in values):
        raise ContractError("video_pts_malformed")
    if any(later <= earlier for earlier, later in zip(values, values[1:], strict=False)):
        raise ContractError("video_pts_not_strictly_increasing")
    return values


def select_frame_indices(
    pts: tuple[float, ...],
    video_start_s: float,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    targets = tuple(video_start_s + frame / VIDEO_FPS for frame in range(16))
    indices: list[int] = []
    for target in targets:
        right = bisect.bisect_left(pts, target)
        candidates = tuple(index for index in (right - 1, right) if 0 <= index < len(pts))
        if not candidates:
            raise ContractError("visual_pts_support_out_of_bounds")
        indices.append(min(candidates, key=lambda index: (abs(pts[index] - target), index)))
    if len(set(indices)) != len(indices):
        raise ContractError("visual_pts_selection_would_duplicate_frames")
    selected = tuple(pts[index] for index in indices)
    return tuple(indices), selected


def cfr_pts_max_error(pts: tuple[float, ...], source_fps: float) -> float:
    origin = pts[0]
    return max(abs(value - (origin + index / source_fps)) for index, value in enumerate(pts))


__all__ = ["cfr_pts_max_error", "probe_frame_pts", "select_frame_indices"]
