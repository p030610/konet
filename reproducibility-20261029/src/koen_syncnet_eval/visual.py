from __future__ import annotations

import bisect
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch
from typing_extensions import assert_never

from temporal_syncnet_eval.visual import PreparedVisual

from .contracts import CohortWindow, ContractError
from .policy import VisualPolicy


@dataclass(frozen=True, slots=True)
class SourceGeometry:
    width: int
    height: int
    display_aspect_ratio: str


@dataclass(frozen=True, slots=True)
class FrameSelection:
    indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    cfr_max_error_s: float
    geometry: SourceGeometry | None = None


def probe_bounded_frames(
    video_path: Path,
    window_start_s: float,
    source_fps: float,
    visual_policy: VisualPolicy = VisualPolicy.NATIVE,
) -> FrameSelection:
    interval_start = max(0.0, window_start_s - 1.0)
    interval_end = window_start_s + 1.64
    try:
        result = subprocess.run(
            (
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-read_intervals",
                f"{interval_start:.9f}%{interval_end:.9f}",
                "-show_entries",
                "stream=start_time,avg_frame_rate,r_frame_rate,time_base,width,height,display_aspect_ratio:frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(video_path),
            ),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ContractError("ffprobe_unavailable") from error
    if result.returncode != 0:
        raise ContractError("video_pts_probe_failed")
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        origin = float(stream["start_time"])
        average_fps = float(Fraction(stream["avg_frame_rate"]))
        reported_fps = float(Fraction(stream["r_frame_rate"]))
        time_base_s = float(Fraction(stream["time_base"]))
        pts = tuple(float(frame["best_effort_timestamp_time"]) for frame in payload["frames"])
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as error:
        raise ContractError("video_pts_malformed") from error
    geometry: SourceGeometry | None = None
    match visual_policy:
        case VisualPolicy.NATIVE:
            pass
        case VisualPolicy.NORMALIZE_1280X720:
            try:
                width = int(stream["width"])
                height = int(stream["height"])
                display_aspect_ratio = str(stream["display_aspect_ratio"])
                numerator_text, denominator_text = display_aspect_ratio.split(":", maxsplit=1)
                display_fraction = Fraction(int(numerator_text), int(denominator_text))
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                raise ContractError("video_geometry_header_malformed") from error
            if width < 1280 or height < 720:
                raise ContractError("video_geometry_below_1280x720")
            if display_fraction != Fraction(16, 9):
                raise ContractError("video_geometry_not_16_9")
            geometry = SourceGeometry(width, height, display_aspect_ratio)
        case unreachable:
            assert_never(unreachable)
    if not pts or not math.isfinite(origin) or not all(math.isfinite(value) for value in pts):
        raise ContractError("video_pts_malformed")
    if any(later <= earlier for earlier, later in zip(pts, pts[1:], strict=False)):
        raise ContractError("video_pts_not_strictly_increasing")
    fps_tolerance = max(1e-6, time_base_s * source_fps)
    if not math.isclose(average_fps, source_fps, abs_tol=fps_tolerance):
        raise ContractError("video_stream_not_cfr")
    if not math.isclose(reported_fps, source_fps, abs_tol=fps_tolerance):
        raise ContractError("video_stream_not_cfr")
    frame_period = 1.0 / source_fps
    if any(
        not math.isclose(later - earlier, frame_period, abs_tol=time_base_s + 1e-6)
        for earlier, later in zip(pts, pts[1:], strict=False)
    ):
        raise ContractError("video_stream_not_cfr")
    targets = tuple(window_start_s + frame / 25.0 for frame in range(16))
    local_indices: list[int] = []
    for target in targets:
        right = bisect.bisect_left(pts, target)
        candidates = tuple(index for index in (right - 1, right) if 0 <= index < len(pts))
        if not candidates:
            raise ContractError("visual_pts_support_out_of_bounds")
        local_indices.append(min(candidates, key=lambda index: (abs(pts[index] - target), index)))
    timestamps = tuple(pts[index] for index in local_indices)
    indices = tuple(round((timestamp - origin) * source_fps) for timestamp in timestamps)
    if len(set(indices)) != len(indices):
        raise ContractError("visual_pts_selection_would_duplicate_frames")
    errors = tuple(
        abs(timestamp - (origin + index / source_fps))
        for index, timestamp in zip(indices, timestamps, strict=True)
    )
    if max(errors) > time_base_s + 1e-6:
        raise ContractError("bounded_pts_not_cfr_indexable")
    return FrameSelection(indices, timestamps, max(errors), geometry)


def decode_rgb_single_pass(
    video_path: Path,
    indices: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    capture = cv2.VideoCapture(str(video_path))
    frames: dict[int, np.ndarray] = {}
    first, last = min(indices), max(indices)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for index in range(first, last + 1):
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise ContractError(f"video_frame_decode_failed:{index}")
            if index in indices:
                frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
    if len(frames) != len(indices):
        raise ContractError("video_frame_decode_incomplete")
    return tuple(frames[index] for index in indices)


def normalize_for_alignment(
    frames: tuple[np.ndarray, ...],
    geometry: SourceGeometry,
) -> tuple[np.ndarray, ...]:
    normalized: list[np.ndarray] = []
    for frame in frames:
        if frame.shape != (geometry.height, geometry.width, 3):
            raise ContractError("decoded_frame_geometry_differs_from_header")
        if geometry.width == 1280 and geometry.height == 720:
            normalized.append(frame)
            continue
        normalized.append(cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA))
    return tuple(normalized)


class BoundedVisualPreparer:
    def __init__(self, visual_policy: VisualPolicy = VisualPolicy.NATIVE) -> None:
        from frozen_korean_syncnet.media import FaceAligner

        self._aligner = FaceAligner(cpu_threads=2)
        self._visual_policy = visual_policy

    def prepare(self, row: CohortWindow, video_start_s: float) -> PreparedVisual:
        from frozen_korean_syncnet.media import _official_visual

        selection = probe_bounded_frames(
            row.video_path, video_start_s, row.source_video_fps, self._visual_policy
        )
        frames = decode_rgb_single_pass(row.video_path, selection.indices)
        match self._visual_policy:
            case VisualPolicy.NATIVE:
                alignment_frames = frames
            case VisualPolicy.NORMALIZE_1280X720:
                if selection.geometry is None:
                    raise ContractError("normalized_policy_geometry_unavailable")
                alignment_frames = normalize_for_alignment(frames, selection.geometry)
            case unreachable:
                assert_never(unreachable)
        self._aligner.reset()
        aligned_and_boxes = tuple(self._aligner.align(frame) for frame in alignment_frames)
        visual = _official_visual(tuple(value[0] for value in aligned_and_boxes))
        if visual.shape != (1, 48, 128, 256) or not bool(torch.isfinite(visual).all()):
            raise ContractError("official_visual_shape_or_finiteness_invalid")
        boxes = tuple(value[1] for value in aligned_and_boxes)
        crop = (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )
        frames_tensor = visual.reshape(16, 3, 128, 256)
        motion = float((frames_tensor[1:] - frames_tensor[:-1]).abs().mean().item())
        return PreparedVisual(
            visual,
            selection.indices,
            selection.timestamps_s,
            crop,
            motion,
            selection.cfr_max_error_s,
        )


__all__ = [
    "BoundedVisualPreparer",
    "FrameSelection",
    "SourceGeometry",
    "decode_rgb_single_pass",
    "normalize_for_alignment",
    "probe_bounded_frames",
]
