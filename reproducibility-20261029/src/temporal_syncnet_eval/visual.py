from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .contracts import CohortRow, ContractError
from .video_timing import cfr_pts_max_error, probe_frame_pts, select_frame_indices


@dataclass(frozen=True, slots=True)
class PreparedVisual:
    tensor: torch.Tensor
    source_frame_indices: tuple[int, ...]
    selected_timestamps_s: tuple[float, ...]
    crop_xyxy: tuple[int, int, int, int]
    motion_value: float
    pts_cfr_max_error_s: float


class VisualPreparer:
    def __init__(self) -> None:
        from frozen_korean_syncnet.media import FaceAligner

        self._aligner = FaceAligner()
        self._pts_cache: dict[Path, tuple[float, ...]] = {}

    def prepare(self, row: CohortRow, video_start_s: float) -> PreparedVisual:
        from frozen_korean_syncnet.media import (
            _decode_rgb,
            _official_visual,
        )

        pts = self._pts_cache.get(row.video_path)
        if pts is None:
            pts = probe_frame_pts(row.video_path)
            self._pts_cache[row.video_path] = pts
        indices, timestamps = select_frame_indices(pts, video_start_s)
        frames = _decode_rgb(row.video_path, indices)
        self._aligner.reset()
        aligned_and_boxes = tuple(self._aligner.align(frame) for frame in frames)
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
            indices,
            timestamps,
            crop,
            motion,
            cfr_pts_max_error(pts, row.source_fps),
        )


def save_visual(root: Path, window_id: str, prepared: PreparedVisual) -> Path:
    visual_dir = root / "visual"
    visual_dir.mkdir(exist_ok=True)
    path = (visual_dir / f"{window_id.replace(':', '_')}.pt").resolve()
    torch.save(prepared.tensor.cpu(), path)
    return path


__all__ = ["PreparedVisual", "VisualPreparer", "save_visual"]
