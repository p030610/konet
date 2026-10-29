from __future__ import annotations

import wave
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from pose_aware_roi_generator.dataset import _latentsync_melspectrogram

from .syncnet_utility_eval_types_v1 import EvalError


@dataclass(frozen=True, slots=True)
class BoundedPairAssetCache:
    capacity: int = 2
    audio: OrderedDict[Path, tuple[int, int, torch.Tensor]] = field(
        default_factory=OrderedDict,
        init=False,
    )
    landmarks: OrderedDict[Path, np.ndarray] = field(
        default_factory=OrderedDict,
        init=False,
    )

    def load_audio(self, path: Path) -> tuple[int, int, torch.Tensor]:
        cached = self.audio.get(path)
        if cached is None:
            try:
                with wave.open(str(path), "rb") as source:
                    sample_frames = source.getnframes()
                    sample_rate = source.getframerate()
                cached = (sample_frames, sample_rate, _latentsync_melspectrogram(path))
            except (OSError, ValueError, wave.Error) as error:
                raise EvalError("unreadable SyncNet audio") from error
            self.audio[path] = cached
        self.audio.move_to_end(path)
        while len(self.audio) > self.capacity:
            self.audio.popitem(last=False)
        return cached

    def load_landmark_windows(
        self,
        crop_root: Path,
        positive_start: int,
        negative_start: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        landmarks_path = (crop_root / "face_landmarks_xy.npy").resolve(strict=True)
        if not landmarks_path.is_relative_to(crop_root):
            raise EvalError("landmark path escapes clip root")
        landmarks = self.landmarks.get(landmarks_path)
        if landmarks is None:
            try:
                landmarks = np.load(landmarks_path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise EvalError("unreadable landmark alignment assets") from error
            if landmarks.ndim != 3 or landmarks.shape[1:] != (478, 2):
                raise EvalError("face landmarks must have shape framesx478x2")
            self.landmarks[landmarks_path] = landmarks
        self.landmarks.move_to_end(landmarks_path)
        while len(self.landmarks) > self.capacity:
            self.landmarks.popitem(last=False)
        required_stop = max(positive_start, negative_start) + 16
        if len(landmarks) < required_stop:
            raise EvalError("face landmarks do not contain requested frames")
        positive = np.asarray(landmarks[positive_start:positive_start + 16], dtype=np.float32)
        negative = np.asarray(landmarks[negative_start:negative_start + 16], dtype=np.float32)
        if not np.isfinite(positive).all() or not np.isfinite(negative).all():
            raise EvalError("face landmarks must be finite")
        affine_path = (crop_root / "crop_to_face_affine.npy").resolve(strict=True)
        if not affine_path.is_relative_to(crop_root):
            raise EvalError("affine path escapes clip root")
        try:
            affine = np.load(affine_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise EvalError("unreadable landmark alignment assets") from error
        if affine.shape != (2, 3) or not np.isfinite(affine).all():
            raise EvalError("crop-to-face affine must be finite 2x3")
        return (
            torch.from_numpy(positive.copy()),
            torch.from_numpy(negative.copy()),
            torch.from_numpy(np.asarray(affine, dtype=np.float32).copy()),
        )
