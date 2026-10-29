from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from .structure_dataset_v4 import CachedTextureDatasetV4


class AudioCachedTextureDatasetV4I(CachedTextureDatasetV4):
    """Coordinate-safe V4 texture samples with centered wav2vec-L6 windows."""

    def __init__(self, *args, audio_window_size: int = 17, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if audio_window_size < 3 or audio_window_size % 2 != 1:
            raise ValueError("audio_window_size must be an odd integer >= 3")
        self.audio_window_size = int(audio_window_size)
        self._audio_arrays: OrderedDict[int, np.ndarray] = OrderedDict()
        missing = [
            self.rows[index]["clip_id"] for index in self.clip_indices
            if not self.rows[index].get("audio_features")
        ]
        if missing:
            raise FileNotFoundError(f"Missing wav2vec features for {len(missing)} clips")

    def _audio(self, clip_index: int) -> np.ndarray:
        cached = self._audio_arrays.get(clip_index)
        if cached is None:
            cached = np.load(self.rows[clip_index]["audio_features"], mmap_mode="r")
            if cached.ndim != 2 or cached.shape[1] != 768:
                raise ValueError(
                    f"Expected wav2vec [T,768], got {cached.shape}: "
                    f"{self.rows[clip_index]['audio_features']}"
                )
            self._audio_arrays[clip_index] = cached
            while len(self._audio_arrays) > self.array_cache_size:
                self._audio_arrays.popitem(last=False)
        else:
            self._audio_arrays.move_to_end(clip_index)
        return cached

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = super().__getitem__(index)
        clip_index, frame, _ = self.records[index]
        row = self.rows[clip_index]
        audio = self._audio(clip_index)
        center = int(round(frame * max(len(audio) - 1, 0) / max(int(row["frame_count"]) - 1, 1)))
        radius = self.audio_window_size // 2
        indices = np.clip(np.arange(center - radius, center + radius + 1), 0, len(audio) - 1)
        sample["audio_window"] = torch.from_numpy(
            np.asarray(audio[indices], dtype=np.float32).copy()
        )
        sample["audio_center_index"] = center
        return sample


__all__ = ["AudioCachedTextureDatasetV4I"]
