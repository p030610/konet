from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_dataset_v4i import AudioCachedTextureDatasetV4I


def resolve_clip_audio(
    row: dict,
    hdtf_audio_roots: tuple[Path, ...],
    voxceleb2_audio_root: Path,
) -> Path:
    clip_id = str(row["clip_id"])
    identity = str(row["identity"])
    if str(row["domain"]).lower() == "hdtf":
        candidates = [root / f"{identity}_16khz" / f"{clip_id}.wav" for root in hdtf_audio_roots]
    elif str(row["domain"]).lower() == "voxceleb2":
        candidates = [voxceleb2_audio_root / f"{identity.upper()}_16khz" / f"{clip_id}.wav"]
    else:
        raise ValueError(f"Unsupported audio domain: {row['domain']}")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No audio for {row['domain']}:{clip_id}; checked {candidates}")
    return path


def crop_latentsync_mel(
    mel: torch.Tensor,
    start_frame: int,
    fps: float,
    window_length: int = 52,
) -> torch.Tensor:
    start = int(80.0 * float(start_frame) / float(fps))
    end = start + int(window_length)
    if start < 0:
        left = torch.full((mel.shape[0], -start), -4.0, dtype=mel.dtype)
        value = torch.cat([left, mel[:, :max(end, 0)]], dim=1)
    else:
        value = mel[:, start:end]
    if value.shape[1] < window_length:
        right = torch.full((mel.shape[0], window_length - value.shape[1]), -4.0, dtype=mel.dtype)
        value = torch.cat([value, right], dim=1)
    return value[:, :window_length].unsqueeze(0).contiguous()


class StableSyncSequenceDatasetV4J(Dataset):
    """Contiguous R7C samples plus aligned, shifted, and cross-speaker mel."""

    def __init__(
        self,
        *args,
        sequence_length: int = 16,
        sequence_stride: int = 16,
        shift_frames: int = 8,
        sync_offset_hdtf: int = 8,
        sync_offset_voxceleb2: int = 4,
        cross_negative_count: int = 3,
        mel_cache_dir: str | Path,
        hdtf_audio_roots: tuple[str | Path, ...] = (
            r"C:\LipSyncAnim_HDTF_Cache\raw\audio",
            r"D:\LipSyncAnim_HDTF\raw\audio",
        ),
        voxceleb2_audio_root: str | Path = r"C:\LipSyncAnim_VoxCeleb2_Work\audio_16k",
        mel_memory_cache_size: int = 8,
        **kwargs,
    ) -> None:
        self.frames = AudioCachedTextureDatasetV4I(*args, **kwargs)
        self.sequence_length = int(sequence_length)
        self.sequence_stride = max(1, int(sequence_stride))
        self.shift_frames = int(shift_frames)
        self.sync_offsets = {"hdtf": int(sync_offset_hdtf), "voxceleb2": int(sync_offset_voxceleb2)}
        self.cross_negative_count = max(1, int(cross_negative_count))
        if self.sequence_length != 16:
            raise ValueError("Stable SyncNet requires sequence_length=16")
        self.mel_cache_dir = Path(mel_cache_dir)
        self.mel_cache_dir.mkdir(parents=True, exist_ok=True)
        self.hdtf_audio_roots = tuple(Path(root) for root in hdtf_audio_roots)
        self.voxceleb2_audio_root = Path(voxceleb2_audio_root)
        self.mel_memory_cache_size = max(1, int(mel_memory_cache_size))
        self._mel_memory: OrderedDict[int, torch.Tensor] = OrderedDict()

        by_clip: dict[int, dict[int, int]] = {}
        for sample_index, (clip_index, frame, _) in enumerate(self.frames.records):
            by_clip.setdefault(clip_index, {})[frame] = sample_index
        sequences: list[tuple[int, int, tuple[int, ...]]] = []
        for clip_index, lookup in sorted(by_clip.items()):
            frames = sorted(lookup)
            if len(frames) < self.sequence_length:
                continue
            first, last = frames[0], frames[-1] - self.sequence_length + 1
            for start in range(first, last + 1, self.sequence_stride):
                values = tuple(lookup.get(start + offset, -1) for offset in range(self.sequence_length))
                if all(value >= 0 for value in values):
                    sequences.append((clip_index, start, values))
        if not sequences:
            raise ValueError("No contiguous 16-frame sequences exist in the transition cache")
        self.sequences = sequences
        self.domain_to_indices: dict[str, list[int]] = {}
        for index, (clip_index, _, _) in enumerate(sequences):
            domain = str(self.frames.rows[clip_index]["domain"])
            self.domain_to_indices.setdefault(domain, []).append(index)

    def __len__(self) -> int:
        return len(self.sequences)

    def _mel(self, clip_index: int) -> torch.Tensor:
        value = self._mel_memory.get(clip_index)
        if value is not None:
            self._mel_memory.move_to_end(clip_index)
            return value
        row = self.frames.rows[clip_index]
        cache = self.mel_cache_dir / str(row["domain"]) / f"{row['clip_id']}_mel.pt"
        if cache.is_file():
            value = torch.load(cache, map_location="cpu", weights_only=True)
        else:
            from pose_aware_roi_generator.dataset import _latentsync_melspectrogram

            audio = resolve_clip_audio(row, self.hdtf_audio_roots, self.voxceleb2_audio_root)
            value = _latentsync_melspectrogram(audio)
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(value, cache)
        self._mel_memory[clip_index] = value
        while len(self._mel_memory) > self.mel_memory_cache_size:
            self._mel_memory.popitem(last=False)
        return value

    def _cross_sequence_indices(self, index: int, clip_index: int) -> list[int]:
        row = self.frames.rows[clip_index]
        candidates = [
            value for value in self.domain_to_indices[str(row["domain"])]
            if self.sequences[value][0] != clip_index
            and self.frames.rows[self.sequences[value][0]]["identity"] != row["identity"]
        ]
        if not candidates:
            candidates = [value for value in range(len(self.sequences)) if self.sequences[value][0] != clip_index]
        digest = hashlib.sha256(f"r7d-cross:{row['clip_id']}:{index}".encode()).digest()
        start = int.from_bytes(digest[:4], "little") % len(candidates)
        step = max(1, int.from_bytes(digest[4:8], "little") % len(candidates))
        selected: list[int] = []
        cursor = start
        while len(selected) < min(self.cross_negative_count, len(candidates)):
            value = candidates[cursor % len(candidates)]
            if value not in selected:
                selected.append(value)
            cursor += step
            if cursor % len(candidates) == start and len(selected) < min(self.cross_negative_count, len(candidates)):
                cursor += 1
        return selected

    def _wav2vec_window(self, clip_index: int, frame: int) -> torch.Tensor:
        row = self.frames.rows[clip_index]
        audio = self.frames._audio(clip_index)
        center = int(round(frame * max(len(audio) - 1, 0) / max(int(row["frame_count"]) - 1, 1)))
        radius = self.frames.audio_window_size // 2
        indices = np.clip(np.arange(center - radius, center + radius + 1), 0, len(audio) - 1)
        return torch.from_numpy(np.asarray(audio[indices], dtype=np.float32).copy())

    def __getitem__(self, index: int) -> dict:
        clip_index, start, frame_indices = self.sequences[index]
        samples = [self.frames[value] for value in frame_indices]
        result: dict = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            result[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
        row = self.frames.rows[clip_index]
        fps = float(row.get("fps", 25.0))
        mel = self._mel(clip_index)
        sync_offset = self.sync_offsets[str(row["domain"]).lower()]
        result["matched_mel"] = crop_latentsync_mel(mel, start + sync_offset, fps)
        result["shifted_mels"] = torch.stack([
            crop_latentsync_mel(mel, start + sync_offset - self.shift_frames, fps),
            crop_latentsync_mel(mel, start + sync_offset + self.shift_frames, fps),
        ])
        result["shifted_audio_window"] = torch.stack([
            self._wav2vec_window(clip_index, start + offset + self.shift_frames)
            for offset in range(self.sequence_length)
        ])
        cross_rows = []
        cross_mels = []
        for cross_index in self._cross_sequence_indices(index, clip_index):
            cross_clip_index, cross_start, _ = self.sequences[cross_index]
            cross_row = self.frames.rows[cross_clip_index]
            cross_offset = self.sync_offsets[str(cross_row["domain"]).lower()]
            cross_mels.append(crop_latentsync_mel(
                self._mel(cross_clip_index), cross_start + cross_offset,
                float(cross_row.get("fps", 25.0)),
            ))
            cross_rows.append(cross_row)
        result["cross_mels"] = torch.stack(cross_mels)
        result["sequence_start"] = start
        result["sync_offset_frames"] = sync_offset
        result["cross_clip_id"] = [str(value["clip_id"]) for value in cross_rows]
        return result


__all__ = ["StableSyncSequenceDatasetV4J", "crop_latentsync_mel", "resolve_clip_audio"]
