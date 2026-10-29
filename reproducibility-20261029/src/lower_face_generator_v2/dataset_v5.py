from __future__ import annotations

import bisect
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .mask_cache_v5 import load_mask_chunk_v5


def _rows(path: str | Path, split: str) -> list[dict]:
    values = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [row for row in values if row["split"] == split]
    if not selected:
        raise ValueError(f"No V5 manifest rows for split {split!r}")
    for row in selected:
        if row.get("source_pairing_v5") != "same_frame_reconstruction":
            raise ValueError(
                "V5 texture training requires same_frame_reconstruction caches; "
                f"{row['clip_id']} uses {row.get('source_pairing_v5')!r}"
            )
    return selected


def _rgb(path: Path) -> torch.Tensor:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(value.transpose(2, 0, 1).copy()).float().div_(255.0)


class LowerFaceTextureDatasetV5(Dataset):
    """Pixel-aligned V5 reconstruction pairs with mouth-free references."""

    coordinate_mode = "same_frame_reconstruction"

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        *,
        reference_count: int = 5,
        chunk_cache_size: int = 12,
        metadata_cache_size: int = 8,
        image_cache_size: int = 96,
    ) -> None:
        self.manifest = Path(manifest)
        self.rows = _rows(self.manifest, split)
        self.reference_count = int(reference_count)
        self.chunk_cache_size = max(1, int(chunk_cache_size))
        self.metadata_cache_size = max(1, int(metadata_cache_size))
        self.image_cache_size = max(1, int(image_cache_size))
        counts = np.asarray([int(row["frame_count"]) for row in self.rows], dtype=np.int64)
        self.ends = np.cumsum(counts)
        self.starts = np.concatenate((np.zeros(1, dtype=np.int64), self.ends[:-1]))
        self._chunks: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._metadata: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._images: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self.domain_to_identities: dict[str, dict[str, list[int]]] = {}
        for clip_index, row in enumerate(self.rows):
            domain = str(row["domain"])
            identity = str(row["identity"])
            self.domain_to_identities.setdefault(domain, {}).setdefault(identity, []).append(
                clip_index
            )

    def __len__(self) -> int:
        return int(self.ends[-1])

    def locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        clip = bisect.bisect_right(self.ends, index)
        return clip, int(index - self.starts[clip])

    def global_index(self, clip_index: int, frame: int) -> int:
        return int(self.starts[clip_index] + int(frame))

    def _metadata_arrays(self, clip_index: int) -> dict[str, np.ndarray]:
        value = self._metadata.get(clip_index)
        if value is None:
            row = self.rows[clip_index]
            value = {
                "points": np.load(row["lower_face_points"], mmap_mode="r"),
                "geometry81": np.load(row["geometry81"], mmap_mode="r"),
            }
            self._metadata[clip_index] = value
            while len(self._metadata) > self.metadata_cache_size:
                self._metadata.popitem(last=False)
        else:
            self._metadata.move_to_end(clip_index)
        return value

    def _chunk(self, row: dict, frame: int) -> tuple[dict[str, np.ndarray], int]:
        chunk_frames = int(row["mask_cache_chunk_frames_v5"])
        chunk_index = int(frame) // chunk_frames
        local = int(frame) - chunk_index * chunk_frames
        path = Path(row["mask_cache_v5"]) / f"chunk_{chunk_index:06d}.npz"
        value = self._chunks.get(path)
        if value is None:
            value = load_mask_chunk_v5(path)
            self._chunks[path] = value
            while len(self._chunks) > self.chunk_cache_size:
                self._chunks.popitem(last=False)
        else:
            self._chunks.move_to_end(path)
        start = int(value["target_frame_start"].item())
        if start + local != int(frame):
            raise ValueError(f"V5 cache frame mismatch in {path}: {start}+{local} != {frame}")
        if int(value["source_frame_indices"][local]) != int(frame):
            raise ValueError(f"V5 training cache is not same-frame at {path}, frame {frame}")
        return value, local

    def _image_path(self, row: dict, frame: int) -> Path:
        return Path(row["root"]) / "crops" / f"{row['clip_id']}_f{int(frame):06d}.jpg"

    def _image(self, row: dict, frame: int) -> torch.Tensor:
        path = self._image_path(row, frame)
        value = self._images.get(path)
        if value is None:
            value = _rgb(path)
            self._images[path] = value
            while len(self._images) > self.image_cache_size:
                self._images.popitem(last=False)
        else:
            self._images.move_to_end(path)
        return value

    def _reference_frames(self, clip_index: int, frame: int) -> list[int]:
        row = self.rows[clip_index]
        points = self._metadata_arrays(clip_index)["points"]
        count = int(row["frame_count"])
        candidates = np.unique(
            np.linspace(0, count - 1, min(48, count)).round().astype(np.int64)
        )
        candidates = candidates[np.abs(candidates - frame) > 8]
        if not len(candidates):
            candidates = np.asarray([frame], dtype=np.int64)
        values = np.asarray(points[candidates], dtype=np.float32).reshape(len(candidates), -1)
        target = np.asarray(points[frame], dtype=np.float32).reshape(1, -1)
        scale = np.std(values, axis=0, keepdims=True).clip(1.0, None)
        normalized = (values - target) / scale
        selected = [int(np.linalg.norm(normalized, axis=1).argmin())]
        nearest = np.linalg.norm(normalized - normalized[selected[0]], axis=1)
        target_distance = np.linalg.norm(normalized, axis=1)
        while len(selected) < min(self.reference_count, len(candidates)):
            score = nearest + 0.1 * target_distance
            score[np.asarray(selected)] = -1
            choice = int(score.argmax())
            selected.append(choice)
            nearest = np.minimum(
                nearest, np.linalg.norm(normalized - normalized[choice], axis=1)
            )
        while len(selected) < self.reference_count:
            selected.append(selected[len(selected) % len(selected)])
        return [int(candidates[index]) for index in selected]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        clip_index, frame = self.locate(index)
        row = self.rows[clip_index]
        arrays = self._metadata_arrays(clip_index)
        chunk, local = self._chunk(row, frame)
        masks = {}
        for name, value in chunk.items():
            if name in ("source_frame_indices", "target_points_px", "target_frame_start"):
                continue
            frame_value = np.asarray(value[local], dtype=np.float32).copy()
            if name == "alpha_target" and np.issubdtype(value.dtype, np.integer):
                frame_value /= 255.0
            masks[name] = torch.from_numpy(frame_value)[None]
        target = self._image(row, frame)
        edit = masks["edit_union"]
        reference_frames = self._reference_frames(clip_index, frame)
        references = []
        references_skin = []
        reference_points = []
        for reference_frame in reference_frames:
            image = self._image(row, reference_frame)
            reference_chunk, reference_local = self._chunk(row, reference_frame)
            mouth = torch.from_numpy(
                np.asarray(
                    reference_chunk["target_lip_support"][reference_local], dtype=np.float32
                ).copy()
            )[None]
            references.append(image)
            references_skin.append(image * (1.0 - mouth))
            reference_points.append(
                torch.from_numpy(
                    np.asarray(arrays["points"][reference_frame], dtype=np.float32).copy()
                )
            )
        geometry81 = torch.from_numpy(
            np.asarray(arrays["geometry81"][frame], dtype=np.float32).copy()
        )
        target_points = torch.from_numpy(
            np.asarray(chunk["target_points_px"][local], dtype=np.float32).copy()
        )
        return {
            "source_rgb": target,
            "source_masked": target * (1.0 - edit),
            "target_rgb": target,
            "references_rgb": torch.stack(references),
            "references_skin_rgb": torch.stack(references_skin),
            "reference_points_px": torch.stack(reference_points),
            "source_points_px": torch.from_numpy(
                np.asarray(arrays["points"][frame], dtype=np.float32).copy()
            ),
            "target_points_px": target_points,
            "geometry81": geometry81,
            "teeth_score": geometry81[80:81],
            **{f"{name}_mask": value for name, value in masks.items()},
            "domain": str(row["domain"]),
            "identity": str(row["identity"]),
            "clip_id": str(row["clip_id"]),
            "frame_index": int(frame),
            "source_frame_index": int(frame),
            "reference_frame_indices": torch.tensor(reference_frames, dtype=torch.int64),
            "coordinate_mode": self.coordinate_mode,
        }


class DomainIdentityBalancedSamplerV5(Sampler[int]):
    def __init__(self, dataset: LowerFaceTextureDatasetV5, samples: int, seed: int = 523) -> None:
        self.dataset = dataset
        self.samples = int(samples)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        domains = sorted(self.dataset.domain_to_identities)
        counters = defaultdict(int)
        orders = {
            domain: list(self.dataset.domain_to_identities[domain]) for domain in domains
        }
        for values in orders.values():
            rng.shuffle(values)
        for sample in range(self.samples):
            domain = domains[sample % len(domains)]
            identities = orders[domain]
            identity = identities[counters[domain] % len(identities)]
            counters[domain] += 1
            clips = self.dataset.domain_to_identities[domain][identity]
            clip = int(rng.choice(clips))
            frame = int(rng.integers(0, int(self.dataset.rows[clip]["frame_count"])))
            yield self.dataset.global_index(clip, frame)


class DomainIdentityBalancedBatchSamplerV5(Sampler[list[int]]):
    """Balance domains/identities while grouping each batch within one clip."""

    def __init__(
        self,
        dataset: LowerFaceTextureDatasetV5,
        samples: int,
        batch_size: int,
        seed: int = 523,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.batches = int(samples) // self.batch_size
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size < 1 or self.batches < 1:
            raise ValueError("Grouped V5 sampling requires at least one complete batch")

    def __len__(self) -> int:
        return self.batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        domains = sorted(self.dataset.domain_to_identities)
        counters = defaultdict(int)
        orders = {
            domain: list(self.dataset.domain_to_identities[domain]) for domain in domains
        }
        for identities in orders.values():
            rng.shuffle(identities)
        for batch_index in range(self.batches):
            domain = domains[batch_index % len(domains)]
            identities = orders[domain]
            identity = identities[counters[domain] % len(identities)]
            counters[domain] += 1
            clips = self.dataset.domain_to_identities[domain][identity]
            clip = int(rng.choice(clips))
            frame_count = int(self.dataset.rows[clip]["frame_count"])
            replace = frame_count < self.batch_size
            frames = rng.choice(frame_count, size=self.batch_size, replace=replace)
            yield [self.dataset.global_index(clip, int(frame)) for frame in frames]


def deterministic_validation_indices_v5(
    dataset: LowerFaceTextureDatasetV5, samples_per_identity: int = 2
) -> list[int]:
    result = []
    for domain in sorted(dataset.domain_to_identities):
        for identity in sorted(dataset.domain_to_identities[domain]):
            clips = dataset.domain_to_identities[domain][identity]
            count = max(1, int(samples_per_identity))
            for value in range(count):
                clip = clips[value % len(clips)]
                frames = int(dataset.rows[clip]["frame_count"])
                frame = int(round((value + 1) * (frames - 1) / (count + 1)))
                result.append(dataset.global_index(clip, frame))
    return result


__all__ = [
    "DomainIdentityBalancedBatchSamplerV5",
    "DomainIdentityBalancedSamplerV5",
    "LowerFaceTextureDatasetV5",
    "deterministic_validation_indices_v5",
]
