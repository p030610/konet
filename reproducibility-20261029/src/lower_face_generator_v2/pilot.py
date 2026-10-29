from __future__ import annotations

import bisect
import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .dataset import LowerFaceCompletionDatasetV2
from .indexing import LowerFaceTrainingIndexV2


def stable_key(seed: int, domain: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{domain}:{identity}".encode("utf-8")).hexdigest()


def build_mixed_50_split(
    index_path: str | Path,
    output_path: str | Path,
    *,
    hdtf_train: int = 30,
    voxceleb2_train: int = 20,
    validation_per_domain: int = 5,
    minimum_frames: int = 300,
    seed: int = 325,
) -> dict:
    index = LowerFaceTrainingIndexV2.load(index_path)
    arrays = index.arrays
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for clip_index, (domain, identity, split) in enumerate(
        zip(arrays["clip_domains"], arrays["clip_identities"], arrays["clip_splits"])
    ):
        groups[(str(split), str(domain), str(identity))].append(clip_index)

    def select(split: str, domain: str, count: int) -> list[str]:
        eligible = []
        for (value_split, value_domain, identity), clips in groups.items():
            frames = int(arrays["clip_frame_counts"][clips].sum())
            if value_split == split and value_domain == domain and frames >= minimum_frames:
                eligible.append(identity)
        eligible.sort(key=lambda identity: stable_key(seed, domain, identity))
        if len(eligible) < count:
            raise ValueError(f"Only {len(eligible)} eligible {split}/{domain} identities; need {count}")
        return eligible[:count]

    train = {
        "hdtf": select("train", "hdtf", hdtf_train),
        "voxceleb2": select("train", "voxceleb2", voxceleb2_train),
    }
    val = {
        "hdtf": select("val", "hdtf", validation_per_domain),
        "voxceleb2": select("val", "voxceleb2", validation_per_domain),
    }
    all_keys = {(domain, identity) for domain, values in train.items() for identity in values}
    val_keys = {(domain, identity) for domain, values in val.items() for identity in values}
    if all_keys & val_keys:
        raise ValueError("Pilot train/validation identity leakage")

    def clips_for(split: str, selected: dict[str, list[str]]) -> list[int]:
        result = []
        for domain, identities in selected.items():
            for identity in identities:
                result.extend(groups[(split, domain, identity)])
        return sorted(result)

    train_clips = clips_for("train", train)
    val_clips = clips_for("val", val)
    payload = {
        "stage": "lower_face_generator_v2_mixed_50_identity_pilot",
        "seed": seed,
        "index": str(Path(index_path).resolve()),
        "minimum_frames": minimum_frames,
        "train_identities": train,
        "validation_identities": val,
        "train_identity_count": sum(map(len, train.values())),
        "validation_identity_count": sum(map(len, val.values())),
        "train_clip_indices": train_clips,
        "validation_clip_indices": val_clips,
        "train_frames": int(arrays["clip_frame_counts"][train_clips].sum()),
        "validation_frames": int(arrays["clip_frame_counts"][val_clips].sum()),
        "identity_leaks": 0,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_full_dataset_split(
    index_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Export every indexed train/validation clip without identity leakage."""
    index = LowerFaceTrainingIndexV2.load(index_path)
    arrays = index.arrays
    train_clips = np.flatnonzero(arrays["clip_splits"] == "train").astype(int).tolist()
    val_clips = np.flatnonzero(arrays["clip_splits"] == "val").astype(int).tolist()

    def identities(clips: list[int]) -> dict[str, list[str]]:
        grouped: dict[str, set[str]] = defaultdict(set)
        for clip in clips:
            grouped[str(arrays["clip_domains"][clip])].add(
                str(arrays["clip_identities"][clip])
            )
        return {domain: sorted(values) for domain, values in sorted(grouped.items())}

    train_identities = identities(train_clips)
    validation_identities = identities(val_clips)
    train_keys = {
        (domain, identity)
        for domain, values in train_identities.items()
        for identity in values
    }
    validation_keys = {
        (domain, identity)
        for domain, values in validation_identities.items()
        for identity in values
    }
    leaks = sorted(train_keys & validation_keys)
    if leaks:
        raise ValueError(f"Full split contains identity leakage: {leaks[:5]}")

    payload = {
        "stage": "lower_face_generator_v2_full_mixed_domain_training",
        "index": str(Path(index_path).resolve()),
        "train_identities": train_identities,
        "validation_identities": validation_identities,
        "train_identity_count": len(train_keys),
        "validation_identity_count": len(validation_keys),
        "train_clip_indices": train_clips,
        "validation_clip_indices": val_clips,
        "train_clip_count": len(train_clips),
        "validation_clip_count": len(val_clips),
        "train_frames": int(arrays["clip_frame_counts"][train_clips].sum()),
        "validation_frames": int(arrays["clip_frame_counts"][val_clips].sum()),
        "identity_leaks": 0,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _image(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)


class LowerFaceGeometryProbeDataset(Dataset):
    def __init__(
        self,
        index_path: str | Path,
        split_path: str | Path,
        split: str,
        cache_size: int = 16,
    ) -> None:
        self.index = LowerFaceTrainingIndexV2.load(index_path)
        self.rows = [
            json.loads(line)
            for line in Path(str(self.index.arrays["manifest"].item())).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pilot = json.loads(Path(split_path).read_text(encoding="utf-8"))
        key = "train_clip_indices" if split == "train" else "validation_clip_indices"
        self.global_clip_indices = [int(value) for value in pilot[key]]
        self.frame_counts = np.asarray(
            [int(self.rows[value]["frame_count"]) for value in self.global_clip_indices], dtype=np.int64
        )
        self.ends = np.cumsum(self.frame_counts, dtype=np.int64)
        self.starts = np.concatenate([np.zeros(1, dtype=np.int64), self.ends[:-1]])
        self.cache_size = max(1, int(cache_size))
        self.cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.identity_to_clips: dict[tuple[str, str], list[int]] = defaultdict(list)
        for local_clip, global_clip in enumerate(self.global_clip_indices):
            row = self.rows[global_clip]
            self.identity_to_clips[(row["domain"], row["identity"])].append(local_clip)
        self.identities = sorted(self.identity_to_clips)

    def __len__(self) -> int:
        return int(self.ends[-1])

    def locate(self, index: int) -> tuple[int, int]:
        local_clip = bisect.bisect_right(self.ends, index)
        return local_clip, int(index - self.starts[local_clip])

    def global_index(self, local_clip: int, frame: int) -> int:
        return int(self.starts[local_clip] + frame)

    def _arrays(self, local_clip: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self.cache.get(local_clip)
        if cached is None:
            row = self.rows[self.global_clip_indices[local_clip]]
            cached = (
                np.load(row["lower_face_points"], mmap_mode="r"),
                np.load(row["geometry81"], mmap_mode="r"),
            )
            self.cache[local_clip] = cached
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(local_clip)
        return cached

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        local_clip, frame = self.locate(index)
        row = self.rows[self.global_clip_indices[local_clip]]
        points, geometry81 = self._arrays(local_clip)
        target_path = Path(row["root"]) / "crops" / f"{row['clip_id']}_f{frame:06d}.jpg"
        normalized = np.asarray(points[frame], dtype=np.float32).copy()
        normalized[:, 0] /= 319.0
        normalized[:, 1] /= 255.0
        teeth = np.asarray(geometry81[frame, 80:81], dtype=np.float32).copy()
        return {
            "image": _image(target_path),
            "points_normalized": torch.from_numpy(normalized),
            "teeth_score": torch.from_numpy(teeth),
            "domain": row["domain"],
            "identity": row["identity"],
            "clip_id": row["clip_id"],
            "frame_index": frame,
        }


class IdentityBalancedSampler(Sampler[int]):
    def __init__(self, dataset: LowerFaceGeometryProbeDataset, samples: int, seed: int = 325) -> None:
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
        identities = list(self.dataset.identities)
        for sample_index in range(self.samples):
            if sample_index % len(identities) == 0:
                rng.shuffle(identities)
            identity = identities[sample_index % len(identities)]
            local_clips = self.dataset.identity_to_clips[identity]
            local_clip = int(rng.choice(local_clips))
            frame = int(rng.integers(0, self.dataset.frame_counts[local_clip]))
            yield self.dataset.global_index(local_clip, frame)


def deterministic_validation_indices(
    dataset: LowerFaceGeometryProbeDataset, samples_per_identity: int
) -> list[int]:
    result = []
    for identity in dataset.identities:
        clips = dataset.identity_to_clips[identity]
        candidates = []
        for local_clip in clips:
            frames = np.linspace(
                0, dataset.frame_counts[local_clip] - 1, samples_per_identity, dtype=np.int64
            )
            candidates.extend(dataset.global_index(local_clip, int(frame)) for frame in frames)
        result.extend(candidates[:samples_per_identity])
    return result


class Mixed50RendererDataset(Dataset):
    """Selected mixed-50 frames with an independent same-identity source frame."""

    def __init__(
        self,
        index_path: str | Path,
        split_path: str | Path,
        split: str,
        reference_count: int = 5,
        seed: int = 325,
        upper_boundary_mode: str = "legacy_horizontal",
        upper_overlap_px: int = 0,
        boundary_tracking_correction: float = 0.05,
        transition_safety_dilate_px: int = 3,
        transition_cache_dir: str | Path | None = None,
    ) -> None:
        base_split = "train" if split == "train" else "val"
        self.base = LowerFaceCompletionDatasetV2(
            index_path,
            split=base_split,
            reference_count=reference_count,
            metadata_cache_size=4 if upper_boundary_mode == "tracked_transition_v4" else 24,
            upper_boundary_mode=upper_boundary_mode,
            upper_overlap_px=upper_overlap_px,
            boundary_tracking_correction=boundary_tracking_correction,
            transition_safety_dilate_px=transition_safety_dilate_px,
            transition_cache_dir=transition_cache_dir,
        )
        pilot = json.loads(Path(split_path).read_text(encoding="utf-8"))
        key = "train_clip_indices" if split == "train" else "validation_clip_indices"
        selected = {int(value) for value in pilot[key]}
        self.local_clips = [index for index, clip in enumerate(self.base.clips) if clip.global_index in selected]
        if not self.local_clips:
            raise ValueError(f"No selected renderer clips for split {split!r}")
        self.frame_counts = np.asarray(
            [int(self.base.clips[value].row["frame_count"]) for value in self.local_clips], dtype=np.int64
        )
        self.ends = np.cumsum(self.frame_counts, dtype=np.int64)
        self.starts = np.concatenate([np.zeros(1, dtype=np.int64), self.ends[:-1]])
        self.base_starts = np.concatenate([np.zeros(1, dtype=np.int64), self.base.ends[:-1]])
        self.identity_to_clips: dict[tuple[str, str], list[int]] = defaultdict(list)
        for selected_clip, base_clip in enumerate(self.local_clips):
            row = self.base.clips[base_clip].row
            self.identity_to_clips[(row["domain"], row["identity"])].append(selected_clip)
        self.identities = sorted(self.identity_to_clips)
        self.seed = int(seed)

    def __len__(self) -> int:
        return int(self.ends[-1])

    def locate(self, index: int) -> tuple[int, int]:
        local_clip = bisect.bisect_right(self.ends, index)
        return local_clip, int(index - self.starts[local_clip])

    def global_index(self, local_clip: int, frame: int) -> int:
        return int(self.starts[local_clip] + frame)

    def _base_index(self, selected_clip: int, frame: int) -> int:
        return int(self.base_starts[self.local_clips[selected_clip]] + frame)

    def __getitem__(self, index: int) -> dict:
        selected_clip, frame = self.locate(index)
        target = self.base[self._base_index(selected_clip, frame)]
        digest = hashlib.sha256(f"{self.seed}:{index}".encode("ascii")).digest()
        # Keep identity, crop calibration, background, and coarse pose coherent.
        # A separated frame from the same clip removes target-mouth leakage while
        # preserving the real inference contract: edit articulation, not footage.
        condition_clip = selected_clip
        frame_count = int(self.frame_counts[condition_clip])
        offset = 8 + int.from_bytes(digest[:4], "little") % min(41, max(frame_count - 1, 1))
        direction = -1 if digest[4] & 1 else 1
        condition_frame = int(np.clip(frame + direction * offset, 0, frame_count - 1))
        if condition_frame == frame and frame_count > 1:
            condition_frame = frame - 1 if frame else 1
        condition = self.base[self._base_index(condition_clip, condition_frame)]
        target["source_rgb"] = condition["source_rgb"]
        target["source_masked"] = condition["source_rgb"] * (1.0 - target["edit_mask"])
        target["source_points_px"] = condition["source_points_px"]
        target["source_geometry_xyz"] = condition["source_geometry_xyz"]
        target["source_clip_id"] = condition["clip_id"]
        target["source_frame_index"] = condition["frame_index"]
        return target


class MixedSequenceRendererDataset(Dataset):
    """Contiguous target/source windows with one fixed reference bank."""

    def __init__(
        self,
        frame_dataset: Mixed50RendererDataset,
        sequence_length: int = 16,
    ) -> None:
        self.frame_dataset = frame_dataset
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 2:
            raise ValueError("Temporal sequences require at least two frames")
        self.counts = np.maximum(frame_dataset.frame_counts - self.sequence_length + 1, 0)
        self.ends = np.cumsum(self.counts, dtype=np.int64)
        self.starts = np.concatenate([np.zeros(1, dtype=np.int64), self.ends[:-1]])
        self.identity_to_clips = frame_dataset.identity_to_clips
        self.identities = frame_dataset.identities
        if not len(self.ends) or int(self.ends[-1]) <= 0:
            raise ValueError("No valid contiguous temporal sequences")

    def __len__(self) -> int:
        return int(self.ends[-1])

    def locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        selected_clip = bisect.bisect_right(self.ends, index)
        return selected_clip, int(index - self.starts[selected_clip])

    def global_index(self, selected_clip: int, start_frame: int) -> int:
        return int(self.starts[selected_clip] + start_frame)

    def condition_start(self, selected_clip: int, start_frame: int) -> int:
        frame_count = int(self.frame_dataset.frame_counts[selected_clip])
        latest = frame_count - self.sequence_length
        if latest <= 0:
            return 0
        digest = hashlib.sha256(
            f"temporal:{self.frame_dataset.seed}:{selected_clip}:{start_frame}".encode("ascii")
        ).digest()
        distance = self.sequence_length + 8 + int.from_bytes(digest[:2], "little") % 33
        candidates = [
            int(np.clip(start_frame - distance, 0, latest)),
            int(np.clip(start_frame + distance, 0, latest)),
        ]
        return max(candidates, key=lambda value: abs(value - start_frame))

    def __getitem__(self, index: int) -> dict:
        selected_clip, start_frame = self.locate(index)
        condition_start = self.condition_start(selected_clip, start_frame)
        base_clip = self.frame_dataset.local_clips[selected_clip]
        center = start_frame + self.sequence_length // 2
        center_sample = self.frame_dataset.base[
            self.frame_dataset._base_index(selected_clip, center)
        ]
        fixed_reference_keys = (
            "references_rgb",
            "reference_geometry_xyz",
            "reference_geometry81",
            "reference_points_px",
            "reference_clip_indices",
            "reference_frame_indices",
        )
        samples = []
        for offset in range(self.sequence_length):
            target = self.frame_dataset.base[
                self.frame_dataset._base_index(selected_clip, start_frame + offset)
            ]
            condition = self.frame_dataset.base[
                self.frame_dataset._base_index(selected_clip, condition_start + offset)
            ]
            target["source_rgb"] = condition["source_rgb"]
            target["source_masked"] = condition["source_rgb"] * (1.0 - target["edit_mask"])
            target["source_points_px"] = condition["source_points_px"]
            target["source_geometry_xyz"] = condition["source_geometry_xyz"]
            for key in fixed_reference_keys:
                target[key] = center_sample[key]
            samples.append(target)
        tensor_keys = [key for key, value in samples[0].items() if torch.is_tensor(value)]
        row = self.frame_dataset.base.clips[base_clip].row
        return {
            **{key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys},
            "domain": row["domain"],
            "identity": row["identity"],
            "clip_id": row["clip_id"],
            "start_frame": int(start_frame),
            "condition_start_frame": int(condition_start),
            "source_frame_indices": torch.arange(
                condition_start,
                condition_start + self.sequence_length,
                dtype=torch.int64,
            ),
        }


class IdentityBalancedSequenceSampler(Sampler[int]):
    def __init__(self, dataset: MixedSequenceRendererDataset, samples: int, seed: int = 325) -> None:
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
        identities = list(self.dataset.identities)
        for sample_index in range(self.samples):
            if sample_index % len(identities) == 0:
                rng.shuffle(identities)
            identity = identities[sample_index % len(identities)]
            clips = [
                value for value in self.dataset.identity_to_clips[identity]
                if self.dataset.counts[value] > 0
            ]
            if not clips:
                continue
            selected_clip = int(rng.choice(clips))
            start = int(rng.integers(0, self.dataset.counts[selected_clip]))
            yield self.dataset.global_index(selected_clip, start)


def deterministic_validation_sequence_indices(
    dataset: MixedSequenceRendererDataset,
    samples_per_identity: int,
) -> list[int]:
    result = []
    for identity in dataset.identities:
        candidates = []
        for selected_clip in dataset.identity_to_clips[identity]:
            count = int(dataset.counts[selected_clip])
            if count <= 0:
                continue
            starts = np.linspace(0, count - 1, samples_per_identity, dtype=np.int64)
            candidates.extend(
                dataset.global_index(selected_clip, int(start)) for start in starts
            )
        result.extend(candidates[:samples_per_identity])
    return result


__all__ = [
    "IdentityBalancedSampler",
    "IdentityBalancedSequenceSampler",
    "LowerFaceGeometryProbeDataset",
    "Mixed50RendererDataset",
    "MixedSequenceRendererDataset",
    "build_full_dataset_split",
    "build_mixed_50_split",
    "deterministic_validation_indices",
    "deterministic_validation_sequence_indices",
]
