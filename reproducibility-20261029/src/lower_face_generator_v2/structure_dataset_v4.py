from __future__ import annotations

import json
import hashlib
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .indexing import LowerFaceTrainingIndexV2
from .coordinate_contract import FrameSpaceBinding, validate_frame_space_binding


class CachedStructureDatasetV4(Dataset):
    """Geometry and cached structure masks without RGB/reference decoding."""

    def __init__(
        self,
        index_path: str | Path,
        split_path: str | Path,
        split: str,
        transition_cache_dir: str | Path,
        upper_overlap_px: int = 14,
        boundary_tracking_correction: float = 0.05,
        transition_safety_dilate_px: int = 3,
        array_cache_size: int = 32,
    ) -> None:
        index = LowerFaceTrainingIndexV2.load(index_path)
        manifest = Path(str(index.arrays["manifest"].item()))
        self.rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pilot = json.loads(Path(split_path).read_text(encoding="utf-8"))
        key = "train_clip_indices" if split == "train" else "validation_clip_indices"
        self.clip_indices = [int(value) for value in pilot[key]]
        self.transition_cache_dir = Path(transition_cache_dir)
        self.upper_overlap_px = int(upper_overlap_px)
        self.boundary_tracking_correction = float(boundary_tracking_correction)
        self.transition_safety_dilate_px = int(transition_safety_dilate_px)
        self.array_cache_size = max(1, int(array_cache_size))
        self._arrays: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._placement_arrays: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self.records: list[tuple[int, int, Path]] = []
        self.identity_to_indices: dict[tuple[str, str], list[int]] = defaultdict(list)

        for clip_index in self.clip_indices:
            row = self.rows[clip_index]
            directory = self._cache_path(row, 0).parent
            for path in sorted(directory.glob("f*.npz")):
                frame = int(path.stem[1:])
                if frame >= int(row["frame_count"]):
                    continue
                record_index = len(self.records)
                self.records.append((clip_index, frame, path))
                self.identity_to_indices[(str(row["domain"]), str(row["identity"]))].append(record_index)
        self.identities = sorted(self.identity_to_indices)
        self.domain_to_identities: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for identity in self.identities:
            self.domain_to_identities[identity[0]].append(identity)
        if not self.records:
            raise ValueError(f"No cached V4 structure frames for split {split!r}")

    def _cache_path(self, row: dict, frame: int) -> Path:
        stem = (
            f"{row['clip_id']}_o{self.upper_overlap_px}"
            f"_c{self.boundary_tracking_correction:.4f}"
            f"_d{self.transition_safety_dilate_px}"
        )
        return self.transition_cache_dir / str(row["domain"]) / stem / f"f{frame:06d}.npz"

    def __len__(self) -> int:
        return len(self.records)

    def _geometry_arrays(self, clip_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._arrays.get(clip_index)
        if cached is None:
            row = self.rows[clip_index]
            cached = (
                np.load(row["lower_face_points"], mmap_mode="r"),
                np.load(row["geometry81"], mmap_mode="r"),
            )
            self._arrays[clip_index] = cached
            while len(self._arrays) > self.array_cache_size:
                self._arrays.popitem(last=False)
        else:
            self._arrays.move_to_end(clip_index)
        return cached

    def _placement_metadata(self, clip_index: int) -> dict[str, np.ndarray]:
        cached = self._placement_arrays.get(clip_index)
        if cached is None:
            row = self.rows[clip_index]
            root = Path(row["root"])
            cached = {
                "geometry_xyz": np.load(row["lower_face_geometry"], mmap_mode="r"),
                "projection19": np.load(row["projection19"], mmap_mode="r"),
                "pose6": np.load(row["pose6"], mmap_mode="r"),
                "face_to_crop_affine": np.load(root / "face_to_crop_affine.npy"),
            }
            frame_count = int(row["frame_count"])
            expected = {
                "geometry_xyz": (frame_count, 93, 3),
                "projection19": (frame_count, 19),
                "pose6": (frame_count, 6),
                "face_to_crop_affine": (2, 3),
            }
            for name, shape in expected.items():
                if cached[name].shape != shape:
                    raise ValueError(
                        f"{row['clip_id']}: placement metadata {name} has "
                        f"shape {cached[name].shape}, expected {shape}"
                    )
            self._placement_arrays[clip_index] = cached
            while len(self._placement_arrays) > self.array_cache_size:
                self._placement_arrays.popitem(last=False)
        else:
            self._placement_arrays.move_to_end(clip_index)
        return cached

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        clip_index, frame, cache_path = self.records[index]
        row = self.rows[clip_index]
        points, geometry81 = self._geometry_arrays(clip_index)
        placement = self._placement_metadata(clip_index)
        with np.load(cache_path) as masks:
            missing = {"lip_oral", "motion", "edit"}.difference(masks.files)
            if missing:
                raise ValueError(f"Incomplete V4 structure cache {cache_path}: {sorted(missing)}")
            mask_tensors = {
                f"{name}_mask": torch.from_numpy(
                    np.asarray(masks[name], dtype=np.float32).copy() / 255.0
                )[None]
                for name in ("lip_oral", "motion", "edit")
            }
        return {
            "target_points_px": torch.from_numpy(
                np.asarray(points[frame], dtype=np.float32).copy()
            ),
            "geometry81": torch.from_numpy(
                np.asarray(geometry81[frame], dtype=np.float32).copy()
            ),
            "target_geometry_xyz": torch.from_numpy(
                np.asarray(placement["geometry_xyz"][frame], dtype=np.float32).copy()
            ),
            "projection19": torch.from_numpy(
                np.asarray(placement["projection19"][frame], dtype=np.float32).copy()
            ),
            "pose6": torch.from_numpy(
                np.asarray(placement["pose6"][frame], dtype=np.float32).copy()
            ),
            "face_to_crop_affine": torch.from_numpy(
                np.asarray(placement["face_to_crop_affine"], dtype=np.float32).copy()
            ),
            **mask_tensors,
            "domain": str(row["domain"]),
            "identity": str(row["identity"]),
            "clip_id": str(row["clip_id"]),
            "frame_index": int(frame),
        }


def _rgb_image(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)


class CachedTextureDatasetV4(CachedStructureDatasetV4):
    """V4B RGB samples anchored to the audited V4 transition cache."""

    def __init__(
        self, *args, reference_count: int = 5, seed: int = 413,
        context_mode: str = "same_frame_reconstruction", **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reference_count = int(reference_count)
        self.seed = int(seed)
        if context_mode != "same_frame_reconstruction":
            raise ValueError(
                "CachedTextureDatasetV4 only supports coordinate-safe same-frame reconstruction. "
                "Use a source-space cross-driving dataset after reprojection and mask regeneration."
            )
        self.context_mode = context_mode
        self.clip_to_cached_frames: dict[int, list[int]] = defaultdict(list)
        for clip_index, frame, _ in self.records:
            self.clip_to_cached_frames[clip_index].append(frame)

    def _image_path(self, row: dict, frame: int) -> Path:
        return Path(row["root"]) / "crops" / f"{row['clip_id']}_f{frame:06d}.jpg"

    def _source_frame(self, clip_index: int, frame: int) -> int:
        row = self.rows[clip_index]
        count = int(row["frame_count"])
        digest = hashlib.sha256(
            f"v4b-source:{self.seed}:{row['clip_id']}:{frame}".encode("utf-8")
        ).digest()
        distance = 12 + int.from_bytes(digest[:2], "little") % min(61, max(count - 1, 1))
        direction = -1 if digest[2] & 1 else 1
        source = int(np.clip(frame + direction * distance, 0, count - 1))
        if source == frame and count > 1:
            source = frame - 1 if frame else 1
        return source

    def _reference_frames(self, clip_index: int, frame: int) -> list[int]:
        row = self.rows[clip_index]
        points, _ = self._geometry_arrays(clip_index)
        count = int(row["frame_count"])
        candidates = np.unique(np.linspace(0, count - 1, min(48, count)).round().astype(np.int64))
        candidates = candidates[np.abs(candidates - frame) > 8]
        if not len(candidates):
            candidates = np.asarray([frame], dtype=np.int64)
        values = np.asarray(points[candidates], dtype=np.float32).reshape(len(candidates), -1)
        target = np.asarray(points[frame], dtype=np.float32).reshape(1, -1)
        scale = np.std(values, axis=0, keepdims=True).clip(1.0, None)
        values = (values - target) / scale
        selected = [int(np.linalg.norm(values, axis=1).argmin())]
        nearest = np.linalg.norm(values - values[selected[0]], axis=1)
        target_distance = np.linalg.norm(values, axis=1)
        while len(selected) < min(self.reference_count, len(candidates)):
            score = nearest + 0.1 * target_distance
            score[selected] = -1
            choice = int(score.argmax())
            selected.append(choice)
            nearest = np.minimum(nearest, np.linalg.norm(values - values[choice], axis=1))
        while len(selected) < self.reference_count:
            selected.append(selected[len(selected) % len(selected)])
        return [int(candidates[index]) for index in selected]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        structure = super().__getitem__(index)
        clip_index, frame, cache_path = self.records[index]
        row = self.rows[clip_index]
        points, geometry81 = self._geometry_arrays(clip_index)
        placement = self._placement_metadata(clip_index)
        source_frame = frame
        binding = FrameSpaceBinding(
            clip_id=str(row["clip_id"]), crop_contract_id=str(row["contract"]),
            context_frame_index=source_frame, mask_frame_index=frame,
            geometry_frame_index=frame, projection_frame_index=frame,
            mode=self.context_mode,
        )
        validate_frame_space_binding(binding)
        reference_frames = self._reference_frames(clip_index, frame)
        target_rgb = _rgb_image(self._image_path(row, frame))
        source_rgb = _rgb_image(self._image_path(row, source_frame))
        with np.load(cache_path) as masks:
            all_masks = {
                f"{name}_mask": torch.from_numpy(
                    np.asarray(masks[name], dtype=np.float32).copy() / 255.0
                )[None]
                for name in masks.files
            }
        edit = all_masks["edit_mask"]
        references = torch.stack(
            [_rgb_image(self._image_path(row, value)) for value in reference_frames]
        )
        reference_points = torch.stack(
            [
                torch.from_numpy(np.asarray(points[value], dtype=np.float32).copy())
                for value in reference_frames
            ]
        )
        reference_geometry81 = torch.stack(
            [
                torch.from_numpy(np.asarray(geometry81[value], dtype=np.float32).copy())
                for value in reference_frames
            ]
        )
        return {
            **structure,
            **all_masks,
            "target_rgb": target_rgb,
            "source_rgb": source_rgb,
            "source_masked": source_rgb * (1.0 - edit),
            "source_points_px": torch.from_numpy(
                np.asarray(points[source_frame], dtype=np.float32).copy()
            ),
            "source_geometry_xyz": torch.from_numpy(
                np.asarray(placement["geometry_xyz"][source_frame], dtype=np.float32).copy()
            ),
            "references_rgb": references,
            "reference_points_px": reference_points,
            "reference_geometry81": reference_geometry81,
            "source_frame_index": source_frame,
            "context_frame_index": source_frame,
            "mask_frame_index": frame,
            "geometry_frame_index": frame,
            "projection_frame_index": frame,
            "coordinate_contract_version": binding.version,
            "coordinate_mode": binding.mode,
            "crop_contract_id": binding.crop_contract_id,
            "reference_frame_indices": torch.tensor(reference_frames, dtype=torch.int64),
        }


class DomainIdentityBalancedStructureSamplerV4(Sampler[int]):
    """Alternates domains, then cycles identities within each domain."""

    def __init__(self, dataset: CachedStructureDatasetV4, samples: int, seed: int = 413) -> None:
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
        identity_orders = {
            domain: list(self.dataset.domain_to_identities[domain]) for domain in domains
        }
        counters = {domain: 0 for domain in domains}
        for identities in identity_orders.values():
            rng.shuffle(identities)
        for sample_index in range(self.samples):
            domain = domains[sample_index % len(domains)]
            identities = identity_orders[domain]
            position = counters[domain]
            if position and position % len(identities) == 0:
                rng.shuffle(identities)
            identity = identities[position % len(identities)]
            counters[domain] += 1
            yield int(rng.choice(self.dataset.identity_to_indices[identity]))


def deterministic_structure_validation_indices_v4(
    dataset: CachedStructureDatasetV4,
    samples_per_identity: int,
) -> list[int]:
    result: list[int] = []
    for identity in dataset.identities:
        values = dataset.identity_to_indices[identity]
        positions = np.linspace(0, len(values) - 1, min(samples_per_identity, len(values)))
        result.extend(values[int(round(position))] for position in positions)
    return result


__all__ = [
    "CachedStructureDatasetV4",
    "CachedTextureDatasetV4",
    "DomainIdentityBalancedStructureSamplerV4",
    "deterministic_structure_validation_indices_v4",
]
