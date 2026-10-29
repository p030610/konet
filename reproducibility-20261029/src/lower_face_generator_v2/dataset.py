from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lower_face_generator.preprocess import (
    StableLowerFaceCrop,
    build_crop_masks,
    track_upper_transition_sequence,
)
from lower_face_generator.geometry import (
    TEMPORAL_STABILITY_INDICES,
    causal_smooth_upper_boundary_parameters,
    track_upper_boundary_curve_sequence,
    upper_boundary_anchor_parameters_sequence,
)

from .indexing import LowerFaceTrainingIndexV2, articulation_descriptor


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _crop(path: Path) -> StableLowerFaceCrop:
    row = json.loads(path.read_text(encoding="utf-8"))["crop"]
    return StableLowerFaceCrop(
        x=float(row["x"]),
        y=float(row["y"]),
        width=float(row["width"]),
        height=float(row["height"]),
        output_width=int(row["output_width"]),
        output_height=int(row["output_height"]),
    )


def _image(path: Path) -> torch.Tensor:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(value.transpose(2, 0, 1).copy()).float().div_(255.0)


@dataclass(frozen=True)
class _ClipRuntime:
    row: dict
    global_index: int
    identity_index: int


class LowerFaceCompletionDatasetV2(Dataset):
    """Indexed lower-face frame dataset with deterministic five-reference banks."""

    def __init__(
        self,
        index_path: str | Path,
        split: str = "train",
        reference_count: int = 5,
        audio_context: int = 4,
        fill_value: float = 0.0,
        metadata_cache_size: int = 16,
        upper_boundary_mode: str = "legacy_horizontal",
        upper_overlap_px: int = 0,
        boundary_smoothing_alpha: float = 0.18,
        boundary_tracking_correction: float = 0.15,
        transition_safety_dilate_px: int = 2,
        transition_cache_dir: str | Path | None = None,
    ) -> None:
        self.index = LowerFaceTrainingIndexV2.load(index_path)
        manifest = Path(str(self.index.arrays["manifest"].item()))
        self.rows = _read_rows(manifest)
        if len(self.rows) != self.index.clip_count:
            raise ValueError("Manifest and training index clip counts differ")
        self.reference_count = int(reference_count)
        self.audio_context = int(audio_context)
        self.fill_value = float(fill_value)
        self.upper_boundary_mode = str(upper_boundary_mode)
        self.upper_overlap_px = max(0, int(upper_overlap_px))
        self.boundary_smoothing_alpha = float(boundary_smoothing_alpha)
        self.boundary_tracking_correction = float(boundary_tracking_correction)
        self.transition_safety_dilate_px = max(0, int(transition_safety_dilate_px))
        self.transition_cache_dir = (
            Path(transition_cache_dir) if transition_cache_dir is not None else None
        )
        self.metadata_cache_size = max(1, int(metadata_cache_size))
        identity_indices = self.index.arrays["clip_identity_indices"]
        self.clips = [
            _ClipRuntime(row, global_index, int(identity_indices[global_index]))
            for global_index, row in enumerate(self.rows)
            if row["split"] == split
        ]
        if not self.clips:
            raise ValueError(f"No clips for split {split!r}")
        self.ends = np.cumsum([int(clip.row["frame_count"]) for clip in self.clips], dtype=np.int64)
        self._metadata: OrderedDict[int, dict[str, np.ndarray | StableLowerFaceCrop]] = OrderedDict()

    def transition_frame_cache_path(self, row: dict, frame: int) -> Path:
        if self.transition_cache_dir is None:
            raise ValueError("A transition cache directory was not configured")
        stem = (
            f"{row['clip_id']}_o{self.upper_overlap_px}"
            f"_c{self.boundary_tracking_correction:.4f}"
            f"_d{self.transition_safety_dilate_px}"
        )
        return (
            self.transition_cache_dir
            / str(row["domain"])
            / stem
            / f"f{int(frame):06d}.npz"
        )

    def __len__(self) -> int:
        return int(self.ends[-1])

    def _clip_frame(self, index: int) -> tuple[_ClipRuntime, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        clip_index = bisect.bisect_right(self.ends, index)
        previous = 0 if clip_index == 0 else int(self.ends[clip_index - 1])
        return self.clips[clip_index], int(index - previous)

    def _load_metadata(self, clip_index: int, row: dict) -> dict[str, np.ndarray | StableLowerFaceCrop]:
        cached = self._metadata.get(clip_index)
        if cached is None:
            root = Path(row["root"])
            cached = {
                "landmarks": np.load(row["face_landmarks"], mmap_mode="r"),
                "points": np.load(row["lower_face_points"], mmap_mode="r"),
                "geometry": np.load(row["lower_face_geometry"], mmap_mode="r"),
                "geometry81": np.load(row["geometry81"], mmap_mode="r"),
                "projection": np.load(row["projection19"], mmap_mode="r"),
                "pose": np.load(row["pose6"], mmap_mode="r"),
                "audio": np.load(row["audio_features"], mmap_mode="r"),
                "crop": _crop(Path(row["contract"])),
                "root": root,
            }
            if (
                self.upper_boundary_mode == "tracked_transition_v4"
                and self.transition_cache_dir is not None
            ):
                cached["transition_frame_cache"] = np.asarray(True)
            elif self.upper_boundary_mode in (
                "anchor_curved_v2",
                "tracked_anchor_curved_v3",
                "tracked_transition_v4",
            ):
                raw_parameters = upper_boundary_anchor_parameters_sequence(
                    np.asarray(cached["landmarks"])
                )
                cached["upper_boundary_parameters"] = (
                    causal_smooth_upper_boundary_parameters(
                        raw_parameters, alpha=self.boundary_smoothing_alpha
                    )
                )
                if self.upper_boundary_mode in (
                    "tracked_anchor_curved_v3",
                    "tracked_transition_v4",
                ):
                    cached["upper_boundary_curve_xy"] = (
                        track_upper_boundary_curve_sequence(
                            np.asarray(cached["landmarks"]),
                            cached["upper_boundary_parameters"],
                            correction=self.boundary_tracking_correction,
                        )
                    )
                if self.upper_boundary_mode == "tracked_transition_v4":
                    crop = cached["crop"]
                    assert isinstance(crop, StableLowerFaceCrop)
                    frame_masks = [
                            build_crop_masks(
                                np.asarray(cached["landmarks"][frame], dtype=np.float32),
                                crop,
                                upper_boundary_mode=self.upper_boundary_mode,
                                upper_overlap_px=self.upper_overlap_px,
                                upper_boundary_parameters=np.asarray(
                                    cached["upper_boundary_parameters"][frame], dtype=np.float32
                                ),
                                upper_boundary_curve_xy=np.asarray(
                                    cached["upper_boundary_curve_xy"][frame], dtype=np.float32
                                ),
                            )
                            for frame in range(len(cached["landmarks"]))
                    ]
                    stable_face = np.asarray(cached["landmarks"])[
                            :, np.asarray(TEMPORAL_STABILITY_INDICES), :2
                        ]
                    stable_crop = (
                            stable_face @ crop.face_to_crop[:, :2].T
                            + crop.face_to_crop[:, 2]
                    ).astype(np.float32)
                    tracked_overlap, tracked_alpha = track_upper_transition_sequence(
                            np.stack([value["upper_overlap"] for value in frame_masks]),
                            np.stack([value["upper_alpha_target"] for value in frame_masks]).astype(np.float32) / 255.0,
                            np.stack([value["edit"] for value in frame_masks]),
                            np.stack([value["nose_exclusion"] for value in frame_masks]),
                            stable_crop,
                            correction=self.boundary_tracking_correction,
                            safety_dilate_px=self.transition_safety_dilate_px,
                    )
                    expected = (
                        len(cached["landmarks"]), crop.output_height, crop.output_width
                    )
                    if tracked_overlap.shape != expected or tracked_alpha.shape != expected:
                        raise ValueError(
                            f"Invalid transition fields for {row['clip_id']}: "
                            f"{tracked_overlap.shape}, {tracked_alpha.shape}; expected {expected}"
                        )
                    cached["tracked_upper_overlap"] = tracked_overlap
                    cached["tracked_upper_alpha_target"] = tracked_alpha
            self._metadata[clip_index] = cached
            while len(self._metadata) > self.metadata_cache_size:
                self._metadata.popitem(last=False)
        else:
            self._metadata.move_to_end(clip_index)
        return cached

    def _descriptor(self, clip: _ClipRuntime, frame: int) -> np.ndarray:
        metadata = self._load_metadata(clip.global_index, clip.row)
        raw = articulation_descriptor(
            np.asarray(metadata["geometry81"][frame]),
            np.asarray(metadata["geometry"][frame]),
            self.index.arrays["clip_geometry_centers"][clip.global_index],
            self.index.arrays["pca_basis"],
        )[0]
        return (raw - self.index.arrays["descriptor_mean"]) / self.index.arrays["descriptor_std"]

    def reference_frames(self, clip: _ClipRuntime, frame: int) -> list[tuple[int, int]]:
        # Milestone 3 is an articulation renderer, not cross-recording appearance
        # transfer. Build the bank from the target clip so crop calibration,
        # lighting, background, and facial hair remain coherent.
        metadata = self._load_metadata(clip.global_index, clip.row)
        frame_count = int(clip.row["frame_count"])
        candidate_count = min(64, frame_count)
        frames = np.unique(np.linspace(0, frame_count - 1, candidate_count).round().astype(np.int32))
        exclusion = int(self.index.arrays["temporal_exclusion"].item())
        valid = np.abs(frames - frame) > exclusion
        if not valid.any():
            valid = frames != frame
        frames = frames[valid]
        if not len(frames):
            return [(clip.global_index, frame)] * self.reference_count

        descriptors = articulation_descriptor(
            np.asarray(metadata["geometry81"])[frames],
            np.asarray(metadata["geometry"])[frames],
            self.index.arrays["clip_geometry_centers"][clip.global_index],
            self.index.arrays["pca_basis"],
        )
        descriptors = (descriptors - self.index.arrays["descriptor_mean"]) / self.index.arrays["descriptor_std"]

        target = self._descriptor(clip, frame)
        target_distance = np.sum((descriptors - target) ** 2, axis=1)
        selected = [int(np.argmin(target_distance))]
        nearest = np.sum((descriptors - descriptors[selected[0]]) ** 2, axis=1)
        while len(selected) < min(self.reference_count, len(frames)):
            score = nearest + 0.15 * target_distance
            score[np.asarray(selected)] = -np.inf
            choice = int(np.argmax(score))
            selected.append(choice)
            nearest = np.minimum(nearest, np.sum((descriptors - descriptors[choice]) ** 2, axis=1))
        while len(selected) < self.reference_count:
            selected.append(selected[len(selected) % len(selected)])
        return [(clip.global_index, int(frames[value])) for value in selected]

    def _audio_window(self, audio: np.ndarray, frame: int, frame_count: int) -> torch.Tensor:
        center = int(round((frame + 0.5) * len(audio) / max(frame_count, 1) - 0.5))
        offsets = np.arange(-self.audio_context, self.audio_context + 1, dtype=np.int64)
        indices = np.clip(center + offsets, 0, len(audio) - 1)
        return torch.from_numpy(np.asarray(audio[indices], dtype=np.float32).copy())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        clip, frame = self._clip_frame(index)
        row = clip.row
        metadata = self._load_metadata(clip.global_index, row)
        crop = metadata["crop"]
        assert isinstance(crop, StableLowerFaceCrop)
        landmarks = np.asarray(metadata["landmarks"][frame], dtype=np.float32)
        if self.upper_boundary_mode == "tracked_transition_v4" and "transition_frame_cache" in metadata:
            cache_path = self.transition_frame_cache_path(row, frame)
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"Missing M7D frame cache: {cache_path}. "
                    "Run cache_lower_face_v2_transitions.py for this training schedule."
                )
            with np.load(cache_path) as transition:
                if "edit" not in transition.files:
                    raise ValueError(
                        f"Incomplete M7D mask cache: {cache_path}. "
                        "Run cache_lower_face_v2_transitions.py --upgrade-existing."
                    )
                masks = {
                    name: np.asarray(transition[name], dtype=np.uint8)
                    for name in transition.files
                }
        else:
            masks = build_crop_masks(
                landmarks,
                crop,
                upper_boundary_mode=self.upper_boundary_mode,
                upper_overlap_px=self.upper_overlap_px,
                upper_boundary_parameters=(
                    np.asarray(metadata["upper_boundary_parameters"][frame], dtype=np.float32)
                    if "upper_boundary_parameters" in metadata
                    else None
                ),
                upper_boundary_curve_xy=(
                    np.asarray(metadata["upper_boundary_curve_xy"][frame], dtype=np.float32)
                    if "upper_boundary_curve_xy" in metadata
                    else None
                ),
            )
            if self.upper_boundary_mode == "tracked_transition_v4":
                overlap = np.asarray(
                    metadata["tracked_upper_overlap"][frame], dtype=np.uint8
                )
                alpha_target = np.asarray(
                    metadata["tracked_upper_alpha_target"][frame], dtype=np.uint8
                )
                active = overlap > 0
                masks["upper_overlap"] = overlap.copy()
                masks["upper_alpha_target"] = alpha_target.copy()
                masks["edit"] = np.maximum(masks["edit"], overlap)
                masks["ring"] = np.maximum(masks["ring"], overlap)
                masks["alpha_core"][active] = 0
        mask_tensors = {
            f"{name}_mask": torch.from_numpy(value.astype(np.float32) / 255.0)[None]
            for name, value in masks.items()
        }
        root = Path(row["root"])
        target_path = root / "crops" / f"{row['clip_id']}_f{frame:06d}.jpg"
        target = _image(target_path)
        edit = mask_tensors["edit_mask"]
        source_masked = target * (1.0 - edit) + self.fill_value * edit

        reference_pairs = self.reference_frames(clip, frame)
        reference_rgb = []
        reference_geometry = []
        reference_geometry81 = []
        reference_points = []
        for reference_clip_index, reference_frame in reference_pairs:
            reference_row = self.rows[reference_clip_index]
            reference_metadata = self._load_metadata(reference_clip_index, reference_row)
            reference_root = Path(reference_row["root"])
            reference_rgb.append(
                _image(reference_root / "crops" / f"{reference_row['clip_id']}_f{reference_frame:06d}.jpg")
            )
            reference_geometry.append(
                torch.from_numpy(np.asarray(reference_metadata["geometry"][reference_frame], dtype=np.float32).copy())
            )
            reference_geometry81.append(
                torch.from_numpy(np.asarray(reference_metadata["geometry81"][reference_frame], dtype=np.float32).copy())
            )
            reference_points.append(
                torch.from_numpy(np.asarray(reference_metadata["points"][reference_frame], dtype=np.float32).copy())
            )
        geometry81 = torch.from_numpy(np.asarray(metadata["geometry81"][frame], dtype=np.float32).copy())
        stable_face_points = landmarks[np.asarray(TEMPORAL_STABILITY_INDICES), :2]
        stable_crop_points = (
            stable_face_points @ crop.face_to_crop[:, :2].T
            + crop.face_to_crop[:, 2]
        ).astype(np.float32)
        return {
            "source_rgb": target,
            "source_masked": source_masked,
            "target_rgb": target,
            "references_rgb": torch.stack(reference_rgb),
            "source_points_px": torch.from_numpy(np.asarray(metadata["points"][frame], dtype=np.float32).copy()),
            "target_points_px": torch.from_numpy(np.asarray(metadata["points"][frame], dtype=np.float32).copy()),
            "source_geometry_xyz": torch.from_numpy(
                np.asarray(metadata["geometry"][frame], dtype=np.float32).copy()
            ),
            "target_geometry_xyz": torch.from_numpy(
                np.asarray(metadata["geometry"][frame], dtype=np.float32).copy()
            ),
            "reference_geometry_xyz": torch.stack(reference_geometry),
            "reference_geometry81": torch.stack(reference_geometry81),
            "reference_points_px": torch.stack(reference_points),
            "geometry81": geometry81,
            "teeth_score": geometry81[80:81],
            "projection19": torch.from_numpy(
                np.asarray(metadata["projection"][frame], dtype=np.float32).copy()
            ),
            "pose6": torch.from_numpy(np.asarray(metadata["pose"][frame], dtype=np.float32).copy()),
            "wav2vec_window": self._audio_window(
                metadata["audio"], frame, int(row["frame_count"])
            ),
            "face_to_crop_affine": torch.from_numpy(crop.face_to_crop.copy()),
            "stability_points_px": torch.from_numpy(stable_crop_points.copy()),
            **mask_tensors,
            "domain": row["domain"],
            "identity": row["identity"],
            "clip_id": row["clip_id"],
            "frame_index": frame,
            "reference_clip_indices": torch.tensor([value[0] for value in reference_pairs]),
            "reference_frame_indices": torch.tensor([value[1] for value in reference_pairs]),
        }


class LowerFaceSequenceIndexV2(Dataset):
    """Compact contiguous-window index; frame decoding remains the caller's choice."""

    def __init__(self, index_path: str | Path, split: str = "train") -> None:
        self.index = LowerFaceTrainingIndexV2.load(index_path)
        self.sequence_length = int(self.index.arrays["sequence_length"].item())
        splits = self.index.arrays["clip_splits"]
        counts = self.index.arrays["sequence_counts"]
        self.clip_indices = np.flatnonzero(splits == split).astype(np.int32)
        self.counts = counts[self.clip_indices]
        self.ends = np.cumsum(self.counts, dtype=np.int64)
        if not len(self.ends) or self.ends[-1] == 0:
            raise ValueError(f"No sequences for split {split!r}")

    def __len__(self) -> int:
        return int(self.ends[-1])

    def __getitem__(self, index: int) -> dict[str, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        local_clip = bisect.bisect_right(self.ends, index)
        previous = 0 if local_clip == 0 else int(self.ends[local_clip - 1])
        return {
            "clip_index": int(self.clip_indices[local_clip]),
            "start_frame": int(index - previous),
            "length": self.sequence_length,
        }


__all__ = ["LowerFaceCompletionDatasetV2", "LowerFaceSequenceIndexV2"]
