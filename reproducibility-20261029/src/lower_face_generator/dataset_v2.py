from __future__ import annotations

import bisect
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import StableLowerFaceCrop, build_crop_masks


@dataclass(frozen=True)
class LowerFaceClipV2:
    clip_id: str
    identity: str
    domain: str
    split: str
    root: Path
    frame_count: int


def read_manifest(path: Path) -> list[LowerFaceClipV2]:
    clips: list[LowerFaceClipV2] = []
    base = path.resolve().parent
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        root = Path(row["root"])
        if not root.is_absolute():
            root = base / root
        clips.append(
            LowerFaceClipV2(
                clip_id=str(row["clip_id"]),
                identity=str(row["identity"]),
                domain=str(row["domain"]),
                split=str(row["split"]),
                root=root,
                frame_count=int(row["frame_count"]),
            )
        )
    return clips


def _crop_from_contract(path: Path) -> StableLowerFaceCrop:
    row = json.loads(path.read_text(encoding="utf-8"))["crop"]
    return StableLowerFaceCrop(
        x=float(row["x"]),
        y=float(row["y"]),
        width=float(row["width"]),
        height=float(row["height"]),
        output_width=int(row["output_width"]),
        output_height=int(row["output_height"]),
    )


def _image_tensor(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255.0


class LowerFaceCompletionDatasetV2(Dataset):
    """Lower-face completion samples with broad nose-excluded edit support."""

    def __init__(
        self,
        manifest: str | Path,
        split: str = "train",
        reference_count: int = 5,
        fill_value: float = 0.0,
        seed: int = 1337,
    ) -> None:
        self.reference_count = int(reference_count)
        self.fill_value = float(fill_value)
        self.seed = int(seed)
        self.clips = [clip for clip in read_manifest(Path(manifest)) if clip.split == split]
        if not self.clips:
            raise ValueError(f"No {split!r} clips in {manifest}")
        self.ends = np.cumsum([clip.frame_count for clip in self.clips], dtype=np.int64)
        self._arrays: dict[Path, dict[str, np.ndarray | StableLowerFaceCrop]] = {}

    def __len__(self) -> int:
        return int(self.ends[-1])

    def _metadata(self, clip: LowerFaceClipV2) -> dict[str, np.ndarray | StableLowerFaceCrop]:
        cached = self._arrays.get(clip.root)
        if cached is None:
            cached = {
                "landmarks": np.load(clip.root / "face_landmarks_xy.npy", mmap_mode="r"),
                "points": np.load(clip.root / "lower_face_landmarks_crop.npy", mmap_mode="r"),
                "geometry": np.load(clip.root / "lower_face_geometry_xyz.npy", mmap_mode="r"),
                "projection": np.load(clip.root / "lower_face_projection19.npy", mmap_mode="r"),
                "pose": np.load(clip.root / "pose6.npy", mmap_mode="r"),
                "crop": _crop_from_contract(clip.root / "contract.json"),
            }
            self._arrays[clip.root] = cached
        return cached

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        clip_index = bisect.bisect_right(self.ends, index)
        previous = 0 if clip_index == 0 else int(self.ends[clip_index - 1])
        frame = int(index - previous)
        clip = self.clips[clip_index]
        metadata = self._metadata(clip)
        crop = metadata["crop"]
        assert isinstance(crop, StableLowerFaceCrop)
        landmarks = np.asarray(metadata["landmarks"][frame], dtype=np.float32)
        masks = build_crop_masks(landmarks, crop)
        edit = torch.from_numpy(masks["edit"].astype(np.float32) / 255.0)[None]
        target_path = clip.root / "crops" / f"{clip.clip_id}_f{frame:06d}.jpg"
        target = _image_tensor(target_path)
        source = target * (1.0 - edit) + self.fill_value * edit

        rng = random.Random(self.seed + int(index))
        candidates = [value for value in range(clip.frame_count) if value != frame]
        if not candidates:
            candidates = [frame]
        references = [rng.choice(candidates) for _ in range(self.reference_count)]
        reference_rgb = torch.stack(
            [
                _image_tensor(clip.root / "crops" / f"{clip.clip_id}_f{value:06d}.jpg")
                for value in references
            ]
        )
        mask_tensors = {
            name: torch.from_numpy(value.astype(np.float32) / 255.0)[None]
            for name, value in masks.items()
        }
        return {
            "source_masked": source,
            "target": target,
            "references": reference_rgb,
            "target_points_px": torch.from_numpy(
                np.asarray(metadata["points"][frame], dtype=np.float32).copy()
            ),
            "target_geometry_xyz": torch.from_numpy(
                np.asarray(metadata["geometry"][frame], dtype=np.float32).copy()
            ),
            "projection19": torch.from_numpy(
                np.asarray(metadata["projection"][frame], dtype=np.float32).copy()
            ),
            "pose6": torch.from_numpy(
                np.asarray(metadata["pose"][frame], dtype=np.float32).copy()
            ),
            **mask_tensors,
            "clip_id": clip.clip_id,
            "identity": clip.identity,
            "domain": clip.domain,
            "frame": frame,
        }


__all__ = ["LowerFaceClipV2", "LowerFaceCompletionDatasetV2", "read_manifest"]
