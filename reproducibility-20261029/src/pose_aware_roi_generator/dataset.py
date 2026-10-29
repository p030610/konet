from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import librosa
import numpy as np
import torch
from PIL import Image
from scipy import signal
from torch.utils.data import Dataset

try:
    from torchvision.io import ImageReadMode, read_image
except (ImportError, OSError):
    ImageReadMode = None
    read_image = None

from .geometry import (
    NEUTRAL_POSE6,
    adapt_canonical_points_to_view,
    load_geometry81,
    load_pose6,
    points_to_pixels,
)
from .rasterizer import OUTER_LIP_ORDER


ROI_RE = re.compile(r"(?P<video>.+)_f(?P<frame>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
FRAME_RE = re.compile(r"(?:^|_)f(?P<frame>\d+)(?:_|\.|$)")


@dataclass(frozen=True)
class RoiItem:
    video_id: str
    frame_idx: int
    roi_path: Path
    geometry_path: Path
    pose_path: Path | None


@dataclass(frozen=True)
class RoiClip:
    video_id: str
    start_frame: int
    item_indices: tuple[int, ...]


def parse_roi_filename(path: Path) -> tuple[str, int] | None:
    match = ROI_RE.match(path.name)
    if not match:
        return None
    return match.group("video"), int(match.group("frame"))


def parse_frame_idx(path: Path) -> int | None:
    match = FRAME_RE.search(path.name)
    return int(match.group("frame")) if match else None


def load_split(path: Path | None, split_name: str) -> set[str] | None:
    if path is None:
        return None
    data = json.loads(path.read_text())
    candidates = [split_name, f"{split_name}_clips"]
    for key in candidates:
        values = data.get(key)
        if isinstance(values, list):
            return set(str(v) for v in values)
    raise KeyError(f"Split file {path} does not contain {split_name!r} or {split_name + '_clips'!r}")


def build_roi_map(roi_root: Path) -> dict[tuple[str, int], Path]:
    out: dict[tuple[str, int], Path] = {}
    for path in roi_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parsed = parse_roi_filename(path)
        if parsed is not None:
            out[parsed] = path
    return out


def resolve_video_roi_root(roi_root: Path, video_id: str) -> Path:
    """Narrow HDTF ROI scans to one identity directory when available."""
    identity = video_id.rsplit("_", 1)[0]
    candidate = roi_root / f"{identity}_roi"
    return candidate if candidate.is_dir() else roi_root


def build_frame_file_map(root: Path) -> dict[tuple[str, int], Path]:
    out: dict[tuple[str, int], Path] = {}
    if not root.exists():
        return out
    for path in root.rglob("*.npy"):
        video_id = path.parent.name
        frame_idx = parse_frame_idx(path)
        if frame_idx is None:
            continue
        out[(video_id, frame_idx)] = path
    return out


def load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    if read_image is not None and ImageReadMode is not None:
        tensor = read_image(str(path), mode=ImageReadMode.RGB)
        if tensor.shape[-2:] == (image_size, image_size):
            return tensor.to(dtype=torch.float32).div_(255.0)
    image = Image.open(path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def estimate_roi_lip_mask(path: Path, image_size: int) -> torch.Tensor:
    """Estimate a conservative lip/oral hull directly in an ROI training crop."""
    image = np.asarray(
        Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]
    gate = np.zeros((height, width), dtype=np.uint8)
    gate[int(height * 0.04) : int(height * 0.98), int(width * 0.04) : int(width * 0.96)] = 1
    a_channel = lab[..., 1]
    saturation = hsv[..., 1]
    a_threshold = max(145.0, float(np.percentile(a_channel[gate > 0], 74.0)))
    s_threshold = max(35.0, float(np.percentile(saturation[gate > 0], 48.0)))
    candidate = np.uint8((a_channel >= a_threshold) & (saturation >= s_threshold) & (gate > 0))
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13)),
    )
    count, labels, stats, centers = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    scored: list[tuple[float, int]] = []
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centers[label]
        center_penalty = abs(cx / width - 0.5) + 0.5 * abs(cy / height - 0.58)
        scored.append((area / (1.0 + 2.0 * center_penalty), label))
    selected_labels = [label for _, label in sorted(scored, reverse=True)[:2]]
    selected = np.uint8(np.isin(labels, selected_labels)) if selected_labels else candidate
    points = cv2.findNonZero(selected)
    mask = np.zeros((height, width), dtype=np.uint8)
    if points is not None:
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 1)
    if int(mask.sum()) < max(32, image_size * image_size // 1024):
        # A small center prior is safer than silently disabling skin supervision.
        cv2.ellipse(
            mask,
            (width // 2, int(height * 0.58)),
            (max(2, int(width * 0.28)), max(2, int(height * 0.22))),
            0,
            0,
            360,
            1,
            -1,
        )
    return torch.from_numpy(mask.astype(np.float32))[None].contiguous()


def geometry_lip_mask(geometry81: np.ndarray, pose6: np.ndarray, image_size: int) -> torch.Tensor:
    """Rasterize the topology-defined outer lip contour into an ROI-space mask."""
    points = torch.from_numpy(geometry81[:80].reshape(1, 40, 2).astype(np.float32))
    pose = torch.from_numpy(pose6[:6].reshape(1, 6).astype(np.float32))
    with torch.no_grad():
        view_points = adapt_canonical_points_to_view(points, pose)
        pixels = points_to_pixels(view_points, image_size, geometry_scale=0.92)[0]
        pixels = pixels[list(OUTER_LIP_ORDER)]
    contour = np.rint(pixels.numpy()).astype(np.int32)
    contour[:, 0] = np.clip(contour[:, 0], 0, image_size - 1)
    contour[:, 1] = np.clip(contour[:, 1], 0, image_size - 1)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 1)
    return torch.from_numpy(mask.astype(np.float32))[None].contiguous()


class PoseAwareRoiDataset(Dataset):
    def __init__(
        self,
        roi_root: str | Path,
        geometry_root: str | Path,
        pose_root: str | Path | None = None,
        split_path: str | Path | None = None,
        split_name: str = "train",
        image_size: int = 192,
        allow_missing_pose: bool = True,
        max_samples: int = 0,
        include_color_lip_masks: bool = True,
        include_geometry_lip_masks: bool = False,
        include_reference_geometry: bool = False,
    ) -> None:
        self.roi_root = Path(roi_root)
        self.geometry_root = Path(geometry_root)
        self.pose_root = Path(pose_root) if pose_root else None
        self.split_name = split_name
        self.image_size = int(image_size)
        self.allow_missing_pose = bool(allow_missing_pose)
        self.include_color_lip_masks = bool(include_color_lip_masks)
        self.include_geometry_lip_masks = bool(include_geometry_lip_masks)
        self.include_reference_geometry = bool(include_reference_geometry)

        clip_set = load_split(Path(split_path) if split_path else None, split_name)
        roi_map = build_roi_map(self.roi_root)
        geometry_map = build_frame_file_map(self.geometry_root)
        pose_map = build_frame_file_map(self.pose_root) if self.pose_root else {}

        items: list[RoiItem] = []
        for key, roi_path in sorted(roi_map.items()):
            video_id, frame_idx = key
            if clip_set is not None and video_id not in clip_set:
                continue
            geometry_path = geometry_map.get(key)
            if geometry_path is None:
                continue
            pose_path = pose_map.get(key)
            if pose_path is None and not self.allow_missing_pose:
                continue
            items.append(RoiItem(video_id, frame_idx, roi_path, geometry_path, pose_path))

        if max_samples and max_samples > 0:
            items = items[: int(max_samples)]
        if not items:
            raise RuntimeError(
                f"No ROI/geometry pairs found. roi_root={self.roi_root}, geometry_root={self.geometry_root}, split={split_name}"
            )

        self.items = items
        self.by_video: dict[str, list[int]] = {}
        for idx, item in enumerate(self.items):
            self.by_video.setdefault(item.video_id, []).append(idx)

    def __len__(self) -> int:
        return len(self.items)

    def _choose_reference_item(self, item_idx: int) -> RoiItem:
        item = self.items[item_idx]
        candidates = self.by_video.get(item.video_id, [item_idx])
        if self.split_name == "train" and len(candidates) > 1:
            ref_idx = item_idx
            for _ in range(8):
                ref_idx = random.choice(candidates)
                if ref_idx != item_idx:
                    break
            return self.items[ref_idx]
        if len(candidates) > 1:
            return self.items[candidates[len(candidates) // 2]]
        return item

    def _choose_reference(self, item_idx: int) -> Path:
        return self._choose_reference_item(item_idx).roi_path

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        ref_item = self._choose_reference_item(idx)
        ref_path = ref_item.roi_path
        geometry = load_geometry81(item.geometry_path)
        pose = load_pose6(item.pose_path) if item.pose_path is not None else NEUTRAL_POSE6.copy()
        reference_geometry = load_geometry81(ref_item.geometry_path)
        reference_pose = load_pose6(ref_item.pose_path) if ref_item.pose_path is not None else NEUTRAL_POSE6.copy()
        sample = {
            "reference": load_image_tensor(ref_path, self.image_size),
            "target": load_image_tensor(item.roi_path, self.image_size),
            "geometry": torch.from_numpy(geometry),
            "pose": torch.from_numpy(pose.astype(np.float32)),
            "video_id": item.video_id,
            "frame_idx": item.frame_idx,
            "target_path": str(item.roi_path),
            "reference_path": str(ref_path),
        }
        if self.include_color_lip_masks:
            sample["reference_lip_mask"] = estimate_roi_lip_mask(ref_path, self.image_size)
            sample["target_lip_mask"] = estimate_roi_lip_mask(item.roi_path, self.image_size)
        if self.include_geometry_lip_masks:
            sample["reference_geometry_lip_mask"] = geometry_lip_mask(reference_geometry, reference_pose, self.image_size)
            sample["target_geometry_lip_mask"] = geometry_lip_mask(geometry, pose, self.image_size)
        if self.include_reference_geometry:
            sample["reference_geometry"] = torch.from_numpy(reference_geometry)
            sample["reference_pose"] = torch.from_numpy(reference_pose.astype(np.float32))
        return sample


def _latentsync_melspectrogram(audio_path: Path) -> torch.Tensor:
    """Reproduce LatentSync's 16 kHz, 80-bin mel preprocessing."""
    wav, _ = librosa.load(audio_path, sr=16000, mono=True)
    emphasized = signal.lfilter([1.0, -0.97], [1.0], wav)
    spectrum = librosa.stft(
        y=emphasized,
        n_fft=800,
        hop_length=200,
        win_length=800,
    )
    mel_basis = librosa.filters.mel(
        sr=16000,
        n_fft=800,
        n_mels=80,
        fmin=55,
        fmax=7600,
    )
    amplitude = np.matmul(mel_basis, np.abs(spectrum))
    min_level = np.exp(-100.0 / 20.0 * np.log(10.0))
    decibels = 20.0 * np.log10(np.maximum(min_level, amplitude)) - 20.0
    normalized = np.clip(8.0 * ((decibels + 100.0) / 100.0) - 4.0, -4.0, 4.0)
    return torch.from_numpy(normalized.astype(np.float32))


def _audio_path_for_video(audio_root: Path, video_id: str) -> Path | None:
    if "_" not in video_id:
        return None
    speaker, clip_token = video_id.rsplit("_", 1)
    speaker_dir = audio_root / f"{speaker}_16khz"
    candidates = [speaker_dir / f"{video_id}.wav"]
    if clip_token.isdigit():
        clip_number = int(clip_token)
        candidates.extend(
            [
                speaker_dir / f"{speaker}_{clip_number:03d}.wav",
                speaker_dir / f"{speaker}_{clip_number}.wav",
            ]
        )
    return next((path for path in candidates if path.exists()), None)


class PoseAwareRoiClipDataset(Dataset):
    """Consecutive ROI clips plus full-face context for Stable SyncNet loss."""

    def __init__(
        self,
        roi_root: str | Path,
        geometry_root: str | Path,
        pose_root: str | Path,
        split_path: str | Path,
        split_name: str,
        face_root: str | Path,
        roi_metadata_root: str | Path,
        audio_root: str | Path,
        audio_mel_cache_dir: str | Path,
        image_size: int = 192,
        sync_face_size: int = 256,
        num_frames: int = 16,
        source_fps: float = 29.97,
        sync_fps: float = 25.0,
        clip_stride: int = 8,
        allow_missing_pose: bool = False,
        max_samples: int = 0,
    ) -> None:
        self.base = PoseAwareRoiDataset(
            roi_root=roi_root,
            geometry_root=geometry_root,
            pose_root=pose_root,
            split_path=split_path,
            split_name=split_name,
            image_size=image_size,
            allow_missing_pose=allow_missing_pose,
            max_samples=0,
            include_color_lip_masks=False,
            include_geometry_lip_masks=False,
            include_reference_geometry=False,
        )
        self.split_name = split_name
        self.face_root = Path(face_root)
        self.roi_metadata_root = Path(roi_metadata_root)
        self.audio_root = Path(audio_root)
        self.audio_mel_cache_dir = Path(audio_mel_cache_dir)
        self.audio_mel_cache_dir.mkdir(parents=True, exist_ok=True)
        self.sync_face_size = int(sync_face_size)
        self.num_frames = int(num_frames)
        self.source_fps = float(source_fps)
        self.sync_fps = float(sync_fps)
        self.clip_stride = max(1, int(clip_stride))
        self.frame_offsets = tuple(
            int(round(frame_index * self.source_fps / self.sync_fps))
            for frame_index in range(self.num_frames)
        )
        if len(set(self.frame_offsets)) != self.num_frames:
            raise ValueError(
                f"source_fps={self.source_fps} cannot provide {self.num_frames} unique samples at sync_fps={self.sync_fps}"
            )

        self._metadata_by_video: dict[str, dict[int, tuple[Path, np.ndarray]]] = {}
        self._audio_by_video: dict[str, Path] = {}
        self._mel_memory_cache: dict[str, torch.Tensor] = {}
        item_by_video_frame = {
            (item.video_id, item.frame_idx): item_index
            for item_index, item in enumerate(self.base.items)
        }

        clips: list[RoiClip] = []
        for video_id, item_indices in sorted(self.base.by_video.items()):
            metadata = self._load_video_metadata(video_id)
            audio_path = _audio_path_for_video(self.audio_root, video_id)
            if not metadata or audio_path is None:
                continue
            self._audio_by_video[video_id] = audio_path
            sorted_frames = sorted(self.base.items[item_index].frame_idx for item_index in item_indices)
            for start_frame in sorted_frames[:: self.clip_stride]:
                frame_indices = tuple(start_frame + offset for offset in self.frame_offsets)
                if not all((video_id, frame_idx) in item_by_video_frame for frame_idx in frame_indices):
                    continue
                if not all(frame_idx in metadata for frame_idx in frame_indices):
                    continue
                clips.append(
                    RoiClip(
                        video_id=video_id,
                        start_frame=start_frame,
                        item_indices=tuple(item_by_video_frame[(video_id, frame_idx)] for frame_idx in frame_indices),
                    )
                )

        if max_samples > 0:
            clips = clips[: int(max_samples)]
        if not clips:
            raise RuntimeError(
                "No consecutive clips with ROI, geometry, pose, face metadata, and audio were found. "
                f"split={split_name}, face_root={self.face_root}, metadata_root={self.roi_metadata_root}"
            )
        self.clips = clips

    def _load_video_metadata(self, video_id: str) -> dict[int, tuple[Path, np.ndarray]]:
        if video_id in self._metadata_by_video:
            return self._metadata_by_video[video_id]
        speaker = video_id.split("_", 1)[0]
        metadata_dir = self.roi_metadata_root / f"{speaker}_roi" / video_id
        meta_path = metadata_dir / "meta.csv"
        boxes_path = metadata_dir / "lip_rois.npy"
        if not meta_path.exists() or not boxes_path.exists():
            self._metadata_by_video[video_id] = {}
            return {}

        boxes = np.load(boxes_path)
        rows = list(csv.DictReader(meta_path.open("r", encoding="utf-8", newline="")))
        metadata: dict[int, tuple[Path, np.ndarray]] = {}
        for row_index, row in enumerate(rows):
            if row_index >= len(boxes) or int(row.get("ok", "0")) != 1:
                continue
            frame_idx = int(row["frame_idx"])
            filename = row["filename"]
            candidates = [
                self.face_root / video_id / filename,
                self.face_root / video_id / "crops" / filename,
            ]
            face_path = next((path for path in candidates if path.exists()), None)
            box = np.asarray(boxes[row_index], dtype=np.int32)
            if face_path is None or box.shape != (4,) or np.any(box < 0):
                continue
            metadata[frame_idx] = (face_path, box)
        self._metadata_by_video[video_id] = metadata
        return metadata

    def _load_face_and_box(self, video_id: str, frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        face_path, box = self._metadata_by_video[video_id][frame_idx]
        image = Image.open(face_path).convert("RGB")
        original_width, original_height = image.size
        image = image.resize((self.sync_face_size, self.sync_face_size), Image.Resampling.BICUBIC)
        face = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()

        x, y, width, height = [int(value) for value in box]
        x1 = int(round(x * self.sync_face_size / original_width))
        y1 = int(round(y * self.sync_face_size / original_height))
        x2 = int(round((x + width) * self.sync_face_size / original_width))
        y2 = int(round((y + height) * self.sync_face_size / original_height))
        x1 = min(max(x1, 0), self.sync_face_size - 1)
        y1 = min(max(y1, 0), self.sync_face_size - 1)
        x2 = min(max(x2, x1 + 1), self.sync_face_size)
        y2 = min(max(y2, y1 + 1), self.sync_face_size)
        return face, torch.tensor([x1, y1, x2, y2], dtype=torch.int64)

    def load_video_mel(self, video_id: str) -> torch.Tensor:
        cached = self._mel_memory_cache.get(video_id)
        if cached is not None:
            return cached
        cache_path = self.audio_mel_cache_dir / f"{video_id}_mel.pt"
        if cache_path.exists():
            mel = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            mel = _latentsync_melspectrogram(self._audio_by_video[video_id])
            temporary_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(mel, temporary_path)
            try:
                os.replace(temporary_path, cache_path)
            except FileNotFoundError:
                pass
            finally:
                temporary_path.unlink(missing_ok=True)
        self._mel_memory_cache[video_id] = mel
        return mel

    def crop_mel(self, video_id: str, start_frame: int) -> tuple[torch.Tensor, int]:
        mel = self.load_video_mel(video_id)
        start_index = int(80.0 * (float(start_frame) / self.source_fps))
        window_length = int(np.ceil(self.num_frames / 5.0 * 16.0))
        window = mel[:, start_index : start_index + window_length]
        if window.shape[1] < window_length:
            padding = torch.full(
                (window.shape[0], window_length - window.shape[1]),
                -4.0,
                dtype=window.dtype,
            )
            window = torch.cat([window, padding], dim=1)
        return window.unsqueeze(0).contiguous(), start_index

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip = self.clips[idx]
        reference_item = self.base._choose_reference_item(clip.item_indices[0])
        reference = load_image_tensor(reference_item.roi_path, self.base.image_size)
        targets: list[torch.Tensor] = []
        geometries: list[torch.Tensor] = []
        poses: list[torch.Tensor] = []
        faces: list[torch.Tensor] = []
        roi_boxes: list[torch.Tensor] = []
        frame_indices: list[int] = []

        for item_index in clip.item_indices:
            item = self.base.items[item_index]
            geometry = load_geometry81(item.geometry_path)
            pose = load_pose6(item.pose_path) if item.pose_path is not None else NEUTRAL_POSE6.copy()
            face, roi_box = self._load_face_and_box(item.video_id, item.frame_idx)
            targets.append(load_image_tensor(item.roi_path, self.base.image_size))
            geometries.append(torch.from_numpy(geometry))
            poses.append(torch.from_numpy(pose.astype(np.float32)))
            faces.append(face)
            roi_boxes.append(roi_box)
            frame_indices.append(item.frame_idx)

        mel, mel_start_index = self.crop_mel(clip.video_id, clip.start_frame)
        return {
            "reference": reference.unsqueeze(0).expand(self.num_frames, -1, -1, -1).contiguous(),
            "target": torch.stack(targets),
            "geometry": torch.stack(geometries),
            "pose": torch.stack(poses),
            "face": torch.stack(faces),
            "roi_box": torch.stack(roi_boxes),
            "mel": mel,
            "video_id": clip.video_id,
            "frame_idx": torch.tensor(frame_indices, dtype=torch.int64),
            "clip_start_frame": clip.start_frame,
            "mel_start_index": mel_start_index,
        }
