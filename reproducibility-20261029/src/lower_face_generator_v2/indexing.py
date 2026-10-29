from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lower_face_generator.geometry import FLESH_INDICES, JAW_INDICES, LOWER_FACE_INDICES


INDEX_VERSION = "lower_face_training_index_v2"


def _read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def _geometry81_path(row: dict) -> Path:
    if row.get("geometry81"):
        return Path(row["geometry81"])
    clip_id = str(row["clip_id"])
    identity = str(row["identity"])
    if row["domain"] == "hdtf":
        base = Path(r"C:\LipSyncAnim_HDTF_ProbeData\processed\roi_metadata_frontal2d_v2")
    elif row["domain"] == "voxceleb2":
        base = Path(r"C:\LipSyncAnim_VoxCeleb2_Work\processed\roi_metadata_frontal2d_v3")
    else:
        raise ValueError(f"Unsupported domain {row['domain']!r}")
    return base / f"{identity}_roi" / clip_id / "geometry81.npy"


def _even_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, maximum).round().astype(np.int64))


def _fit_pca(samples: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray]:
    mean = samples.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = samples.astype(np.float32) - mean
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    basis = right[:components].astype(np.float32)
    return mean, basis


def _group_positions(indices: tuple[int, ...]) -> np.ndarray:
    lookup = {landmark: index for index, landmark in enumerate(LOWER_FACE_INDICES)}
    return np.asarray([lookup[index] for index in indices], dtype=np.int64)


_JAW_POSITIONS = _group_positions(JAW_INDICES)
_FLESH_POSITIONS = _group_positions(FLESH_INDICES)


def articulation_descriptor(
    geometry81: np.ndarray,
    lower_geometry: np.ndarray,
    clip_center: np.ndarray,
    pca_basis: np.ndarray,
) -> np.ndarray:
    geometry = np.asarray(geometry81, dtype=np.float32).reshape(-1, 81)
    lower = np.asarray(lower_geometry, dtype=np.float32).reshape(-1, len(LOWER_FACE_INDICES), 3)
    lips = geometry[:, :80].reshape(-1, 40, 2)
    lip_width = np.ptp(lips[..., 0], axis=1)
    lip_height = np.ptp(lips[..., 1], axis=1)
    aspect = lip_height / np.maximum(lip_width, 1e-6)
    jaw = lower[:, _JAW_POSITIONS]
    flesh = lower[:, _FLESH_POSITIONS]
    jaw_span = np.ptp(jaw[..., 1], axis=1)
    jaw_center_y = jaw[..., 1].mean(axis=1)
    flesh_center_y = flesh[..., 1].mean(axis=1)
    centered = lower.reshape(len(lower), -1) - np.asarray(clip_center, dtype=np.float32)
    pca = centered @ np.asarray(pca_basis, dtype=np.float32).T
    base = np.stack(
        [lip_width, lip_height, aspect, geometry[:, 80], jaw_span, jaw_center_y, flesh_center_y],
        axis=1,
    )
    return np.concatenate([base.astype(np.float32), pca.astype(np.float32)], axis=1)


def _farthest_point_indices(descriptors: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float32)
    if len(values) <= count:
        return np.arange(len(values), dtype=np.int64)
    center = values.mean(axis=0)
    first = int(np.argmin(np.sum((values - center) ** 2, axis=1)))
    selected = [first]
    nearest = np.sum((values - values[first]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(nearest))
        selected.append(index)
        nearest = np.minimum(nearest, np.sum((values - values[index]) ** 2, axis=1))
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True)
class LowerFaceTrainingIndexV2:
    path: Path
    arrays: dict[str, np.ndarray]

    @classmethod
    def load(cls, path: str | Path) -> "LowerFaceTrainingIndexV2":
        resolved = Path(path)
        with np.load(resolved, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        version = str(arrays["version"].item())
        if version != INDEX_VERSION:
            raise ValueError(f"Unsupported index version {version!r}")
        return cls(resolved, arrays)

    @property
    def clip_count(self) -> int:
        return len(self.arrays["clip_ids"])

    def reference_slice(self, identity_index: int) -> slice:
        offsets = self.arrays["identity_offsets"]
        return slice(int(offsets[identity_index]), int(offsets[identity_index + 1]))


def build_training_index(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    pca_components: int = 8,
    pca_samples: int = 12000,
    center_samples_per_clip: int = 256,
    candidates_per_clip: int = 64,
    reference_bank_size: int = 32,
    sequence_length: int = 16,
    temporal_exclusion: int = 16,
    seed: int = 325,
) -> dict:
    manifest_path = Path(manifest).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(manifest_path)
    rng = np.random.default_rng(seed)

    seen_clips: set[tuple[str, str]] = set()
    identity_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    enriched: list[dict] = []
    centers: list[np.ndarray] = []
    pca_pool: list[np.ndarray] = []
    pca_seen = 0

    for clip_index, row in enumerate(tqdm(rows, desc="validate/assets+pca", unit="clip")):
        key = (str(row["domain"]), str(row["clip_id"]))
        if key in seen_clips:
            raise ValueError(f"Duplicate clip in manifest: {key}")
        seen_clips.add(key)
        identity_key = (str(row["domain"]), str(row["identity"]))
        identity_splits[identity_key].add(str(row["split"]))
        root = Path(row["root"])
        geometry81_path = _geometry81_path(row)
        required = {
            "root": root,
            "geometry81": geometry81_path,
            "audio_features": Path(row["audio_features"]),
            "lower_face_geometry": root / "lower_face_geometry_xyz.npy",
            "lower_face_points": root / "lower_face_landmarks_crop.npy",
            "projection19": root / "lower_face_projection19.npy",
            "pose6": root / "pose6.npy",
            "landmarks": root / "face_landmarks_xy.npy",
            "contract": root / "contract.json",
        }
        missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{row['clip_id']}: " + "; ".join(missing))
        frame_count = int(row["frame_count"])
        geometry81 = np.load(geometry81_path, mmap_mode="r")
        lower = np.load(required["lower_face_geometry"], mmap_mode="r")
        if geometry81.shape != (frame_count, 81):
            raise ValueError(f"{row['clip_id']}: geometry81 {geometry81.shape}, expected {(frame_count, 81)}")
        if lower.shape != (frame_count, len(LOWER_FACE_INDICES), 3):
            raise ValueError(f"{row['clip_id']}: lower geometry {lower.shape}")
        center_indices = _even_indices(frame_count, center_samples_per_clip)
        center = np.asarray(lower[center_indices], dtype=np.float32).reshape(len(center_indices), -1).mean(axis=0)
        centers.append(center.astype(np.float32))

        if row["split"] == "train":
            sample_indices = _even_indices(frame_count, min(32, frame_count))
            centered = np.asarray(lower[sample_indices], dtype=np.float32).reshape(len(sample_indices), -1) - center
            for sample in centered:
                pca_seen += 1
                if len(pca_pool) < pca_samples:
                    pca_pool.append(sample.copy())
                else:
                    replace = int(rng.integers(0, pca_seen))
                    if replace < pca_samples:
                        pca_pool[replace] = sample.copy()

        enriched_row = dict(row)
        enriched_row.update(
            {
                "training_index_clip": clip_index,
                "geometry81": str(geometry81_path.resolve()),
                "teeth_source": "geometry81[-1]",
                "lower_face_geometry": str(required["lower_face_geometry"].resolve()),
                "lower_face_points": str(required["lower_face_points"].resolve()),
                "projection19": str(required["projection19"].resolve()),
                "pose6": str(required["pose6"].resolve()),
                "face_landmarks": str(required["landmarks"].resolve()),
                "contract": str(required["contract"].resolve()),
            }
        )
        enriched.append(enriched_row)

    leaks = {key: value for key, value in identity_splits.items() if len(value) != 1}
    if leaks:
        example = next(iter(leaks.items()))
        raise ValueError(f"Identity split leakage: {example}")
    if len(pca_pool) < pca_components:
        raise ValueError("Insufficient training samples for lower-face PCA")
    pca_mean, pca_basis = _fit_pca(np.stack(pca_pool), pca_components)
    clip_centers = np.stack(centers).astype(np.float32)

    identities = sorted({(str(row["domain"]), str(row["identity"])) for row in enriched})
    identity_lookup = {key: index for index, key in enumerate(identities)}
    clips_by_identity: dict[int, list[int]] = defaultdict(list)
    for clip_index, row in enumerate(enriched):
        clips_by_identity[identity_lookup[(row["domain"], row["identity"])]].append(clip_index)

    candidate_clip: list[int] = []
    candidate_frame: list[int] = []
    candidate_descriptor: list[np.ndarray] = []
    identity_offsets = [0]
    raw_descriptor_pool: list[np.ndarray] = []
    per_identity_candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for identity_index in tqdm(range(len(identities)), desc="reference banks", unit="identity"):
        local_clips: list[int] = []
        local_frames: list[int] = []
        local_descriptors: list[np.ndarray] = []
        for clip_index in clips_by_identity[identity_index]:
            row = enriched[clip_index]
            frames = _even_indices(int(row["frame_count"]), candidates_per_clip)
            geometry81 = np.load(row["geometry81"], mmap_mode="r")
            lower = np.load(row["lower_face_geometry"], mmap_mode="r")
            descriptors = articulation_descriptor(
                np.asarray(geometry81[frames]), np.asarray(lower[frames]), clip_centers[clip_index], pca_basis
            )
            local_clips.extend([clip_index] * len(frames))
            local_frames.extend(frames.tolist())
            local_descriptors.extend(descriptors)
        descriptor_array = np.asarray(local_descriptors, dtype=np.float32)
        per_identity_candidates.append(
            (
                np.asarray(local_clips, dtype=np.int32),
                np.asarray(local_frames, dtype=np.int32),
                descriptor_array,
            )
        )
        if identities[identity_index] and enriched[clips_by_identity[identity_index][0]]["split"] == "train":
            raw_descriptor_pool.append(descriptor_array)

    train_descriptors = np.concatenate(raw_descriptor_pool, axis=0)
    descriptor_mean = train_descriptors.mean(axis=0).astype(np.float32)
    descriptor_std = train_descriptors.std(axis=0).clip(1e-5).astype(np.float32)
    for clips, frames, descriptors in per_identity_candidates:
        normalized = (descriptors - descriptor_mean) / descriptor_std
        chosen = _farthest_point_indices(normalized, reference_bank_size)
        candidate_clip.extend(clips[chosen].tolist())
        candidate_frame.extend(frames[chosen].tolist())
        candidate_descriptor.extend(normalized[chosen])
        identity_offsets.append(len(candidate_clip))

    enriched_path = output / "manifest_mixed_training_v2.jsonl"
    enriched_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in enriched), encoding="utf-8"
    )
    sequence_counts = np.asarray(
        [max(0, int(row["frame_count"]) - sequence_length + 1) for row in enriched], dtype=np.int64
    )
    index_path = output / "training_index_v2.npz"
    np.savez_compressed(
        index_path,
        version=np.asarray(INDEX_VERSION),
        manifest=np.asarray(str(enriched_path)),
        clip_ids=np.asarray([row["clip_id"] for row in enriched]),
        clip_domains=np.asarray([row["domain"] for row in enriched]),
        clip_identities=np.asarray([row["identity"] for row in enriched]),
        clip_splits=np.asarray([row["split"] for row in enriched]),
        clip_frame_counts=np.asarray([row["frame_count"] for row in enriched], dtype=np.int64),
        clip_identity_indices=np.asarray(
            [identity_lookup[(row["domain"], row["identity"])] for row in enriched], dtype=np.int32
        ),
        clip_geometry_centers=clip_centers,
        identity_domains=np.asarray([value[0] for value in identities]),
        identity_names=np.asarray([value[1] for value in identities]),
        identity_offsets=np.asarray(identity_offsets, dtype=np.int64),
        reference_clip_indices=np.asarray(candidate_clip, dtype=np.int32),
        reference_frame_indices=np.asarray(candidate_frame, dtype=np.int32),
        reference_descriptors=np.asarray(candidate_descriptor, dtype=np.float32),
        descriptor_mean=descriptor_mean,
        descriptor_std=descriptor_std,
        pca_mean=pca_mean,
        pca_basis=pca_basis,
        sequence_length=np.asarray(sequence_length, dtype=np.int32),
        sequence_counts=sequence_counts,
        sequence_ends=np.cumsum(sequence_counts, dtype=np.int64),
        temporal_exclusion=np.asarray(temporal_exclusion, dtype=np.int32),
    )
    summary = {
        "version": INDEX_VERSION,
        "source_manifest": str(manifest_path),
        "enriched_manifest": str(enriched_path),
        "index": str(index_path),
        "clips": len(enriched),
        "identities": len(identities),
        "frames": int(sum(int(row["frame_count"]) for row in enriched)),
        "sequences": int(sequence_counts.sum()),
        "sequence_length": sequence_length,
        "reference_candidates": len(candidate_clip),
        "reference_bank_size": reference_bank_size,
        "temporal_exclusion": temporal_exclusion,
        "pca_components": pca_components,
        "pca_fit_samples": len(pca_pool),
        "identity_split_leaks": 0,
    }
    (output / "training_index_v2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "INDEX_VERSION",
    "LowerFaceTrainingIndexV2",
    "articulation_descriptor",
    "build_training_index",
]
