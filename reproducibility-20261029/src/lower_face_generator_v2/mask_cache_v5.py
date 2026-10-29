from __future__ import annotations

from pathlib import Path

import numpy as np


MASK_CACHE_V5_VERSION = "lower_face_mask_cache_v5_chunked_bitpack"
MASK_CACHE_V5_BINARY_NAMES = (
    "source_lip_support",
    "target_lip_support",
    "aligned_reference_guard",
    "swept_mouth_union",
    "edit_union",
    "nose_exclusion",
    "direct_rgb_ownership",
    "safe_reference_skin",
    "mouth_skin_transition",
    "opaque_inpaint_core",
    "outer_feather",
)


def encode_mask_chunk_v5(
    frames: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if not frames:
        raise ValueError("Cannot encode an empty V5 mask chunk")
    shape = np.asarray(frames[0][MASK_CACHE_V5_BINARY_NAMES[0]]).shape
    if len(shape) != 2:
        raise ValueError(f"V5 masks must be 2D, got {shape}")
    binary = np.empty(
        (len(frames), len(MASK_CACHE_V5_BINARY_NAMES), shape[0], shape[1]),
        dtype=np.uint8,
    )
    alpha = np.empty((len(frames), shape[0], shape[1]), dtype=np.uint8)
    for frame_index, masks in enumerate(frames):
        for mask_index, name in enumerate(MASK_CACHE_V5_BINARY_NAMES):
            value = np.asarray(masks[name])
            if value.shape != shape:
                raise ValueError(f"V5 mask {name} has shape {value.shape}, expected {shape}")
            binary[frame_index, mask_index] = np.uint8(value > 0.5)
        alpha_value = np.asarray(masks["alpha_target"], dtype=np.float32)
        if alpha_value.shape != shape or not np.all(np.isfinite(alpha_value)):
            raise ValueError("Invalid V5 alpha_target")
        alpha[frame_index] = np.uint8(np.clip(alpha_value, 0, 1) * 255.0 + 0.5)
    return np.packbits(binary, axis=-1, bitorder="little"), alpha


def decode_mask_chunk_v5(
    packed_masks: np.ndarray,
    alpha_target_u8: np.ndarray,
    *,
    image_width: int,
    compact: bool = False,
) -> dict[str, np.ndarray]:
    packed = np.asarray(packed_masks, dtype=np.uint8)
    alpha = np.asarray(alpha_target_u8, dtype=np.uint8)
    if packed.ndim != 4 or packed.shape[1] != len(MASK_CACHE_V5_BINARY_NAMES):
        raise ValueError(f"Invalid packed V5 mask shape {packed.shape}")
    binary = np.unpackbits(packed, axis=-1, count=int(image_width), bitorder="little")
    expected = (packed.shape[0], packed.shape[2], int(image_width))
    if alpha.shape != expected:
        raise ValueError(f"Invalid V5 alpha shape {alpha.shape}, expected {expected}")
    if compact:
        result = {
            name: binary[:, index]
            for index, name in enumerate(MASK_CACHE_V5_BINARY_NAMES)
        }
        # Keep the cached chunk compact. Dataset access normalizes only the
        # requested frame instead of allocating a float copy of all 64 frames.
        result["alpha_target"] = alpha
    else:
        result = {
            name: binary[:, index].astype(np.float32)
            for index, name in enumerate(MASK_CACHE_V5_BINARY_NAMES)
        }
        result["alpha_target"] = alpha.astype(np.float32) / 255.0
    return result


def load_mask_chunk_v5(path: str | Path) -> dict[str, np.ndarray]:
    resolved = Path(path)
    with np.load(resolved, allow_pickle=False) as payload:
        version = str(payload["version"].item())
        if version != MASK_CACHE_V5_VERSION:
            raise ValueError(f"Unsupported V5 mask cache version {version!r}")
        decoded = decode_mask_chunk_v5(
            payload["packed_masks"],
            payload["alpha_target_u8"],
            image_width=int(payload["image_width"].item()),
            compact=True,
        )
        decoded["source_frame_indices"] = np.asarray(
            payload["source_frame_indices"], dtype=np.int32
        )
        decoded["target_points_px"] = np.asarray(payload["target_points_px"], dtype=np.float32)
        decoded["target_frame_start"] = np.asarray(
            payload["target_frame_start"], dtype=np.int32
        )
    return decoded


__all__ = [
    "MASK_CACHE_V5_BINARY_NAMES",
    "MASK_CACHE_V5_VERSION",
    "decode_mask_chunk_v5",
    "encode_mask_chunk_v5",
    "load_mask_chunk_v5",
]
