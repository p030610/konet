from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lower_face_generator.geometry import LIP_INDICES, LOWER_FACE_INDICES

from .source_space import build_source_space_union_masks


MASK_CONTRACT_V5_VERSION = "lower_face_exclusive_ownership_v5"

_LOWER_LOOKUP = {landmark: index for index, landmark in enumerate(LOWER_FACE_INDICES)}
_LIP_POSITIONS = np.asarray(
    [_LOWER_LOOKUP[index] for index in LIP_INDICES], dtype=np.int32
)


class MaskContractV5Error(ValueError):
    pass


@dataclass(frozen=True)
class MaskContractV5Config:
    lip_dilate_x: int = 9
    lip_dilate_y: int = 7
    swept_dilate_x: int = 11
    swept_dilate_y: int = 9
    reference_guard_dilate_x: int = 17
    reference_guard_dilate_y: int = 13
    reference_exclusion_dilate_px: int = 7
    support_margin_px: int = 4
    feather_px: int = 7
    component_close_px: int = 5

    def validate(self) -> None:
        odd_fields = (
            "lip_dilate_x",
            "lip_dilate_y",
            "swept_dilate_x",
            "swept_dilate_y",
            "reference_guard_dilate_x",
            "reference_guard_dilate_y",
            "component_close_px",
        )
        for name in odd_fields:
            value = int(getattr(self, name))
            if value < 1 or value % 2 == 0:
                raise MaskContractV5Error(f"{name} must be a positive odd integer, got {value}")
        if self.reference_exclusion_dilate_px < 0:
            raise MaskContractV5Error("reference_exclusion_dilate_px must be non-negative")
        if self.support_margin_px < 0:
            raise MaskContractV5Error("support_margin_px must be non-negative")
        if self.feather_px < 1:
            raise MaskContractV5Error("feather_px must be positive")


def _binary(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise MaskContractV5Error(f"{name} must be a 2D mask, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise MaskContractV5Error(f"{name} contains NaN or Inf")
    return np.uint8(array > 0.5)


def _points(value: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    expected = (len(LOWER_FACE_INDICES), 2)
    if points.shape != expected:
        raise MaskContractV5Error(f"{name} must have shape {expected}, got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise MaskContractV5Error(f"{name} contains NaN or Inf")
    return points


def _ellipse(width: int, height: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(width), int(height)))


def _is_binary(value: np.ndarray) -> bool:
    return bool(np.all((value == 0) | (value == 1)))


def build_lip_support_v5(
    points_px: np.ndarray,
    image_shape: tuple[int, int],
    *,
    dilate_x: int = 9,
    dilate_y: int = 7,
) -> np.ndarray:
    """Rasterize a lower-face lip hull in the supplied crop coordinates."""

    points = _points(points_px, name="points_px")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height < 1 or width < 1:
        raise MaskContractV5Error(f"Invalid image shape {image_shape}")
    if dilate_x < 1 or dilate_y < 1 or dilate_x % 2 == 0 or dilate_y % 2 == 0:
        raise MaskContractV5Error("Lip dilation dimensions must be positive odd integers")
    support = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(np.rint(points[_LIP_POSITIONS]).astype(np.int32))
    cv2.fillConvexPoly(support, hull, 1, lineType=cv2.LINE_AA)
    return cv2.dilate(support, _ellipse(dilate_x, dilate_y))


def build_swept_mouth_union_v5(
    source_points_px: np.ndarray,
    target_points_px: np.ndarray,
    image_shape: tuple[int, int],
    *,
    dilate_x: int = 11,
    dilate_y: int = 9,
) -> np.ndarray:
    """Fill the complete source-to-target lip trajectory in source pixels."""

    source = _points(source_points_px, name="source_points_px")
    target = _points(target_points_px, name="target_points_px")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height < 1 or width < 1:
        raise MaskContractV5Error(f"Invalid image shape {image_shape}")
    if dilate_x < 1 or dilate_y < 1 or dilate_x % 2 == 0 or dilate_y % 2 == 0:
        raise MaskContractV5Error("Swept-union dilation dimensions must be positive odd integers")
    lip_points = np.concatenate(
        (source[_LIP_POSITIONS], target[_LIP_POSITIONS]), axis=0
    )
    support = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(np.rint(lip_points).astype(np.int32))
    cv2.fillConvexPoly(support, hull, 1, lineType=cv2.LINE_AA)
    return cv2.dilate(support, _ellipse(dilate_x, dilate_y))


def _retain_native_component(value: np.ndarray, native: np.ndarray, close_px: int) -> np.ndarray:
    binary = np.uint8(value > 0)
    if close_px > 1:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _ellipse(close_px, close_px))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary
    overlaps = np.asarray(
        [np.count_nonzero((labels == index) & (native > 0)) for index in range(1, count)],
        dtype=np.int64,
    )
    if int(overlaps.max(initial=0)) > 0:
        selected = 1 + int(np.argmax(overlaps))
    else:
        selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.uint8(labels == selected)


def _mask_metrics(masks: dict[str, np.ndarray]) -> dict[str, int | float]:
    direct = masks["direct_rgb_ownership"] > 0
    reference = masks["safe_reference_skin"] > 0
    transition = masks["mouth_skin_transition"] > 0
    edit = masks["edit_union"] > 0
    nose = masks["nose_exclusion"] > 0
    partition = direct.astype(np.uint8) + reference.astype(np.uint8) + transition.astype(np.uint8)
    components = cv2.connectedComponents(np.uint8(edit), 8)[0] - 1
    return {
        "direct_reference_overlap_pixels": int(np.count_nonzero(direct & reference)),
        "direct_transition_overlap_pixels": int(np.count_nonzero(direct & transition)),
        "reference_transition_overlap_pixels": int(np.count_nonzero(reference & transition)),
        "ownership_unassigned_edit_pixels": int(np.count_nonzero(edit & (partition == 0))),
        "ownership_multiply_assigned_pixels": int(np.count_nonzero(partition > 1)),
        "edit_nose_overlap_pixels": int(np.count_nonzero(edit & nose)),
        "source_lip_uncovered_pixels": int(
            np.count_nonzero((masks["source_lip_support"] > 0) & ~direct)
        ),
        "target_lip_uncovered_pixels": int(
            np.count_nonzero((masks["target_lip_support"] > 0) & ~direct)
        ),
        "opaque_core_nonopaque_pixels": int(
            np.count_nonzero(
                (masks["opaque_inpaint_core"] > 0) & (masks["alpha_target"] < 1.0)
            )
        ),
        "nose_alpha_pixels": int(
            np.count_nonzero(nose & (masks["alpha_target"] > 0))
        ),
        "edit_components": int(components),
        "edit_fraction": float(np.mean(edit)),
        "direct_fraction": float(np.mean(direct)),
        "reference_skin_fraction": float(np.mean(reference)),
        "transition_fraction": float(np.mean(transition)),
    }


def validate_mask_contract_v5(masks: dict[str, np.ndarray]) -> dict[str, int | float]:
    required = (
        "source_lip_support",
        "target_lip_support",
        "aligned_reference_guard",
        "swept_mouth_union",
        "source_native_edit",
        "target_driven_support",
        "edit_union",
        "nose_exclusion",
        "direct_rgb_ownership",
        "safe_reference_skin",
        "mouth_skin_transition",
        "opaque_inpaint_core",
        "outer_feather",
        "alpha_target",
    )
    missing = [name for name in required if name not in masks]
    if missing:
        raise MaskContractV5Error(f"Missing V5 masks: {missing}")
    shape = np.asarray(masks[required[0]]).shape
    for name in required:
        value = np.asarray(masks[name])
        if value.shape != shape:
            raise MaskContractV5Error(f"V5 mask {name} has shape {value.shape}, expected {shape}")
        if not np.all(np.isfinite(value)):
            raise MaskContractV5Error(f"V5 mask {name} contains NaN or Inf")
        if name != "alpha_target" and not _is_binary(value):
            raise MaskContractV5Error(f"V5 mask {name} is not binary")
    alpha = np.asarray(masks["alpha_target"], dtype=np.float32)
    if np.any(alpha < 0) or np.any(alpha > 1):
        raise MaskContractV5Error("V5 alpha_target must remain in [0, 1]")
    metrics = _mask_metrics(masks)
    zero_metrics = (
        "direct_reference_overlap_pixels",
        "direct_transition_overlap_pixels",
        "reference_transition_overlap_pixels",
        "ownership_unassigned_edit_pixels",
        "ownership_multiply_assigned_pixels",
        "edit_nose_overlap_pixels",
        "source_lip_uncovered_pixels",
        "target_lip_uncovered_pixels",
        "opaque_core_nonopaque_pixels",
        "nose_alpha_pixels",
    )
    failures = {name: metrics[name] for name in zero_metrics if metrics[name] != 0}
    if metrics["edit_components"] != 1:
        failures["edit_components"] = metrics["edit_components"]
    if failures:
        raise MaskContractV5Error(f"V5 mask invariants failed: {failures}")
    return metrics


def build_mask_contract_v5(
    source_edit: np.ndarray,
    source_nose_exclusion: np.ndarray,
    source_points_px: np.ndarray,
    target_points_px: np.ndarray,
    *,
    config: MaskContractV5Config | None = None,
) -> dict[str, np.ndarray]:
    """Build validated V5 lower-face masks in source-crop coordinates."""

    cfg = config or MaskContractV5Config()
    cfg.validate()
    native = _binary(source_edit, name="source_edit")
    nose = _binary(source_nose_exclusion, name="source_nose_exclusion")
    if native.shape != nose.shape:
        raise MaskContractV5Error(
            f"source_edit and source_nose_exclusion differ: {native.shape} vs {nose.shape}"
        )
    source_points = _points(source_points_px, name="source_points_px")
    target_points = _points(target_points_px, name="target_points_px")
    image_shape = native.shape

    source_lip = build_lip_support_v5(
        source_points,
        image_shape,
        dilate_x=cfg.lip_dilate_x,
        dilate_y=cfg.lip_dilate_y,
    )
    target_lip = build_lip_support_v5(
        target_points,
        image_shape,
        dilate_x=cfg.lip_dilate_x,
        dilate_y=cfg.lip_dilate_y,
    )
    source_lip *= 1 - nose
    target_lip *= 1 - nose
    aligned_reference_guard = cv2.dilate(
        target_lip,
        _ellipse(cfg.reference_guard_dilate_x, cfg.reference_guard_dilate_y),
    ) * (1 - nose)
    swept = build_swept_mouth_union_v5(
        source_points,
        target_points,
        image_shape,
        dilate_x=cfg.swept_dilate_x,
        dilate_y=cfg.swept_dilate_y,
    )
    swept = np.maximum(swept, aligned_reference_guard) * (1 - nose)

    if np.max(np.abs(source_points - target_points)) <= 1e-4:
        margin_size = 2 * int(cfg.support_margin_px) + 1
        driven_support = native.copy()
        if cfg.support_margin_px:
            driven_support = cv2.dilate(
                driven_support, _ellipse(margin_size, margin_size)
            )
        driven_support *= 1 - nose
        anatomy = {
            "edit": np.maximum(native, driven_support).astype(np.float32),
            "source_native_edit": native.astype(np.float32),
            "driven_support": driven_support.astype(np.float32),
        }
    else:
        anatomy = build_source_space_union_masks(
            native,
            nose,
            source_points,
            target_points,
            support_margin_px=cfg.support_margin_px,
            core_erode_px=0,
            feather_px=cfg.feather_px,
        )
    edit = np.maximum(np.uint8(anatomy["edit"] > 0), swept) * (1 - nose)
    edit = _retain_native_component(edit, native * (1 - nose), cfg.component_close_px)
    edit *= 1 - nose

    direct = swept * edit
    exclusion_size = 2 * int(cfg.reference_exclusion_dilate_px) + 1
    reference_exclusion = cv2.dilate(direct, _ellipse(exclusion_size, exclusion_size))
    reference_skin = edit * (1 - reference_exclusion)
    transition = edit * (1 - direct) * (1 - reference_skin)

    distance = cv2.distanceTransform(edit, cv2.DIST_L2, 5)
    alpha = np.clip(distance / float(cfg.feather_px), 0.0, 1.0) * edit
    opaque_core = np.uint8((edit > 0) & (distance >= float(cfg.feather_px)))
    alpha[opaque_core > 0] = 1.0
    outer_feather = np.uint8((edit > 0) & (opaque_core == 0))

    masks: dict[str, np.ndarray] = {
        "source_lip_support": source_lip.astype(np.float32),
        "target_lip_support": target_lip.astype(np.float32),
        "aligned_reference_guard": aligned_reference_guard.astype(np.float32),
        "swept_mouth_union": swept.astype(np.float32),
        "source_native_edit": np.uint8(anatomy["source_native_edit"] > 0).astype(np.float32),
        "target_driven_support": np.uint8(anatomy["driven_support"] > 0).astype(np.float32),
        "edit_union": edit.astype(np.float32),
        "nose_exclusion": nose.astype(np.float32),
        "direct_rgb_ownership": direct.astype(np.float32),
        "safe_reference_skin": reference_skin.astype(np.float32),
        "mouth_skin_transition": transition.astype(np.float32),
        "opaque_inpaint_core": opaque_core.astype(np.float32),
        "outer_feather": outer_feather.astype(np.float32),
        "alpha_target": alpha.astype(np.float32),
    }
    validate_mask_contract_v5(masks)
    return masks


def composite_owned_rgb_v5(
    source_rgb: np.ndarray,
    direct_mouth_rgb: np.ndarray,
    reference_skin_rgb: np.ndarray,
    transition_rgb: np.ndarray,
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    """Compose exclusive V5 RGB while preserving source pixels outside alpha."""

    validate_mask_contract_v5(masks)
    source = np.asarray(source_rgb)
    direct_rgb = np.asarray(direct_mouth_rgb)
    reference_rgb = np.asarray(reference_skin_rgb)
    transition_value = np.asarray(transition_rgb)
    if source.ndim != 3 or source.shape[2] != 3:
        raise MaskContractV5Error(f"source_rgb must have shape [H,W,3], got {source.shape}")
    if any(value.shape != source.shape for value in (direct_rgb, reference_rgb, transition_value)):
        raise MaskContractV5Error("All V5 RGB inputs must share the source RGB shape")
    if source.shape[:2] != masks["edit_union"].shape:
        raise MaskContractV5Error("RGB and V5 mask dimensions differ")

    direct = masks["direct_rgb_ownership"][..., None]
    reference = masks["safe_reference_skin"][..., None]
    transition = masks["mouth_skin_transition"][..., None]
    generated = direct_rgb * direct + reference_rgb * reference + transition_value * transition
    alpha = masks["alpha_target"][..., None].astype(np.float32)
    output = source.astype(np.float32) * (1.0 - alpha) + generated.astype(np.float32) * alpha
    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        output = np.rint(np.clip(output, info.min, info.max)).astype(source.dtype)
    else:
        output = output.astype(source.dtype, copy=False)
    output[alpha[..., 0] == 0] = source[alpha[..., 0] == 0]
    return output


__all__ = [
    "MASK_CONTRACT_V5_VERSION",
    "MaskContractV5Config",
    "MaskContractV5Error",
    "build_lip_support_v5",
    "build_mask_contract_v5",
    "build_swept_mouth_union_v5",
    "composite_owned_rgb_v5",
    "validate_mask_contract_v5",
]
