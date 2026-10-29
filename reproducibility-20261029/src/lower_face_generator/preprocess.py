from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import (
    LIP_INDICES,
    LOWER_FACE_INDICES,
    build_lower_face_mask,
    canonicalize_lower_face,
    pack_lower_face_projection19,
    project_lower_face_to_image,
)


@dataclass(frozen=True)
class StableLowerFaceCrop:
    x: float
    y: float
    width: float
    height: float
    output_width: int = 320
    output_height: int = 256

    @property
    def face_to_crop(self) -> np.ndarray:
        scale_x = self.output_width / self.width
        scale_y = self.output_height / self.height
        if abs(scale_x - scale_y) > 1e-4:
            raise ValueError("Lower-face crop must preserve aspect ratio")
        return np.asarray(
            [
                [scale_x, 0.0, -self.x * scale_x],
                [0.0, scale_y, -self.y * scale_y],
            ],
            dtype=np.float32,
        )

    @property
    def crop_to_face(self) -> np.ndarray:
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2] = self.face_to_crop
        return np.linalg.inv(matrix)[:2].astype(np.float32)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "padding_left": max(0.0, -self.x),
            "padding_top": max(0.0, -self.y),
            "padding_right": max(0.0, self.x + self.width - 512.0),
            "padding_bottom": max(0.0, self.y + self.height - 512.0),
        }


def transform_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    matrix = np.asarray(affine, dtype=np.float32).reshape(2, 3)
    return values[..., :2] @ matrix[:, :2].T + matrix[:, 2]


def fit_stable_lower_face_crop(
    face_landmarks: np.ndarray,
    valid: np.ndarray,
    output_size: tuple[int, int] = (320, 256),
    margin_x: float = 12.0,
    margin_top: float = 14.0,
    margin_bottom: float = 12.0,
    image_size: int = 512,
) -> StableLowerFaceCrop:
    points = np.asarray(face_landmarks, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if points.ndim != 3 or points.shape[1] < 468 or points.shape[2] < 2:
        raise ValueError(f"Expected landmarks [T,>=468,>=2], got {points.shape}")
    if not valid.any():
        raise ValueError("No valid frames for lower-face crop fitting")

    selected = points[valid][:, np.asarray(LOWER_FACE_INDICES), :2]
    # The crop is a persistent geometric contract, so every valid lower-face
    # point must fit. Robust quantiles are suitable for appearance crops but
    # can silently discard an extreme cheek/jaw landmark.
    left = float(np.min(selected[..., 0])) - margin_x
    right = float(np.max(selected[..., 0])) + margin_x
    top = float(np.min(selected[..., 1])) - margin_top
    bottom = float(np.max(selected[..., 1])) + margin_bottom

    output_width, output_height = output_size
    target_ratio = float(output_width) / float(output_height)
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    width = max(right - left, 32.0)
    height = max(bottom - top, 32.0)
    if width / height < target_ratio:
        width = height * target_ratio
    else:
        height = width / target_ratio
    # Do not clip the crop to the aligned face. A few HDTF faces place the
    # chin or cheek outside 512x512. Reflected padding preserves aspect and
    # full geometry coverage, whereas independently clamping width and height
    # both drops landmarks and breaks the required 320:256 aspect ratio.
    x = float(center_x - width / 2.0)
    y = float(center_y - height / 2.0)
    return StableLowerFaceCrop(
        x=x,
        y=y,
        width=width,
        height=height,
        output_width=output_width,
        output_height=output_height,
    )


def warp_lower_face(face_bgr: np.ndarray, crop: StableLowerFaceCrop) -> np.ndarray:
    return cv2.warpAffine(
        face_bgr,
        crop.face_to_crop,
        (crop.output_width, crop.output_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def soft_rectangle_mask(
    width: int,
    height: int,
    inset: int = 4,
    feather: int = 7,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    inset = int(np.clip(inset, 0, min(width, height) // 3))
    mask[inset : height - inset, inset : width - inset] = 1.0
    if feather > 0:
        sigma = max(0.5, float(feather) / 2.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def build_crop_masks(
    face_landmarks_xy: np.ndarray,
    crop: StableLowerFaceCrop,
    face_size: int = 512,
    upper_boundary_mode: str = "legacy_horizontal",
    upper_overlap_px: int = 0,
    upper_boundary_parameters: np.ndarray | None = None,
    upper_boundary_curve_xy: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    masks = build_lower_face_mask(
        face_landmarks_xy,
        face_size,
        upper_boundary_mode=upper_boundary_mode,
        upper_boundary_parameters=upper_boundary_parameters,
        upper_boundary_curve_xy=upper_boundary_curve_xy,
    )
    lips = np.zeros((face_size, face_size), dtype=np.uint8)
    lip_points = np.rint(
        np.asarray(face_landmarks_xy)[np.asarray(LIP_INDICES), :2]
    ).astype(np.int32)
    cv2.fillConvexPoly(lips, cv2.convexHull(lip_points), 1, lineType=cv2.LINE_AA)
    lips = cv2.dilate(
        lips,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7)),
    )
    allowed = masks["allowed"]
    lip_oral = lips * allowed
    motion = allowed * (1 - lip_oral)
    outer = cv2.dilate(allowed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    inner = cv2.erode(allowed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    ring = np.uint8((outer - inner) > 0)
    # V2 trains a true lower-face completion task. The opaque interior is the
    # whole valid lower face, not only the lips; the ring remains available for
    # boundary supervision and the lip mask for region-weighted losses.
    inpaint_core = cv2.erode(
        allowed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    def warp(mask: np.ndarray) -> np.ndarray:
        return cv2.warpAffine(
            np.uint8(mask) * 255,
            crop.face_to_crop,
            (crop.output_width, crop.output_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )

    result = {
        "core": warp(lip_oral),
        "lip_oral": warp(lip_oral),
        "inpaint_core": warp(inpaint_core),
        "edit": warp(allowed),
        "motion": warp(motion),
        "ring": warp(ring),
        "anatomical": warp(allowed),
        "nose_exclusion": warp(masks["nose_exclusion"]),
        "upper_boundary": warp(masks["upper_boundary"]),
        "soft_rectangle": np.uint8(
            soft_rectangle_mask(crop.output_width, crop.output_height) * 255.0 + 0.5
        ),
    }
    result["alpha_core"] = result["inpaint_core"].copy()
    overlap_width = max(0, int(upper_overlap_px))
    upper_overlap = np.zeros_like(result["edit"], dtype=np.uint8)
    upper_alpha_target = np.zeros_like(result["edit"], dtype=np.uint8)
    if overlap_width > 0 and upper_boundary_mode in (
        "pose_curved_v1",
        "anchor_curved_v2",
        "tracked_anchor_curved_v3",
        "tracked_transition_v4",
    ):
        edit_binary = np.uint8(result["edit"] > 0)
        boundary_binary = np.uint8(result["upper_boundary"] > 0)
        distance = cv2.distanceTransform(1 - boundary_binary, cv2.DIST_L2, 5)
        overlap_binary = np.uint8(
            (edit_binary > 0) & (distance <= float(overlap_width))
        )
        alpha_target = np.clip(distance / float(overlap_width), 0.0, 1.0) * overlap_binary
        upper_overlap = overlap_binary * 255
        upper_alpha_target = np.uint8(alpha_target * 255.0 + 0.5)
        # The upper band is transition-authoritative. The generated completion
        # becomes opaque only after this directional ramp, while the existing
        # jaw/chin core remains unchanged.
        result["alpha_core"][overlap_binary > 0] = 0
        result["ring"] = np.maximum(result["ring"], upper_overlap)
    result["upper_overlap"] = upper_overlap
    result["upper_alpha_target"] = upper_alpha_target
    return result


def track_upper_transition_step(
    previous_support: np.ndarray,
    previous_alpha: np.ndarray,
    current_overlap: np.ndarray,
    current_alpha: np.ndarray,
    current_edit: np.ndarray,
    current_nose: np.ndarray,
    previous_points_px: np.ndarray,
    current_points_px: np.ndarray,
    correction: float = 0.05,
    safety_dilate_px: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one causal upper-transition state."""

    support = np.uint8(np.asarray(previous_support) > 0)
    alpha = np.asarray(previous_alpha, dtype=np.float32)
    overlap = np.uint8(np.asarray(current_overlap) > 0)
    target = np.clip(np.asarray(current_alpha, dtype=np.float32), 0.0, 1.0)
    edit = np.uint8(np.asarray(current_edit) > 0)
    nose = np.uint8(np.asarray(current_nose) > 0)
    if not (support.shape == alpha.shape == overlap.shape == target.shape == edit.shape == nose.shape):
        raise ValueError("Transition-step masks must share one [H,W] shape")
    previous_points = np.ascontiguousarray(previous_points_px, dtype=np.float32)
    current_points = np.ascontiguousarray(current_points_px, dtype=np.float32)
    transform, _ = cv2.estimateAffinePartial2D(
        previous_points, current_points, method=cv2.LMEDS
    )
    height, width = support.shape
    if transform is None or not np.isfinite(transform).all():
        warped_support = np.zeros_like(support)
        warped_alpha = np.zeros_like(alpha)
    else:
        transform = transform.astype(np.float32)
        warped_support = cv2.warpAffine(
            support, transform, (width, height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_alpha = cv2.warpAffine(
            alpha, transform, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
    radius = max(0, int(safety_dilate_px))
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
        edit = cv2.dilate(edit, kernel)
    safe = (edit > 0) & ~(nose > 0)
    correction = float(np.clip(correction, 0.0, 1.0))
    candidate = warped_alpha + correction * (target - warped_alpha)
    active = ((warped_support > 0) | (overlap > 0)) & safe
    candidate *= active
    next_support = np.uint8(active & (candidate > (0.5 / 255.0)))
    return next_support * 255, np.uint8(
        np.clip(candidate * next_support, 0.0, 1.0) * 255.0 + 0.5
    )


def track_upper_transition_sequence(
    overlap_masks: np.ndarray,
    alpha_targets: np.ndarray,
    edit_masks: np.ndarray,
    nose_masks: np.ndarray,
    stability_points_px: np.ndarray,
    correction: float = 0.05,
    safety_dilate_px: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Causally transport the seam support and its soft alpha field."""

    overlap = np.asarray(overlap_masks, dtype=np.uint8)
    alpha = np.asarray(alpha_targets, dtype=np.float32)
    edit = np.asarray(edit_masks, dtype=np.uint8)
    nose = np.asarray(nose_masks, dtype=np.uint8)
    points = np.asarray(stability_points_px, dtype=np.float32)
    if overlap.ndim != 3 or alpha.shape != overlap.shape:
        raise ValueError("Upper transition arrays must have shape [T,H,W]")
    if edit.shape != overlap.shape or nose.shape != overlap.shape:
        raise ValueError("Transition safety masks must match upper transition shape")
    if points.ndim != 3 or points.shape[0] != overlap.shape[0] or points.shape[2] != 2:
        raise ValueError("Stability points must have shape [T,N,2]")
    if not len(overlap):
        return overlap.copy(), np.uint8(np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5)

    correction = float(np.clip(correction, 0.0, 1.0))
    kernel_size = max(0, int(safety_dilate_px)) * 2 + 1
    kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if kernel_size > 1
        else None
    )
    tracked_support = np.zeros_like(overlap, dtype=np.uint8)
    tracked_alpha = np.zeros_like(alpha, dtype=np.float32)

    def safety(frame: int) -> np.ndarray:
        allowed = np.uint8(edit[frame] > 0)
        if kernel is not None:
            allowed = cv2.dilate(allowed, kernel)
        return (allowed > 0) & ~(nose[frame] > 0)

    tracked_support[0] = np.uint8((overlap[0] > 0) & safety(0))
    tracked_alpha[0] = np.clip(alpha[0], 0.0, 1.0) * tracked_support[0]
    for frame in range(1, len(overlap)):
        next_support, next_alpha = track_upper_transition_step(
            tracked_support[frame - 1],
            tracked_alpha[frame - 1],
            overlap[frame],
            alpha[frame],
            edit[frame],
            nose[frame],
            points[frame - 1],
            points[frame],
            correction=correction,
            safety_dilate_px=safety_dilate_px,
        )
        tracked_support[frame] = np.uint8(next_support > 0)
        tracked_alpha[frame] = next_alpha.astype(np.float32) / 255.0
    return tracked_support * 255, np.uint8(
        np.clip(tracked_alpha, 0.0, 1.0) * 255.0 + 0.5
    )


def frame_geometry_contract(
    landmarks_xyz: np.ndarray,
    face_rotation: np.ndarray,
    original_to_face_affine: np.ndarray,
) -> dict[str, np.ndarray | float]:
    canonical, origin, scale = canonicalize_lower_face(
        landmarks_xyz,
        face_rotation,
    )
    projection = pack_lower_face_projection19(
        face_rotation,
        origin,
        scale,
        original_to_face_affine,
    )
    projected = project_lower_face_to_image(canonical, projection)
    return {
        "canonical_xyz": canonical,
        "projection19": projection,
        "projected_face_xy": projected,
        "interocular_scale": scale,
    }


def write_contract_json(path: Path, crop: StableLowerFaceCrop) -> None:
    payload = {
        "version": "hdtf_lower_face_contract_v1",
        "aligned_face_size": [512, 512],
        "network_crop_size": [crop.output_width, crop.output_height],
        "crop_policy": "temporally_stable_aspect_preserving",
        "crop": crop.as_dict(),
        "dense_masks": "derived_from_face_landmarks_xy_at_load_time",
        "composite_modes": ["soft_rectangle", "anatomical_nose_excluded"],
        "soft_rectangle": {"inset_px": 4, "feather_px": 7},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "StableLowerFaceCrop",
    "build_crop_masks",
    "fit_stable_lower_face_crop",
    "frame_geometry_contract",
    "soft_rectangle_mask",
    "track_upper_transition_sequence",
    "track_upper_transition_step",
    "transform_points",
    "warp_lower_face",
    "write_contract_json",
]
