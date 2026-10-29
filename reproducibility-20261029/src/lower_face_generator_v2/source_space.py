from __future__ import annotations

import cv2
import numpy as np
import torch

from lower_face_generator.geometry import LIP_INDICES, LOWER_FACE_INDICES

from .rasterizer import project_geometry_to_crop


_LOWER_LOOKUP = {landmark: index for index, landmark in enumerate(LOWER_FACE_INDICES)}
_LIP_POSITIONS = np.asarray([_LOWER_LOOKUP[index] for index in LIP_INDICES], dtype=np.int32)


def reproject_driving_geometry_to_source(
    driving_geometry_xyz: torch.Tensor,
    source_projection19: torch.Tensor,
    source_face_to_crop_affine: torch.Tensor,
) -> torch.Tensor:
    """Place canonical driving geometry in the source crop coordinate system."""
    return project_geometry_to_crop(
        driving_geometry_xyz, source_projection19, source_face_to_crop_affine
    )


def _binary(value: np.ndarray) -> np.ndarray:
    return np.uint8(np.asarray(value, dtype=np.float32) > 0.5)


def _warp_mask_by_landmarks(
    source_mask: np.ndarray,
    source_points_px: np.ndarray,
    driven_points_px: np.ndarray,
    *,
    sigma_px: float = 18.0,
    grid_scale: int = 4,
) -> np.ndarray:
    """Warp a source-space anatomical mask to driven landmark positions.

    The inverse map is estimated from Gaussian-weighted control-point motion on
    a reduced grid. This preserves the native lower-face silhouette instead of
    replacing it with the convex hull of every lower-face landmark.
    """
    height, width = source_mask.shape
    scale = max(1, int(grid_scale))
    grid_height = max(1, int(np.ceil(height / scale)))
    grid_width = max(1, int(np.ceil(width / scale)))
    source_small = cv2.resize(
        source_mask.astype(np.float32), (grid_width, grid_height), interpolation=cv2.INTER_AREA
    )
    source_points = source_points_px / float(scale)
    driven_points = driven_points_px / float(scale)
    yy, xx = np.mgrid[0:grid_height, 0:grid_width].astype(np.float32)
    grid = np.stack((xx, yy), axis=-1)
    delta = source_points - driven_points
    distance_sq = np.sum((grid[..., None, :] - driven_points[None, None, ...]) ** 2, axis=-1)
    sigma = max(1.0, float(sigma_px) / float(scale))
    weights = np.exp(-distance_sq / (2.0 * sigma * sigma)).astype(np.float32)
    weight_sum = weights.sum(axis=-1, keepdims=True)
    displacement = np.sum(weights[..., None] * delta[None, None, ...], axis=-2)
    displacement /= np.maximum(weight_sum, 1e-6)
    confidence = np.clip(weight_sum[..., 0], 0.0, 1.0)[..., None]
    source_coordinates = grid + displacement * confidence
    warped_small = cv2.remap(
        source_small,
        source_coordinates[..., 0],
        source_coordinates[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped = cv2.resize(warped_small, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.uint8(warped > 0.35)


def build_source_space_union_masks(
    source_edit: np.ndarray,
    source_nose_exclusion: np.ndarray,
    source_points_px: np.ndarray,
    driven_points_px: np.ndarray,
    *,
    support_margin_px: int = 4,
    core_erode_px: int = 2,
    feather_px: int = 7,
) -> dict[str, np.ndarray]:
    """Build all edit fields in source pixels from native and driven geometry."""
    edit_source = _binary(source_edit)
    nose = _binary(source_nose_exclusion)
    if edit_source.shape != nose.shape:
        raise ValueError(f"Source edit/nose shapes differ: {edit_source.shape}, {nose.shape}")
    source_points = np.asarray(source_points_px, dtype=np.float32)
    driven_points = np.asarray(driven_points_px, dtype=np.float32)
    if source_points.shape != (len(LOWER_FACE_INDICES), 2) or driven_points.shape != source_points.shape:
        raise ValueError(
            f"Expected source/driven points {(len(LOWER_FACE_INDICES), 2)}, "
            f"got {source_points.shape} and {driven_points.shape}"
        )
    height, width = edit_source.shape
    driven_support = _warp_mask_by_landmarks(edit_source, source_points, driven_points)
    # Preserve jaw/cheek excursions that a smooth inverse warp can miss. Only
    # controls belonging to the native editable anatomy may seed pixels; upper
    # cheek/nose controls remain deformation clues and must not form islands.
    point_gate = cv2.dilate(
        edit_source, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ) * (1 - nose)
    rounded_source = np.rint(source_points).astype(np.int32)
    rounded_driven = np.rint(driven_points).astype(np.int32)
    support_point_valid = np.zeros(len(source_points), dtype=bool)
    for index, (x, y) in enumerate(rounded_source):
        if 0 <= x < width and 0 <= y < height:
            support_point_valid[index] = point_gate[y, x] > 0
    for index, (x, y) in enumerate(rounded_driven):
        if not (0 <= x < width and 0 <= y < height) or nose[y, x] > 0:
            support_point_valid[index] = False
    for x, y in rounded_driven[support_point_valid]:
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(driven_support, (int(x), int(y)), 2, 1, -1, cv2.LINE_AA)
    margin = max(0, int(support_margin_px))
    if margin:
        driven_support = cv2.dilate(
            driven_support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
        )
    edit = np.maximum(edit_source, driven_support) * (1 - nose)
    edit = np.uint8(edit > 0)

    lip = np.zeros_like(edit)
    lip_hull = cv2.convexHull(np.rint(driven_points[_LIP_POSITIONS]).astype(np.int32))
    cv2.fillConvexPoly(lip, lip_hull, 1, lineType=cv2.LINE_AA)
    lip = cv2.dilate(lip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7)))
    lip = lip * edit * (1 - nose)
    erode = max(0, int(core_erode_px))
    inpaint_core = edit.copy()
    if erode:
        inpaint_core = cv2.erode(
            inpaint_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode + 1,) * 2)
        )
    feather = max(1, int(feather_px))
    distance = cv2.distanceTransform(edit, cv2.DIST_L2, 5)
    alpha = np.clip(distance / float(feather), 0.0, 1.0) * edit
    ring = np.uint8((edit > 0) & (distance <= float(feather)))
    motion = edit * (1 - lip)
    return {
        "edit": edit.astype(np.float32),
        "lip_oral": lip.astype(np.float32),
        "motion": motion.astype(np.float32),
        "inpaint_core": inpaint_core.astype(np.float32),
        "alpha_core": inpaint_core.astype(np.float32),
        "ring": ring.astype(np.float32),
        "alpha_target": alpha.astype(np.float32),
        "nose_exclusion": nose.astype(np.float32),
        "source_native_edit": edit_source.astype(np.float32),
        "driven_support": driven_support.astype(np.float32),
        "support_point_valid": support_point_valid,
    }


__all__ = ["build_source_space_union_masks", "reproject_driving_geometry_to_source"]
