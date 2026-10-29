from __future__ import annotations

import cv2
import numpy as np
import torch


# Lips, lower face oval, and cheek/perioral flesh. The order is a persistent
# data contract, not MediaPipe's mesh traversal order.
LIP_INDICES = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
    402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270,
    409, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
)
JAW_INDICES = (
    454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
)
FLESH_INDICES = (
    280, 330, 425, 266, 426, 423, 358, 327, 326, 2, 97, 98, 129,
    203, 206, 36, 205, 101, 50, 187, 207, 216, 212, 202, 204, 194,
    418, 424, 422, 432, 436, 427,
)
LOWER_FACE_INDICES = tuple(dict.fromkeys((*LIP_INDICES, *JAW_INDICES, *FLESH_INDICES)))
LOWER_FACE_POINT_COUNT = len(LOWER_FACE_INDICES)
LOWER_FACE_GEOMETRY_DIM = LOWER_FACE_POINT_COUNT * 3

FACE_OVAL_ORDER = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
    365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
    132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)
NOSE_HULL_INDICES = (168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 97, 326)
UPPER_BOUNDARY_MODES = (
    "legacy_horizontal",
    "pose_curved_v1",
    "anchor_curved_v2",
    "tracked_anchor_curved_v3",
    "tracked_transition_v4",
)
UPPER_BOUNDARY_ANCHOR_INDICES = (2, 94, 97, 326, 205, 425, 50, 280)
TEMPORAL_STABILITY_INDICES = (
    33, 133, 159, 145, 263, 362, 386, 374,
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 97, 326,
)


def upper_boundary_anchor_parameters_sequence(landmarks_xy: np.ndarray) -> np.ndarray:
    """Vectorized articulation-independent parameters for a landmark sequence."""

    points = np.asarray(landmarks_xy, dtype=np.float32)
    if points.ndim != 3 or points.shape[1] < 468 or points.shape[2] < 2:
        raise ValueError(f"Expected image landmarks [T,>=468,2], got {points.shape}")
    left_eye = points[:, 33, :2]
    right_eye = points[:, 263, :2]
    eye_center = 0.5 * (left_eye + right_eye)
    axis_x = right_eye - left_eye
    eye_scale = np.maximum(np.linalg.norm(axis_x, axis=1), 1e-6)
    axis_x = axis_x / eye_scale[:, None]
    axis_y = np.stack([-axis_x[:, 1], axis_x[:, 0]], axis=1)
    orientation = np.sum((points[:, 152, :2] - eye_center) * axis_y, axis=1)
    axis_y[orientation < 0] *= -1.0

    required = np.asarray(
        tuple(dict.fromkeys((*FACE_OVAL_ORDER, 2, 94, 97, 326, 205, 50, 425, 280))),
        dtype=np.int32,
    )
    selected = points[:, required, :2] - eye_center[:, None]
    local_x_selected = np.sum(selected * axis_x[:, None], axis=2) / eye_scale[:, None]
    local_y_selected = np.sum(selected * axis_y[:, None], axis=2) / eye_scale[:, None]
    lookup = {int(value): index for index, value in enumerate(required)}

    def positions(indices: tuple[int, ...]) -> np.ndarray:
        return np.asarray([lookup[value] for value in indices], dtype=np.int32)

    oval = positions(FACE_OVAL_ORDER)
    nose_base = np.asarray((2, 94, 97, 326), dtype=np.int32)
    cheek_left = (205, 50)
    cheek_right = (425, 280)
    nose = positions(tuple(int(value) for value in nose_base))
    left = positions(cheek_left)
    right = positions(cheek_right)
    center_x = np.median(local_x_selected[:, nose], axis=1)
    nose_y = np.median(local_y_selected[:, nose], axis=1)
    cheek_y = 0.5 * (
        np.median(local_y_selected[:, left], axis=1)
        + np.median(local_y_selected[:, right], axis=1)
    )
    cheek_half_width = np.maximum(
        0.5
        * np.abs(
            np.median(local_x_selected[:, right], axis=1)
            - np.median(local_x_selected[:, left], axis=1)
        ),
        0.55,
    )
    center_y = nose_y + 0.045
    side_drop = np.clip(cheek_y - center_y, 0.10, 0.42)
    return np.stack(
        [
            center_x,
            center_y,
            cheek_half_width,
            side_drop,
            np.min(local_x_selected[:, oval], axis=1) - 0.02,
            np.max(local_x_selected[:, oval], axis=1) + 0.02,
        ],
        axis=1,
    ).astype(np.float32)


def upper_boundary_anchor_parameters(landmarks_xy: np.ndarray) -> np.ndarray:
    """Return articulation-independent curve parameters in eye-normalized coordinates."""

    points = np.asarray(landmarks_xy, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"Expected one landmark frame, got {points.shape}")
    return upper_boundary_anchor_parameters_sequence(points[None])[0]


def causal_smooth_upper_boundary_parameters(
    values: np.ndarray,
    alpha: float = 0.18,
    max_step: float = 0.035,
) -> np.ndarray:
    """Causally damp small jitter while following genuine pose changes quickly."""

    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 6:
        raise ValueError(f"Expected boundary parameters [T,6], got {source.shape}")
    if not len(source):
        return source.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    max_step = max(float(max_step), 1e-6)
    output = np.empty_like(source)
    output[0] = source[0]
    for frame in range(1, len(source)):
        delta = source[frame] - output[frame - 1]
        adaptive = alpha + (1.0 - alpha) * np.clip(
            np.abs(delta) / max_step, 0.0, 1.0
        )
        output[frame] = output[frame - 1] + adaptive * delta
    return output


def anchor_upper_boundary_curve(
    landmarks_xy: np.ndarray,
    normalized_parameters: np.ndarray | None = None,
    samples: int = 161,
) -> np.ndarray:
    """Raster-order anchor curve in the current face-image coordinates."""

    points = np.asarray(landmarks_xy, dtype=np.float32)
    normalized = (
        upper_boundary_anchor_parameters(points)
        if normalized_parameters is None
        else np.asarray(normalized_parameters, dtype=np.float32)
    )
    if normalized.shape != (6,):
        raise ValueError(f"Expected six upper-boundary parameters, got {normalized.shape}")
    left_eye = points[33, :2]
    right_eye = points[263, :2]
    axis_x = right_eye - left_eye
    eye_scale = max(float(np.linalg.norm(axis_x)), 1e-6)
    axis_x /= eye_scale
    axis_y = np.asarray([-axis_x[1], axis_x[0]], dtype=np.float32)
    eye_center = 0.5 * (left_eye + right_eye)
    if float(np.dot(points[152, :2] - eye_center, axis_y)) < 0:
        axis_y *= -1.0
    center_x, center_y, half_width, side_drop, left_x, right_x = (
        normalized * eye_scale
    )
    sample_x = np.linspace(float(left_x), float(right_x), int(samples))
    distance = np.abs(sample_x - center_x) / max(float(half_width), 1e-6)
    bend = 1.0 - np.exp(-0.70 * distance * distance)
    sample_y = center_y + side_drop * bend
    return (
        eye_center + sample_x[:, None] * axis_x + sample_y[:, None] * axis_y
    ).astype(np.float32)


def track_upper_boundary_curve_sequence(
    landmarks_xy: np.ndarray,
    normalized_parameters: np.ndarray,
    correction: float = 0.15,
    reset_error_px: float = 18.0,
) -> np.ndarray:
    """Causally propagate a curve with rigid face motion, then correct toward anchors."""

    points = np.asarray(landmarks_xy, dtype=np.float32)
    parameters = np.asarray(normalized_parameters, dtype=np.float32)
    if points.ndim != 3 or len(points) != len(parameters):
        raise ValueError("Landmark and boundary-parameter sequences differ")
    raw = np.stack(
        [anchor_upper_boundary_curve(frame, value) for frame, value in zip(points, parameters)]
    )
    if not len(raw):
        return raw
    stable = points[:, np.asarray(TEMPORAL_STABILITY_INDICES), :2]
    correction = float(np.clip(correction, 0.0, 1.0))
    tracked = np.empty_like(raw)
    tracked[0] = raw[0]
    for frame in range(1, len(raw)):
        previous_stable = np.ascontiguousarray(stable[frame - 1], dtype=np.float32)
        current_stable = np.ascontiguousarray(stable[frame], dtype=np.float32)
        transform, _ = cv2.estimateAffinePartial2D(
            previous_stable, current_stable, method=cv2.LMEDS
        )
        if transform is None or not np.isfinite(transform).all():
            tracked[frame] = raw[frame]
            continue
        predicted = (
            tracked[frame - 1] @ transform[:, :2].T + transform[:, 2]
        ).astype(np.float32)
        error = float(np.median(np.linalg.norm(predicted - raw[frame], axis=1)))
        if error > float(reset_error_px):
            tracked[frame] = raw[frame]
        else:
            tracked[frame] = predicted + correction * (raw[frame] - predicted)
    return tracked


def _rotation(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    left, _, right = np.linalg.svd(value)
    result = left @ right
    if np.linalg.det(result) < 0:
        left[:, -1] *= -1.0
        result = left @ right
    return result.astype(np.float32)


def canonicalize_lower_face(
    landmarks_xyz: np.ndarray,
    face_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return head-frontalized lower-face points and rigid eye anchor."""

    points = np.asarray(landmarks_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] < 3:
        raise ValueError(f"Expected MediaPipe landmarks [>=468,3], got {points.shape}")
    rotation = _rotation(face_rotation)
    camera = points[:, :3].copy()
    camera[:, 1:] *= -1.0
    left_eye = camera[33]
    right_eye = camera[263]
    eye_origin = 0.5 * (left_eye + right_eye)
    eye_vector_local = (right_eye - left_eye) @ rotation
    scale = max(float(np.linalg.norm(eye_vector_local)), 1e-6)
    selected = camera[np.asarray(LOWER_FACE_INDICES)]
    canonical = ((selected - eye_origin) @ rotation) / scale
    return canonical.astype(np.float32), eye_origin.astype(np.float32), scale


def pack_lower_face_projection19(
    face_rotation: np.ndarray,
    eye_origin_camera: np.ndarray,
    interocular_scale: float,
    original_to_face_affine: np.ndarray,
) -> np.ndarray:
    value = np.concatenate(
        [
            _rotation(face_rotation).reshape(-1),
            np.asarray(eye_origin_camera, dtype=np.float32).reshape(3),
            np.asarray([max(float(interocular_scale), 1e-6)], dtype=np.float32),
            np.asarray(original_to_face_affine, dtype=np.float32).reshape(6),
        ]
    ).astype(np.float32)
    if value.shape != (19,):
        raise RuntimeError(f"Invalid lower-face projection shape {value.shape}")
    return value


def project_lower_face_to_image(
    canonical_points: np.ndarray,
    projection19: np.ndarray,
) -> np.ndarray:
    points = np.asarray(canonical_points, dtype=np.float32).reshape(LOWER_FACE_POINT_COUNT, 3)
    projection = np.asarray(projection19, dtype=np.float32).reshape(19)
    rotation = projection[:9].reshape(3, 3)
    origin = projection[9:12]
    scale = max(float(projection[12]), 1e-6)
    affine = projection[13:19].reshape(2, 3)
    camera = points * scale @ rotation.T + origin
    original_xy = np.stack([camera[:, 0], -camera[:, 1]], axis=1)
    return (original_xy @ affine[:, :2].T + affine[:, 2]).astype(np.float32)


def build_lower_face_mask(
    landmarks_xy: np.ndarray,
    image_size: int,
    nose_margin: int = 5,
    face_erode: int = 2,
    jaw_boundary: int = 5,
    upper_boundary_mode: str = "legacy_horizontal",
    upper_boundary_parameters: np.ndarray | None = None,
    upper_boundary_curve_xy: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build the face-below-nose union with explicit nose exclusion."""

    points = np.asarray(landmarks_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] < 2:
        raise ValueError(f"Expected image landmarks [>=468,2], got {points.shape}")
    shape = (int(image_size), int(image_size))
    face = np.zeros(shape, dtype=np.uint8)
    oval = np.round(points[np.asarray(FACE_OVAL_ORDER), :2]).astype(np.int32)
    cv2.fillPoly(face, [oval], 1, lineType=cv2.LINE_AA)
    if face_erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (face_erode * 2 + 1,) * 2)
        face = cv2.erode(face, k)

    nose = np.zeros(shape, dtype=np.uint8)
    nose_points = np.round(points[np.asarray(NOSE_HULL_INDICES), :2]).astype(np.int32)
    cv2.fillConvexPoly(nose, cv2.convexHull(nose_points), 1, lineType=cv2.LINE_AA)
    if nose_margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nose_margin * 2 + 1,) * 2)
        nose = cv2.dilate(nose, k)
    if upper_boundary_mode not in UPPER_BOUNDARY_MODES:
        raise ValueError(
            f"Unknown upper boundary mode {upper_boundary_mode!r}; "
            f"expected one of {UPPER_BOUNDARY_MODES}"
        )
    below = np.zeros(shape, dtype=np.uint8)
    upper_boundary = np.zeros(shape, dtype=np.uint8)
    if upper_boundary_mode == "legacy_horizontal":
        nose_bottom = int(np.clip(np.max(nose_points[:, 1]) + 2, 0, image_size))
        below[nose_bottom:] = 1
        upper_boundary[max(0, nose_bottom - 1) : min(image_size, nose_bottom + 2)] = 1
    else:
        # Build the boundary in a roll-neutral coordinate system. Its center is
        # tied to the detected nose base while its sides descend toward the
        # upper-lip/cheek region, avoiding the conspicuous horizontal cut.
        left_eye = points[33, :2]
        right_eye = points[263, :2]
        axis_x = right_eye - left_eye
        axis_x /= max(float(np.linalg.norm(axis_x)), 1e-6)
        axis_y = np.asarray([-axis_x[1], axis_x[0]], dtype=np.float32)
        eye_center = 0.5 * (left_eye + right_eye)
        if float(np.dot(points[152, :2] - eye_center, axis_y)) < 0:
            axis_y *= -1.0

        local_x = (points[:, :2] - eye_center) @ axis_x
        local_y = (points[:, :2] - eye_center) @ axis_y
        if upper_boundary_curve_xy is not None:
            curve = np.asarray(upper_boundary_curve_xy, dtype=np.float32)
            if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2:
                raise ValueError(f"Invalid upper-boundary curve {curve.shape}")
            sample_x = (curve - eye_center) @ axis_x
        elif upper_boundary_mode == "pose_curved_v1":
            oval_x = local_x[np.asarray(FACE_OVAL_ORDER)]
            nose_bottom_y = float(np.max(local_y[np.asarray(NOSE_HULL_INDICES)]))
            upper_lip_y = float(np.mean(local_y[np.asarray((0, 13, 37, 267))]))
            center_x = 0.5 * float(local_x[61] + local_x[291])
            half_width = max(0.5 * abs(float(local_x[291] - local_x[61])), 8.0)
            lip_gap = max(upper_lip_y - nose_bottom_y, 6.0)
            center_y = nose_bottom_y + 0.12 * lip_gap + 2.0
            side_drop = 0.78 * lip_gap
            sample_x = np.linspace(
                float(np.min(oval_x)) - 2.0, float(np.max(oval_x)) + 2.0, 161
            )
        else:
            curve = anchor_upper_boundary_curve(points, upper_boundary_parameters)
            sample_x = (curve - eye_center) @ axis_x
        if upper_boundary_curve_xy is None and upper_boundary_mode == "pose_curved_v1":
            distance = np.abs(sample_x - center_x) / max(float(half_width), 1e-6)
            bend = 1.0 - np.exp(-0.70 * distance * distance)
            sample_y = center_y + side_drop * bend
            curve = eye_center + sample_x[:, None] * axis_x + sample_y[:, None] * axis_y
        far_y = float(max(image_size * 2, np.max(local_y[np.asarray(FACE_OVAL_ORDER)]) + 32.0))
        bottom_right = eye_center + sample_x[-1] * axis_x + far_y * axis_y
        bottom_left = eye_center + sample_x[0] * axis_x + far_y * axis_y
        polygon = np.concatenate([curve, bottom_right[None], bottom_left[None]], axis=0)
        cv2.fillPoly(below, [np.rint(polygon).astype(np.int32)], 1, lineType=cv2.LINE_AA)
        cv2.polylines(
            upper_boundary,
            [np.rint(curve).astype(np.int32)],
            False,
            1,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
    skin_lower = face * below * (1 - nose)

    jaw_band = np.zeros(shape, dtype=np.uint8)
    jaw = np.round(points[np.asarray(JAW_INDICES), :2]).astype(np.int32)
    cv2.polylines(jaw_band, [jaw], False, 1, thickness=max(1, jaw_boundary * 2), lineType=cv2.LINE_AA)
    allowed = np.maximum(skin_lower, jaw_band) * (1 - nose)
    allowed = np.uint8(allowed > 0)
    return {
        "allowed": allowed,
        "skin_lower": np.uint8(skin_lower > 0),
        "nose_exclusion": np.uint8(nose > 0),
        "jaw_band": np.uint8(jaw_band > 0),
        "upper_boundary": np.uint8(upper_boundary > 0),
    }


def project_lower_face_torch(points: torch.Tensor, projection19: torch.Tensor) -> torch.Tensor:
    if points.shape[-2:] != (LOWER_FACE_POINT_COUNT, 3):
        raise ValueError(f"Invalid lower-face point shape {tuple(points.shape)}")
    rotation = projection19[..., :9].reshape(*points.shape[:-2], 3, 3)
    origin = projection19[..., 9:12].unsqueeze(-2)
    scale = projection19[..., 12:13].abs().clamp_min(1e-6).unsqueeze(-1)
    affine = projection19[..., 13:19].reshape(*points.shape[:-2], 2, 3)
    camera = torch.matmul(points * scale, rotation.transpose(-1, -2)) + origin
    original_xy = torch.stack([camera[..., 0], -camera[..., 1]], dim=-1)
    return torch.matmul(original_xy, affine[..., :2].transpose(-1, -2)) + affine[..., 2].unsqueeze(-2)


__all__ = [
    "FLESH_INDICES",
    "JAW_INDICES",
    "LIP_INDICES",
    "LOWER_FACE_GEOMETRY_DIM",
    "LOWER_FACE_INDICES",
    "LOWER_FACE_POINT_COUNT",
    "build_lower_face_mask",
    "canonicalize_lower_face",
    "pack_lower_face_projection19",
    "project_lower_face_to_image",
    "project_lower_face_torch",
    "UPPER_BOUNDARY_MODES",
    "UPPER_BOUNDARY_ANCHOR_INDICES",
    "TEMPORAL_STABILITY_INDICES",
    "upper_boundary_anchor_parameters",
    "upper_boundary_anchor_parameters_sequence",
    "causal_smooth_upper_boundary_parameters",
    "anchor_upper_boundary_curve",
    "track_upper_boundary_curve_sequence",
]
