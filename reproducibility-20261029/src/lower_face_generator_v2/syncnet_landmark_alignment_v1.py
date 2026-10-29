from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn.functional as functional

MEDIAPIPE_IMAGE_LEFT_EYEBROW: Final = (46, 52, 53, 55, 63, 65, 66, 70, 105, 107)
MEDIAPIPE_IMAGE_RIGHT_EYEBROW: Final = (276, 282, 283, 285, 293, 295, 296, 300, 334, 336)
MEDIAPIPE_NOSE: Final = (
    1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98,
    115, 168, 195, 197, 220, 275, 278, 294, 326, 327, 344, 440,
)
MEDIAPIPE_LIPS: Final = (
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91,
    95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311,
    312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
)
OFFICIAL_FACE_TEMPLATE: Final = torch.tensor(
    [[17.0, 20.0], [58.0, 20.0], [37.5, 40.0]], dtype=torch.float32
) * 2.8
INTERMEDIATE_HEIGHT: Final = 280
INTERMEDIATE_WIDTH: Final = 210
OUTPUT_SIZE: Final = 256
NEUTRAL_RGB: Final = 127.0 / 255.0
ANCHOR_ADAPTATION: Final = (
    "official 106-point image-left eyebrow mean -> MediaPipe image-left eyebrow topology centroid",
    "official 106-point image-right eyebrow mean -> MediaPipe image-right eyebrow topology centroid",
    "official 106-point four-nose mean -> MediaPipe full nose topology centroid",
)
RESAMPLING_DEVIATION: Final = (
    "official cv2 INTER_LANCZOS4 uint8 210x280-to-256x256 resize is replaced by "
    "float differentiable torch bicubic antialiased resize and [0,1] color clamp"
)


class LandmarkAlignmentError(RuntimeError):
    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class AlignmentCoverage:
    padding_fraction: torch.Tensor
    lip_coverage_fraction: torch.Tensor


@dataclass(frozen=True, slots=True)
class AlignmentWindow:
    face_to_intermediate: torch.Tensor
    crop_to_intermediate: torch.Tensor
    normalized_nose_bias: torch.Tensor
    coverage: AlignmentCoverage


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    syncnet_rgb: torch.Tensor
    alignment: AlignmentWindow
    coverage: AlignmentCoverage


def _require_shape(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if value.shape != shape:
        raise LandmarkAlignmentError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise LandmarkAlignmentError(f"{name} must be finite")


def _homogeneous(matrix: torch.Tensor) -> torch.Tensor:
    bottom = matrix.new_tensor([[0.0, 0.0, 1.0]])
    return torch.cat((matrix, bottom), dim=0)


def _transform_points(matrix: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)
    return torch.cat((points, ones), dim=-1) @ matrix.transpose(-1, -2)


def similarity_transform_from_points(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the orientation-preserving similarity mapping source pixels to target pixels."""
    _require_shape(source, (3, 2), "source anchors")
    _require_shape(target, (3, 2), "target anchors")
    source64 = source.to(dtype=torch.float64)
    target64 = target.to(device=source.device, dtype=torch.float64)
    source_center = source64.mean(dim=0)
    target_center = target64.mean(dim=0)
    source_centered = source64 - source_center
    target_centered = target64 - target_center
    source_scale = torch.std(source_centered)
    target_scale = torch.std(target_centered)
    if float(source_scale) <= 1e-6 or float(target_scale) <= 1e-6:
        raise LandmarkAlignmentError("anchor geometry is degenerate")
    source_normalized = source_centered / source_scale
    target_normalized = target_centered / target_scale
    covariance = source_normalized.T @ target_normalized
    left, _, right_h = torch.linalg.svd(covariance)
    right = right_h.T
    rotation = right @ left.T
    if float(torch.det(rotation)) < 0.0:
        right = right.clone()
        right[:, -1] = -right[:, -1]
        rotation = right @ left.T
    scale = target_scale / source_scale
    translation = target_center - scale * (rotation @ source_center)
    return torch.cat((scale * rotation, translation[:, None]), dim=1).to(dtype=source.dtype)


def _anchor_points(landmarks: torch.Tensor) -> torch.Tensor:
    left = landmarks[:, MEDIAPIPE_IMAGE_LEFT_EYEBROW].mean(dim=1)
    right = landmarks[:, MEDIAPIPE_IMAGE_RIGHT_EYEBROW].mean(dim=1)
    nose = landmarks[:, MEDIAPIPE_NOSE].mean(dim=1)
    return torch.round(torch.stack((left, right, nose), dim=1))


def _coverage(
    landmarks: torch.Tensor,
    crop_to_face: torch.Tensor,
    crop_to_intermediate: torch.Tensor,
) -> AlignmentCoverage:
    device = crop_to_intermediate.device
    dtype = crop_to_intermediate.dtype
    final_x = torch.arange(OUTPUT_SIZE, device=device, dtype=dtype)
    final_y = torch.arange(OUTPUT_SIZE // 2, OUTPUT_SIZE, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(final_y, final_x, indexing="ij")
    intermediate = torch.stack(
        (
            (grid_x + 0.5) * INTERMEDIATE_WIDTH / OUTPUT_SIZE - 0.5,
            (grid_y + 0.5) * INTERMEDIATE_HEIGHT / OUTPUT_SIZE - 0.5,
        ),
        dim=-1,
    ).reshape(-1, 2)
    inverse = torch.linalg.inv(torch.vmap(_homogeneous)(crop_to_intermediate))[:, :2]
    source = torch.vmap(_transform_points, in_dims=(0, None))(inverse, intermediate)
    supported = (
        (source[..., 0] >= 0.0)
        & (source[..., 0] <= 319.0)
        & (source[..., 1] >= 0.0)
        & (source[..., 1] <= 255.0)
    )
    face_to_crop = torch.linalg.inv(_homogeneous(crop_to_face))[:2]
    lips_face = landmarks[:, MEDIAPIPE_LIPS]
    lips_crop = torch.vmap(_transform_points, in_dims=(None, 0))(face_to_crop, lips_face)
    lips_intermediate = torch.vmap(_transform_points)(crop_to_intermediate, lips_crop)
    lips_final_x = (lips_intermediate[..., 0] + 0.5) * OUTPUT_SIZE / INTERMEDIATE_WIDTH - 0.5
    lips_final_y = (lips_intermediate[..., 1] + 0.5) * OUTPUT_SIZE / INTERMEDIATE_HEIGHT - 0.5
    lips_available = (
        (lips_crop[..., 0] >= 0.0)
        & (lips_crop[..., 0] <= 319.0)
        & (lips_crop[..., 1] >= 0.0)
        & (lips_crop[..., 1] <= 255.0)
        & (lips_final_x >= 0.0)
        & (lips_final_x <= 255.0)
        & (lips_final_y >= 128.0)
        & (lips_final_y <= 255.0)
    )
    return AlignmentCoverage(
        padding_fraction=1.0 - supported.float().mean(dim=1),
        lip_coverage_fraction=lips_available.float().mean(dim=1),
    )


def estimate_mediapipe_alignment(
    face_landmarks_xy: torch.Tensor,
    crop_to_face_affine: torch.Tensor,
) -> AlignmentWindow:
    _require_shape(face_landmarks_xy, (16, 478, 2), "face_landmarks_xy")
    _require_shape(crop_to_face_affine, (2, 3), "crop_to_face_affine")
    anchors = _anchor_points(face_landmarks_xy.float())
    target = OFFICIAL_FACE_TEMPLATE.to(device=anchors.device)
    face_transforms: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    previous_bias: torch.Tensor | None = None
    for frame_anchors in anchors:
        transform = similarity_transform_from_points(frame_anchors, target)
        source_centered = frame_anchors - frame_anchors.mean(dim=0)
        target_centered = target - target.mean(dim=0)
        current_bias = (
            target_centered / torch.std(target_centered)
            - source_centered / torch.std(source_centered)
        )[2]
        smoothed_bias = current_bias if previous_bias is None else previous_bias * 0.2 + current_bias * 0.8
        transform = transform.clone()
        transform[:, 2] += smoothed_bias
        face_transforms.append(transform)
        biases.append(smoothed_bias)
        previous_bias = smoothed_bias
    face_to_intermediate = torch.stack(face_transforms)
    crop_to_intermediate = (
        torch.vmap(_homogeneous)(face_to_intermediate)
        @ _homogeneous(crop_to_face_affine.float())
    )[:, :2]
    coverage = _coverage(face_landmarks_xy.float(), crop_to_face_affine.float(), crop_to_intermediate)
    return AlignmentWindow(
        face_to_intermediate=face_to_intermediate,
        crop_to_intermediate=crop_to_intermediate,
        normalized_nose_bias=torch.stack(biases),
        coverage=coverage,
    )


def apply_syncnet_alignment(crop_rgb: torch.Tensor, alignment: AlignmentWindow) -> torch.Tensor:
    _require_shape(crop_rgb, (16, 3, 256, 320), "crop_rgb")
    if not crop_rgb.is_floating_point():
        raise LandmarkAlignmentError("crop_rgb must be floating point")
    if bool((crop_rgb < 0.0).any()) or bool((crop_rgb > 1.0).any()):
        raise LandmarkAlignmentError("crop_rgb must be in [0,1]")
    matrices = alignment.crop_to_intermediate.to(device=crop_rgb.device, dtype=torch.float32)
    inverse = torch.linalg.inv(torch.vmap(_homogeneous)(matrices))[:, :2]
    y, x = torch.meshgrid(
        torch.arange(INTERMEDIATE_HEIGHT, device=crop_rgb.device, dtype=torch.float32),
        torch.arange(INTERMEDIATE_WIDTH, device=crop_rgb.device, dtype=torch.float32),
        indexing="ij",
    )
    destination = torch.stack((x, y), dim=-1).reshape(-1, 2)
    source = torch.vmap(_transform_points, in_dims=(0, None))(inverse, destination)
    grid = torch.stack(
        (
            2.0 * (source[..., 0] + 0.5) / 320.0 - 1.0,
            2.0 * (source[..., 1] + 0.5) / 256.0 - 1.0,
        ),
        dim=-1,
    ).reshape(16, INTERMEDIATE_HEIGHT, INTERMEDIATE_WIDTH, 2)
    working = crop_rgb.float()
    warped = functional.grid_sample(
        working - NEUTRAL_RGB,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ) + NEUTRAL_RGB
    square = functional.interpolate(
        warped,
        size=(OUTPUT_SIZE, OUTPUT_SIZE),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)
    normalized = square.mul(2.0).sub(1.0)
    return normalized.reshape(1, 48, OUTPUT_SIZE, OUTPUT_SIZE)[:, :, OUTPUT_SIZE // 2 :]


def align_mediapipe_syncnet_window(
    crop_rgb: torch.Tensor,
    face_landmarks_xy: torch.Tensor,
    crop_to_face_affine: torch.Tensor,
) -> AlignmentResult:
    alignment = estimate_mediapipe_alignment(face_landmarks_xy, crop_to_face_affine)
    syncnet_rgb = apply_syncnet_alignment(crop_rgb, alignment)
    return AlignmentResult(syncnet_rgb=syncnet_rgb, alignment=alignment, coverage=alignment.coverage)
