from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Final

import cv2
import importlib.util
import numpy as np
import onnxruntime as ort
import soundfile
import torch
from pose_aware_roi_generator.dataset import _latentsync_melspectrogram

from .models import ManifestRow
from .windows import FRAME_COUNT, MEL_COLUMNS, SYNC_FPS, mel_bounds


PREPROCESS_NAME: Final = "latentsync_official_insightface106_cpu_provider_v1"
DEFAULT_NATIVE_ROOT: Final = Path(r"D:\LipSyncAnim_NativeCompare_20261007\runtime\LatentSync")
INSIGHTFACE_MODEL_ASSETS: Final = {
    "det_10g.onnx": 16_923_827,
    "2d106det.onnx": 5_030_888,
}
SETUPTOOLS_COMPAT_SITE: Final = Path(
    r"C:\Users\allen\AppData\Local\Programs\Python\Python310\Lib\site-packages"
)

if importlib.util.find_spec("pkg_resources") is None and SETUPTOOLS_COMPAT_SITE.is_dir():
    sys.path.append(str(SETUPTOOLS_COMPAT_SITE))


@dataclass(frozen=True, slots=True)
class MediaPreparationError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class MediaProbe:
    source_fps: float
    frame_count: int
    audio_duration_s: float


@dataclass(frozen=True, slots=True)
class MediaSource:
    row: ManifestRow
    probe: MediaProbe
    mel: torch.Tensor


@dataclass(frozen=True, slots=True)
class PreparedWindow:
    visual: torch.Tensor
    mel: torch.Tensor
    source_frame_indices: tuple[int, ...]
    selected_timestamps_s: tuple[float, ...]
    crop_xyxy: tuple[int, int, int, int]


class FaceAligner:
    """Official LatentSync InsightFace-106 alignment on CPUExecutionProvider."""

    def __init__(self, *, cpu_threads: int | None = None) -> None:
        if cpu_threads is not None and cpu_threads not in (1, 2):
            raise MediaPreparationError("worker_cpu_threads_invalid")
        native_root = Path(os.environ.get("LATENTSYNC_NATIVE_ROOT", DEFAULT_NATIVE_ROOT)).resolve()
        model_root = native_root / "checkpoints" / "auxiliary" / "models" / "buffalo_l"
        for name, expected_bytes in INSIGHTFACE_MODEL_ASSETS.items():
            asset = model_root / name
            if not asset.is_file():
                raise MediaPreparationError(f"insightface_model_asset_missing:{name}")
            if asset.stat().st_size != expected_bytes:
                raise MediaPreparationError(f"insightface_model_asset_size_mismatch:{name}")
        sys.path.insert(0, str(native_root))
        from insightface.app import FaceAnalysis
        from latentsync.utils.affine_transform import AlignRestore

        if cpu_threads is None:
            self._detector = FaceAnalysis(
                name="buffalo_l",
                root=str(native_root / "checkpoints/auxiliary"),
                allowed_modules=("detection", "landmark_2d_106"),
                providers=("CPUExecutionProvider",),
            )
        else:
            from insightface.model_zoo.landmark import Landmark
            from insightface.model_zoo.model_zoo import PickableInferenceSession
            from insightface.model_zoo.retinaface import RetinaFace

            ort.set_default_logger_severity(3)
            options = ort.SessionOptions()
            options.intra_op_num_threads = cpu_threads
            options.inter_op_num_threads = 1
            providers = ("CPUExecutionProvider",)
            detector_path = model_root / "det_10g.onnx"
            landmark_path = model_root / "2d106det.onnx"
            detector = RetinaFace(
                model_file=str(detector_path),
                session=PickableInferenceSession(
                    str(detector_path),
                    sess_options=options,
                    providers=providers,
                ),
            )
            landmark = Landmark(
                model_file=str(landmark_path),
                session=PickableInferenceSession(
                    str(landmark_path),
                    sess_options=options,
                    providers=providers,
                ),
            )
            analysis = FaceAnalysis.__new__(FaceAnalysis)
            analysis.models = {"detection": detector, "landmark_2d_106": landmark}
            analysis.det_model = detector
            self._detector = analysis
        self._detector.prepare(ctx_id=-1, det_size=(512, 512))
        self._restorer = AlignRestore(resolution=256, device="cpu", dtype=torch.float32)

    def align(self, rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        faces = self._detector.get(rgb)
        if not faces:
            raise MediaPreparationError("face landmarks unavailable")
        ranked = sorted(
            faces,
            key=lambda face: float(
                (face.bbox[2] - face.bbox[0])
                * (face.bbox[3] - face.bbox[1])
            ),
            reverse=True,
        )
        face = ranked[0]
        points = np.asarray(face.landmark_2d_106, dtype=np.float32)
        if points.shape != (106, 2) or not np.isfinite(points).all():
            raise MediaPreparationError("face landmarks are malformed")
        anchors = np.round(
            (
                points[[43, 48, 49, 51, 50]].mean(axis=0),
                points[101:106].mean(axis=0),
                points[[74, 77, 83, 86]].mean(axis=0),
            )
        )
        aligned, _ = self._restorer.align_warp_face(rgb.copy(), landmarks3=anchors, smooth=True)
        resized = cv2.resize(aligned, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        bbox = tuple(int(value) for value in face.bbox)
        return resized, bbox

    def reset(self) -> None:
        self._restorer.p_bias = None


def probe_media(row: ManifestRow) -> MediaProbe:
    capture = cv2.VideoCapture(str(row.video_path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frame_count < FRAME_COUNT:
        raise MediaPreparationError("video timing metadata is invalid")
    audio_path = row.audio_path or row.video_path
    try:
        audio_duration = float(soundfile.info(str(audio_path)).duration)
    except (RuntimeError, TypeError) as error:
        raise MediaPreparationError(f"audio probe failed: {audio_path}") from error
    return MediaProbe(fps, frame_count, audio_duration)


def source_frame_indices(anchor_s: float, source_fps: float) -> tuple[int, ...]:
    """Timestamp-resample source video onto Stable SyncNet's fixed 25 Hz grid."""
    return tuple(round((anchor_s + frame / SYNC_FPS) * source_fps) for frame in range(FRAME_COUNT))


def _decode_rgb(path: Path, indices: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise MediaPreparationError(f"video frame decode failed at {index}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return tuple(frames)


def _official_visual(aligned: tuple[np.ndarray, ...]) -> torch.Tensor:
    images = torch.stack(
        tuple(torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div(255) for frame in aligned)
    )
    normalized = images.mul(2).sub(1)
    return normalized.reshape(1, 48, 256, 256)[:, :, 128:, :]


def load_media_source(row: ManifestRow) -> MediaSource:
    probe = probe_media(row)
    mel = _latentsync_melspectrogram(row.audio_path or row.video_path)
    if mel.ndim != 2 or mel.shape[0] != 80 or not bool(torch.isfinite(mel).all()):
        raise MediaPreparationError("LatentSync mel is malformed")
    return MediaSource(row, probe, mel)


def prepare_window(source: MediaSource, anchor_s: float, aligner: FaceAligner) -> PreparedWindow:
    indices = source_frame_indices(anchor_s, source.probe.source_fps)
    if indices[-1] >= source.probe.frame_count:
        raise MediaPreparationError("visual window exceeds decoded video")
    frames = _decode_rgb(source.row.video_path, indices)
    aligner.reset()
    aligned_and_boxes = tuple(aligner.align(frame) for frame in frames)
    visual = _official_visual(tuple(value[0] for value in aligned_and_boxes))
    boxes = tuple(value[1] for value in aligned_and_boxes)
    crop = (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )
    start, stop = mel_bounds(anchor_s, 0)
    native = source.mel[:, start:stop]
    if visual.shape != (1, 48, 128, 256) or native.shape != (80, MEL_COLUMNS):
        raise MediaPreparationError("prepared Stable SyncNet tensor shape drift")
    timestamps = tuple(index / source.probe.source_fps for index in indices)
    return PreparedWindow(visual, native, indices, timestamps, crop)


__all__ = [
    "FaceAligner",
    "MediaPreparationError",
    "MediaProbe",
    "MediaSource",
    "PREPROCESS_NAME",
    "PreparedWindow",
    "load_media_source",
    "prepare_window",
    "probe_media",
    "source_frame_indices",
]
