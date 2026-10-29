from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from pose_aware_roi_generator.stable_syncnet_loss import StableSyncNetLoss

from .mask_cache_v5 import load_mask_chunk_v5
from .syncnet_utility_eval_types_v1 import (
    CandidateAdapter,
    EvalError,
    EvaluationItem,
    PairPayload,
)
from .syncnet_utility_assets_v1 import BoundedPairAssetCache


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FilePairMaterializer:
    candidate_adapter: CandidateAdapter = "v5_target_lip_support_mask_crop"
    _assets: BoundedPairAssetCache = field(default_factory=BoundedPairAssetCache, init=False, repr=False)

    def _rgb(self, row: EvaluationItem) -> torch.Tensor:
        try:
            crop_root = Path(row.crop_root).resolve(strict=True)
            frames_root = (crop_root / "crops").resolve(strict=True)
        except OSError as error:
            raise EvalError("unreadable crop root") from error
        if not frames_root.is_relative_to(crop_root):
            raise EvalError("crop path escapes clip root")
        frames: list[torch.Tensor] = []
        for frame_index in range(row.visual_start, row.visual_start + 16):
            frame_path = self._inside(frames_root, frames_root / f"{row.clip_id}_f{frame_index:06d}.jpg")
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None or image.shape != (256, 320, 3):
                raise EvalError(f"unreadable 256x320 crop: {frame_path}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb.transpose(2, 0, 1).copy()))
        return torch.stack(frames).to(torch.float32).div(255)

    def _masks(self, row: EvaluationItem) -> torch.Tensor:
        try:
            mask_root = Path(row.mask_root).resolve(strict=True)
        except OSError as error:
            raise EvalError("unreadable mask root") from error
        chunks: dict[int, dict[str, np.ndarray]] = {}
        masks: list[torch.Tensor] = []
        for frame_index in range(row.visual_start, row.visual_start + 16):
            chunk_index, local_index = divmod(frame_index, 64)
            if chunk_index not in chunks:
                try:
                    chunk_path = self._inside(
                        mask_root,
                        (mask_root / f"chunk_{chunk_index:06d}.npz").resolve(strict=True),
                    )
                except OSError as error:
                    raise EvalError("unreadable V5 mask chunk") from error
                try:
                    chunks[chunk_index] = load_mask_chunk_v5(chunk_path)
                except (OSError, ValueError) as error:
                    raise EvalError(f"unreadable V5 mask chunk: {chunk_path}") from error
            chunk = chunks[chunk_index]
            starts_at = int(chunk["target_frame_start"].item())
            source_indices = chunk["source_frame_indices"]
            if local_index >= len(source_indices):
                raise EvalError("mask chunk does not contain requested frame")
            if starts_at + local_index != frame_index or int(source_indices[local_index]) != frame_index:
                raise EvalError("mask frame binding is stale")
            value = np.asarray(chunk["target_lip_support"][local_index], dtype=np.float32)
            if value.shape != (256, 320) or not np.isfinite(value).all():
                raise EvalError("target lip support mask is malformed")
            masks.append(torch.from_numpy(value.copy())[None])
        return torch.stack(masks)

    def load(self, positive: EvaluationItem, negative: EvaluationItem) -> PairPayload:
        try:
            audio_path = Path(positive.audio_path).resolve(strict=True)
        except OSError as error:
            raise EvalError("unreadable SyncNet audio") from error
        sample_frames, sample_rate, mel = self._assets.load_audio(audio_path)
        conservative_columns = sample_frames // 200
        if sample_rate != 16_000 or positive.mel_columns != conservative_columns:
            raise EvalError("inventory mel columns do not match the 16 kHz WAV bound")
        if mel.shape != (80, conservative_columns + 1) or not bool(torch.isfinite(mel).all()):
            raise EvalError("audio mel shape differs from exact centered-STFT convention")
        if positive.mel_stop > positive.mel_columns:
            raise EvalError("anchor mel exceeds the conservative inventory bound")
        anchor_mel = mel[:, positive.mel_start:positive.mel_stop]
        if anchor_mel.shape != (80, 52):
            raise EvalError("anchor mel must be an unpadded 80x52 slice")
        positive_landmarks: torch.Tensor | None = None
        negative_landmarks: torch.Tensor | None = None
        crop_to_face: torch.Tensor | None = None
        if self.candidate_adapter == "mediapipe_landmark_aligned_crop_v1":
            try:
                crop_root = Path(positive.crop_root).resolve(strict=True)
            except OSError as error:
                raise EvalError("unreadable landmark alignment assets") from error
            positive_landmarks, negative_landmarks, crop_to_face = (
                self._assets.load_landmark_windows(
                    crop_root,
                    positive.visual_start,
                    negative.visual_start,
                )
            )

        zero_mask = torch.zeros(16, 1, 256, 320)
        return PairPayload(
            positive_rgb=self._rgb(positive),
            positive_mask=(
                self._masks(positive)
                if self.candidate_adapter == "v5_target_lip_support_mask_crop"
                else zero_mask
            ),
            negative_rgb=self._rgb(negative),
            negative_mask=(
                self._masks(negative)
                if self.candidate_adapter == "v5_target_lip_support_mask_crop"
                else zero_mask
            ),
            anchor_mel=anchor_mel.unsqueeze(0).unsqueeze(0),
            positive_face_landmarks_xy=positive_landmarks,
            negative_face_landmarks_xy=negative_landmarks,
            crop_to_face_affine=crop_to_face,
        )

    @staticmethod
    def _inside(root: Path, candidate: Path) -> Path:
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise EvalError("asset path escapes declared root")
        return resolved


@dataclass(frozen=True, slots=True)
class FrozenSyncNetEncoder:
    checkpoint_path: Path
    device: torch.device
    network: StableSyncNetLoss = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        network = StableSyncNetLoss(
            self.checkpoint_path,
            self.device,
            dtype=dtype,
            gradient_checkpointing=False,
        ).eval()
        if any(parameter.requires_grad for parameter in network.parameters()):
            raise EvalError("SyncNet parameters must be frozen")
        object.__setattr__(self, "network", network)

    def audio_embedding(self, mel: torch.Tensor) -> torch.Tensor:
        return self.network.audio_embedding(mel.to(self.device))

    def visual_embedding(self, visual: torch.Tensor) -> torch.Tensor:
        value = visual.to(device=self.device, dtype=self.network.model_dtype)
        embedding = self.network.syncnet.visual_encoder(value).reshape(value.shape[0], -1)
        return embedding.float()

    def state_identity(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.network.training).encode())
        for name, parameter in self.network.named_parameters():
            digest.update(name.encode())
            digest.update(str((parameter.data_ptr(), parameter._version, parameter.requires_grad)).encode())
        for name, buffer in self.network.named_buffers():
            digest.update(name.encode())
            digest.update(str((buffer.data_ptr(), buffer._version)).encode())
        return digest.hexdigest()

    def content_identity(self) -> str:
        digest = hashlib.sha256()
        for name, value in self.network.state_dict().items():
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode())
            digest.update(str((array.dtype, array.shape)).encode())
            digest.update(memoryview(array))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FixturePairMaterializer:
    candidate_adapter: CandidateAdapter = "v5_target_lip_support_mask_crop"

    def load(self, positive: EvaluationItem, negative: EvaluationItem) -> PairPayload:
        mask = torch.zeros(16, 1, 256, 320)
        mask[:, :, 152:184, 128:192] = 1
        landmarks: torch.Tensor | None = None
        affine: torch.Tensor | None = None
        if self.candidate_adapter == "mediapipe_landmark_aligned_crop_v1":
            from .syncnet_landmark_alignment_v1 import (
                MEDIAPIPE_IMAGE_LEFT_EYEBROW,
                MEDIAPIPE_IMAGE_RIGHT_EYEBROW,
                MEDIAPIPE_LIPS,
                MEDIAPIPE_NOSE,
            )

            landmarks = torch.full((16, 478, 2), 160.0)
            landmarks[:, MEDIAPIPE_IMAGE_LEFT_EYEBROW] = torch.tensor([110.0, 90.0])
            landmarks[:, MEDIAPIPE_IMAGE_RIGHT_EYEBROW] = torch.tensor([210.0, 90.0])
            landmarks[:, MEDIAPIPE_NOSE] = torch.tensor([160.0, 140.0])
            landmarks[:, MEDIAPIPE_LIPS] = torch.tensor([160.0, 190.0])
            affine = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        return PairPayload(
            positive_rgb=torch.ones(16, 3, 256, 320),
            positive_mask=mask,
            negative_rgb=torch.zeros(16, 3, 256, 320),
            negative_mask=mask,
            anchor_mel=torch.zeros(1, 1, 80, 52),
            positive_face_landmarks_xy=landmarks,
            negative_face_landmarks_xy=landmarks,
            crop_to_face_affine=affine,
        )


@dataclass(frozen=True, slots=True)
class FixtureEncoder:
    def audio_embedding(self, mel: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(mel.shape[0], 2048)
        result[:, 0] = 1
        return result

    def visual_embedding(self, visual: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(visual.shape[0], 2048)
        positive = visual.mean(dim=(1, 2, 3)) > 0
        result[positive, 0] = 1
        result[~positive, 1] = 1
        return result

    def state_identity(self) -> str:
        return "fixture-syncnet-state-v1"

    def content_identity(self) -> str:
        return "fixture-syncnet-content-v1"
