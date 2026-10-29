from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


LATENTSYNC_MODEL_CONFIG = {
    "audio_encoder": {
        "in_channels": 1,
        "block_out_channels": [32, 64, 128, 256, 512, 1024, 2048],
        "downsample_factors": [[2, 1], 2, 2, 1, 2, 2, [2, 3]],
        "attn_blocks": [0, 0, 0, 1, 1, 0, 0],
        "dropout": 0.0,
    },
    "visual_encoder": {
        "in_channels": 48,
        "block_out_channels": [64, 128, 256, 256, 512, 1024, 2048, 2048],
        "downsample_factors": [[1, 2], 2, 2, 2, 2, 2, 2, 2],
        "attn_blocks": [0, 0, 0, 0, 1, 1, 0, 0],
        "dropout": 0.0,
    },
}


def _import_official_stable_syncnet():
    project_0325 = Path(__file__).resolve().parents[2]
    vendor_root = project_0325 / "vendor" / "LatentSync"
    if not vendor_root.exists():
        raise FileNotFoundError(
            f"LatentSync vendor checkout is missing: {vendor_root}. "
            "Clone https://github.com/bytedance/LatentSync there."
        )
    vendor_text = str(vendor_root)
    if vendor_text not in sys.path:
        sys.path.insert(0, vendor_text)
    from latentsync.models.stable_syncnet import StableSyncNet

    return StableSyncNet


def composite_generated_rois(
    generated_rois: torch.Tensor,
    original_faces: torch.Tensor,
    roi_boxes: torch.Tensor,
) -> torch.Tensor:
    """Differentiably place generated mouth ROIs into their 256px face crops."""
    if generated_rois.ndim != 5 or original_faces.ndim != 5:
        raise ValueError("generated_rois and original_faces must have shape [B, T, C, H, W]")
    if generated_rois.shape[:2] != original_faces.shape[:2]:
        raise ValueError("generated ROI and face clip dimensions do not match")
    if roi_boxes.shape[:2] != generated_rois.shape[:2] or roi_boxes.shape[-1] != 4:
        raise ValueError("roi_boxes must have shape [B, T, 4]")

    batch_frames: list[torch.Tensor] = []
    for batch_index in range(generated_rois.shape[0]):
        clip_frames: list[torch.Tensor] = []
        for frame_index in range(generated_rois.shape[1]):
            x1, y1, x2, y2 = [int(value) for value in roi_boxes[batch_index, frame_index].tolist()]
            face = original_faces[batch_index, frame_index].clone()
            patch = F.interpolate(
                generated_rois[batch_index, frame_index].unsqueeze(0),
                size=(y2 - y1, x2 - x1),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0]
            face[:, y1:y2, x1:x2] = patch
            clip_frames.append(face)
        batch_frames.append(torch.stack(clip_frames))
    return torch.stack(batch_frames)


class StableSyncNetLoss(nn.Module):
    """Frozen LatentSync Stable SyncNet used as a differentiable AV critic."""

    num_frames = 16
    resolution = 256

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if device.type != "cuda":
            dtype = torch.float32
        model_class = _import_official_stable_syncnet()
        self.syncnet = model_class(
            LATENTSYNC_MODEL_CONFIG,
            gradient_checkpointing=gradient_checkpointing,
        )
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.syncnet.load_state_dict(state_dict, strict=True)
        self.syncnet.to(device=device, dtype=dtype)
        self.syncnet.requires_grad_(False)
        self.syncnet.eval()
        self.model_dtype = dtype

    def train(self, mode: bool = True):
        super().train(False)
        self.syncnet.eval()
        return self

    def prepare_visual(self, face_clips: torch.Tensor) -> torch.Tensor:
        if face_clips.ndim != 5:
            raise ValueError("face_clips must have shape [B, T, 3, H, W]")
        if face_clips.shape[1] != self.num_frames or face_clips.shape[2] != 3:
            raise ValueError(f"Stable SyncNet requires [B, {self.num_frames}, 3, H, W]")
        batch, frames, channels, height, width = face_clips.shape
        flat = face_clips.reshape(batch * frames, channels, height, width)
        if (height, width) != (self.resolution, self.resolution):
            flat = F.interpolate(
                flat,
                size=(self.resolution, self.resolution),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        flat = flat.clamp(0.0, 1.0).mul(2.0).sub(1.0)
        packed = flat.reshape(batch, frames * channels, self.resolution, self.resolution)
        return packed[:, :, self.resolution // 2 :, :].to(dtype=self.model_dtype)

    def embeddings(
        self,
        face_clips: torch.Tensor,
        mel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vision_embedding = self.visual_embedding(face_clips)
        audio_embedding = self.audio_embedding(
            mel.to(device=face_clips.device)
        )
        return vision_embedding, audio_embedding

    def visual_embedding(self, face_clips: torch.Tensor) -> torch.Tensor:
        visual = self.prepare_visual(face_clips)
        embedding = self.syncnet.visual_encoder(visual).reshape(visual.shape[0], -1)
        return F.normalize(embedding.float(), p=2, dim=1)

    def audio_embedding(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.to(device=next(self.syncnet.parameters()).device, dtype=self.model_dtype)
        with torch.no_grad():
            embedding = self.syncnet.audio_encoder(mel).reshape(mel.shape[0], -1)
        return F.normalize(embedding.float(), p=2, dim=1)

    def forward(self, face_clips: torch.Tensor, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vision_embedding, audio_embedding = self.embeddings(face_clips, mel)
        similarity = F.cosine_similarity(vision_embedding, audio_embedding, dim=1)
        return (1.0 - similarity).mean(), similarity.mean()
