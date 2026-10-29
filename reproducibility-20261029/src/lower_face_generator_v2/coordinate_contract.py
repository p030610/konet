from __future__ import annotations

from dataclasses import asdict, dataclass


COORDINATE_CONTRACT_VERSION = "lower_face_frame_space_v1"


class CoordinateContractError(ValueError):
    pass


@dataclass(frozen=True)
class FrameSpaceBinding:
    clip_id: str
    crop_contract_id: str
    context_frame_index: int
    mask_frame_index: int
    geometry_frame_index: int
    projection_frame_index: int
    mode: str = "same_frame_reconstruction"
    version: str = COORDINATE_CONTRACT_VERSION

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


def validate_frame_space_binding(binding: FrameSpaceBinding) -> None:
    if binding.version != COORDINATE_CONTRACT_VERSION:
        raise CoordinateContractError(f"Unsupported coordinate contract {binding.version!r}")
    if not binding.clip_id or not binding.crop_contract_id:
        raise CoordinateContractError("Frame-space binding requires clip and crop contract identifiers")
    indices = {
        "context": int(binding.context_frame_index),
        "mask": int(binding.mask_frame_index),
        "geometry": int(binding.geometry_frame_index),
        "projection": int(binding.projection_frame_index),
    }
    if any(value < 0 for value in indices.values()):
        raise CoordinateContractError(f"Negative frame index in coordinate binding: {indices}")
    if binding.mode == "same_frame_reconstruction":
        if len(set(indices.values())) != 1:
            raise CoordinateContractError(
                "Same-frame reconstruction requires context, mask, geometry, and projection "
                f"from one frame; received {indices}"
            )
    elif binding.mode == "source_space_cross_drive":
        if indices["context"] != indices["mask"] or indices["context"] != indices["projection"]:
            raise CoordinateContractError(
                "Cross-driving requires context RGB, edit mask, and projection in source-frame "
                f"coordinates; received {indices}"
            )
    else:
        raise CoordinateContractError(f"Unknown frame-space mode {binding.mode!r}")


__all__ = [
    "COORDINATE_CONTRACT_VERSION", "CoordinateContractError", "FrameSpaceBinding",
    "validate_frame_space_binding",
]
