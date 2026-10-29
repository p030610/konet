from __future__ import annotations

from enum import Enum, unique
from dataclasses import dataclass
from pathlib import Path
from typing import NewType

from pydantic import BaseModel, ConfigDict, Field, field_validator


ClipId = NewType("ClipId", str)
SpeakerId = NewType("SpeakerId", str)
UtteranceId = NewType("UtteranceId", str)


@dataclass(frozen=True, slots=True)
class ManifestContractError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@unique
class Language(str, Enum):
    KO = "ko"
    EN = "en"
    UNKNOWN = "unknown"


@unique
class Condition(str, Enum):
    NATIVE = "native"
    SHIFT = "shift"
    WRONG = "wrong"


class ManifestRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    clip_id: ClipId
    speaker_id: SpeakerId
    language: Language
    video_path: Path
    audio_path: Path | None
    utterance_id: UtteranceId
    start_s: float = Field(ge=0.0)
    duration_s: float = Field(gt=0.0)
    speaker_provenance: str
    language_provenance: str

    @field_validator("video_path", "audio_path")
    @classmethod
    def require_absolute_path(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ManifestContractError("media paths must be absolute")
        return value

    @classmethod
    def fixture(cls, clip_id: str, speaker_id: str, utterance_id: str) -> ManifestRow:
        return cls(
            clip_id=ClipId(clip_id),
            speaker_id=SpeakerId(speaker_id),
            language=Language.KO,
            video_path=Path.cwd() / f"{clip_id}.mp4",
            audio_path=None,
            utterance_id=UtteranceId(utterance_id),
            start_s=0.0,
            duration_s=3.0,
            speaker_provenance="fixture",
            language_provenance="fixture",
        )


class RawScoreRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    clip_id: ClipId
    speaker_id: SpeakerId
    utterance_id: UtteranceId
    language: Language
    window_id: str
    preprocess: str
    condition: Condition
    shift_ms: int | None
    score: float | None
    valid: bool
    reason: str | None
    negative_clip_id: ClipId | None
    negative_speaker_id: SpeakerId | None
    negative_utterance_id: UtteranceId | None


class ExpectedScoreRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    clip_id: ClipId
    speaker_id: SpeakerId
    utterance_id: UtteranceId
    window_id: str
    preprocess: str
    condition: Condition
    shift_ms: int | None
    negative_clip_id: ClipId | None
    negative_speaker_id: SpeakerId | None
    negative_utterance_id: UtteranceId | None


__all__ = [
    "ClipId",
    "Condition",
    "ExpectedScoreRow",
    "Language",
    "ManifestContractError",
    "ManifestRow",
    "RawScoreRow",
    "SpeakerId",
    "UtteranceId",
]
