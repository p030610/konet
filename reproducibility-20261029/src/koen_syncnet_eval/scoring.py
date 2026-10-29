from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as functional

from .contracts import CohortWindow, ContractError, ExpectedObservation, RawObservation
from .policy import VisualPolicy


Backend = Literal["fixture", "frozen"]


@dataclass(frozen=True, slots=True)
class ScoringError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def _raw(
    expected: ExpectedObservation,
    status: Literal["valid", "missing_support", "nonfinite", "input_error"],
    score: float | None,
    reason: str | None,
) -> RawObservation:
    return RawObservation(**expected.model_dump(), score_status=status, score=score, reason=reason)


def audio_sample_count_16k(path: Path) -> int:
    return int(_load_waveform_16k(path).shape[0])


def score_fixture(
    expected: tuple[ExpectedObservation, ...],
    audio_samples_16k: int,
) -> tuple[RawObservation, ...]:
    rows: list[RawObservation] = []
    for item in expected:
        if item.support_start_sample_16k < 0 or item.support_end_sample_16k > audio_samples_16k:
            rows.append(_raw(item, "missing_support", None, "waveform_support_out_of_bounds"))
            continue
        score = 1.0 / (1.0 + abs(item.lag_ms) / 80.0)
        rows.append(_raw(item, "valid", score, None))
    return tuple(rows)


def _load_waveform_16k(path: Path) -> np.ndarray:
    from temporal_syncnet_eval.audio import load_waveform

    @dataclass(frozen=True, slots=True)
    class AudioReference:
        audio_path: Path

    try:
        return load_waveform(AudioReference(path))
    except RuntimeError as error:
        raise ScoringError("audio decode failed") from error


class FrozenScorer:
    def __init__(
        self,
        checkpoint: Path,
        native_root: Path,
        device: str,
        visual_policy: VisualPolicy = VisualPolicy.NATIVE,
    ) -> None:
        from lower_face_generator_v2.syncnet_utility_runtime_v1 import FrozenSyncNetEncoder
        from temporal_syncnet_eval.audio import load_official_mel_function
        from .visual import BoundedVisualPreparer

        self._encoder = FrozenSyncNetEncoder(checkpoint, torch.device(device))
        self._mel_function = load_official_mel_function(native_root)
        self._visual_preparer = BoundedVisualPreparer(visual_policy)
        self._waveform_path: Path | None = None
        self._waveform: np.ndarray | None = None

    def score(
        self,
        row: CohortWindow,
        expected: tuple[ExpectedObservation, ...],
        batch_size: int,
    ) -> tuple[RawObservation, ...]:
        from temporal_syncnet_eval.audio import prepare_mel
        from temporal_syncnet_eval.timing import WaveformWindow

        started = time.perf_counter()
        audio_started = started
        try:
            if self._waveform_path != row.audio_path:
                self._waveform = _load_waveform_16k(row.audio_path)
                self._waveform_path = row.audio_path
            if self._waveform is None:
                raise ScoringError("audio cache is unavailable")
            waveform = self._waveform
            audio_finished = time.perf_counter()
            visual = self._visual_preparer.prepare(row, row.window_start_pts_seconds).tensor
            visual_finished = time.perf_counter()
        except (ContractError, RuntimeError, ScoringError) as error:
            return tuple(_raw(item, "input_error", None, str(error)) for item in expected)
        valid_items: list[ExpectedObservation] = []
        mels: list[torch.Tensor] = []
        terminal: dict[str, RawObservation] = {}
        for item in expected:
            if item.support_start_sample_16k < 0 or item.support_end_sample_16k > waveform.shape[0]:
                terminal[item.observation_key] = _raw(
                    item, "missing_support", None, "waveform_support_out_of_bounds"
                )
                continue
            support = WaveformWindow(
                frame_start=row.frame_start,
                frame_phase_mod5=row.frame_start % 5,
                lag_ms=item.lag_ms,
                first_center_sample=item.support_start_sample_16k + 400,
                audio_start_sample=item.support_start_sample_16k,
                audio_end_sample=item.support_end_sample_16k,
                mel_start=2,
                mel_end=54,
            )
            try:
                mels.append(prepare_mel(waveform, support, self._mel_function))
                valid_items.append(item)
            except RuntimeError as error:
                terminal[item.observation_key] = _raw(item, "input_error", None, str(error))
        mel_finished = time.perf_counter()
        if valid_items:
            with torch.inference_mode():
                visual_embedding = functional.normalize(self._encoder.visual_embedding(visual), dim=1)
                audio = torch.stack(mels)
                for start in range(0, len(valid_items), batch_size):
                    items = valid_items[start : start + batch_size]
                    audio_embedding = functional.normalize(
                        self._encoder.audio_embedding(audio[start : start + batch_size]), dim=1
                    )
                    scores = (audio_embedding * visual_embedding.expand(len(items), -1)).sum(dim=1)
                    for item, value in zip(items, scores.detach().cpu().tolist(), strict=True):
                        score = float(value)
                        terminal[item.observation_key] = (
                            _raw(item, "valid", score, None)
                            if math.isfinite(score)
                            else _raw(item, "nonfinite", None, "model score is nonfinite")
                        )
        finished = time.perf_counter()
        print(
            json.dumps(
                {
                    "phase": "window_timing",
                    "window_id": row.window_id,
                    "audio_load_seconds": audio_finished - audio_started,
                    "visual_prepare_seconds": visual_finished - audio_finished,
                    "mel_prepare_seconds": mel_finished - visual_finished,
                    "inference_seconds": finished - mel_finished,
                    "total_seconds": finished - started,
                    "valid_lags": len(valid_items),
                    "effective_audio_batch": min(batch_size, len(valid_items)),
                }
            ),
            flush=True,
        )
        return tuple(terminal[item.observation_key] for item in expected)


__all__ = ["Backend", "FrozenScorer", "ScoringError", "audio_sample_count_16k", "score_fixture"]
