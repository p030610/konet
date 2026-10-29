from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as functional

from koen_syncnet_eval.contracts import CohortWindow, ExpectedObservation, RawObservation
from koen_syncnet_eval.policy import VisualPolicy
from koen_syncnet_eval.scoring import ScoringError, _load_waveform_16k, _raw

from .adapters import KoNetAdapterConfig, KoNetEncoderAdapter


class AdapterScorer:
    def __init__(
        self,
        config: KoNetAdapterConfig,
        native_root: Path,
    ) -> None:
        from koen_syncnet_eval.visual import BoundedVisualPreparer
        from temporal_syncnet_eval.audio import load_official_mel_function

        self.encoder = KoNetEncoderAdapter(config)
        self._mel_function = load_official_mel_function(native_root)
        self._visual_preparer = BoundedVisualPreparer(VisualPolicy.NORMALIZE_1280X720)
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

        if self._waveform_path != row.audio_path:
            self._waveform = _load_waveform_16k(row.audio_path)
            self._waveform_path = row.audio_path
        if self._waveform is None:
            raise ScoringError("audio cache is unavailable")
        waveform = self._waveform
        visual = self._visual_preparer.prepare(row, row.window_start_pts_seconds).tensor
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
        if valid_items:
            with torch.inference_mode():
                visual_embedding = functional.normalize(self.encoder.visual_embedding(visual), dim=1)
                audio = torch.stack(mels)
                for start in range(0, len(valid_items), batch_size):
                    items = valid_items[start : start + batch_size]
                    audio_embedding = functional.normalize(
                        self.encoder.audio_embedding(audio[start : start + batch_size]), dim=1
                    )
                    values = (audio_embedding * visual_embedding.expand(len(items), -1)).sum(dim=1)
                    for item, value in zip(items, values.detach().cpu().tolist(), strict=True):
                        score = float(value)
                        terminal[item.observation_key] = (
                            _raw(item, "valid", score, None)
                            if math.isfinite(score)
                            else _raw(item, "nonfinite", None, "model score is nonfinite")
                        )
        return tuple(terminal[item.observation_key] for item in expected)


__all__ = ["AdapterScorer"]
