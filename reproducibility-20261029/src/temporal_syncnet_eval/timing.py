from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .contracts import ContractError, Role


SAMPLE_RATE: Final = 16_000
VIDEO_FPS: Final = 25
SAMPLES_PER_FRAME: Final = 640
MEL_HOP_SAMPLES: Final = 200
STFT_WINDOW_SAMPLES: Final = 800
STFT_HALF_WINDOW_SAMPLES: Final = STFT_WINDOW_SAMPLES // 2
MEL_COLUMNS: Final = 52
MEL_CONTEXT_COLUMNS: Final = STFT_HALF_WINDOW_SAMPLES // MEL_HOP_SAMPLES
CORE_LAGS_MS: Final = tuple(range(-320, 321, 40))
FAR_LAGS_MS: Final = (-1280, -640, 640, 1280)
HELDOUT_SUPPORT_LAGS_MS: Final = (-400, -360, 360, 400)
ALL_LAGS_MS: Final = CORE_LAGS_MS + FAR_LAGS_MS + HELDOUT_SUPPORT_LAGS_MS


@dataclass(frozen=True, slots=True)
class WaveformWindow:
    frame_start: int
    frame_phase_mod5: int
    lag_ms: int
    first_center_sample: int
    audio_start_sample: int
    audio_end_sample: int
    mel_start: int
    mel_end: int

    @property
    def supported_sample_count(self) -> int:
        return self.audio_end_sample - self.audio_start_sample


def lag_role(lag_ms: int) -> Role:
    if lag_ms in CORE_LAGS_MS:
        return "core"
    if lag_ms in FAR_LAGS_MS:
        return "far"
    if lag_ms in HELDOUT_SUPPORT_LAGS_MS:
        return "heldout_support"
    raise ContractError("lag is outside the frozen evaluation grid")


def plan_waveform_window(frame_start: int, lag_ms: int, audio_origin_sample: int) -> WaveformWindow:
    if lag_ms not in ALL_LAGS_MS:
        raise ContractError("lag is outside the frozen evaluation grid")
    lag_samples = lag_ms * SAMPLE_RATE // 1000
    first_center = audio_origin_sample + frame_start * SAMPLES_PER_FRAME - lag_samples
    audio_start = first_center - STFT_HALF_WINDOW_SAMPLES
    audio_end = first_center + (MEL_COLUMNS - 1) * MEL_HOP_SAMPLES + STFT_HALF_WINDOW_SAMPLES
    return WaveformWindow(
        frame_start=frame_start,
        frame_phase_mod5=frame_start % 5,
        lag_ms=lag_ms,
        first_center_sample=first_center,
        audio_start_sample=audio_start,
        audio_end_sample=audio_end,
        mel_start=MEL_CONTEXT_COLUMNS,
        mel_end=MEL_CONTEXT_COLUMNS + MEL_COLUMNS,
    )


def extract_waveform_window(waveform: np.ndarray, plan: WaveformWindow) -> np.ndarray:
    if waveform.ndim != 1:
        raise ContractError("waveform_must_be_mono")
    if plan.audio_start_sample < 0 or plan.audio_end_sample > waveform.shape[0]:
        raise ContractError("waveform_support_out_of_bounds")
    result = waveform[plan.audio_start_sample:plan.audio_end_sample]
    if result.shape != (plan.supported_sample_count,):
        raise ContractError("waveform_support_out_of_bounds")
    return result


__all__ = [
    "ALL_LAGS_MS",
    "CORE_LAGS_MS",
    "FAR_LAGS_MS",
    "HELDOUT_SUPPORT_LAGS_MS",
    "MEL_COLUMNS",
    "STFT_HALF_WINDOW_SAMPLES",
    "WaveformWindow",
    "extract_waveform_window",
    "lag_role",
    "plan_waveform_window",
]
