from __future__ import annotations

from dataclasses import dataclass
from typing import Final


SAMPLE_RATE: Final = 16_000
SHIFT_80MS_SAMPLES: Final = 1_280
STFT_HALF_WINDOW_SAMPLES: Final = 400
MEL_COLUMNS: Final = 52
MEL_HOP_SAMPLES: Final = 200
SUPPORT_SAMPLE_COUNT: Final = 11_000
NATIVE_LAGS_MS: Final = (0, -40, 40, -80, 80, -160, 160, -320, 320, -640, 640)
DIAGNOSTIC_SEARCH_LAGS_MS: Final = tuple(range(-320, 321, 40))
DIAGNOSTIC_SCORE_LAGS_MS: Final = tuple(range(-360, 361, 40))


@dataclass(frozen=True, slots=True)
class SupportWindow:
    start_sample_16k: int
    end_sample_16k: int

    @property
    def sample_count(self) -> int:
        return self.end_sample_16k - self.start_sample_16k


def plan_support(audio_start_sample_16k: int, lag_ms: int) -> SupportWindow:
    """Plan centered-STFT support; positive lag means delayed (earlier-source) audio."""
    lag_samples = lag_ms * SAMPLE_RATE // 1000
    start = audio_start_sample_16k - lag_samples - STFT_HALF_WINDOW_SAMPLES
    return SupportWindow(start, start + SUPPORT_SAMPLE_COUNT)


__all__ = [
    "DIAGNOSTIC_SCORE_LAGS_MS",
    "DIAGNOSTIC_SEARCH_LAGS_MS",
    "NATIVE_LAGS_MS",
    "SHIFT_80MS_SAMPLES",
    "SupportWindow",
    "plan_support",
]
