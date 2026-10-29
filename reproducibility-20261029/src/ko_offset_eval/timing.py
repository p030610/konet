from __future__ import annotations

from typing import Final

from koen_syncnet_eval.timing import SupportWindow, plan_support


CALIBRATION_LAGS_MS: Final = tuple(range(-640, 641, 40))
TARGET_DISTANCES_MS: Final = (40, 80, 120, 160, 240, 320)
MAXIMUM_HELDOUT_LAG_MS: Final = 960


def plan_offset_support(audio_start_sample_16k: int, lag_ms: int) -> SupportWindow:
    return plan_support(audio_start_sample_16k, lag_ms)


def heldout_lags(selected_offset_ms: int) -> tuple[int, ...]:
    values = {selected_offset_ms}
    for distance in TARGET_DISTANCES_MS:
        values.add(selected_offset_ms - distance)
        values.add(selected_offset_ms + distance)
    return tuple(sorted(values))


__all__ = [
    "CALIBRATION_LAGS_MS",
    "MAXIMUM_HELDOUT_LAG_MS",
    "TARGET_DISTANCES_MS",
    "heldout_lags",
    "plan_offset_support",
]
