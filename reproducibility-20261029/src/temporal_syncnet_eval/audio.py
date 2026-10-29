from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import soundfile
import torch

from .contracts import CohortRow, ContractError
from .timing import MEL_COLUMNS, WaveformWindow, extract_waveform_window


MelFunction = Callable[[np.ndarray], np.ndarray]
SETUPTOOLS_COMPAT_SITE = Path(
    r"C:\Users\allen\AppData\Local\Programs\Python\Python310\Lib\site-packages"
)


def load_waveform(row: CohortRow) -> np.ndarray:
    try:
        waveform, sample_rate = soundfile.read(
            row.audio_path,
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError, TypeError):
        try:
            from decord import AudioReader, cpu

            reader = AudioReader(str(row.audio_path), ctx=cpu(0), sample_rate=16_000, mono=True)
            waveform = reader[:].asnumpy().reshape(-1, 1).astype(np.float32, copy=False)
            sample_rate = 16_000
        except (ImportError, RuntimeError) as error:
            raise ContractError("audio_decode_failed") from error
    if sample_rate != 16_000:
        raise ContractError("waveform_sample_rate_must_be_16000")
    if waveform.shape[1] != 1:
        raise ContractError("waveform_must_be_mono")
    mono = np.ascontiguousarray(waveform[:, 0], dtype=np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ContractError("waveform_is_empty_or_nonfinite")
    return mono


def load_official_mel_function(native_root: Path) -> MelFunction:
    if not (native_root / "latentsync" / "utils" / "audio.py").is_file():
        raise ContractError("LatentSync native runtime is unavailable")
    original = Path.cwd()
    if importlib.util.find_spec("pkg_resources") is None and SETUPTOOLS_COMPAT_SITE.is_dir():
        sys.path.append(str(SETUPTOOLS_COMPAT_SITE))
    sys.path.insert(0, str(native_root))
    try:
        os.chdir(native_root)
        module = importlib.import_module("latentsync.utils.audio")
    finally:
        os.chdir(original)
    function = module.melspectrogram
    return function


def prepare_mel(
    waveform: np.ndarray,
    plan: WaveformWindow,
    mel_function: MelFunction,
) -> torch.Tensor:
    exact_support = extract_waveform_window(waveform, plan)
    mel = np.asarray(mel_function(exact_support), dtype=np.float32)
    selected = mel[:, plan.mel_start:plan.mel_end]
    if selected.shape != (80, MEL_COLUMNS) or not np.isfinite(selected).all():
        raise ContractError("official_mel_shape_or_finiteness_invalid")
    return torch.from_numpy(np.ascontiguousarray(selected))[None]


__all__ = ["load_official_mel_function", "load_waveform", "prepare_mel"]
