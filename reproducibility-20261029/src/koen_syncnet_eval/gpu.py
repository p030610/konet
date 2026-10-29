from __future__ import annotations

from dataclasses import dataclass
from io import BufferedRandom
import msvcrt
import os
from pathlib import Path
from typing import Final

import torch


GIB: Final = 1024**3


@dataclass(frozen=True, slots=True)
class GpuGuardError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


class GpuLease:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.pid = os.getpid()
        self.acquired = False
        self._stream: BufferedRandom | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream: BufferedRandom | None = None
        try:
            try:
                stream = self.path.open("x+b")
                stream.write(b"0" * 32)
                stream.flush()
            except FileExistsError:
                stream = self.path.open("r+b")
            os.lseek(stream.fileno(), 0, os.SEEK_SET)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            os.lseek(stream.fileno(), 0, os.SEEK_SET)
            stream.write(str(self.pid).encode().ljust(32, b" "))
            stream.flush()
        except OSError as error:
            if stream is not None:
                stream.close()
            raise GpuGuardError("GPU ownership acquisition failed") from error
        self._stream = stream
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        if self._stream is None:
            raise GpuGuardError("GPU ownership stream is unavailable")
        self._stream.flush()
        os.lseek(self._stream.fileno(), 0, os.SEEK_SET)
        msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
        self._stream.close()
        self._stream = None
        self.acquired = False


def check_cuda_headroom() -> None:
    if not torch.cuda.is_available():
        raise GpuGuardError("CUDA is unavailable")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < GIB:
        raise GpuGuardError("CUDA free memory is below 1 GiB headroom")


__all__ = ["GpuGuardError", "GpuLease", "check_cuda_headroom"]
