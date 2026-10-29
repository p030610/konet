from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Final


GIB: Final = 1024**3
MIB: Final = 1024**2
MINIMUM_FREE_BYTES: Final = 25 * GIB
MAXIMUM_CACHE_BYTES: Final = 512 * MIB
MAXIMUM_OUTPUT_BYTES: Final = 512 * MIB


@dataclass(frozen=True, slots=True)
class ResourceError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError as error:
        raise ResourceError(f"resource size unavailable: {path}") from error


def check_resources(output_root: Path, cache_root: Path) -> None:
    anchor = output_root.anchor or str(output_root.resolve().anchor)
    try:
        free = shutil.disk_usage(anchor).free
    except OSError as error:
        raise ResourceError("output disk usage unavailable") from error
    if free < MINIMUM_FREE_BYTES:
        raise ResourceError("output disk free space is below 25 GiB")
    if tree_size(cache_root) > MAXIMUM_CACHE_BYTES:
        raise ResourceError("rolling task cache exceeds 512 MiB")
    if tree_size(output_root) > MAXIMUM_OUTPUT_BYTES:
        raise ResourceError("task output exceeds 512 MiB")


__all__ = ["ResourceError", "check_resources", "tree_size"]
