from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from koen_syncnet_eval.contracts import CohortWindow
from koen_syncnet_eval.ledger import write_atomic

from .contracts import OffsetDecision, OffsetExpectedObservation


@dataclass(frozen=True, slots=True)
class RunnerIOError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def load_expected(path: Path) -> tuple[OffsetExpectedObservation, ...]:
    try:
        return tuple(
            OffsetExpectedObservation.model_validate_json(line)
            for line in path.read_bytes().splitlines()
            if line.strip()
        )
    except (OSError, ValidationError) as error:
        raise RunnerIOError("expected observation manifest is unreadable or malformed") from error


def load_windows(path: Path) -> tuple[CohortWindow, ...]:
    try:
        return tuple(
            CohortWindow.model_validate_json(line)
            for line in path.read_bytes().splitlines()
            if line.strip()
        )
    except (OSError, ValidationError) as error:
        raise RunnerIOError("selected cohort is unreadable or malformed") from error


def write_offsets(root: Path, decisions: tuple[OffsetDecision, ...]) -> None:
    write_atomic(
        root / "offsets.jsonl",
        "".join(row.model_dump_json() + "\n" for row in decisions),
    )


def load_offsets(path: Path) -> tuple[OffsetDecision, ...]:
    if not path.exists():
        return ()
    try:
        return tuple(
            OffsetDecision.model_validate_json(line)
            for line in path.read_bytes().splitlines()
            if line.strip()
        )
    except (OSError, ValidationError) as error:
        raise RunnerIOError("offset ledger is unreadable or malformed") from error


__all__ = ["RunnerIOError", "load_expected", "load_offsets", "load_windows", "write_offsets"]
