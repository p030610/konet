from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from pydantic import ValidationError

from .contracts import RawObservation


@dataclass(frozen=True, slots=True)
class LedgerError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


class RawLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows = self._recover()

    def _recover(self) -> tuple[RawObservation, ...]:
        if not self.path.exists():
            return ()
        try:
            data = self.path.read_bytes()
        except OSError as error:
            raise LedgerError("raw ledger is unreadable") from error
        rows: list[RawObservation] = []
        offset = 0
        for index, line in enumerate(data.splitlines(keepends=True)):
            next_offset = offset + len(line)
            try:
                rows.append(RawObservation.model_validate_json(line))
            except ValidationError as error:
                is_partial_tail = index == len(data.splitlines(keepends=True)) - 1 and not line.endswith(b"\n")
                if not is_partial_tail:
                    raise LedgerError("raw ledger is malformed") from error
                with self.path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            offset = next_offset
        keys = tuple(row.observation_key for row in rows)
        if len(keys) != len(set(keys)):
            raise LedgerError("raw ledger contains duplicate observation key")
        if data and rows and not data.endswith(b"\n"):
            with self.path.open("ab") as stream:
                stream.write(b"\n")
        return tuple(rows)

    def completed_keys(self, config_digest: str) -> set[str]:
        if any(row.config_digest != config_digest for row in self._rows):
            raise LedgerError("raw ledger config digest differs from frozen run")
        return {row.observation_key for row in self._rows}

    def append(self, row: RawObservation) -> None:
        if any(existing.observation_key == row.observation_key for existing in self._rows):
            raise LedgerError("duplicate observation key")
        payload = (row.model_dump_json() + "\n").encode()
        try:
            with self.path.open("ab") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise LedgerError("raw ledger append failed") from error
        self._rows = (*self._rows, row)


def write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as error:
        raise LedgerError(f"atomic output failed: {path.name}") from error


__all__ = ["LedgerError", "RawLedger", "write_atomic"]
