from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from pydantic import ValidationError

from koen_syncnet_eval.contracts import RawObservation

from .contracts import OffsetExpectedObservation, OffsetRawObservation


@dataclass(frozen=True, slots=True)
class ReuseError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


class ObservationLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows = self._recover()
        self._by_key = {row.observation_key: row for row in self._rows}

    def _recover(self) -> tuple[OffsetRawObservation, ...]:
        if not self.path.exists():
            return ()
        data = self.path.read_bytes()
        rows: list[OffsetRawObservation] = []
        offset = 0
        lines = data.splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                rows.append(OffsetRawObservation.model_validate_json(line))
            except ValidationError as error:
                partial_tail = index == len(lines) - 1 and not line.endswith(b"\n")
                if not partial_tail:
                    raise ReuseError("raw ledger is malformed") from error
                with self.path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            offset += len(line)
        by_key: dict[str, OffsetRawObservation] = {}
        for row in rows:
            existing = by_key.get(row.observation_key)
            if existing is not None and existing != row:
                raise ReuseError("raw ledger contains conflicting duplicate")
            by_key[row.observation_key] = row
        return tuple(by_key.values())

    @property
    def rows(self) -> tuple[OffsetRawObservation, ...]:
        return tuple(self._by_key.values())

    def append(self, row: OffsetRawObservation) -> bool:
        existing = self._by_key.get(row.observation_key)
        if existing is not None:
            if existing != row:
                raise ReuseError("conflicting duplicate observation key")
            return False
        with self.path.open("ab") as stream:
            stream.write((row.model_dump_json() + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        self._by_key[row.observation_key] = row
        return True

    def completed_keys(self) -> set[str]:
        return set(self._by_key)


def load_old_rows(path: Path) -> tuple[RawObservation, ...]:
    try:
        rows = tuple(
            RawObservation.model_validate_json(line)
            for line in path.read_bytes().splitlines()
            if line.strip()
        )
    except (OSError, ValidationError) as error:
        raise ReuseError("old raw score ledger is unreadable or malformed") from error
    by_identity: dict[tuple[str, int, int, int], RawObservation] = {}
    rejected: set[tuple[str, int, int, int]] = set()
    for row in rows:
        identity = (
            row.window_id,
            row.lag_ms,
            row.support_start_sample_16k,
            row.support_end_sample_16k,
        )
        existing = by_identity.get(identity)
        if existing is not None and (
            existing.score_status != row.score_status
            or existing.score != row.score
            or existing.reason != row.reason
        ):
            rejected.add(identity)
            continue
        by_identity.setdefault(identity, row)
    return tuple(row for identity, row in by_identity.items() if identity not in rejected)


def reusable_rows(
    expected: tuple[OffsetExpectedObservation, ...],
    old_rows: tuple[RawObservation, ...],
) -> tuple[OffsetRawObservation, ...]:
    lookup = {
        (row.window_id, row.lag_ms, row.support_start_sample_16k, row.support_end_sample_16k): row
        for row in old_rows
    }
    reused: list[OffsetRawObservation] = []
    for item in expected:
        old = lookup.get(
            (
                item.window_id,
                item.lag_ms,
                item.support_start_sample_16k,
                item.support_end_sample_16k,
            )
        )
        if old is None:
            continue
        immutable_fields_match = (
            old.recording_id == item.recording_id
            and old.speaker_id == item.speaker_id
            and old.source_id == item.source_id
            and old.source_group_id == item.source_group_id
            and old.frame_start == item.frame_start
            and old.window_start_pts_seconds == item.window_start_pts_seconds
            and old.source_video_fps == item.source_video_fps
        )
        if not immutable_fields_match:
            raise ReuseError("old score identity conflicts with expected window provenance")
        reused.append(
            OffsetRawObservation(
                **item.model_dump(),
                score_status=old.score_status,
                score=old.score,
                reason=old.reason,
            )
        )
    return tuple(reused)


__all__ = ["ObservationLedger", "ReuseError", "load_old_rows", "reusable_rows"]
