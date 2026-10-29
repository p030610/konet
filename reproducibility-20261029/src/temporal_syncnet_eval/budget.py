from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


AttemptStatus = Literal["running", "complete", "failed", "interrupted"]


@dataclass(frozen=True, slots=True)
class BudgetError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


class BudgetAttempt(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempt_id: str
    run_id: str
    pid: int
    started_s: float
    finished_s: float | None = None
    elapsed_seconds: float = Field(ge=0.0)
    status: AttemptStatus


class BudgetLedger(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    limit_seconds: float = Field(gt=0.0)
    attempts: tuple[BudgetAttempt, ...] = ()

    @property
    def total_elapsed_seconds(self) -> float:
        return sum(attempt.elapsed_seconds for attempt in self.attempts)


class GpuBudget:
    def __init__(
        self,
        path: Path,
        limit_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self.limit_seconds = limit_seconds
        self.clock = clock

    def _read(self) -> BudgetLedger:
        if not self.path.exists():
            return BudgetLedger(limit_seconds=self.limit_seconds)
        try:
            ledger = BudgetLedger.model_validate_json(self.path.read_bytes())
        except (OSError, ValidationError) as error:
            raise BudgetError("GPU budget ledger is unreadable or malformed") from error
        if ledger.limit_seconds != self.limit_seconds:
            raise BudgetError("GPU budget limit differs from the shared ledger")
        return ledger

    def _write(self, ledger: BudgetLedger) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(ledger.model_dump_json(), encoding="utf-8")
        os.replace(temporary, self.path)

    def _locked(self, operation: Callable[[BudgetLedger], BudgetLedger]) -> BudgetLedger:
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise BudgetError("GPU budget ledger is locked by another executor") from error
        try:
            os.write(descriptor, str(os.getpid()).encode())
            ledger = operation(self._read())
            self._write(ledger)
            return ledger
        finally:
            os.close(descriptor)
            lock.unlink(missing_ok=True)

    def begin(self, run_id: str, started_s: float | None = None) -> str:
        now = self.clock()
        effective_start = now if started_s is None else started_s
        if effective_start > now:
            raise BudgetError("GPU budget start time is in the future")
        attempt_id = uuid.uuid4().hex

        def update(ledger: BudgetLedger) -> BudgetLedger:
            reconciled = tuple(
                attempt.model_copy(
                    update={
                        "finished_s": now,
                        "elapsed_seconds": max(0.0, now - attempt.started_s),
                        "status": "interrupted",
                    }
                )
                if attempt.status == "running"
                else attempt
                for attempt in ledger.attempts
            )
            if sum(attempt.elapsed_seconds for attempt in reconciled) >= self.limit_seconds:
                raise BudgetError("GPU wall budget exhausted")
            current = BudgetAttempt(
                attempt_id=attempt_id,
                run_id=run_id,
                pid=os.getpid(),
                started_s=effective_start,
                elapsed_seconds=0.0,
                status="running",
            )
            return ledger.model_copy(update={"attempts": (*reconciled, current)})

        self._locked(update)
        return attempt_id

    def finish(self, attempt_id: str, status: Literal["complete", "failed"]) -> BudgetLedger:
        now = self.clock()

        def update(ledger: BudgetLedger) -> BudgetLedger:
            found = False
            attempts: list[BudgetAttempt] = []
            for attempt in ledger.attempts:
                if attempt.attempt_id == attempt_id and attempt.status == "running":
                    found = True
                    attempts.append(
                        attempt.model_copy(
                            update={
                                "finished_s": now,
                                "elapsed_seconds": max(0.0, now - attempt.started_s),
                                "status": status,
                            }
                        )
                    )
                else:
                    attempts.append(attempt)
            if not found:
                raise BudgetError("GPU budget attempt is not active")
            return ledger.model_copy(update={"attempts": tuple(attempts)})

        return self._locked(update)

    def remaining_seconds(self) -> float:
        ledger = self._read()
        now = self.clock()
        used = sum(
            max(0.0, now - attempt.started_s)
            if attempt.status == "running"
            else attempt.elapsed_seconds
            for attempt in ledger.attempts
        )
        remaining = self.limit_seconds - used
        if remaining <= 0:
            raise BudgetError("GPU wall budget exhausted")
        return remaining


__all__ = ["BudgetError", "BudgetLedger", "GpuBudget"]
