from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Literal, Protocol

from koen_syncnet_eval.contracts import CohortWindow, ExpectedObservation
from koen_syncnet_eval.gpu import check_cuda_headroom
from koen_syncnet_eval.resources import check_resources
from koen_syncnet_eval.scoring import FrozenScorer, score_fixture
from temporal_syncnet_eval.budget import GpuBudget

from .contracts import OffsetExpectedObservation, OffsetRawObservation
from .reuse import ObservationLedger
from .progress import write_progress


class ScoreOptions(Protocol):
    output_root: Path
    device: Literal["cpu", "cuda"]
    batch_size: int
    maximum_recordings: int | None


def _base_expected(row: OffsetExpectedObservation) -> ExpectedObservation:
    return ExpectedObservation.model_validate(row.model_dump(exclude={"model_id"}))


def _enrich(
    expected: OffsetExpectedObservation,
    score_status: str,
    score: float | None,
    reason: str | None,
) -> OffsetRawObservation:
    return OffsetRawObservation(
        **expected.model_dump(), score_status=score_status, score=score, reason=reason
    )


def score_pending(
    request: ScoreOptions,
    expected: tuple[OffsetExpectedObservation, ...],
    windows: tuple[CohortWindow, ...],
    ledger: ObservationLedger,
    scorer: FrozenScorer | None,
    budget: GpuBudget | None = None,
) -> int:
    complete = ledger.completed_keys()
    window_by_id = {row.window_id: row for row in windows}
    recording_order = tuple(dict.fromkeys(row.recording_id for row in expected))
    pending_recordings = tuple(
        recording_id
        for recording_id in recording_order
        if any(
            row.recording_id == recording_id and row.observation_key not in complete
            for row in expected
        )
    )
    selected_recordings = set(
        pending_recordings[: request.maximum_recordings]
        if request.maximum_recordings is not None
        else pending_recordings
    )
    appended = 0
    for window_id in dict.fromkeys(row.window_id for row in expected):
        pending = tuple(
            row
            for row in expected
            if row.window_id == window_id
            and row.recording_id in selected_recordings
            and row.observation_key not in complete
        )
        if not pending:
            continue
        base = tuple(_base_expected(row) for row in pending)
        raw_rows = (
            score_fixture(base, 10**12)
            if scorer is None
            else scorer.score(window_by_id[window_id], base, request.batch_size)
        )
        for expected_row, raw in zip(pending, raw_rows, strict=True):
            ledger.append(_enrich(expected_row, raw.score_status, raw.score, raw.reason))
            complete.add(expected_row.observation_key)
            appended += 1
        print(
            json.dumps(
                {
                    "phase": pending[0].fold,
                    "recording_id": pending[0].recording_id,
                    "window_id": window_id,
                    "completed_window_observations": len(pending),
                    "durable_observations": len(complete),
                }
            ),
            flush=True,
        )
        write_progress(
            request.output_root / "runtime" / "progress.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "phase": pending[0].fold,
                    "recording_id": pending[0].recording_id,
                    "window_id": window_id,
                    "completed_observations": len(complete),
                    "updated_monotonic_seconds": time.perf_counter(),
                },
                indent=2,
            )
            + "\n",
        )
        check_resources(request.output_root, request.output_root / "cache")
        if request.device == "cuda":
            check_cuda_headroom()
            if budget is not None:
                budget.remaining_seconds()
    return appended


__all__ = ["ScoreOptions", "score_pending"]
