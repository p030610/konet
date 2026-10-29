from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Literal

from koen_syncnet_eval.gpu import GpuLease, check_cuda_headroom
from koen_syncnet_eval.ledger import write_atomic
from koen_syncnet_eval.policy import VisualPolicy
from koen_syncnet_eval.scoring import FrozenScorer
from temporal_syncnet_eval.budget import GpuBudget

from .contracts import OffsetDecision
from .planning import plan_heldout
from .progress import write_progress
from .preparation import load_recordings
from .reuse import ObservationLedger, load_old_rows, reusable_rows
from .runner_io import load_expected, load_offsets, load_windows, write_offsets
from .scoring import calibrate_recording
from .window_scoring import score_pending


Backend = Literal["fixture", "frozen"]


@dataclass(frozen=True, slots=True)
class OffsetScoreRequest:
    output_root: Path
    old_raw: Path
    backend: Backend
    device: Literal["cpu", "cuda"]
    batch_size: int
    maximum_recordings: int | None
    shared_gpu_owner_path: Path


def score(request: OffsetScoreRequest) -> None:
    started = time.perf_counter()
    config = json.loads((request.output_root / "config.json").read_text(encoding="utf-8"))
    checkpoint = Path(str(config["checkpoint"]))
    native_root = Path(str(config["native_root"]))
    recordings = load_recordings(request.output_root / "recordings.jsonl")
    windows = load_windows(request.output_root / "cohort.jsonl")
    calibration = load_expected(request.output_root / "runtime" / "expected-calibration.jsonl")
    ledger = ObservationLedger(request.output_root / "runtime" / "raw.jsonl")
    lease = None
    budget = None
    attempt_id = None
    invocation_finished = False
    try:
        if request.device == "cuda":
            check_cuda_headroom()
            lease = GpuLease(request.shared_gpu_owner_path)
            lease.acquire()
            budget = GpuBudget(request.output_root / "runtime" / "gpu-budget.json", 36_000.0)
            attempt_id = budget.begin(str(config["run_id"]))
        write_atomic(
            request.output_root / "runtime" / "terminal.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": config["run_id"],
                    "status": "running",
                    "phase": "recording_pipeline",
                    "expected_observations": len(calibration),
                    "completed_observations": len(ledger.completed_keys()),
                    "gpu_launched": request.device == "cuda",
                    "elapsed_seconds": 0.0,
                },
                indent=2,
            )
            + "\n",
        )
        scorer = (
            FrozenScorer(
                checkpoint,
                native_root,
                request.device,
                VisualPolicy.NORMALIZE_1280X720,
            )
            if request.backend == "frozen"
            else None
        )
        decisions = list(load_offsets(request.output_root / "offsets.jsonl"))
        decision_by_recording = {row.recording_id: row for row in decisions}
        old_rows = load_old_rows(request.old_raw)
        candidates = []
        for recording in recordings:
            decision = decision_by_recording.get(recording.recording_id)
            heldout_rows = (
                plan_heldout(
                    windows,
                    (recording,),
                    (decision,),
                    str(config["run_id"]),
                    str(config["config_digest"]),
                    str(config["model_id"]),
                )
                if decision is not None
                else ()
            )
            unfinished = decision is None or any(
                row.observation_key not in ledger.completed_keys() for row in heldout_rows
            )
            if unfinished:
                candidates.append(recording)
        selected = (
            candidates[: request.maximum_recordings]
            if request.maximum_recordings is not None
            else candidates
        )
        for recording in selected:
            decision = decision_by_recording.get(recording.recording_id)
            if decision is None:
                if recording.decision_status == "construction_exclusion":
                    decision = OffsetDecision(
                        recording_id=recording.recording_id,
                        selected_offset_ms=None,
                        observed_peak_ms=None,
                        status="incomplete",
                        window_count=0,
                        peak_mean_score=None,
                        prominence=None,
                        per_window_peak_agreement=None,
                    )
                else:
                    recording_calibration = tuple(
                        row for row in calibration if row.recording_id == recording.recording_id
                    )
                    score_pending(request, recording_calibration, windows, ledger, scorer, budget)
                    decision = calibrate_recording(
                        recording.recording_id, recording.a_window_ids, ledger.rows
                    )
                decisions.append(decision)
                decision_by_recording[recording.recording_id] = decision
                write_offsets(request.output_root, tuple(decisions))
            recording_heldout = plan_heldout(
                windows,
                (recording,),
                (decision,),
                str(config["run_id"]),
                str(config["config_digest"]),
                str(config["model_id"]),
            )
            if bool(config.get("numeric_reuse_enabled", False)):
                for row in reusable_rows(recording_heldout, old_rows):
                    ledger.append(row)
            score_pending(request, recording_heldout, windows, ledger, scorer, budget)
            write_progress(
                request.output_root / "runtime" / "progress.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "phase": "recording_complete",
                        "recording_id": recording.recording_id,
                        "recordings_completed": len(decisions),
                        "recordings_total": len(recordings),
                        "raw_rows": len(ledger.rows),
                        "pid": os.getpid(),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    indent=2,
                )
                + "\n",
            )
            all_heldout = plan_heldout(
                windows,
                recordings,
                tuple(decisions),
                str(config["run_id"]),
                str(config["config_digest"]),
                str(config["model_id"]),
            )
            expected = (*calibration, *all_heldout)
            write_atomic(
                request.output_root / "runtime" / "expected.jsonl",
                "".join(row.model_dump_json() + "\n" for row in expected),
            )
        all_heldout = plan_heldout(
            windows,
            recordings,
            tuple(decisions),
            str(config["run_id"]),
            str(config["config_digest"]),
            str(config["model_id"]),
        )
        expected = (*calibration, *all_heldout)
        complete = ledger.completed_keys()
        expected_keys = {row.observation_key for row in expected}
        status = (
            "complete"
            if len(decisions) == len(recordings) and expected_keys == complete
            else "running"
        )
        terminal = {
            "schema_version": 1,
            "run_id": config["run_id"],
            "status": status,
            "phase": "complete" if status == "complete" else "recording_pipeline",
            "expected_observations": len(expected_keys),
            "completed_observations": len(expected_keys & complete),
            "gpu_launched": request.device == "cuda",
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_atomic(
            request.output_root / "runtime" / "terminal.json",
            json.dumps(terminal, indent=2) + "\n",
        )
        invocation_finished = True
    finally:
        if budget is not None and attempt_id is not None:
            budget.finish(attempt_id, "complete" if invocation_finished else "failed")
        if lease is not None:
            lease.release()


__all__ = ["OffsetScoreRequest", "score"]
