from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

import torch

from koen_syncnet_eval.contracts import ExpectedObservation
from koen_syncnet_eval.gpu import GpuLease, check_cuda_headroom
from koen_syncnet_eval.ledger import write_atomic
from koen_syncnet_eval.resources import check_resources
from temporal_syncnet_eval.budget import GpuBudget

from .adapter_scoring import AdapterScorer
from .adapters import KoNetAdapterConfig
from .contracts import OffsetExpectedObservation, OffsetRawObservation
from .planning import plan_heldout
from .progress import write_progress
from .preparation import load_recordings
from .reuse import ObservationLedger
from .runner_io import load_offsets, load_windows


TEST_SPEAKERS = frozenset(("C282", "C371", "C580", "C613"))


@dataclass(frozen=True, slots=True)
class ArmRequest:
    source_root: Path
    output_root: Path
    model_id: str
    checkpoint: Path
    training_config: Path
    native_root: Path
    shared_gpu_owner: Path
    batch_size: int = 32


def prepare_arm(request: ArmRequest) -> int:
    recordings = tuple(
        row
        for row in load_recordings(request.source_root / "recordings.jsonl")
        if row.speaker_id in TEST_SPEAKERS
    )
    windows = load_windows(request.source_root / "cohort.jsonl")
    decisions = tuple(
        row
        for row in load_offsets(request.source_root / "offsets.jsonl")
        if any(item.recording_id == row.recording_id for item in recordings)
    )
    source_config = json.loads((request.source_root / "config.json").read_text(encoding="utf-8"))
    expected = plan_heldout(
        windows,
        recordings,
        decisions,
        str(source_config["run_id"]),
        str(source_config["config_digest"]),
        request.model_id,
    )
    request.output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "model_id": request.model_id,
        "checkpoint": str(request.checkpoint.resolve()),
        "training_config": str(request.training_config.resolve()),
        "native_root": str(request.native_root.resolve()),
        "source_root": str(request.source_root.resolve()),
        "test_speakers": sorted(TEST_SPEAKERS),
        "numeric_reuse_enabled": False,
        "offsets_source": str((request.source_root / "offsets.jsonl").resolve()),
    }
    write_atomic(request.output_root / "config.json", json.dumps(config, indent=2) + "\n")
    write_atomic(
        request.output_root / "expected.jsonl",
        "".join(row.model_dump_json() + "\n" for row in expected),
    )
    write_atomic(
        request.output_root / "terminal.json",
        json.dumps(
            {
                "status": "prepared",
                "model_id": request.model_id,
                "expected_observations": len(expected),
                "completed_observations": 0,
                "recordings": len(recordings),
                "gpu_launched": False,
            },
            indent=2,
        )
        + "\n",
    )
    return len(expected)


def score_arm(request: ArmRequest) -> None:
    started = time.perf_counter()
    expected = tuple(
        OffsetExpectedObservation.model_validate_json(line)
        for line in (request.output_root / "expected.jsonl").read_bytes().splitlines()
        if line.strip()
    )
    windows = load_windows(request.source_root / "cohort.jsonl")
    by_window = {row.window_id: row for row in windows}
    ledger = ObservationLedger(request.output_root / "raw.jsonl")
    check_resources(request.output_root, request.output_root / "cache")
    check_cuda_headroom()
    lease = GpuLease(request.shared_gpu_owner)
    lease.acquire()
    budget = GpuBudget(request.source_root / "runtime" / "gpu-budget.json", 36_000.0)
    attempt_id = budget.begin(request.model_id)
    completed = False
    try:
        write_atomic(
            request.output_root / "terminal.json",
            json.dumps(
                {
                    "status": "running",
                    "model_id": request.model_id,
                    "expected_observations": len(expected),
                    "completed_observations": len(ledger.rows),
                    "gpu_launched": True,
                    "elapsed_seconds": 0.0,
                },
                indent=2,
            )
            + "\n",
        )
        scorer = AdapterScorer(
            KoNetAdapterConfig(
                request.model_id, request.checkpoint, request.training_config, torch.device("cuda")
            ),
            request.native_root,
        )
        write_atomic(
            request.output_root / "adapter-receipt.json",
            json.dumps(
                {
                    **asdict(scorer.encoder.receipt),
                    "checkpoint": str(scorer.encoder.receipt.checkpoint),
                    "training_config": str(scorer.encoder.receipt.training_config),
                    "base_checkpoint": str(scorer.encoder.receipt.base_checkpoint),
                    "architecture_config": str(scorer.encoder.receipt.architecture_config),
                },
                indent=2,
            )
            + "\n",
        )
        for window_id in dict.fromkeys(row.window_id for row in expected):
            pending = tuple(
                row
                for row in expected
                if row.window_id == window_id and row.observation_key not in ledger.completed_keys()
            )
            if not pending:
                continue
            base = tuple(
                ExpectedObservation.model_validate(row.model_dump(exclude={"model_id"}))
                for row in pending
            )
            raw = scorer.score(by_window[window_id], base, request.batch_size)
            for item, observation in zip(pending, raw, strict=True):
                ledger.append(
                    OffsetRawObservation(
                        **item.model_dump(),
                        score_status=observation.score_status,
                        score=observation.score,
                        reason=observation.reason,
                    )
                )
            write_progress(
                request.output_root / "progress.json",
                json.dumps(
                    {
                        "model_id": request.model_id,
                        "window_id": window_id,
                        "completed_observations": len(ledger.rows),
                        "expected_observations": len(expected),
                        "pid": os.getpid(),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    indent=2,
                )
                + "\n",
            )
            budget.remaining_seconds()
            check_cuda_headroom()
        completed = len(ledger.completed_keys()) == len(expected)
        write_atomic(
            request.output_root / "terminal.json",
            json.dumps(
                {
                    "status": "complete" if completed else "failed",
                    "model_id": request.model_id,
                    "expected_observations": len(expected),
                    "completed_observations": len(ledger.rows),
                    "gpu_launched": True,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                indent=2,
            )
            + "\n",
        )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        write_atomic(
            request.output_root / "terminal.json",
            json.dumps(
                {
                    "status": "failed",
                    "model_id": request.model_id,
                    "reason": str(error),
                    "expected_observations": len(expected),
                    "completed_observations": len(ledger.rows),
                    "gpu_launched": True,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                indent=2,
            )
            + "\n",
        )
        raise
    finally:
        budget.finish(attempt_id, "complete" if completed else "failed")
        lease.release()


__all__ = ["ArmRequest", "prepare_arm", "score_arm"]
