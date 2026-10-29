from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from koen_syncnet_eval.contracts import CohortWindow, ExpectedObservation
from koen_syncnet_eval.gpu import GpuLease, check_cuda_headroom
from koen_syncnet_eval.ledger import write_atomic
from koen_syncnet_eval.policy import VisualPolicy
from koen_syncnet_eval.resources import check_resources
from koen_syncnet_eval.scoring import FrozenScorer

from .contracts import OffsetExpectedObservation
from .reuse import ObservationLedger


@dataclass(frozen=True, slots=True)
class ParityError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def run_parity_probe(output_root: Path, shared_gpu_owner: Path, batch_size: int = 3) -> Path:
    check_resources(output_root, output_root / "cache")
    check_cuda_headroom()
    config = json.loads((output_root / "config.json").read_text(encoding="utf-8"))
    windows = tuple(
        CohortWindow.model_validate_json(line)
        for line in (output_root / "cohort.jsonl").read_bytes().splitlines()
        if line.strip()
    )
    expected = tuple(
        OffsetExpectedObservation.model_validate_json(line)
        for line in (output_root / "runtime" / "expected-calibration.jsonl").read_bytes().splitlines()
        if line.strip()
    )
    ledger = ObservationLedger(output_root / "runtime" / "raw.jsonl")
    valid = {row.observation_key: row for row in ledger.rows if row.score_status == "valid"}
    first_window = next(row.window_id for row in expected if row.observation_key in valid)
    selected = tuple(
        row for row in expected if row.window_id == first_window and row.observation_key in valid
    )[:3]
    if len(selected) != 3:
        raise ParityError("three reusable parity cells are unavailable")
    window = next(row for row in windows if row.window_id == first_window)
    base = tuple(
        ExpectedObservation.model_validate(row.model_dump(exclude={"model_id"})) for row in selected
    )
    lease = GpuLease(shared_gpu_owner)
    lease.acquire()
    try:
        scorer = FrozenScorer(
            Path(str(config["checkpoint"])),
            Path(str(config["native_root"])),
            "cuda",
            VisualPolicy.NORMALIZE_1280X720,
        )
        observed = scorer.score(window, base, batch_size)
    finally:
        lease.release()
    comparisons = []
    for expected_row, fresh in zip(selected, observed, strict=True):
        reused = valid[expected_row.observation_key]
        if fresh.score_status != "valid" or fresh.score is None or reused.score is None:
            raise ParityError("parity recomputation did not produce finite valid scores")
        difference = abs(fresh.score - reused.score)
        comparisons.append(
            {
                "observation_key": expected_row.observation_key,
                "lag_ms": expected_row.lag_ms,
                "reused_score": reused.score,
                "fresh_score": fresh.score,
                "absolute_difference": difference,
                "finite": math.isfinite(fresh.score),
            }
        )
    payload = {
        "schema_version": 1,
        "model_id": config["model_id"],
        "window_id": first_window,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "comparisons": comparisons,
    }
    path = output_root / "runtime" / "parity.json"
    write_atomic(path, json.dumps(payload, indent=2) + "\n")
    check_resources(output_root, output_root / "cache")
    return path


__all__ = ["ParityError", "run_parity_probe"]
