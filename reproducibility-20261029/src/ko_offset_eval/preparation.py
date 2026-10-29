from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from koen_syncnet_eval.contracts import ContractError
from koen_syncnet_eval.ledger import write_atomic
from koen_syncnet_eval.resources import check_resources
from koen_syncnet_eval.scoring import audio_sample_count_16k

from .cohort import build_recordings, load_korean_windows, selected_windows
from .contracts import OffsetRecording, PreparationRequest
from .planning import plan_calibration
from .reuse import ObservationLedger, ReuseError, load_old_rows, reusable_rows


ORIGINAL_CHECKPOINT_SHA256 = "77678d13861a02d6e83d0b169962175b6266be773f1f6f85fda2b0182e225118"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ContractError("source cohort is unreadable") from error
    return digest.hexdigest()


def _validate_reuse_identity(request: PreparationRequest) -> None:
    try:
        old_config = json.loads(request.old_config.read_text(encoding="utf-8"))
        old_score = json.loads(request.old_score_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReuseError("old run identity is unreadable or malformed") from error
    accepted = (
        old_config.get("protocol_version") == "koen-signed80-v3"
        and old_config.get("visual_policy") == "normalize-1280x720"
        and old_score.get("backend") == "frozen"
        and old_score.get("visual_policy") == "normalize-1280x720"
        and Path(str(old_score.get("checkpoint"))).resolve() == request.checkpoint.resolve()
        and Path(str(old_score.get("native_root"))).resolve() == request.native_root.resolve()
    )
    if not accepted:
        raise ReuseError("old scores do not match checkpoint/preprocess/runtime identity")


def _configuration(request: PreparationRequest) -> tuple[str, str]:
    if not request.checkpoint.is_file() or not request.native_root.is_dir():
        raise ContractError("checkpoint or native runtime root is unavailable")
    checkpoint_stat = request.checkpoint.stat()
    checkpoint_sha256 = _sha256(request.checkpoint)
    if request.model_id == "original" and checkpoint_sha256 != ORIGINAL_CHECKPOINT_SHA256:
        raise ContractError("original checkpoint SHA256 differs from canonical receipt")
    body = {
        "schema_version": 1,
        "run_id": request.run_id,
        "model_id": request.model_id,
        "source_cohort": str(request.source_cohort.resolve()),
        "source_cohort_sha256": _sha256(request.source_cohort),
        "checkpoint": str(request.checkpoint.resolve()),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "checkpoint_sha256": checkpoint_sha256,
        "native_root": str(request.native_root.resolve()),
        "visual_policy": "normalize-1280x720",
        "audio_interface": "official_latentsync_mel_16khz_80x52_centered_stft",
        "visual_interface": "720p_before_alignment_normalize_1280x720_16_frames",
        "protocol_version": "ko-offset-a-to-b-v1",
        "positive_lag": "delayed_audio_earlier_source_samples",
        "numeric_reuse_enabled": request.reuse_old_scores,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return json.dumps({**body, "config_digest": digest}, indent=2) + "\n", digest


def prepare(request: PreparationRequest) -> tuple[int, int, int]:
    _validate_reuse_identity(request)
    config_payload, config_digest = _configuration(request)
    if request.reference_preparation is None:
        source_windows = load_korean_windows(request.source_cohort)
        audio_sample_counts = {
            path: audio_sample_count_16k(path) for path in {row.audio_path for row in source_windows}
        }
        recordings = build_recordings(source_windows, audio_sample_counts)
        windows = selected_windows(source_windows, recordings)
    else:
        recordings = load_recordings(request.reference_preparation / "recordings.jsonl")
        windows = load_korean_windows(request.reference_preparation / "cohort.jsonl")
    if len(recordings) != 276:
        raise ContractError("source cohort does not contain exactly 276 Korean recordings")
    expected = plan_calibration(
        windows, recordings, request.run_id, config_digest, request.model_id
    )
    request.output_root.mkdir(parents=True, exist_ok=True)
    check_resources(request.output_root, request.output_root / "cache")
    config_path = request.output_root / "config.json"
    if config_path.exists() and config_path.read_text(encoding="utf-8") != config_payload:
        raise ContractError("existing offset run configuration differs")
    write_atomic(config_path, config_payload)
    write_atomic(
        request.output_root / "recordings.jsonl",
        "".join(row.model_dump_json() + "\n" for row in recordings),
    )
    write_atomic(
        request.output_root / "cohort.jsonl",
        "".join(row.model_dump_json() + "\n" for row in windows),
    )
    write_atomic(
        request.output_root / "runtime" / "expected-calibration.jsonl",
        "".join(row.model_dump_json() + "\n" for row in expected),
    )
    ledger = ObservationLedger(request.output_root / "runtime" / "raw.jsonl")
    old_rows = load_old_rows(request.old_raw)
    reused = reusable_rows(expected, old_rows)
    reuse_map = tuple(
        {
            "observation_key": row.observation_key,
            "old_run_id": "run-final-720p",
            "old_score_status": row.score_status,
            "numeric_reuse": False,
        }
        for row in reused
    )
    write_atomic(
        request.output_root / "runtime" / "reuse-map.jsonl",
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in reuse_map),
    )
    if request.reuse_old_scores:
        for row in reused:
            ledger.append(row)
    terminal = {
        "schema_version": 1,
        "run_id": request.run_id,
        "status": "prepared",
        "recordings": len(recordings),
        "included_recordings": sum(row.decision_status == "included" for row in recordings),
        "expected_observations": len(expected),
        "completed_observations": len(ledger.completed_keys()),
        "reusable_reference_observations": len(reused),
        "reused_observations": len(reused) if request.reuse_old_scores else 0,
        "gpu_launched": False,
    }
    write_atomic(
        request.output_root / "runtime" / "terminal.json",
        json.dumps(terminal, indent=2) + "\n",
    )
    return len(recordings), len(expected), len(reused)


def load_recordings(path: Path) -> tuple[OffsetRecording, ...]:
    try:
        return tuple(
            OffsetRecording.model_validate_json(line)
            for line in path.read_bytes().splitlines()
            if line.strip()
        )
    except (OSError, ValidationError) as error:
        raise ContractError("recording metadata is unreadable or malformed") from error


__all__ = ["load_recordings", "prepare"]
