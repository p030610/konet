#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

# How to run
# uv run --offline --no-project --no-sync --python python scripts/eval_korean_offsets.py --help

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ko_offset_eval.contracts import PreparationRequest
from ko_offset_eval.arms import ArmRequest, prepare_arm, score_arm
from ko_offset_eval.preparation import prepare
from ko_offset_eval.parity import run_parity_probe
from ko_offset_eval.runner import OffsetScoreRequest, score


DEFAULT_SOURCE = Path(r"G:\StableSyncNet_720p_ExpandedHDTF_20261020\cohort\run-final\cohort.jsonl")
DEFAULT_OLD_RUNTIME = Path(r"G:\StableSyncNet_720p_ExpandedHDTF_20261020\runtime\run-final-720p")
DEFAULT_OUTPUT = Path(r"G:\StableSyncNet_Korean_Offsets_20261024")
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "latentsync" / "stable_syncnet.pt"
DEFAULT_NATIVE_ROOT = PROJECT_ROOT / "vendor" / "LatentSync"
DEFAULT_SHARED_GPU_OWNER = Path(r"G:\StableSyncNet_shared_gpu.owner")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--source-cohort", type=Path, default=DEFAULT_SOURCE)
    prepare_parser.add_argument("--old-runtime", type=Path, default=DEFAULT_OLD_RUNTIME)
    prepare_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    prepare_parser.add_argument("--native-root", type=Path, default=DEFAULT_NATIVE_ROOT)
    prepare_parser.add_argument("--run-id", default="korean-offsets-original-20261024")
    prepare_parser.add_argument("--reuse-old-scores", action="store_true")
    prepare_parser.add_argument("--reference-preparation", type=Path)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    score_parser.add_argument("--old-runtime", type=Path, default=DEFAULT_OLD_RUNTIME)
    score_parser.add_argument("--backend", choices=("fixture", "frozen"), default="frozen")
    score_parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    score_parser.add_argument("--batch-size", type=int, default=32)
    score_parser.add_argument("--max-recordings", type=int)
    score_parser.add_argument("--shared-gpu-owner", type=Path, default=DEFAULT_SHARED_GPU_OWNER)
    parity_parser = commands.add_parser("parity")
    parity_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parity_parser.add_argument("--shared-gpu-owner", type=Path, default=DEFAULT_SHARED_GPU_OWNER)
    for name in ("prepare-arm", "score-arm"):
        arm_parser = commands.add_parser(name)
        arm_parser.add_argument("--source-root", type=Path, required=True)
        arm_parser.add_argument("--output", type=Path, required=True)
        arm_parser.add_argument("--model-id", required=True)
        arm_parser.add_argument("--checkpoint", type=Path, required=True)
        arm_parser.add_argument("--training-config", type=Path, required=True)
        arm_parser.add_argument("--native-root", type=Path, default=DEFAULT_NATIVE_ROOT)
        arm_parser.add_argument("--shared-gpu-owner", type=Path, default=DEFAULT_SHARED_GPU_OWNER)
        arm_parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        recordings, expected, reused = prepare(
            PreparationRequest(
                source_cohort=arguments.source_cohort.resolve(),
                output_root=arguments.output.resolve(),
                old_raw=(arguments.old_runtime / "raw.jsonl").resolve(),
                old_config=(arguments.old_runtime / "config.json").resolve(),
                old_score_config=(arguments.old_runtime / "score-config.json").resolve(),
                checkpoint=arguments.checkpoint.resolve(),
                native_root=arguments.native_root.resolve(),
                run_id=arguments.run_id,
                reuse_old_scores=arguments.reuse_old_scores,
                reference_preparation=(
                    arguments.reference_preparation.resolve()
                    if arguments.reference_preparation is not None
                    else None
                ),
            )
        )
        print(f"recordings={recordings} expected_calibration={expected} reused={reused}")
        print(f"output={arguments.output.resolve()}")
        return 0
    if arguments.command == "parity":
        path = run_parity_probe(arguments.output.resolve(), arguments.shared_gpu_owner.resolve())
        print(path.read_text(encoding="utf-8"))
        return 0
    if arguments.command in {"prepare-arm", "score-arm"}:
        request = ArmRequest(
            source_root=arguments.source_root.resolve(),
            output_root=arguments.output.resolve(),
            model_id=arguments.model_id,
            checkpoint=arguments.checkpoint.resolve(),
            training_config=arguments.training_config.resolve(),
            native_root=arguments.native_root.resolve(),
            shared_gpu_owner=arguments.shared_gpu_owner.resolve(),
            batch_size=arguments.batch_size,
        )
        if arguments.command == "prepare-arm":
            print(f"expected={prepare_arm(request)}")
        else:
            score_arm(request)
            print((request.output_root / "terminal.json").read_text(encoding="utf-8"))
        return 0
    if arguments.batch_size < 1 or (
        arguments.max_recordings is not None and arguments.max_recordings < 1
    ):
        parser.error("batch-size and max-recordings must be positive")
    score(
        OffsetScoreRequest(
            output_root=arguments.output.resolve(),
            old_raw=(arguments.old_runtime / "raw.jsonl").resolve(),
            backend=arguments.backend,
            device=arguments.device,
            batch_size=arguments.batch_size,
            maximum_recordings=arguments.max_recordings,
            shared_gpu_owner_path=arguments.shared_gpu_owner.resolve(),
        )
    )
    print((arguments.output / "runtime" / "terminal.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
