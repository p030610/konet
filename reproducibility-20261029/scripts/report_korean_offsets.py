# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.23"]
# ///
# How to run: uv run --offline --no-project --no-sync --python <python> python report_korean_offsets.py --help
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ko_offset_eval.metrics import OffsetError, analyze, read_raw_jsonl
from ko_offset_eval.reporting import read_offsets, read_recordings, write_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the Korean held-out offset metrics report from append-only scores.")
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--recordings", required=True, type=Path)
    parser.add_argument("--offsets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", default=10_000, type=int)
    parser.add_argument("--seed", default=2701, type=int)
    args = parser.parse_args()
    try:
        analysis = analyze(read_raw_jsonl(args.raw), read_recordings(args.recordings), read_offsets(args.offsets), bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        paths = write_outputs(analysis, args.output)
    except (OffsetError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for path in paths:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
