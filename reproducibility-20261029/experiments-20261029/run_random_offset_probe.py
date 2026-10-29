#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.23"]
# ///
# How to run: uv run --offline --no-project --no-sync run_random_offset_probe.py
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final

import numpy as np


ROOT: Final = Path(__file__).resolve().parent
SOURCE: Final = ROOT.parent / "datasets" / "heldout_scores.csv"
TARGETS: Final = (-320, -240, -160, -120, -80, -40, 40, 80, 120, 160, 240, 320)
NEAR: Final = (-120, -80, -40, 40, 80, 120)
FAR: Final = (-320, -240, -160, 160, 240, 320)
EPSILON: Final = 1e-6
SEED: Final = 1701


def identity(row: dict[str, str]) -> tuple[str, str, str]:
    return row["speaker_alias"], row["recording_alias"], row["window_alias"]


def score(row: dict[str, str], lag: int) -> float:
    return float(row[f"score_negative_{-lag}" if lag < 0 else f"score_positive_{lag}"])


def aggregate(rows: tuple[dict[str, str], ...], selected: dict[tuple[str, str, str], int] | None, lags: tuple[int, ...]) -> dict[str, float]:
    records: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        use_lags = (selected[identity(row)],) if selected is not None else lags
        center = float(row["score_center"])
        comparisons = tuple(center - score(row, lag) for lag in use_lags)
        rank = float(np.mean([1.0 if value > EPSILON else 0.0 if value < -EPSILON else 0.5 for value in comparisons]))
        strict = float(np.mean([float(value > EPSILON) for value in comparisons]))
        records[(row["speaker_alias"], row["recording_alias"])].append((rank, strict, float(np.mean(comparisons))))
    speakers: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for (speaker, _), values in records.items():
        speakers[speaker].append(tuple(float(np.mean([row[index] for row in values])) for index in range(3)))
    if len(records) != 44 or len(speakers) != 4:
        raise RuntimeError("expected_44_records_4_speakers")
    rank, strict, margin = (float(value) for value in np.mean([np.mean(values, axis=0) for values in speakers.values()], axis=0))
    return {"rank_win_tie_half": rank, "strict_center_greater": strict, "mean_margin": margin}


def main() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    models: dict[str, dict[tuple[str, str, str], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        models[row["model_id"]][identity(row)] = row
    keys = set(next(iter(models.values())))
    if len(keys) != 96 or any(set(index) != keys for index in models.values()):
        raise RuntimeError("expected_identical_96_window_coverage")
    draw = {key: int(lag) for key, lag in zip(sorted(keys), np.random.default_rng(SEED).choice(np.asarray(TARGETS), size=96), strict=True)}
    with (ROOT / "random_offset_selection.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("speaker_alias", "recording_alias", "window_alias", "selected_target_lag_ms", "target_group"))
        writer.writeheader()
        writer.writerows({"speaker_alias": key[0], "recording_alias": key[1], "window_alias": key[2], "selected_target_lag_ms": lag, "target_group": "near" if lag in NEAR else "far"} for key, lag in sorted(draw.items()))
    exact = {model: {"all_12": aggregate(tuple(index.values()), None, TARGETS), "near_6": aggregate(tuple(index.values()), None, NEAR), "far_6": aggregate(tuple(index.values()), None, FAR)} for model, index in models.items()}
    sampled = {model: aggregate(tuple(index.values()), draw, TARGETS) for model, index in models.items()}
    manifest = {"kind": "cached_random_offset_probe_v1", "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "seed": SEED, "targets_ms": TARGETS, "primary": "exact uniform expectation across 12 cached signed targets", "secondary": "one fixed shared random draw per window", "support": {"windows": 96, "recordings": 44, "speakers": 4}, "draw_target_counts": dict(sorted(Counter(draw.values()).items())), "note": "Exploratory score-ordering diagnostic; no held-out model selection or binary synchronization claim."}
    (ROOT / "random_offset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (ROOT / "random_offset_metrics.json").write_text(json.dumps({"exact_uniform_expectation": exact, "fixed_seed1701_draw": sampled}, indent=2) + "\n", encoding="utf-8")
    lines = ["# Cached random-offset score probe", "", "| method | all-12 rank / strict / margin | near-6 rank / strict / margin | far-6 rank / strict / margin | fixed-draw rank / strict / margin |", "|---|---|---|---|---|"]
    for model in models:
        render = lambda values: f"{values['rank_win_tie_half']:.4f} / {values['strict_center_greater']:.4f} / {values['mean_margin']:.4f}"
        lines.append(f"| {model} | {render(exact[model]['all_12'])} | {render(exact[model]['near_6'])} | {render(exact[model]['far_6'])} | {render(sampled[model])} |")
    lines.extend(("", f"Exact primary uses all 12 equally weighted targets. Fixed draw uses NumPy default_rng({SEED}) and counts {manifest['draw_target_counts']}.", "Rank assigns ties 0.5; strict assigns ties 0. This is not binary synchronization accuracy.", "No method was selected from held-out outcomes."))
    (ROOT / "random_offset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "models": list(models), "selection_rows": len(draw), "seed": SEED}))


if __name__ == "__main__":
    main()
