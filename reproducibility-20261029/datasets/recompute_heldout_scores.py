#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.23"]
# ///
# How to run: uv run --offline --no-project --no-sync recompute_heldout_scores.py
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent
DISTANCES = (40, 80, 120, 160, 240, 320)
EPSILON = 1e-6


def win(center: float, target: float) -> float:
    return 1.0 if center > target + EPSILON else 0.0 if center < target - EPSILON else 0.5


def main() -> None:
    with (ROOT / "heldout_scores.csv").open(encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    models = tuple(sorted({row["model_id"] for row in rows}))
    keys = {(row["speaker_alias"], row["recording_alias"], row["window_alias"]) for row in rows}
    recordings = {row["recording_alias"] for row in rows}
    speakers = {row["speaker_alias"] for row in rows}
    by_model = {model: {key: row for row in rows if row["model_id"] == model for key in [(row["speaker_alias"], row["recording_alias"], row["window_alias"]) ]} for model in models}
    score_columns = tuple(column for column in rows[0] if column.startswith("score_"))
    scores_finite = all(math.isfinite(float(row[column])) for row in rows for column in score_columns)
    if len(rows) != 288 or len(keys) != 96 or len(recordings) != 44 or len(speakers) != 4 or len(models) != 3 or any(len(index) != 96 or set(index) != keys for index in by_model.values()) or not scores_finite:
        raise SystemExit("expected 288 rows: 96 identical paired windows × 3 models")
    output: dict[str, dict[str, dict[str, float]]] = {}
    for model, index in by_model.items():
        model_output: dict[str, dict[str, float]] = {}
        for distance in DISTANCES:
            records: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
            for (speaker, recording, _), row in index.items():
                center = float(row["score_center"])
                negative = float(row[f"score_negative_{distance}"])
                positive = float(row[f"score_positive_{distance}"])
                records[(speaker, recording)].append((win(center, negative), win(center, positive), float(center > negative + EPSILON and center > positive + EPSILON), center - max(negative, positive)))
            speakers: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
            for (speaker, _), values in records.items():
                speakers[speaker].append(tuple(float(np.mean([item[column] for item in values])) for column in range(4)))
            negative = {speaker: float(np.mean([item[0] for item in values])) for speaker, values in speakers.items()}
            positive = {speaker: float(np.mean([item[1] for item in values])) for speaker, values in speakers.items()}
            both_win = {speaker: float(np.mean([item[2] for item in values])) for speaker, values in speakers.items()}
            minimum_margin = {speaker: float(np.mean([item[3] for item in values])) for speaker, values in speakers.items()}
            signed = {speaker: (negative[speaker] + positive[speaker]) / 2 for speaker in speakers}
            model_output[str(distance)] = {"negative_win": float(np.mean(list(negative.values()))), "positive_win": float(np.mean(list(positive.values()))), "signed_directional": float(np.mean(list(signed.values())),), "strict_both_win": float(np.mean(list(both_win.values()))), "mean_minimum_margin": float(np.mean(list(minimum_margin.values()))), **{f"speaker_{speaker}": value for speaker, value in signed.items()}}
        output[model] = model_output
    (ROOT / "recomputed_heldout_metrics.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("PASS: 96 paired windows, 44 recording aliases, 4 speaker aliases, 3 models")
    for distance in DISTANCES:
        print(distance, *(f"{model}: both={output[model][str(distance)]['strict_both_win']:.4f}, margin={output[model][str(distance)]['mean_minimum_margin']:.4f}, signed={output[model][str(distance)]['signed_directional']:.4f}" for model in models))


if __name__ == "__main__":
    main()
