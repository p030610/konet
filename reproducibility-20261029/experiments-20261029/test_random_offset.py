from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

import numpy as np
import run_random_offset_probe as probe

ROOT = Path(__file__).resolve().parent
MODELS = {'original', 'konet_full_epoch9', 'konet_lora_r8a16_epoch5'}


class RandomOffsetTests(unittest.TestCase):
    def test_published_three_model_metrics_reproduce(self) -> None:
        expected = json.loads((ROOT / 'reference/random_offset_metrics.json').read_text(encoding='utf-8-sig'))
        actual = json.loads((ROOT / 'random_offset_metrics.json').read_text(encoding='utf-8'))
        for section in expected:
            self.assertEqual(set(actual[section]), MODELS)
            for model in MODELS:
                groups = ('all_12', 'near_6', 'far_6') if section == 'exact_uniform_expectation' else (None,)
                for group in groups:
                    left = expected[section][model] if group is None else expected[section][model][group]
                    right = actual[section][model] if group is None else actual[section][model][group]
                    for metric, value in left.items():
                        self.assertAlmostEqual(value, right[metric], delta=1e-12)

    def test_identical_nonempty_paired_inputs(self) -> None:
        with probe.SOURCE.open(encoding='utf-8', newline='') as stream:
            rows = tuple(csv.DictReader(stream))
        groups = {model: {probe.identity(row) for row in rows if row['model_id'] == model} for model in MODELS}
        self.assertEqual(len(rows), 288)
        self.assertEqual(len(groups['original']), 96)
        self.assertEqual(groups['original'], groups['konet_full_epoch9'])
        self.assertEqual(groups['original'], groups['konet_lora_r8a16_epoch5'])
        self.assertEqual(len({key[0] for key in groups['original']}), 4)
        self.assertTrue(all(all(part for part in key) for key in groups['original']))

    def test_seed_draw_matches_shared_selection(self) -> None:
        with (ROOT / 'random_offset_selection.csv').open(encoding='utf-8', newline='') as stream:
            rows = tuple(csv.DictReader(stream))
        actual = {probe.identity(row): int(row['selected_target_lag_ms']) for row in rows}
        expected = np.random.default_rng(1701).choice(np.asarray(probe.TARGETS), size=96)
        self.assertEqual(tuple(actual[key] for key in sorted(actual)), tuple(int(value) for value in expected))

    def test_independent_rank_and_hierarchical_aggregation(self) -> None:
        with probe.SOURCE.open(encoding='utf-8', newline='') as stream:
            rows = tuple(csv.DictReader(stream))
        actual = json.loads((ROOT / 'random_offset_metrics.json').read_text(encoding='utf-8'))
        for model in MODELS:
            selected = [row for row in rows if row['model_id'] == model]
            speaker_scores = []
            for speaker in sorted({row['speaker_alias'] for row in selected}):
                records = []
                for recording in sorted({row['recording_alias'] for row in selected if row['speaker_alias'] == speaker}):
                    windows = []
                    for row in selected:
                        if row['speaker_alias'] != speaker or row['recording_alias'] != recording:
                            continue
                        differences = np.array([float(row['score_center']) - probe.score(row, lag) for lag in probe.TARGETS])
                        windows.append((float(np.mean((differences > 1e-6) + 0.5 * (np.abs(differences) <= 1e-6))), float(np.mean(differences > 1e-6)), float(np.mean(differences))))
                    records.append(np.mean(windows, axis=0))
                speaker_scores.append(np.mean(records, axis=0))
            result = np.mean(speaker_scores, axis=0)
            for index, metric in enumerate(('rank_win_tie_half', 'strict_center_greater', 'mean_margin')):
                self.assertAlmostEqual(float(result[index]), actual['exact_uniform_expectation'][model]['all_12'][metric], delta=1e-12)


if __name__ == '__main__':
    unittest.main()