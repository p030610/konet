# Deidentified held-out score-level evaluation dataset

This is a score-level evaluation dataset, not an audio-visual inference dataset. It contains no media, paths, source IDs, transcripts, identities, or training data.

- Rows: 288 = 96 B-windows x 3 completed score models.
- Models: `original`, `konet_full_epoch9`, `konet_lora_r8a16_epoch5`.
- Common support: 44 valid-calibration Korean held-out recordings and 4 deidentified speakers. The original selected cohort had 48 recordings; 4 boundary A-fold calibrations were excluded before the fixed paired intersection.
- Fields: aliases, frozen center lag, center cosine, and raw cosine at plus/minus 40/80/120/160/240/320 ms.
- The aliases are stable only within this released score table and cannot be joined to source media by this release.

`recompute_heldout_scores.py` is the definitive CPU evaluator. At every distance it reproduces the historical primary metrics from the 13 raw scores per row: strict both-win is `center > negative + 1e-6` and `center > positive + 1e-6`; mean minimum margin is the equal-speaker mean of `center - max(negative, positive)`. It also emits a secondary directional W: center versus one signed target, with a tie worth 0.5. The hierarchy is window -> recording -> equal speaker. These are model-score ranking measures, not physical synchronization ground truth, media accuracy, or SOTA evidence.

The evaluator rejects any non-finite score, duplicate/missing model-window key, or cardinality other than 288 rows, 96 paired windows, 44 recording aliases, 4 speaker aliases, and 3 models. Its output is `recomputed_heldout_metrics.json`.
