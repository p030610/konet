Verified 2026-10-29; CPU only, no GPU model construction/inference.
Runtime: existing LatentSync virtual environment, Python 3.10.11 on Windows; observed package versions below.
Both scripts/eval_korean_offsets.py --help and scripts/report_korean_offsets.py --help: exit 0.
python -m pytest tests/test_ko_offset_metrics.py -q -p no:cacheprovider: 13 passed in 2.39s, exit 0.
Synthetic report command documented in README: exit 0; report.md, metrics.json, metrics.csv, heldout_curve.svg generated.
Observed key versions: numpy1.26.4; pydantic2.13.5; typing_extensions4.16.0; torch2.5.1+cu121; omegaconf2.3.0; pytest8.3.5.
Clean environment installation, actual AV decoding, baseline/LoRA weight loading and GPU parity were not verified by this packaging task.
Production Python files are copied without behavioral edits. Existing implementation style and platform limits are preserved.
Final documentation update: source confirms baseline ko_offset_eval/runner.py passes VisualPolicy.NORMALIZE_1280X720 to FrozenScorer; KoNet adapter_scoring.py uses identical explicit policy. This is source-policy verification, not a new GPU numerical experiment.
Added 48-record authored selection manifest with four speaker pseudonyms, hashed recording identifiers, A/B frame indices and calibration outcomes; 44 have valid calibration. No raw IDs/paths/transcripts/AV/source annotations were added.

Real score dataset: definitive recompute_heldout_scores.py executed on CPU, exit0. 288 rows / 96 paired windows / 44 recording aliases / 4 nonempty speaker aliases / 3 models.
Independent parity receipt verification/score-table-parity.json: all 18 model-distance cells, 36 strict-both-win/minimum-margin scalar checks agree with archived paired table within 1e-12.
Public animation inputs are separately verified by their own data worker; see public_animation_eval/verification/checks.json. No model performance on those four inputs is claimed.
