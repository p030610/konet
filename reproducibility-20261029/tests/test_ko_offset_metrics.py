from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ko_offset_eval.metrics import DISTANCES_MS, OffsetError, analyze, read_raw_jsonl
from ko_offset_eval.models import OffsetSelection, Recording
from ko_offset_eval.reporting import write_outputs


ROOT = Path(__file__).resolve().parents[1]


def _recording(recording_id: str, speaker_id: str, windows: tuple[str, ...] = ("w1",)) -> Recording:
    return Recording(recording_id, speaker_id, "included", windows)


def _offset(recording_id: str, offset_ms: int | None = 0, status: str = "valid") -> OffsetSelection:
    return OffsetSelection(recording_id, "A", offset_ms, status)


def _rows(
    recording_id: str,
    window_id: str,
    *,
    model_id: str = "original",
    center: float = 0.9,
    left: float = 0.2,
    right: float = 0.2,
    distance: int = 40,
) -> list[dict[str, str | int | float | None]]:
    return [
        {"recording_id": recording_id, "window_id": window_id, "lag_ms": lag, "score_status": "valid", "score": score, "model_id": model_id, "fold": "B", "direction": "A_to_B", "evaluation_mode": "diagnostic_score"}
        for lag, score in ((0, center), (-distance, left), (distance, right))
    ]


def test_four_outcomes_margin_and_ties_are_hand_computed() -> None:
    # Given: four held-out windows with joint win, negative-only, positive-only, and both-side failures.
    rows = _rows("r1", "w1")
    rows += _rows("r1", "w2", left=0.9 - 5e-7)
    rows += _rows("r1", "w3", right=0.95)
    rows += _rows("r1", "w4", left=1.0, right=1.1)
    recording = _recording("r1", "s1", ("w1", "w2", "w3", "w4"))
    # When: the strict epsilon joint metric is aggregated.
    result = analyze(rows, (recording,), (_offset("r1"),), bootstrap_samples=40)
    metric = result.models["original"].per_distance[40]
    # Then: ties fail their side, categories stay mutually exclusive, and margin is never replaced by zero.
    assert metric.conditional_success.estimate == pytest.approx(0.25)
    assert metric.outcomes == {"neither_failure": 1, "negative_only_failure": 1, "positive_only_failure": 1, "both_failures": 1}
    assert metric.tie_comparisons == 1
    assert metric.tie_rate == pytest.approx(1 / 8)
    assert metric.any_tie_windows == 1
    assert metric.any_tie_window_rate == pytest.approx(1 / 4)
    assert metric.negative_failure_rate.estimate == pytest.approx(0.5)
    assert metric.positive_failure_rate.estimate == pytest.approx(0.5)
    assert metric.margin.estimate == pytest.approx((0.7 + 5e-7 - 0.05 - 0.2) / 4)


def test_incomplete_records_are_zero_only_for_end_to_end_and_common_support_is_intersection() -> None:
    # Given: one usable record and one record whose calibration has no selected offset.
    rows = [item for distance in DISTANCES_MS for item in _rows("r1", "w1", distance=distance)]
    recordings = (_recording("r1", "s1"), _recording("r2", "s2"))
    offsets = (_offset("r1"), _offset("r2", None, "tied"))
    # When: all-recording and conditional results are both emitted.
    result = analyze(rows, recordings, offsets, bootstrap_samples=40)
    at40 = result.models["original"].per_distance[40]
    # Then: r2 remains in the all-276-style denominator but cannot fabricate a valid margin.
    assert at40.conditional_success.estimate == pytest.approx(1.0)
    assert at40.end_to_end_success.estimate == pytest.approx(0.5)
    assert at40.margin.estimate == pytest.approx(0.7)
    assert at40.support.available_records == 1
    assert result.models["original"].common_support_records == 1


def test_common_support_requires_the_same_held_out_windows_at_every_distance() -> None:
    # Given: each distance has support for the record, but the two distances use different B windows.
    rows = _rows("r1", "w1", distance=40) + _rows("r1", "w2", distance=80)
    recordings = (_recording("r1", "s1", ("w1", "w2")),)
    # When: common support is computed over every approved distance and window identity.
    result = analyze(rows, recordings, (_offset("r1"),), bootstrap_samples=40)
    # Then: record overlap alone cannot fabricate a common-support cohort.
    assert result.models["original"].common_support_records == 0


def test_record_then_speaker_equal_weighting_is_not_window_weighted() -> None:
    # Given: speaker A has a successful short record and failed long record; speaker B has one tied record.
    rows = _rows("a1", "w1")
    rows += _rows("a2", "w1", center=0.2, left=0.9, right=0.9)
    rows += _rows("a2", "w2", center=0.2, left=0.9, right=0.9)
    rows += _rows("a2", "w3", center=0.2, left=0.9, right=0.9)
    rows += _rows("b1", "w1", left=0.9 - 5e-7, right=0.2)
    recordings = (_recording("a1", "A"), _recording("a2", "A", ("w1", "w2", "w3")), _recording("b1", "B"))
    offsets = tuple(_offset(item.recording_id) for item in recordings)
    # When: success is nested window -> recording -> speaker.
    result = analyze(rows, recordings, offsets, bootstrap_samples=40)
    # Then: A=(1+0)/2, B=0, equal-speaker estimate=.25 rather than the window-weighted .2.
    assert result.models["original"].per_distance[40].conditional_success.estimate == pytest.approx(0.25)


def test_conflicting_duplicate_raw_observation_is_rejected(tmp_path: Path) -> None:
    # Given: duplicate score keys with incompatible values in an append-only raw ledger.
    import json

    rows = _rows("r1", "w1")
    duplicate = dict(rows[0])
    duplicate["score"] = 0.1
    path = tmp_path / "raw.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in (*rows, duplicate)) + "\n", encoding="utf-8")
    # When: the raw reader validates the ledger boundary.
    # Then: a conflicting duplicate cannot silently choose an arbitrary score.
    with pytest.raises(OffsetError, match="conflicting duplicate"):
        read_raw_jsonl(path)


def test_each_required_distance_is_present() -> None:
    # Given: the approved distance grid.
    # When: the public metric constant is inspected.
    # Then: all six requested temporal distances are fixed and score-subset free.
    assert DISTANCES_MS == (40, 80, 120, 160, 240, 320)


def test_report_cli_writes_korean_markdown_json_csv_and_svg(tmp_path: Path) -> None:
    # Given: an append-only original-model fixture and canonical recording/offset metadata.
    rows = [item for distance in DISTANCES_MS for item in _rows("r1", "w1", distance=distance)]
    raw, recordings, offsets, output = (tmp_path / "raw.jsonl", tmp_path / "recordings.jsonl", tmp_path / "offsets.jsonl", tmp_path / "reports")
    raw.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")
    recordings.write_text(json.dumps({"recording_id": "r1", "speaker_id": "C282", "decision_status": "included", "include_in_all276": True, "a_window_ids": ["a1", "a2"], "b_window_ids": ["w1"]}) + "\n", encoding="utf-8")
    offsets.write_text(json.dumps({"recording_id": "r1", "fold": "A", "selected_offset_ms": 0, "status": "valid"}) + "\n", encoding="utf-8")
    # When: the real report command consumes only CPU fixture files.
    completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "report_korean_offsets.py"), "--raw", str(raw), "--recordings", str(recordings), "--offsets", str(offsets), "--output", str(output), "--bootstrap-samples", "40"], check=False, capture_output=True, text=True)
    # Then: all required final artifact formats exist and Korean report text avoids physical-GT claims.
    assert completed.returncode == 0, completed.stderr
    assert (output / "report.md").is_file()
    assert (output / "metrics.json").is_file()
    assert (output / "metrics.csv").is_file()
    assert (output / "heldout_curve.svg").read_text(encoding="utf-8").startswith("<svg")
    assert "물리적 동기화 정답" in (output / "report.md").read_text(encoding="utf-8")


def test_paired_fine_comparison_uses_only_matched_original_test_windows() -> None:
    # Given: original rows include one train speaker while a fine model has only an approved test speaker.
    rows = [item for distance in DISTANCES_MS for item in _rows("test", "w1", distance=distance)]
    rows += [item for distance in DISTANCES_MS for item in _rows("train", "w1", distance=distance)]
    rows += [item for distance in DISTANCES_MS for item in _rows("test", "w1", model_id="fine", distance=distance, center=0.2, left=0.9, right=0.9)]
    recordings = (_recording("test", "C282"), _recording("train", "C999"))
    offsets = (_offset("test"), _offset("train"))
    # When: primary and fine-model summaries are emitted together.
    result = analyze(rows, recordings, offsets, bootstrap_samples=40)
    paired = result.paired_comparisons["fine"]
    # Then: train-original rows remain in primary only; paired arms share the single test B window.
    assert result.models["original"].per_distance[40].support.all_recordings == 2
    assert paired.common_support_records == 1
    assert paired.original.per_distance[40].conditional_success.estimate == pytest.approx(1.0)
    assert paired.fine.per_distance[40].conditional_success.estimate == pytest.approx(0.0)


def test_final_analysis_rejects_included_recordings_without_an_offset_decision() -> None:
    # Given: a claimed included record has B scores but no frozen A-fold decision row.
    rows = _rows("r1", "w1")
    # When: final analysis is invoked without its calibration decision.
    # Then: a partial run cannot masquerade as an all-record end-to-end report.
    with pytest.raises(OffsetError, match="missing A-fold offset decisions"):
        analyze(rows, (_recording("r1", "C282"),), (), bootstrap_samples=40)


def test_report_writes_separate_common_support_panels_for_primary_and_paired_arms(tmp_path: Path) -> None:
    # Given: all-record original data and a fine arm limited to one approved test speaker.
    rows = [item for distance in DISTANCES_MS for item in _rows("test", "w1", distance=distance)]
    rows += [item for distance in DISTANCES_MS for item in _rows("train", "w1", distance=distance)]
    rows += [item for distance in DISTANCES_MS for item in _rows("test", "w1", model_id="fine", distance=distance, center=0.2, left=0.9, right=0.9)]
    recordings = (_recording("test", "C282"), _recording("train", "C999"))
    analysis = analyze(rows, recordings, (_offset("test"), _offset("train")), bootstrap_samples=40)
    # When: output files are rendered from the fully paired analysis.
    write_outputs(analysis, tmp_path / "reports")
    primary = (tmp_path / "reports" / "heldout_curve.svg").read_text(encoding="utf-8")
    paired = (tmp_path / "reports" / "paired_original_vs_fine.svg").read_text(encoding="utf-8")
    # Then: no all-record/fine overlay exists, axes are explicit, and paired denominator is visible.
    assert "Original all-record common support" in primary
    assert "n=2 records" in primary
    assert ">0.0<" in primary and ">1.0<" in primary
    assert "fine" not in primary
    assert "Paired original vs fine common support" in paired
    assert "n=1 records" in paired


def test_analysis_reports_calibration_statuses_and_lowest_80ms_margin_examples() -> None:
    # Given: two A-fold outcomes and one valid recording with two complete held-out B windows.
    rows = [
        item
        for distance in DISTANCES_MS
        for item in (
            _rows("r1", "w1", distance=distance, center=0.8, left=0.5, right=0.4)
            if distance != 80
            else _rows("r1", "w1", distance=distance, center=0.8, left=0.7999995, right=0.4)
        )
    ]
    rows += [item for distance in DISTANCES_MS for item in _rows("r1", "w2", distance=distance, center=0.8, left=0.2, right=0.1)]
    rows += [item for distance in DISTANCES_MS for item in _rows("r2", "w3", distance=distance)]
    recordings = (_recording("r1", "C282", ("w1", "w2")), _recording("r2", "C371", ("w3",)))
    offsets = (_offset("r1"), _offset("r2", None, "tied"))

    # When: the all-record analysis is assembled.
    result = analyze(rows, recordings, offsets, bootstrap_samples=40)

    # Then: every calibration status remains countable and the lowest 80ms record example is deterministic.
    assert result.calibration_status_counts == {"valid": 1, "tied": 1}
    assert [(item.recording_id, item.window_id, item.minimum_margin) for item in result.failure_examples_80ms] == [("r1", "w1", pytest.approx(5e-7))]


def test_svg_uses_linear_distance_positions_precise_quarter_ticks_and_fitting_long_legends(tmp_path: Path) -> None:
    # Given: original and both production fine-model IDs on the same approved test B-window.
    rows = [item for distance in DISTANCES_MS for item in _rows("r1", "w1", distance=distance)]
    for model_id in ("konet_full_epoch9", "konet_lora_r8a16_epoch5"):
        rows += [item for distance in DISTANCES_MS for item in _rows("r1", "w1", model_id=model_id, distance=distance)]
    analysis = analyze(rows, (_recording("r1", "C282"),), (_offset("r1"),), bootstrap_samples=40)

    # When: all primary and paired figures are emitted.
    write_outputs(analysis, tmp_path / "reports")
    paired = (tmp_path / "reports" / "paired_original_vs_konet_lora_r8a16_epoch5.svg").read_text(encoding="utf-8")

    # Then: millisecond geometry is linear, quarter labels are precise, and the reserved legend fits both IDs.
    for coordinate in ("86.0", "177.4", "268.9", "360.3", "543.1", "726.0"):
        assert f'x1="{coordinate}" y1="344"' in paired
    assert ">0.25<" in paired and ">0.75<" in paired
    assert '<rect x="478" y="60" width="240" height="40"' in paired
    assert "konet_lora_r8a16_epoch5" in paired


def test_report_keeps_primary_and_paired_confidence_intervals_visible(tmp_path: Path) -> None:
    # Given: original and fine rows on a common test window.
    rows = [item for distance in DISTANCES_MS for item in _rows("r1", "w1", distance=distance)]
    rows += [item for distance in DISTANCES_MS for item in _rows("r1", "w1", model_id="konet_full_epoch9", distance=distance)]
    analysis = analyze(rows, (_recording("r1", "C282"),), (_offset("r1"),), bootstrap_samples=40)

    # When: report output is written.
    write_outputs(analysis, tmp_path / "reports")
    report = (tmp_path / "reports" / "report.md").read_text(encoding="utf-8")

    # Then: the three primary estimates and paired margin/positive-rate fields remain visible.
    assert "Primary estimates (95% CI)" in report
    assert "conditional both-win (95% CI)" in report
    assert "end-to-end both-win (95% CI)" in report
    assert "original minimum margin (95% CI)" in report
    assert "fine positive failure rate" in report
