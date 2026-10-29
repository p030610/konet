from __future__ import annotations

import csv
import json
from dataclasses import asdict
from html import escape
from pathlib import Path

from .metrics import DISTANCES_MS
from .models import Analysis, DistanceMetric, Estimate, OffsetSelection, Recording


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    try:
        payload = tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSONL: {path}") from error
    if not all(isinstance(row, dict) for row in payload):
        raise RuntimeError("JSONL rows must be objects")
    return payload


def _text(row: dict[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"metadata {field} must be a nonempty string")
    return value


def read_recordings(path: Path) -> tuple[Recording, ...]:
    records: list[Recording] = []
    for row in _jsonl(path):
        windows = row.get("b_window_ids")
        if not isinstance(windows, list) or not all(isinstance(value, str) and value for value in windows):
            raise RuntimeError("recordings b_window_ids must be a nonempty string list")
        records.append(Recording(_text(row, "recording_id"), _text(row, "speaker_id"), _text(row, "decision_status"), tuple(windows)))
    if not records or len({item.recording_id for item in records}) != len(records):
        raise RuntimeError("recordings metadata is empty or duplicates recording_id")
    return tuple(records)


def read_offsets(path: Path) -> tuple[OffsetSelection, ...]:
    offsets: list[OffsetSelection] = []
    for row in _jsonl(path):
        offset = row.get("selected_offset_ms")
        if offset is not None and (isinstance(offset, bool) or not isinstance(offset, int)):
            raise RuntimeError("offset selected_offset_ms must be an integer or null")
        offsets.append(OffsetSelection(_text(row, "recording_id"), _text(row, "fold"), offset, _text(row, "status")))
    if not offsets or len({item.recording_id for item in offsets}) != len(offsets):
        raise RuntimeError("offset metadata is empty or duplicates recording_id")
    if any(item.fold != "A" for item in offsets):
        raise RuntimeError("offset metadata must contain only A-fold selections")
    return tuple(offsets)


def _number(value: float | None) -> str:
    return "사용 불가" if value is None else f"{value:.4f}"


def _estimate_with_ci(value: Estimate) -> str:
    if value.estimate is None or value.ci_low is None or value.ci_high is None:
        return "unavailable"
    return f"{value.estimate:.4f} [{value.ci_low:.4f}, {value.ci_high:.4f}]"


def _confidence_interval_tables(analysis: Analysis) -> tuple[str, ...]:
    output = ["", "## Primary estimates (95% CI)"]
    for model_id, model in analysis.models.items():
        output.extend(("", f"### {model_id}", "", "| distance (ms) | conditional both-win (95% CI) | end-to-end both-win (95% CI) | minimum margin (95% CI) |", "|---:|---:|---:|---:|"))
        for distance in DISTANCES_MS:
            metric = model.per_distance[distance]
            output.append(f"| {distance} | {_estimate_with_ci(metric.conditional_success)} | {_estimate_with_ci(metric.end_to_end_success)} | {_estimate_with_ci(metric.margin)} |")
    for fine_model_id, paired in analysis.paired_comparisons.items():
        output.extend(("", f"### Paired {fine_model_id}", "", "| distance (ms) | original minimum margin (95% CI) | fine minimum margin (95% CI) | original positive failure rate | fine positive failure rate |", "|---:|---:|---:|---:|---:|"))
        for distance in DISTANCES_MS:
            original = paired.original.per_distance[distance]
            fine = paired.fine.per_distance[distance]
            output.append(f"| {distance} | {_estimate_with_ci(original.margin)} | {_estimate_with_ci(fine.margin)} | {_number(original.positive_failure_rate.estimate)} | {_number(fine.positive_failure_rate.estimate)} |")
    return tuple(output)


def _markdown(analysis: Analysis) -> str:
    lines = ["# 한국어 원본-참조 held-out 오프셋 보고서", "", "이 보고서는 고정된 원본 참조 오프셋에서의 모델 점수 판별 결과다. 물리적 동기화 정답, 인과 효과, 또는 일반적 성능 주장을 제공하지 않는다."]
    lines.extend(("", "## A-fold 보정 상태", "", "| 상태 | 기록 수 |", "|---|---:|"))
    for status in ("valid", "boundary", "tied", "flat", "incomplete"):
        lines.append(f"| {status} | {analysis.calibration_status_counts.get(status, 0)} |")
    for model_id, model in analysis.models.items():
        all_records = model.per_distance[DISTANCES_MS[0]].support.all_recordings
        arm = f"원본 모델 · 전체 {all_records}개 기록 end-to-end" if model_id == "original" else "미세조정 비교 · 테스트 화자 C282/C371/C580/C613만"
        lines.extend(("", f"## {model_id}", "", arm, "", "| 거리(ms) | 조건부 strict both-win | 전체 end-to-end strict both-win | 최소 마진 | 음(-)측 실패율 (기록→화자) | 양(+)측 실패율 (기록→화자) | 지원 기록/전체 | 양측 성공 창 수 | 음(-)측만 실패 창 수 | 양(+)측만 실패 창 수 | 양방향 실패 창 수 | 비교 tie율 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"))
        for distance in DISTANCES_MS:
            metric = model.per_distance[distance]
            lines.append(f"| {distance} | {_number(metric.conditional_success.estimate)} | {_number(metric.end_to_end_success.estimate)} | {_number(metric.margin.estimate)} | {_number(metric.negative_failure_rate.estimate)} | {_number(metric.positive_failure_rate.estimate)} | {metric.support.available_records}/{metric.support.all_recordings} | {metric.outcomes['neither_failure']} | {metric.outcomes['negative_only_failure']} | {metric.outcomes['positive_only_failure']} | {metric.outcomes['both_failures']} | {_number(metric.tie_rate)} |")
        lines.extend(("", "실패율은 valid triplet 창을 기록 평균 후 화자 동등 가중한 조건부 비율이다. 네 outcome 열은 같은 valid 창의 상호배타적 원시 창 수이며, 실패율의 원시 분자는 음(-)측=음(-)측만+양방향 실패, 양(+)측=양(+)측만+양방향 실패다.", f"여섯 거리 모두 동일한 B-window를 가진 공통 지원 기록: {model.common_support_records}개. `heldout_curve.svg`는 원본 전체 공통 지원 패널이며, 미세조정 비교는 별도 paired SVG에서만 그린다."))
    for fine_model_id, paired in analysis.paired_comparisons.items():
        lines.extend(("", f"## 원본-미세조정 짝비교: {fine_model_id}", "", f"동일 테스트 화자와 동일 B-window의 공통 지원 기록 {paired.common_support_records}개만 사용했다. 원본 전체 결과와 이 표를 직접 비교하지 않는다.", "", "| 거리(ms) | 원본 both-win | 미세조정 both-win | 원본 음(-) 실패율 | 미세조정 음(-) 실패율 |", "|---:|---:|---:|---:|---:|"))
        for distance in DISTANCES_MS:
            original = paired.original.per_distance[distance]
            fine = paired.fine.per_distance[distance]
            lines.append(f"| {distance} | {_number(original.conditional_success.estimate)} | {_number(fine.conditional_success.estimate)} | {_number(original.negative_failure_rate.estimate)} | {_number(fine.negative_failure_rate.estimate)} |")
    lines.extend(("", "## 점수-정의 80 ms 저마진 예시", "", "아래는 물리적 동기화 오류가 아니라 고정된 원본 오프셋에서의 모델 점수 마진이 낮은 B-window 예시다. 기록당 최대 하나를 선택하고, 최소 마진·기록 ID·window ID 순으로 결정적으로 정렬했다.", "", "| 기록 ID | B-window ID | 선택 오프셋(ms) | center 점수 | -80 점수 | +80 점수 | 최소 마진 |", "|---|---|---:|---:|---:|---:|---:|"))
    for item in analysis.failure_examples_80ms:
        lines.append(f"| {item.recording_id} | {item.window_id} | {item.selected_offset_ms} | {item.center_score:.4f} | {item.negative_80_score:.4f} | {item.positive_80_score:.4f} | {item.minimum_margin:.4f} |")
    lines.extend(_confidence_interval_tables(analysis))
    return "\n".join(lines) + "\n"


def _segments(curve: dict[int, DistanceMetric]) -> tuple[tuple[tuple[int, float], ...], ...]:
    groups: list[tuple[tuple[int, float], ...]] = []
    current: list[tuple[int, float]] = []
    for distance in DISTANCES_MS:
        value = curve[distance].conditional_success.estimate
        if value is None:
            if current:
                groups.append(tuple(current))
                current = []
        else:
            current.append((distance, value))
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _svg(title: str, subtitle: str, curves: tuple[tuple[str, str, dict[int, DistanceMetric]], ...]) -> str:
    width, height, left, bottom = 780, 410, 86, 344
    plot_width, plot_height = 640, 240
    minimum_distance, maximum_distance = DISTANCES_MS[0], DISTANCES_MS[-1]
    x = {distance: left + (distance - minimum_distance) * plot_width / (maximum_distance - minimum_distance) for distance in DISTANCES_MS}
    y = lambda value: bottom - value * plot_height
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/><text x="{left}" y="32" font-family="sans-serif" font-size="18">{escape(title)}</text><text x="{left}" y="54" font-family="sans-serif" font-size="13">{escape(subtitle)}</text><line x1="{left}" y1="{bottom}" x2="{left + plot_width}" y2="{bottom}" stroke="#333"/><line x1="{left}" y1="{bottom - plot_height}" x2="{left}" y2="{bottom}" stroke="#333"/>']
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        tick_y = y(tick)
        label = f"{tick:.2f}" if tick in (0.25, 0.75) else f"{tick:.1f}"
        parts.append(f'<line x1="{left}" y1="{tick_y:.1f}" x2="{left + plot_width}" y2="{tick_y:.1f}" stroke="#ddd"/><text x="42" y="{tick_y + 4:.1f}" font-family="sans-serif" font-size="12">{label}</text>')
    for distance in DISTANCES_MS:
        tick_x = x[distance]
        parts.append(f'<line x1="{tick_x:.1f}" y1="{bottom}" x2="{tick_x:.1f}" y2="{bottom + 5}" stroke="#333"/><text x="{tick_x - 10:.1f}" y="{bottom + 24}" font-family="sans-serif" font-size="12">{distance}</text>')
    parts.append(f'<text x="{left + 270}" y="{bottom + 48}" font-family="sans-serif" font-size="13">distance (ms)</text><text x="14" y="{bottom - 110}" transform="rotate(-90 14 {bottom - 110})" font-family="sans-serif" font-size="13">strict both-win</text>')
    legend_x, legend_y, legend_width = 478, 60, 240
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="{4 + 18 * len(curves)}" fill="white" stroke="#999"/>')
    for index, (label, color, curve) in enumerate(curves):
        for group in _segments(curve):
            points = " ".join(f"{x[distance]:.1f},{y(value):.1f}" for distance, value in group)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
            for distance, value in group:
                parts.append(f'<circle cx="{x[distance]:.1f}" cy="{y(value):.1f}" r="4" fill="{color}"/>')
        legend_item_y = 73 + index * 18
        parts.append(f'<line x1="{legend_x + 10}" y1="{legend_item_y}" x2="{legend_x + 22}" y2="{legend_item_y}" stroke="{color}" stroke-width="3"/><text x="{legend_x + 28}" y="{legend_item_y + 4}" font-family="sans-serif" font-size="11" fill="{color}">{escape(label)}</text>')
    return "".join((*parts, "</svg>"))


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _csv_row(model_id: str, support: str, distance: int, metric: DistanceMetric) -> dict[str, object]:
    return {"model_id": model_id, "support": support, "distance_ms": distance, "conditional_success": metric.conditional_success.estimate, "end_to_end_success": metric.end_to_end_success.estimate, "minimum_margin": metric.margin.estimate, "negative_failure_rate_recording_speaker": metric.negative_failure_rate.estimate, "positive_failure_rate_recording_speaker": metric.positive_failure_rate.estimate, "available_records": metric.support.available_records, "all_recordings": metric.support.all_recordings, "tie_rate": metric.tie_rate, "any_tie_window_rate": metric.any_tie_window_rate, "negative_only_failure_windows": metric.outcomes["negative_only_failure"], "positive_only_failure_windows": metric.outcomes["positive_only_failure"], "both_failure_windows": metric.outcomes["both_failures"]}


def write_outputs(analysis: Analysis, output: Path) -> tuple[Path, ...]:
    output.mkdir(parents=True, exist_ok=True)
    primary_figure = output / "heldout_curve.svg"
    paired_figures = tuple(output / f"paired_original_vs_{_safe_name(fine_id)}.svg" for fine_id in analysis.paired_comparisons)
    paths = (output / "report.md", output / "metrics.json", output / "metrics.csv", primary_figure, *paired_figures)
    if any(path.exists() for path in paths):
        raise RuntimeError("report output already exists; refusing to overwrite")
    paths[0].write_text(_markdown(analysis), encoding="utf-8")
    paths[1].write_text(json.dumps(asdict(analysis), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fieldnames = ("model_id", "support", "distance_ms", "conditional_success", "end_to_end_success", "minimum_margin", "negative_failure_rate_recording_speaker", "positive_failure_rate_recording_speaker", "available_records", "all_recordings", "tie_rate", "any_tie_window_rate", "negative_only_failure_windows", "positive_only_failure_windows", "both_failure_windows")
    with paths[2].open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model_id, model in analysis.models.items():
            for support, curve in (("per_distance", model.per_distance), ("common_support", model.common_support_per_distance)):
                for distance, metric in curve.items():
                    writer.writerow(_csv_row(model_id, support, distance, metric))
        for fine_model_id, paired in analysis.paired_comparisons.items():
            for model_id, model in (("original", paired.original), (fine_model_id, paired.fine)):
                for distance, metric in model.per_distance.items():
                    writer.writerow(_csv_row(model_id, f"paired_common_support:{fine_model_id}", distance, metric))
    original = analysis.models.get("original")
    if original is None:
        raise RuntimeError("original model report is required for the primary figure")
    primary_figure.write_text(_svg("Original all-record common support", f"n={original.common_support_records} records; per-distance points are common-support only", (("original", "#1565c0", original.common_support_per_distance),)), encoding="utf-8")
    for fine_model_id, paired in analysis.paired_comparisons.items():
        output.joinpath(f"paired_original_vs_{_safe_name(fine_model_id)}.svg").write_text(_svg("Paired original vs fine common support", f"n={paired.common_support_records} records; same test speakers and B-windows", (("original", "#1565c0", paired.original.per_distance), (fine_model_id, "#d84315", paired.fine.per_distance))), encoding="utf-8")
    return paths
