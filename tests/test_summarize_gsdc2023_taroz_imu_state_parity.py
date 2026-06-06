from __future__ import annotations

import argparse
import csv
import json

import pytest

from experiments.summarize_gsdc2023_taroz_imu_state_parity import (
    aggregate_summary_rows,
    load_summary_rows,
    main,
    parse_summary_spec,
    parse_threshold_spec,
    run,
    summary_row,
    threshold_violations,
)


def _summary(position_mean: float, position_max: float, clock_max: float) -> dict[str, object]:
    return {
        "mode": "taroz_imu_state",
        "native_imu_state_path": "native.csv",
        "matlab_imu_state_path": "matlab.csv",
        "delta_stats": {
            "matched_rows": 2,
            "groups": {
                "position_m": {
                    "finite_rows": 2,
                    "component_rms": position_mean / 2.0,
                    "component_max_abs": position_max / 2.0,
                    "mean_norm": position_mean,
                    "max_norm": position_max,
                },
                "clock_bias_m": {
                    "finite_rows": 2,
                    "component_rms": clock_max / 3.0,
                    "component_max_abs": clock_max,
                    "mean_norm": clock_max / 2.0,
                    "max_norm": clock_max,
                },
            },
        },
    }


def test_parse_summary_spec_accepts_labeled_and_bare_paths() -> None:
    label, path = parse_summary_spec("pixel5=results/pixel5.json")
    assert label == "pixel5"
    assert path.as_posix() == "results/pixel5.json"

    label, path = parse_summary_spec("results/native_summary.json")
    assert label == "native_summary"
    assert path.as_posix() == "results/native_summary.json"

    with pytest.raises(ValueError, match="empty label"):
        parse_summary_spec("=bad.json")


def test_parse_threshold_spec_validates_group_metric_and_limit() -> None:
    threshold = parse_threshold_spec("position_m.max_norm=0.002")

    assert threshold == {"group": "position_m", "metric": "max_norm", "threshold": 0.002}
    with pytest.raises(ValueError, match="group.metric"):
        parse_threshold_spec("position_m=0.1")
    with pytest.raises(ValueError, match="unknown threshold group"):
        parse_threshold_spec("bad.max_norm=0.1")
    with pytest.raises(ValueError, match="unknown threshold metric"):
        parse_threshold_spec("position_m.finite_rows=100")
    with pytest.raises(ValueError, match="non-negative"):
        parse_threshold_spec("position_m.max_norm=-1")


def test_summary_row_flattens_groups_and_preserves_missing_metrics(tmp_path) -> None:
    path = tmp_path / "pixel5.json"
    row = summary_row("pixel5", path, _summary(0.1, 0.4, 0.2))

    assert row["label"] == "pixel5"
    assert row["matched_rows"] == 2
    assert row["position_m_mean_norm"] == pytest.approx(0.1)
    assert row["position_m_max_norm"] == pytest.approx(0.4)
    assert row["clock_bias_m_component_max_abs"] == pytest.approx(0.2)
    assert row["rpy_rad_mean_norm"] is None


def test_load_summary_rows_and_aggregate_reports_worst_values(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_summary(0.1, 0.4, 0.2)), encoding="utf-8")
    second.write_text(json.dumps(_summary(0.3, 0.5, 0.1)), encoding="utf-8")

    rows = load_summary_rows([f"pixel5={first}", f"pixel6pro={second}"])
    aggregate = aggregate_summary_rows(rows)

    assert aggregate["summary_count"] == 2
    assert aggregate["labels"] == ["pixel5", "pixel6pro"]
    assert aggregate["groups"]["position_m"]["worst_mean_norm"] == {"label": "pixel6pro", "value": 0.3}
    assert aggregate["groups"]["position_m"]["worst_max_norm"] == {"label": "pixel6pro", "value": 0.5}
    assert aggregate["groups"]["clock_bias_m"]["worst_component_max_abs"] == {"label": "pixel5", "value": 0.2}
    assert aggregate["passed"] is True
    assert aggregate["threshold_violations"] == []


def test_threshold_violations_report_exceeded_and_missing_values(tmp_path) -> None:
    row = summary_row("pixel5", tmp_path / "pixel5.json", _summary(0.1, 0.4, 0.2))

    violations = threshold_violations(
        [row],
        [
            parse_threshold_spec("position_m.max_norm=0.3"),
            parse_threshold_spec("rpy_rad.mean_norm=1e-6"),
        ],
    )

    assert violations == [
        {
            "label": "pixel5",
            "summary_path": str(tmp_path / "pixel5.json"),
            "group": "position_m",
            "metric": "max_norm",
            "value": 0.4,
            "threshold": 0.3,
            "reason": "exceeded",
        },
        {
            "label": "pixel5",
            "summary_path": str(tmp_path / "pixel5.json"),
            "group": "rpy_rad",
            "metric": "mean_norm",
            "value": None,
            "threshold": 1e-6,
            "reason": "missing",
        },
    ]


def test_run_writes_csv_and_json(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output_csv = tmp_path / "summary.csv"
    output_json = tmp_path / "summary.json"
    first.write_text(json.dumps(_summary(0.1, 0.4, 0.2)), encoding="utf-8")
    second.write_text(json.dumps(_summary(0.3, 0.5, 0.1)), encoding="utf-8")

    aggregate = run(
        argparse.Namespace(
            summary=[f"pixel5={first}", f"pixel6pro={second}"],
            threshold=["position_m.max_norm=0.6", "clock_bias_m.component_max_abs=0.25"],
            output_csv=output_csv,
            output_json=output_json,
        )
    )

    written_aggregate = json.loads(output_json.read_text(encoding="utf-8"))
    assert written_aggregate == aggregate
    assert aggregate["passed"] is True
    assert aggregate["threshold_violations"] == []
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == ["pixel5", "pixel6pro"]
    assert float(rows[1]["position_m_max_norm"]) == pytest.approx(0.5)


def test_main_exits_nonzero_when_threshold_fails(tmp_path, capsys) -> None:
    first = tmp_path / "first.json"
    first.write_text(json.dumps(_summary(0.1, 0.4, 0.2)), encoding="utf-8")

    with pytest.raises(SystemExit, match="parity gate failed: 1 violation"):
        main([f"--summary=pixel5={first}", "--threshold=position_m.max_norm=0.3"])

    output = json.loads(capsys.readouterr().out)
    assert output["passed"] is False
    assert output["threshold_violations"][0]["label"] == "pixel5"
