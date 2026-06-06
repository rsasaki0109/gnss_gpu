from __future__ import annotations

import json

from experiments.run_gsdc2023_taroz_full_parity_gate import (
    DEFAULT_THRESHOLDS,
    default_summary_specs,
    main,
    run_gate,
)


def _summary(position_max: float, clock_max: float) -> dict[str, object]:
    return {
        "mode": "taroz_imu_state",
        "native_imu_state_path": "native.csv",
        "matlab_imu_state_path": "matlab.csv",
        "delta_stats": {
            "matched_rows": 2,
            "groups": {
                "position_m": {
                    "finite_rows": 2,
                    "component_rms": position_max / 3.0,
                    "component_max_abs": position_max / 2.0,
                    "mean_norm": position_max / 2.0,
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


def test_default_summary_specs_are_labeled() -> None:
    specs = default_summary_specs()

    assert len(specs) == 3
    assert specs[0].startswith("pixel5=experiments/results/")
    assert set(DEFAULT_THRESHOLDS) >= {"position_m.max_norm=0.002", "clock_bias_m.max_norm=0.01"}


def test_run_gate_writes_outputs_and_passes_thresholds(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output_csv = tmp_path / "summary.csv"
    output_json = tmp_path / "summary.json"
    first.write_text(json.dumps(_summary(0.4, 0.2)), encoding="utf-8")
    second.write_text(json.dumps(_summary(0.5, 0.1)), encoding="utf-8")

    payload = run_gate(
        summaries=[f"pixel5={first}", f"pixel6pro={second}"],
        thresholds=["position_m.max_norm=0.6", "clock_bias_m.max_norm=0.3"],
        output_csv=output_csv,
        output_json=output_json,
    )

    assert payload["passed"] is True
    assert payload["summary_count"] == 2
    assert output_csv.is_file()
    assert json.loads(output_json.read_text(encoding="utf-8")) == payload


def test_main_returns_failure_when_gate_fails(tmp_path, capsys) -> None:
    summary = tmp_path / "summary.json"
    output_csv = tmp_path / "out.csv"
    output_json = tmp_path / "out.json"
    summary.write_text(json.dumps(_summary(0.4, 0.2)), encoding="utf-8")

    status = main(
        [
            f"--summary=pixel5={summary}",
            "--threshold=position_m.max_norm=0.3",
            f"--output-csv={output_csv}",
            f"--output-json={output_json}",
        ]
    )

    assert status == 2
    assert "passed=False" in capsys.readouterr().out
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["threshold_violations"][0]["group"] == "position_m"
