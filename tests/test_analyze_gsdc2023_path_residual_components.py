from __future__ import annotations

import json

import pandas as pd

from experiments.analyze_gsdc2023_path_residual_components import analyze_path_residual_components, main


TRIP = "trip-a/pixel5"
ONE_METER_DEG = 1.0 / 111320.0


def _write_row_summary(path) -> None:
    pd.DataFrame(
        {
            "tripId": [TRIP] * 6,
            "UnixTimeMillis": [1000, 2000, 3000, 4000, 5000, 6000],
            "epoch_index": [0, 1, 2, 3, 4, 5],
            "LatitudeDegrees_reference": [ONE_METER_DEG] * 6,
            "LongitudeDegrees_reference": [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
        },
    ).to_csv(path, index=False)


def _write_bridge_rows(path) -> None:
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 2000, 3000, 4000, 5000, 6000],
            "FgoLatitudeDegrees": [0.0] * 6,
            "FgoLongitudeDegrees": [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
            "BaselineLatitudeDegrees": [0.0] * 6,
            "BaselineLongitudeDegrees": [-0.0001, 0.0, 0.0001, 0.0002, 0.0003, 0.0004],
        },
    ).to_csv(path, index=False)


def test_analyze_path_residual_components_detects_lateral_bias(tmp_path) -> None:
    row_summary = tmp_path / "rows.csv"
    bridge_rows = tmp_path / "bridge.csv"
    _write_row_summary(row_summary)
    _write_bridge_rows(bridge_rows)

    rows, component_summary, summary = analyze_path_residual_components(
        row_summary=row_summary,
        bridge_rows=bridge_rows,
        chunk_epochs=3,
    )

    fgo_overall = component_summary[
        (component_summary["source"] == "fgo") & (component_summary["chunk_start_epoch"] == -1)
    ].iloc[0]
    assert abs(float(fgo_overall["median_tangent_residual_m"])) < 0.05
    assert 0.9 < float(fgo_overall["median_normal_residual_m"]) < 1.1
    assert float(fgo_overall["distance_after_normal_median_p95_m"]) < 0.05
    assert rows["fgo_abs_normal_residual_m"].between(0.9, 1.1).all()
    assert summary["rows"] == 6


def test_analyze_path_residual_components_cli_writes_outputs(tmp_path, capsys) -> None:
    row_summary = tmp_path / "rows.csv"
    bridge_rows = tmp_path / "bridge.csv"
    output = tmp_path / "out"
    _write_row_summary(row_summary)
    _write_bridge_rows(bridge_rows)

    assert (
        main(
            [
                "--row-summary",
                str(row_summary),
                "--bridge-rows",
                str(bridge_rows),
                "--chunk-epochs",
                "3",
                "--output-dir",
                str(output),
            ],
        )
        == 0
    )

    assert "analyzed: 6 row(s)" in capsys.readouterr().out
    assert (output / "path_residual_component_rows.csv").is_file()
    assert (output / "path_residual_component_summary.csv").is_file()
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert payload["rows"] == 6
