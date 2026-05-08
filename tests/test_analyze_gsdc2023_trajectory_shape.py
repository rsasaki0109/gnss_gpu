from __future__ import annotations

import json

import pandas as pd

from experiments.analyze_gsdc2023_trajectory_shape import analyze_trajectory_shape, main


TRIP = "trip-a/pixel5"


def _write_row_summary(path) -> None:
    pd.DataFrame(
        {
            "tripId": [TRIP] * 6,
            "UnixTimeMillis": [1000, 2000, 3000, 4000, 5000, 6000],
            "epoch_index": [0, 1, 2, 3, 4, 5],
            "LatitudeDegrees_reference": [0.0] * 6,
            "LongitudeDegrees_reference": [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
        },
    ).to_csv(path, index=False)


def _write_bridge_rows(path) -> None:
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 2000, 3000, 4000, 5000, 6000],
            "BaselineLatitudeDegrees": [0.0] * 6,
            "BaselineLongitudeDegrees": [-0.0001, 0.0, 0.0001, 0.0002, 0.0003, 0.0004],
            "RawWlsLatitudeDegrees": [0.0] * 6,
            "RawWlsLongitudeDegrees": [0.0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
        },
    ).to_csv(path, index=False)


def test_analyze_trajectory_shape_detects_source_lag(tmp_path) -> None:
    row_summary = tmp_path / "rows.csv"
    bridge_rows = tmp_path / "bridge.csv"
    _write_row_summary(row_summary)
    _write_bridge_rows(bridge_rows)

    rows, lag_summary, chunk_lag_summary, summary = analyze_trajectory_shape(
        row_summary=row_summary,
        bridge_rows=bridge_rows,
        chunk_epochs=3,
        max_lag_epochs=2,
    )

    baseline_best = lag_summary[lag_summary["source"] == "baseline"].sort_values("distance_p95_m").iloc[0]
    raw_best = lag_summary[lag_summary["source"] == "raw_wls"].sort_values("distance_p95_m").iloc[0]
    assert int(baseline_best["lag_epochs"]) == 1
    assert int(raw_best["lag_epochs"]) == 0
    assert rows.loc[1, "raw_wls_distance_m"] < 1e-6
    assert set(chunk_lag_summary["chunk_start_epoch"]) == {0, 3}
    assert summary["rows"] == 6


def test_analyze_trajectory_shape_cli_writes_outputs(tmp_path, capsys) -> None:
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
                "--max-lag-epochs",
                "2",
                "--output-dir",
                str(output),
            ],
        )
        == 0
    )

    assert "analyzed: 6 row(s)" in capsys.readouterr().out
    assert (output / "trajectory_shape_rows.csv").is_file()
    assert (output / "trajectory_lag_summary.csv").is_file()
    assert (output / "trajectory_lag_chunks.csv").is_file()
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert payload["rows"] == 6
