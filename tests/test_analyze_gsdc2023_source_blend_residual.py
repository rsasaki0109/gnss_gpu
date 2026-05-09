from __future__ import annotations

import json

import pandas as pd

from experiments.analyze_gsdc2023_source_blend_residual import (
    analyze_source_blend_residual,
    main,
)


TRIP = "trip-a/pixel5"


def _write_row_summary(path) -> None:
    pd.DataFrame(
        {
            "tripId": [TRIP, TRIP, TRIP],
            "UnixTimeMillis": [1000, 2000, 3000],
            "epoch_index": [0, 1, 2],
            "LatitudeDegrees_reference": [0.0, 0.0, 0.00025],
            "LongitudeDegrees_reference": [0.0005, 0.00025, 0.00025],
        },
    ).to_csv(path, index=False)


def _write_bridge_rows(path) -> None:
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 2000, 3000],
            "BaselineLatitudeDegrees": [0.0, 0.0, 0.0],
            "BaselineLongitudeDegrees": [0.0, 0.0, 0.0],
            "RawWlsLatitudeDegrees": [0.0, 0.0, 0.0],
            "RawWlsLongitudeDegrees": [0.001, 0.001, 0.001],
            "FgoLatitudeDegrees": [0.001, 0.001, 0.001],
            "FgoLongitudeDegrees": [0.0, 0.0, 0.0],
        },
    ).to_csv(path, index=False)


def test_analyze_source_blend_residual_detects_segment_and_triangle(tmp_path) -> None:
    row_summary = tmp_path / "rows.csv"
    bridge_rows = tmp_path / "bridge.csv"
    _write_row_summary(row_summary)
    _write_bridge_rows(bridge_rows)

    rows, chunks, summary = analyze_source_blend_residual(
        row_summary=row_summary,
        bridge_rows=bridge_rows,
        chunk_epochs=2,
    )

    assert rows.loc[0, "best_segment_distance_m"] < 1e-6
    assert rows.loc[0, "best_segment_pair"] == "baseline-raw_wls"
    assert rows.loc[1, "best_hull_kind"] == "segment"
    assert rows.loc[2, "best_hull_kind"] == "triangle"
    assert rows.loc[2, "best_hull_distance_m"] < 1e-6
    assert int(chunks.loc[0, "rows"]) == 2
    assert summary["rows"] == 3


def test_analyze_source_blend_residual_cli_writes_outputs(tmp_path, capsys) -> None:
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
                "2",
                "--output-dir",
                str(output),
            ],
        )
        == 0
    )

    assert "analyzed: 3 row(s)" in capsys.readouterr().out
    assert (output / "source_blend_residual_rows.csv").is_file()
    assert (output / "source_blend_residual_chunks.csv").is_file()
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert payload["rows"] == 3
