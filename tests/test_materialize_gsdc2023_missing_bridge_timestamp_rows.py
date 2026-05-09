from __future__ import annotations

import json

import pandas as pd

from experiments.materialize_gsdc2023_missing_bridge_timestamp_rows import (
    main,
    materialize_missing_bridge_timestamp_rows,
)


def test_materialize_missing_bridge_timestamp_rows_uses_nearest_selected(tmp_path) -> None:
    bridge_dir = tmp_path / "bridges/course/phone"
    bridge_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 3000],
            "LatitudeDegrees": [10.0, 30.0],
            "LongitudeDegrees": [100.0, 300.0],
        },
    ).to_csv(bridge_dir / "bridge_positions.csv", index=False)
    submission = pd.DataFrame(
        {
            "tripId": ["course/phone"] * 4,
            "UnixTimeMillis": [1000, 1500, 2500, 3000],
            "LatitudeDegrees": [0.0] * 4,
            "LongitudeDegrees": [0.0] * 4,
        },
    )

    rows, summary = materialize_missing_bridge_timestamp_rows(
        submission=submission,
        bridge_root=tmp_path / "bridges",
    )

    assert rows["UnixTimeMillis"].tolist() == [1500, 2500]
    assert rows["best_source_latitude_degrees"].tolist() == [10.0, 30.0]
    assert rows["best_source_longitude_degrees"].tolist() == [100.0, 300.0]
    assert rows["materialized_source"].tolist() == ["nearest_selected_previous", "nearest_selected_next"]
    assert summary["rows"] == 2
    assert summary["trips"] == 1


def test_materialize_missing_bridge_timestamp_rows_tie_uses_previous(tmp_path) -> None:
    bridge_dir = tmp_path / "bridges/course/phone"
    bridge_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 3000],
            "LatitudeDegrees": [10.0, 30.0],
            "LongitudeDegrees": [100.0, 300.0],
        },
    ).to_csv(bridge_dir / "bridge_positions.csv", index=False)
    submission = pd.DataFrame(
        {
            "tripId": ["course/phone"],
            "UnixTimeMillis": [2000],
            "LatitudeDegrees": [0.0],
            "LongitudeDegrees": [0.0],
        },
    )

    rows, _ = materialize_missing_bridge_timestamp_rows(
        submission=submission,
        bridge_root=tmp_path / "bridges",
    )

    assert rows["best_source_latitude_degrees"].tolist() == [10.0]
    assert rows["materialized_source"].tolist() == ["nearest_selected_previous"]


def test_materialize_missing_bridge_timestamp_rows_cli_writes_outputs(tmp_path, capsys) -> None:
    bridge_dir = tmp_path / "bridges/course/phone"
    bridge_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000],
            "LatitudeDegrees": [10.0],
            "LongitudeDegrees": [100.0],
        },
    ).to_csv(bridge_dir / "bridge_positions.csv", index=False)
    submission = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "tripId": ["course/phone", "course/phone"],
            "UnixTimeMillis": [1000, 2000],
            "LatitudeDegrees": [0.0, 0.0],
            "LongitudeDegrees": [0.0, 0.0],
        },
    ).to_csv(submission, index=False)

    assert (
        main(
            [
                "--submission",
                str(submission),
                "--bridge-root",
                str(tmp_path / "bridges"),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        == 0
    )

    assert "materialized missing bridge timestamps: rows=1 trips=1" in capsys.readouterr().out
    rows = pd.read_csv(tmp_path / "out/missing_bridge_timestamp_rows.csv")
    assert rows["best_source_latitude_degrees"].tolist() == [10.0]
    payload = json.loads((tmp_path / "out/summary.json").read_text(encoding="utf-8"))
    assert payload["rows"] == 1
