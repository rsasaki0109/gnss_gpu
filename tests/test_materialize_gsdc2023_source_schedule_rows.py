from __future__ import annotations

import json

import pandas as pd

from experiments.materialize_gsdc2023_source_schedule_rows import (
    main,
    materialize_source_schedule_rows,
)


def _write_bridge(path, *, trip_id: str | None = None) -> None:
    payload = {
        "UnixTimeMillis": [1, 2],
        "BaselineLatitudeDegrees": [10.0, 20.0],
        "BaselineLongitudeDegrees": [30.0, 40.0],
        "FgoLatitudeDegrees": [11.0, 21.0],
        "FgoLongitudeDegrees": [31.0, 41.0],
    }
    if trip_id is not None:
        payload["tripId"] = [trip_id, trip_id]
    pd.DataFrame(payload).to_csv(path, index=False)


def test_materialize_multi_bridge_schedule_rows(tmp_path) -> None:
    ref = tmp_path / "ref.csv"
    local = tmp_path / "local.csv"
    _write_bridge(ref)
    _write_bridge(local)
    schedule = pd.DataFrame(
        {
            "tripId": ["trip/a", "trip/a"],
            "UnixTimeMillis": [1, 2],
            "best_bridge_source": ["ref:baseline", "local:fgo"],
        },
    )

    rows, summary = materialize_source_schedule_rows(
        schedule_rows=schedule,
        bridge_sources=[("ref", ref), ("local", local)],
        target_trip="trip/a",
    )

    assert rows["best_source_latitude_degrees"].tolist() == [10.0, 21.0]
    assert rows["best_source_longitude_degrees"].tolist() == [30.0, 41.0]
    assert rows["materialized_bridge_label"].tolist() == ["ref", "local"]
    assert rows["materialized_source"].tolist() == ["baseline", "fgo"]
    assert summary["materialized_source_counts"] == {"local:fgo": 1, "ref:baseline": 1}


def test_materialize_single_bridge_reference_source_schedule(tmp_path) -> None:
    bridge = tmp_path / "bridge.csv"
    _write_bridge(bridge, trip_id="trip/a")
    schedule = pd.DataFrame(
        {
            "tripId": ["trip/a", "trip/a"],
            "UnixTimeMillis": [1, 2],
            "best_reference_source": ["baseline", "fgo"],
        },
    )

    rows, summary = materialize_source_schedule_rows(
        schedule_rows=schedule,
        bridge_sources=[("bridge", bridge)],
        target_trip="trip/a",
    )

    assert rows["best_source_latitude_degrees"].tolist() == [10.0, 21.0]
    assert rows["best_source_longitude_degrees"].tolist() == [30.0, 41.0]
    assert summary["schedule_source_column"] == "best_reference_source"


def test_materialize_source_schedule_cli_writes_outputs(tmp_path, capsys) -> None:
    bridge = tmp_path / "bridge.csv"
    schedule = tmp_path / "schedule.csv"
    output = tmp_path / "out"
    _write_bridge(bridge)
    pd.DataFrame(
        {
            "tripId": ["trip/a"],
            "UnixTimeMillis": [1],
            "best_source": ["baseline"],
        },
    ).to_csv(schedule, index=False)

    assert (
        main(
            [
                "--schedule-rows",
                str(schedule),
                "--bridge-source",
                f"bridge={bridge}",
                "--target-trip",
                "trip/a",
                "--output-dir",
                str(output),
            ],
        )
        == 0
    )

    assert "materialized: rows=1" in capsys.readouterr().out
    rows = pd.read_csv(output / "materialized_source_schedule_rows.csv")
    assert rows["best_source_latitude_degrees"].tolist() == [10.0]
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert payload["materialized_source_counts"] == {"bridge:baseline": 1}
