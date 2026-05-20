from __future__ import annotations

import json
from pathlib import Path

from experiments.gsdc2023_ab_source_mix import (
    SourceCounts,
    aggregate_source_totals,
    diff_source_counts,
    load_bridge_source_counts,
    load_taroz_source_counts,
)


def _write_taroz_summary(path: Path, trip_metrics: list[dict]) -> None:
    payload = {"trip_metrics": trip_metrics}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_bridge_metrics(root: Path, trip: str, phone: str, payload: dict) -> Path:
    trip_dir = root / trip / phone
    trip_dir.mkdir(parents=True, exist_ok=True)
    out = trip_dir / "bridge_metrics.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return out


def test_source_counts_total_nonbaseline_excludes_baseline():
    counts = SourceCounts(trip_id="a", n_epochs=1000, baseline=800, fgo=100, fgo_no_tdcp=50, raw_wls=50)
    assert counts.total_nonbaseline == 200


def test_load_taroz_strips_train_prefix(tmp_path: Path):
    path = tmp_path / "summary.json"
    _write_taroz_summary(
        path,
        [
            {
                "trip": "train/2023-05-09-23-10-us-ca-sjc-r/sm-a505u",
                "n_epochs": 500,
                "selected_source_counts": {"baseline": 400, "fgo": 50, "fgo_dd_carrier": 50},
            },
            {
                "trip": "test/2022-04-04-16-31-us-ca-lax-x/pixel5",
                "n_epochs": 2171,
                "selected_source_counts": {"baseline": 1771, "fgo_dd_carrier": 400},
            },
        ],
    )
    rows = load_taroz_source_counts(path)
    assert [r.trip_id for r in rows] == [
        "2023-05-09-23-10-us-ca-sjc-r/sm-a505u",
        "2022-04-04-16-31-us-ca-lax-x/pixel5",
    ]
    assert rows[0].baseline == 400
    assert rows[0].fgo_dd_carrier == 50
    assert rows[1].fgo_dd_carrier == 400


def test_load_taroz_handles_missing_selected_source_counts(tmp_path: Path):
    path = tmp_path / "summary.json"
    _write_taroz_summary(path, [{"trip": "x/y", "n_epochs": 10}])
    rows = load_taroz_source_counts(path)
    assert rows == [SourceCounts(trip_id="x/y", n_epochs=10)]


def test_load_bridge_walks_per_trip_phone_dirs(tmp_path: Path):
    root = tmp_path / "bridge_root"
    _write_bridge_metrics(
        root,
        "2021-09-14-20-32-us-ca-mtv-k",
        "pixel4",
        {"n_epochs": 1274, "selected_source_counts": {"baseline": 1074, "fgo": 200}},
    )
    _write_bridge_metrics(
        root,
        "2022-02-23-17-46-us-ca-lax-n",
        "pixel5",
        {"n_epochs": 2407, "selected_source_counts": {"baseline": 2407}},
    )
    rows = load_bridge_source_counts(root)
    assert sorted(r.trip_id for r in rows) == [
        "2021-09-14-20-32-us-ca-mtv-k/pixel4",
        "2022-02-23-17-46-us-ca-lax-n/pixel5",
    ]


def test_load_bridge_ignores_misplaced_metrics_files(tmp_path: Path):
    root = tmp_path / "bridge_root"
    (root / "stray").mkdir(parents=True)
    (root / "stray" / "bridge_metrics.json").write_text("{}", encoding="utf-8")
    assert load_bridge_source_counts(root) == []


def test_diff_source_counts_aligns_on_trip_id_and_pads_missing():
    bridge = [
        SourceCounts(trip_id="a", n_epochs=100, baseline=100),
        SourceCounts(trip_id="b", n_epochs=200, baseline=180, fgo=20),
    ]
    taroz = [
        SourceCounts(trip_id="b", n_epochs=200, baseline=160, fgo=10, fgo_dd_carrier=30),
        SourceCounts(trip_id="c", n_epochs=50, baseline=40, fgo_no_tdcp=10),
    ]
    diffs = diff_source_counts(bridge, taroz)
    assert [d.trip_id for d in diffs] == ["a", "b", "c"]

    a = diffs[0]
    assert a.d_baseline == -100  # taroz missing => zero - 100
    assert a.n_epochs_bridge == 100
    assert a.n_epochs_taroz == 0

    b = diffs[1]
    assert b.d_baseline == -20
    assert b.d_fgo == -10
    assert b.d_fgo_dd_carrier == 30
    assert b.gained_dd_carrier is True
    assert b.gained_no_tdcp is False

    c = diffs[2]
    assert c.d_baseline == 40
    assert c.d_fgo_no_tdcp == 10
    assert c.gained_no_tdcp is True


def test_aggregate_source_totals_collects_interpolated_legacy_name():
    counts = [
        SourceCounts(trip_id="a", n_epochs=10, interpolated=5),
        SourceCounts(trip_id="b", n_epochs=10, interpolated=7),
    ]
    totals = aggregate_source_totals(counts)
    assert totals["interpolated"] == 12
    assert totals["baseline"] == 0


def test_load_taroz_merges_interpolated_missing_with_interpolated(tmp_path: Path):
    path = tmp_path / "summary.json"
    _write_taroz_summary(
        path,
        [
            {
                "trip": "train/x/p",
                "n_epochs": 100,
                "selected_source_counts": {"baseline": 70, "interpolated": 10, "interpolated_missing": 20},
            }
        ],
    )
    [row] = load_taroz_source_counts(path)
    assert row.interpolated == 30
