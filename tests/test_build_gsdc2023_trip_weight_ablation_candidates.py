from __future__ import annotations

import csv

import pandas as pd
import pytest

from experiments.build_gsdc2023_trip_weight_ablation_candidates import (
    build_trip_weight_ablation_candidates,
    main,
)


TRIP_A = "2022-01-01-00-00-us-ca-a/pixel5"
TRIP_B = "2022-01-02-00-00-us-ca-b/pixel5"
TRIP_STATIC = "2022-01-03-00-00-us-ca-c/pixel5"


def _reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tripId": [TRIP_A, TRIP_A, TRIP_B, TRIP_STATIC],
            "UnixTimeMillis": [1000, 2000, 1000, 1000],
            "LatitudeDegrees": [37.0, 37.1, 38.0, 39.0],
            "LongitudeDegrees": [-122.0, -122.1, -123.0, -124.0],
        },
    )


def _target() -> pd.DataFrame:
    target = _reference()
    target.loc[target["tripId"] == TRIP_A, "LatitudeDegrees"] += 0.0001
    target.loc[target["tripId"] == TRIP_B, "LongitudeDegrees"] -= 0.0001
    return target


def test_build_trip_weight_ablation_candidates_writes_single_and_leave_one_out(tmp_path) -> None:
    reference_path = tmp_path / "reference.csv"
    target_path = tmp_path / "target.csv"
    output_dir = tmp_path / "out"
    _reference().to_csv(reference_path, index=False)
    _target().to_csv(target_path, index=False)

    rows = build_trip_weight_ablation_candidates(
        reference_path=reference_path,
        target_path=target_path,
        output_dir=output_dir,
        tag="test",
        alpha=0.5,
    )

    assert len(rows) == 4
    assert {row["mode"] for row in rows} == {"single", "leave_one_out"}
    assert all(row["output_sha256"] for row in rows)
    assert (output_dir / "trip_weight_ablation_manifest_test.csv").is_file()
    assert (output_dir / "trip_weight_ablation_summary_test.json").is_file()

    manifest_rows = list(csv.DictReader((output_dir / "trip_weight_ablation_manifest_test.csv").open()))
    assert len(manifest_rows) == 4
    single_a = next(row for row in manifest_rows if row["mode"] == "single" and row["active_trip"] == TRIP_A)
    leave_a = next(row for row in manifest_rows if row["mode"] == "leave_one_out" and row["held_trip"] == TRIP_A)
    assert int(single_a["active_trip_count"]) == 1
    assert int(single_a["changed_rows"]) == 2
    assert int(leave_a["active_trip_count"]) == 1
    assert int(leave_a["changed_rows"]) == 1


def test_build_trip_weight_ablation_candidates_rejects_static_selected_trip(tmp_path) -> None:
    reference_path = tmp_path / "reference.csv"
    target_path = tmp_path / "target.csv"
    _reference().to_csv(reference_path, index=False)
    _target().to_csv(target_path, index=False)

    with pytest.raises(SystemExit, match="do not move"):
        build_trip_weight_ablation_candidates(
            reference_path=reference_path,
            target_path=target_path,
            output_dir=tmp_path / "out",
            tag="test",
            trips=(TRIP_STATIC,),
        )


def test_build_trip_weight_ablation_candidates_writes_leave_group_out(tmp_path) -> None:
    reference_path = tmp_path / "reference.csv"
    target_path = tmp_path / "target.csv"
    output_dir = tmp_path / "out"
    _reference().to_csv(reference_path, index=False)
    _target().to_csv(target_path, index=False)

    rows = build_trip_weight_ablation_candidates(
        reference_path=reference_path,
        target_path=target_path,
        output_dir=output_dir,
        tag="test",
        modes=("leave_group_out",),
        trips=(TRIP_A, TRIP_B),
        group_held_trips=(TRIP_A,),
        group_extra_count=1,
    )

    assert len(rows) == 1
    assert rows[0]["mode"] == "leave_group_out"
    assert rows[0]["held_trip"] == f"{TRIP_A}|{TRIP_B}"
    assert rows[0]["held_trip_count"] == 2
    assert rows[0]["active_trip_count"] == 0
    assert rows[0]["changed_rows"] == 0

    manifest_rows = list(csv.DictReader((output_dir / "trip_weight_ablation_manifest_test.csv").open()))
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["held_trip"] == f"{TRIP_A}|{TRIP_B}"
    assert manifest_rows[0]["held_trip_count"] == "2"


def test_build_trip_weight_ablation_candidates_cli(tmp_path, capsys) -> None:
    reference_path = tmp_path / "reference.csv"
    target_path = tmp_path / "target.csv"
    output_dir = tmp_path / "out"
    _reference().to_csv(reference_path, index=False)
    _target().to_csv(target_path, index=False)

    assert main(
        [
            "--reference",
            str(reference_path),
            "--target",
            str(target_path),
            "--output-dir",
            str(output_dir),
            "--tag",
            "test",
            "--mode",
            "single",
        ],
    ) == 0
    assert "prepared: 2 candidate(s)" in capsys.readouterr().out
