from __future__ import annotations

import csv

import pandas as pd
import pytest

from experiments.build_gsdc2023_source_family_ablation_candidates import (
    SourceFamilySpec,
    build_source_family_ablation_candidates,
    main,
    parse_source_family,
)


TRIP_PIXEL4 = "2021-01-01-00-00-us-ca-a/pixel4"
TRIP_PIXEL6PRO = "2021-01-02-00-00-us-ca-b/pixel6pro"


def _reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tripId": [TRIP_PIXEL4, TRIP_PIXEL4, TRIP_PIXEL6PRO, TRIP_PIXEL6PRO],
            "UnixTimeMillis": [1000, 2000, 1000, 2000],
            "LatitudeDegrees": [37.0, 37.1, 38.0, 38.1],
            "LongitudeDegrees": [-122.0, -122.1, -123.0, -123.1],
        },
    )


def _target() -> pd.DataFrame:
    target = _reference()
    target.loc[0, "LatitudeDegrees"] += 0.0001
    target.loc[1, "LongitudeDegrees"] -= 0.0001
    target.loc[2, "LatitudeDegrees"] += 0.0002
    target.loc[3, "LongitudeDegrees"] -= 0.0002
    return target


def _write_bridge_sources(root) -> None:
    trip4_dir = root / "2021-01-01-00-00-us-ca-a" / "pixel4"
    trip4_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000],
            "SelectedSource": ["fgo"],
        },
    ).to_csv(trip4_dir / "bridge_positions.csv", index=False)

    trip6_dir = root / "2021-01-02-00-00-us-ca-b" / "pixel6pro"
    trip6_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 2000],
            "SelectedSource": ["fgo", "baseline"],
        },
    ).to_csv(trip6_dir / "bridge_positions.csv", index=False)


def test_parse_source_family() -> None:
    spec = parse_source_family("pixel6=fgo,raw_wls@pixel6pro,pixel7")

    assert spec == SourceFamilySpec("pixel6", ("fgo", "raw_wls"), ("pixel6pro", "pixel7"))


def test_parse_source_family_rejects_missing_sources() -> None:
    with pytest.raises(SystemExit, match="at least one source"):
        parse_source_family("bad=@pixel6pro")


def test_build_source_family_ablation_candidates_writes_only_and_revert(tmp_path) -> None:
    reference_path = tmp_path / "reference.csv"
    target_path = tmp_path / "target.csv"
    bridge_root = tmp_path / "bridge"
    missing_rows = tmp_path / "missing.csv"
    output_dir = tmp_path / "out"
    _reference().to_csv(reference_path, index=False)
    _target().to_csv(target_path, index=False)
    _write_bridge_sources(bridge_root)
    pd.DataFrame(
        {
            "tripId": [TRIP_PIXEL4],
            "UnixTimeMillis": [2000],
        },
    ).to_csv(missing_rows, index=False)

    rows = build_source_family_ablation_candidates(
        reference_path=reference_path,
        target_path=target_path,
        bridge_output_root=bridge_root,
        missing_rows_path=missing_rows,
        output_dir=output_dir,
        tag="test",
        specs=(
            SourceFamilySpec("pixel4_fgo", ("fgo",), ("pixel4",)),
            SourceFamilySpec("interpolated", ("interpolated_missing",)),
        ),
    )

    assert len(rows) == 4
    assert {row["mode"] for row in rows} == {"only", "revert"}
    pixel4_only = next(row for row in rows if row["family"] == "pixel4_fgo" and row["mode"] == "only")
    interpolated_only = next(row for row in rows if row["family"] == "interpolated" and row["mode"] == "only")
    assert pixel4_only["selected_rows"] == 1
    assert interpolated_only["selected_rows"] == 1
    assert (output_dir / "source_family_ablation_manifest_test.csv").is_file()
    assert (output_dir / "source_family_ablation_summary_test.json").is_file()

    manifest_rows = list(csv.DictReader((output_dir / "source_family_ablation_manifest_test.csv").open()))
    assert len(manifest_rows) == 4
    assert {row["family"] for row in manifest_rows} == {"pixel4_fgo", "interpolated"}


def test_build_source_family_ablation_candidates_cli(tmp_path, capsys) -> None:
    reference_path = tmp_path / "reference.csv"
    target_path = tmp_path / "target.csv"
    bridge_root = tmp_path / "bridge"
    missing_rows = tmp_path / "missing.csv"
    output_dir = tmp_path / "out"
    _reference().to_csv(reference_path, index=False)
    _target().to_csv(target_path, index=False)
    _write_bridge_sources(bridge_root)
    pd.DataFrame(
        {
            "tripId": [TRIP_PIXEL4],
            "UnixTimeMillis": [2000],
        },
    ).to_csv(missing_rows, index=False)

    assert main(
        [
            "--reference",
            str(reference_path),
            "--target",
            str(target_path),
            "--bridge-output-root",
            str(bridge_root),
            "--missing-rows",
            str(missing_rows),
            "--output-dir",
            str(output_dir),
            "--tag",
            "test",
            "--family",
            "pixel6=fgo@pixel6pro",
            "--mode",
            "only",
        ],
    ) == 0
    assert "prepared: 1 candidate(s)" in capsys.readouterr().out
