from __future__ import annotations

import json

import pandas as pd

from experiments.reconstruct_gsdc2023_matlab_reference_submission import (
    apply_row_summary_coordinates,
    main,
    reconstruct_matlab_reference_submission,
)


def _write_submission(path, *, trip_b_lat: float = 0.001) -> None:
    pd.DataFrame(
        {
            "tripId": ["course-a/pixel5", "course-a/pixel5", "course-b/mi8"],
            "UnixTimeMillis": [1000, 2000, 1000],
            "LatitudeDegrees": [0.0, 0.0, trip_b_lat],
            "LongitudeDegrees": [0.0, 0.001, 0.001],
        },
    ).to_csv(path, index=False)


def _write_bridge_tree(root) -> None:
    trip_a = root / "course-a" / "pixel5"
    trip_a.mkdir(parents=True)
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 2000],
            "BaselineLatitudeDegrees": [0.0, 0.0],
            "BaselineLongitudeDegrees": [0.0, 0.0],
            "FgoLatitudeDegrees": [0.0, 0.0],
            "FgoLongitudeDegrees": [0.0, 0.001],
        },
    ).to_csv(trip_a / "bridge_positions.csv", index=False)

    trip_b = root / "course-b" / "mi8"
    trip_b.mkdir(parents=True)
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000],
            "BaselineLatitudeDegrees": [0.0],
            "BaselineLongitudeDegrees": [0.0],
        },
    ).to_csv(trip_b / "bridge_positions.csv", index=False)


def test_apply_row_summary_coordinates_replaces_keyed_rows() -> None:
    submission = pd.DataFrame(
        {
            "tripId": ["trip/a", "trip/a", "trip/b"],
            "UnixTimeMillis": [1, 2, 1],
            "LatitudeDegrees": [10.0, 20.0, 30.0],
            "LongitudeDegrees": [40.0, 50.0, 60.0],
        },
    )
    rows = pd.DataFrame(
        {
            "tripId": ["trip/a"],
            "UnixTimeMillis": [2],
            "best_source_latitude_degrees": [21.0],
            "best_source_longitude_degrees": [51.0],
        },
    )

    reconstructed, summary = apply_row_summary_coordinates(submission, rows, source_label="override")

    assert reconstructed["LatitudeDegrees"].tolist() == [10.0, 21.0, 30.0]
    assert reconstructed["LongitudeDegrees"].tolist() == [40.0, 51.0, 60.0]
    assert summary["rows_replaced"] == 1
    assert summary["rows_by_trip"] == {"trip/a": 1}


def test_apply_row_summary_coordinates_accepts_reference_source_columns() -> None:
    submission = pd.DataFrame(
        {
            "tripId": ["trip/a"],
            "UnixTimeMillis": [1],
            "LatitudeDegrees": [10.0],
            "LongitudeDegrees": [40.0],
        },
    )
    rows = pd.DataFrame(
        {
            "tripId": ["trip/a"],
            "UnixTimeMillis": [1],
            "best_reference_source_latitude_degrees": [11.0],
            "best_reference_source_longitude_degrees": [41.0],
        },
    )

    reconstructed, summary = apply_row_summary_coordinates(submission, rows, source_label="target_trip")

    assert reconstructed["LatitudeDegrees"].tolist() == [11.0]
    assert reconstructed["LongitudeDegrees"].tolist() == [41.0]
    assert summary["rows_replaced"] == 1


def test_reconstruct_matlab_reference_submission_applies_bridge_and_overrides(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    bridge_root = tmp_path / "bridge"
    override = tmp_path / "override.csv"
    _write_submission(reference)
    _write_submission(candidate, trip_b_lat=9.0)
    _write_bridge_tree(bridge_root)
    pd.DataFrame(
        {
            "tripId": ["course-b/mi8"],
            "UnixTimeMillis": [1000],
            "best_source_latitude_degrees": [0.001],
            "best_source_longitude_degrees": [0.001],
        },
    ).to_csv(override, index=False)

    reconstructed, _, _, source_runs, _, summary = reconstruct_matlab_reference_submission(
        reference_submission=reference,
        candidate_submission=candidate,
        bridge_root=bridge_root,
        override_row_summaries=[("manual", override)],
    )

    assert reconstructed["LatitudeDegrees"].tolist() == [0.0, 0.0, 0.001]
    assert reconstructed["LongitudeDegrees"].tolist() == [0.0, 0.001, 0.001]
    assert summary["base_reconstruction"]["rows_replaced"] == 3
    assert summary["override_reconstructions"][0]["rows_replaced"] == 1
    assert summary["delta_vs_reference"]["max_delta_m"] == 0.0
    assert len(source_runs) == 3


def test_reconstruct_matlab_reference_submission_cli_writes_outputs(tmp_path, capsys) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    bridge_root = tmp_path / "bridge"
    output = tmp_path / "out"
    _write_submission(reference)
    _write_submission(candidate, trip_b_lat=9.0)
    _write_bridge_tree(bridge_root)

    assert (
        main(
            [
                "--reference-submission",
                str(reference),
                "--candidate-submission",
                str(candidate),
                "--bridge-root",
                str(bridge_root),
                "--output-dir",
                str(output),
            ],
        )
        == 0
    )

    assert "reconstructed: rows=3" in capsys.readouterr().out
    assert (output / "submission_reconstructed_matlab_reference.csv").is_file()
    assert (output / "all_trip_bridge_source_runs.csv").is_file()
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert payload["base_reconstruction"]["rows_replaced"] == 3
