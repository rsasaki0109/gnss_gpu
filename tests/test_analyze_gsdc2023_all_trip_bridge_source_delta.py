from __future__ import annotations

import json

import pandas as pd

from experiments.analyze_gsdc2023_all_trip_bridge_source_delta import (
    analyze_all_trip_bridge_source_delta,
    main,
    reconstruct_candidate_submission,
)


def _write_reference(path) -> None:
    pd.DataFrame(
        {
            "tripId": ["course-a/pixel5", "course-a/pixel5", "course-b/mi8"],
            "UnixTimeMillis": [1000, 2000, 1000],
            "LatitudeDegrees": [0.0, 0.0, 0.001],
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


def test_analyze_all_trip_bridge_source_delta_summarizes_trips(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    bridge_root = tmp_path / "bridge"
    _write_reference(reference)
    _write_bridge_tree(bridge_root)

    rows, trips, summary = analyze_all_trip_bridge_source_delta(
        reference_submission=reference,
        bridge_root=bridge_root,
    )

    trip_a = trips[trips["tripId"] == "course-a/pixel5"].iloc[0]
    trip_b = trips[trips["tripId"] == "course-b/mi8"].iloc[0]
    assert trip_a["status"] == "compared"
    assert trip_a["best_source_p95_m"] < 1e-6
    assert trip_a["best_baseline_rows"] == 1
    assert trip_a["best_fgo_rows"] == 1
    assert trip_b["best_source_rows_gt_1m"] == 1
    assert len(rows) == 3
    assert rows["best_source_latitude_degrees"].tolist() == [0.0, 0.0, 0.0]
    assert rows["best_source_longitude_degrees"].tolist() == [0.0, 0.001, 0.0]
    assert summary["trip_count"] == 2


def test_reconstruct_candidate_submission_replaces_matched_rows(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    bridge_root = tmp_path / "bridge"
    _write_reference(reference)
    _write_reference(candidate)
    _write_bridge_tree(bridge_root)

    rows, _, _ = analyze_all_trip_bridge_source_delta(
        reference_submission=reference,
        bridge_root=bridge_root,
    )
    reconstructed, summary = reconstruct_candidate_submission(candidate, rows.iloc[:2])

    assert reconstructed["LatitudeDegrees"].tolist() == [0.0, 0.0, 0.001]
    assert reconstructed["LongitudeDegrees"].tolist() == [0.0, 0.001, 0.001]
    assert summary["rows_replaced"] == 2
    assert summary["rows_unmatched"] == 1


def test_analyze_all_trip_bridge_source_delta_cli_writes_outputs(tmp_path, capsys) -> None:
    reference = tmp_path / "reference.csv"
    bridge_root = tmp_path / "bridge"
    output = tmp_path / "out"
    _write_reference(reference)
    _write_bridge_tree(bridge_root)

    assert (
        main(
            [
                "--reference-submission",
                str(reference),
                "--bridge-root",
                str(bridge_root),
                "--output-dir",
                str(output),
            ],
        )
        == 0
    )

    assert "analyzed: 2 trip(s), 3 matched row(s)" in capsys.readouterr().out
    assert (output / "all_trip_bridge_source_delta_rows.csv").is_file()
    assert (output / "all_trip_bridge_source_delta_trips.csv").is_file()
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert payload["matched_rows"] == 3


def test_analyze_all_trip_bridge_source_delta_cli_reconstructs_submission(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    bridge_root = tmp_path / "bridge"
    output = tmp_path / "out"
    _write_reference(reference)
    _write_reference(candidate)
    _write_bridge_tree(bridge_root)

    assert (
        main(
            [
                "--reference-submission",
                str(reference),
                "--bridge-root",
                str(bridge_root),
                "--output-dir",
                str(output),
                "--candidate-submission",
                str(candidate),
                "--write-reconstructed-submission",
            ],
        )
        == 0
    )

    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    reconstructed = output / "submission_with_all_trip_best_reference_bridge_source.csv"
    assert reconstructed.is_file()
    assert payload["reconstructed_submission"]["rows_replaced"] == 3
