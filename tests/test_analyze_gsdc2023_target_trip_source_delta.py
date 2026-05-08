from __future__ import annotations

import csv
import json

import pandas as pd

from experiments.analyze_gsdc2023_target_trip_source_delta import (
    analyze_target_trip_source_delta,
    main,
    reconstruct_candidate_submission,
    source_coordinate_columns,
)


TRIP = "trip-a/pixel5"


def _write_submission(path, lats: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "tripId": [TRIP] * len(lats),
            "UnixTimeMillis": [1000 + 1000 * index for index in range(len(lats))],
            "LatitudeDegrees": lats,
            "LongitudeDegrees": [0.0] * len(lats),
        },
    )
    frame.to_csv(path, index=False)


def _write_bridge(path) -> None:
    frame = pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 2000, 3000, 4000],
            "SelectedSource": ["baseline", "baseline", "raw_wls", "raw_wls"],
            "BaselineLatitudeDegrees": [0.0, 0.0, 0.0, 0.0],
            "BaselineLongitudeDegrees": [0.0, 0.0, 0.0, 0.0],
            "RawWlsLatitudeDegrees": [0.001, 0.001, 0.001, 0.001],
            "RawWlsLongitudeDegrees": [0.0, 0.0, 0.0, 0.0],
            "FgoLatitudeDegrees": [0.002, 0.002, 0.002, 0.002],
            "FgoLongitudeDegrees": [0.0, 0.0, 0.0, 0.0],
            "LatitudeDegrees": [0.0, 0.0, 0.001, 0.001],
            "LongitudeDegrees": [0.0, 0.0, 0.0, 0.0],
            "GroundTruthLatitudeDegrees": [float("nan")] * 4,
            "GroundTruthLongitudeDegrees": [float("nan")] * 4,
        },
    )
    frame.to_csv(path, index=False)


def test_source_coordinate_columns_detects_bridge_sources() -> None:
    frame = pd.DataFrame(
        {
            "BaselineLatitudeDegrees": [0.0],
            "BaselineLongitudeDegrees": [0.0],
            "RawWlsLatitudeDegrees": [0.0],
            "RawWlsLongitudeDegrees": [0.0],
            "LatitudeDegrees": [0.0],
            "LongitudeDegrees": [0.0],
            "GroundTruthLatitudeDegrees": [float("nan")],
            "GroundTruthLongitudeDegrees": [float("nan")],
        },
    )

    assert source_coordinate_columns(frame) == {
        "baseline": ("BaselineLatitudeDegrees", "BaselineLongitudeDegrees"),
        "raw_wls": ("RawWlsLatitudeDegrees", "RawWlsLongitudeDegrees"),
        "selected": ("LatitudeDegrees", "LongitudeDegrees"),
    }


def test_analyze_target_trip_source_delta_summarizes_best_sources(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    bridge = tmp_path / "bridge.csv"
    _write_submission(reference, [0.0, 0.0, 0.001, 0.001])
    _write_submission(candidate, [0.001, 0.001, 0.0, 0.0])
    _write_bridge(bridge)

    rows, chunks, summary = analyze_target_trip_source_delta(
        reference_submission=reference,
        candidate_submission=candidate,
        bridge_rows=bridge,
        target_trip=TRIP,
        chunk_epochs=2,
    )

    assert rows["best_reference_source"].tolist() == ["baseline", "baseline", "raw_wls", "raw_wls"]
    assert summary["best_reference_source_counts"] == {"baseline": 2, "raw_wls": 2}
    assert chunks.loc[0, "best_reference_source_baseline_rows"] == 2
    assert chunks.loc[1, "best_reference_source_raw_wls_rows"] == 2
    assert chunks.loc[0, "selected_source_baseline_rows"] == 2
    assert chunks.loc[1, "selected_source_raw_wls_rows"] == 2
    assert rows["best_reference_source_latitude_degrees"].tolist() == [0.0, 0.0, 0.001, 0.001]


def test_reconstruct_candidate_submission_replaces_target_trip_with_best_source(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    bridge = tmp_path / "bridge.csv"
    _write_submission(reference, [0.0, 0.0, 0.001, 0.001])
    _write_submission(candidate, [0.001, 0.001, 0.0, 0.0])
    _write_bridge(bridge)
    rows, _, _ = analyze_target_trip_source_delta(
        reference_submission=reference,
        candidate_submission=candidate,
        bridge_rows=bridge,
        target_trip=TRIP,
        chunk_epochs=2,
    )

    reconstructed, summary = reconstruct_candidate_submission(candidate, rows, target_trip=TRIP)

    assert reconstructed["LatitudeDegrees"].tolist() == [0.0, 0.0, 0.001, 0.001]
    assert summary["rows_replaced"] == 4


def test_analyze_target_trip_source_delta_cli(tmp_path, capsys) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    bridge = tmp_path / "bridge.csv"
    output = tmp_path / "out"
    _write_submission(reference, [0.0, 0.0, 0.001, 0.001])
    _write_submission(candidate, [0.001, 0.001, 0.0, 0.0])
    _write_bridge(bridge)

    assert (
        main(
            [
                "--reference-submission",
                str(reference),
                "--candidate-submission",
                str(candidate),
                "--bridge-rows",
                str(bridge),
                "--target-trip",
                TRIP,
                "--chunk-epochs",
                "2",
                "--write-reconstructed-submission",
                "--output-dir",
                str(output),
            ],
        )
        == 0
    )

    assert "analyzed: 4 row(s)" in capsys.readouterr().out
    assert (output / "summary.json").is_file()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["rows"] == 4
    assert summary["reconstructed_submission"]["rows_replaced"] == 4
    assert (output / "submission_with_target_trip_best_reference_source.csv").is_file()
    rows = list(csv.DictReader((output / "target_trip_source_delta_rows.csv").open(encoding="utf-8")))
    assert rows[0]["best_reference_source"] == "baseline"
