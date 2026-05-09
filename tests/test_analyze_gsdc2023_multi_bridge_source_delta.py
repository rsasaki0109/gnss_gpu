from __future__ import annotations

import json

import pandas as pd

from experiments.analyze_gsdc2023_multi_bridge_source_delta import analyze_multi_bridge_source_delta, main


TRIP = "trip-a/pixel5"


def _write_reference(path) -> None:
    pd.DataFrame(
        {
            "tripId": [TRIP, TRIP, TRIP, "other/pixel5"],
            "UnixTimeMillis": [1000, 2000, 3000, 1000],
            "LatitudeDegrees": [0.0, 0.0, 0.0001, 1.0],
            "LongitudeDegrees": [0.0, 0.0001, 0.0002, 1.0],
        },
    ).to_csv(path, index=False)


def _write_candidate(path) -> None:
    pd.DataFrame(
        {
            "tripId": [TRIP, TRIP, TRIP, "other/pixel5"],
            "UnixTimeMillis": [1000, 2000, 3000, 1000],
            "LatitudeDegrees": [1.0, 1.0, 1.0, 1.0],
            "LongitudeDegrees": [1.0, 1.0, 1.0, 1.0],
        },
    ).to_csv(path, index=False)


def _write_bridge_a(path) -> None:
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 2000, 3000],
            "BaselineLatitudeDegrees": [0.0, 0.0, 0.0],
            "BaselineLongitudeDegrees": [0.0, 0.0, 0.0],
            "FgoLatitudeDegrees": [0.0005, 0.0005, 0.0005],
            "FgoLongitudeDegrees": [0.0005, 0.0005, 0.0005],
        },
    ).to_csv(path, index=False)


def _write_bridge_b(path) -> None:
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 2000, 3000],
            "RawWlsLatitudeDegrees": [0.001, 0.0, 0.0001],
            "RawWlsLongitudeDegrees": [0.001, 0.0001, 0.0002],
        },
    ).to_csv(path, index=False)


def test_analyze_multi_bridge_source_delta_picks_best_artifact_source(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    bridge_a = tmp_path / "bridge_a.csv"
    bridge_b = tmp_path / "bridge_b.csv"
    _write_reference(reference)
    _write_bridge_a(bridge_a)
    _write_bridge_b(bridge_b)

    rows, chunks, summary = analyze_multi_bridge_source_delta(
        reference_submission=reference,
        target_trip=TRIP,
        bridge_sources=[("a", bridge_a), ("b", bridge_b)],
        chunk_epochs=2,
    )

    assert rows["best_bridge_source"].tolist() == ["a:baseline", "b:raw_wls", "b:raw_wls"]
    assert rows["best_source_distance_m"].max() < 1e-6
    assert chunks.loc[0, "top_bridge_source"] == "a:baseline"
    assert summary["best_bridge_source_counts"] == {"a:baseline": 1, "b:raw_wls": 2}


def test_analyze_multi_bridge_source_delta_cli_reconstructs_submission(tmp_path, capsys) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    bridge_a = tmp_path / "bridge_a.csv"
    bridge_b = tmp_path / "bridge_b.csv"
    output = tmp_path / "out"
    _write_reference(reference)
    _write_candidate(candidate)
    _write_bridge_a(bridge_a)
    _write_bridge_b(bridge_b)

    assert (
        main(
            [
                "--reference-submission",
                str(reference),
                "--target-trip",
                TRIP,
                "--bridge-source",
                f"a={bridge_a}",
                "--bridge-source",
                f"b={bridge_b}",
                "--candidate-submission",
                str(candidate),
                "--write-reconstructed-submission",
                "--output-dir",
                str(output),
            ],
        )
        == 0
    )

    assert "analyzed: 3 row(s)" in capsys.readouterr().out
    assert (output / "multi_bridge_source_delta_rows.csv").is_file()
    reconstructed = pd.read_csv(output / "submission_with_target_trip_multi_bridge_best_source.csv")
    patched = reconstructed[reconstructed["tripId"] == TRIP].sort_values("UnixTimeMillis")
    assert patched["LatitudeDegrees"].tolist() == [0.0, 0.0, 0.0001]
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert payload["reconstructed_submission"]["rows_replaced"] == 3
