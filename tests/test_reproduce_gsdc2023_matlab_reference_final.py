from __future__ import annotations

import json

import pandas as pd

from experiments.reproduce_gsdc2023_matlab_reference_final import (
    main,
    reproduce_matlab_reference_final,
)


def _write_submission(path, *, trip_b_lat: float = 9.0) -> None:
    pd.DataFrame(
        {
            "tripId": ["course-a/pixel5", "course-a/pixel5", "course-a/pixel5", "course-b/mi8"],
            "UnixTimeMillis": [1000, 2000, 3000, 1000],
            "LatitudeDegrees": [1.0, 1.0, 2.0, trip_b_lat],
            "LongitudeDegrees": [10.0, 10.0, 20.0, 90.0],
        },
    ).to_csv(path, index=False)


def _write_bridge_tree(root) -> None:
    trip_a = root / "course-a" / "pixel5"
    trip_a.mkdir(parents=True)
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000, 3000],
            "LatitudeDegrees": [1.0, 2.0],
            "LongitudeDegrees": [10.0, 20.0],
            "BaselineLatitudeDegrees": [1.0, 2.0],
            "BaselineLongitudeDegrees": [10.0, 20.0],
        },
    ).to_csv(trip_a / "bridge_positions.csv", index=False)

    trip_b = root / "course-b" / "mi8"
    trip_b.mkdir(parents=True)
    pd.DataFrame(
        {
            "UnixTimeMillis": [1000],
            "LatitudeDegrees": [3.0],
            "LongitudeDegrees": [90.0],
            "BaselineLatitudeDegrees": [3.0],
            "BaselineLongitudeDegrees": [90.0],
        },
    ).to_csv(trip_b / "bridge_positions.csv", index=False)


def test_reproduce_matlab_reference_final_materializes_missing_rows(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    bridge_root = tmp_path / "bridge"
    output_dir = tmp_path / "out"
    _write_submission(reference, trip_b_lat=3.0)
    _write_submission(candidate, trip_b_lat=0.0)
    _write_bridge_tree(bridge_root)

    payload = reproduce_matlab_reference_final(
        reference_submission=reference,
        candidate_submission=candidate,
        bridge_root=bridge_root,
        output_dir=output_dir,
    )

    assert payload["missing_bridge_timestamp_summary"]["rows"] == 1
    assert payload["reconstruction_summary"]["delta_vs_reference"]["max_delta_m"] == 0.0
    reconstructed = pd.read_csv(output_dir / "reconstruction/submission_reconstructed_matlab_reference.csv")
    assert reconstructed["LatitudeDegrees"].tolist() == [1.0, 1.0, 2.0, 3.0]


def test_reproduce_matlab_reference_final_cli_writes_summary(tmp_path, capsys) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    bridge_root = tmp_path / "bridge"
    output_dir = tmp_path / "out"
    _write_submission(reference, trip_b_lat=3.0)
    _write_submission(candidate, trip_b_lat=0.0)
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
                str(output_dir),
            ],
        )
        == 0
    )

    assert "reproduced MATLAB reference final: rows=4" in capsys.readouterr().out
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["missing_bridge_timestamp_summary"]["rows"] == 1
