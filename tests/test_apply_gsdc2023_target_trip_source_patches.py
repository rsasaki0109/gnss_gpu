from __future__ import annotations

import json

import pandas as pd

from experiments.apply_gsdc2023_target_trip_source_patches import (
    apply_source_patches,
    main,
    parse_patch_spec,
    read_submission,
)


TRIP_A = "trip-a/pixel5"
TRIP_B = "trip-b/mi8"


def _write_submission(path, trip_ids: list[str], lats: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "tripId": trip_ids,
            "UnixTimeMillis": [1000 + 1000 * index for index in range(len(trip_ids))],
            "LatitudeDegrees": lats,
            "LongitudeDegrees": [0.0] * len(trip_ids),
        },
    )
    frame.to_csv(path, index=False)


def _write_row_summary(path, trip_ids: list[str], patch_lats: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "tripId": trip_ids,
            "UnixTimeMillis": [1000 + 1000 * index for index in range(len(trip_ids))],
            "epoch_index": list(range(len(trip_ids))),
            "best_reference_source_latitude_degrees": patch_lats,
            "best_reference_source_longitude_degrees": [1.0] * len(trip_ids),
        },
    )
    frame.to_csv(path, index=False)


def test_parse_patch_spec() -> None:
    spec = parse_patch_spec(f"{TRIP_A}=rows.csv#10-20")

    assert spec.trip_id == TRIP_A
    assert str(spec.row_summary_path) == "rows.csv"
    assert spec.start_epoch == 10
    assert spec.end_epoch == 20


def test_apply_source_patches_replaces_epoch_range(tmp_path) -> None:
    submission = tmp_path / "submission.csv"
    rows = tmp_path / "rows.csv"
    _write_submission(submission, [TRIP_A] * 4 + [TRIP_B], [0.0, 0.1, 0.2, 0.3, 9.0])
    _write_row_summary(rows, [TRIP_A] * 4 + [TRIP_B], [10.0, 10.1, 10.2, 10.3, 19.0])

    patched, summaries = apply_source_patches(read_submission(submission), [parse_patch_spec(f"{TRIP_A}={rows}#1-3")])

    assert patched["LatitudeDegrees"].tolist() == [0.0, 10.1, 10.2, 0.3, 9.0]
    assert patched["LongitudeDegrees"].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert summaries[0]["rows_replaced"] == 2


def test_apply_source_patches_cli_writes_patched_submission(tmp_path, capsys) -> None:
    candidate = tmp_path / "candidate.csv"
    reference = tmp_path / "reference.csv"
    rows = tmp_path / "rows.csv"
    output = tmp_path / "out"
    _write_submission(candidate, [TRIP_A] * 3, [0.0, 0.1, 0.2])
    _write_submission(reference, [TRIP_A] * 3, [0.0, 10.1, 10.2])
    _write_row_summary(rows, [TRIP_A] * 3, [10.0, 10.1, 10.2])

    assert (
        main(
            [
                "--candidate-submission",
                str(candidate),
                "--reference-submission",
                str(reference),
                "--patch",
                f"{TRIP_A}={rows}#1-3",
                "--output-dir",
                str(output),
            ],
        )
        == 0
    )

    assert "rows replaced: 2" in capsys.readouterr().out
    patched = pd.read_csv(output / "submission_with_target_source_patches.csv")
    assert patched["LatitudeDegrees"].tolist() == [0.0, 10.1, 10.2]
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert payload["rows_replaced"] == 2
    assert payload["patched_vs_reference"]["rows"] == 3
