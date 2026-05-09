from __future__ import annotations

import csv

import pandas as pd

from experiments.screen_gsdc2023_local_submissions import main, screen_local_submissions


RISKY_TRIP = "2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro"
OTHER_TRIP = "2022-10-06-20-46-us-ca-sjc-r/pixel5"


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tripId": [RISKY_TRIP, RISKY_TRIP, OTHER_TRIP],
            "UnixTimeMillis": [1000, 2000, 1000],
            "LatitudeDegrees": [37.0, 37.00001, 38.0],
            "LongitudeDegrees": [-122.0, -121.99999, -123.0],
        },
    )


def test_screen_local_submissions_marks_submitted_duplicates_and_risky_delta(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    base = _submission()
    submitted = base.copy()
    tiny_float_noise = base.copy()
    tiny_float_noise.loc[tiny_float_noise["tripId"] == RISKY_TRIP, "LatitudeDegrees"] += 1.0e-12
    risky_changed = base.copy()
    risky_changed.loc[risky_changed["tripId"] == RISKY_TRIP, "LatitudeDegrees"] += 0.00001
    submitted_path = root / "submission_submitted.csv"
    same_bytes_path = root / "submission_same_bytes_new_name.csv"
    noise_path = root / "submission_tiny_float_noise.csv"
    risky_path = root / "submission_risky.csv"
    base_path = tmp_path / "base.csv"
    submitted.to_csv(submitted_path, index=False)
    submitted.to_csv(same_bytes_path, index=False)
    tiny_float_noise.to_csv(noise_path, index=False)
    risky_changed.to_csv(risky_path, index=False)
    base.to_csv(base_path, index=False)
    submitted_csv = tmp_path / "kaggle.csv"
    submitted_csv.write_text(
        "fileName,date,description,status,publicScore,privateScore\n"
        "submission_submitted.csv,2026-05-05,msg,complete,1.0,2.0\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "screen.csv"

    rows = screen_local_submissions(
        root=root,
        output_csv=output_csv,
        submitted_csv=submitted_csv,
        reference_best=base_path,
        previous_safe=base_path,
        risky_trips=(RISKY_TRIP,),
    )

    by_name = {row["filename"]: row for row in rows}
    assert by_name["submission_submitted.csv"]["submitted_filename"] is True
    assert by_name["submission_same_bytes_new_name.csv"]["duplicate_submitted_local_sha"] is True
    assert by_name["submission_tiny_float_noise.csv"]["risky_previous_changed_rows"] == 0
    assert by_name["submission_tiny_float_noise.csv"]["reference_changed_rows"] == 0
    assert by_name["submission_risky.csv"]["risky_previous_changed_rows"] == 2
    assert by_name["submission_risky.csv"]["reference_changed_rows"] == 2
    assert output_csv.is_file()
    assert output_csv.with_suffix(".summary.json").is_file()

    csv_rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert {row["filename"] for row in csv_rows} == {
        "submission_submitted.csv",
        "submission_same_bytes_new_name.csv",
        "submission_tiny_float_noise.csv",
        "submission_risky.csv",
    }


def test_screen_local_submissions_cli(tmp_path, capsys) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "submission_candidate.csv"
    _submission().to_csv(path, index=False)
    output_csv = tmp_path / "screen.csv"

    assert main(["--root", str(root), "--output-csv", str(output_csv)]) == 0
    assert "screened: 1 candidate(s)" in capsys.readouterr().out
