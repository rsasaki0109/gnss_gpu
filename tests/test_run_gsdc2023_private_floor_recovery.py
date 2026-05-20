from __future__ import annotations

import csv

import pandas as pd

from experiments.build_gsdc2023_basecorr_posoffset_pixel5_patch_candidates import (
    PIXEL5_PATCH_TRIP,
    PIXEL5_SJC_R_TRIP,
)
from experiments.run_gsdc2023_private_floor_recovery import main, run_private_floor_recovery


def _write_score_history(path, filename: str, *, private_score: str = "4.711") -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["fileName", "date", "description", "status", "publicScore", "privateScore"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "fileName": filename,
                "date": "2026-05-01",
                "description": "pixel5 3.375 sjc r scale 0.84375",
                "status": "complete",
                "publicScore": "3.725",
                "privateScore": private_score,
            },
        )


def _write_submission(path) -> None:
    pd.DataFrame(
        {
            "tripId": [
                PIXEL5_PATCH_TRIP,
                PIXEL5_PATCH_TRIP,
                PIXEL5_SJC_R_TRIP,
                PIXEL5_SJC_R_TRIP,
            ],
            "UnixTimeMillis": [1000, 2000, 1000, 2000],
            "LatitudeDegrees": [37.0, 37.00001, 37.5, 37.50001],
            "LongitudeDegrees": [-122.0, -121.99999, -122.5, -122.49999],
        },
    ).to_csv(path, index=False)


def _write_patch(path) -> None:
    pd.DataFrame(
        {
            "tripId": [PIXEL5_PATCH_TRIP, PIXEL5_PATCH_TRIP],
            "UnixTimeMillis": [1000, 2000],
            "LatitudeDegrees": [38.0, 38.00001],
            "LongitudeDegrees": [-123.0, -122.99999],
        },
    ).to_csv(path, index=False)


def test_recovery_one_shot_blocks_when_direct_builder_inputs_are_missing(tmp_path) -> None:
    score_history = tmp_path / "history.csv"
    _write_score_history(
        score_history,
        "submission_best_basecorr_posoffset_pixel5phone_3p375_sjc_r0p84375_plus_pixel5_patch_20260501.csv",
    )

    payload = run_private_floor_recovery(
        output_dir=tmp_path / "recovery",
        score_history_csv=score_history,
        input_path=tmp_path / "missing_input.csv",
        pixel5_patch_path=tmp_path / "missing_patch.csv",
    )

    assert payload["status"] == "blocked"
    assert payload["direct_private_floor_builder_ready"] is False
    assert payload["build_skipped_reason"] == "direct_private_floor_builder_not_ready"
    assert (tmp_path / "recovery" / "summary.json").is_file()


def test_recovery_one_shot_builds_selected_candidate_when_inputs_are_ready(tmp_path) -> None:
    input_path = tmp_path / "basecorr_input.csv"
    patch_path = tmp_path / "pixel5_patch.csv"
    score_history = tmp_path / "history.csv"
    _write_submission(input_path)
    _write_patch(patch_path)
    _write_score_history(
        score_history,
        "submission_best_basecorr_posoffset_pixel5phone_3p375_sjc_r0p84375_plus_pixel5_patch_20260501.csv",
    )

    payload = run_private_floor_recovery(
        output_dir=tmp_path / "recovery",
        score_history_csv=score_history,
        input_path=input_path,
        pixel5_patch_path=patch_path,
        candidate_names=["pixel5phone_3p375_sjc_r0p84375"],
        tag="test",
    )

    assert payload["status"] == "ready"
    assert payload["built_candidate_count"] == 1
    assert payload["screen_candidate_count"] == 1
    built = (
        tmp_path
        / "recovery"
        / "basecorr_candidates"
        / "pixel5phone_3p375_sjc_r0p84375"
        / "submission_best_basecorr_posoffset_pixel5phone_3p375_sjc_r0p84375_plus_pixel5_patch_test.csv"
    )
    assert built.is_file()
    assert (tmp_path / "recovery" / "local_submission_screen.csv").is_file()


def test_recovery_cli_returns_two_when_unready(tmp_path) -> None:
    score_history = tmp_path / "history.csv"
    _write_score_history(
        score_history,
        "submission_best_basecorr_posoffset_pixel5phone_3p375_sjc_r0p84375_plus_pixel5_patch_20260501.csv",
    )

    code = main(
        [
            "--output-dir",
            str(tmp_path / "recovery"),
            "--score-history-csv",
            str(score_history),
            "--input",
            str(tmp_path / "missing_input.csv"),
            "--pixel5-patch",
            str(tmp_path / "missing_patch.csv"),
        ],
    )

    assert code == 2
