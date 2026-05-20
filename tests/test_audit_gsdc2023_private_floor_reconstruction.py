from __future__ import annotations

import csv

import pandas as pd

from experiments.audit_gsdc2023_private_floor_reconstruction import (
    audit_private_floor_reconstruction,
)


def _submission(offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tripId": ["trip-a/pixel5", "trip-a/pixel5"],
            "UnixTimeMillis": [1000, 2000],
            "LatitudeDegrees": [37.0 + offset, 37.1 + offset],
            "LongitudeDegrees": [-122.0, -122.1],
        },
    )


def test_audit_private_floor_reconstruction_finds_exact_private_floor_match(tmp_path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    local = results_root / "gsdc2023_submission.csv"
    scored = results_root / "submission_private_floor_weighted_best_p3p25_a0p0625_20260505.csv"
    _submission().to_csv(local, index=False)
    _submission(0.0001).to_csv(scored, index=False)

    score_history = tmp_path / "history.csv"
    with score_history.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["fileName", "date", "description", "status", "publicScore", "privateScore"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "fileName": scored.name,
                "date": "2026-05-05",
                "description": "private floor",
                "status": "complete",
                "publicScore": "3.687",
                "privateScore": "4.710",
            },
        )

    payload = audit_private_floor_reconstruction(
        results_root=results_root,
        score_history_csv=score_history,
        output_dir=tmp_path / "audit",
        search_roots=(results_root,),
    )

    assert payload["exact_private_floor_local_matches"] == 1
    assert payload["private_floor_reconstructable_from_available_files"] is True
    assert (tmp_path / "audit" / "private_floor_reconstruction_audit.md").is_file()


def test_audit_private_floor_reconstruction_reports_missing_inputs(tmp_path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    _submission().to_csv(results_root / "gsdc2023_submission.csv", index=False)

    score_history = tmp_path / "history.csv"
    with score_history.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["fileName", "date", "description", "status", "publicScore", "privateScore"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "fileName": "submission_private_floor_missing.csv",
                "date": "2026-05-05",
                "description": "private floor",
                "status": "complete",
                "publicScore": "3.687",
                "privateScore": "4.710",
            },
        )

    payload = audit_private_floor_reconstruction(
        results_root=results_root,
        score_history_csv=score_history,
        output_dir=tmp_path / "audit",
        search_roots=(results_root,),
    )

    assert payload["exact_private_floor_local_matches"] == 0
    assert payload["private_floor_reconstructable_from_available_files"] is False
    assert payload["missing_prerequisite_count"] > 0
