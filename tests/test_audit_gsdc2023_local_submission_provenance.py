from __future__ import annotations

import csv
import json

import pandas as pd

from experiments.audit_gsdc2023_local_submission_provenance import (
    audit_local_submission_provenance,
    main,
)


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tripId": ["trip-a/pixel5"],
            "UnixTimeMillis": [1000],
            "LatitudeDegrees": [37.0],
            "LongitudeDegrees": [-122.0],
        },
    )


def _write_score_history(path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["fileName", "date", "description", "status", "publicScore", "privateScore"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_audit_local_submission_provenance_marks_unscored_local_as_legacy(tmp_path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    _submission().to_csv(results_root / "gsdc2023_submission_v15.csv", index=False)
    score_history = tmp_path / "scores.csv"
    _write_score_history(
        score_history,
        [
            {
                "fileName": "submission_private_floor_weighted_best_p3p25_a0p0625_20260505.csv",
                "date": "2026-05-05 22:07:32.843000",
                "description": "private floor",
                "status": "complete",
                "publicScore": "3.687",
                "privateScore": "4.710",
            },
        ],
    )

    payload = audit_local_submission_provenance(
        results_root=results_root,
        score_history_csv=score_history,
        output_dir=tmp_path / "audit",
        repo_root=tmp_path,
    )

    assert payload["local_score_backed_private_floor_count"] == 0
    rows = list(csv.DictReader((tmp_path / "audit" / "local_submission_provenance.csv").open()))
    assert rows[0]["provenance_class"] == "legacy_pf_local_unscored"
    assert rows[0]["usable_as_private_floor_base"] == "False"


def test_audit_local_submission_provenance_detects_exact_private_floor_match(tmp_path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    local = results_root / "gsdc2023_submission.csv"
    _submission().to_csv(local, index=False)
    score_history = tmp_path / "scores.csv"
    _write_score_history(
        score_history,
        [
            {
                "fileName": local.name,
                "date": "2026-05-05 22:07:32.843000",
                "description": "private floor",
                "status": "complete",
                "publicScore": "3.687",
                "privateScore": "4.710",
            },
        ],
    )

    payload = audit_local_submission_provenance(
        results_root=results_root,
        score_history_csv=score_history,
        output_dir=tmp_path / "audit",
        repo_root=tmp_path,
    )

    assert payload["local_score_backed_private_floor_count"] == 1
    rows = list(csv.DictReader((tmp_path / "audit" / "local_submission_provenance.csv").open()))
    assert rows[0]["provenance_class"] == "score_backed_private_floor"
    assert rows[0]["usable_as_private_floor_base"] == "True"


def test_audit_local_submission_provenance_cli(tmp_path, capsys) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    _submission().to_csv(results_root / "gsdc2023_submission_v15.csv", index=False)
    score_history = tmp_path / "scores.csv"
    _write_score_history(score_history, [])
    output_dir = tmp_path / "audit"

    assert main(
        [
            "--results-root",
            str(results_root),
            "--score-history-csv",
            str(score_history),
            "--output-dir",
            str(output_dir),
            "--repo-root",
            str(tmp_path),
        ],
    ) == 0

    payload = json.loads((output_dir / "summary.json").read_text())
    assert payload["local_candidate_count"] == 1
    assert "local_submission_provenance.md" in capsys.readouterr().out
