from __future__ import annotations

import csv

from experiments.filter_gsdc2023_safe_submission_candidates import main, safe_submission_shortlist


def _write_screen(path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "filename",
        "sha256",
        "submitted_filename",
        "duplicate_submitted_local_sha",
        "reference_score_m",
        "reference_p95_m",
        "reference_max_m",
        "risky_previous_changed_rows",
        "risky_previous_max_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_safe_submission_shortlist_filters_merges_score_audit_and_deduplicates(tmp_path) -> None:
    screen = tmp_path / "screen.csv"
    _write_screen(
        screen,
        [
            {
                "path": "a.csv",
                "filename": "submission_trip_weight_single_a.csv",
                "sha256": "aaa",
                "submitted_filename": "False",
                "duplicate_submitted_local_sha": "False",
                "reference_score_m": "0.1",
                "reference_p95_m": "0.2",
                "reference_max_m": "0.3",
                "risky_previous_changed_rows": "0",
                "risky_previous_max_m": "0.0",
            },
            {
                "path": "duplicate.csv",
                "filename": "submission_trip_weight_single_a_copy.csv",
                "sha256": "aaa",
                "submitted_filename": "False",
                "duplicate_submitted_local_sha": "False",
                "reference_score_m": "0.1",
                "reference_p95_m": "0.2",
                "reference_max_m": "0.3",
                "risky_previous_changed_rows": "0",
                "risky_previous_max_m": "0.0",
            },
            {
                "path": "submitted.csv",
                "filename": "submission_submitted.csv",
                "sha256": "bbb",
                "submitted_filename": "True",
                "duplicate_submitted_local_sha": "False",
                "reference_score_m": "0.0",
                "reference_p95_m": "0.0",
                "reference_max_m": "0.0",
                "risky_previous_changed_rows": "0",
                "risky_previous_max_m": "0.0",
            },
            {
                "path": "risky.csv",
                "filename": "submission_risky.csv",
                "sha256": "ccc",
                "submitted_filename": "False",
                "duplicate_submitted_local_sha": "False",
                "reference_score_m": "0.0",
                "reference_p95_m": "0.0",
                "reference_max_m": "0.0",
                "risky_previous_changed_rows": "1",
                "risky_previous_max_m": "1.0",
            },
        ],
    )
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "filename,copies,groups,score_log_status,score_log_public,score_log_private\n"
        "submission_trip_weight_single_a.csv,1,grp,not_found_in_score_logs,,\n",
        encoding="utf-8",
    )
    out = tmp_path / "shortlist.csv"

    rows = safe_submission_shortlist(
        screens=[("screen_a", screen)],
        output_csv=out,
        score_audits=[audit],
    )

    assert len(rows) == 1
    assert rows[0]["filename"] == "submission_trip_weight_single_a.csv"
    assert rows[0]["candidate_family"] == "trip_weight_single"
    assert rows[0]["recommended_action"] == "discovery_only"
    assert rows[0]["score_log_status"] == "not_found_in_score_logs"
    csv_rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(csv_rows) == 1
    assert out.with_suffix(".summary.json").is_file()


def test_safe_submission_shortlist_cli(tmp_path, capsys) -> None:
    screen = tmp_path / "screen.csv"
    _write_screen(
        screen,
        [
            {
                "path": "a.csv",
                "filename": "submission_private_floor_weighted_best_p3p25_a0p125.csv",
                "sha256": "aaa",
                "submitted_filename": "False",
                "duplicate_submitted_local_sha": "False",
                "reference_score_m": "0.1",
                "reference_p95_m": "0.2",
                "reference_max_m": "0.3",
                "risky_previous_changed_rows": "0",
                "risky_previous_max_m": "0.0",
            },
        ],
    )
    out = tmp_path / "shortlist.csv"

    assert main(["--screen", "weighted", str(screen), "--output-csv", str(out)]) == 0
    assert "shortlisted: 1 candidate(s)" in capsys.readouterr().out
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows[0]["candidate_family"] == "weighted_private_floor"
    assert rows[0]["recommended_action"] == "hold_bracketed_blend"
