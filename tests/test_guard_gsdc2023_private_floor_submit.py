from __future__ import annotations

import csv
import json

from experiments.guard_gsdc2023_private_floor_submit import guard_private_floor_submit, main


def _write_audit(path, *, reconstructable: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "private_floor_reconstructable_from_available_files": reconstructable,
                "read": "audit read",
            },
        ),
        encoding="utf-8",
    )


def _write_screen(path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "filename",
        "path",
        "sha256",
        "submitted_filename",
        "duplicate_submitted_local_sha",
        "coordinate_sanity_pass",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_recovery(path, *, recoverable: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "recoverable_from_current_workspace": recoverable,
                "direct_private_floor_builder_ready": recoverable,
                "missing_core_artifacts": [] if recoverable else ["basecorr_builder_input"],
                "available_routes": ["direct_basecorr_private_floor_builder"] if recoverable else [],
            },
        ),
        encoding="utf-8",
    )


def test_guard_blocks_when_private_floor_is_not_reconstructable(tmp_path) -> None:
    audit = tmp_path / "audit.json"
    screen = tmp_path / "screen.csv"
    output_dir = tmp_path / "guard"
    _write_audit(audit, reconstructable=False)
    _write_screen(
        screen,
        [
            {
                "filename": "submission_source_family_only_pixel4.csv",
                "path": "candidate.csv",
                "sha256": "abc",
                "submitted_filename": "False",
                "duplicate_submitted_local_sha": "False",
                "coordinate_sanity_pass": "True",
            },
        ],
    )

    payload = guard_private_floor_submit(
        private_floor_audit_summary=audit,
        screen_csvs=(screen,),
        output_dir=output_dir,
    )

    assert payload["submit_allowed"] is False
    assert payload["blocked_count"] == 1
    rows = list(csv.DictReader((output_dir / "submit_guard_report.csv").open()))
    assert rows[0]["guard_status"] == "blocked"
    assert "private_floor_not_reconstructable" in rows[0]["guard_blockers"]


def test_guard_blocks_when_recovery_dependencies_are_not_available(tmp_path) -> None:
    audit = tmp_path / "audit.json"
    recovery = tmp_path / "recovery.json"
    screen = tmp_path / "screen.csv"
    output_dir = tmp_path / "guard"
    _write_audit(audit, reconstructable=True)
    _write_recovery(recovery, recoverable=False)
    _write_screen(
        screen,
        [
            {
                "filename": "submission_private_floor_pixel4_probe.csv",
                "path": "candidate.csv",
                "sha256": "abc",
                "submitted_filename": "False",
                "duplicate_submitted_local_sha": "False",
                "coordinate_sanity_pass": "True",
            },
        ],
    )

    payload = guard_private_floor_submit(
        private_floor_audit_summary=audit,
        recovery_dependencies_summary=recovery,
        screen_csvs=(screen,),
        output_dir=output_dir,
    )

    assert payload["submit_allowed"] is False
    assert payload["recovery_dependencies_ready"] is False
    rows = list(csv.DictReader((output_dir / "submit_guard_report.csv").open()))
    assert "recovery_dependencies_not_available" in rows[0]["guard_blockers"]


def test_guard_blocks_source_family_non_private_floor_reference(tmp_path) -> None:
    audit = tmp_path / "audit.json"
    summary = tmp_path / "source_family_summary.json"
    output_dir = tmp_path / "guard"
    candidate = tmp_path / "submission_source_family_only_pixel4.csv"
    _write_audit(audit, reconstructable=True)
    summary.write_text(json.dumps({"reference": "experiments/results/gsdc2023_submission_v15.csv"}), encoding="utf-8")
    candidate.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")

    payload = guard_private_floor_submit(
        private_floor_audit_summary=audit,
        candidate_csvs=(candidate,),
        source_family_summaries=(summary,),
        output_dir=output_dir,
    )

    assert payload["submit_allowed"] is False
    rows = list(csv.DictReader((output_dir / "submit_guard_report.csv").open()))
    assert "source_family_reference_not_private_floor" in rows[0]["guard_blockers"]


def test_guard_allows_reconstructable_private_floor_without_other_blockers(tmp_path) -> None:
    audit = tmp_path / "audit.json"
    screen = tmp_path / "screen.csv"
    output_dir = tmp_path / "guard"
    _write_audit(audit, reconstructable=True)
    _write_screen(
        screen,
        [
            {
                "filename": "submission_private_floor_pixel4_probe.csv",
                "path": "candidate.csv",
                "sha256": "abc",
                "submitted_filename": "False",
                "duplicate_submitted_local_sha": "False",
                "coordinate_sanity_pass": "True",
            },
        ],
    )

    payload = guard_private_floor_submit(
        private_floor_audit_summary=audit,
        screen_csvs=(screen,),
        output_dir=output_dir,
    )

    assert payload["submit_allowed"] is True
    assert payload["blocked_count"] == 0


def test_guard_cli_fail_on_blocked_returns_two(tmp_path, capsys) -> None:
    audit = tmp_path / "audit.json"
    screen = tmp_path / "screen.csv"
    _write_audit(audit, reconstructable=False)
    _write_screen(
        screen,
        [
            {
                "filename": "submission_bridge.csv",
                "path": "candidate.csv",
                "sha256": "abc",
                "submitted_filename": "False",
                "duplicate_submitted_local_sha": "False",
                "coordinate_sanity_pass": "True",
            },
        ],
    )

    code = main(
        [
            "--private-floor-audit-summary",
            str(audit),
            "--screen-csv",
            str(screen),
            "--output-dir",
            str(tmp_path / "guard"),
            "--fail-on-blocked",
        ],
    )

    assert code == 2
    assert "Submit is blocked" in capsys.readouterr().out
