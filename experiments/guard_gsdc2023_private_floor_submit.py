#!/usr/bin/env python3
"""Guard GSDC2023 candidate submissions against missing private-floor base state."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PRIVATE_FLOOR_HINTS = ("private_floor", "privatefloor", "mtv700")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read JSON object from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _candidate_rows(screen_csvs: tuple[Path, ...], candidate_csvs: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for screen_csv in screen_csvs:
        for row in _read_csv_rows(screen_csv):
            row["input_kind"] = "screen_csv"
            row["input_source"] = str(screen_csv)
            rows.append(row)
    for candidate_csv in candidate_csvs:
        rows.append(
            {
                "input_kind": "candidate_csv",
                "input_source": str(candidate_csv),
                "path": str(candidate_csv),
                "filename": candidate_csv.name,
            },
        )
    return rows


def _private_floor_reconstructable(audit: dict[str, Any]) -> bool:
    return bool(audit.get("private_floor_reconstructable_from_available_files", False))


def _private_floor_reference_like(path_text: str) -> bool:
    lowered = path_text.lower()
    return any(hint in lowered for hint in PRIVATE_FLOOR_HINTS)


def _source_family_reference_blockers(source_family_summaries: tuple[Path, ...]) -> list[str]:
    blockers: list[str] = []
    for path in source_family_summaries:
        payload = _read_json_object(path)
        reference = str(payload.get("reference", ""))
        if not reference:
            blockers.append(f"{path}:missing_reference")
        elif not _private_floor_reference_like(reference):
            blockers.append(f"{path}:reference_not_private_floor:{reference}")
    return blockers


def _recovery_dependencies_ready(summary: dict[str, Any] | None) -> bool | None:
    if summary is None:
        return None
    if "recoverable_from_current_workspace" in summary:
        return bool(summary.get("recoverable_from_current_workspace"))
    if "direct_private_floor_builder_ready" in summary:
        return bool(summary.get("direct_private_floor_builder_ready"))
    return False


def _row_blockers(
    row: dict[str, Any],
    *,
    audit_reconstructable: bool,
    recovery_dependencies_ready: bool | None,
    source_family_blockers: list[str],
) -> list[str]:
    blockers: list[str] = []
    if not audit_reconstructable:
        blockers.append("private_floor_not_reconstructable")
    if recovery_dependencies_ready is False:
        blockers.append("recovery_dependencies_not_available")
    if source_family_blockers:
        blockers.append("source_family_reference_not_private_floor")
    if "coordinate_sanity_pass" in row and not _as_bool(row.get("coordinate_sanity_pass")):
        blockers.append("coordinate_sanity_failed")
    if _as_bool(row.get("submitted_filename")):
        blockers.append("already_submitted_filename")
    if _as_bool(row.get("duplicate_submitted_local_sha")):
        blockers.append("duplicate_submitted_local_sha")
    return blockers


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# GSDC2023 Submit Guard",
        "",
        f"- Submit allowed: `{payload['submit_allowed']}`",
        f"- Candidate count: `{payload['candidate_count']}`",
        f"- Blocked count: `{payload['blocked_count']}`",
        f"- Private-floor reconstructable: `{payload['private_floor_reconstructable']}`",
        f"- Recovery dependencies ready: `{payload['recovery_dependencies_ready']}`",
        f"- Read: {payload['read']}",
        "",
        "## Candidates",
        "",
        "| Status | File | Blockers | Source |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| `{status}` | `{filename}` | {blockers} | `{source}` |".format(
                status=row["guard_status"],
                filename=row.get("filename", ""),
                blockers=row["guard_blockers"],
                source=row.get("input_source", ""),
            ),
        )
    lines.append("")
    return "\n".join(lines)


def guard_private_floor_submit(
    *,
    private_floor_audit_summary: Path,
    output_dir: Path,
    screen_csvs: tuple[Path, ...] = (),
    candidate_csvs: tuple[Path, ...] = (),
    source_family_summaries: tuple[Path, ...] = (),
    recovery_dependencies_summary: Path | None = None,
) -> dict[str, Any]:
    audit = _read_json_object(private_floor_audit_summary)
    audit_reconstructable = _private_floor_reconstructable(audit)
    recovery_summary = _read_json_object(recovery_dependencies_summary) if recovery_dependencies_summary else None
    recovery_ready = _recovery_dependencies_ready(recovery_summary)
    source_family_blockers = _source_family_reference_blockers(source_family_summaries)
    rows = _candidate_rows(screen_csvs, candidate_csvs)
    if not rows:
        raise SystemExit("submit guard requires at least one --screen-csv or --candidate-csv")

    guarded_rows: list[dict[str, Any]] = []
    for row in rows:
        blockers = _row_blockers(
            row,
            audit_reconstructable=audit_reconstructable,
            recovery_dependencies_ready=recovery_ready,
            source_family_blockers=source_family_blockers,
        )
        guarded = dict(row)
        guarded["guard_status"] = "blocked" if blockers else "allowed"
        guarded["guard_blockers"] = ";".join(blockers)
        guarded_rows.append(guarded)

    blocked_count = sum(1 for row in guarded_rows if row["guard_status"] == "blocked")
    submit_allowed = blocked_count == 0
    output_dir.mkdir(parents=True, exist_ok=True)
    report_csv = output_dir / "submit_guard_report.csv"
    fieldnames = [
        "guard_status",
        "guard_blockers",
        "filename",
        "path",
        "sha256",
        "submitted_filename",
        "duplicate_submitted_local_sha",
        "coordinate_sanity_pass",
        "recommended_action",
        "candidate_family",
        "screen_label",
        "input_kind",
        "input_source",
    ]
    _write_csv(report_csv, guarded_rows, fieldnames=fieldnames)

    payload = {
        "private_floor_audit_summary": str(private_floor_audit_summary),
        "private_floor_reconstructable": audit_reconstructable,
        "private_floor_read": audit.get("read", ""),
        "recovery_dependencies_summary": str(recovery_dependencies_summary) if recovery_dependencies_summary else "",
        "recovery_dependencies_ready": recovery_ready,
        "recovery_dependencies_missing_core_artifacts": (
            recovery_summary.get("missing_core_artifacts", []) if recovery_summary else []
        ),
        "recovery_dependencies_available_routes": (
            recovery_summary.get("available_routes", []) if recovery_summary else []
        ),
        "source_family_summaries": [str(path) for path in source_family_summaries],
        "source_family_blockers": source_family_blockers,
        "candidate_count": len(guarded_rows),
        "blocked_count": blocked_count,
        "allowed_count": len(guarded_rows) - blocked_count,
        "submit_allowed": submit_allowed,
        "report_csv": str(report_csv),
        "read": (
            "Submit is blocked; recover/reconstruct a private-floor base before spending a Kaggle submit."
            if not submit_allowed
            else "Submit guard passed for the supplied candidates."
        ),
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = output_dir / "submit_guard.md"
    md_path.write_text(_render_markdown(payload, guarded_rows), encoding="utf-8")
    print(f"saved: {summary_json}")
    print(f"saved: {report_csv}")
    print(f"saved: {md_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-floor-audit-summary", type=Path, required=True)
    parser.add_argument("--screen-csv", action="append", type=Path, default=[])
    parser.add_argument("--candidate-csv", action="append", type=Path, default=[])
    parser.add_argument("--source-family-summary", action="append", type=Path, default=[])
    parser.add_argument("--recovery-dependencies-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fail-on-blocked", action="store_true")
    args = parser.parse_args(argv)

    payload = guard_private_floor_submit(
        private_floor_audit_summary=args.private_floor_audit_summary,
        output_dir=args.output_dir,
        screen_csvs=tuple(args.screen_csv),
        candidate_csvs=tuple(args.candidate_csv),
        source_family_summaries=tuple(args.source_family_summary),
        recovery_dependencies_summary=args.recovery_dependencies_summary,
    )
    if args.fail_on_blocked and not bool(payload["submit_allowed"]):
        print(payload["read"])
        return 2
    print(f"submit_allowed={payload['submit_allowed']} blocked={payload['blocked_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
