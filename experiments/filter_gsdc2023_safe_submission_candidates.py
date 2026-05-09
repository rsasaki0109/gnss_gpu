#!/usr/bin/env python3
"""Build a safe/unsubmitted shortlist from local GSDC2023 submission screens."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _as_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _screen_rows(label: str, path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["screen_label"] = label
        item["screen_csv"] = str(path)
        out.append(item)
    return out


def _score_audit_by_filename(paths: list[Path]) -> dict[str, dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                filename = row.get("filename")
                if filename:
                    by_name[filename] = row
    return by_name


def _safe_candidate(row: dict[str, Any], *, max_risky_previous_m: float) -> bool:
    if _as_bool(row.get("submitted_filename")):
        return False
    if _as_bool(row.get("duplicate_submitted_local_sha")):
        return False
    if _as_int(row.get("risky_previous_changed_rows")) != 0:
        return False
    if _as_float(row.get("risky_previous_max_m")) > max_risky_previous_m:
        return False
    return True


def _family_from_row(row: dict[str, Any]) -> str:
    label = str(row.get("screen_label", ""))
    filename = str(row.get("filename", ""))
    if "trip_weight_leave_group_out" in filename:
        return "trip_weight_leave_group_out"
    if "trip_weight_leave_one_out" in filename:
        return "trip_weight_leave_one_out"
    if "trip_weight_single" in filename:
        return "trip_weight_single"
    if "weighted_best" in filename:
        return "weighted_private_floor"
    if label:
        return label
    return "screen"


def _recommendation(row: dict[str, Any]) -> tuple[str, str]:
    status = str(row.get("score_log_status", ""))
    filename = str(row.get("filename", ""))
    if status == "submitted_in_score_log":
        private = row.get("score_log_private", "")
        return "reject_known_score", f"already scored in local logs with private={private}"
    if "raw_wls" in filename:
        return "reject_spike_risk", "raw WLS candidate family has known large local spikes"
    if "weighted_best_p3p25" in filename:
        return "hold_bracketed_blend", "p3p25 blend family is bracketed by submitted alpha results"
    if "weighted_best_p3p0" in filename:
        return "hold_public_only_blend", "p3p0 source already lost private floor in prior A/B"
    if "trip_weight" in filename:
        return "discovery_only", "trip-weight candidate is useful only as explicit split-discovery"
    return "review_manually", "safe screen gates pass but no stronger positive signal is recorded"


def safe_submission_shortlist(
    *,
    screens: list[tuple[str, Path]],
    output_csv: Path,
    score_audits: list[Path] | None = None,
    max_risky_previous_m: float = 1.0e-6,
    deduplicate_sha: bool = True,
) -> list[dict[str, Any]]:
    audit_by_filename = _score_audit_by_filename(list(score_audits or []))
    rows: list[dict[str, Any]] = []
    for label, path in screens:
        rows.extend(_screen_rows(label, path))

    kept: list[dict[str, Any]] = []
    seen_sha: set[str] = set()
    for row in rows:
        if not _safe_candidate(row, max_risky_previous_m=max_risky_previous_m):
            continue
        sha = str(row.get("sha256", ""))
        if deduplicate_sha and sha and sha in seen_sha:
            continue
        if sha:
            seen_sha.add(sha)
        audit = audit_by_filename.get(str(row.get("filename", "")), {})
        item = dict(row)
        for key in ("copies", "groups", "score_log_status", "score_log_public", "score_log_private"):
            item[key] = audit.get(key, "")
        item["candidate_family"] = _family_from_row(item)
        action, reason = _recommendation(item)
        item["recommended_action"] = action
        item["recommendation_reason"] = reason
        kept.append(item)

    kept = sorted(
        kept,
        key=lambda row: (
            str(row.get("recommended_action", "")),
            _as_float(row.get("reference_score_m"), 0.0),
            _as_float(row.get("reference_max_m"), 0.0),
            str(row.get("filename", "")),
        ),
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "screen_label",
        "candidate_family",
        "recommended_action",
        "recommendation_reason",
        "filename",
        "path",
        "sha256",
        "submitted_filename",
        "duplicate_submitted_local_sha",
        "risky_previous_changed_rows",
        "risky_previous_max_m",
        "reference_changed_rows",
        "reference_score_m",
        "reference_p95_m",
        "reference_max_m",
        "score_log_status",
        "score_log_public",
        "score_log_private",
        "copies",
        "groups",
        "screen_csv",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    summary = {
        "screen_count": len(screens),
        "input_row_count": len(rows),
        "shortlist_count": len(kept),
        "deduplicate_sha": bool(deduplicate_sha),
        "max_risky_previous_m": float(max_risky_previous_m),
        "by_recommended_action": {
            action: sum(1 for row in kept if row.get("recommended_action") == action)
            for action in sorted({str(row.get("recommended_action", "")) for row in kept})
        },
        "by_screen_label": {
            label: sum(1 for row in kept if row.get("screen_label") == label)
            for label, _path in screens
        },
    }
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return kept


def _screen_arg(value: list[str]) -> tuple[str, Path]:
    if len(value) != 2:
        raise argparse.ArgumentTypeError("--screen requires LABEL PATH")
    return value[0], Path(value[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", action="append", nargs=2, metavar=("LABEL", "CSV"), required=True)
    parser.add_argument("--score-audit-csv", action="append", type=Path, default=[])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-risky-previous-m", type=float, default=1.0e-6)
    parser.add_argument("--keep-duplicate-sha", action="store_true")
    args = parser.parse_args(argv)

    rows = safe_submission_shortlist(
        screens=[_screen_arg(item) for item in args.screen],
        output_csv=args.output_csv,
        score_audits=args.score_audit_csv,
        max_risky_previous_m=args.max_risky_previous_m,
        deduplicate_sha=not args.keep_duplicate_sha,
    )
    print(f"shortlisted: {len(rows)} candidate(s)")
    print(f"saved: {args.output_csv}")
    print(f"saved: {args.output_csv.with_suffix('.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
