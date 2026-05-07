#!/usr/bin/env python3
"""Audit final-submission equivalence against a MATLAB/reference GSDC2023 CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.build_gsdc2023_pre_submit_manifest import DELTA_CHANGED_THRESHOLD_M, sha256_file
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


REQUIRED_COLUMNS = ("tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees")


def _read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    return frame


def _submission_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.expanduser().resolve().rglob("submission*.csv")
        if path.is_file()
        and "trip_summary" not in path.name
        and not path.name.startswith("submissions_by_")
    )


def _score_logs(paths: list[Path]) -> dict[str, dict[str, str]]:
    by_filename: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.expanduser().open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                filename = row.get("fileName")
                if filename:
                    by_filename[filename] = row
    return by_filename


def _candidate_delta(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, float | int | str]:
    if len(reference) != len(candidate):
        return {
            "status": "row_count_mismatch",
            "rows": int(len(candidate)),
            "matched_rows": 0,
            "changed_rows": None,
            "score_m": None,
            "p50_m": None,
            "p95_m": None,
            "max_m": None,
        }
    for column in ("tripId", "UnixTimeMillis"):
        if not reference[column].equals(candidate[column]):
            return {
                "status": f"{column}_mismatch",
                "rows": int(len(candidate)),
                "matched_rows": 0,
                "changed_rows": None,
                "score_m": None,
                "p50_m": None,
                "p95_m": None,
                "max_m": None,
            }
    delta_m = haversine_m(
        reference["LatitudeDegrees"].to_numpy(),
        reference["LongitudeDegrees"].to_numpy(),
        candidate["LatitudeDegrees"].to_numpy(),
        candidate["LongitudeDegrees"].to_numpy(),
    )
    score = gsdc_score_m(delta_m)
    return {
        "status": "compared",
        "rows": int(len(candidate)),
        "matched_rows": int(len(candidate)),
        "changed_rows": int(np.count_nonzero(delta_m > DELTA_CHANGED_THRESHOLD_M)),
        "score_m": float(score["score_m"]),
        "p50_m": float(score["p50_m"]),
        "p95_m": float(score["p95_m"]),
        "max_m": float(score["max_m"]),
    }


def audit_matlab_submission_score_equivalence(
    *,
    matlab_reference: Path,
    output_csv: Path,
    candidate_roots: list[Path] | None = None,
    candidate_paths: list[Path] | None = None,
    submitted_csvs: list[Path] | None = None,
) -> list[dict[str, Any]]:
    reference_path = matlab_reference.expanduser().resolve()
    reference = _read_submission(reference_path)
    reference_sha = sha256_file(reference_path)
    score_by_filename = _score_logs(list(submitted_csvs or []))

    paths: list[Path] = []
    for root in candidate_roots or []:
        paths.extend(_submission_paths(root))
    paths.extend(path.expanduser().resolve() for path in candidate_paths or [])
    unique_paths = sorted({path.resolve() for path in paths})

    rows: list[dict[str, Any]] = []
    for path in unique_paths:
        path_sha = sha256_file(path)
        score_row = score_by_filename.get(path.name, {})
        item: dict[str, Any] = {
            "path": str(path),
            "filename": path.name,
            "sha256": path_sha,
            "reference_sha256": reference_sha,
            "byte_identical": path_sha == reference_sha,
            "score_log_status": "submitted_in_score_log" if score_row else "not_found_in_score_logs",
            "score_log_public": score_row.get("publicScore", ""),
            "score_log_private": score_row.get("privateScore", ""),
            "score_log_description": score_row.get("description", ""),
        }
        try:
            candidate = _read_submission(path)
            item.update(_candidate_delta(reference, candidate))
        except Exception as exc:  # noqa: BLE001 - keep audit rows for malformed local artifacts.
            item.update(
                {
                    "status": "read_error",
                    "error": str(exc),
                    "rows": None,
                    "matched_rows": 0,
                    "changed_rows": None,
                    "score_m": None,
                    "p50_m": None,
                    "p95_m": None,
                    "max_m": None,
                },
            )
        rows.append(item)

    rows = sorted(
        rows,
        key=lambda row: (
            not bool(row.get("byte_identical")),
            str(row.get("status")) != "compared",
            float(row.get("score_m") if row.get("score_m") not in (None, "") else float("inf")),
            float(row.get("max_m") if row.get("max_m") not in (None, "") else float("inf")),
            str(row.get("filename", "")),
        ),
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "filename",
        "sha256",
        "reference_sha256",
        "byte_identical",
        "status",
        "rows",
        "matched_rows",
        "changed_rows",
        "score_m",
        "p50_m",
        "p95_m",
        "max_m",
        "score_log_status",
        "score_log_public",
        "score_log_private",
        "score_log_description",
        "error",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    compared = [row for row in rows if row.get("status") == "compared"]
    closest = compared[0] if compared else None
    summary = {
        "matlab_reference": str(reference_path),
        "matlab_reference_sha256": reference_sha,
        "matlab_reference_rows": int(len(reference)),
        "candidate_count": int(len(rows)),
        "compared_count": int(len(compared)),
        "byte_identical_count": int(sum(1 for row in rows if row.get("byte_identical"))),
        "submitted_score_log_count": int(sum(1 for row in rows if row.get("score_log_status") == "submitted_in_score_log")),
        "closest": {
            key: closest.get(key)
            for key in (
                "filename",
                "path",
                "sha256",
                "byte_identical",
                "changed_rows",
                "score_m",
                "p95_m",
                "max_m",
                "score_log_public",
                "score_log_private",
            )
        }
        if closest is not None
        else None,
    }
    output_csv.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-reference", type=Path, required=True)
    parser.add_argument("--candidate-root", action="append", type=Path, default=[])
    parser.add_argument("--candidate", action="append", type=Path, default=[])
    parser.add_argument("--submitted-csv", action="append", type=Path, default=[])
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = audit_matlab_submission_score_equivalence(
        matlab_reference=args.matlab_reference,
        candidate_roots=args.candidate_root,
        candidate_paths=args.candidate,
        submitted_csvs=args.submitted_csv,
        output_csv=args.output_csv,
    )
    print(f"audited: {len(rows)} candidate(s)")
    print(f"saved: {args.output_csv}")
    print(f"saved: {args.output_csv.with_suffix('.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
