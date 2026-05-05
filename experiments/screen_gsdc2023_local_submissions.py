#!/usr/bin/env python3
"""Screen local GSDC2023 submission CSVs against submitted and safe baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.build_gsdc2023_pre_submit_manifest import (
    DEFAULT_RISKY_TRIPS,
    DELTA_CHANGED_THRESHOLD_M,
    sha256_file,
)
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


REQUIRED_COLUMNS = ("tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees")


def _read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")
    return frame


def _assert_same_keys(reference: pd.DataFrame, candidate: pd.DataFrame, *, label: str) -> None:
    if len(reference) != len(candidate):
        raise SystemExit(f"{label}: row count mismatch {len(reference)} != {len(candidate)}")
    for column in ("tripId", "UnixTimeMillis"):
        if not reference[column].equals(candidate[column]):
            raise SystemExit(f"{label}: {column} mismatch")


def _delta_summary(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, float | int]:
    deltas = haversine_m(
        reference["LatitudeDegrees"].to_numpy(),
        reference["LongitudeDegrees"].to_numpy(),
        candidate["LatitudeDegrees"].to_numpy(),
        candidate["LongitudeDegrees"].to_numpy(),
    )
    score = gsdc_score_m(deltas)
    return {
        "changed_rows": int(np.count_nonzero(deltas > DELTA_CHANGED_THRESHOLD_M)),
        "score_m": float(score["score_m"]),
        "p50_m": float(score["p50_m"]),
        "p95_m": float(score["p95_m"]),
        "max_m": float(score["max_m"]),
    }


def _submitted_filenames(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {row.get("fileName", "") for row in csv.DictReader(fh) if row.get("fileName")}


def _local_submission_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("submission*.csv")
        if path.is_file()
        and "trip_summary" not in path.name
        and not path.name.startswith("submissions_by_")
    )


def _submitted_local_sha(paths: list[Path], submitted_names: set[str]) -> set[str]:
    return {sha256_file(path) for path in paths if path.name in submitted_names}


def _risky_previous_delta(
    *,
    previous_safe: pd.DataFrame | None,
    candidate: pd.DataFrame,
    risky_trips: tuple[str, ...],
) -> dict[str, float | int | bool]:
    if previous_safe is None:
        return {
            "previous_safe_exists": False,
            "risky_previous_changed_rows": 0,
            "risky_previous_max_m": 0.0,
        }
    total_changed = 0
    max_delta = 0.0
    for trip in risky_trips:
        mask = previous_safe["tripId"] == trip
        if not bool(mask.any()):
            continue
        prev_trip = previous_safe[mask].reset_index(drop=True)
        cand_trip = candidate[mask].reset_index(drop=True)
        delta = _delta_summary(prev_trip, cand_trip)
        total_changed += int(delta["changed_rows"])
        max_delta = max(max_delta, float(delta["max_m"]))
    return {
        "previous_safe_exists": True,
        "risky_previous_changed_rows": total_changed,
        "risky_previous_max_m": max_delta,
    }


def screen_local_submissions(
    *,
    root: Path,
    output_csv: Path,
    submitted_csv: Path | None = None,
    reference_best: Path | None = None,
    previous_safe: Path | None = None,
    risky_trips: tuple[str, ...] = DEFAULT_RISKY_TRIPS,
) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    paths = _local_submission_paths(root)
    submitted_names = _submitted_filenames(submitted_csv.expanduser().resolve() if submitted_csv else None)
    submitted_shas = _submitted_local_sha(paths, submitted_names)

    reference_frame = _read_submission(reference_best.expanduser().resolve()) if reference_best else None
    previous_frame = _read_submission(previous_safe.expanduser().resolve()) if previous_safe else None

    rows: list[dict[str, Any]] = []
    for path in paths:
        path_sha = sha256_file(path)
        candidate_frame: pd.DataFrame | None = None
        delta_vs_reference = {
            "reference_changed_rows": None,
            "reference_score_m": None,
            "reference_p95_m": None,
            "reference_max_m": None,
        }
        risky_delta = {
            "previous_safe_exists": previous_frame is not None,
            "risky_previous_changed_rows": None,
            "risky_previous_max_m": None,
        }
        if reference_frame is not None or previous_frame is not None:
            candidate_frame = _read_submission(path)
        if reference_frame is not None and candidate_frame is not None:
            _assert_same_keys(reference_frame, candidate_frame, label=str(path))
            delta = _delta_summary(reference_frame, candidate_frame)
            delta_vs_reference = {
                "reference_changed_rows": delta["changed_rows"],
                "reference_score_m": delta["score_m"],
                "reference_p95_m": delta["p95_m"],
                "reference_max_m": delta["max_m"],
            }
        if previous_frame is not None and candidate_frame is not None:
            _assert_same_keys(previous_frame, candidate_frame, label=str(path))
            risky_delta = _risky_previous_delta(
                previous_safe=previous_frame,
                candidate=candidate_frame,
                risky_trips=risky_trips,
            )
        rows.append(
            {
                "path": str(path),
                "filename": path.name,
                "sha256": path_sha,
                "submitted_filename": path.name in submitted_names,
                "duplicate_submitted_local_sha": path_sha in submitted_shas,
                **delta_vs_reference,
                **risky_delta,
            },
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "path",
        "filename",
        "sha256",
        "submitted_filename",
        "duplicate_submitted_local_sha",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_csv.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "root": str(root),
                "output_csv": str(output_csv),
                "candidate_count": len(rows),
                "submitted_filename_count": sum(1 for row in rows if row["submitted_filename"]),
                "duplicate_submitted_local_sha_count": sum(
                    1 for row in rows if row["duplicate_submitted_local_sha"]
                ),
                "risky_previous_changed_count": sum(
                    1 for row in rows if int(row.get("risky_previous_changed_rows") or 0) > 0
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved: {output_csv}")
    print(f"saved: {summary_path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--submitted-csv", type=Path)
    parser.add_argument("--reference-best", type=Path)
    parser.add_argument("--previous-safe", type=Path)
    parser.add_argument("--risky-trip", action="append", dest="risky_trips")
    args = parser.parse_args(argv)

    rows = screen_local_submissions(
        root=args.root,
        output_csv=args.output_csv,
        submitted_csv=args.submitted_csv,
        reference_best=args.reference_best,
        previous_safe=args.previous_safe,
        risky_trips=tuple(args.risky_trips or DEFAULT_RISKY_TRIPS),
    )
    print(f"screened: {len(rows)} candidate(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
