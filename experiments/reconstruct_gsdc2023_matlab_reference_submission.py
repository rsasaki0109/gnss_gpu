"""Reconstruct a MATLAB-reference GSDC2023 submission from bridge source rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.analyze_gsdc2023_all_trip_bridge_source_delta import (
    analyze_all_trip_bridge_source_delta,
    reconstruct_candidate_submission,
)
from experiments.analyze_gsdc2023_source_ab import compare_submissions, comparison_summary, read_submission


KEY_COLUMNS = ["tripId", "UnixTimeMillis"]
SOURCE_COORDINATE_COLUMNS = ["best_source_latitude_degrees", "best_source_longitude_degrees"]
REFERENCE_SOURCE_COORDINATE_COLUMNS = [
    "best_reference_source_latitude_degrees",
    "best_reference_source_longitude_degrees",
]
SUBMISSION_COORDINATE_COLUMNS = ["LatitudeDegrees", "LongitudeDegrees"]


def _coordinate_columns(row_summary: pd.DataFrame) -> list[str]:
    if set(SOURCE_COORDINATE_COLUMNS).issubset(row_summary.columns):
        return SOURCE_COORDINATE_COLUMNS
    if set(REFERENCE_SOURCE_COORDINATE_COLUMNS).issubset(row_summary.columns):
        return REFERENCE_SOURCE_COORDINATE_COLUMNS
    raise ValueError(
        "row summary is missing coordinate columns: expected "
        f"{SOURCE_COORDINATE_COLUMNS} or {REFERENCE_SOURCE_COORDINATE_COLUMNS}",
    )


def apply_row_summary_coordinates(
    submission: pd.DataFrame,
    row_summary: pd.DataFrame,
    *,
    source_label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    coordinate_columns = _coordinate_columns(row_summary)
    required = set(KEY_COLUMNS + coordinate_columns)
    missing = required.difference(row_summary.columns)
    if missing:
        raise ValueError(f"row summary is missing columns: {sorted(missing)}")
    if row_summary[KEY_COLUMNS].duplicated().any():
        raise ValueError("row summary contains duplicate tripId/UnixTimeMillis values")

    patch = row_summary[KEY_COLUMNS + coordinate_columns].copy()
    patch = patch.rename(
        columns={
            coordinate_columns[0]: SOURCE_COORDINATE_COLUMNS[0],
            coordinate_columns[1]: SOURCE_COORDINATE_COLUMNS[1],
        },
    )
    keyed_submission = submission[KEY_COLUMNS].copy()
    joined = keyed_submission.merge(patch, on=KEY_COLUMNS, how="left", validate="one_to_one")
    replace_mask = joined[SOURCE_COORDINATE_COLUMNS].notna().all(axis=1)
    expected_keys = set(zip(patch["tripId"], patch["UnixTimeMillis"]))
    matched_keys = set(zip(joined.loc[replace_mask, "tripId"], joined.loc[replace_mask, "UnixTimeMillis"]))
    missing_keys = expected_keys.difference(matched_keys)
    if missing_keys:
        raise ValueError(f"row summary has {len(missing_keys)} key(s) absent from candidate submission")

    out = submission.copy()
    out.loc[replace_mask, SUBMISSION_COORDINATE_COLUMNS] = joined.loc[replace_mask, SOURCE_COORDINATE_COLUMNS].to_numpy()
    rows_by_trip = joined.loc[replace_mask, "tripId"].value_counts().sort_index().to_dict()
    return out, {
        "source_label": source_label,
        "rows_replaced": int(replace_mask.sum()),
        "rows_by_trip": {str(trip): int(count) for trip, count in rows_by_trip.items()},
    }


def reconstruct_matlab_reference_submission(
    *,
    reference_submission: Path,
    candidate_submission: Path,
    bridge_root: Path,
    override_row_summaries: list[tuple[str, Path]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows, trips, source_runs, scan_summary = analyze_all_trip_bridge_source_delta(
        reference_submission=reference_submission,
        bridge_root=bridge_root,
    )
    reconstructed, base_reconstruction = reconstruct_candidate_submission(candidate_submission, rows)

    override_summaries: list[dict[str, object]] = []
    for label, path in override_row_summaries or []:
        override_rows = pd.read_csv(path)
        reconstructed, override_summary = apply_row_summary_coordinates(
            reconstructed,
            override_rows,
            source_label=label,
        )
        override_summary["path"] = str(path)
        override_summaries.append(override_summary)

    reference = read_submission(reference_submission)
    row_delta, trip_delta = compare_submissions(reference, reconstructed, "reconstructed")
    delta_summary = comparison_summary(row_delta).iloc[0].to_dict()
    summary = {
        "reference_submission": str(reference_submission),
        "candidate_submission": str(candidate_submission),
        "bridge_root": str(bridge_root),
        "base_reconstruction": base_reconstruction,
        "override_reconstructions": override_summaries,
        "bridge_scan": scan_summary,
        "delta_vs_reference": delta_summary,
    }
    return reconstructed, rows, trips, source_runs, trip_delta, summary


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected LABEL=path, got {value!r}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"expected non-empty LABEL in {value!r}")
    return label, Path(path)


def write_outputs(
    output_dir: Path,
    reconstructed: pd.DataFrame,
    rows: pd.DataFrame,
    trips: pd.DataFrame,
    source_runs: pd.DataFrame,
    trip_delta: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstructed_path = output_dir / "submission_reconstructed_matlab_reference.csv"
    rows_path = output_dir / "all_trip_bridge_source_delta_rows.csv"
    trips_path = output_dir / "all_trip_bridge_source_delta_trips.csv"
    source_runs_path = output_dir / "all_trip_bridge_source_runs.csv"
    trip_delta_path = output_dir / "trip_delta_vs_reference.csv"
    reconstructed.to_csv(reconstructed_path, index=False)
    rows.to_csv(rows_path, index=False)
    trips.to_csv(trips_path, index=False)
    source_runs.to_csv(source_runs_path, index=False)
    trip_delta.to_csv(trip_delta_path, index=False)
    payload = {
        **summary,
        "reconstructed_submission_csv": str(reconstructed_path),
        "rows_csv": str(rows_path),
        "trip_summary_csv": str(trips_path),
        "source_runs_csv": str(source_runs_path),
        "trip_delta_csv": str(trip_delta_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-submission", type=Path, required=True)
    parser.add_argument("--candidate-submission", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--override-row-summary", action="append", default=[], help="LABEL=rows.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    reconstructed, rows, trips, source_runs, trip_delta, summary = reconstruct_matlab_reference_submission(
        reference_submission=args.reference_submission,
        candidate_submission=args.candidate_submission,
        bridge_root=args.bridge_root,
        override_row_summaries=[_parse_label_path(value) for value in args.override_row_summary],
    )
    write_outputs(args.output_dir, reconstructed, rows, trips, source_runs, trip_delta, summary)
    delta = summary["delta_vs_reference"]
    print(
        "reconstructed: "
        f"rows={delta['rows']} p95={delta['p95_delta_m']:.6g}m max={delta['max_delta_m']:.6g}m",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
