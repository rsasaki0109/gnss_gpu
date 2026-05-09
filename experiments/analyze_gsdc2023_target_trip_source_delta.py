"""Decompose one GSDC submission trip against bridge source columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


KEY_COLUMNS = ["tripId", "UnixTimeMillis"]


def _source_name_from_lat_column(column: str) -> str:
    if column == "LatitudeDegrees":
        return "selected"
    prefix = column.removesuffix("LatitudeDegrees")
    tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\\d+", prefix)
    return "_".join(token.lower() for token in tokens if token) or prefix.lower()


def source_coordinate_columns(frame: pd.DataFrame) -> dict[str, tuple[str, str]]:
    columns: dict[str, tuple[str, str]] = {}
    for column in frame.columns:
        if not column.endswith("LatitudeDegrees"):
            continue
        lon_column = column.removesuffix("LatitudeDegrees") + "LongitudeDegrees"
        if lon_column not in frame.columns:
            continue
        if not frame[column].notna().any() or not frame[lon_column].notna().any():
            continue
        source_name = _source_name_from_lat_column(column)
        columns[source_name] = (column, lon_column)
    return columns


def _read_submission_trip(path: Path, target_trip: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS + ["LatitudeDegrees", "LongitudeDegrees"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    trip = frame[frame["tripId"] == target_trip].copy()
    if trip.empty:
        raise ValueError(f"{path} contains no rows for {target_trip!r}")
    return trip.sort_values("UnixTimeMillis").reset_index(drop=True)


def _read_bridge_rows(path: Path, target_trip: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "tripId" in frame.columns:
        frame = frame[frame["tripId"] == target_trip].copy()
    if "UnixTimeMillis" not in frame.columns:
        raise ValueError(f"{path} is missing UnixTimeMillis")
    if frame.empty:
        raise ValueError(f"{path} contains no bridge rows for {target_trip!r}")
    sources = source_coordinate_columns(frame)
    if not sources:
        raise ValueError(f"{path} contains no *LatitudeDegrees/*LongitudeDegrees source pairs")
    return frame.sort_values("UnixTimeMillis").reset_index(drop=True)


def _score_columns(delta_m: np.ndarray, suffix: str = "") -> dict[str, float | int]:
    score = gsdc_score_m(delta_m)
    return {
        f"mean{suffix}_m": score["mean_m"],
        f"p50{suffix}_m": score["p50_m"],
        f"p95{suffix}_m": score["p95_m"],
        f"max{suffix}_m": score["max_m"],
    }


def _count_columns(prefix: str, values: pd.Series) -> dict[str, int]:
    counts = values.value_counts(dropna=False).sort_index()
    return {f"{prefix}_{str(name).replace('-', '_')}_rows": int(count) for name, count in counts.items()}


def analyze_target_trip_source_delta(
    *,
    reference_submission: Path,
    candidate_submission: Path,
    bridge_rows: Path,
    target_trip: str,
    chunk_epochs: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if chunk_epochs <= 0:
        raise ValueError("chunk_epochs must be positive")

    reference = _read_submission_trip(reference_submission, target_trip)
    candidate = _read_submission_trip(candidate_submission, target_trip)
    bridge = _read_bridge_rows(bridge_rows, target_trip)
    source_columns = source_coordinate_columns(bridge)

    rows = reference.merge(
        candidate,
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    rows = rows.merge(bridge, on="UnixTimeMillis", how="inner", validate="one_to_one")
    if len(rows) != len(reference) or len(rows) != len(candidate):
        raise ValueError(
            f"matched row count mismatch for {target_trip}: "
            f"reference={len(reference)} candidate={len(candidate)} bridge={len(bridge)} matched={len(rows)}"
        )

    rows.insert(2, "epoch_index", np.arange(len(rows), dtype=np.int64))
    rows["candidate_delta_m"] = haversine_m(
        rows["LatitudeDegrees_reference"].to_numpy(),
        rows["LongitudeDegrees_reference"].to_numpy(),
        rows["LatitudeDegrees_candidate"].to_numpy(),
        rows["LongitudeDegrees_candidate"].to_numpy(),
    )
    for source, (lat_column, lon_column) in source_columns.items():
        rows[f"reference_to_{source}_m"] = haversine_m(
            rows["LatitudeDegrees_reference"].to_numpy(),
            rows["LongitudeDegrees_reference"].to_numpy(),
            rows[lat_column].to_numpy(),
            rows[lon_column].to_numpy(),
        )
        rows[f"candidate_to_{source}_m"] = haversine_m(
            rows["LatitudeDegrees_candidate"].to_numpy(),
            rows["LongitudeDegrees_candidate"].to_numpy(),
            rows[lat_column].to_numpy(),
            rows[lon_column].to_numpy(),
        )

    distance_columns = [f"reference_to_{source}_m" for source in source_columns]
    candidate_distance_columns = [f"candidate_to_{source}_m" for source in source_columns]
    distance_matrix = rows[distance_columns].to_numpy(dtype=np.float64)
    best_indices = np.nanargmin(distance_matrix, axis=1)
    source_names = list(source_columns)
    rows["best_reference_source"] = [source_names[index] for index in best_indices]
    rows["best_reference_source_distance_m"] = np.nanmin(distance_matrix, axis=1)
    candidate_distance_matrix = rows[candidate_distance_columns].to_numpy(dtype=np.float64)
    best_candidate_indices = np.nanargmin(candidate_distance_matrix, axis=1)
    rows["best_candidate_source"] = [source_names[index] for index in best_candidate_indices]
    rows["best_candidate_source_distance_m"] = np.nanmin(candidate_distance_matrix, axis=1)
    rows["best_reference_source_latitude_degrees"] = np.nan
    rows["best_reference_source_longitude_degrees"] = np.nan
    for source, (lat_column, lon_column) in source_columns.items():
        mask = rows["best_reference_source"] == source
        rows.loc[mask, "best_reference_source_latitude_degrees"] = rows.loc[mask, lat_column]
        rows.loc[mask, "best_reference_source_longitude_degrees"] = rows.loc[mask, lon_column]
    if "SelectedSource" not in rows.columns:
        rows["SelectedSource"] = ""

    chunk_rows: list[dict[str, object]] = []
    for start in range(0, len(rows), chunk_epochs):
        chunk = rows.iloc[start : start + chunk_epochs]
        candidate_delta = chunk["candidate_delta_m"].to_numpy()
        best_delta = chunk["best_reference_source_distance_m"].to_numpy()
        best_candidate_delta = chunk["best_candidate_source_distance_m"].to_numpy()
        record: dict[str, object] = {
            "tripId": target_trip,
            "start_epoch": int(start),
            "end_epoch": int(start + len(chunk)),
            "rows": int(len(chunk)),
            "changed_rows_gt_0p01m": int(np.count_nonzero(candidate_delta > 0.01)),
            "rows_gt_1m": int(np.count_nonzero(candidate_delta > 1.0)),
            "rows_gt_5m": int(np.count_nonzero(candidate_delta > 5.0)),
            **_score_columns(candidate_delta, "_candidate_delta"),
            **_score_columns(best_delta, "_best_reference_source_distance"),
            **_score_columns(best_candidate_delta, "_best_candidate_source_distance"),
        }
        record.update(_count_columns("best_reference_source", chunk["best_reference_source"]))
        record.update(_count_columns("best_candidate_source", chunk["best_candidate_source"]))
        record.update(_count_columns("selected_source", chunk["SelectedSource"].astype(str)))
        chunk_rows.append(record)
    chunk_summary = pd.DataFrame(chunk_rows)

    summary = {
        "reference_submission": str(reference_submission),
        "candidate_submission": str(candidate_submission),
        "bridge_rows": str(bridge_rows),
        "target_trip": target_trip,
        "rows": int(len(rows)),
        "source_columns": {source: [lat, lon] for source, (lat, lon) in source_columns.items()},
        "candidate_delta": {
            "rows_gt_0p01m": int(np.count_nonzero(rows["candidate_delta_m"].to_numpy() > 0.01)),
            "rows_gt_1m": int(np.count_nonzero(rows["candidate_delta_m"].to_numpy() > 1.0)),
            "rows_gt_5m": int(np.count_nonzero(rows["candidate_delta_m"].to_numpy() > 5.0)),
            **_score_columns(rows["candidate_delta_m"].to_numpy(), ""),
        },
        "best_reference_source_distance": _score_columns(rows["best_reference_source_distance_m"].to_numpy(), ""),
        "best_candidate_source_distance": _score_columns(rows["best_candidate_source_distance_m"].to_numpy(), ""),
        "best_reference_source_counts": rows["best_reference_source"].value_counts().sort_index().to_dict(),
        "best_candidate_source_counts": rows["best_candidate_source"].value_counts().sort_index().to_dict(),
        "selected_source_counts": rows["SelectedSource"].astype(str).value_counts().sort_index().to_dict(),
        "top_chunks_by_candidate_p95_delta": (
            chunk_summary.sort_values(
                ["p95_candidate_delta_m", "max_candidate_delta_m"],
                ascending=[False, False],
            )
            .head(10)
            .to_dict(orient="records")
        ),
        "top_rows_by_candidate_delta": (
            rows.sort_values("candidate_delta_m", ascending=False)
            .head(10)[
                [
                    "tripId",
                    "UnixTimeMillis",
                    "epoch_index",
                    "candidate_delta_m",
                    "SelectedSource",
                    "best_reference_source",
                    "best_reference_source_distance_m",
                    "best_candidate_source",
                    "best_candidate_source_distance_m",
                ]
            ]
            .to_dict(orient="records")
        ),
    }
    output_columns = [
        "tripId",
        "UnixTimeMillis",
        "epoch_index",
        "candidate_delta_m",
        "SelectedSource",
        "best_reference_source",
        "best_reference_source_distance_m",
        "best_candidate_source",
        "best_candidate_source_distance_m",
        "best_reference_source_latitude_degrees",
        "best_reference_source_longitude_degrees",
        "LatitudeDegrees_reference",
        "LongitudeDegrees_reference",
        "LatitudeDegrees_candidate",
        "LongitudeDegrees_candidate",
    ]
    output_columns.extend(distance_columns)
    output_columns.extend(candidate_distance_columns)
    return rows[output_columns], chunk_summary, summary


def reconstruct_candidate_submission(
    candidate_submission: Path,
    row_summary: pd.DataFrame,
    *,
    target_trip: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(candidate_submission)
    required = set(KEY_COLUMNS + ["LatitudeDegrees", "LongitudeDegrees"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{candidate_submission} is missing columns: {sorted(missing)}")
    patch_columns = {
        "UnixTimeMillis",
        "best_reference_source_latitude_degrees",
        "best_reference_source_longitude_degrees",
    }
    missing_patch = patch_columns.difference(row_summary.columns)
    if missing_patch:
        raise ValueError(f"row summary is missing columns: {sorted(missing_patch)}")

    mask = frame["tripId"] == target_trip
    if not mask.any():
        raise ValueError(f"{candidate_submission} contains no rows for {target_trip!r}")
    target = frame.loc[mask, ["UnixTimeMillis"]].copy()
    patch = row_summary[
        [
            "UnixTimeMillis",
            "best_reference_source_latitude_degrees",
            "best_reference_source_longitude_degrees",
        ]
    ].copy()
    if patch["UnixTimeMillis"].duplicated().any():
        raise ValueError("row summary contains duplicate UnixTimeMillis values")
    joined = target.merge(patch, on="UnixTimeMillis", how="left", validate="one_to_one")
    if joined[["best_reference_source_latitude_degrees", "best_reference_source_longitude_degrees"]].isna().any(axis=None):
        raise ValueError("row summary does not cover every target trip timestamp")

    out = frame.copy()
    out.loc[mask, "LatitudeDegrees"] = joined["best_reference_source_latitude_degrees"].to_numpy()
    out.loc[mask, "LongitudeDegrees"] = joined["best_reference_source_longitude_degrees"].to_numpy()
    return out, {
        "candidate_submission": str(candidate_submission),
        "target_trip": target_trip,
        "rows_replaced": int(mask.sum()),
        "source": "best_reference_source",
    }


def write_outputs(
    output_dir: Path,
    row_summary: pd.DataFrame,
    chunk_summary: pd.DataFrame,
    summary: dict[str, object],
    reconstructed_submission: pd.DataFrame | None = None,
    reconstructed_summary: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_summary.to_csv(output_dir / "target_trip_source_delta_rows.csv", index=False)
    chunk_summary.to_csv(output_dir / "target_trip_source_delta_chunks.csv", index=False)
    reconstructed_path = output_dir / "submission_with_target_trip_best_reference_source.csv"
    if reconstructed_submission is not None:
        reconstructed_submission.to_csv(reconstructed_path, index=False)
    payload = {
        **summary,
        "row_summary_csv": str(output_dir / "target_trip_source_delta_rows.csv"),
        "chunk_summary_csv": str(output_dir / "target_trip_source_delta_chunks.csv"),
    }
    if reconstructed_submission is not None:
        payload["reconstructed_submission_csv"] = str(reconstructed_path)
        payload["reconstructed_submission"] = reconstructed_summary or {}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-submission", type=Path, required=True)
    parser.add_argument("--candidate-submission", type=Path, required=True)
    parser.add_argument("--bridge-rows", type=Path, required=True)
    parser.add_argument("--target-trip", required=True)
    parser.add_argument("--chunk-epochs", type=int, default=200)
    parser.add_argument(
        "--write-reconstructed-submission",
        action="store_true",
        help="write a full candidate submission with the target trip replaced by best-reference-source bridge rows",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    row_summary, chunk_summary, summary = analyze_target_trip_source_delta(
        reference_submission=args.reference_submission,
        candidate_submission=args.candidate_submission,
        bridge_rows=args.bridge_rows,
        target_trip=args.target_trip,
        chunk_epochs=args.chunk_epochs,
    )
    reconstructed_submission = None
    reconstructed_summary = None
    if args.write_reconstructed_submission:
        reconstructed_submission, reconstructed_summary = reconstruct_candidate_submission(
            args.candidate_submission,
            row_summary,
            target_trip=args.target_trip,
        )
    write_outputs(
        args.output_dir,
        row_summary,
        chunk_summary,
        summary,
        reconstructed_submission=reconstructed_submission,
        reconstructed_summary=reconstructed_summary,
    )
    print(f"analyzed: {summary['rows']} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
