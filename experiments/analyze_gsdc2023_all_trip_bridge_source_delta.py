"""Scan all GSDC trips against per-trip bridge source columns."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_gsdc2023_target_trip_source_delta import source_coordinate_columns
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


REQUIRED_SUBMISSION_COLUMNS = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]


def _read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(REQUIRED_SUBMISSION_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)


def _bridge_path(bridge_root: Path, trip_id: str) -> Path:
    course, phone = trip_id.split("/", 1)
    return bridge_root / course / phone / "bridge_positions.csv"


def _score_record(values: np.ndarray, prefix: str = "") -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}mean_m": float("nan"),
            f"{prefix}p50_m": float("nan"),
            f"{prefix}p95_m": float("nan"),
            f"{prefix}max_m": float("nan"),
            f"{prefix}rows_gt_1m": 0,
            f"{prefix}rows_gt_5m": 0,
        }
    score = gsdc_score_m(values)
    return {
        f"{prefix}mean_m": score["mean_m"],
        f"{prefix}p50_m": score["p50_m"],
        f"{prefix}p95_m": score["p95_m"],
        f"{prefix}max_m": score["max_m"],
        f"{prefix}rows_gt_1m": int(np.count_nonzero(values > 1.0)),
        f"{prefix}rows_gt_5m": int(np.count_nonzero(values > 5.0)),
    }


def _analyze_trip(reference: pd.DataFrame, bridge_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    trip_id = str(reference["tripId"].iloc[0])
    phone = trip_id.split("/", 1)[1]
    if not bridge_path.exists():
        return pd.DataFrame(), {
            "tripId": trip_id,
            "phone": phone,
            "status": "missing_bridge",
            "rows": int(len(reference)),
            "matched_rows": 0,
            "missing_rows": int(len(reference)),
        }

    bridge = pd.read_csv(bridge_path)
    sources = source_coordinate_columns(bridge)
    if not sources:
        return pd.DataFrame(), {
            "tripId": trip_id,
            "phone": phone,
            "status": "missing_source_columns",
            "rows": int(len(reference)),
            "matched_rows": 0,
            "missing_rows": int(len(reference)),
        }

    rows = reference.reset_index(drop=True).copy()
    rows.insert(2, "epoch_index", np.arange(len(rows), dtype=np.int64))
    rows = rows.rename(columns={"LatitudeDegrees": "LatitudeDegrees_reference", "LongitudeDegrees": "LongitudeDegrees_reference"})
    merged = rows.merge(bridge, on="UnixTimeMillis", how="inner", validate="one_to_one", suffixes=("_reference", ""))
    if merged.empty:
        return pd.DataFrame(), {
            "tripId": trip_id,
            "phone": phone,
            "status": "no_matched_rows",
            "rows": int(len(reference)),
            "matched_rows": 0,
            "missing_rows": int(len(reference)),
            "bridge_path": str(bridge_path),
        }

    distance_columns: list[str] = []
    source_names: list[str] = []
    for source, (lat_column, lon_column) in sources.items():
        distance_column = f"distance_to_{source}_m"
        merged[distance_column] = haversine_m(
            merged["LatitudeDegrees_reference"].to_numpy(),
            merged["LongitudeDegrees_reference"].to_numpy(),
            merged[lat_column].to_numpy(),
            merged[lon_column].to_numpy(),
        )
        distance_columns.append(distance_column)
        source_names.append(source)

    distance_matrix = merged[distance_columns].to_numpy(dtype=np.float64)
    best_index = np.nanargmin(distance_matrix, axis=1)
    best_distance = distance_matrix[np.arange(len(merged)), best_index]
    merged["best_source"] = [source_names[index] for index in best_index]
    merged["best_source_distance_m"] = best_distance
    merged["best_source_latitude_degrees"] = np.nan
    merged["best_source_longitude_degrees"] = np.nan
    for index, source in enumerate(source_names):
        lat_column, lon_column = sources[source]
        mask = best_index == index
        merged.loc[mask, "best_source_latitude_degrees"] = merged.loc[mask, lat_column]
        merged.loc[mask, "best_source_longitude_degrees"] = merged.loc[mask, lon_column]
    counts = Counter(merged["best_source"].astype(str))
    status = "compared" if len(merged) == len(reference) else "partial_match"
    summary: dict[str, object] = {
        "tripId": trip_id,
        "phone": phone,
        "status": status,
        "rows": int(len(reference)),
        "matched_rows": int(len(merged)),
        "missing_rows": int(len(reference) - len(merged)),
        "bridge_path": str(bridge_path),
        **_score_record(best_distance, "best_source_"),
    }
    for source, count in sorted(counts.items()):
        summary[f"best_{source}_rows"] = int(count)

    output_columns = [
        "tripId",
        "UnixTimeMillis",
        "epoch_index",
        "LatitudeDegrees_reference",
        "LongitudeDegrees_reference",
        "best_source",
        "best_source_distance_m",
        "best_source_latitude_degrees",
        "best_source_longitude_degrees",
    ]
    output_columns.extend(distance_columns)
    return merged[output_columns], summary


def analyze_all_trip_bridge_source_delta(
    *,
    reference_submission: Path,
    bridge_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    reference = _read_submission(reference_submission)
    row_frames: list[pd.DataFrame] = []
    trip_records: list[dict[str, object]] = []
    for trip_id, trip_rows in reference.groupby("tripId", sort=True):
        rows, summary = _analyze_trip(trip_rows, _bridge_path(bridge_root, str(trip_id)))
        if not rows.empty:
            row_frames.append(rows)
        trip_records.append(summary)

    row_summary = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    trip_summary = pd.DataFrame(trip_records)
    compared = trip_summary[trip_summary["matched_rows"] > 0].copy()
    if row_summary.empty:
        overall = _score_record(np.array([], dtype=np.float64), "best_source_")
    else:
        overall = _score_record(row_summary["best_source_distance_m"].to_numpy(dtype=np.float64), "best_source_")
    phone_summary = (
        compared.groupby("phone", dropna=False)
        .agg(
            trips=("tripId", "count"),
            rows=("rows", "sum"),
            matched_rows=("matched_rows", "sum"),
            missing_rows=("missing_rows", "sum"),
            rows_gt_1m=("best_source_rows_gt_1m", "sum"),
            rows_gt_5m=("best_source_rows_gt_5m", "sum"),
            max_trip_p95_m=("best_source_p95_m", "max"),
            max_row_m=("best_source_max_m", "max"),
        )
        .reset_index()
        if not compared.empty
        else pd.DataFrame()
    )
    payload = {
        "reference_submission": str(reference_submission),
        "bridge_root": str(bridge_root),
        "rows": int(len(reference)),
        "matched_rows": int(trip_summary["matched_rows"].sum()) if "matched_rows" in trip_summary else 0,
        "missing_rows": int(trip_summary["missing_rows"].sum()) if "missing_rows" in trip_summary else 0,
        "trip_count": int(len(trip_summary)),
        "status_counts": trip_summary["status"].value_counts().sort_index().to_dict() if "status" in trip_summary else {},
        "overall": overall,
        "top_trips_by_best_source_p95": (
            compared.sort_values(["best_source_p95_m", "best_source_max_m"], ascending=[False, False])
            .head(20)
            .to_dict(orient="records")
            if not compared.empty
            else []
        ),
        "top_phones_by_best_source_p95": (
            phone_summary.sort_values(["max_trip_p95_m", "max_row_m"], ascending=[False, False])
            .head(20)
            .to_dict(orient="records")
            if not phone_summary.empty
            else []
        ),
    }
    return row_summary, trip_summary, payload


def reconstruct_candidate_submission(
    candidate_submission: Path,
    row_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = _read_submission(candidate_submission)
    patch_columns = {
        "tripId",
        "UnixTimeMillis",
        "best_source_latitude_degrees",
        "best_source_longitude_degrees",
    }
    missing_patch = patch_columns.difference(row_summary.columns)
    if missing_patch:
        raise ValueError(f"row summary is missing columns: {sorted(missing_patch)}")

    patch = row_summary[
        [
            "tripId",
            "UnixTimeMillis",
            "best_source_latitude_degrees",
            "best_source_longitude_degrees",
        ]
    ].copy()
    if patch[["tripId", "UnixTimeMillis"]].duplicated().any():
        raise ValueError("row summary contains duplicate tripId/UnixTimeMillis values")

    joined = frame[["tripId", "UnixTimeMillis"]].merge(
        patch,
        on=["tripId", "UnixTimeMillis"],
        how="left",
        validate="one_to_one",
    )
    replace_mask = joined[["best_source_latitude_degrees", "best_source_longitude_degrees"]].notna().all(axis=1)
    out = frame.copy()
    out.loc[replace_mask, "LatitudeDegrees"] = joined.loc[replace_mask, "best_source_latitude_degrees"].to_numpy()
    out.loc[replace_mask, "LongitudeDegrees"] = joined.loc[replace_mask, "best_source_longitude_degrees"].to_numpy()
    return out, {
        "candidate_submission": str(candidate_submission),
        "rows": int(len(frame)),
        "rows_replaced": int(replace_mask.sum()),
        "rows_unmatched": int(len(frame) - replace_mask.sum()),
        "source": "all_trip_best_reference_bridge_source",
    }


def write_outputs(
    output_dir: Path,
    rows: pd.DataFrame,
    trips: pd.DataFrame,
    summary: dict[str, object],
    reconstructed_submission: pd.DataFrame | None = None,
    reconstructed_summary: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "all_trip_bridge_source_delta_rows.csv"
    trips_path = output_dir / "all_trip_bridge_source_delta_trips.csv"
    rows.to_csv(rows_path, index=False)
    trips.to_csv(trips_path, index=False)
    reconstructed_path = output_dir / "submission_with_all_trip_best_reference_bridge_source.csv"
    if reconstructed_submission is not None:
        reconstructed_submission.to_csv(reconstructed_path, index=False)
        summary = {
            **summary,
            "reconstructed_submission_csv": str(reconstructed_path),
            "reconstructed_submission": reconstructed_summary or {},
        }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({**summary, "rows_csv": str(rows_path), "trip_summary_csv": str(trips_path)}, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-submission", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-submission", type=Path)
    parser.add_argument("--write-reconstructed-submission", action="store_true")
    args = parser.parse_args(argv)

    rows, trips, summary = analyze_all_trip_bridge_source_delta(
        reference_submission=args.reference_submission,
        bridge_root=args.bridge_root,
    )
    reconstructed_submission = None
    reconstructed_summary = None
    if args.write_reconstructed_submission:
        if args.candidate_submission is None:
            raise ValueError("--candidate-submission is required with --write-reconstructed-submission")
        reconstructed_submission, reconstructed_summary = reconstruct_candidate_submission(
            args.candidate_submission,
            rows,
        )
    write_outputs(
        args.output_dir,
        rows,
        trips,
        summary,
        reconstructed_submission=reconstructed_submission,
        reconstructed_summary=reconstructed_summary,
    )
    print(f"analyzed: {summary['trip_count']} trip(s), {summary['matched_rows']} matched row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
