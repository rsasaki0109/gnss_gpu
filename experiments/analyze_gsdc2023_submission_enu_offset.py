"""Analyze ENU offset structure between two GSDC2023 submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_gsdc2023_source_ab import KEY_COLUMNS, phone_from_trip_id
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m, latlon_to_local_m


COORD_COLUMNS = ["LatitudeDegrees", "LongitudeDegrees"]


def read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS + COORD_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def _score_columns(values: np.ndarray, prefix: str) -> dict[str, float]:
    score = gsdc_score_m(values)
    return {
        f"{prefix}_mean_m": score["mean_m"],
        f"{prefix}_p50_m": score["p50_m"],
        f"{prefix}_p95_m": score["p95_m"],
        f"{prefix}_max_m": score["max_m"],
    }


def _selected_rows(frame: pd.DataFrame, target_trips: set[str], target_phones: set[str]) -> pd.Series:
    selected = pd.Series(True, index=frame.index)
    if target_trips:
        selected &= frame["tripId"].isin(target_trips)
    if target_phones:
        selected &= frame["tripId"].map(phone_from_trip_id).isin(target_phones)
    return selected


def _compute_offsets(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    if len(reference) != len(candidate):
        raise ValueError(f"row count mismatch {len(reference)} != {len(candidate)}")
    for column in KEY_COLUMNS:
        if not reference[column].equals(candidate[column]):
            raise ValueError(f"key column mismatch: {column}")

    rows = reference[KEY_COLUMNS].copy()
    rows["phone"] = rows["tripId"].map(phone_from_trip_id)
    rows["reference_latitude_degrees"] = reference["LatitudeDegrees"].to_numpy()
    rows["reference_longitude_degrees"] = reference["LongitudeDegrees"].to_numpy()
    rows["candidate_latitude_degrees"] = candidate["LatitudeDegrees"].to_numpy()
    rows["candidate_longitude_degrees"] = candidate["LongitudeDegrees"].to_numpy()
    rows["original_delta_m"] = haversine_m(
        reference["LatitudeDegrees"].to_numpy(),
        reference["LongitudeDegrees"].to_numpy(),
        candidate["LatitudeDegrees"].to_numpy(),
        candidate["LongitudeDegrees"].to_numpy(),
    )

    east_m = np.empty(len(rows), dtype=np.float64)
    north_m = np.empty(len(rows), dtype=np.float64)
    for _, index in rows.groupby("tripId", sort=False).groups.items():
        ref_east, ref_north, origin_lat, origin_lon = latlon_to_local_m(
            rows.loc[index, "reference_latitude_degrees"].to_numpy(),
            rows.loc[index, "reference_longitude_degrees"].to_numpy(),
        )
        cand_east, cand_north, _, _ = latlon_to_local_m(
            rows.loc[index, "candidate_latitude_degrees"].to_numpy(),
            rows.loc[index, "candidate_longitude_degrees"].to_numpy(),
            origin_lat_deg=origin_lat,
            origin_lon_deg=origin_lon,
        )
        east_m[np.asarray(index)] = cand_east - ref_east
        north_m[np.asarray(index)] = cand_north - ref_north

    rows["candidate_minus_reference_east_m"] = east_m
    rows["candidate_minus_reference_north_m"] = north_m
    return rows


def _summarize_group(group_type: str, group_name: str, group: pd.DataFrame) -> dict[str, object]:
    east = group["candidate_minus_reference_east_m"].to_numpy(dtype=np.float64)
    north = group["candidate_minus_reference_north_m"].to_numpy(dtype=np.float64)
    original_delta = group["original_delta_m"].to_numpy(dtype=np.float64)
    median_east = float(np.nanmedian(east))
    median_north = float(np.nanmedian(north))
    residual_delta = np.hypot(east - median_east, north - median_north)
    original_p95 = float(gsdc_score_m(original_delta)["p95_m"])
    residual_p95 = float(gsdc_score_m(residual_delta)["p95_m"])
    return {
        "group_type": group_type,
        "group": group_name,
        "phone": phone_from_trip_id(group_name) if group_type == "trip" else group_name,
        "rows": int(len(group)),
        "original_rows_gt_1m": int(np.count_nonzero(original_delta > 1.0)),
        "original_rows_gt_5m": int(np.count_nonzero(original_delta > 5.0)),
        **_score_columns(original_delta, "original_delta"),
        "median_candidate_minus_reference_east_m": median_east,
        "median_candidate_minus_reference_north_m": median_north,
        "median_offset_norm_m": float(np.hypot(median_east, median_north)),
        "residual_after_median_rows_gt_1m": int(np.count_nonzero(residual_delta > 1.0)),
        "residual_after_median_rows_gt_5m": int(np.count_nonzero(residual_delta > 5.0)),
        **_score_columns(residual_delta, "residual_after_median"),
        "p95_reduction_m": original_p95 - residual_p95,
        "p95_residual_ratio": residual_p95 / original_p95 if original_p95 > 0.0 else float("nan"),
    }


def summarize_offsets(row_offsets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trip_id, group in row_offsets.groupby("tripId", sort=False):
        rows.append(_summarize_group("trip", str(trip_id), group))
    for phone, group in row_offsets.groupby("phone", sort=False):
        rows.append(_summarize_group("phone", str(phone), group))
    return pd.DataFrame(rows)


def add_trip_median_residuals(row_offsets: pd.DataFrame) -> pd.DataFrame:
    rows = row_offsets.copy()
    rows["trip_median_candidate_minus_reference_east_m"] = np.nan
    rows["trip_median_candidate_minus_reference_north_m"] = np.nan
    rows["residual_after_trip_median_east_m"] = np.nan
    rows["residual_after_trip_median_north_m"] = np.nan
    rows["residual_after_trip_median_m"] = np.nan
    for _, index in rows.groupby("tripId", sort=False).groups.items():
        east = rows.loc[index, "candidate_minus_reference_east_m"].to_numpy(dtype=np.float64)
        north = rows.loc[index, "candidate_minus_reference_north_m"].to_numpy(dtype=np.float64)
        median_east = float(np.nanmedian(east))
        median_north = float(np.nanmedian(north))
        residual_east = east - median_east
        residual_north = north - median_north
        rows.loc[index, "trip_median_candidate_minus_reference_east_m"] = median_east
        rows.loc[index, "trip_median_candidate_minus_reference_north_m"] = median_north
        rows.loc[index, "residual_after_trip_median_east_m"] = residual_east
        rows.loc[index, "residual_after_trip_median_north_m"] = residual_north
        rows.loc[index, "residual_after_trip_median_m"] = np.hypot(residual_east, residual_north)
    return rows


def analyze_submission_enu_offset(
    *,
    reference_submission: Path,
    candidate_submission: Path,
    target_trips: set[str] | None = None,
    target_phones: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    target_trips = target_trips or set()
    target_phones = target_phones or set()
    reference = read_submission(reference_submission)
    candidate = read_submission(candidate_submission)
    selected = _selected_rows(reference, target_trips, target_phones)
    if not selected.any():
        raise ValueError("selection matched no rows")

    reference = reference.loc[selected].reset_index(drop=True)
    candidate = candidate.loc[selected].reset_index(drop=True)
    row_offsets = add_trip_median_residuals(_compute_offsets(reference, candidate))
    summary = summarize_offsets(row_offsets)
    top_groups = summary.sort_values(
        ["original_delta_p95_m", "original_delta_max_m"],
        ascending=[False, False],
    ).head(20)
    top_residual_groups = summary.sort_values(
        ["residual_after_median_p95_m", "residual_after_median_max_m"],
        ascending=[False, False],
    ).head(20)
    worst_rows = row_offsets.sort_values("original_delta_m", ascending=False).head(20)
    payload = {
        "reference_submission": str(reference_submission),
        "candidate_submission": str(candidate_submission),
        "target_trips": sorted(target_trips),
        "target_phones": sorted(target_phones),
        "rows": int(len(row_offsets)),
        "summary_csv": "enu_offset_summary.csv",
        "row_offsets_csv": "enu_row_offsets.csv",
        "top_groups_by_original_p95": top_groups.to_dict(orient="records"),
        "top_groups_by_residual_p95": top_residual_groups.to_dict(orient="records"),
        "worst_rows_by_original_delta": worst_rows[
            [
                "tripId",
                "UnixTimeMillis",
                "phone",
                "original_delta_m",
                "candidate_minus_reference_east_m",
                "candidate_minus_reference_north_m",
                "residual_after_trip_median_m",
            ]
        ].to_dict(orient="records"),
    }
    return row_offsets, summary, payload


def write_outputs(output_dir: Path, row_offsets: pd.DataFrame, summary: pd.DataFrame, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "enu_row_offsets.csv"
    summary_path = output_dir / "enu_offset_summary.csv"
    row_offsets.to_csv(row_path, index=False)
    summary.to_csv(summary_path, index=False)
    payload = {
        **payload,
        "row_offsets_csv": str(row_path),
        "summary_csv": str(summary_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-submission", type=Path, required=True)
    parser.add_argument("--candidate-submission", type=Path, required=True)
    parser.add_argument("--target-trip", action="append", default=[])
    parser.add_argument("--target-phone", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    row_offsets, summary, payload = analyze_submission_enu_offset(
        reference_submission=args.reference_submission,
        candidate_submission=args.candidate_submission,
        target_trips=set(args.target_trip),
        target_phones=set(args.target_phone),
    )
    write_outputs(args.output_dir, row_offsets, summary, payload)
    print(f"analyzed: {len(row_offsets)} row(s), {len(summary)} group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
