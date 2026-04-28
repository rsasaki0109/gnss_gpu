"""Analyze GSDC2023 source-selection A/B submission and chunk metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.gsdc2023_chunk_selection import (
    GATED_RAW_WLS_RESCUE_BASELINE_GAP_MAX_M,
    GATED_RAW_WLS_RESCUE_BASELINE_MSE_PR_MIN,
    GATED_RAW_WLS_RESCUE_MSE_PR_MAX,
    GATED_RAW_WLS_RESCUE_MSE_PR_RATIO_MAX,
)
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


KEY_COLUMNS = ["tripId", "UnixTimeMillis"]


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"empty comparison name in {value!r}")
    return name, Path(raw_path)


def read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS + ["LatitudeDegrees", "LongitudeDegrees"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def compare_submissions(base: pd.DataFrame, candidate: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(base) != len(candidate):
        raise ValueError(f"{name}: row count mismatch {len(base)} != {len(candidate)}")
    for column in KEY_COLUMNS:
        if not base[column].equals(candidate[column]):
            raise ValueError(f"{name}: key column mismatch: {column}")

    dist_m = haversine_m(
        base["LatitudeDegrees"].to_numpy(),
        base["LongitudeDegrees"].to_numpy(),
        candidate["LatitudeDegrees"].to_numpy(),
        candidate["LongitudeDegrees"].to_numpy(),
    )
    row_delta = base[KEY_COLUMNS].copy()
    row_delta["comparison"] = name
    row_delta["delta_m"] = dist_m
    row_delta["base_lat"] = base["LatitudeDegrees"].to_numpy()
    row_delta["base_lon"] = base["LongitudeDegrees"].to_numpy()
    row_delta["candidate_lat"] = candidate["LatitudeDegrees"].to_numpy()
    row_delta["candidate_lon"] = candidate["LongitudeDegrees"].to_numpy()

    trip_rows: list[dict[str, object]] = []
    grouped = row_delta.groupby("tripId", sort=False)
    for trip_id, group in grouped:
        score = gsdc_score_m(group["delta_m"].to_numpy())
        changed = group["delta_m"].to_numpy() > 0.01
        trip_rows.append(
            {
                "comparison": name,
                "tripId": trip_id,
                "rows": int(len(group)),
                "changed_rows_gt_0p01m": int(np.count_nonzero(changed)),
                "rows_gt_1m": int(np.count_nonzero(group["delta_m"].to_numpy() > 1.0)),
                "rows_gt_5m": int(np.count_nonzero(group["delta_m"].to_numpy() > 5.0)),
                "mean_delta_m": score["mean_m"],
                "p50_delta_m": score["p50_m"],
                "p95_delta_m": score["p95_m"],
                "max_delta_m": score["max_m"],
                "first_time_ms": int(group["UnixTimeMillis"].iloc[0]),
                "last_time_ms": int(group["UnixTimeMillis"].iloc[-1]),
            }
        )
    return row_delta, pd.DataFrame(trip_rows)


def chunk_delta_summary(row_delta: pd.DataFrame, target_trips: set[str], chunk_epochs: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (comparison, trip_id), group in row_delta.groupby(["comparison", "tripId"], sort=False):
        if target_trips and trip_id not in target_trips:
            continue
        group = group.reset_index(drop=True)
        for start in range(0, len(group), chunk_epochs):
            chunk = group.iloc[start : start + chunk_epochs]
            score = gsdc_score_m(chunk["delta_m"].to_numpy())
            rows.append(
                {
                    "comparison": comparison,
                    "tripId": trip_id,
                    "start_epoch": int(start),
                    "end_epoch": int(start + len(chunk)),
                    "rows": int(len(chunk)),
                    "changed_rows_gt_0p01m": int(np.count_nonzero(chunk["delta_m"].to_numpy() > 0.01)),
                    "rows_gt_1m": int(np.count_nonzero(chunk["delta_m"].to_numpy() > 1.0)),
                    "rows_gt_5m": int(np.count_nonzero(chunk["delta_m"].to_numpy() > 5.0)),
                    "mean_delta_m": score["mean_m"],
                    "p50_delta_m": score["p50_m"],
                    "p95_delta_m": score["p95_m"],
                    "max_delta_m": score["max_m"],
                    "first_time_ms": int(chunk["UnixTimeMillis"].iloc[0]),
                    "last_time_ms": int(chunk["UnixTimeMillis"].iloc[-1]),
                }
            )
    return pd.DataFrame(rows)


def _candidate_value(candidates: dict[str, object], source: str, field: str) -> float:
    value = candidates.get(source, {})
    if not isinstance(value, dict):
        return float("nan")
    raw = value.get(field, float("nan"))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def _raw_rescue_passes(baseline_mse: float, raw_mse: float, raw_gap_max_m: float) -> bool:
    return (
        math.isfinite(baseline_mse)
        and math.isfinite(raw_mse)
        and math.isfinite(raw_gap_max_m)
        and baseline_mse >= GATED_RAW_WLS_RESCUE_BASELINE_MSE_PR_MIN
        and raw_mse <= GATED_RAW_WLS_RESCUE_MSE_PR_MAX
        and raw_mse <= baseline_mse * GATED_RAW_WLS_RESCUE_MSE_PR_RATIO_MAX
        and raw_gap_max_m <= GATED_RAW_WLS_RESCUE_BASELINE_GAP_MAX_M
    )


def metrics_chunk_summary(named_metrics: list[tuple[str, Path]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, path in named_metrics:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        trip = str(payload.get("trip", name))
        records = payload.get("chunk_selection_records", [])
        if not isinstance(records, list):
            continue
        for record in records:
            candidates = record.get("candidates", {})
            if not isinstance(candidates, dict):
                candidates = {}
            baseline_mse = _candidate_value(candidates, "baseline", "mse_pr")
            raw_mse = _candidate_value(candidates, "raw_wls", "mse_pr")
            raw_gap_max_m = _candidate_value(candidates, "raw_wls", "baseline_gap_max_m")
            rows.append(
                {
                    "metrics_name": name,
                    "trip": trip,
                    "start_epoch": int(record.get("start_epoch", -1)),
                    "end_epoch": int(record.get("end_epoch", -1)),
                    "auto_source": str(record.get("auto_source", "")),
                    "gated_source": str(record.get("gated_source", "")),
                    "baseline_mse_pr": baseline_mse,
                    "raw_wls_mse_pr": raw_mse,
                    "raw_wls_mse_ratio": raw_mse / baseline_mse if math.isfinite(baseline_mse) and baseline_mse else float("nan"),
                    "fgo_mse_pr": _candidate_value(candidates, "fgo", "mse_pr"),
                    "baseline_step_p95_m": _candidate_value(candidates, "baseline", "step_p95_m"),
                    "raw_wls_step_p95_m": _candidate_value(candidates, "raw_wls", "step_p95_m"),
                    "raw_wls_baseline_gap_p95_m": _candidate_value(candidates, "raw_wls", "baseline_gap_p95_m"),
                    "raw_wls_baseline_gap_max_m": raw_gap_max_m,
                    "raw_wls_quality_score": _candidate_value(candidates, "raw_wls", "quality_score"),
                    "fgo_baseline_gap_p95_m": _candidate_value(candidates, "fgo", "baseline_gap_p95_m"),
                    "fgo_baseline_gap_max_m": _candidate_value(candidates, "fgo", "baseline_gap_max_m"),
                    "fgo_quality_score": _candidate_value(candidates, "fgo", "quality_score"),
                    "raw_wls_high_pr_rescue_pass": _raw_rescue_passes(baseline_mse, raw_mse, raw_gap_max_m),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: Path,
    row_delta: pd.DataFrame,
    trip_summary: pd.DataFrame,
    chunk_summary: pd.DataFrame,
    metrics_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_delta.to_csv(output_dir / "row_deltas.csv", index=False)
    trip_summary.to_csv(output_dir / "trip_delta_summary.csv", index=False)
    chunk_summary.to_csv(output_dir / "target_chunk_delta_summary.csv", index=False)
    metrics_summary.to_csv(output_dir / "chunk_metrics_summary.csv", index=False)

    top_trips = (
        trip_summary.sort_values(["comparison", "p95_delta_m", "max_delta_m"], ascending=[True, False, False])
        .groupby("comparison", sort=False)
        .head(10)
    )
    summary = {
        "comparisons": sorted(trip_summary["comparison"].unique().tolist()) if not trip_summary.empty else [],
        "row_deltas_csv": str(output_dir / "row_deltas.csv"),
        "trip_delta_summary_csv": str(output_dir / "trip_delta_summary.csv"),
        "target_chunk_delta_summary_csv": str(output_dir / "target_chunk_delta_summary.csv"),
        "chunk_metrics_summary_csv": str(output_dir / "chunk_metrics_summary.csv"),
        "top_trips_by_p95_delta": top_trips.to_dict(orient="records"),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--comparison", action="append", default=[], help="NAME=path/to/submission.csv")
    parser.add_argument("--metrics", action="append", default=[], help="NAME=path/to/bridge_metrics.json")
    parser.add_argument("--target-trip", action="append", default=[])
    parser.add_argument("--chunk-epochs", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.chunk_epochs <= 0:
        raise ValueError("--chunk-epochs must be positive")
    comparisons = [parse_named_path(value) for value in args.comparison]
    if not comparisons:
        raise ValueError("at least one --comparison is required")
    metrics = [parse_named_path(value) for value in args.metrics]

    base = read_submission(args.base_submission)
    row_frames: list[pd.DataFrame] = []
    trip_frames: list[pd.DataFrame] = []
    for name, path in comparisons:
        row_delta, trip_summary = compare_submissions(base, read_submission(path), name)
        row_frames.append(row_delta)
        trip_frames.append(trip_summary)

    all_rows = pd.concat(row_frames, ignore_index=True)
    all_trips = pd.concat(trip_frames, ignore_index=True)
    target_trips = set(args.target_trip)
    chunks = chunk_delta_summary(all_rows, target_trips, args.chunk_epochs)
    metrics_summary = metrics_chunk_summary(metrics)
    write_outputs(args.output_dir, all_rows, all_trips, chunks, metrics_summary)


if __name__ == "__main__":
    main()
