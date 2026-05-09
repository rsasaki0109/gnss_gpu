"""Compare a MATLAB/reference trip trajectory against bridge source paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_gsdc2023_target_trip_source_delta import source_coordinate_columns
from experiments.smooth_gsdc2023_submission import gsdc_score_m, latlon_to_local_m


def _score_record(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean_m": float("nan"),
            f"{prefix}_p50_m": float("nan"),
            f"{prefix}_p95_m": float("nan"),
            f"{prefix}_max_m": float("nan"),
            f"{prefix}_rows_gt_1m": 0,
            f"{prefix}_rows_gt_5m": 0,
        }
    score = gsdc_score_m(values)
    return {
        f"{prefix}_mean_m": score["mean_m"],
        f"{prefix}_p50_m": score["p50_m"],
        f"{prefix}_p95_m": score["p95_m"],
        f"{prefix}_max_m": score["max_m"],
        f"{prefix}_rows_gt_1m": int(np.count_nonzero(values > 1.0)),
        f"{prefix}_rows_gt_5m": int(np.count_nonzero(values > 5.0)),
    }


def _read_rows(row_summary: Path, bridge_rows: Path) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    rows = pd.read_csv(row_summary)
    required = {
        "UnixTimeMillis",
        "epoch_index",
        "LatitudeDegrees_reference",
        "LongitudeDegrees_reference",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"{row_summary} is missing columns: {sorted(missing)}")
    bridge = pd.read_csv(bridge_rows)
    if "UnixTimeMillis" not in bridge.columns:
        raise ValueError(f"{bridge_rows} is missing UnixTimeMillis")
    sources = source_coordinate_columns(bridge)
    if not sources:
        raise ValueError(f"{bridge_rows} contains no source coordinate columns")
    merged = rows.merge(bridge, on="UnixTimeMillis", how="inner", validate="one_to_one")
    if len(merged) != len(rows):
        raise ValueError(f"matched row count mismatch: rows={len(rows)} matched={len(merged)}")
    return merged.sort_values("epoch_index").reset_index(drop=True), sources


def _local_tracks(rows: pd.DataFrame, sources: dict[str, tuple[str, str]]) -> dict[str, np.ndarray]:
    ref_east, ref_north, origin_lat, origin_lon = latlon_to_local_m(
        rows["LatitudeDegrees_reference"].to_numpy(),
        rows["LongitudeDegrees_reference"].to_numpy(),
    )
    tracks = {"reference": np.stack([ref_east, ref_north], axis=1)}
    for source, (lat_column, lon_column) in sources.items():
        east, north, _, _ = latlon_to_local_m(
            rows[lat_column].to_numpy(),
            rows[lon_column].to_numpy(),
            origin_lat_deg=origin_lat,
            origin_lon_deg=origin_lon,
        )
        tracks[source] = np.stack([east, north], axis=1)
    return tracks


def _step_m(track: np.ndarray) -> np.ndarray:
    out = np.full(len(track), np.nan, dtype=np.float64)
    if len(track) > 1:
        out[1:] = np.linalg.norm(np.diff(track, axis=0), axis=1)
    return out


def _curvature_m(track: np.ndarray) -> np.ndarray:
    out = np.full(len(track), np.nan, dtype=np.float64)
    if len(track) > 2:
        out[1:-1] = np.linalg.norm(track[2:] - 2.0 * track[1:-1] + track[:-2], axis=1)
    return out


def _lag_slices(row_count: int, lag_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    if abs(lag_epochs) >= row_count:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if lag_epochs >= 0:
        reference_index = np.arange(0, row_count - lag_epochs, dtype=np.int64)
        source_index = reference_index + lag_epochs
    else:
        source_index = np.arange(0, row_count + lag_epochs, dtype=np.int64)
        reference_index = source_index - lag_epochs
    return reference_index, source_index


def _lag_distance(reference: np.ndarray, source: np.ndarray, lag_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    reference_index, source_index = _lag_slices(len(reference), lag_epochs)
    if len(reference_index) == 0:
        return reference_index, np.array([], dtype=np.float64)
    return reference_index, np.linalg.norm(reference[reference_index] - source[source_index], axis=1)


def _best_lag_record(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    ordered = frame.sort_values(group_columns + ["distance_p95_m", "distance_max_m", "lag_abs_epochs"])
    return ordered.groupby(group_columns, as_index=False).head(1).reset_index(drop=True)


def analyze_trajectory_shape(
    *,
    row_summary: Path,
    bridge_rows: Path,
    chunk_epochs: int,
    max_lag_epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if chunk_epochs <= 0:
        raise ValueError("chunk_epochs must be positive")
    if max_lag_epochs < 0:
        raise ValueError("max_lag_epochs must be non-negative")

    rows, sources = _read_rows(row_summary, bridge_rows)
    tracks = _local_tracks(rows, sources)
    reference = tracks["reference"]
    reference_step = _step_m(reference)
    reference_curvature = _curvature_m(reference)
    out = rows[["tripId", "UnixTimeMillis", "epoch_index"]].copy() if "tripId" in rows.columns else rows[["UnixTimeMillis", "epoch_index"]].copy()
    out["reference_step_m"] = reference_step
    out["reference_curvature_m"] = reference_curvature

    lag_rows: list[dict[str, object]] = []
    chunk_lag_rows: list[dict[str, object]] = []
    for source_name, source_track in tracks.items():
        if source_name == "reference":
            continue
        distance = np.linalg.norm(source_track - reference, axis=1)
        source_step = _step_m(source_track)
        source_curvature = _curvature_m(source_track)
        out[f"{source_name}_distance_m"] = distance
        out[f"{source_name}_step_m"] = source_step
        out[f"{source_name}_step_delta_m"] = source_step - reference_step
        out[f"{source_name}_curvature_m"] = source_curvature
        out[f"{source_name}_curvature_delta_m"] = source_curvature - reference_curvature

        for lag in range(-max_lag_epochs, max_lag_epochs + 1):
            reference_index, lag_distance = _lag_distance(reference, source_track, lag)
            record: dict[str, object] = {
                "source": source_name,
                "lag_epochs": int(lag),
                "lag_abs_epochs": int(abs(lag)),
                "rows": int(len(lag_distance)),
                **_score_record(lag_distance, "distance"),
            }
            lag_rows.append(record)
            chunk_start = (rows.loc[reference_index, "epoch_index"].to_numpy(dtype=np.int64) // chunk_epochs) * chunk_epochs
            for start in sorted(np.unique(chunk_start)):
                mask = chunk_start == start
                chunk_values = lag_distance[mask]
                chunk_lag_rows.append(
                    {
                        "source": source_name,
                        "chunk_start_epoch": int(start),
                        "chunk_end_epoch": int(min(start + chunk_epochs, int(rows["epoch_index"].max()) + 1)),
                        "lag_epochs": int(lag),
                        "lag_abs_epochs": int(abs(lag)),
                        "rows": int(len(chunk_values)),
                        **_score_record(chunk_values, "distance"),
                    },
                )

    lag_summary = pd.DataFrame(lag_rows)
    chunk_lag_summary = pd.DataFrame(chunk_lag_rows)
    best_lag = _best_lag_record(lag_summary, ["source"])
    best_chunk_lag = _best_lag_record(chunk_lag_summary, ["source", "chunk_start_epoch"])
    zero_chunk_lag = chunk_lag_summary[chunk_lag_summary["lag_epochs"] == 0].copy()
    summary = {
        "row_summary": str(row_summary),
        "bridge_rows": str(bridge_rows),
        "rows": int(len(out)),
        "source_names": list(sources),
        "max_lag_epochs": int(max_lag_epochs),
        "best_lag_by_source": best_lag.to_dict(orient="records"),
        "top_zero_lag_chunks_by_p95": (
            zero_chunk_lag.sort_values(["distance_p95_m", "distance_max_m"], ascending=[False, False])
            .head(20)
            .to_dict(orient="records")
        ),
        "top_best_lag_chunks_by_p95": (
            best_chunk_lag.sort_values(["distance_p95_m", "distance_max_m"], ascending=[False, False])
            .head(20)
            .to_dict(orient="records")
        ),
    }
    return out, lag_summary, chunk_lag_summary, summary


def write_outputs(
    output_dir: Path,
    rows: pd.DataFrame,
    lag_summary: pd.DataFrame,
    chunk_lag_summary: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "trajectory_shape_rows.csv"
    lag_path = output_dir / "trajectory_lag_summary.csv"
    chunk_lag_path = output_dir / "trajectory_lag_chunks.csv"
    rows.to_csv(rows_path, index=False)
    lag_summary.to_csv(lag_path, index=False)
    chunk_lag_summary.to_csv(chunk_lag_path, index=False)
    payload = {
        **summary,
        "rows_csv": str(rows_path),
        "lag_summary_csv": str(lag_path),
        "chunk_lag_summary_csv": str(chunk_lag_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-summary", type=Path, required=True)
    parser.add_argument("--bridge-rows", type=Path, required=True)
    parser.add_argument("--chunk-epochs", type=int, default=200)
    parser.add_argument("--max-lag-epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows, lag_summary, chunk_lag_summary, summary = analyze_trajectory_shape(
        row_summary=args.row_summary,
        bridge_rows=args.bridge_rows,
        chunk_epochs=args.chunk_epochs,
        max_lag_epochs=args.max_lag_epochs,
    )
    write_outputs(args.output_dir, rows, lag_summary, chunk_lag_summary, summary)
    print(f"analyzed: {summary['rows']} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
