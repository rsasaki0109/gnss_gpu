"""Analyze whether a reference trip lies on bridge source blends."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_gsdc2023_target_trip_source_delta import source_coordinate_columns
from experiments.smooth_gsdc2023_submission import gsdc_score_m, latlon_to_local_m


def _score_record(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    score = gsdc_score_m(values)
    return {
        f"{prefix}_mean_m": score["mean_m"],
        f"{prefix}_p50_m": score["p50_m"],
        f"{prefix}_p95_m": score["p95_m"],
        f"{prefix}_max_m": score["max_m"],
        f"{prefix}_rows_gt_1m": int(np.count_nonzero(np.asarray(values) > 1.0)),
        f"{prefix}_rows_gt_5m": int(np.count_nonzero(np.asarray(values) > 5.0)),
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
    return merged, sources


def _source_points_m(rows: pd.DataFrame, sources: dict[str, tuple[str, str]]) -> tuple[np.ndarray, list[str]]:
    ref_east, ref_north, origin_lat, origin_lon = latlon_to_local_m(
        rows["LatitudeDegrees_reference"].to_numpy(),
        rows["LongitudeDegrees_reference"].to_numpy(),
    )
    names = list(sources)
    points: list[np.ndarray] = []
    for lat_column, lon_column in sources.values():
        east, north, _, _ = latlon_to_local_m(
            rows[lat_column].to_numpy(),
            rows[lon_column].to_numpy(),
            origin_lat_deg=origin_lat,
            origin_lon_deg=origin_lon,
        )
        points.append(np.stack([east, north], axis=1))
    ref = np.stack([ref_east, ref_north], axis=1)
    return np.stack(points, axis=1) - ref[:, None, :], names


def _closest_points(source_vectors: np.ndarray, source_names: list[str]) -> pd.DataFrame:
    distances = np.linalg.norm(source_vectors, axis=2)
    best = np.nanargmin(distances, axis=1)
    return pd.DataFrame(
        {
            "best_point_source": [source_names[index] for index in best],
            "best_point_distance_m": distances[np.arange(len(best)), best],
        },
    )


def _closest_segments(source_vectors: np.ndarray, source_names: list[str]) -> pd.DataFrame:
    row_count = source_vectors.shape[0]
    best_distance = np.full(row_count, np.inf, dtype=np.float64)
    best_pair = np.full(row_count, "", dtype=object)
    best_t = np.zeros(row_count, dtype=np.float64)
    for left, right in combinations(range(len(source_names)), 2):
        a = source_vectors[:, left, :]
        b = source_vectors[:, right, :]
        direction = b - a
        denom = np.einsum("ij,ij->i", direction, direction)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -np.einsum("ij,ij->i", a, direction) / denom
        t = np.where(np.isfinite(t), np.clip(t, 0.0, 1.0), 0.0)
        projection = a + direction * t[:, None]
        distance = np.linalg.norm(projection, axis=1)
        mask = distance < best_distance
        best_distance[mask] = distance[mask]
        best_t[mask] = t[mask]
        best_pair[mask] = f"{source_names[left]}-{source_names[right]}"
    return pd.DataFrame(
        {
            "best_segment_pair": best_pair,
            "best_segment_t": best_t,
            "best_segment_distance_m": best_distance,
        },
    )


def _closest_triangle_interiors(source_vectors: np.ndarray, source_names: list[str]) -> pd.DataFrame:
    row_count = source_vectors.shape[0]
    best_distance = np.full(row_count, np.inf, dtype=np.float64)
    best_triangle = np.full(row_count, "", dtype=object)
    best_weights = np.full((row_count, 3), np.nan, dtype=np.float64)
    for a_idx, b_idx, c_idx in combinations(range(len(source_names)), 3):
        a = source_vectors[:, a_idx, :]
        b = source_vectors[:, b_idx, :]
        c = source_vectors[:, c_idx, :]
        v0 = b - a
        v1 = c - a
        rhs = -a
        d00 = np.einsum("ij,ij->i", v0, v0)
        d01 = np.einsum("ij,ij->i", v0, v1)
        d11 = np.einsum("ij,ij->i", v1, v1)
        d20 = np.einsum("ij,ij->i", rhs, v0)
        d21 = np.einsum("ij,ij->i", rhs, v1)
        denom = d00 * d11 - d01 * d01
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            projection = a + v[:, None] * v0 + w[:, None] * v1
            distance = np.linalg.norm(projection, axis=1)
        inside = (u >= 0.0) & (v >= 0.0) & (w >= 0.0) & np.isfinite(u) & np.isfinite(v) & np.isfinite(w)
        mask = inside & (distance < best_distance)
        best_distance[mask] = distance[mask]
        best_triangle[mask] = f"{source_names[a_idx]}-{source_names[b_idx]}-{source_names[c_idx]}"
        best_weights[mask, 0] = u[mask]
        best_weights[mask, 1] = v[mask]
        best_weights[mask, 2] = w[mask]
    return pd.DataFrame(
        {
            "best_triangle": best_triangle,
            "best_triangle_u": best_weights[:, 0],
            "best_triangle_v": best_weights[:, 1],
            "best_triangle_w": best_weights[:, 2],
            "best_triangle_distance_m": best_distance,
        },
    )


def analyze_source_blend_residual(
    *,
    row_summary: Path,
    bridge_rows: Path,
    chunk_epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if chunk_epochs <= 0:
        raise ValueError("chunk_epochs must be positive")
    rows, sources = _read_rows(row_summary, bridge_rows)
    source_vectors, source_names = _source_points_m(rows, sources)
    out = rows[["tripId", "UnixTimeMillis", "epoch_index"]].copy() if "tripId" in rows.columns else rows[["UnixTimeMillis", "epoch_index"]].copy()
    out = pd.concat(
        [
            out.reset_index(drop=True),
            _closest_points(source_vectors, source_names),
            _closest_segments(source_vectors, source_names),
            _closest_triangle_interiors(source_vectors, source_names),
        ],
        axis=1,
    )
    out["best_hull_kind"] = "segment"
    out["best_hull_source"] = out["best_segment_pair"]
    out["best_hull_distance_m"] = out["best_segment_distance_m"]
    triangle_mask = out["best_triangle_distance_m"] < out["best_hull_distance_m"]
    out.loc[triangle_mask, "best_hull_kind"] = "triangle"
    out.loc[triangle_mask, "best_hull_source"] = out.loc[triangle_mask, "best_triangle"]
    out.loc[triangle_mask, "best_hull_distance_m"] = out.loc[triangle_mask, "best_triangle_distance_m"]
    out["chunk_start_epoch"] = (out["epoch_index"] // chunk_epochs) * chunk_epochs

    chunk_rows: list[dict[str, object]] = []
    for chunk_start, group in out.groupby("chunk_start_epoch", sort=True):
        record: dict[str, object] = {
            "chunk_start_epoch": int(chunk_start),
            "chunk_end_epoch": int(group["epoch_index"].max()) + 1,
            "rows": int(len(group)),
            "best_point_top_source": str(group["best_point_source"].value_counts().idxmax()),
            "best_segment_top_pair": str(group["best_segment_pair"].value_counts().idxmax()),
            "best_hull_top_source": str(group["best_hull_source"].value_counts().idxmax()),
            **_score_record(group["best_point_distance_m"].to_numpy(), "best_point_distance"),
            **_score_record(group["best_segment_distance_m"].to_numpy(), "best_segment_distance"),
            **_score_record(group["best_hull_distance_m"].to_numpy(), "best_hull_distance"),
        }
        chunk_rows.append(record)
    chunks = pd.DataFrame(chunk_rows)
    summary = {
        "row_summary": str(row_summary),
        "bridge_rows": str(bridge_rows),
        "rows": int(len(out)),
        "source_names": source_names,
        "overall": {
            **_score_record(out["best_point_distance_m"].to_numpy(), "best_point_distance"),
            **_score_record(out["best_segment_distance_m"].to_numpy(), "best_segment_distance"),
            **_score_record(out["best_hull_distance_m"].to_numpy(), "best_hull_distance"),
        },
        "top_chunks_by_hull_p95": (
            chunks.sort_values(["best_hull_distance_p95_m", "best_hull_distance_max_m"], ascending=[False, False])
            .head(10)
            .to_dict(orient="records")
        ),
        "worst_rows_by_hull_distance": (
            out.sort_values("best_hull_distance_m", ascending=False)
            .head(20)
            .to_dict(orient="records")
        ),
    }
    return out, chunks, summary


def write_outputs(output_dir: Path, rows: pd.DataFrame, chunks: pd.DataFrame, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "source_blend_residual_rows.csv"
    chunks_path = output_dir / "source_blend_residual_chunks.csv"
    rows.to_csv(rows_path, index=False)
    chunks.to_csv(chunks_path, index=False)
    payload = {
        **summary,
        "rows_csv": str(rows_path),
        "chunks_csv": str(chunks_path),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-summary", type=Path, required=True)
    parser.add_argument("--bridge-rows", type=Path, required=True)
    parser.add_argument("--chunk-epochs", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows, chunks, summary = analyze_source_blend_residual(
        row_summary=args.row_summary,
        bridge_rows=args.bridge_rows,
        chunk_epochs=args.chunk_epochs,
    )
    write_outputs(args.output_dir, rows, chunks, summary)
    print(f"analyzed: {summary['rows']} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
