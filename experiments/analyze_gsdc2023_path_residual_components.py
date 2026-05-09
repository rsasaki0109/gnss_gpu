"""Decompose reference-vs-source path residuals into tangent/normal components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analyze_gsdc2023_trajectory_shape import _local_tracks, _read_rows, _score_record


def _unit_tangent(track: np.ndarray) -> np.ndarray:
    tangent = np.full_like(track, np.nan, dtype=np.float64)
    if len(track) == 1:
        return tangent
    tangent[0] = track[1] - track[0]
    tangent[-1] = track[-1] - track[-2]
    if len(track) > 2:
        tangent[1:-1] = track[2:] - track[:-2]
    norm = np.linalg.norm(tangent, axis=1)
    valid = norm > 1e-9
    tangent[valid] /= norm[valid, None]
    return tangent


def _component_rows(rows: pd.DataFrame, tracks: dict[str, np.ndarray]) -> pd.DataFrame:
    reference = tracks["reference"]
    out = rows[["tripId", "UnixTimeMillis", "epoch_index"]].copy() if "tripId" in rows.columns else rows[["UnixTimeMillis", "epoch_index"]].copy()
    for source_name, source_track in tracks.items():
        if source_name == "reference":
            continue
        tangent = _unit_tangent(source_track)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        residual = reference - source_track
        tangent_residual = np.einsum("ij,ij->i", residual, tangent)
        normal_residual = np.einsum("ij,ij->i", residual, normal)
        out[f"{source_name}_distance_m"] = np.linalg.norm(residual, axis=1)
        out[f"{source_name}_tangent_residual_m"] = tangent_residual
        out[f"{source_name}_normal_residual_m"] = normal_residual
        out[f"{source_name}_abs_tangent_residual_m"] = np.abs(tangent_residual)
        out[f"{source_name}_abs_normal_residual_m"] = np.abs(normal_residual)
    return out


def _median_corrected_distance(
    tangent_residual: np.ndarray,
    normal_residual: np.ndarray,
    *,
    remove_tangent: bool,
    remove_normal: bool,
) -> np.ndarray:
    tangent = np.asarray(tangent_residual, dtype=np.float64).copy()
    normal = np.asarray(normal_residual, dtype=np.float64).copy()
    if remove_tangent:
        valid_tangent = np.isfinite(tangent)
        if np.any(valid_tangent):
            tangent -= float(np.nanmedian(tangent[valid_tangent]))
    if remove_normal:
        valid_normal = np.isfinite(normal)
        if np.any(valid_normal):
            normal -= float(np.nanmedian(normal[valid_normal]))
    return np.sqrt(tangent * tangent + normal * normal)


def _component_record(group: pd.DataFrame, source_name: str) -> dict[str, float | int]:
    tangent = group[f"{source_name}_tangent_residual_m"].to_numpy(dtype=np.float64)
    normal = group[f"{source_name}_normal_residual_m"].to_numpy(dtype=np.float64)
    distance = group[f"{source_name}_distance_m"].to_numpy(dtype=np.float64)
    abs_tangent = np.abs(tangent)
    abs_normal = np.abs(normal)
    return {
        "rows": int(len(group)),
        "median_tangent_residual_m": float(np.nanmedian(tangent)),
        "median_normal_residual_m": float(np.nanmedian(normal)),
        "median_abs_tangent_residual_m": float(np.nanmedian(abs_tangent)),
        "median_abs_normal_residual_m": float(np.nanmedian(abs_normal)),
        **_score_record(distance, "distance"),
        **_score_record(abs_tangent, "abs_tangent_residual"),
        **_score_record(abs_normal, "abs_normal_residual"),
        **_score_record(
            _median_corrected_distance(tangent, normal, remove_tangent=True, remove_normal=False),
            "distance_after_tangent_median",
        ),
        **_score_record(
            _median_corrected_distance(tangent, normal, remove_tangent=False, remove_normal=True),
            "distance_after_normal_median",
        ),
        **_score_record(
            _median_corrected_distance(tangent, normal, remove_tangent=True, remove_normal=True),
            "distance_after_component_medians",
        ),
    }


def analyze_path_residual_components(
    *,
    row_summary: Path,
    bridge_rows: Path,
    chunk_epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if chunk_epochs <= 0:
        raise ValueError("chunk_epochs must be positive")
    rows, sources = _read_rows(row_summary, bridge_rows)
    component_rows = _component_rows(rows, _local_tracks(rows, sources))
    source_names = list(sources)

    summary_rows: list[dict[str, object]] = []
    max_epoch = int(component_rows["epoch_index"].max())
    for source_name in source_names:
        summary_rows.append(
            {
                "source": source_name,
                "chunk_start_epoch": -1,
                "chunk_end_epoch": max_epoch + 1,
                **_component_record(component_rows, source_name),
            },
        )
        chunk_start = (component_rows["epoch_index"] // chunk_epochs) * chunk_epochs
        for start, group in component_rows.groupby(chunk_start, sort=True):
            summary_rows.append(
                {
                    "source": source_name,
                    "chunk_start_epoch": int(start),
                    "chunk_end_epoch": int(min(start + chunk_epochs, max_epoch + 1)),
                    **_component_record(group, source_name),
                },
            )
    component_summary = pd.DataFrame(summary_rows)
    overall = component_summary[component_summary["chunk_start_epoch"] == -1].copy()
    chunks = component_summary[component_summary["chunk_start_epoch"] >= 0].copy()
    payload = {
        "row_summary": str(row_summary),
        "bridge_rows": str(bridge_rows),
        "rows": int(len(component_rows)),
        "source_names": source_names,
        "overall_by_source": overall.to_dict(orient="records"),
        "top_chunks_by_component_corrected_p95": (
            chunks.sort_values(
                ["distance_after_component_medians_p95_m", "distance_after_component_medians_max_m"],
                ascending=[False, False],
            )
            .head(20)
            .to_dict(orient="records")
        ),
        "top_chunks_by_normal_median_gain": (
            chunks.assign(
                normal_median_p95_gain_m=chunks["distance_p95_m"] - chunks["distance_after_normal_median_p95_m"],
            )
            .sort_values(["normal_median_p95_gain_m", "distance_p95_m"], ascending=[False, False])
            .head(20)
            .to_dict(orient="records")
        ),
    }
    return component_rows, component_summary, payload


def write_outputs(
    output_dir: Path,
    rows: pd.DataFrame,
    component_summary: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "path_residual_component_rows.csv"
    summary_path = output_dir / "path_residual_component_summary.csv"
    rows.to_csv(rows_path, index=False)
    component_summary.to_csv(summary_path, index=False)
    payload = {
        **summary,
        "rows_csv": str(rows_path),
        "component_summary_csv": str(summary_path),
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

    rows, component_summary, summary = analyze_path_residual_components(
        row_summary=args.row_summary,
        bridge_rows=args.bridge_rows,
        chunk_epochs=args.chunk_epochs,
    )
    write_outputs(args.output_dir, rows, component_summary, summary)
    print(f"analyzed: {summary['rows']} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
