"""Find the closest source row across multiple GSDC bridge artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from experiments.analyze_gsdc2023_target_trip_source_delta import (
    KEY_COLUMNS,
    _read_submission_trip,
    source_coordinate_columns,
)
from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


def _safe_label(label: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z]+", "_", label.strip()).strip("_").lower()
    if not safe:
        raise ValueError("bridge source label must not be empty")
    return safe


def _parse_bridge_source(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"bridge source must be LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    return _safe_label(label), Path(path)


def _read_bridge(path: Path, target_trip: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "tripId" in frame.columns:
        frame = frame[frame["tripId"] == target_trip].copy()
    if "UnixTimeMillis" not in frame.columns:
        raise ValueError(f"{path} is missing UnixTimeMillis")
    if frame.empty:
        raise ValueError(f"{path} contains no rows for {target_trip!r}")
    if not source_coordinate_columns(frame):
        raise ValueError(f"{path} contains no source coordinate columns")
    return frame.sort_values("UnixTimeMillis").reset_index(drop=True)


def _score_columns(values: np.ndarray, suffix: str = "") -> dict[str, float | int]:
    score = gsdc_score_m(values)
    return {
        f"mean{suffix}_m": score["mean_m"],
        f"p50{suffix}_m": score["p50_m"],
        f"p95{suffix}_m": score["p95_m"],
        f"max{suffix}_m": score["max_m"],
        f"rows_gt_1m{suffix}": int(np.count_nonzero(values > 1.0)),
        f"rows_gt_5m{suffix}": int(np.count_nonzero(values > 5.0)),
    }


def analyze_multi_bridge_source_delta(
    *,
    reference_submission: Path,
    target_trip: str,
    bridge_sources: list[tuple[str, Path]],
    chunk_epochs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if chunk_epochs <= 0:
        raise ValueError("chunk_epochs must be positive")
    if not bridge_sources:
        raise ValueError("at least one bridge source is required")
    labels = [label for label, _ in bridge_sources]
    if len(set(labels)) != len(labels):
        raise ValueError(f"bridge source labels must be unique: {labels}")

    reference = _read_submission_trip(reference_submission, target_trip)
    rows = reference.rename(
        columns={
            "LatitudeDegrees": "LatitudeDegrees_reference",
            "LongitudeDegrees": "LongitudeDegrees_reference",
        },
    )
    rows.insert(2, "epoch_index", np.arange(len(rows), dtype=np.int64))
    candidate_columns: list[tuple[str, str, str, str, str, str]] = []
    source_columns_payload: dict[str, dict[str, list[str]]] = {}
    for label, path in bridge_sources:
        bridge = _read_bridge(path, target_trip)
        sources = source_coordinate_columns(bridge)
        source_columns_payload[label] = {source: [lat, lon] for source, (lat, lon) in sources.items()}
        rename_columns: dict[str, str] = {}
        keep_columns = ["UnixTimeMillis"]
        for source, (lat_column, lon_column) in sources.items():
            out_lat = f"{label}__{source}__LatitudeDegrees"
            out_lon = f"{label}__{source}__LongitudeDegrees"
            rename_columns[lat_column] = out_lat
            rename_columns[lon_column] = out_lon
            keep_columns.extend([lat_column, lon_column])
            candidate_columns.append((label, source, out_lat, out_lon, f"{label}:{source}", f"distance_to_{label}__{source}_m"))
        rows = rows.merge(
            bridge[keep_columns].rename(columns=rename_columns),
            on="UnixTimeMillis",
            how="inner",
            validate="one_to_one",
        )

    if len(rows) != len(reference):
        raise ValueError(f"matched row count mismatch: reference={len(reference)} matched={len(rows)}")

    distance_matrix = np.empty((len(rows), len(candidate_columns)), dtype=np.float64)
    for index, (_, _, lat_column, lon_column, _, distance_column) in enumerate(candidate_columns):
        distance = haversine_m(
            rows["LatitudeDegrees_reference"].to_numpy(),
            rows["LongitudeDegrees_reference"].to_numpy(),
            rows[lat_column].to_numpy(),
            rows[lon_column].to_numpy(),
        )
        rows[distance_column] = distance
        distance_matrix[:, index] = distance

    best_index = np.nanargmin(distance_matrix, axis=1)
    best_distance = distance_matrix[np.arange(len(rows)), best_index]
    rows["best_bridge_label"] = [candidate_columns[index][0] for index in best_index]
    rows["best_source"] = [candidate_columns[index][1] for index in best_index]
    rows["best_bridge_source"] = [candidate_columns[index][4] for index in best_index]
    rows["best_source_distance_m"] = best_distance
    rows["best_source_latitude_degrees"] = np.nan
    rows["best_source_longitude_degrees"] = np.nan
    for index, (_, _, lat_column, lon_column, bridge_source, _) in enumerate(candidate_columns):
        mask = rows["best_bridge_source"] == bridge_source
        rows.loc[mask, "best_source_latitude_degrees"] = rows.loc[mask, lat_column]
        rows.loc[mask, "best_source_longitude_degrees"] = rows.loc[mask, lon_column]

    chunk_rows: list[dict[str, object]] = []
    for start in range(0, len(rows), chunk_epochs):
        chunk = rows.iloc[start : start + chunk_epochs]
        distance = chunk["best_source_distance_m"].to_numpy()
        record: dict[str, object] = {
            "tripId": target_trip,
            "start_epoch": int(start),
            "end_epoch": int(start + len(chunk)),
            "rows": int(len(chunk)),
            "top_bridge_source": str(chunk["best_bridge_source"].value_counts().idxmax()),
            **_score_columns(distance, "_best_source_distance"),
        }
        for source_name, count in chunk["best_bridge_source"].value_counts().sort_index().items():
            record[f"best_{str(source_name).replace(':', '__')}_rows"] = int(count)
        chunk_rows.append(record)
    chunk_summary = pd.DataFrame(chunk_rows)

    summary = {
        "reference_submission": str(reference_submission),
        "target_trip": target_trip,
        "rows": int(len(rows)),
        "bridge_sources": [{"label": label, "path": str(path)} for label, path in bridge_sources],
        "source_columns": source_columns_payload,
        "best_source_distance": {
            **_score_columns(rows["best_source_distance_m"].to_numpy(), ""),
        },
        "best_bridge_source_counts": rows["best_bridge_source"].value_counts().sort_index().to_dict(),
        "top_chunks_by_best_source_p95": (
            chunk_summary.sort_values(
                ["p95_best_source_distance_m", "max_best_source_distance_m"],
                ascending=[False, False],
            )
            .head(10)
            .to_dict(orient="records")
        ),
        "top_rows_by_best_source_distance": (
            rows.sort_values("best_source_distance_m", ascending=False)
            .head(20)[
                [
                    "tripId",
                    "UnixTimeMillis",
                    "epoch_index",
                    "best_bridge_label",
                    "best_source",
                    "best_bridge_source",
                    "best_source_distance_m",
                ]
            ]
            .to_dict(orient="records")
        ),
    }
    output_columns = [
        "tripId",
        "UnixTimeMillis",
        "epoch_index",
        "LatitudeDegrees_reference",
        "LongitudeDegrees_reference",
        "best_bridge_label",
        "best_source",
        "best_bridge_source",
        "best_source_distance_m",
        "best_source_latitude_degrees",
        "best_source_longitude_degrees",
    ]
    output_columns.extend(column[-1] for column in candidate_columns)
    return rows[output_columns], chunk_summary, summary


def reconstruct_submission(
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
    patch_columns = {"UnixTimeMillis", "best_source_latitude_degrees", "best_source_longitude_degrees"}
    missing_patch = patch_columns.difference(row_summary.columns)
    if missing_patch:
        raise ValueError(f"row summary is missing columns: {sorted(missing_patch)}")
    mask = frame["tripId"] == target_trip
    if not mask.any():
        raise ValueError(f"{candidate_submission} contains no rows for {target_trip!r}")
    joined = frame.loc[mask, ["UnixTimeMillis"]].merge(
        row_summary[["UnixTimeMillis", "best_source_latitude_degrees", "best_source_longitude_degrees"]],
        on="UnixTimeMillis",
        how="left",
        validate="one_to_one",
    )
    if joined[["best_source_latitude_degrees", "best_source_longitude_degrees"]].isna().any(axis=None):
        raise ValueError("row summary does not cover every target trip timestamp")
    out = frame.copy()
    out.loc[mask, "LatitudeDegrees"] = joined["best_source_latitude_degrees"].to_numpy()
    out.loc[mask, "LongitudeDegrees"] = joined["best_source_longitude_degrees"].to_numpy()
    return out, {
        "candidate_submission": str(candidate_submission),
        "target_trip": target_trip,
        "rows_replaced": int(mask.sum()),
        "source": "multi_bridge_best_source",
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
    row_path = output_dir / "multi_bridge_source_delta_rows.csv"
    chunk_path = output_dir / "multi_bridge_source_delta_chunks.csv"
    row_summary.to_csv(row_path, index=False)
    chunk_summary.to_csv(chunk_path, index=False)
    payload = {
        **summary,
        "row_summary_csv": str(row_path),
        "chunk_summary_csv": str(chunk_path),
    }
    if reconstructed_submission is not None:
        reconstructed_path = output_dir / "submission_with_target_trip_multi_bridge_best_source.csv"
        reconstructed_submission.to_csv(reconstructed_path, index=False)
        payload["reconstructed_submission_csv"] = str(reconstructed_path)
        payload["reconstructed_submission"] = reconstructed_summary or {}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-submission", type=Path, required=True)
    parser.add_argument("--target-trip", required=True)
    parser.add_argument("--bridge-source", action="append", default=[], help="LABEL=bridge_positions.csv")
    parser.add_argument("--chunk-epochs", type=int, default=200)
    parser.add_argument("--candidate-submission", type=Path)
    parser.add_argument("--write-reconstructed-submission", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    row_summary, chunk_summary, summary = analyze_multi_bridge_source_delta(
        reference_submission=args.reference_submission,
        target_trip=args.target_trip,
        bridge_sources=[_parse_bridge_source(spec) for spec in args.bridge_source],
        chunk_epochs=args.chunk_epochs,
    )
    reconstructed_submission = None
    reconstructed_summary = None
    if args.write_reconstructed_submission:
        if args.candidate_submission is None:
            raise ValueError("--candidate-submission is required with --write-reconstructed-submission")
        reconstructed_submission, reconstructed_summary = reconstruct_submission(
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
