"""Materialize GSDC2023 source schedules from bridge artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.analyze_gsdc2023_target_trip_source_delta import KEY_COLUMNS, source_coordinate_columns


OUTPUT_COORDINATE_COLUMNS = ["best_source_latitude_degrees", "best_source_longitude_degrees"]


def _parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected LABEL=path, got {value!r}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"expected non-empty LABEL in {value!r}")
    return label, Path(path)


def _read_bridge_source(path: Path, target_trip: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if target_trip is not None and "tripId" in frame.columns:
        frame = frame[frame["tripId"] == target_trip].copy()
    if "UnixTimeMillis" not in frame.columns:
        raise ValueError(f"{path} is missing UnixTimeMillis")
    if frame["UnixTimeMillis"].duplicated().any():
        raise ValueError(f"{path} contains duplicate UnixTimeMillis values")
    if not source_coordinate_columns(frame):
        raise ValueError(f"{path} contains no source coordinate column pairs")
    return frame.sort_values("UnixTimeMillis").reset_index(drop=True)


def _schedule_bridge_source_columns(schedule_rows: pd.DataFrame) -> tuple[str, str | None]:
    if "best_bridge_source" in schedule_rows.columns:
        return "best_bridge_source", "bridge_source"
    if "best_reference_source" in schedule_rows.columns:
        return "best_reference_source", "single_source"
    if "best_source" in schedule_rows.columns:
        return "best_source", "single_source"
    raise ValueError("schedule rows must contain best_bridge_source, best_reference_source, or best_source")


def _split_bridge_source(value: object, mode: str | None, bridge_labels: list[str]) -> tuple[str, str]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid schedule source value: {value!r}")
    if mode == "bridge_source":
        if ":" not in value:
            raise ValueError(f"best_bridge_source must be LABEL:source, got {value!r}")
        label, source = value.split(":", 1)
    else:
        if len(bridge_labels) != 1:
            raise ValueError("single-source schedules require exactly one --bridge-source")
        label, source = bridge_labels[0], value
    if not label or not source:
        raise ValueError(f"invalid schedule source value: {value!r}")
    return label, source


def materialize_source_schedule_rows(
    *,
    schedule_rows: pd.DataFrame,
    bridge_sources: list[tuple[str, Path]],
    target_trip: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = set(KEY_COLUMNS)
    missing = required.difference(schedule_rows.columns)
    if missing:
        raise ValueError(f"schedule rows are missing columns: {sorted(missing)}")
    if schedule_rows[KEY_COLUMNS].duplicated().any():
        raise ValueError("schedule rows contain duplicate tripId/UnixTimeMillis values")
    if not bridge_sources:
        raise ValueError("at least one bridge source is required")
    labels = [label for label, _ in bridge_sources]
    if len(set(labels)) != len(labels):
        raise ValueError(f"bridge source labels must be unique: {labels}")

    source_column, mode = _schedule_bridge_source_columns(schedule_rows)
    materialized = schedule_rows.copy()
    materialized["materialized_bridge_label"] = ""
    materialized["materialized_source"] = ""
    materialized[OUTPUT_COORDINATE_COLUMNS[0]] = pd.NA
    materialized[OUTPUT_COORDINATE_COLUMNS[1]] = pd.NA

    bridge_frames: dict[str, pd.DataFrame] = {}
    bridge_source_columns: dict[str, dict[str, tuple[str, str]]] = {}
    for label, path in bridge_sources:
        frame = _read_bridge_source(path, target_trip)
        bridge_frames[label] = frame
        bridge_source_columns[label] = source_coordinate_columns(frame)

    source_pairs = materialized[source_column].map(lambda value: _split_bridge_source(value, mode, labels))
    materialized["materialized_bridge_label"] = [label for label, _ in source_pairs]
    materialized["materialized_source"] = [source for _, source in source_pairs]

    for label, frame in bridge_frames.items():
        label_mask = materialized["materialized_bridge_label"] == label
        if not label_mask.any():
            continue
        bridge_columns = bridge_source_columns[label]
        for source, (lat_column, lon_column) in bridge_columns.items():
            mask = label_mask & (materialized["materialized_source"] == source)
            if not mask.any():
                continue
            patch = materialized.loc[mask, KEY_COLUMNS].merge(
                frame[["UnixTimeMillis", lat_column, lon_column]],
                on="UnixTimeMillis",
                how="left",
                validate="one_to_one",
            )
            if patch[[lat_column, lon_column]].isna().any(axis=None):
                raise ValueError(f"bridge source {label}:{source} does not cover every scheduled timestamp")
            materialized.loc[mask, OUTPUT_COORDINATE_COLUMNS] = patch[[lat_column, lon_column]].to_numpy()

    missing_coordinates = materialized[OUTPUT_COORDINATE_COLUMNS].isna().any(axis=1)
    if missing_coordinates.any():
        missing_sources = (
            materialized.loc[missing_coordinates, ["materialized_bridge_label", "materialized_source"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(f"schedule references unavailable bridge source(s): {missing_sources}")

    counts = materialized["materialized_bridge_label"].str.cat(materialized["materialized_source"], sep=":").value_counts()
    summary = {
        "rows": int(len(materialized)),
        "target_trip": target_trip,
        "schedule_source_column": source_column,
        "bridge_sources": [{"label": label, "path": str(path)} for label, path in bridge_sources],
        "materialized_source_counts": {str(name): int(count) for name, count in counts.sort_index().items()},
    }
    return materialized, summary


def write_outputs(output_dir: Path, rows: pd.DataFrame, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "materialized_source_schedule_rows.csv"
    rows.to_csv(rows_path, index=False)
    payload = {**summary, "rows_csv": str(rows_path)}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-rows", type=Path, required=True)
    parser.add_argument("--bridge-source", action="append", default=[], help="LABEL=bridge_positions.csv")
    parser.add_argument("--target-trip")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows, summary = materialize_source_schedule_rows(
        schedule_rows=pd.read_csv(args.schedule_rows),
        bridge_sources=[_parse_label_path(value) for value in args.bridge_source],
        target_trip=args.target_trip,
    )
    write_outputs(args.output_dir, rows, summary)
    print(f"materialized: rows={summary['rows']} sources={len(summary['materialized_source_counts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
