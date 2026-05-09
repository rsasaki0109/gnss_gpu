"""Materialize rows for submission timestamps absent from GSDC bridge artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.reconstruct_gsdc2023_matlab_reference_submission import (
    KEY_COLUMNS,
    SOURCE_COORDINATE_COLUMNS,
)


def _bridge_path(bridge_root: Path, trip_id: str) -> Path:
    course, phone = str(trip_id).split("/", 1)
    return bridge_root / course / phone / "bridge_positions.csv"


def _nearest_selected_row(bridge: pd.DataFrame, unix_time_millis: int) -> tuple[pd.Series, str, int]:
    times = bridge["UnixTimeMillis"].astype("int64")
    before = bridge.loc[times < unix_time_millis].tail(1)
    after = bridge.loc[times > unix_time_millis].head(1)
    if before.empty and after.empty:
        raise ValueError("bridge contains no neighboring rows")
    if before.empty:
        row = after.iloc[0]
        return row, "next", int(row["UnixTimeMillis"]) - unix_time_millis
    if after.empty:
        row = before.iloc[0]
        return row, "previous", unix_time_millis - int(row["UnixTimeMillis"])

    prev_row = before.iloc[0]
    next_row = after.iloc[0]
    prev_dt = unix_time_millis - int(prev_row["UnixTimeMillis"])
    next_dt = int(next_row["UnixTimeMillis"]) - unix_time_millis
    if prev_dt <= next_dt:
        return prev_row, "previous", prev_dt
    return next_row, "next", next_dt


def materialize_missing_bridge_timestamp_rows(
    *,
    submission: pd.DataFrame,
    bridge_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    missing = set(KEY_COLUMNS + ["LatitudeDegrees", "LongitudeDegrees"]).difference(submission.columns)
    if missing:
        raise ValueError(f"submission is missing columns: {sorted(missing)}")
    if submission[KEY_COLUMNS].duplicated().any():
        raise ValueError("submission contains duplicate tripId/UnixTimeMillis values")

    output_rows: list[dict[str, object]] = []
    trip_summary: list[dict[str, object]] = []
    for trip_id, trip_rows in submission.groupby("tripId", sort=False):
        trip_id = str(trip_id)
        path = _bridge_path(bridge_root, trip_id)
        if not path.is_file():
            continue
        bridge = pd.read_csv(path)
        required = {"UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"}
        missing_bridge = required.difference(bridge.columns)
        if missing_bridge:
            raise ValueError(f"{path} is missing columns: {sorted(missing_bridge)}")
        bridge = bridge.sort_values("UnixTimeMillis").reset_index(drop=True)
        if bridge["UnixTimeMillis"].duplicated().any():
            raise ValueError(f"{path} contains duplicate UnixTimeMillis values")

        bridge_times = set(bridge["UnixTimeMillis"].astype("int64"))
        trip_missing_count = 0
        for _, row in trip_rows.sort_values("UnixTimeMillis").iterrows():
            unix_time_millis = int(row["UnixTimeMillis"])
            if unix_time_millis in bridge_times:
                continue
            nearest, side, delta_ms = _nearest_selected_row(bridge, unix_time_millis)
            output_rows.append(
                {
                    "tripId": trip_id,
                    "UnixTimeMillis": unix_time_millis,
                    SOURCE_COORDINATE_COLUMNS[0]: float(nearest["LatitudeDegrees"]),
                    SOURCE_COORDINATE_COLUMNS[1]: float(nearest["LongitudeDegrees"]),
                    "materialized_bridge_label": "bridge",
                    "materialized_source": f"nearest_selected_{side}",
                    "nearest_bridge_unix_time_millis": int(nearest["UnixTimeMillis"]),
                    "nearest_bridge_delta_ms": int(delta_ms),
                },
            )
            trip_missing_count += 1
        if trip_missing_count:
            trip_summary.append(
                {
                    "tripId": trip_id,
                    "bridge_path": str(path),
                    "missing_rows": trip_missing_count,
                },
            )

    rows = pd.DataFrame(output_rows)
    if rows.empty:
        rows = pd.DataFrame(
            columns=KEY_COLUMNS
            + SOURCE_COORDINATE_COLUMNS
            + [
                "materialized_bridge_label",
                "materialized_source",
                "nearest_bridge_unix_time_millis",
                "nearest_bridge_delta_ms",
            ],
        )
    counts = rows["materialized_source"].value_counts().sort_index().to_dict()
    summary = {
        "bridge_root": str(bridge_root),
        "rows": int(len(rows)),
        "trips": int(rows["tripId"].nunique()) if not rows.empty else 0,
        "materialized_source_counts": {str(name): int(count) for name, count in counts.items()},
        "trip_summary": trip_summary,
    }
    return rows, summary


def write_outputs(output_dir: Path, rows: pd.DataFrame, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "missing_bridge_timestamp_rows.csv"
    rows.to_csv(rows_path, index=False)
    payload = {**summary, "rows_csv": str(rows_path)}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--bridge-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows, summary = materialize_missing_bridge_timestamp_rows(
        submission=pd.read_csv(args.submission),
        bridge_root=args.bridge_root,
    )
    write_outputs(args.output_dir, rows, summary)
    print(f"materialized missing bridge timestamps: rows={summary['rows']} trips={summary['trips']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
