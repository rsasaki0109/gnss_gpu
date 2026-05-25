"""Build a chunk-merged submission CSV from a chunk_epochs=100 bridge output.

Applies the Phase 76-77 *adjacent sub-chunk merge* approximation to a
chunk_epochs=100 production run.  For each trip, the per-chunk
``ChunkSelectionRecord`` set is merged pairwise and the production
``select_gated_chunk_source`` gate is re-evaluated on the merged-window
``ChunkCandidateQuality`` summaries.  Rows whose chunk's ``gated_source``
prediction differs from the recorded ``SelectedSource`` are re-pointed
to the per-source position columns already stored in
``bridge_positions.csv`` (``BaselineLatitudeDegrees``,
``RawWlsLatitudeDegrees``, ``FgoLatitudeDegrees`` and altitudes).

Approximations: see ``experiments.gsdc2023_ab_chunk_merge`` for the full
list.  The merge approximation recovers ~48% of the chunk_epochs=200
``A_ROOT`` baseline's row diff (audited 2026-05-22 on 40 trips), with the
remaining 53% requiring an actual chunk_epochs=200 run (fix #1).

Source-column mapping
---------------------

The bridge writes only three per-source position columns even though
``fgo_no_tdcp`` is a distinct source label: positions for ``fgo_no_tdcp``
selections live in the ``LatitudeDegrees``/``LongitudeDegrees`` row
itself (already SelectedSource-stamped).  The merge fix only redirects
*away* from ``fgo_no_tdcp`` (toward ``baseline``/``fgo``) per the
2026-05-22 40-trip audit, so the absence of a per-source column for
``fgo_no_tdcp`` does not block the post-process.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.build_gsdc2023_bridge_submission import (
    COORDINATE_COLUMNS,
    submission_from_bridge_tables,
)
from experiments.gsdc2023_ab_chunk_merge import (
    load_bridge_chunk_records,
    predict_merged_gated_sources,
)


_SOURCE_LATITUDE_COLUMNS: dict[str, str] = {
    "baseline": "BaselineLatitudeDegrees",
    "raw_wls": "RawWlsLatitudeDegrees",
    "fgo": "FgoLatitudeDegrees",
}
_SOURCE_LONGITUDE_COLUMNS: dict[str, str] = {
    "baseline": "BaselineLongitudeDegrees",
    "raw_wls": "RawWlsLongitudeDegrees",
    "fgo": "FgoLongitudeDegrees",
}
_SOURCE_ALTITUDE_COLUMNS: dict[str, str] = {
    "baseline": "BaselineAltitudeMeters",
    "raw_wls": "RawWlsAltitudeMeters",
    "fgo": "FgoAltitudeMeters",
}


@dataclass(frozen=True)
class TripMergeResult:
    trip_id: str
    n_epochs: int
    chunks_total: int
    chunks_changed: int
    rows_total: int
    rows_changed: int
    rows_skipped_missing_column: int
    flips: Counter
    source_counts_before: Counter
    source_counts_after: Counter


def _apply_merge_to_trip(
    bridge_csv_path: Path,
    chunk_records: list[dict[str, Any]],
    merged_predictions: list[tuple[int, int, str]],
) -> tuple[pd.DataFrame, TripMergeResult]:
    """Apply merged-source predictions to a single trip's bridge_positions.csv.

    Returns (modified_dataframe, result_summary).
    """

    df = pd.read_csv(bridge_csv_path)
    n_epochs = len(df)
    rows_changed = 0
    rows_skipped = 0
    chunks_changed = 0
    flips: Counter = Counter()
    source_before = Counter(df["SelectedSource"].astype(str))

    # Build epoch -> bridge-actual chunk's gated_source map.
    bridge_gated_by_epoch: list[str | None] = [None] * n_epochs
    for c in chunk_records:
        s = max(0, min(int(c["start_epoch"]), n_epochs))
        e = max(0, min(int(c["end_epoch"]), n_epochs))
        src = str(c.get("gated_source") or "baseline")
        for i in range(s, e):
            bridge_gated_by_epoch[i] = src

    for start, end, predicted in merged_predictions:
        start = max(0, min(start, n_epochs))
        end = max(0, min(end, n_epochs))
        if start >= end:
            continue
        # Determine bridge actual source for this merged window via majority.
        sources = [bridge_gated_by_epoch[i] for i in range(start, end) if bridge_gated_by_epoch[i] is not None]
        if not sources:
            continue
        bridge_src = Counter(sources).most_common(1)[0][0]
        if predicted == bridge_src:
            continue
        # Apply the swap if the predicted source has per-source columns.
        if predicted not in _SOURCE_LATITUDE_COLUMNS:
            # We cannot honestly route rows to a per-source destination we do
            # not store; record and skip.
            rows_skipped += end - start
            continue
        chunks_changed += 1
        flips[(bridge_src, predicted)] += end - start
        lat_col = _SOURCE_LATITUDE_COLUMNS[predicted]
        lon_col = _SOURCE_LONGITUDE_COLUMNS[predicted]
        alt_col = _SOURCE_ALTITUDE_COLUMNS[predicted]
        valid = df.loc[start:end - 1, [lat_col, lon_col, alt_col]].notna().all(axis=1)
        valid_idx = df.index[start:end][valid.to_numpy()]
        if len(valid_idx) == 0:
            rows_skipped += end - start
            continue
        df.loc[valid_idx, "LatitudeDegrees"] = df.loc[valid_idx, lat_col].to_numpy()
        df.loc[valid_idx, "LongitudeDegrees"] = df.loc[valid_idx, lon_col].to_numpy()
        df.loc[valid_idx, "AltitudeMeters"] = df.loc[valid_idx, alt_col].to_numpy()
        df.loc[valid_idx, "SelectedSource"] = predicted
        rows_changed += int(len(valid_idx))
        rows_skipped += int((end - start) - len(valid_idx))

    source_after = Counter(df["SelectedSource"].astype(str))
    return df, TripMergeResult(
        trip_id=str(bridge_csv_path.parent.parent.name) + "/" + str(bridge_csv_path.parent.name),
        n_epochs=n_epochs,
        chunks_total=len(merged_predictions),
        chunks_changed=chunks_changed,
        rows_total=n_epochs,
        rows_changed=rows_changed,
        rows_skipped_missing_column=rows_skipped,
        flips=flips,
        source_counts_before=source_before,
        source_counts_after=source_after,
    )


def _resolve_bridge_csv(bridge_root: Path, trip_id: str) -> Path | None:
    path = bridge_root / trip_id / "bridge_positions.csv"
    return path if path.exists() else None


def run(
    *,
    bridge_root: Path,
    sample_submission_path: Path,
    output_csv: Path,
    output_summary: Path,
    allow_partial: bool,
    interpolate_missing: bool,
) -> int:
    chunk_records = load_bridge_chunk_records(bridge_root)
    sample = pd.read_csv(sample_submission_path)
    bridge_tables: dict[str, pd.DataFrame] = {}
    per_trip: list[TripMergeResult] = []
    total_chunks = 0
    total_chunks_changed = 0
    total_rows_changed = 0
    total_rows_skipped = 0
    aggregate_flips: Counter = Counter()
    for trip_id, payload in sorted(chunk_records.items()):
        csv_path = _resolve_bridge_csv(bridge_root, trip_id)
        if csv_path is None:
            print(f"[skip] {trip_id}: bridge_positions.csv missing", flush=True)
            continue
        merged_preds = predict_merged_gated_sources(payload["chunks"])
        df, res = _apply_merge_to_trip(csv_path, payload["chunks"], merged_preds)
        bridge_tables[trip_id] = df
        per_trip.append(res)
        total_chunks += res.chunks_total
        total_chunks_changed += res.chunks_changed
        total_rows_changed += res.rows_changed
        total_rows_skipped += res.rows_skipped_missing_column
        for k, v in res.flips.items():
            aggregate_flips[k] += v
        print(
            f"[merge] {trip_id} chunks_changed={res.chunks_changed}/{res.chunks_total}"
            f" rows_changed={res.rows_changed}/{res.rows_total}"
            f" rows_skipped={res.rows_skipped_missing_column}",
            flush=True,
        )

    output, sub_summary = submission_from_bridge_tables(
        sample,
        bridge_tables,
        allow_partial=allow_partial,
        interpolate_missing=interpolate_missing,
    )
    output.to_csv(output_csv, index=False)
    summary = {
        "bridge_output_root": str(bridge_root),
        "sample_submission": str(sample_submission_path),
        "output": str(output_csv),
        "trips_processed": len(per_trip),
        "total_chunks": int(total_chunks),
        "total_chunks_changed": int(total_chunks_changed),
        "total_rows_changed": int(total_rows_changed),
        "total_rows_skipped_missing_column": int(total_rows_skipped),
        "aggregate_flip_table": {
            f"{a}->{b}": int(n) for (a, b), n in sorted(aggregate_flips.items(), key=lambda x: -x[1])
        },
        "submission": sub_summary,
        "per_trip": [
            {
                "trip_id": r.trip_id,
                "n_epochs": int(r.n_epochs),
                "chunks_total": int(r.chunks_total),
                "chunks_changed": int(r.chunks_changed),
                "rows_changed": int(r.rows_changed),
                "rows_skipped_missing_column": int(r.rows_skipped_missing_column),
                "flip_table": {
                    f"{a}->{b}": int(n) for (a, b), n in sorted(r.flips.items(), key=lambda x: -x[1])
                },
                "source_counts_before": dict(r.source_counts_before),
                "source_counts_after": dict(r.source_counts_after),
            }
            for r in per_trip
        ],
    }
    output_summary.write_text(json.dumps(summary, indent=2))
    print(f"wrote: {output_csv}")
    print(f"wrote: {output_summary}")
    print(
        f"summary: trips={len(per_trip)} chunks_changed={total_chunks_changed}/{total_chunks}"
        f" rows_changed={total_rows_changed} rows_skipped={total_rows_skipped}"
    )
    print("aggregate flips (bridge_actual -> merged_predicted : rows):")
    for (a, b), n in sorted(aggregate_flips.items(), key=lambda x: -x[1]):
        print(f"  {a:>16}  ->  {b:<16}  : {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-output-root", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--interpolate-missing", action="store_true")
    args = parser.parse_args(argv)
    return run(
        bridge_root=args.bridge_output_root,
        sample_submission_path=args.sample_submission,
        output_csv=args.output,
        output_summary=args.summary,
        allow_partial=bool(args.allow_partial),
        interpolate_missing=bool(args.interpolate_missing),
    )


if __name__ == "__main__":
    sys.exit(main())
