"""Post-process a GSDC2023 submission CSV with stationary-segment median snap.

Complements :mod:`postprocess_gsdc2023_submission_hampel` (position-MAD
outliers) and :mod:`postprocess_gsdc2023_submission_accel_smooth`
(motion-physics outliers).  Those two attack epochs where the receiver
*moves* impossibly fast / hard.  This module attacks epochs where the
receiver is *standing still* but the reported lat / lng still wobbles by
1 – 3 m due to ordinary GNSS noise.

Algorithm (per tripId)
----------------------

1. Sort rows by UnixTimeMillis.
2. Compute frame-to-frame haversine distance ``d[t]`` between consecutive
   epochs.
3. Detect maximal stationary runs: contiguous index ranges
   ``[lo, hi]`` such that ``d[lo..hi-1] < move_threshold_m`` and
   ``hi - lo >= min_run_length``.
4. For each detected run, replace the lat / lng of every member epoch with
   the *median* lat / lng over the run.  Median (not mean) survives the
   occasional brief GNSS spike inside an otherwise stationary segment.

Defaults: ``move_threshold_m = 0.5, min_run_length = 10`` (~ 10 s of
contiguous near-zero motion at 1 Hz).  Conservative — only fires when
the car is clearly parked at a traffic light or pulled over.

This is a *kinematic-segmentation* post-process — orthogonal to Hampel
(position-MAD) and accel-smoother (acceleration-MAD).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _haversine_m(lat1: np.ndarray, lng1: np.ndarray, lat2: np.ndarray, lng2: np.ndarray) -> np.ndarray:
    R = 6371000.0
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlng / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def detect_stationary_runs(
    d: np.ndarray, *, move_threshold_m: float, min_run_length: int,
) -> list[tuple[int, int]]:
    """Return list of (start, end_inclusive) index ranges of stationary runs.

    ``d[i]`` is the haversine distance between row i and row i+1, length n-1.
    A stationary "run" spans rows ``[lo, hi]`` such that
    ``d[lo..hi-1] < move_threshold_m`` and the run is at least
    ``min_run_length`` rows long.
    """
    n_d = len(d)
    if n_d == 0:
        return []
    is_still = d < move_threshold_m
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n_d:
        if not is_still[i]:
            i += 1
            continue
        j = i
        while j < n_d and is_still[j]:
            j += 1
        # rows [i, j] are stationary (j-i+1 rows total, j-i edges all under threshold)
        run_rows = j - i + 1
        if run_rows >= min_run_length:
            runs.append((i, j))
        i = j + 1
    return runs


def apply_stop_snap_to_submission(
    df: pd.DataFrame,
    *,
    move_threshold_m: float = 0.5,
    min_run_length: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return new DataFrame with stationary-segment medians snapped per tripId."""
    out = df.copy()
    out = out.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)
    stats: dict[str, object] = {
        "rows_total": len(out),
        "rows_changed": 0,
        "trips": 0,
        "runs": 0,
        "move_threshold_m": move_threshold_m,
        "min_run_length": min_run_length,
    }
    original_lat = out["LatitudeDegrees"].to_numpy(dtype=np.float64).copy()
    original_lng = out["LongitudeDegrees"].to_numpy(dtype=np.float64).copy()
    for _, group in out.groupby("tripId", sort=False):
        idx = group.index.to_numpy()
        lat = group["LatitudeDegrees"].to_numpy(dtype=np.float64)
        lng = group["LongitudeDegrees"].to_numpy(dtype=np.float64)
        if len(lat) < 2:
            stats["trips"] = int(stats["trips"]) + 1  # type: ignore[arg-type]
            continue
        d = _haversine_m(lat[:-1], lng[:-1], lat[1:], lng[1:])
        runs = detect_stationary_runs(
            d, move_threshold_m=move_threshold_m, min_run_length=min_run_length,
        )
        stats["runs"] = int(stats["runs"]) + len(runs)  # type: ignore[arg-type]
        stats["trips"] = int(stats["trips"]) + 1  # type: ignore[arg-type]
        for lo, hi in runs:
            med_lat = float(np.median(lat[lo : hi + 1]))
            med_lng = float(np.median(lng[lo : hi + 1]))
            lat[lo : hi + 1] = med_lat
            lng[lo : hi + 1] = med_lng
        out.loc[idx, "LatitudeDegrees"] = lat
        out.loc[idx, "LongitudeDegrees"] = lng
    final_lat = out["LatitudeDegrees"].to_numpy(dtype=np.float64)
    final_lng = out["LongitudeDegrees"].to_numpy(dtype=np.float64)
    stats["rows_changed"] = int(
        np.sum(
            (np.abs(final_lat - original_lat) > 1e-12) | (np.abs(final_lng - original_lng) > 1e-12)
        )
    )
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply stationary-segment median snap to a GSDC2023 submission CSV.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--move-threshold-m",
        type=float,
        default=0.5,
        help="frame-to-frame haversine distance threshold (m) below which the row is considered stationary",
    )
    parser.add_argument(
        "--min-run-length",
        type=int,
        default=10,
        help="minimum number of consecutive stationary rows to qualify as a snap target",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        print(f"[error] input not found: {args.input}", file=sys.stderr)
        return 1
    df = pd.read_csv(args.input)
    required = {"tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"}
    missing = required - set(df.columns)
    if missing:
        print(f"[error] missing columns: {missing}", file=sys.stderr)
        return 2
    out, stats = apply_stop_snap_to_submission(
        df, move_threshold_m=args.move_threshold_m, min_run_length=args.min_run_length,
    )
    out.to_csv(args.output, index=False)
    print(
        f"trips={stats['trips']} rows_total={stats['rows_total']} "
        f"rows_changed={stats['rows_changed']} "
        f"({100 * int(stats['rows_changed']) / max(1, int(stats['rows_total'])):.2f}%) "
        f"runs={stats['runs']} "
        f"move_threshold_m={args.move_threshold_m} "
        f"min_run_length={args.min_run_length}",
        flush=True,
    )
    print(f"wrote: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
