"""Post-process a GSDC2023 submission CSV with motion-acceleration smoothing.

Complements :mod:`postprocess_gsdc2023_submission_hampel`.  Hampel attacks
*position-space* outliers (median + MAD on lat / lng).  This module attacks
*acceleration-space* outliers: a car cannot physically exceed ~5 m/s²
laterally on a paved road, so any epoch whose finite-difference acceleration
exceeds ``accel_max`` is replaced by the average of its two neighbours.

Algorithm (per tripId, per axis: lat / lng)
-------------------------------------------

1. Sort rows by UnixTimeMillis.
2. Convert lat / lng to local east/north metres around the trip median (small
   patch so a degree of latitude ≈ 111 km and longitude scales with cos(lat0)).
3. Finite-difference twice: velocity[t] = (x[t+1] - x[t-1]) / (2 dt),
   accel[t] = (x[t+1] - 2 x[t] + x[t-1]) / dt².
4. If ``|accel[t]| > accel_max`` on *either* axis: replace x[t] with the mean
   of x[t-1] and x[t+1] (linear interpolation across the bad epoch).
5. Iterate ``passes`` times.  Default ``accel_max = 5.0 m/s², passes = 2``.

Boundary rows (first / last) are left untouched (no central difference).

This is intentionally a *lighter* filter than Hampel: it only fires when the
trajectory is locally inconsistent with vehicle motion, leaving smooth motion
(including high-speed straight-line driving) completely intact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _local_metres(lat: np.ndarray, lng: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    lat0 = float(np.median(lat))
    lng0 = float(np.median(lng))
    mlat = 111_320.0  # metres per degree latitude
    mlng = 111_320.0 * np.cos(np.radians(lat0))
    east = (lng - lng0) * mlng
    north = (lat - lat0) * mlat
    return east, north, lat0, lng0


def _back_to_deg(east: np.ndarray, north: np.ndarray, lat0: float, lng0: float) -> tuple[np.ndarray, np.ndarray]:
    mlat = 111_320.0
    mlng = 111_320.0 * np.cos(np.radians(lat0))
    lat = north / mlat + lat0
    lng = east / mlng + lng0
    return lat, lng


def smooth_axis_by_accel(
    x: np.ndarray, dt: np.ndarray, *, accel_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One pass of accel-outlier replacement on a single axis (metres).

    Two-stage algorithm (avoids contamination spread):

    1. Compute |accel| from the *original* signal at every interior epoch and
       mark rows with ``|accel| > accel_max`` as flagged.  An isolated impulse
       at epoch t flags rows t-1, t, t+1 in the raw mask, so we then
       *contract* the flag set: keep only rows where the flag is a local
       maximum of |accel| (the impulse centre).  This isolates the source
       outlier and leaves the innocent neighbours alone.
    2. Replace each surviving flagged row by linear interpolation from the
       nearest *non-flagged* rows on either side (preserves the smooth motion
       envelope around the outlier).

    Boundary rows (first / last) are left untouched.
    """
    n = len(x)
    out = x.copy()
    flagged = np.zeros(n, dtype=bool)
    if n < 3:
        return out, flagged
    abs_accel = np.zeros(n, dtype=np.float64)
    for t in range(1, n - 1):
        dt_prev = float(dt[t - 1])
        dt_next = float(dt[t])
        if dt_prev <= 0 or dt_next <= 0:
            continue
        v_prev = (x[t] - x[t - 1]) / dt_prev
        v_next = (x[t + 1] - x[t]) / dt_next
        abs_accel[t] = abs((v_next - v_prev) / (0.5 * (dt_prev + dt_next)))
    raw_flag = abs_accel > accel_max
    # Contract to local maxima of |accel|: keep t only if abs_accel[t] is
    # >= both neighbours.  Removes the "propagation flags" at t-1, t+1.
    for t in range(1, n - 1):
        if not raw_flag[t]:
            continue
        if abs_accel[t] >= abs_accel[t - 1] and abs_accel[t] >= abs_accel[t + 1]:
            flagged[t] = True
    if not flagged.any():
        return out, flagged
    # Step 2: linear interpolation from nearest non-flagged neighbours.
    for t in np.where(flagged)[0]:
        lo = t - 1
        while lo >= 0 and flagged[lo]:
            lo -= 1
        hi = t + 1
        while hi < n and flagged[hi]:
            hi += 1
        if lo < 0 or hi >= n:
            continue  # cannot interpolate at boundary cluster
        # Linear interpolation using time stamps embedded in cumulative dt.
        # Build cumulative time index relative to lo.
        t_lo = 0.0
        t_t = float(np.sum(dt[lo:t]))
        t_hi = float(np.sum(dt[lo:hi]))
        if t_hi <= 0:
            continue
        alpha = (t_t - t_lo) / (t_hi - t_lo)
        out[t] = (1.0 - alpha) * x[lo] + alpha * x[hi]
    return out, flagged


def apply_accel_smoothing_to_submission(
    df: pd.DataFrame,
    *,
    accel_max: float = 5.0,
    passes: int = 2,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return new DataFrame with accel-outlier-smoothed lat/lng per tripId."""
    if passes < 1:
        raise ValueError("passes must be >= 1")
    out = df.copy()
    out = out.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)
    stats: dict[str, object] = {
        "rows_total": len(out),
        "rows_changed": 0,
        "trips": 0,
        "passes": passes,
        "accel_max": accel_max,
    }
    per_pass_changed: list[int] = []
    original_lat = out["LatitudeDegrees"].to_numpy(dtype=np.float64).copy()
    original_lng = out["LongitudeDegrees"].to_numpy(dtype=np.float64).copy()
    for pass_idx in range(passes):
        pass_changed = 0
        for _, group in out.groupby("tripId", sort=False):
            idx = group.index.to_numpy()
            if len(idx) < 3:
                continue
            lat = group["LatitudeDegrees"].to_numpy(dtype=np.float64)
            lng = group["LongitudeDegrees"].to_numpy(dtype=np.float64)
            t_ms = group["UnixTimeMillis"].to_numpy(dtype=np.float64)
            dt_s = np.diff(t_ms) / 1000.0
            east, north, lat0, lng0 = _local_metres(lat, lng)
            east_s, rep_e = smooth_axis_by_accel(east, dt_s, accel_max=accel_max)
            north_s, rep_n = smooth_axis_by_accel(north, dt_s, accel_max=accel_max)
            replaced_any = rep_e | rep_n
            pass_changed += int(replaced_any.sum())
            lat_s, lng_s = _back_to_deg(east_s, north_s, lat0, lng0)
            if pass_idx == 0:
                stats["trips"] = int(stats["trips"]) + 1  # type: ignore[arg-type]
            out.loc[idx, "LatitudeDegrees"] = lat_s
            out.loc[idx, "LongitudeDegrees"] = lng_s
        per_pass_changed.append(pass_changed)
    final_lat = out["LatitudeDegrees"].to_numpy(dtype=np.float64)
    final_lng = out["LongitudeDegrees"].to_numpy(dtype=np.float64)
    stats["rows_changed"] = int(
        np.sum(
            (np.abs(final_lat - original_lat) > 1e-12) | (np.abs(final_lng - original_lng) > 1e-12)
        )
    )
    stats["per_pass_changed"] = per_pass_changed
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply acceleration-outlier smoothing to a GSDC2023 submission CSV.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accel-max",
        type=float,
        default=5.0,
        help="m/s² threshold above which the epoch is treated as outlier (default: 5.0)",
    )
    parser.add_argument("--passes", type=int, default=2)
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
    out, stats = apply_accel_smoothing_to_submission(
        df, accel_max=args.accel_max, passes=args.passes,
    )
    out.to_csv(args.output, index=False)
    print(
        f"trips={stats['trips']} rows_total={stats['rows_total']} "
        f"rows_changed={stats['rows_changed']} "
        f"({100 * int(stats['rows_changed']) / max(1, int(stats['rows_total'])):.2f}%) "
        f"accel_max={args.accel_max} passes={args.passes} "
        f"per_pass_changed={stats['per_pass_changed']}",
        flush=True,
    )
    print(f"wrote: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
