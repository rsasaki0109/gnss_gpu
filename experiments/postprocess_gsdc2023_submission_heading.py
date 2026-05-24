"""Post-process a GSDC2023 submission CSV with heading-consistency smoothing.

Fourth layer in the trajectory post-process stack (after Hampel + accel +
stop-snap).  Targets a signal none of those see directly: the *direction*
of motion changes too fast for the vehicle to physically execute.

A car cannot change heading faster than ~30 °/s in sharp turns (and far less
at highway speed).  At 1 Hz sampling, ``|Δheading[t]| > heading_max_dps``
between two consecutive segments means at least one of the surrounding
positions is wrong.  The middle row is the cheapest assumption.

Algorithm (per tripId)
----------------------

1. Sort rows by UnixTimeMillis.
2. Compute per-edge bearing ``b[t] = atan2(dlng, dlat)`` between rows t and t+1
   (geodetic bearing approximation in flat-earth small patch; sufficient for
   < 100 m segments).  Skip near-zero displacement edges (no defined bearing).
3. Compute angular delta ``d[t] = wrap_pi(b[t] - b[t-1])`` between consecutive
   bearing measurements, in degrees / second.
4. Mark rows where ``|d[t]| > heading_max_dps`` as flagged.  Contract to local
   maxima of ``|d|`` (same trick as accel-smoother: isolates the source
   outlier, not the propagation neighbours).
5. Replace the flagged row's lat / lng by linear interpolation across the
   nearest non-flagged neighbours.

Default ``heading_max_dps = 30.0`` (≈ aggressive intersection turn).  No
iterations needed — single pass typically clears the residual outliers v6
leaves behind.
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


def _bearings(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """Per-edge bearing in radians (length n-1).  NaN for near-zero displacement."""
    n = len(lat)
    if n < 2:
        return np.zeros(0)
    lat0 = float(np.median(lat))
    mlat = 111_320.0
    mlng = 111_320.0 * np.cos(np.radians(lat0))
    east = (lng - lng[0]) * mlng
    north = (lat - lat[0]) * mlat
    de = np.diff(east)
    dn = np.diff(north)
    d2 = de * de + dn * dn
    b = np.arctan2(de, dn)
    b[d2 < 1e-6] = np.nan  # < 1 mm displacement -> bearing undefined
    return b


def smooth_heading_outliers(
    lat: np.ndarray, lng: np.ndarray, dt: np.ndarray, *, heading_max_dps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flag rows with physically impossible yaw rate; linearly interpolate.

    Returns (lat_out, lng_out, flagged_mask).
    """
    n = len(lat)
    out_lat = lat.copy()
    out_lng = lng.copy()
    flagged = np.zeros(n, dtype=bool)
    if n < 3:
        return out_lat, out_lng, flagged
    b = _bearings(lat, lng)
    # Per-row yaw rate at row t uses bearings on either side: edge (t-1, t)
    # and edge (t, t+1).  So row index t in [1, n-2].
    abs_yaw = np.zeros(n, dtype=np.float64)
    for t in range(1, n - 1):
        b_prev = b[t - 1]
        b_next = b[t]
        if not (np.isfinite(b_prev) and np.isfinite(b_next)):
            continue
        delta = b_next - b_prev
        # Wrap to (-pi, pi]
        while delta > np.pi:
            delta -= 2 * np.pi
        while delta < -np.pi:
            delta += 2 * np.pi
        # Average dt across the two edges (≈ 1 s at 1 Hz)
        dt_avg = 0.5 * (float(dt[t - 1]) + float(dt[t]))
        if dt_avg <= 0:
            continue
        abs_yaw[t] = abs(np.degrees(delta) / dt_avg)
    raw_flag = abs_yaw > heading_max_dps
    # Contract to local maxima of |yaw rate| to isolate the source outlier.
    for t in range(1, n - 1):
        if not raw_flag[t]:
            continue
        if abs_yaw[t] >= abs_yaw[t - 1] and abs_yaw[t] >= abs_yaw[t + 1]:
            flagged[t] = True
    if not flagged.any():
        return out_lat, out_lng, flagged
    # Linear interpolation across the bad row(s) using non-flagged neighbours.
    for t in np.where(flagged)[0]:
        lo = t - 1
        while lo >= 0 and flagged[lo]:
            lo -= 1
        hi = t + 1
        while hi < n and flagged[hi]:
            hi += 1
        if lo < 0 or hi >= n:
            continue
        t_lo = 0.0
        t_t = float(np.sum(dt[lo:t]))
        t_hi = float(np.sum(dt[lo:hi]))
        if t_hi <= 0:
            continue
        alpha = (t_t - t_lo) / (t_hi - t_lo)
        out_lat[t] = (1.0 - alpha) * lat[lo] + alpha * lat[hi]
        out_lng[t] = (1.0 - alpha) * lng[lo] + alpha * lng[hi]
    return out_lat, out_lng, flagged


def apply_heading_smoothing_to_submission(
    df: pd.DataFrame,
    *,
    heading_max_dps: float = 30.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return new DataFrame with heading-outlier rows replaced per tripId."""
    out = df.copy()
    out = out.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)
    stats: dict[str, object] = {
        "rows_total": len(out),
        "rows_changed": 0,
        "trips": 0,
        "heading_max_dps": heading_max_dps,
    }
    original_lat = out["LatitudeDegrees"].to_numpy(dtype=np.float64).copy()
    original_lng = out["LongitudeDegrees"].to_numpy(dtype=np.float64).copy()
    for _, group in out.groupby("tripId", sort=False):
        idx = group.index.to_numpy()
        lat = group["LatitudeDegrees"].to_numpy(dtype=np.float64)
        lng = group["LongitudeDegrees"].to_numpy(dtype=np.float64)
        t_ms = group["UnixTimeMillis"].to_numpy(dtype=np.float64)
        if len(idx) < 3:
            stats["trips"] = int(stats["trips"]) + 1  # type: ignore[arg-type]
            continue
        dt_s = np.diff(t_ms) / 1000.0
        lat_s, lng_s, _ = smooth_heading_outliers(
            lat, lng, dt_s, heading_max_dps=heading_max_dps,
        )
        out.loc[idx, "LatitudeDegrees"] = lat_s
        out.loc[idx, "LongitudeDegrees"] = lng_s
        stats["trips"] = int(stats["trips"]) + 1  # type: ignore[arg-type]
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
        description="Apply heading-consistency smoothing to a GSDC2023 submission CSV.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--heading-max-dps",
        type=float,
        default=30.0,
        help="degrees/second maximum yaw rate; rows above are outliers (default: 30)",
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
    out, stats = apply_heading_smoothing_to_submission(
        df, heading_max_dps=args.heading_max_dps,
    )
    out.to_csv(args.output, index=False)
    print(
        f"trips={stats['trips']} rows_total={stats['rows_total']} "
        f"rows_changed={stats['rows_changed']} "
        f"({100 * int(stats['rows_changed']) / max(1, int(stats['rows_total'])):.2f}%) "
        f"heading_max_dps={args.heading_max_dps}",
        flush=True,
    )
    print(f"wrote: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
