"""Post-process a GSDC2023 submission CSV with per-trip Hampel filter.

The baseline kaggle_wls trajectory (used for 98%+ of Selected output rows in
gate baseline-locked submissions) has isolated outlier spikes — most extreme
on dense-urban trips like lax-o where the 1-Hz position can jump >200 m in a
single epoch (physically impossible for a car).  A Hampel filter (median +
3*MAD outlier replacement) removes those spikes in trajectory space without
affecting the smooth majority.

Algorithm (per tripId)
----------------------

1. Sort by UnixTimeMillis.
2. For each row t, take the window of rows ``[t - W/2, t + W/2]``.
3. Compute per-axis (lat / lng) median + MAD on the window.
4. If ``|pos[t] - median| > k * 1.4826 * MAD``: replace pos[t] with median.
5. Else: keep pos[t].

Defaults ``W=21, k=2.5`` selected empirically on 4 train trips:
  aggregate metric (P50+P95)/2 reduction = -24 cm vs raw baseline,
  driven by lax-o P95 -1.81 m gain.

This is a *trajectory-space* post-process — no need to rebuild the pipeline.
The submission's ``LatitudeDegrees / LongitudeDegrees`` columns are filtered
in place.  Original input is preserved by writing to a new file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def hampel_filter_1d(values: np.ndarray, window: int, k: float, mad_floor: float) -> np.ndarray:
    """Single-axis Hampel filter.  Returns the filtered series."""
    n = len(values)
    out = values.copy()
    half = window // 2
    for t in range(n):
        lo = max(0, t - half)
        hi = min(n, t + half + 1)
        win = values[lo:hi]
        med = float(np.median(win))
        mad = float(np.median(np.abs(win - med)))
        sigma = max(1.4826 * mad, mad_floor)
        if abs(values[t] - med) > k * sigma:
            out[t] = med
    return out


def apply_hampel_to_submission(
    df: pd.DataFrame,
    *,
    window: int,
    k: float,
    mad_floor_deg: float,
    passes: int = 1,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return new DataFrame with Hampel-smoothed lat/lng per tripId.

    ``passes >= 2`` runs the Hampel filter iteratively. Each pass uses the
    *previous* pass output as input, allowing the filter to peel away
    consecutive outlier clusters that a single pass would otherwise treat as
    a local plateau (and miss).
    """
    if passes < 1:
        raise ValueError("passes must be >= 1")
    out = df.copy()
    out = out.sort_values(["tripId", "UnixTimeMillis"]).reset_index(drop=True)
    stats: dict[str, int] = {
        "rows_total": len(out),
        "rows_changed": 0,
        "trips": 0,
        "passes": passes,
    }
    per_pass_changed: list[int] = []
    original_lat = out["LatitudeDegrees"].to_numpy(dtype=np.float64).copy()
    original_lng = out["LongitudeDegrees"].to_numpy(dtype=np.float64).copy()
    for pass_idx in range(passes):
        pass_changed = 0
        for _, group in out.groupby("tripId", sort=False):
            idx = group.index.to_numpy()
            lat = group["LatitudeDegrees"].to_numpy(dtype=np.float64)
            lng = group["LongitudeDegrees"].to_numpy(dtype=np.float64)
            lat_filt = hampel_filter_1d(lat, window=window, k=k, mad_floor=mad_floor_deg)
            lng_filt = hampel_filter_1d(lng, window=window, k=k, mad_floor=mad_floor_deg)
            changed = (lat_filt != lat) | (lng_filt != lng)
            pass_changed += int(changed.sum())
            if pass_idx == 0:
                stats["trips"] += 1
            out.loc[idx, "LatitudeDegrees"] = lat_filt
            out.loc[idx, "LongitudeDegrees"] = lng_filt
        per_pass_changed.append(pass_changed)
    final_lat = out["LatitudeDegrees"].to_numpy(dtype=np.float64)
    final_lng = out["LongitudeDegrees"].to_numpy(dtype=np.float64)
    stats["rows_changed"] = int(
        np.sum((final_lat != original_lat) | (final_lng != original_lng))
    )
    stats["per_pass_changed"] = per_pass_changed  # type: ignore[assignment]
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply Hampel filter to a GSDC2023 submission CSV.",
    )
    parser.add_argument("--input", type=Path, required=True, help="input submission CSV")
    parser.add_argument("--output", type=Path, required=True, help="output (Hampel-smoothed) CSV")
    parser.add_argument("--window", type=int, default=21)
    parser.add_argument("--k", type=float, default=2.5)
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help=(
            "Number of Hampel passes (>=1). Iterative passes peel away "
            "consecutive outlier clusters that a single pass would miss."
        ),
    )
    parser.add_argument(
        "--mad-floor-deg",
        type=float,
        default=5e-7,
        help=(
            "Floor for the per-window sigma estimate, in degrees.  "
            "5e-7 deg ~= 5 cm at the equator; prevents over-aggressive "
            "replacement when MAD is very small (= stationary segments)."
        ),
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
    out, stats = apply_hampel_to_submission(
        df,
        window=args.window,
        k=args.k,
        mad_floor_deg=args.mad_floor_deg,
        passes=args.passes,
    )
    out.to_csv(args.output, index=False)
    print(
        f"trips={stats['trips']} rows_total={stats['rows_total']} "
        f"rows_changed={stats['rows_changed']} "
        f"({100 * stats['rows_changed'] / max(1, stats['rows_total']):.2f}%) "
        f"window={args.window} k={args.k} passes={args.passes} "
        f"per_pass_changed={stats.get('per_pass_changed', [])}",
        flush=True,
    )
    print(f"wrote: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
