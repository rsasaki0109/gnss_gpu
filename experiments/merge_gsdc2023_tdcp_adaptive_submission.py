#!/usr/bin/env python3
"""Merge aggressive and conservative TDCP_on_v8 submissions by trip metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


_A32_PHONES = {"samsunga325g", "samsunga32", "sm-a325f"}
_EARTH_RADIUS_M = 6_371_000.0


def _phone(trip_id: str) -> str:
    return str(trip_id).split("/")[-1].lower()


def _is_lax_pixel5(trip_id: str) -> bool:
    text = str(trip_id).lower()
    return "us-ca-lax-" in text and _phone(text) == "pixel5"


def _is_february_lax_pixel5(trip_id: str) -> bool:
    text = str(trip_id).lower()
    return _is_lax_pixel5(text) and text.startswith("2022-02-")


def _use_conservative(trip_id: str, lax_pixel5_mode: str) -> bool:
    phone = _phone(trip_id)
    if phone in _A32_PHONES:
        return True
    if lax_pixel5_mode == "all":
        return _is_lax_pixel5(trip_id)
    if lax_pixel5_mode == "february":
        return _is_february_lax_pixel5(trip_id)
    if lax_pixel5_mode == "none":
        return False
    raise ValueError(f"unknown LAX/pixel5 mode: {lax_pixel5_mode}")


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_rad = np.deg2rad(np.asarray(lat1, dtype=np.float64))
    lon1_rad = np.deg2rad(np.asarray(lon1, dtype=np.float64))
    lat2_rad = np.deg2rad(np.asarray(lat2, dtype=np.float64))
    lon2_rad = np.deg2rad(np.asarray(lon2, dtype=np.float64))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _row_gate_threshold_m(trip_id: str, *, lax_pixel5_threshold_m: float, a32_threshold_m: float) -> float:
    phone = _phone(trip_id)
    if phone in _A32_PHONES:
        return float(a32_threshold_m)
    if _is_lax_pixel5(trip_id):
        return float(lax_pixel5_threshold_m)
    return -1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggressive", type=Path, required=True)
    parser.add_argument("--conservative", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, default=None)
    parser.add_argument("--lax-pixel5-mode", choices=("all", "february", "none"), default="all")
    parser.add_argument(
        "--lax-pixel5-row-gate-m",
        type=float,
        default=-1.0,
        help="For conservative LAX/pixel5 trips, keep aggressive rows whose aggressive-vs-conservative displacement is at most this many metres. Negative disables row gating.",
    )
    parser.add_argument(
        "--a32-row-gate-m",
        type=float,
        default=-1.0,
        help="For conservative A32-family trips, keep aggressive rows whose aggressive-vs-conservative displacement is at most this many metres. Negative disables row gating.",
    )
    args = parser.parse_args()

    aggressive = pd.read_csv(args.aggressive)
    conservative = pd.read_csv(args.conservative)
    if list(aggressive.columns) != list(conservative.columns):
        raise ValueError("input columns differ")
    key_cols = ["tripId", "UnixTimeMillis"]
    if not aggressive[key_cols].equals(conservative[key_cols]):
        raise ValueError("input row order or keys differ")

    out = aggressive.copy()
    latlon_cols = ["LatitudeDegrees", "LongitudeDegrees"]
    use_cons = out["tripId"].map(lambda trip_id: _use_conservative(trip_id, args.lax_pixel5_mode)).to_numpy(dtype=bool)
    source = np.where(use_cons, "conservative", "aggressive").astype(object)
    if np.any(use_cons):
        out.loc[use_cons, latlon_cols] = conservative.loc[use_cons, latlon_cols].to_numpy()

    thresholds = out["tripId"].map(
        lambda trip_id: _row_gate_threshold_m(
            trip_id,
            lax_pixel5_threshold_m=args.lax_pixel5_row_gate_m,
            a32_threshold_m=args.a32_row_gate_m,
        ),
    ).to_numpy(dtype=np.float64)
    row_gate = use_cons & (thresholds >= 0.0)
    if np.any(row_gate):
        displacement_m = _haversine_m(
            aggressive.loc[row_gate, "LatitudeDegrees"].to_numpy(),
            aggressive.loc[row_gate, "LongitudeDegrees"].to_numpy(),
            conservative.loc[row_gate, "LatitudeDegrees"].to_numpy(),
            conservative.loc[row_gate, "LongitudeDegrees"].to_numpy(),
        )
        keep_aggressive = displacement_m <= thresholds[row_gate]
        row_gate_idx = np.flatnonzero(row_gate)
        aggressive_idx = row_gate_idx[keep_aggressive]
        if aggressive_idx.size:
            out.iloc[aggressive_idx, [out.columns.get_loc(col) for col in latlon_cols]] = aggressive.iloc[
                aggressive_idx
            ][latlon_cols].to_numpy()
            source[aggressive_idx] = "aggressive_row_gate"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    stats = (
        pd.DataFrame(
            {
                "tripId": out["tripId"],
                "source": source,
            },
        )
        .groupby(["tripId", "source"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    stats_path = args.stats_output or args.output.with_name(args.output.stem + "_adaptive_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"rows={len(out)} conservative_rows={int(use_cons.sum())} aggressive_rows={int((~use_cons).sum())}")
    print(f"row_gate_rows={int(np.count_nonzero(source == 'aggressive_row_gate'))}")
    print(f"conservative_trips={stats.loc[stats['source'] == 'conservative', 'tripId'].nunique()}")
    print(f"wrote: {args.output}")
    print(f"wrote: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
