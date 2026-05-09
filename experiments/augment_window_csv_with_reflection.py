#!/usr/bin/env python3
"""Merge BVH multipath per-window features into the §7.16 augmented
window CSV, producing a new CSV ready for retraining the nested stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_BASE_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_window_predictions.csv"
)
DEFAULT_REFL_CSV = RESULTS_DIR / "ppc_reflection_bvh_pooled_per_window.csv"
DEFAULT_OUT_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_with_reflection_window_predictions.csv"
)

REFL_FEATURES = [
    "reflection_count_mean",
    "reflection_count_max",
    "excess_delay_m_max_mean",
    "excess_delay_m_max_max",
    "excess_delay_m_p90_mean",
    "nlos_count_mean",
    "sat_count_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--reflection-csv", type=Path, default=DEFAULT_REFL_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_csv)
    refl = pd.read_csv(args.reflection_csv)
    print(f"base: {len(base)} windows × {len(base.columns)} cols")
    print(f"reflection: {len(refl)} rows; using cols: {[c for c in REFL_FEATURES if c in refl.columns]}")
    keep = ["city", "run", "window_index"] + [c for c in REFL_FEATURES if c in refl.columns]
    refl_sub = refl[keep].copy()
    # Rename refl features to make them obviously deployable in the stack
    rename = {c: f"refl_{c}" for c in REFL_FEATURES if c in refl.columns}
    refl_sub = refl_sub.rename(columns=rename)
    merged = base.merge(refl_sub, on=["city", "run", "window_index"], how="left")
    added = list(rename.values())
    nan_rows = merged[added].isna().any(axis=1).sum()
    print(f"merged: {len(merged)} rows; rows with NaN in reflection cols: {nan_rows}")
    merged[added] = merged[added].fillna(0.0)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"saved: {args.output_csv} ({len(merged)} rows × {len(merged.columns)} cols)")
    print(f"added columns: {added}")


if __name__ == "__main__":
    main()
