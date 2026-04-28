#!/usr/bin/env python3
"""Merge UTD edge candidate features into the §7.16 augmented window CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_BASE_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_window_predictions.csv"
)
DEFAULT_UTD_CSV = RESULTS_DIR / "ppc_utd_edges_pooled_per_window.csv"
DEFAULT_OUT_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_with_utd_window_predictions.csv"
)
UTD_FEATURES = [
    "utd_candidate_sat_count_mean",
    "utd_candidate_sat_count_max",
    "utd_candidate_nlos_sat_count_mean",
    "utd_candidate_nlos_sat_count_max",
    "utd_candidate_count_total_mean",
    "utd_candidate_count_total_max",
    "utd_candidate_count_nlos_mean",
    "utd_candidate_count_nlos_max",
    "utd_min_excess_path_m_mean",
    "utd_min_excess_path_m_min",
    "utd_min_edge_distance_m_mean",
    "utd_min_edge_distance_m_min",
    "utd_min_fresnel_v_mean",
    "utd_min_fresnel_v_min",
    "utd_score_sum_mean",
    "utd_score_sum_max",
    "utd_score_nlos_sum_mean",
    "utd_score_nlos_sum_max",
    "utd_edge_count_total",
    "utd_edge_boundary_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--utd-csv", type=Path, default=DEFAULT_UTD_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_csv)
    utd = pd.read_csv(args.utd_csv)
    keep = ["city", "run", "window_index"] + [c for c in UTD_FEATURES if c in utd.columns]
    utd_sub = utd[keep].copy()
    rename = {c: f"utd_edge_{c[4:]}" for c in UTD_FEATURES if c in utd.columns}
    utd_sub = utd_sub.rename(columns=rename)
    merged = base.merge(utd_sub, on=["city", "run", "window_index"], how="left")
    added = list(rename.values())
    nan_rows = int(merged[added].isna().any(axis=1).sum()) if added else 0
    print(f"base: {len(base)} rows x {len(base.columns)} cols")
    print(f"utd: {len(utd)} rows; added columns: {added}")
    print(f"merged: {len(merged)} rows; rows with NaN in UTD cols: {nan_rows}")
    if added:
        merged[added] = merged[added].fillna(0.0)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"saved: {args.output_csv} ({len(merged)} rows x {len(merged.columns)} cols)")

    if "actual_fix_rate_pct" in merged.columns:
        print("\nCorrelations vs actual_fix_rate_pct:")
        for col in added:
            print(f"  {col}: {merged[col].corr(merged['actual_fix_rate_pct']):+.3f}")


if __name__ == "__main__":
    main()
