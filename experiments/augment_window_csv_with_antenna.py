#!/usr/bin/env python3
"""Aggregate per-run antenna feature CSVs into a single per-window CSV
and merge into the §7.16 augmented window CSV for retraining."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_BASE_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_window_predictions.csv"
)
DEFAULT_OUT_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_current_tight_hold_carry_with_antenna_window_predictions.csv"
)
ANTENNA_FEATURES = [
    "eff_db_p10_mean", "eff_db_p10_min",
    "eff_db_p50_mean", "eff_db_p90_mean",
    "eff_db_max_mean", "eff_db_max_max",
    "eff_db_mean_mean",
    "usable_count_mean", "usable_count_min",
    "marginal_count_mean",
    "nlos_at_high_elev_count_mean", "nlos_at_high_elev_count_max",
    "gain_db_mean_mean",
    "elev_deg_p50_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="ppc_antenna_features_s5")
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--pooled-csv", type=Path,
                        default=RESULTS_DIR / "ppc_antenna_features_pooled_per_window.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern = re.compile(rf"^{re.escape(args.prefix)}_(?P<city>nagoya|tokyo)_(?P<run>run\d+)_per_window\.csv$")
    rows: list[pd.DataFrame] = []
    for path in sorted(RESULTS_DIR.glob(f"{args.prefix}_*_per_window.csv")):
        m = pattern.match(path.name)
        if not m:
            continue
        df = pd.read_csv(path)
        if "city" not in df.columns:
            df.insert(0, "city", m.group("city"))
        if "run" not in df.columns:
            df.insert(1, "run", m.group("run"))
        rows.append(df)
        print(f"loaded {path.name}: {len(df)} windows")
    if not rows:
        raise SystemExit("no per-window CSVs matched")
    pool = pd.concat(rows, ignore_index=True)
    pool.to_csv(args.pooled_csv, index=False)
    print(f"\nsaved pooled: {args.pooled_csv} ({len(pool)} rows)")

    base = pd.read_csv(args.base_csv)
    keep = ["city", "run", "window_index"] + [c for c in ANTENNA_FEATURES if c in pool.columns]
    pool_sub = pool[keep].copy()
    rename = {c: f"ant_{c}" for c in ANTENNA_FEATURES if c in pool.columns}
    pool_sub = pool_sub.rename(columns=rename)
    merged = base.merge(pool_sub, on=["city", "run", "window_index"], how="left")
    added = list(rename.values())
    nan_rows = int(merged[added].isna().any(axis=1).sum())
    print(f"merged: {len(merged)} rows; rows with NaN in antenna cols: {nan_rows}")
    merged[added] = merged[added].fillna(0.0)
    merged.to_csv(args.output_csv, index=False)
    print(f"saved augmented: {args.output_csv} ({len(merged)} rows × {len(merged.columns)} cols)")
    print(f"added columns: {added}")

    # Also print correlations vs the §7.16 prediction targets if columns exist
    if "actual_fix_rate_pct" in merged.columns:
        print("\nCorrelations vs actual_fix_rate_pct (n=197):")
        for c in added:
            r = merged[c].corr(merged["actual_fix_rate_pct"])
            print(f"  {c}: {r:+.3f}")


if __name__ == "__main__":
    main()
