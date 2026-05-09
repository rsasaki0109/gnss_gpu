#!/usr/bin/env python3
"""Aggregate per-run BVH multipath outputs into a single per-window CSV
and analyse correlation with demo5 actual FIX rate / §7.16 prediction
error.

Input: `ppc_reflection_bvh_s<STRIDE>_<city>_<run>_per_window.csv` files
    produced by `exp_ppc_reflection_poc.py`.

Output:
- `ppc_reflection_bvh_pooled_per_window.csv` (all 6 runs concatenated)
- prints correlations with demo5 actual FIX and §7.16 error
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
ADOPTED_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45_window_predictions.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="ppc_reflection_bvh_s60")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ppc_reflection_bvh_pooled_per_window.csv")
    parser.add_argument("--adopted-csv", type=Path, default=ADOPTED_CSV)
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
        raise SystemExit("no per-window CSVs matched the prefix")
    pool = pd.concat(rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(args.output, index=False)
    print(f"\nsaved: {args.output} ({len(pool)} windows total)")

    # Correlate with §7.16
    adopted = pd.read_csv(args.adopted_csv)
    keep = ["city", "run", "window_index", "actual_fix_rate_pct", "base_pred_fix_rate_pct", "corrected_pred_fix_rate_pct"]
    adopted = adopted[keep].copy()
    adopted["error_pp"] = adopted["corrected_pred_fix_rate_pct"] - adopted["actual_fix_rate_pct"]
    merged = pool.merge(adopted, on=["city", "run", "window_index"], how="inner")
    print(f"\nmerged: {len(merged)} (refl pool {len(pool)} ∩ adopted {len(adopted)})")

    refl_features = [c for c in pool.columns if c not in {"city", "run", "window_index", "epoch_count"}]
    print(f"\nCorrelations of {len(refl_features)} reflection features:")
    print(f"{'feature':<40s} {'vs actual':>10s} {'vs §7.16 err':>14s}")
    for c in refl_features:
        r_actual = merged[c].corr(merged["actual_fix_rate_pct"])
        r_err = merged[c].corr(merged["error_pp"])
        print(f"  {c:<38s} {r_actual:+.3f}      {r_err:+.3f}")

    print("\nFocus windows (Tokyo run2):")
    focus = merged[(merged.city == "tokyo") & (merged.run == "run2") & merged.window_index.isin([7, 9, 23, 24, 25, 26, 27])]
    cols = ["window_index", "actual_fix_rate_pct", "corrected_pred_fix_rate_pct", "reflection_count_mean", "excess_delay_m_max_mean", "nlos_count_mean"]
    cols = [c for c in cols if c in focus.columns]
    print(focus[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
