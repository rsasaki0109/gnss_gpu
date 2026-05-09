#!/usr/bin/env python3
"""Aggregate per-run UTD edge candidate features and print correlations."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
ADOPTED_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45_window_predictions.csv"
)
PRODUCT_WINDOW_CSV = (
    Path(__file__).resolve().parents[1]
    / "internal_docs"
    / "product_deliverable"
    / "window_level_details.csv"
)
UTD_FEATURE_PREFIXES = ("utd_",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="ppc_utd_edges_s60")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "ppc_utd_edges_pooled_per_window.csv")
    parser.add_argument("--adopted-csv", type=Path, default=ADOPTED_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern = re.compile(rf"^{re.escape(args.prefix)}_(?P<city>nagoya|tokyo)_(?P<run>run\d+)_per_window\.csv$")
    rows: list[pd.DataFrame] = []
    for path in sorted(RESULTS_DIR.glob(f"{args.prefix}_*_per_window.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        df = pd.read_csv(path)
        if "city" not in df.columns:
            df.insert(0, "city", match.group("city"))
        if "run" not in df.columns:
            df.insert(1, "run", match.group("run"))
        rows.append(df)
        print(f"loaded {path.name}: {len(df)} windows")
    if not rows:
        raise SystemExit("no UTD per-window CSVs matched the prefix")

    pool = pd.concat(rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(args.output, index=False)
    print(f"\nsaved: {args.output} ({len(pool)} windows total)")

    if args.adopted_csv.exists():
        adopted = pd.read_csv(args.adopted_csv)
        keep = [
            "city",
            "run",
            "window_index",
            "actual_fix_rate_pct",
            "base_pred_fix_rate_pct",
            "corrected_pred_fix_rate_pct",
        ]
        adopted = adopted[keep].copy()
    elif PRODUCT_WINDOW_CSV.exists():
        print(f"\nadopted CSV missing; falling back to {PRODUCT_WINDOW_CSV}")
        adopted = pd.read_csv(PRODUCT_WINDOW_CSV)
        adopted = adopted.rename(columns={"adopted_pred_fix_rate_pct": "corrected_pred_fix_rate_pct"})
        keep = [
            "city",
            "run",
            "window_index",
            "actual_fix_rate_pct",
            "base_pred_fix_rate_pct",
            "corrected_pred_fix_rate_pct",
        ]
        adopted = adopted[keep].copy()
    else:
        print(f"\nadopted CSV missing: {args.adopted_csv}")
        print("wrote pooled CSV only; skipping correlation analysis")
        return
    adopted["error_pp"] = adopted["corrected_pred_fix_rate_pct"] - adopted["actual_fix_rate_pct"]
    merged = pool.merge(adopted, on=["city", "run", "window_index"], how="inner")
    print(f"\nmerged: {len(merged)} (UTD pool {len(pool)} ∩ adopted {len(adopted)})")

    feature_cols = [
        c
        for c in pool.columns
        if c not in {"city", "run", "window_index", "epoch_count"}
        and c.startswith(UTD_FEATURE_PREFIXES)
    ]
    print(f"\nCorrelations of {len(feature_cols)} UTD features:")
    print(f"{'feature':<48s} {'vs actual':>10s} {'vs §7.16 err':>14s}")
    for col in feature_cols:
        r_actual = merged[col].corr(merged["actual_fix_rate_pct"])
        r_err = merged[col].corr(merged["error_pp"])
        print(f"  {col:<46s} {r_actual:+.3f}      {r_err:+.3f}")

    print("\nFocus windows (Tokyo run2):")
    focus = merged[
        (merged.city == "tokyo")
        & (merged.run == "run2")
        & merged.window_index.isin([7, 9, 23, 24, 25, 26, 27])
    ]
    cols = [
        "window_index",
        "actual_fix_rate_pct",
        "corrected_pred_fix_rate_pct",
        "utd_candidate_nlos_sat_count_mean",
        "utd_candidate_count_nlos_mean",
        "utd_score_nlos_sum_mean",
        "utd_min_excess_path_m_min",
    ]
    cols = [c for c in cols if c in focus.columns]
    print(focus[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
