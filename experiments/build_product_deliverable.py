#!/usr/bin/env python3
"""Build the product deliverable package for the adopted strict-best model.

Reads the §7.16 adopted `transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45`
window predictions and emits:

1. `internal_docs/product_deliverable/route_level_fix_rate_prediction.csv`
   - One row per (city, run), epoch-weighted aggregate predicted FIX rate,
     actual FIX rate, aggregate error, and qualitative confidence tier.
2. `internal_docs/product_deliverable/window_level_details.csv`
   - Per-window predictions with flags for known failure cases.

Confidence tiers derive from the §7.16 LORO metrics and the presence of
known-failure windows:

- high: no focus-case windows, expected error <= 3 pp
- medium: contains hidden-high or carry-boundary windows, expected 3-8 pp
- low: contains false-high or false-lift windows, wide uncertainty
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DELIVERABLE_DIR = Path(__file__).resolve().parent.parent / "internal_docs" / "product_deliverable"
DEFAULT_PRED_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45_window_predictions.csv"
)

# Known failure cases from plan.md §7.11 / §7.13 / §7.16 diagnostics.
FOCUS_CASES = {
    ("tokyo", "run2", 7): ("false_high", "reject partially absorbed; actual 0%, pred 39.5%"),
    ("tokyo", "run2", 9): ("false_high", "reject partially absorbed; actual 0%, pred 71%"),
    ("tokyo", "run2", 23): ("hidden_high", "under-lifted; actual 100%, pred 32%"),
    ("tokyo", "run2", 24): ("hidden_high", "under-lifted; actual 100%, pred 20%"),
    ("tokyo", "run2", 25): ("hidden_high", "better under §7.16; actual 100%, pred 48%"),
    ("tokyo", "run2", 26): ("hidden_high", "better under §7.16; actual 97%, pred 50%"),
    ("tokyo", "run2", 27): ("hidden_high", "better under §7.16; actual 75%, pred 39%"),
    ("tokyo", "run3", 17): ("false_lift", "over-lifted; actual 0%, pred 19%"),
    ("tokyo", "run3", 16): ("false_lift", "mild over-lift; actual 1%, pred 10%"),
    ("nagoya", "run2", 17): ("false_lift_mild", "deployable readiness fires; actual 0%, pred 17%"),
    ("nagoya", "run2", 27): ("false_lift_resolved", "successfully rejected; actual 0%, pred 9%"),
}


def _confidence_tier(city: str, run: str, window_indices: list[int]) -> tuple[str, str]:
    """Pick a qualitative confidence tier for a (city, run) based on focus cases present."""
    tags = []
    for wi in window_indices:
        key = (city, run, wi)
        if key in FOCUS_CASES:
            tags.append(FOCUS_CASES[key][0])
    if any(t == "false_high" for t in tags):
        return "low", "run contains at least one false-high window (pred high, actual 0%); individual window trust low"
    if any(t == "false_lift" for t in tags):
        return "low", "run contains at least one false-lift window (pred lifted, actual 0%); individual window trust low"
    if any(t.startswith("hidden_high") for t in tags):
        return "medium", "run contains hidden-high windows (actual 100%, pred under-lifted)"
    if any(t.startswith("false_lift_mild") for t in tags):
        return "medium", "mild false-lift risk"
    return "high", "no focus-case windows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build product deliverable CSVs")
    parser.add_argument("--prediction-csv", type=Path, default=DEFAULT_PRED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DELIVERABLE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.prediction_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # window-level details
    window_rows: list[dict[str, object]] = []
    for _, row in df.sort_values(["city", "run", "window_index"]).iterrows():
        key = (row["city"], row["run"], int(row["window_index"]))
        focus_tag, focus_note = FOCUS_CASES.get(key, ("", ""))
        window_rows.append(
            {
                "city": row["city"],
                "run": row["run"],
                "window_index": int(row["window_index"]),
                "actual_fix_rate_pct": float(row["actual_fix_rate_pct"]),
                "base_pred_fix_rate_pct": float(row["base_pred_fix_rate_pct"]),
                "adopted_pred_fix_rate_pct": float(row["corrected_pred_fix_rate_pct"]),
                "abs_error_pp": float(abs(row["corrected_pred_fix_rate_pct"] - row["actual_fix_rate_pct"])),
                "focus_case_tag": focus_tag,
                "focus_case_note": focus_note,
            }
        )
    window_df = pd.DataFrame(window_rows)
    window_path = args.output_dir / "window_level_details.csv"
    window_df.to_csv(window_path, index=False)
    print(f"saved: {window_path} ({len(window_df)} rows)")

    # route-level aggregate
    route_rows: list[dict[str, object]] = []
    for (city, run), group in df.groupby(["city", "run"], sort=True):
        weights = group["sim_matched_epochs"].to_numpy(dtype=np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
        actual = float(np.average(group["actual_fix_rate_pct"].to_numpy(dtype=np.float64), weights=weights))
        pred = float(np.average(group["corrected_pred_fix_rate_pct"].to_numpy(dtype=np.float64), weights=weights))
        base = float(np.average(group["base_pred_fix_rate_pct"].to_numpy(dtype=np.float64), weights=weights))
        window_indices = sorted(int(v) for v in group["window_index"].tolist())
        tier, note = _confidence_tier(str(city), str(run), window_indices)
        n_focus = sum(1 for wi in window_indices if (str(city), str(run), wi) in FOCUS_CASES)
        route_rows.append(
            {
                "city": city,
                "run": run,
                "window_count": len(group),
                "focus_case_window_count": n_focus,
                "actual_fix_rate_pct": round(actual, 3),
                "baseline_pred_fix_rate_pct": round(base, 3),
                "adopted_pred_fix_rate_pct": round(pred, 3),
                "adopted_abs_error_pp": round(abs(pred - actual), 3),
                "adopted_signed_error_pp": round(pred - actual, 3),
                "confidence_tier": tier,
                "confidence_note": note,
            }
        )
    route_df = pd.DataFrame(route_rows)
    route_path = args.output_dir / "route_level_fix_rate_prediction.csv"
    route_df.to_csv(route_path, index=False)
    print(f"saved: {route_path} ({len(route_df)} rows)")

    # summary stats
    print("\nRoute-level summary (adopted §7.16):")
    print(route_df[["city", "run", "window_count", "actual_fix_rate_pct", "adopted_pred_fix_rate_pct", "adopted_abs_error_pp", "confidence_tier"]].to_string(index=False))

    overall_actual = float(np.average(window_df["actual_fix_rate_pct"], weights=df["sim_matched_epochs"].fillna(1.0).clip(lower=1.0)))
    overall_pred = float(np.average(window_df["adopted_pred_fix_rate_pct"], weights=df["sim_matched_epochs"].fillna(1.0).clip(lower=1.0)))
    print(f"\nOverall actual FIX rate: {overall_actual:.3f} %")
    print(f"Overall adopted prediction: {overall_pred:.3f} %")
    print(f"Overall aggregate error: {overall_pred - overall_actual:+.3f} pp")


if __name__ == "__main__":
    main()
