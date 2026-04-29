#!/usr/bin/env python3
"""Build the product deliverable package for the adopted strict-best model.

Reads the adopted `transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_isotonic75_meta_run45`
window predictions and emits:

1. `internal_docs/product_deliverable/route_level_fix_rate_prediction.csv`
   - One row per (city, run), epoch-weighted aggregate predicted FIX rate,
     actual FIX rate, aggregate error, and qualitative confidence tier.
2. `internal_docs/product_deliverable/window_level_details.csv`
   - Per-window predictions with flags for known failure cases.

Confidence tiers derive from the strict LORO metrics and the presence of
known-failure windows:

- high: no focus-case windows, expected error <= 3 pp
- medium: contains hidden-high or carry-boundary windows, expected 3-8 pp
- low: contains false-high or false-lift windows, wide uncertainty
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
DELIVERABLE_DIR = Path(__file__).resolve().parent.parent / "internal_docs" / "product_deliverable"
DEFAULT_PRED_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_isotonic75_meta_run45_window_predictions.csv"
)
REQUIRED_PREDICTION_COLUMNS = {
    "city",
    "run",
    "window_index",
    "sim_matched_epochs",
    "actual_fix_rate_pct",
    "base_pred_fix_rate_pct",
    "corrected_pred_fix_rate_pct",
}

# Threshold-based focus-case classification.  Applied per window to every
# input row; no hardcoded window-index list, so new runs with similar
# failure archetypes are tagged automatically.  Thresholds are chosen to
# flag only the material failure modes after the adopted correction is
# applied (mild residual error on low-actual windows is normal and is NOT
# tagged).
# Default thresholds; CLI overrides below in parse_args().
# Applied per window to every input row; no hardcoded window-index list,
# so new runs with similar failure archetypes are tagged automatically.
DEFAULT_THRESHOLDS = {
    "actual_low_pct": 5.0,             # actual FIX rate <= this counts as "actual low"
    "actual_high_pct": 75.0,           # actual FIX rate >= this counts as "actual high"
    "false_high_corrected_pct": 35.0,  # corrected >= this on an actual-low window
    "hidden_high_gap_pp": 40.0,        # actual - corrected >= this on an actual-high window
    "lift_pp": 15.0,                   # corrected - base >= this counts as a material lift
    "lift_corrected_pct": 15.0,        # absolute corrected floor for lift classification
    "reject_pp": 15.0,                 # base - corrected >= this counts as a material reject
    "reject_base_pct": 25.0,           # minimum base for reject classification
}


def _classify_window(row: "pd.Series", thresholds: dict | None = None) -> tuple[str, str]:
    """Return (tag, note) for a window based on actual / base / corrected values.

    Thresholds intentionally flag only material failures: everyday
    ~15 pp residual noise on low-actual windows is NOT tagged.
    """
    actual = float(row["actual_fix_rate_pct"])
    base = float(row["base_pred_fix_rate_pct"])
    corrected = float(row["corrected_pred_fix_rate_pct"])
    thr = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    delta = corrected - base

    # false_high: corrected prediction still inflates an actual-zero window
    if actual <= thr["actual_low_pct"] and corrected >= thr["false_high_corrected_pct"]:
        return "false_high", f"corrected prediction {corrected:.1f}% against actual {actual:.1f}%; deployable features misread"

    # hidden_high: actual is high but corrected still under-predicts by a large gap
    if actual >= thr["actual_high_pct"] and (actual - corrected) >= thr["hidden_high_gap_pp"]:
        return "hidden_high", f"actual {actual:.1f}% but corrected only {corrected:.1f}%; under-predicted by {actual - corrected:.1f} pp"

    # Remaining cases: actual-low windows where lift or rejection is material
    if actual <= thr["actual_low_pct"]:
        if delta >= thr["lift_pp"] and corrected >= thr["lift_corrected_pct"]:
            return "false_lift", f"lifted by {delta:+.1f} pp against actual {actual:.1f}%; corrected {corrected:.1f}%"
        if -delta >= thr["reject_pp"] and base >= thr["reject_base_pct"]:
            return "false_lift_resolved", f"successfully rejected by {-delta:.1f} pp against actual {actual:.1f}% (base was {base:.1f}%)"

    return "", ""


def _confidence_tier(tags: list[str]) -> tuple[str, str]:
    """Pick a qualitative confidence tier for a run based on the set of focus tags present."""
    if any(t == "false_high" for t in tags):
        return "low", "run contains at least one false-high window (pred high, actual low); individual window trust low"
    if any(t == "false_lift" for t in tags):
        return "low", "run contains at least one false-lift window (pred lifted, actual low); individual window trust low"
    if any(t == "hidden_high" for t in tags):
        return "medium", "run contains hidden-high windows (actual high, pred under-lifted)"
    # false_lift_resolved is a positive outcome (model caught the lift) and does not downgrade the tier.
    return "high", "no material focus-case windows"


def _require_prediction_columns(df: pd.DataFrame, path: Path) -> None:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(df.columns))
    if missing:
        raise SystemExit(
            f"prediction CSV is missing required columns: {', '.join(missing)}\n"
            f"path: {path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build product deliverable CSVs")
    parser.add_argument("--prediction-csv", type=Path, default=DEFAULT_PRED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DELIVERABLE_DIR)
    g = parser.add_argument_group("focus-case thresholds")
    g.add_argument("--actual-low-pct", type=float, default=DEFAULT_THRESHOLDS["actual_low_pct"],
                   help=f"actual FIX rate <= this counts as 'actual low' (default: {DEFAULT_THRESHOLDS['actual_low_pct']})")
    g.add_argument("--actual-high-pct", type=float, default=DEFAULT_THRESHOLDS["actual_high_pct"],
                   help=f"actual FIX rate >= this counts as 'actual high' (default: {DEFAULT_THRESHOLDS['actual_high_pct']})")
    g.add_argument("--false-high-corrected-pct", type=float, default=DEFAULT_THRESHOLDS["false_high_corrected_pct"],
                   help=f"corrected >= this on an actual-low window -> false_high (default: {DEFAULT_THRESHOLDS['false_high_corrected_pct']})")
    g.add_argument("--hidden-high-gap-pp", type=float, default=DEFAULT_THRESHOLDS["hidden_high_gap_pp"],
                   help=f"(actual - corrected) >= this on actual-high window -> hidden_high (default: {DEFAULT_THRESHOLDS['hidden_high_gap_pp']})")
    g.add_argument("--lift-pp", type=float, default=DEFAULT_THRESHOLDS["lift_pp"],
                   help=f"(corrected - base) >= this -> material lift (default: {DEFAULT_THRESHOLDS['lift_pp']})")
    g.add_argument("--lift-corrected-pct", type=float, default=DEFAULT_THRESHOLDS["lift_corrected_pct"],
                   help=f"absolute corrected floor for lift classification (default: {DEFAULT_THRESHOLDS['lift_corrected_pct']})")
    g.add_argument("--reject-pp", type=float, default=DEFAULT_THRESHOLDS["reject_pp"],
                   help=f"(base - corrected) >= this -> material reject (default: {DEFAULT_THRESHOLDS['reject_pp']})")
    g.add_argument("--reject-base-pct", type=float, default=DEFAULT_THRESHOLDS["reject_base_pct"],
                   help=f"minimum base for reject classification (default: {DEFAULT_THRESHOLDS['reject_base_pct']})")
    parser.add_argument("--skip-dashboard", action="store_true",
                        help="do not regenerate internal_docs/product_deliverable/dashboard.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = {
        "actual_low_pct": args.actual_low_pct,
        "actual_high_pct": args.actual_high_pct,
        "false_high_corrected_pct": args.false_high_corrected_pct,
        "hidden_high_gap_pp": args.hidden_high_gap_pp,
        "lift_pp": args.lift_pp,
        "lift_corrected_pct": args.lift_corrected_pct,
        "reject_pp": args.reject_pp,
        "reject_base_pct": args.reject_base_pct,
    }
    df = pd.read_csv(args.prediction_csv)
    _require_prediction_columns(df, args.prediction_csv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # window-level details
    window_rows: list[dict[str, object]] = []
    for _, row in df.sort_values(["city", "run", "window_index"]).iterrows():
        focus_tag, focus_note = _classify_window(row, thresholds)
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
    print(f"saved: {window_path} ({len(window_df)} rows)", flush=True)

    tags_by_run = window_df.groupby(["city", "run"])["focus_case_tag"].apply(lambda s: [t for t in s if t]).to_dict()

    # route-level aggregate
    route_rows: list[dict[str, object]] = []
    for (city, run), group in df.groupby(["city", "run"], sort=True):
        weights = group["sim_matched_epochs"].to_numpy(dtype=np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
        actual = float(np.average(group["actual_fix_rate_pct"].to_numpy(dtype=np.float64), weights=weights))
        pred = float(np.average(group["corrected_pred_fix_rate_pct"].to_numpy(dtype=np.float64), weights=weights))
        base = float(np.average(group["base_pred_fix_rate_pct"].to_numpy(dtype=np.float64), weights=weights))
        run_tags = tags_by_run.get((city, run), [])
        tier, note = _confidence_tier(run_tags)
        n_focus = len(run_tags)
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
    print(f"saved: {route_path} ({len(route_df)} rows)", flush=True)

    # summary stats
    print("\nRoute-level summary (adopted model):", flush=True)
    print(
        route_df[
            [
                "city",
                "run",
                "window_count",
                "actual_fix_rate_pct",
                "adopted_pred_fix_rate_pct",
                "adopted_abs_error_pp",
                "confidence_tier",
            ]
        ].to_string(index=False),
        flush=True,
    )

    overall_actual = float(np.average(window_df["actual_fix_rate_pct"], weights=df["sim_matched_epochs"].fillna(1.0).clip(lower=1.0)))
    overall_pred = float(np.average(window_df["adopted_pred_fix_rate_pct"], weights=df["sim_matched_epochs"].fillna(1.0).clip(lower=1.0)))
    print(f"\nOverall actual FIX rate: {overall_actual:.3f} %", flush=True)
    print(f"Overall adopted prediction: {overall_pred:.3f} %", flush=True)
    print(f"Overall aggregate error: {overall_pred - overall_actual:+.3f} pp", flush=True)

    # Regenerate the dashboard HTML unless the operator opted out.
    if not args.skip_dashboard:
        dashboard_script = EXPERIMENTS_DIR / "build_product_dashboard.py"
        if dashboard_script.exists():
            print("\nregenerating dashboard.html...", flush=True)
            result = subprocess.run(
                [sys.executable, str(dashboard_script),
                 "--route-csv", str(route_path),
                 "--window-csv", str(window_path),
                 "--output", str(args.output_dir / "dashboard.html")],
                check=False,
            )
            if result.returncode != 0:
                print("WARNING: dashboard regeneration failed; CSVs are still up to date", flush=True)
        else:
            print("note: build_product_dashboard.py not found; skipping dashboard regeneration", flush=True)


if __name__ == "__main__":
    main()
