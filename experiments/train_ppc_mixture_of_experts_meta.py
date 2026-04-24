#!/usr/bin/env python3
"""Mixture-of-experts meta-learner over the §7.11 / §7.13 / §7.16 experts.

STATUS: experimental / null result.  NOT part of the adopted production
pipeline.  The three MoE variants (full ridge, experts-only ridge, convex
combination) were all dominated by the single best expert (§7.16) on the
current 6-run dataset — see plan.md §7.20 on the source branch for the
full null-result write-up.  This script is retained as a reference
implementation for future experiments with structurally more diverse
experts (e.g. different base models, different feature families).  Do
not invoke from `predict.py` or the product deliverable pipeline.

Each expert produces strict-nested-LORO window predictions using a
different (threshold preset, residual alpha) combination.  Their failure
modes are complementary:

- §7.11 (hold_ready_thr=0.55, alpha=0.5): best on Tokyo run3 w17 (no
  false-lift regression) but under-lifts Tokyo run2 w26-w27.
- §7.13 (hold_ready_thr=0.60, alpha=0.5): best wmae; lifts Tokyo run2
  w25-w27 more but reintroduces Tokyo run3 w17 regression.
- §7.16 (hold_ready_thr=0.60, alpha=0.75): best run MAE and corr;
  amplifies §7.13 in both directions.

This meta-learner trains a ridge regressor to predict actual FIX rate
from the three experts' predictions plus 10 diagnostic validationhold
features.  Because each expert is already held out from its test run,
using them as features for a LORO ridge is strictly valid (no leakage).

Outputs:

- `..._moe_meta_run45_window_predictions.csv`
- `..._moe_meta_run45_best_model.csv`
- `..._moe_meta_run45_top_gains.csv`
- `..._moe_meta_run45_top_regressions.csv`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RESULTS_DIR = Path(__file__).resolve().parent / "results"
PREFIX = "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_"

EXPERTS = {
    "expert_p11": PREFIX + "solver_transition_surrogate_nested_et80_validationhold_carry_meta_run45",
    "expert_p13": PREFIX + "solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_meta_run45",
    "expert_p16": PREFIX + "solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45",
}

DIAGNOSTIC_FEATURES = [
    "clean_streak_s_at_start",
    "clean_streak_s_max",
    "hold_ready_frac",
    "hold_strict_ready_frac",
    "hold_carry_score_mean",
    "validation_pass_frac",
    "validation_reject_block_frac",
    "validation_block_score_p90",
    "validationhold_high_pred_reject_flag",
    "validationhold_low_pred_lift_flag",
]

DEFAULT_SUMMARY_CSV = RESULTS_DIR / "ppc_validationhold_window_summary_current_tight_hold.csv"
DEFAULT_RESULTS_PREFIX = PREFIX + "solver_transition_surrogate_nested_et80_validationhold_moe_meta_run45"


def _weighted_metrics(actual: np.ndarray, pred: np.ndarray, weights: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    w = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
    abs_err = np.abs(pred - actual)
    wmae = float(np.average(abs_err, weights=w))
    rmse = float(np.sqrt(np.average((pred - actual) ** 2, weights=w)))
    corr = float(np.corrcoef(actual, pred)[0, 1]) if np.std(pred) > 0 else float("nan")

    run_actuals = []
    run_preds = []
    for gid in np.unique(groups):
        m = groups == gid
        gw = w[m]
        run_actuals.append(float(np.average(actual[m], weights=gw)))
        run_preds.append(float(np.average(pred[m], weights=gw)))
    run_mae = float(np.mean(np.abs(np.asarray(run_actuals) - np.asarray(run_preds))))
    agg_err = float(np.average(pred, weights=w) - np.average(actual, weights=w))
    over_20 = int(np.count_nonzero(abs_err > 20.0))
    return {
        "window_weighted_mae_pp": wmae,
        "window_rmse_pp": rmse,
        "corr_window": corr,
        "run_mae_pp": run_mae,
        "aggregate_error_pp": agg_err,
        "over_20pp_windows": over_20,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixture-of-experts meta-learner")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--include-diagnostic-features", dest="include_diagnostic_features", action="store_true", default=True)
    parser.add_argument("--experts-only", dest="include_diagnostic_features", action="store_false")
    parser.add_argument("--convex", action="store_true", help="Use convex combination (weights >=0, sum=1) instead of ridge")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load expert predictions
    expert_dfs: dict[str, pd.DataFrame] = {}
    for name, prefix in EXPERTS.items():
        path = RESULTS_DIR / f"{prefix}_window_predictions.csv"
        df = pd.read_csv(path)
        expert_dfs[name] = df[["city", "run", "window_index", "actual_fix_rate_pct", "base_pred_fix_rate_pct", "sim_matched_epochs", "corrected_pred_fix_rate_pct"]].rename(columns={"corrected_pred_fix_rate_pct": name})

    # Merge all experts on (city, run, window_index)
    merged = expert_dfs["expert_p11"]
    for name in ("expert_p13", "expert_p16"):
        merged = merged.merge(expert_dfs[name][["city", "run", "window_index", name]], on=["city", "run", "window_index"], how="inner")

    # Load diagnostic features from validationhold summary
    summary = pd.read_csv(args.summary_csv)
    keep = ["city", "run", "window_index"] + [c for c in DIAGNOSTIC_FEATURES if c in summary.columns]
    missing = [c for c in DIAGNOSTIC_FEATURES if c not in summary.columns]
    if missing:
        # map signal -> flag
        rename_map = {"validationhold_high_pred_reject_signal": "validationhold_high_pred_reject_flag",
                      "validationhold_low_pred_lift_signal": "validationhold_low_pred_lift_flag"}
        for src, dst in rename_map.items():
            if dst in missing and src in summary.columns:
                summary[dst] = summary[src]
                keep.append(dst)
                missing.remove(dst)
    print(f"diagnostic features used: {len(keep) - 3} (missing: {missing})")
    merged = merged.merge(summary[keep], on=["city", "run", "window_index"], how="left")

    # Build feature matrix
    feature_cols = list(EXPERTS.keys())
    if args.include_diagnostic_features:
        feature_cols += [c for c in DIAGNOSTIC_FEATURES if c in merged.columns]
    X = merged[feature_cols].to_numpy(dtype=np.float64)
    X[~np.isfinite(X)] = 0.0
    y = merged["actual_fix_rate_pct"].to_numpy(dtype=np.float64)
    weights = merged["sim_matched_epochs"].to_numpy(dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)

    # Group encoding for LORO
    group_keys = list(zip(merged["city"].astype(str), merged["run"].astype(str)))
    unique_groups = sorted(set(group_keys))
    group_idx = np.array([unique_groups.index(k) for k in group_keys], dtype=np.int64)

    # LORO meta training
    pred_meta = np.full(len(merged), np.nan, dtype=np.float64)
    coef_log: list[tuple[str, str, dict[str, float]]] = []
    if args.convex:
        # Convex combination of the 3 experts only (ignores diagnostic features).
        expert_cols = list(EXPERTS.keys())
        for gid, group in enumerate(unique_groups):
            train_mask = group_idx != gid
            test_mask = group_idx == gid
            X_train = merged.loc[train_mask, expert_cols].to_numpy(dtype=np.float64)
            y_train = y[train_mask]
            w_train = weights[train_mask]

            def objective(weights_vec: np.ndarray) -> float:
                pred = X_train @ weights_vec
                return float(np.sum(w_train * (pred - y_train) ** 2))

            res = minimize(
                objective,
                x0=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
                method="SLSQP",
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
                bounds=[(0.0, 1.0)] * 3,
            )
            weight_vec = res.x
            X_test = merged.loc[test_mask, expert_cols].to_numpy(dtype=np.float64)
            pred_meta[test_mask] = X_test @ weight_vec
            coef_log.append((group[0], group[1], dict(zip(expert_cols, weight_vec))))
    else:
        for gid, group in enumerate(unique_groups):
            train_mask = group_idx != gid
            test_mask = group_idx == gid
            pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=args.ridge_alpha))])
            pipe.fit(X[train_mask], y[train_mask], model__sample_weight=weights[train_mask])
            pred_meta[test_mask] = pipe.predict(X[test_mask])
            coef_log.append((group[0], group[1], dict(zip(feature_cols, pipe.named_steps["model"].coef_))))

    pred_meta = np.clip(pred_meta, 0.0, 100.0)
    merged["meta_pred_fix_rate_pct"] = pred_meta

    # Metrics
    results: list[dict[str, object]] = []
    for name in feature_cols[:3] + ["meta"]:
        col = name if name != "meta" else "meta_pred_fix_rate_pct"
        m = _weighted_metrics(y, merged[col].to_numpy(dtype=np.float64), weights, group_idx)
        m["model"] = name
        results.append(m)
    metrics_df = pd.DataFrame(results)
    print("\nMetrics:")
    print(metrics_df[["model", "window_weighted_mae_pp", "run_mae_pp", "corr_window", "aggregate_error_pp", "over_20pp_windows"]].to_string(index=False))

    # Save predictions
    prefix = RESULTS_DIR / args.results_prefix
    out = merged[["city", "run", "window_index", "sim_matched_epochs", "actual_fix_rate_pct", "base_pred_fix_rate_pct", "expert_p11", "expert_p13", "expert_p16", "meta_pred_fix_rate_pct"]].copy()
    out["corrected_pred_fix_rate_pct"] = out["meta_pred_fix_rate_pct"]
    out["base_error_pp"] = out["base_pred_fix_rate_pct"] - out["actual_fix_rate_pct"]
    out["corrected_error_pp"] = out["corrected_pred_fix_rate_pct"] - out["actual_fix_rate_pct"]
    out["abs_error_gain_pp"] = np.abs(out["base_error_pp"]) - np.abs(out["corrected_error_pp"])
    out.to_csv(str(prefix) + "_window_predictions.csv", index=False)
    print(f"\nsaved: {str(prefix)}_window_predictions.csv")

    # Best model summary (meta row)
    best_row = metrics_df[metrics_df.model == "meta"].iloc[0]
    best_row["model"] = "meta_ridge_moe"
    best_row["residual_model"] = "ridge"
    best_row["alpha"] = args.ridge_alpha
    best_df = pd.DataFrame([best_row])
    best_df.to_csv(str(prefix) + "_best_model.csv", index=False)
    print(f"saved: {str(prefix)}_best_model.csv")

    # Mean coef report
    mean_coefs = {name: float(np.mean([c[2][name] for c in coef_log])) for name in feature_cols}
    print("\nMean ridge coefficients across LORO folds:")
    for k, v in sorted(mean_coefs.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k}: {v:+.4f}")


if __name__ == "__main__":
    main()
