#!/usr/bin/env python3
"""Epoch-level FIX classifier with strict leave-one-run-out validation.

Predicts P(actual_fixed = 1 | deployable per-epoch features) at each
0.2 s taroz/PPC epoch.  Training set: 58706 epochs across 6 runs; ~17 %
of epochs are FIX under demo5.  Cross-validation is strict LORO by
run, matching the window-level §7.16 evaluation protocol.

Features exclude demo5 solver internals (`rtk_*`, `solver_*`,
`demo5_*`), the target column `actual_fixed`, and run identifiers.
Validation/hold surrogate state variables ARE included because they
are deployable.

Outputs under `experiments/results/`:

- `ppc_epoch_fix_classifier_predictions.csv`
  columns: city, run, gps_tow, actual_fixed, p_fix_pct
- `ppc_epoch_fix_classifier_metrics.csv`
  per-run + overall AUC / log-loss / Brier score
- `ppc_epoch_fix_classifier_window_aggregated.csv`
  per-window mean of p_fix_pct for comparison with §7.16
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
DEFAULT_EPOCHS_CSV = RESULTS_DIR / (
    "ppc_demo5_fix_rate_compare_fullsim_stride1_phase_proxy_stat_rinex_phasejump_"
    "t0p25_gf0p2_simloscont_focused_simadop_nowt_validationhold_epochs.csv"
)
DEFAULT_OUTPUT_PREFIX = RESULTS_DIR / "ppc_epoch_fix_classifier"


def _is_label_or_metadata(name: str) -> bool:
    if name in {"city", "run", "gps_tow", "actual_fixed", "demo5_pos_file"}:
        return True
    if name.startswith(("rtk_", "solver_", "demo5_")):
        return True
    return False


def _load_epochs(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, low_memory=False)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in numeric_cols if not _is_label_or_metadata(c)]
    return df, feature_cols


def _metrics_row(y_true: np.ndarray, p: np.ndarray, n_epochs: int, label: str) -> dict[str, object]:
    finite = np.isfinite(p)
    y_true = y_true[finite]
    p = p[finite]
    out = {"group": label, "epochs": int(n_epochs)}
    if y_true.size == 0 or len(set(y_true.tolist())) < 2:
        out.update({"auc": float("nan"), "log_loss": float("nan"), "brier": float("nan"), "fix_fraction": float("nan")})
        return out
    out["auc"] = float(roc_auc_score(y_true, p))
    out["log_loss"] = float(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6)))
    out["brier"] = float(brier_score_loss(y_true, p))
    out["fix_fraction"] = float(y_true.mean())
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Epoch-level FIX classifier with LORO evaluation")
    parser.add_argument("--epochs-csv", type=Path, default=DEFAULT_EPOCHS_CSV)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--l2-regularization", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=2034)
    parser.add_argument("--window-duration-s", type=float, default=30.0,
                        help="used for per-window aggregation of epoch predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, feature_cols = _load_epochs(args.epochs_csv)
    print(f"loaded {len(df)} epochs, {len(feature_cols)} deployable features")
    y = df["actual_fixed"].astype(int).to_numpy()
    x = df[feature_cols].to_numpy(dtype=np.float64)
    # HistGradientBoostingClassifier handles NaN natively.
    groups = df["city"].astype(str) + "/" + df["run"].astype(str)
    group_labels = sorted(groups.unique())
    predictions = np.full(len(df), np.nan, dtype=np.float64)

    for outer in group_labels:
        train_mask = (groups != outer).to_numpy()
        test_mask = (groups == outer).to_numpy()
        clf = HistGradientBoostingClassifier(
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            l2_regularization=args.l2_regularization,
            random_state=args.random_state,
        )
        clf.fit(x[train_mask], y[train_mask])
        predictions[test_mask] = clf.predict_proba(x[test_mask])[:, 1]
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        auc = roc_auc_score(y[test_mask], predictions[test_mask]) if len(set(y[test_mask].tolist())) > 1 else float("nan")
        print(f"  held-out {outer}: train={n_train} test={n_test} AUC={auc:.3f}")

    # Write per-epoch predictions
    pred_df = df[["city", "run", "gps_tow", "actual_fixed"]].copy()
    pred_df["p_fix_pct"] = 100.0 * predictions
    pred_path = Path(str(args.output_prefix) + "_predictions.csv")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(pred_path, index=False)
    print(f"\nsaved: {pred_path}")

    # Metrics
    metric_rows: list[dict[str, object]] = []
    for outer in group_labels:
        mask = (groups == outer).to_numpy()
        metric_rows.append(_metrics_row(y[mask], predictions[mask], int(mask.sum()), outer))
    metric_rows.append(_metrics_row(y, predictions, len(df), "OVERALL"))
    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = Path(str(args.output_prefix) + "_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"saved: {metrics_path}")
    print("\nLORO metrics:")
    print(metrics_df.to_string(index=False))

    # Window-level aggregation (mean of p_fix within each 30 s window)
    window_dur = args.window_duration_s
    pred_df = pred_df.sort_values(["city", "run", "gps_tow"]).reset_index(drop=True)
    run_start_tow = pred_df.groupby(["city", "run"])["gps_tow"].transform("min")
    pred_df["window_index"] = ((pred_df["gps_tow"] - run_start_tow) // window_dur).astype(int)
    agg = pred_df.groupby(["city", "run", "window_index"]).agg(
        epoch_count=("actual_fixed", "size"),
        actual_fix_rate_pct=("actual_fixed", lambda s: 100.0 * s.mean()),
        epoch_pred_fix_rate_pct=("p_fix_pct", "mean"),
    ).reset_index()
    agg["abs_err_pp"] = (agg["epoch_pred_fix_rate_pct"] - agg["actual_fix_rate_pct"]).abs()
    agg_path = Path(str(args.output_prefix) + "_window_aggregated.csv")
    agg.to_csv(agg_path, index=False)
    print(f"\nsaved: {agg_path}")

    # Window-level metrics summary
    weights = agg["epoch_count"].to_numpy(dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
    wmae = float(np.average(agg["abs_err_pp"].to_numpy(dtype=np.float64), weights=weights))
    # Per-run aggregate
    run_rows = []
    for (c, r), g in agg.groupby(["city", "run"]):
        gw = g["epoch_count"].to_numpy(dtype=np.float64)
        gw = np.where(np.isfinite(gw) & (gw > 0.0), gw, 1.0)
        actual = float(np.average(g["actual_fix_rate_pct"].to_numpy(dtype=np.float64), weights=gw))
        pred = float(np.average(g["epoch_pred_fix_rate_pct"].to_numpy(dtype=np.float64), weights=gw))
        run_rows.append({"city": c, "run": r, "actual": actual, "pred": pred, "abs_err_pp": abs(pred - actual)})
    run_df = pd.DataFrame(run_rows)
    run_mae = float(run_df["abs_err_pp"].mean())
    corr = float(np.corrcoef(agg["epoch_pred_fix_rate_pct"].to_numpy(dtype=np.float64),
                              agg["actual_fix_rate_pct"].to_numpy(dtype=np.float64))[0, 1])
    print(f"\nwindow-aggregated metrics (for comparison with §7.16):")
    print(f"  weighted MAE: {wmae:.3f} pp")
    print(f"  run MAE:      {run_mae:.3f} pp")
    print(f"  correlation:  {corr:.3f}")


if __name__ == "__main__":
    main()
