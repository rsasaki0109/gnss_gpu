#!/usr/bin/env python3
"""Path 3 pre-gate: does temporal context help under the deployed contract?

PR #45 sketches a sequence-latent state-space model for the pre-demo5
estimator (run-MAE 1.79 pp ceiling).  Before paying the multi-week cost
to install hmmlearn / pykalman / hand-roll an HMM, we run a cheaper
gate: per-epoch GBR regression on actual_fixed using rolling-window
lag features computed from deployable signals only.

If lag features alone do not lift the deployed run-MAE under strict
LORO, the more complex state-space approach is unlikely to either, and
path 3's kill criterion ("discrete + linear-Gaussian both >= 1.79 pp")
applies a fortiori.

Setup
-----

* Per-epoch input CSV with ``city, run, gps_tow, actual_fixed`` and a
  large bundle of deployable sim/RINEX features.
* Window-definition CSV (the same one the deployed §7.16 stack was
  trained on) for ``actual_fix_rate_pct`` per window and for
  ``sim_matched_epochs`` weights.
* Strict LORO at ``(city, run)``.
* sklearn GradientBoostingRegressor with library defaults (no tuning).
* Deployable contract: no ``demo5_*`` / ``rtk_*`` / ``solver_*``
  columns, even though they exist in the per-epoch CSV.  This matches
  the ``_is_metadata_or_label`` gate used by the deployed stack.

Outputs
-------

CSV with per-route predicted vs actual FIX rate, run-MAE / window-MAE,
plus a summary line comparing to deployed 1.79 / 15.85 pp.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

LOGGER = logging.getLogger("path3_gate")


BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    # Simulator outputs
    "sim_satellite_count",
    "sim_carrier_phase_count",
    "sim_carrier_phase_lli_count",
    "sim_n_los",
    "sim_n_nlos",
    "sim_residual_p95_abs_m",
    "sim_fix_probability",
    # RINEX phase quality
    "rinex_phase_present_count",
    "rinex_phase_lli_count",
    "rinex_phase_jump_count",
    "rinex_phase_jump_fraction",
    "rinex_phase_streak_s_p50",
    "rinex_phase_doppler_residual_cycles_p90",
    # RINEX geometry-free / cycle slip
    "rinex_gf_slip_fraction",
)


ROLLING_WINDOWS_S: tuple[int, ...] = (30, 120)


def _compute_lag_features(df: pd.DataFrame, base_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Per-route, per-base-column rolling stats over the past N seconds.

    Uses ``shift(1)`` to make features strictly past-only so a deployable
    predictor at epoch t cannot see epoch t's own observation in its
    rolling stats.  Output column order is deterministic.
    """
    df = df.sort_values(["city", "run", "gps_tow"]).reset_index(drop=True)
    grouped = df.groupby(["city", "run"], sort=False, group_keys=False)
    sample_dt_s = 0.2

    out: dict[str, np.ndarray] = {}
    names: list[str] = []
    for col in base_cols:
        out[f"epoch_{col}"] = df[col].astype(float).to_numpy()
        names.append(f"epoch_{col}")
    for window_s in ROLLING_WINDOWS_S:
        n_samples = max(int(round(window_s / sample_dt_s)), 2)
        for col in base_cols:
            shifted = grouped[col].shift(1).astype(float)
            roller = shifted.groupby([df["city"], df["run"]]).rolling(
                window=n_samples, min_periods=1
            )
            mean = roller.mean().reset_index(level=[0, 1], drop=True)
            std = roller.std().reset_index(level=[0, 1], drop=True)
            max_ = roller.max().reset_index(level=[0, 1], drop=True)
            for stat_name, ser in (("mean", mean), ("std", std), ("max", max_)):
                key = f"{col}_lag{window_s}s_{stat_name}"
                out[key] = ser.to_numpy()
                names.append(key)
    feat = pd.DataFrame(out, index=df.index)
    return feat, names


def _fit_predict_loro(
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    *,
    random_state: int,
) -> np.ndarray:
    df = df.reset_index(drop=True)
    keys = list(zip(df["city"].astype(str), df["run"].astype(str)))
    unique_routes = sorted(set(keys))
    preds = np.full(len(df), np.nan, dtype=np.float64)
    y_full = df["actual_fixed"].astype(float).to_numpy()
    for fold_idx, route in enumerate(unique_routes):
        test_mask = np.array([k == route for k in keys])
        train_mask = ~test_mask
        LOGGER.info(
            "fold %d/%d: held-out=%s n_train=%d n_test=%d",
            fold_idx + 1, len(unique_routes), route,
            int(train_mask.sum()), int(test_mask.sum()),
        )
        model = GradientBoostingRegressor(random_state=random_state + fold_idx)
        model.fit(feature_matrix[train_mask], y_full[train_mask])
        preds[test_mask] = np.clip(model.predict(feature_matrix[test_mask]), 0.0, 1.0)
    if np.isnan(preds).any():
        raise RuntimeError("LORO loop left some epochs unpredicted")
    return preds


def _aggregate_to_windows(
    epoch_df: pd.DataFrame,
    epoch_pred_prob: np.ndarray,
    window_df: pd.DataFrame,
) -> pd.DataFrame:
    """For each window in ``window_df``, average per-epoch fix prob over
    epochs whose ``gps_tow`` falls in [window_start_tow, window_end_tow)."""
    epoch_df = epoch_df.copy().reset_index(drop=True)
    epoch_df["_pred_prob"] = epoch_pred_prob
    rows: list[dict[str, object]] = []
    for _, w in window_df.iterrows():
        mask = (
            (epoch_df["city"] == w["city"])
            & (epoch_df["run"] == w["run"])
            & (epoch_df["gps_tow"] >= w["window_start_tow"])
            & (epoch_df["gps_tow"] < w["window_end_tow"])
        )
        sub = epoch_df.loc[mask]
        if len(sub) == 0:
            continue
        rows.append({
            "city": w["city"],
            "run": w["run"],
            "window_index": int(w["window_index"]),
            "n_epochs_used": int(len(sub)),
            "actual_fix_rate_pct": float(w["actual_fix_rate_pct"]),
            "predicted_fix_rate_pct": float(sub["_pred_prob"].mean() * 100.0),
            "sim_matched_epochs": float(w.get("sim_matched_epochs", len(sub))),
        })
    return pd.DataFrame(rows)


def _per_route_metrics(window_pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (city, run), grp in window_pred_df.groupby(["city", "run"], sort=True):
        weights = grp["sim_matched_epochs"].to_numpy(dtype=np.float64)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        a_route = float(np.average(grp["actual_fix_rate_pct"], weights=weights))
        p_route = float(np.average(grp["predicted_fix_rate_pct"], weights=weights))
        wmae = float(
            np.average(
                np.abs(grp["predicted_fix_rate_pct"] - grp["actual_fix_rate_pct"]),
                weights=weights,
            )
        )
        if len(grp) > 1 and grp["actual_fix_rate_pct"].std() > 0 and grp["predicted_fix_rate_pct"].std() > 0:
            r = float(np.corrcoef(grp["actual_fix_rate_pct"], grp["predicted_fix_rate_pct"])[0, 1])
        else:
            r = float("nan")
        rows.append({
            "city": city,
            "run": run,
            "n_windows": int(len(grp)),
            "route_actual_pct": a_route,
            "route_pred_pct": p_route,
            "route_abs_error_pp": abs(a_route - p_route),
            "window_mae_pp": wmae,
            "window_pearson_r": r,
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch-csv", type=Path, required=True)
    parser.add_argument("--window-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=20260501)
    args = parser.parse_args(argv)

    LOGGER.info("loading epoch CSV: %s", args.epoch_csv)
    epoch_df = pd.read_csv(args.epoch_csv, low_memory=False)
    LOGGER.info("epoch rows=%d cols=%d", len(epoch_df), epoch_df.shape[1])
    LOGGER.info("loading window CSV: %s", args.window_csv)
    window_df = pd.read_csv(args.window_csv)

    epoch_df = epoch_df.dropna(subset=["actual_fixed"]).reset_index(drop=True)
    epoch_df = epoch_df.sort_values(["city", "run", "gps_tow"]).reset_index(drop=True)
    LOGGER.info("after actual_fixed dropna: %d rows", len(epoch_df))

    base_cols = [c for c in BASE_FEATURE_COLUMNS if c in epoch_df.columns]
    LOGGER.info("base features: %d / %d available", len(base_cols), len(BASE_FEATURE_COLUMNS))

    feat_df, feat_names = _compute_lag_features(epoch_df, base_cols)
    feat_arr = feat_df.to_numpy(dtype=np.float64, copy=True)
    feat_arr[~np.isfinite(feat_arr)] = 0.0
    LOGGER.info("feature matrix shape: %s", feat_arr.shape)

    preds = _fit_predict_loro(epoch_df, feat_arr, random_state=args.random_state)

    window_pred = _aggregate_to_windows(epoch_df, preds, window_df)
    LOGGER.info("aggregated to %d windows (window CSV had %d)", len(window_pred), len(window_df))

    per_route = _per_route_metrics(window_pred)
    run_mae = float(per_route["route_abs_error_pp"].mean())
    weights = window_pred["sim_matched_epochs"].to_numpy(dtype=np.float64)
    abs_err = np.abs(window_pred["predicted_fix_rate_pct"] - window_pred["actual_fix_rate_pct"]).to_numpy()
    if weights.sum() > 0:
        window_mae = float(np.average(abs_err, weights=weights))
    else:
        window_mae = float(np.mean(abs_err))
    overall_r = float(np.corrcoef(
        window_pred["actual_fix_rate_pct"], window_pred["predicted_fix_rate_pct"],
    )[0, 1])

    summary_row = pd.DataFrame([{
        "city": "_AGGREGATE_",
        "run": "",
        "n_windows": int(len(window_pred)),
        "route_actual_pct": float(np.average(
            window_pred["actual_fix_rate_pct"], weights=weights,
        )),
        "route_pred_pct": float(np.average(
            window_pred["predicted_fix_rate_pct"], weights=weights,
        )),
        "route_abs_error_pp": float("nan"),
        "window_mae_pp": float(window_mae),
        "window_pearson_r": overall_r,
    }])
    out = pd.concat([per_route, summary_row], ignore_index=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    LOGGER.info("wrote: %s", args.output_csv)

    print()
    print("=== path 3 pre-gate (lag features only, deployed contract) ===")
    print(per_route.to_string(index=False))
    print()
    print(f"  run_mae_pp       = {run_mae:.3f}")
    print(f"  window_mae_pp    = {window_mae:.3f}")
    print(f"  window_pearson_r = {overall_r:.3f}")
    print()
    print("  deployed reference (§7.16 + isotonic75 + phaseguard):")
    print("                       run_mae=1.79 pp  window_mae=15.85 pp  r=0.559")
    print()
    if run_mae < 1.79:
        print(f"  GATE PASSED: temporal context lifts run-MAE below 1.79 (delta {run_mae - 1.79:+.2f}).")
        print("  Path 3 (HMM / state-space) is worth the multi-week investment.")
    else:
        print(f"  GATE NULL: temporal context does not beat 1.79 pp (delta {run_mae - 1.79:+.2f}).")
        print("  Path 3 (HMM / state-space) is unlikely to beat it either; reconsider before pursuing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
