#!/usr/bin/env python3
"""Path 3 HMM PoC: does a hidden-state model lift the deployed run-MAE?

PR #45 sketches a sequence-latent state-space model for the pre-demo5
estimator (run-MAE 1.79 pp ceiling).  PR #46 ran a cheaper lag-feature
GBR pre-gate which returned NULL (run-MAE 6.95 pp).  This script
implements the actual HMM ladder step from the sketch (3-4 state
GaussianHMM with hmmlearn) so the kill criterion is hit head-on rather
than only via the lag-GBR proxy.

Setup
-----

* Same per-epoch CSV and same window CSV as the lag-GBR pre-gate.
* Same 14 deployable per-epoch base features.
* Strict LORO over (city, run): refit the HMM on 5 routes, run forward
  filter on the held-out route, map state posteriors to per-epoch fix
  probability via the training-set state-conditional means, then
  aggregate to windows the same way as the pre-gate.

Kill criteria (sketch §"Kill criteria")
---------------------------------------

* run-MAE >= 1.79 pp under strict LORO  -> HMM null
* tokyo/run2 residual >= 5 pp           -> HMM null on the focal cluster
* posterior collapse onto a single state for most epochs -> degenerate

Outputs
-------

CSV with per-route predicted vs actual FIX rate, run-MAE / window-MAE,
plus diagnostics (state usage histogram, log-likelihood per fold).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

LOGGER = logging.getLogger("path3_hmm")


BASE_FEATURE_COLUMNS: tuple[str, ...] = (
    # Same 14 features as the PR #46 lag-GBR pre-gate; deployed contract
    # (no demo5_* / rtk_* / solver_*).
    "sim_satellite_count",
    "sim_carrier_phase_count",
    "sim_carrier_phase_lli_count",
    "sim_n_los",
    "sim_n_nlos",
    "sim_residual_p95_abs_m",
    "sim_fix_probability",
    "rinex_phase_present_count",
    "rinex_phase_lli_count",
    "rinex_phase_jump_count",
    "rinex_phase_jump_fraction",
    "rinex_phase_streak_s_p50",
    "rinex_phase_doppler_residual_cycles_p90",
    "rinex_gf_slip_fraction",
)


def _standardize_train_apply_test(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score train/test using train statistics. Returns (Z_tr, Z_te, mean, std)."""
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Z_tr = (train - mean) / std
    Z_te = (test - mean) / std
    return Z_tr, Z_te, mean, std


def _route_lengths(df: pd.DataFrame, mask: np.ndarray) -> list[int]:
    """Return per-route epoch counts in the order rows appear in the masked df.

    HMM fit/score consumes a concatenated array with route lengths so it
    knows where one sequence ends and the next begins.
    """
    sub = df.loc[mask, ["city", "run"]].astype(str)
    keys = list(zip(sub["city"], sub["run"]))
    lengths: list[int] = []
    if not keys:
        return lengths
    cur = keys[0]
    count = 0
    for k in keys:
        if k == cur:
            count += 1
        else:
            lengths.append(count)
            cur = k
            count = 1
    lengths.append(count)
    return lengths


def _state_to_fix_prob_table(
    posteriors: np.ndarray, y_train: np.ndarray
) -> np.ndarray:
    """Map state index -> P(fix | state=k) using soft assignments on train.

    posteriors: (T_train, K) gamma from HMM posterior on train.
    y_train: (T_train,) actual_fixed labels (0/1).
    """
    K = posteriors.shape[1]
    table = np.zeros(K, dtype=np.float64)
    eps = 1e-12
    for k in range(K):
        w = posteriors[:, k]
        denom = float(w.sum())
        if denom <= eps:
            table[k] = float(np.nanmean(y_train))
        else:
            table[k] = float(np.sum(w * y_train) / denom)
    return table


def _hmm_per_epoch_fix_prob(
    df: pd.DataFrame,
    feat_arr: np.ndarray,
    *,
    n_states: int,
    covariance_type: str,
    n_iter: int,
    random_state: int,
    diag_records: list[dict[str, object]],
) -> np.ndarray:
    """LORO loop: fit HMM on 5 routes, forward-filter held-out route, map
    posteriors to per-epoch fix probability."""
    df = df.reset_index(drop=True)
    keys = list(zip(df["city"].astype(str), df["run"].astype(str)))
    unique_routes = sorted(set(keys))
    preds = np.full(len(df), np.nan, dtype=np.float64)
    y_full = df["actual_fixed"].astype(float).to_numpy()

    for fold_idx, route in enumerate(unique_routes):
        test_mask = np.array([k == route for k in keys])
        train_mask = ~test_mask

        feat_train = feat_arr[train_mask]
        feat_test = feat_arr[test_mask]
        y_train = y_full[train_mask]

        Z_tr, Z_te, _mean, _std = _standardize_train_apply_test(feat_train, feat_test)
        Z_tr[~np.isfinite(Z_tr)] = 0.0
        Z_te[~np.isfinite(Z_te)] = 0.0

        train_lengths = _route_lengths(df, train_mask)
        test_lengths = _route_lengths(df, test_mask)
        assert sum(train_lengths) == Z_tr.shape[0]
        assert sum(test_lengths) == Z_te.shape[0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hmm = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=1e-3,
                init_params="stmc",
                random_state=random_state + fold_idx,
            )
            hmm.fit(Z_tr, lengths=train_lengths)
            train_post = hmm.predict_proba(Z_tr, lengths=train_lengths)

        state_to_fix = _state_to_fix_prob_table(train_post, y_train)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_post = hmm.predict_proba(Z_te, lengths=test_lengths)
            test_ll = float(hmm.score(Z_te, lengths=test_lengths))

        per_epoch_prob = test_post @ state_to_fix
        preds[test_mask] = np.clip(per_epoch_prob, 0.0, 1.0)

        train_state_usage = train_post.mean(axis=0)
        test_state_usage = test_post.mean(axis=0)
        max_test_state_share = float(test_state_usage.max())
        diag_records.append({
            "fold": fold_idx + 1,
            "held_out_city": route[0],
            "held_out_run": route[1],
            "n_train_epochs": int(train_mask.sum()),
            "n_test_epochs": int(test_mask.sum()),
            "test_log_likelihood": test_ll,
            "test_log_likelihood_per_epoch": test_ll / max(1, int(test_mask.sum())),
            "max_test_state_share": max_test_state_share,
            "state_to_fix": json.dumps([round(float(x), 4) for x in state_to_fix]),
            "train_state_usage": json.dumps([round(float(x), 4) for x in train_state_usage]),
            "test_state_usage": json.dumps([round(float(x), 4) for x in test_state_usage]),
        })
        LOGGER.info(
            "fold %d/%d: held-out=%s n_train=%d n_test=%d "
            "test_ll/epoch=%.4f max_state_share=%.3f state_fix=%s",
            fold_idx + 1, len(unique_routes), route,
            int(train_mask.sum()), int(test_mask.sum()),
            test_ll / max(1, int(test_mask.sum())),
            max_test_state_share,
            [round(float(x), 3) for x in state_to_fix],
        )

    if np.isnan(preds).any():
        raise RuntimeError("LORO loop left some epochs unpredicted")
    return preds


def _aggregate_to_windows(
    epoch_df: pd.DataFrame,
    epoch_pred_prob: np.ndarray,
    window_df: pd.DataFrame,
) -> pd.DataFrame:
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
        if (
            len(grp) > 1
            and grp["actual_fix_rate_pct"].std() > 0
            and grp["predicted_fix_rate_pct"].std() > 0
        ):
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
    parser.add_argument("--diag-csv", type=Path, default=None,
                        help="Optional diagnostics CSV (per-fold state usage / log-likelihood)")
    parser.add_argument("--n-states", type=int, default=3,
                        help="HMM hidden state count (sketch suggests 3-4)")
    parser.add_argument("--covariance-type", type=str, default="diag",
                        choices=("diag", "full", "tied", "spherical"))
    parser.add_argument("--n-iter", type=int, default=50)
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
    if len(base_cols) == 0:
        raise RuntimeError("no base features found in epoch CSV")

    feat_arr = epoch_df[base_cols].to_numpy(dtype=np.float64, copy=True)
    feat_arr[~np.isfinite(feat_arr)] = 0.0
    LOGGER.info("feature matrix shape: %s", feat_arr.shape)

    diag_records: list[dict[str, object]] = []
    preds = _hmm_per_epoch_fix_prob(
        epoch_df,
        feat_arr,
        n_states=args.n_states,
        covariance_type=args.covariance_type,
        n_iter=args.n_iter,
        random_state=args.random_state,
        diag_records=diag_records,
    )

    window_pred = _aggregate_to_windows(epoch_df, preds, window_df)
    LOGGER.info("aggregated to %d windows (window CSV had %d)", len(window_pred), len(window_df))

    per_route = _per_route_metrics(window_pred)
    run_mae = float(per_route["route_abs_error_pp"].mean())
    weights = window_pred["sim_matched_epochs"].to_numpy(dtype=np.float64)
    abs_err = np.abs(
        window_pred["predicted_fix_rate_pct"] - window_pred["actual_fix_rate_pct"]
    ).to_numpy()
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

    if args.diag_csv is not None:
        diag_df = pd.DataFrame(diag_records)
        args.diag_csv.parent.mkdir(parents=True, exist_ok=True)
        diag_df.to_csv(args.diag_csv, index=False)
        LOGGER.info("wrote diag: %s", args.diag_csv)

    print()
    print(f"=== path 3 HMM PoC (n_states={args.n_states}, deployed contract) ===")
    print(per_route.to_string(index=False))
    print()
    print(f"  run_mae_pp       = {run_mae:.3f}")
    print(f"  window_mae_pp    = {window_mae:.3f}")
    print(f"  window_pearson_r = {overall_r:.3f}")
    print()
    print("  deployed reference (§7.16 + isotonic75 + phaseguard):")
    print("                       run_mae=1.79 pp  window_mae=15.85 pp  r=0.559")
    print()
    tokyo_run2 = per_route[(per_route["city"] == "tokyo") & (per_route["run"] == "run2")]
    if len(tokyo_run2) == 1:
        tk_err = float(tokyo_run2.iloc[0]["route_abs_error_pp"])
        print(f"  tokyo/run2 residual = {tk_err:.3f} pp (deployed ceiling 8.13 pp)")
        print()
    max_state_share_overall = float(np.max([d["max_test_state_share"] for d in diag_records]))
    print(f"  max test state share across folds = {max_state_share_overall:.3f}")
    print("    (>=0.95 indicates posterior collapse onto a single state)")
    print()
    if run_mae < 1.79 and (len(tokyo_run2) == 1 and tk_err < 5.0):
        print("  KILL CRITERIA NOT MET: HMM lifts run-MAE below 1.79 pp AND")
        print("  tokyo/run2 residual below 5 pp. Path 3 worth further investment.")
    else:
        print("  KILL CRITERIA MET (sketch §'Kill criteria'):")
        if run_mae >= 1.79:
            print(f"    run-MAE {run_mae:.3f} >= 1.79 pp")
        if len(tokyo_run2) == 1 and tk_err >= 5.0:
            print(f"    tokyo/run2 residual {tk_err:.3f} >= 5 pp")
        if max_state_share_overall >= 0.95:
            print(f"    posterior collapse: max state share {max_state_share_overall:.3f} >= 0.95")
        print("  Path 3 (HMM / state-space) is null on this dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
