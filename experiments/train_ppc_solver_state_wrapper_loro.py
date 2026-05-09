#!/usr/bin/env python3
"""LORO ceiling-lift experiment for D-034 (PR #43 follow-up).

Compares two feature variants under strict leave-one-route-out CV:

* baseline    -- sim/RINEX runtime features only (the deployed contract).
* treatment   -- the same features plus the six curated solver-state columns
                 surfaced via ``SolverStateWrapper``.

Both variants train an identical sklearn GradientBoostingRegressor on
``actual_fix_rate_pct`` directly.  The architecture is intentionally simple
so the only thing that varies between the two runs is the curated-six
feature set, isolating the wrapper's contribution.  Comparing either run
to the deployed §7.16 + isotonic + phaseguard 1.79 pp run-MAE is *not*
apples-to-apples (different model architecture), but tells us whether
the curated feature set has signal worth pursuing in a richer follow-up
architecture.

Outputs
-------

A CSV with columns
``variant, city, run, n_windows, route_actual_pct, route_pred_pct,
route_abs_error_pp, window_mae_pp, window_pearson_r``
plus a final aggregate row per variant
``variant=*_aggregate, run_mae_pp=<weighted mean>, window_mae_pp=<weighted mean>``.

Honest-LORO note
----------------

No hyperparameter tuning is performed.  Defaults from sklearn are used.
This is to avoid the "post-hoc threshold guard inflates LORO" failure
mode flagged in past PPC experiments — any tuning would have to be
nested inside the LORO loop, not selected once on aggregate metrics.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from _common import _is_metadata_or_label  # noqa: E402
from solver_state_wrapper import (  # noqa: E402
    CURATED_SOLVER_STATE_COLUMNS,
    SolverStateWrapper,
)

LOGGER = logging.getLogger("d034_loro")


def _runtime_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the deployable-feature column names (post _is_metadata_or_label gate)."""
    cols = []
    for name in df.columns:
        if _is_metadata_or_label(name):
            continue
        if pd.api.types.is_numeric_dtype(df[name]):
            cols.append(name)
    return cols


def _feature_matrix(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    arr = df.loc[:, feature_names].to_numpy(dtype=np.float64, copy=True)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _fit_predict_loro(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    random_state: int,
) -> np.ndarray:
    """Strict LORO predictions for ``actual_fix_rate_pct``.

    Returns a length-N array of held-out window predictions, indexed in the
    same order as ``df``.
    """
    df = df.reset_index(drop=True)
    keys = list(zip(df["city"].astype(str), df["run"].astype(str)))
    unique_routes = sorted(set(keys))
    preds = np.full(len(df), np.nan, dtype=np.float64)
    x_full = _feature_matrix(df, feature_names)
    y_full = df["actual_fix_rate_pct"].to_numpy(dtype=np.float64)
    for fold_idx, route in enumerate(unique_routes):
        test_mask = np.array([k == route for k in keys])
        train_mask = ~test_mask
        model = GradientBoostingRegressor(random_state=random_state + fold_idx)
        model.fit(x_full[train_mask], y_full[train_mask])
        preds[test_mask] = np.clip(model.predict(x_full[test_mask]), 0.0, 100.0)
    if np.isnan(preds).any():
        raise RuntimeError("LORO loop left some windows unpredicted; this is a bug.")
    return preds


def _aggregate_metrics(
    df: pd.DataFrame, preds: np.ndarray, *, weight_col: str = "sim_matched_epochs"
) -> dict[str, object]:
    """Per-route + aggregate run-MAE and window-MAE."""
    if weight_col not in df.columns:
        raise KeyError(f"weight column {weight_col} missing from input")
    weights = df[weight_col].to_numpy(dtype=np.float64)
    actual = df["actual_fix_rate_pct"].to_numpy(dtype=np.float64)
    abs_err = np.abs(preds - actual)
    rows: list[dict[str, object]] = []
    for (city, run), grp in df.groupby(["city", "run"], sort=True):
        idx = grp.index.to_numpy()
        wv = weights[idx]
        if wv.sum() <= 0:
            wv = np.ones_like(wv)
        a_route = float(np.average(actual[idx], weights=wv))
        p_route = float(np.average(preds[idx], weights=wv))
        wmae = float(np.average(abs_err[idx], weights=wv))
        if len(idx) > 1 and np.std(actual[idx]) > 0 and np.std(preds[idx]) > 0:
            r = float(np.corrcoef(actual[idx], preds[idx])[0, 1])
        else:
            r = float("nan")
        rows.append(
            {
                "city": city,
                "run": run,
                "n_windows": int(len(idx)),
                "route_actual_pct": a_route,
                "route_pred_pct": p_route,
                "route_abs_error_pp": abs(a_route - p_route),
                "window_mae_pp": wmae,
                "window_pearson_r": r,
            }
        )
    run_mae = float(np.mean([r["route_abs_error_pp"] for r in rows]))
    if weights.sum() > 0:
        window_mae = float(np.average(abs_err, weights=weights))
    else:
        window_mae = float(np.mean(abs_err))
    return {"per_route": rows, "run_mae_pp": run_mae, "window_mae_pp": window_mae}


def _run_variant(
    df: pd.DataFrame,
    feature_names: list[str],
    label: str,
    *,
    random_state: int,
) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray]:
    LOGGER.info("variant=%s n_features=%d", label, len(feature_names))
    preds = _fit_predict_loro(df, feature_names, random_state=random_state)
    metrics = _aggregate_metrics(df, preds)
    rows = []
    for r in metrics["per_route"]:
        row = {"variant": label, **r}
        rows.append(row)
    rows.append(
        {
            "variant": f"{label}_aggregate",
            "city": "",
            "run": "",
            "n_windows": int(len(df)),
            "route_actual_pct": float(np.average(
                df["actual_fix_rate_pct"].to_numpy(),
                weights=df["sim_matched_epochs"].to_numpy(),
            )),
            "route_pred_pct": float(np.average(
                preds,
                weights=df["sim_matched_epochs"].to_numpy(),
            )),
            "route_abs_error_pp": float("nan"),
            "window_mae_pp": float(metrics["window_mae_pp"]),
            "window_pearson_r": float(np.corrcoef(
                df["actual_fix_rate_pct"].to_numpy(), preds,
            )[0, 1]),
        }
    )
    return rows, {
        "variant": label,
        "run_mae_pp": metrics["run_mae_pp"],
        "window_mae_pp": metrics["window_mae_pp"],
        "n_features": len(feature_names),
    }, preds


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--per-window-output-csv", type=Path, default=None,
        help=(
            "Optional. If provided, write per-window LORO predictions for the "
            "curated_six_only variant (D-035 adopted post-demo5 QA model). "
            "Columns: city, run, window_index, path1_pred_fix_rate_pct."
        ),
    )
    parser.add_argument("--random-state", type=int, default=20260501)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.input_csv)
    df = df.sort_values(["city", "run", "window_index"]).reset_index(drop=True)
    LOGGER.info("loaded %d rows / %d cols", len(df), df.shape[1])

    wrapper = SolverStateWrapper()
    wrapper.validate(df)
    curated = wrapper.curate(df)
    df_treat = df.copy()
    for col in CURATED_SOLVER_STATE_COLUMNS:
        df_treat[col] = curated[col].to_numpy()

    baseline_features = _runtime_feature_columns(df_treat)
    treatment_features = baseline_features + list(CURATED_SOLVER_STATE_COLUMNS)
    overlap = set(baseline_features) & set(CURATED_SOLVER_STATE_COLUMNS)
    if overlap:
        raise RuntimeError(
            f"unexpected: curated columns leaked into baseline features: {sorted(overlap)}"
        )

    rows_b, summary_b, _ = _run_variant(
        df_treat, baseline_features, "baseline_no_solver_state",
        random_state=args.random_state,
    )
    rows_t, summary_t, _ = _run_variant(
        df_treat, treatment_features, "treatment_with_curated_six",
        random_state=args.random_state,
    )
    rows_c, summary_c, preds_c = _run_variant(
        df_treat, list(CURATED_SOLVER_STATE_COLUMNS), "curated_six_only",
        random_state=args.random_state,
    )

    out = pd.DataFrame(rows_b + rows_t + rows_c)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    LOGGER.info("wrote: %s (%d rows)", args.output_csv, len(out))

    if args.per_window_output_csv is not None:
        per_window = pd.DataFrame({
            "city": df_treat["city"].astype(str).to_numpy(),
            "run": df_treat["run"].astype(str).to_numpy(),
            "window_index": df_treat["window_index"].astype(int).to_numpy(),
            "path1_pred_fix_rate_pct": preds_c,
        })
        args.per_window_output_csv.parent.mkdir(parents=True, exist_ok=True)
        per_window.to_csv(args.per_window_output_csv, index=False)
        LOGGER.info(
            "wrote per-window predictions: %s (%d rows; curated_six_only variant)",
            args.per_window_output_csv, len(per_window),
        )

    print()
    print("=== summary ===")
    for s in (summary_b, summary_t, summary_c):
        print(
            f"  {s['variant']:<32}  run_mae={s['run_mae_pp']:.3f} pp  "
            f"window_mae={s['window_mae_pp']:.3f} pp  n_features={s['n_features']}"
        )
    delta_run = summary_t["run_mae_pp"] - summary_b["run_mae_pp"]
    delta_win = summary_t["window_mae_pp"] - summary_b["window_mae_pp"]
    print()
    print(
        f"  delta (treatment - baseline):  run_mae={delta_run:+.3f} pp  "
        f"window_mae={delta_win:+.3f} pp"
    )
    print("  deployed reference (§7.16 + isotonic75 + phaseguard): run_mae=1.79 pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
