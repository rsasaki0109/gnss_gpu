#!/usr/bin/env python3
"""Furukawa-2019-style paper evaluation.

Furukawa & Kubo (2019, IPNTJ Vol.10 No.2, doi via J-STAGE)
"Prediction of Fixing of RTK-GNSS Positioning in Multipath Environment
Using Radiowave Propagation Simulation" evaluates their simulation
predictor against RTK-GNSS observations in a Tokyo/Hibiya urban course
using:

- Table 2: RTK-FIX matching rate swept over the "LoS-continue satellite
  count" threshold (best 83.9 % at 9 satellites).
- Fig.9: time series of good-signal satellite count plus
  RTK(Simulation) vs RTK FIXED(Measurement) at each epoch.
- Fig.10: side-by-side map of RTK FIXED(Simulation) vs
  RTK FIXED(Measurement) points on the drive route.

This script computes the analogous matching rate for our adopted
§7.16 window predictor aggregated to the epoch level (nearest window)
and for the per-epoch classifier, across a sweep of P(FIX) thresholds.
Outputs go under `internal_docs/product_deliverable/`:

- `paper_style_matching_rate.csv`
- `paper_style_per_run_accuracy.csv`
- `plots/{city}_{run}_fix_comparison_map.png` (Fig.10 style)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
DELIVERABLE_DIR = EXPERIMENTS_DIR.parent / "internal_docs" / "product_deliverable"
PLOTS_DIR = DELIVERABLE_DIR / "plots"

DEFAULT_EPOCH_PRED_CSV = RESULTS_DIR / "ppc_epoch_fix_classifier_predictions.csv"
DEFAULT_WINDOW_PRED_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45_window_predictions.csv"
)
DEFAULT_POS_ROOT = RESULTS_DIR / "demo5_pos"

RUN_ORDER = [
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
]

# Match Furukawa Table 2 sweep granularity (~5 % steps around 50 %).
DEFAULT_THRESHOLDS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]


def _read_pos_file(path: Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            tokens = stripped.split()
            if len(tokens) < 6:
                continue
            rows.append(tokens)
    df = pd.DataFrame(rows)
    df.columns = [
        "date", "time", "lat_deg", "lon_deg", "height_m", "q", "ns",
        "sdn", "sde", "sdu", "sdne", "sdeu", "sdun", "age", "ratio",
    ][: df.shape[1]]
    df["lat_deg"] = df["lat_deg"].astype(float)
    df["lon_deg"] = df["lon_deg"].astype(float)
    df["q"] = df["q"].astype(int)
    ts = pd.to_datetime(df["date"] + " " + df["time"], format="%Y/%m/%d %H:%M:%S.%f")
    df["ts"] = ts
    df["elapsed_s"] = (ts - ts.iloc[0]).dt.total_seconds()
    df["is_fix_actual"] = (df["q"] == 1).astype(int)
    return df[["ts", "elapsed_s", "lat_deg", "lon_deg", "q", "is_fix_actual"]]


def _epoch_preds_for_run(epoch_df: pd.DataFrame, city: str, run: str) -> pd.DataFrame:
    sub = epoch_df[(epoch_df["city"] == city) & (epoch_df["run"] == run)].copy()
    sub = sub.sort_values("gps_tow").reset_index(drop=True)
    sub["elapsed_s"] = sub["gps_tow"] - sub["gps_tow"].iloc[0]
    return sub


def _window_preds_at_epoch(window_df: pd.DataFrame, city: str, run: str, pos_elapsed_s: np.ndarray, window_duration_s: float = 30.0) -> np.ndarray:
    """Expand window-level corrected_pred_fix_rate_pct to each epoch's time."""
    sub = window_df[(window_df["city"] == city) & (window_df["run"] == run)].sort_values("window_index")
    if sub.empty:
        return np.full(len(pos_elapsed_s), np.nan, dtype=np.float64)
    preds = np.full(len(pos_elapsed_s), np.nan, dtype=np.float64)
    for _, row in sub.iterrows():
        start = float(row["window_index"]) * window_duration_s
        end = start + window_duration_s
        mask = (pos_elapsed_s >= start) & (pos_elapsed_s < end)
        preds[mask] = float(row["corrected_pred_fix_rate_pct"])
    return preds


def _matching_rate(predicted_is_fix: np.ndarray, actual_is_fix: np.ndarray) -> dict[str, float]:
    """Compute accuracy + confusion matrix components per Furukawa Table 2."""
    valid = np.isfinite(predicted_is_fix) & np.isfinite(actual_is_fix)
    p = predicted_is_fix[valid].astype(bool)
    a = actual_is_fix[valid].astype(bool)
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "match_rate": float("nan"), "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    tp = int((p & a).sum())
    tn = int((~p & ~a).sum())
    fp = int((p & ~a).sum())
    fn = int((~p & a).sum())
    match = (tp + tn) / n
    return {"n": n, "match_rate": match, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def render_fix_comparison_map(city: str, run: str, pos: pd.DataFrame, pred_is_fix: np.ndarray, output_path: Path, pred_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    # Left: Simulation (predicted) FIX
    ax_sim = axes[0]
    ax_sim.scatter(pos["lon_deg"], pos["lat_deg"], c="#bdc3c7", s=2, alpha=0.3, label="route")
    fix_mask = pred_is_fix.astype(bool)
    if fix_mask.any():
        ax_sim.scatter(pos.loc[fix_mask, "lon_deg"], pos.loc[fix_mask, "lat_deg"],
                       c="#e74c3c", s=6, alpha=0.85, label=f"{pred_label} ({int(fix_mask.sum())})")
    ax_sim.set_title(f"{city} / {run} — RTK FIXED (Simulation)")
    ax_sim.set_xlabel("longitude (deg)")
    ax_sim.set_ylabel("latitude (deg)")
    ax_sim.legend(loc="best", fontsize=9, markerscale=3)
    ax_sim.grid(alpha=0.3)
    ax_sim.set_aspect("equal", adjustable="box")

    # Right: Measurement (actual demo5 Q=1) FIX
    ax_meas = axes[1]
    ax_meas.scatter(pos["lon_deg"], pos["lat_deg"], c="#bdc3c7", s=2, alpha=0.3, label="route")
    actual = (pos["q"] == 1).to_numpy()
    if actual.any():
        ax_meas.scatter(pos.loc[actual, "lon_deg"], pos.loc[actual, "lat_deg"],
                        c="#f1c40f", s=6, alpha=0.85, label=f"RTK FIXED (Measurement) ({int(actual.sum())})")
    ax_meas.set_title(f"{city} / {run} — RTK FIXED (Measurement)")
    ax_meas.set_xlabel("longitude (deg)")
    ax_meas.legend(loc="best", fontsize=9, markerscale=3)
    ax_meas.grid(alpha=0.3)
    ax_meas.set_aspect("equal", adjustable="box")

    fig.suptitle(f"{city} / {run} — Furukawa Fig.10-style: predicted vs observed RTK FIX", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Furukawa-2019-style paper evaluation")
    parser.add_argument("--epoch-pred-csv", type=Path, default=DEFAULT_EPOCH_PRED_CSV)
    parser.add_argument("--window-pred-csv", type=Path, default=DEFAULT_WINDOW_PRED_CSV)
    parser.add_argument("--pos-root", type=Path, default=DEFAULT_POS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DELIVERABLE_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--window-duration-s", type=float, default=30.0)
    parser.add_argument("--reference-threshold-pct", type=float, default=50.0,
                        help="threshold used for the Fig.10-style comparison maps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epoch_df = pd.read_csv(args.epoch_pred_csv)
    window_df = pd.read_csv(args.window_pred_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    per_run_rows: list[dict[str, object]] = []
    for city, run in RUN_ORDER:
        pos_path = args.pos_root / f"{city}_{run}" / "rtklib.pos"
        if not pos_path.exists():
            continue
        pos = _read_pos_file(pos_path)
        # Per-epoch classifier predictions; match by elapsed_s via nearest join.
        epoch_sub = _epoch_preds_for_run(epoch_df, city, run)
        # Join by elapsed_s with 0.5 s tolerance (epoch cadence 0.2 s, pos cadence 0.2 s).
        epoch_preds_at_pos = np.interp(pos["elapsed_s"].to_numpy(dtype=np.float64),
                                        epoch_sub["elapsed_s"].to_numpy(dtype=np.float64),
                                        epoch_sub["p_fix_pct"].to_numpy(dtype=np.float64),
                                        left=np.nan, right=np.nan)
        window_preds_at_pos = _window_preds_at_epoch(window_df, city, run,
                                                     pos["elapsed_s"].to_numpy(dtype=np.float64),
                                                     args.window_duration_s)
        actual = pos["is_fix_actual"].to_numpy(dtype=np.int64)

        # Threshold sweep for both predictors
        for thr in DEFAULT_THRESHOLDS:
            for model_name, preds in [("epoch_classifier", epoch_preds_at_pos),
                                      ("window_adopted_7p16", window_preds_at_pos)]:
                pred_is_fix = (preds >= thr).astype(int)
                m = _matching_rate(pred_is_fix, actual)
                rows.append({"city": city, "run": run, "model": model_name,
                              "threshold_pct": thr, **m})

        # Per-run summary at the reference threshold
        ref_thr = args.reference_threshold_pct
        for model_name, preds in [("epoch_classifier", epoch_preds_at_pos),
                                  ("window_adopted_7p16", window_preds_at_pos)]:
            pred_is_fix = (preds >= ref_thr).astype(int)
            m = _matching_rate(pred_is_fix, actual)
            per_run_rows.append({"city": city, "run": run, "model": model_name,
                                  "threshold_pct": ref_thr, **m})

        # Fig.10-style map at reference threshold using the adopted window predictor
        pred_ref = (window_preds_at_pos >= ref_thr).astype(int)
        map_path = args.plots_dir / f"{city}_{run}_fix_comparison_map.png"
        render_fix_comparison_map(city, run, pos, pred_ref, map_path,
                                   pred_label=f"RTK FIXED (Simulation, §7.16 ≥ {int(ref_thr)} %)")
        print(f"saved: {map_path.name}")

    sweep_df = pd.DataFrame(rows)
    per_run_df = pd.DataFrame(per_run_rows)
    sweep_path = args.output_dir / "paper_style_matching_rate.csv"
    per_run_path = args.output_dir / "paper_style_per_run_accuracy.csv"
    sweep_df.to_csv(sweep_path, index=False)
    per_run_df.to_csv(per_run_path, index=False)
    print(f"saved: {sweep_path}")
    print(f"saved: {per_run_path}")

    # Furukawa-Table-2-style pooled accuracy per threshold per model
    pooled = (sweep_df.groupby(["model", "threshold_pct"])
              .apply(lambda g: pd.Series({
                  "n": g["n"].sum(),
                  "match_rate_pct": 100.0 * (g["tp"].sum() + g["tn"].sum()) / max(g["n"].sum(), 1),
              }), include_groups=False)
              .reset_index())
    print("\nFurukawa-Table-2-style pooled matching rate across all 6 runs:")
    for model, sub in pooled.groupby("model"):
        print(f"\n  model = {model}")
        print(sub[["threshold_pct", "n", "match_rate_pct"]].to_string(index=False))
    pooled_path = args.output_dir / "paper_style_pooled_matching_rate.csv"
    pooled.to_csv(pooled_path, index=False)
    print(f"\nsaved: {pooled_path}")


if __name__ == "__main__":
    main()
