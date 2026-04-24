#!/usr/bin/env python3
"""Render side-by-side visualisations of simulation-predicted FIX rate vs
the per-epoch actual demo5 FIX / NO-FIX trajectory.

For each (city, run) pair:

- Time-series plot: predicted FIX rate (window-level step function, §7.16)
  overlaid with the observed per-epoch Q quality flag from the demo5 .pos
  file, plus a short-window rolling fraction of FIX epochs.
- Trajectory plot: lat/lon scatter with marker colour indicating
  per-epoch Q (green=fix, orange=float, gray=other) plus predicted
  per-window FIX rate annotated at window start points.
- A 2x3 summary grid collects all runs in one figure.

Outputs are saved under `internal_docs/product_deliverable/plots/`.
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

DEFAULT_PRED_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45_window_predictions.csv"
)

RUN_ORDER = [
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
]

DEFAULT_EPOCH_PRED_CSV = RESULTS_DIR / "ppc_epoch_fix_classifier_predictions.csv"

Q_COLOR = {
    1: "#2ecc71",  # FIX (green)
    2: "#f39c12",  # FLOAT (orange)
    3: "#3498db",  # SBAS
    4: "#9b59b6",  # DGPS
    5: "#95a5a6",  # single (gray)
    6: "#1abc9c",  # PPP
}
Q_LABEL = {1: "FIX", 2: "FLOAT", 3: "SBAS", 4: "DGPS", 5: "single", 6: "PPP"}


def _read_pos_file(path: Path) -> pd.DataFrame:
    """Parse an RTKLIB demo5 .pos text file into a DataFrame."""
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
    if not rows:
        raise ValueError(f"no data rows in {path}")
    df = pd.DataFrame(rows)
    df.columns = [
        "date", "time", "lat_deg", "lon_deg", "height_m", "q", "ns",
        "sdn", "sde", "sdu", "sdne", "sdeu", "sdun", "age", "ratio",
    ][: df.shape[1]]
    df["lat_deg"] = df["lat_deg"].astype(float)
    df["lon_deg"] = df["lon_deg"].astype(float)
    df["height_m"] = df["height_m"].astype(float)
    df["q"] = df["q"].astype(int)
    ts = pd.to_datetime(df["date"] + " " + df["time"], format="%Y/%m/%d %H:%M:%S.%f")
    df["ts"] = ts
    df["elapsed_s"] = (ts - ts.iloc[0]).dt.total_seconds()
    return df[["ts", "elapsed_s", "lat_deg", "lon_deg", "height_m", "q"]]


def _window_for_run(pred_df: pd.DataFrame, city: str, run: str) -> pd.DataFrame:
    mask = (pred_df["city"] == city) & (pred_df["run"] == run)
    cols = ["window_index", "actual_fix_rate_pct", "base_pred_fix_rate_pct", "corrected_pred_fix_rate_pct"]
    available = [c for c in cols if c in pred_df.columns]
    return pred_df.loc[mask, available].sort_values("window_index").reset_index(drop=True)


def _run_windows_elapsed(windows: pd.DataFrame, pos: pd.DataFrame, window_duration_s: float = 30.0) -> pd.DataFrame:
    """Compute elapsed_s at the start of each window using uniform 30 s spacing from the pos start."""
    w = windows.copy()
    w["window_start_s"] = w["window_index"] * window_duration_s
    w["window_end_s"] = (w["window_index"] + 1) * window_duration_s
    max_elapsed = pos["elapsed_s"].iloc[-1] if len(pos) else 0.0
    w["window_start_s"] = w["window_start_s"].clip(upper=max_elapsed)
    w["window_end_s"] = w["window_end_s"].clip(upper=max_elapsed)
    return w


def _rolling_fix_fraction(q_series: pd.Series, window: int = 50) -> pd.Series:
    is_fix = (q_series == 1).astype(float)
    return is_fix.rolling(window=window, center=True, min_periods=max(5, window // 4)).mean() * 100.0


def render_time_series(city: str, run: str, pos: pd.DataFrame, windows: pd.DataFrame,
                       epoch_pred: pd.DataFrame | None, output_path: Path) -> None:
    fig, (ax_pred, ax_q) = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})

    ax_pred.step(windows["window_start_s"], windows["corrected_pred_fix_rate_pct"], where="post",
                 color="#e74c3c", linewidth=2.0, label="predicted window FIX % (§7.16)")
    ax_pred.step(windows["window_start_s"], windows["actual_fix_rate_pct"], where="post",
                 color="#34495e", linewidth=1.5, linestyle="--", label="actual window FIX %")
    roll = _rolling_fix_fraction(pos["q"], window=75)
    ax_pred.plot(pos["elapsed_s"], roll, color="#2ecc71", linewidth=1.0, alpha=0.7,
                 label="actual FIX fraction (15 s rolling)")
    if epoch_pred is not None and not epoch_pred.empty:
        ax_pred.plot(epoch_pred["elapsed_s"], epoch_pred["p_fix_smoothed_pct"],
                     color="#2980b9", linewidth=1.2, alpha=0.9,
                     label="predicted epoch P(FIX) (15 s rolling mean)")
    ax_pred.set_ylim(-5, 105)
    ax_pred.set_ylabel("FIX rate (%)")
    ax_pred.set_title(f"{city} / {run} — predicted vs actual FIX rate")
    ax_pred.legend(loc="upper right", fontsize=8, ncol=2)
    ax_pred.grid(alpha=0.3)

    # Per-epoch quality flag as colored band
    for q_val, color in Q_COLOR.items():
        mask = pos["q"] == q_val
        if mask.any():
            ax_q.scatter(pos.loc[mask, "elapsed_s"], np.full(int(mask.sum()), q_val),
                         c=color, s=4, alpha=0.9, label=Q_LABEL.get(q_val, f"Q={q_val}"))
    ax_q.set_yticks(list(Q_COLOR))
    ax_q.set_yticklabels([Q_LABEL.get(q, str(q)) for q in Q_COLOR])
    ax_q.set_xlabel("elapsed seconds from run start")
    ax_q.set_ylabel("demo5 Q")
    ax_q.grid(alpha=0.3)
    ax_q.legend(loc="upper right", fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def render_trajectory(city: str, run: str, pos: pd.DataFrame, windows: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for q_val, color in Q_COLOR.items():
        mask = pos["q"] == q_val
        if mask.any():
            ax.scatter(pos.loc[mask, "lon_deg"], pos.loc[mask, "lat_deg"],
                       c=color, s=3, alpha=0.8, label=f"{Q_LABEL.get(q_val, q_val)} ({int(mask.sum())})")
    # Annotate window-level predicted rate at the pos-index corresponding to each window start
    if len(pos):
        t0 = pos["elapsed_s"].iloc[0]
        for _, row in windows.iterrows():
            target_s = row["window_start_s"] + t0
            idx = (pos["elapsed_s"] - target_s).abs().idxmin()
            lat = pos.loc[idx, "lat_deg"]
            lon = pos.loc[idx, "lon_deg"]
            ax.plot(lon, lat, marker="o", markersize=6, markerfacecolor="none",
                    markeredgecolor="#e74c3c", markeredgewidth=1.2)
            ax.annotate(f"{int(row['corrected_pred_fix_rate_pct'])}",
                        xy=(lon, lat), xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color="#c0392b")
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title(f"{city} / {run} — trajectory coloured by demo5 Q (numbers = predicted FIX % per window)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, markerscale=3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def render_summary_grid(runs: list[tuple[str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
    axes = axes.ravel()
    for ax, (city, run, pos, windows, epoch_pred) in zip(axes, runs):
        ax.step(windows["window_start_s"], windows["corrected_pred_fix_rate_pct"], where="post",
                color="#e74c3c", linewidth=1.8, label="predicted window (§7.16)")
        ax.step(windows["window_start_s"], windows["actual_fix_rate_pct"], where="post",
                color="#34495e", linewidth=1.3, linestyle="--", label="actual window")
        roll = _rolling_fix_fraction(pos["q"], window=75)
        ax.plot(pos["elapsed_s"], roll, color="#2ecc71", linewidth=0.8, alpha=0.7, label="actual rolling FIX %")
        if epoch_pred is not None and not epoch_pred.empty:
            ax.plot(epoch_pred["elapsed_s"], epoch_pred["p_fix_smoothed_pct"],
                    color="#2980b9", linewidth=0.9, alpha=0.9, label="predicted epoch P(FIX)")
        ax.set_title(f"{city} / {run}", fontsize=11)
        ax.set_xlabel("elapsed (s)")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("FIX rate (%)")
    axes[3].set_ylabel("FIX rate (%)")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("PPC demo5 FIX-rate — predicted (§7.16 window, §epoch classifier) vs actual, all 6 runs", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def _epoch_pred_for_run(epoch_pred_df: pd.DataFrame | None, city: str, run: str, pos: pd.DataFrame) -> pd.DataFrame | None:
    if epoch_pred_df is None or epoch_pred_df.empty or len(pos) == 0:
        return None
    mask = (epoch_pred_df["city"] == city) & (epoch_pred_df["run"] == run)
    sub = epoch_pred_df.loc[mask].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("gps_tow").reset_index(drop=True)
    t0 = sub["gps_tow"].iloc[0]
    sub["elapsed_s"] = sub["gps_tow"] - t0
    # 15 s rolling mean (epoch cadence = 0.2 s -> 75 epochs ~= 15 s)
    sub["p_fix_smoothed_pct"] = sub["p_fix_pct"].rolling(window=75, center=True, min_periods=10).mean()
    return sub[["elapsed_s", "p_fix_pct", "p_fix_smoothed_pct"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render simulation vs actual FIX trajectory plots")
    parser.add_argument("--prediction-csv", type=Path, default=DEFAULT_PRED_CSV)
    parser.add_argument("--pos-root", type=Path, default=RESULTS_DIR / "demo5_pos",
                        help="directory containing `<city>_<run>/rtklib.pos` subfolders")
    parser.add_argument("--epoch-pred-csv", type=Path, default=DEFAULT_EPOCH_PRED_CSV,
                        help="optional per-epoch P(FIX) predictions from train_ppc_epoch_fix_classifier.py")
    parser.add_argument("--output-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--window-duration-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.read_csv(args.prediction_csv)
    epoch_pred_df: pd.DataFrame | None = None
    if args.epoch_pred_csv and Path(args.epoch_pred_csv).exists():
        epoch_pred_df = pd.read_csv(args.epoch_pred_csv)
        print(f"loaded epoch predictions: {len(epoch_pred_df)} rows from {args.epoch_pred_csv.name}")
    else:
        print(f"note: epoch prediction CSV not found at {args.epoch_pred_csv}; plots will omit per-epoch P(FIX) line")

    cached: list[tuple[str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]] = []
    for city, run in RUN_ORDER:
        pos_path = args.pos_root / f"{city}_{run}" / "rtklib.pos"
        if not pos_path.exists():
            print(f"skip: {pos_path} not found")
            continue
        pos = _read_pos_file(pos_path)
        windows = _window_for_run(pred_df, city, run)
        if windows.empty:
            print(f"skip: no predictions for {city}/{run}")
            continue
        windows = _run_windows_elapsed(windows, pos, args.window_duration_s)
        epoch_pred = _epoch_pred_for_run(epoch_pred_df, city, run, pos)

        ts_path = args.output_dir / f"{city}_{run}_timeseries.png"
        traj_path = args.output_dir / f"{city}_{run}_trajectory.png"
        render_time_series(city, run, pos, windows, epoch_pred, ts_path)
        render_trajectory(city, run, pos, windows, traj_path)
        print(f"saved: {ts_path.name}, {traj_path.name}")
        cached.append((city, run, pos, windows, epoch_pred))

    if cached:
        summary_path = args.output_dir / "summary_grid.png"
        render_summary_grid(cached, summary_path)
        print(f"saved: {summary_path.name}")


if __name__ == "__main__":
    main()
