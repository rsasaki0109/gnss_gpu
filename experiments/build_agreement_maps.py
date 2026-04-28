#!/usr/bin/env python3
"""Improved spatial comparison between simulation predictions and actual demo5 FIX.

Two additions on top of build_paper_style_eval.py:

(1) Agreement map: a single lat/lon scatter per run with a 4-colour
    legend — TP (both FIX), TN (both not-FIX), FP (predicted FIX but
    actual not), FN (actual FIX but predicted not).  Shows spatially
    where the predictor agrees / disagrees with demo5.

(2) Hybrid prediction: instead of thresholding the §7.16 window FIX
    rate at a single value, allocate inside each window the top-K
    epochs by the experimental per-epoch P(FIX) rank, where K =
    round(window_rate / 100 * epoch_count_in_window).  This keeps the
    per-window FIX count honest to the adopted predictor while using
    the epoch classifier to decide *which* epochs are flagged.

Outputs under `internal_docs/product_deliverable/plots/`:

- `{city}_{run}_agreement_map.png` using the window-threshold
  predictor at 50 %.
- `{city}_{run}_agreement_map_hybrid.png` using the hybrid predictor.
- `{city}_{run}_fix_comparison_map_hybrid.png` — Furukawa Fig.10-style
  two-panel map using the hybrid predictor.
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

DEFAULT_WINDOW_PRED_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45_window_predictions.csv"
)
DEFAULT_EPOCH_PRED_CSV = RESULTS_DIR / "ppc_epoch_fix_classifier_predictions.csv"
DEFAULT_POS_ROOT = RESULTS_DIR / "demo5_pos"

RUN_ORDER = [
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
]

AGREEMENT_COLORS = {
    "TP": "#2ecc71",   # green: both FIX
    "TN": "#bdc3c7",   # light gray: both not-FIX
    "FP": "#e74c3c",   # red: predicted FIX, actual not
    "FN": "#f39c12",   # orange: missed actual FIX
}


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


def _window_rates(window_df: pd.DataFrame, city: str, run: str) -> pd.DataFrame:
    return window_df[(window_df["city"] == city) & (window_df["run"] == run)].sort_values("window_index").reset_index(drop=True)


def _epoch_pred(epoch_df: pd.DataFrame, city: str, run: str) -> pd.DataFrame:
    sub = epoch_df[(epoch_df["city"] == city) & (epoch_df["run"] == run)].copy()
    sub = sub.sort_values("gps_tow").reset_index(drop=True)
    sub["elapsed_s"] = sub["gps_tow"] - sub["gps_tow"].iloc[0]
    return sub


def _window_threshold_pred(pos_elapsed_s: np.ndarray, windows: pd.DataFrame, threshold_pct: float, window_duration_s: float) -> np.ndarray:
    pred = np.zeros(len(pos_elapsed_s), dtype=np.int64)
    for _, row in windows.iterrows():
        start = float(row["window_index"]) * window_duration_s
        end = start + window_duration_s
        mask = (pos_elapsed_s >= start) & (pos_elapsed_s < end)
        if float(row["corrected_pred_fix_rate_pct"]) >= threshold_pct:
            pred[mask] = 1
    return pred


def _hybrid_pred(pos: pd.DataFrame, windows: pd.DataFrame, epoch_pred: pd.DataFrame, window_duration_s: float) -> np.ndarray:
    """Within each window, flag the top-K epochs (K = window_rate * count / 100) by P(FIX) rank."""
    pos_elapsed = pos["elapsed_s"].to_numpy(dtype=np.float64)
    pred = np.zeros(len(pos_elapsed), dtype=np.int64)
    if epoch_pred.empty:
        return pred
    # Interpolate P(FIX) onto pos timeline
    p_fix = np.interp(pos_elapsed,
                      epoch_pred["elapsed_s"].to_numpy(dtype=np.float64),
                      epoch_pred["p_fix_pct"].to_numpy(dtype=np.float64),
                      left=np.nan, right=np.nan)
    for _, row in windows.iterrows():
        start = float(row["window_index"]) * window_duration_s
        end = start + window_duration_s
        mask = (pos_elapsed >= start) & (pos_elapsed < end)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        rate = float(row["corrected_pred_fix_rate_pct"]) / 100.0
        k = int(round(rate * indices.size))
        if k <= 0:
            continue
        local_p = p_fix[indices]
        # Replace NaN with very low so they are ranked last
        local_p = np.where(np.isfinite(local_p), local_p, -1.0)
        top_k_local = np.argsort(-local_p)[:k]
        pred[indices[top_k_local]] = 1
    return pred


def _agreement_tags(pred_is_fix: np.ndarray, actual_is_fix: np.ndarray) -> np.ndarray:
    """Return an array of 'TP' / 'TN' / 'FP' / 'FN' strings."""
    tags = np.empty(len(pred_is_fix), dtype=object)
    p = pred_is_fix.astype(bool)
    a = actual_is_fix.astype(bool)
    tags[p & a] = "TP"
    tags[~p & ~a] = "TN"
    tags[p & ~a] = "FP"
    tags[~p & a] = "FN"
    return tags


def render_agreement_map(city: str, run: str, pos: pd.DataFrame, tags: np.ndarray, title_suffix: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    counts: dict[str, int] = {}
    # Draw TN first so TP/FP/FN sit on top
    draw_order = ["TN", "TP", "FP", "FN"]
    for tag in draw_order:
        mask = tags == tag
        counts[tag] = int(mask.sum())
        if not mask.any():
            continue
        size = 2 if tag == "TN" else 6
        alpha = 0.3 if tag == "TN" else 0.85
        ax.scatter(pos.loc[mask, "lon_deg"], pos.loc[mask, "lat_deg"],
                   c=AGREEMENT_COLORS[tag], s=size, alpha=alpha, label=f"{tag} ({counts[tag]})")
    total = sum(counts.values()) or 1
    accuracy = 100.0 * (counts.get("TP", 0) + counts.get("TN", 0)) / total
    precision = (counts.get("TP", 0) / max(counts.get("TP", 0) + counts.get("FP", 0), 1)) * 100.0
    recall = (counts.get("TP", 0) / max(counts.get("TP", 0) + counts.get("FN", 0), 1)) * 100.0
    ax.set_title(
        f"{city} / {run} — agreement map {title_suffix}\n"
        f"accuracy={accuracy:.1f}%  precision={precision:.1f}%  recall={recall:.1f}%",
        fontsize=11,
    )
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.legend(loc="best", fontsize=9, markerscale=3)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def render_fig10_two_panel(city: str, run: str, pos: pd.DataFrame, pred_is_fix: np.ndarray, output_path: Path, pred_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax_sim, ax_meas = axes
    for ax in axes:
        ax.scatter(pos["lon_deg"], pos["lat_deg"], c="#bdc3c7", s=2, alpha=0.3, label="route")
        ax.set_xlabel("longitude (deg)")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    fix_mask = pred_is_fix.astype(bool)
    if fix_mask.any():
        ax_sim.scatter(pos.loc[fix_mask, "lon_deg"], pos.loc[fix_mask, "lat_deg"],
                       c="#e74c3c", s=6, alpha=0.85, label=f"{pred_label} ({int(fix_mask.sum())})")
    actual = (pos["q"] == 1).to_numpy()
    if actual.any():
        ax_meas.scatter(pos.loc[actual, "lon_deg"], pos.loc[actual, "lat_deg"],
                        c="#f1c40f", s=6, alpha=0.85, label=f"RTK FIXED (Measurement) ({int(actual.sum())})")
    ax_sim.set_title(f"{city} / {run} — RTK FIXED (Simulation, hybrid)")
    ax_meas.set_title(f"{city} / {run} — RTK FIXED (Measurement)")
    ax_sim.set_ylabel("latitude (deg)")
    ax_sim.legend(loc="best", fontsize=9, markerscale=3)
    ax_meas.legend(loc="best", fontsize=9, markerscale=3)
    fig.suptitle(f"{city} / {run} — hybrid (window rate × epoch P(FIX) rank)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agreement maps + hybrid prediction")
    parser.add_argument("--window-pred-csv", type=Path, default=DEFAULT_WINDOW_PRED_CSV)
    parser.add_argument("--epoch-pred-csv", type=Path, default=DEFAULT_EPOCH_PRED_CSV)
    parser.add_argument("--pos-root", type=Path, default=DEFAULT_POS_ROOT)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--deliverable-dir", type=Path, default=DELIVERABLE_DIR)
    parser.add_argument("--threshold-pct", type=float, default=50.0)
    parser.add_argument("--window-duration-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    args.deliverable_dir.mkdir(parents=True, exist_ok=True)
    window_df = pd.read_csv(args.window_pred_csv)
    epoch_df = pd.read_csv(args.epoch_pred_csv)

    summary_rows: list[dict[str, object]] = []
    for city, run in RUN_ORDER:
        pos_path = args.pos_root / f"{city}_{run}" / "rtklib.pos"
        if not pos_path.exists():
            continue
        pos = _read_pos_file(pos_path)
        windows = _window_rates(window_df, city, run)
        epoch_pred = _epoch_pred(epoch_df, city, run)
        actual = pos["is_fix_actual"].to_numpy(dtype=np.int64)
        pos_elapsed = pos["elapsed_s"].to_numpy(dtype=np.float64)

        # Baseline: window-threshold binary predictor
        pred_window = _window_threshold_pred(pos_elapsed, windows, args.threshold_pct, args.window_duration_s)
        tags_window = _agreement_tags(pred_window, actual)
        render_agreement_map(city, run, pos, tags_window,
                             title_suffix=f"(§7.16 window ≥ {int(args.threshold_pct)} %)",
                             output_path=args.plots_dir / f"{city}_{run}_agreement_map.png")

        # Hybrid: window rate × epoch P(FIX) rank
        pred_hybrid = _hybrid_pred(pos, windows, epoch_pred, args.window_duration_s)
        tags_hybrid = _agreement_tags(pred_hybrid, actual)
        render_agreement_map(city, run, pos, tags_hybrid,
                             title_suffix="(hybrid: window rate × epoch rank)",
                             output_path=args.plots_dir / f"{city}_{run}_agreement_map_hybrid.png")
        render_fig10_two_panel(city, run, pos, pred_hybrid,
                               output_path=args.plots_dir / f"{city}_{run}_fix_comparison_map_hybrid.png",
                               pred_label="RTK FIXED (Simulation, hybrid)")

        tp_w = int(((pred_window == 1) & (actual == 1)).sum())
        fp_w = int(((pred_window == 1) & (actual == 0)).sum())
        fn_w = int(((pred_window == 0) & (actual == 1)).sum())
        tp_h = int(((pred_hybrid == 1) & (actual == 1)).sum())
        fp_h = int(((pred_hybrid == 1) & (actual == 0)).sum())
        fn_h = int(((pred_hybrid == 0) & (actual == 1)).sum())
        actual_count = int(actual.sum())
        n = len(actual)
        summary_rows.append({
            "city": city,
            "run": run,
            "n_epochs": n,
            "actual_fix_epochs": actual_count,
            "window_tp": tp_w, "window_fp": fp_w, "window_fn": fn_w,
            "window_accuracy_pct": 100.0 * (tp_w + (n - tp_w - fp_w - fn_w)) / n,
            "window_precision_pct": 100.0 * tp_w / max(tp_w + fp_w, 1),
            "window_recall_pct": 100.0 * tp_w / max(tp_w + fn_w, 1),
            "hybrid_tp": tp_h, "hybrid_fp": fp_h, "hybrid_fn": fn_h,
            "hybrid_accuracy_pct": 100.0 * (tp_h + (n - tp_h - fp_h - fn_h)) / n,
            "hybrid_precision_pct": 100.0 * tp_h / max(tp_h + fp_h, 1),
            "hybrid_recall_pct": 100.0 * tp_h / max(tp_h + fn_h, 1),
        })
        print(f"{city}/{run}: window acc={summary_rows[-1]['window_accuracy_pct']:.1f}% prec={summary_rows[-1]['window_precision_pct']:.1f}% rec={summary_rows[-1]['window_recall_pct']:.1f}%  |  "
              f"hybrid acc={summary_rows[-1]['hybrid_accuracy_pct']:.1f}% prec={summary_rows[-1]['hybrid_precision_pct']:.1f}% rec={summary_rows[-1]['hybrid_recall_pct']:.1f}%")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.deliverable_dir / "agreement_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nsaved: {summary_path}")
    # Pooled metrics
    tot_n = summary_df["n_epochs"].sum()
    for prefix in ("window", "hybrid"):
        tp = summary_df[f"{prefix}_tp"].sum()
        fp = summary_df[f"{prefix}_fp"].sum()
        fn = summary_df[f"{prefix}_fn"].sum()
        tn = tot_n - tp - fp - fn
        acc = 100.0 * (tp + tn) / max(tot_n, 1)
        prec = 100.0 * tp / max(tp + fp, 1)
        rec = 100.0 * tp / max(tp + fn, 1)
        print(f"pooled {prefix}: acc={acc:.1f}% prec={prec:.1f}% rec={rec:.1f}% (TP={tp} FP={fp} FN={fn})")


if __name__ == "__main__":
    main()
