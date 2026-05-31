#!/usr/bin/env python3
"""Diagnose nagoya/run2 ranker wrong picks against pass candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
RESULTS = REPO / "experiments/results"
RUN_ID = "nagoya_run2"

FEATURES = RESULTS / "selector_training_features_v5_nlos.csv"
PREDICTIONS = RESULTS / "selector_ranker_predictions_v3.csv"
OUT_WRONGS = RESULTS / "nr2_phase29_wrongpicks_v3_vs_pass.csv"

KEEP_COLS = [
    "run_id", "tow", "label",
    "rms", "ratio", "abs_max", "update_rows", "sats", "status",
    "pdop", "baseline_m", "spp_valid", "spp_sats", "spp_pdop",
    "candidate_vs_spp_m", "candidate_jump_m",
    "cluster_size_50cm", "cluster_size_25cm", "cluster_size_10cm",
    "max_cluster_size_50cm", "is_in_max_cluster_50cm",
    "n_clusters_50cm", "n_clusters_50cm_ge3",
    "cluster_min_rms_50cm", "cluster_min_abs_max_50cm",
    "dist_to_max_cluster_centroid_m",
    "delta_pos_norm_m", "delta_pos_vs_median_m",
    "rank_by_rms", "n_candidates_in_epoch", "dist_to_median_m",
    "nlos_n_sats", "nlos_count", "nlos_los_count", "nlos_frac",
    "nlos_min_elev_deg", "nlos_mean_elev_deg",
    "err_3d_m", "is_pass_50cm", "path_weight",
]

COMPARE_COLS = [
    "rms", "ratio", "abs_max", "update_rows", "sats", "status", "pdop",
    "candidate_vs_spp_m", "candidate_jump_m",
    "cluster_size_50cm", "cluster_size_25cm", "cluster_size_10cm",
    "max_cluster_size_50cm", "is_in_max_cluster_50cm", "n_clusters_50cm",
    "cluster_min_rms_50cm", "cluster_min_abs_max_50cm",
    "dist_to_max_cluster_centroid_m", "delta_pos_norm_m",
    "delta_pos_vs_median_m", "rank_by_rms", "dist_to_median_m",
    "nlos_count", "nlos_frac", "nlos_min_elev_deg", "nlos_mean_elev_deg",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--out-csv", type=Path, default=OUT_WRONGS)
    parser.add_argument("--title", default="n/r2 wrong-pick summary")
    return parser.parse_args()


def _load_run_frame(features: Path, predictions: Path, run_id: str) -> pd.DataFrame:
    print(f"loading features: {features}", flush=True)
    df = pd.read_csv(features, usecols=KEEP_COLS)
    df = df[df["run_id"] == run_id].copy()
    df["tow"] = df["tow"].round(1)
    print(f"  feature rows={len(df)} epochs={df['tow'].nunique()}", flush=True)

    print(f"loading predictions: {predictions}", flush=True)
    pred = pd.read_csv(predictions, usecols=["run_id", "tow", "label", "p_pass"])
    pred = pred[pred["run_id"] == run_id].copy()
    pred["tow"] = pred["tow"].round(1)
    print(f"  pred rows={len(pred)} epochs={pred['tow'].nunique()}", flush=True)

    merged = df.merge(pred, on=["run_id", "tow", "label"], how="left")
    missing = int(merged["p_pass"].isna().sum())
    if missing:
        print(f"  missing p_pass rows={missing}; filling 0", flush=True)
        merged["p_pass"] = merged["p_pass"].fillna(0.0)
    return merged


def _diagnose(df: pd.DataFrame, title: str) -> pd.DataFrame:
    wrong_rows: list[dict[str, object]] = []
    total_epochs = pick_pass = recoverable = 0
    path_total = path_pick_pass = path_recoverable = 0.0

    for tow, group in df.groupby("tow", sort=True):
        total_epochs += 1
        pick = group.loc[group["p_pass"].idxmax()]
        weight = float(pick["path_weight"])
        path_total += weight
        if int(pick["is_pass_50cm"]) == 1:
            pick_pass += 1
            path_pick_pass += weight
            continue

        passing = group[group["is_pass_50cm"] == 1]
        if passing.empty:
            continue
        recoverable += 1
        path_recoverable += weight
        best = passing.loc[passing["err_3d_m"].idxmin()]
        row: dict[str, object] = {
            "tow": float(tow),
            "path_weight": weight,
            "pick_label": str(pick["label"]),
            "best_label": str(best["label"]),
            "pick_err_3d_m": float(pick["err_3d_m"]),
            "best_err_3d_m": float(best["err_3d_m"]),
            "err_delta_m": float(pick["err_3d_m"] - best["err_3d_m"]),
            "pick_p_pass": float(pick["p_pass"]),
            "best_p_pass": float(best["p_pass"]),
            "p_pass_delta_best_minus_pick": float(best["p_pass"] - pick["p_pass"]),
        }
        for col in COMPARE_COLS:
            row[f"pick_{col}"] = float(pick[col])
            row[f"best_{col}"] = float(best[col])
            row[f"delta_{col}"] = float(best[col] - pick[col])
        wrong_rows.append(row)

    print()
    print(f"========== {title} ==========")
    print(f"epochs={total_epochs} pick_pass={pick_pass} recoverable_wrong={recoverable}")
    print(f"path_total={path_total:.3f} pick_pass_w={path_pick_pass:.3f} recoverable_w={path_recoverable:.3f}")
    if path_total > 0:
        print(f"pick_pass_path_pct={100.0 * path_pick_pass / path_total:.4f}")
        print(f"recoverable_wrong_path_pct={100.0 * path_recoverable / path_total:.4f}")

    out = pd.DataFrame(wrong_rows)
    if out.empty:
        return out

    print()
    print("top wrong pick labels:")
    for label, count in Counter(out["pick_label"]).most_common(20):
        loss = out.loc[out["pick_label"] == label, "path_weight"].sum()
        print(f"  {label:28s} n={count:5d} path_w={loss:9.3f}")

    print()
    print("top best/pass labels ranker missed:")
    for label, count in Counter(out["best_label"]).most_common(20):
        gain = out.loc[out["best_label"] == label, "path_weight"].sum()
        print(f"  {label:28s} n={count:5d} path_w={gain:9.3f}")

    print()
    print("feature deltas on recoverable wrongs (best - pick):")
    for col in COMPARE_COLS:
        values = out[f"delta_{col}"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        print(
            f"  {col:34s} mean={values.mean():+10.4f} "
            f"med={np.median(values):+10.4f} pos%={100.0 * (values > 0).mean():6.2f}"
        )

    print()
    print("NLOS fraction bins for recoverable wrongs:")
    bins = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 1.01)]
    for lo, hi in bins:
        b = out[(out["pick_nlos_frac"] >= lo) & (out["pick_nlos_frac"] < hi)]
        if b.empty:
            continue
        print(
            f"  [{lo:.2f},{hi:.2f}) n={len(b):5d} "
            f"path_w={b['path_weight'].sum():9.3f} "
            f"median_pick_err={b['pick_err_3d_m'].median():8.3f} "
            f"top_pick={b['pick_label'].mode().iat[0]}"
        )

    return out


def main() -> None:
    args = _parse_args()
    df = _load_run_frame(args.features, args.predictions, args.run_id)
    wrongs = _diagnose(df, args.title)
    if not wrongs.empty:
        wrongs.to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv} rows={len(wrongs)}", flush=True)


if __name__ == "__main__":
    main()
