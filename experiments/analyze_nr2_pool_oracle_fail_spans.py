#!/usr/bin/env python3
"""Summarize nagoya/run2 spans where the current candidate pool has no pass."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
RESULTS = REPO / "experiments/results"
FEATURES = RESULTS / "selector_training_features_v5_nlos.csv"
OUT_CSV = RESULTS / "nr2_current_pool_oracle_fail_spans.csv"

USECOLS = [
    "run_id",
    "tow",
    "label",
    "path_weight",
    "err_3d_m",
    "nlos_frac",
    "sats",
    "cluster_size_25cm",
    "dist_to_median_m",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--run-id", default="nagoya_run2")
    parser.add_argument("--gap-s", type=float, default=0.21)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.features, usecols=USECOLS)
    df = df[df["run_id"] == args.run_id].copy()
    df["tow"] = df["tow"].round(1)

    idx = df.groupby("tow")["err_3d_m"].idxmin()
    best = df.loc[idx].sort_values("tow").reset_index(drop=True)
    fail = best[best["err_3d_m"] >= 0.5].reset_index(drop=True)

    epoch = df.groupby("tow").agg(
        path_weight=("path_weight", "first"),
        any_pass=("err_3d_m", lambda s: bool((s < 0.5).any())),
        best_err=("err_3d_m", "min"),
    )
    total_w = float(epoch["path_weight"].sum())
    oracle_pass_w = float(epoch.loc[epoch["any_pass"], "path_weight"].sum())
    oracle_fail_w = total_w - oracle_pass_w

    spans: list[tuple[int, int]] = []
    if not fail.empty:
        start = 0
        for i in range(1, len(fail)):
            if float(fail.loc[i, "tow"] - fail.loc[i - 1, "tow"]) > args.gap_s:
                spans.append((start, i))
                start = i
        spans.append((start, len(fail)))

    rows: list[dict[str, object]] = []
    for start, end in spans:
        span = fail.iloc[start:end]
        rows.append(
            {
                "start_tow": float(span["tow"].iloc[0]),
                "end_tow": float(span["tow"].iloc[-1]),
                "n": int(len(span)),
                "path_w": float(span["path_weight"].sum()),
                "best_label_mode": str(span["label"].mode().iat[0]),
                "med_best_err": float(span["err_3d_m"].median()),
                "min_best_err": float(span["err_3d_m"].min()),
                "med_nlos": float(span["nlos_frac"].median()),
                "med_sats": float(span["sats"].median()),
                "med_c25": float(span["cluster_size_25cm"].median()),
                "med_dist_med": float(span["dist_to_median_m"].median()),
            }
        )

    out = pd.DataFrame(rows).sort_values("path_w", ascending=False)
    out.to_csv(args.out_csv, index=False)
    oracle_pct = 100.0 * oracle_pass_w / total_w if total_w > 0 else 0.0
    fail_pct = 100.0 * oracle_fail_w / total_w if total_w > 0 else 0.0
    print(
        f"oracle_path_pct={oracle_pct:.4f} oracle_fail_path_pct={fail_pct:.4f} "
        f"fail_spans={len(out)}",
        flush=True,
    )
    if not out.empty:
        print(out.head(20).to_string(index=False), flush=True)
    print(f"wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
