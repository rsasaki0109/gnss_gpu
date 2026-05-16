#!/usr/bin/env python3
"""For each recoverable epoch on n/r2, compare ranker pick vs passing candidate
on the existing features. Find what discriminates them."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")


def main():
    feat = pd.read_csv(REPO / "experiments/results/selector_training_features.csv")
    nr2 = feat[feat["run_id"] == "nagoya_run2"].copy()
    recov = pd.read_csv(REPO / "experiments/results/nr2_ranker_recoverable_v1.csv")

    # For each recoverable epoch, get:
    # - the ranker pick features (one row)
    # - the BEST passing candidate features (one row, smallest err)
    pick_rows = []
    best_pass_rows = []
    for _, r in recov.iterrows():
        tow = r["tow"]
        epoch = nr2[nr2["tow"] == tow]
        pick = epoch[epoch["label"] == r["pick_label"]]
        passing = epoch[epoch["is_pass_50cm"] == 1]
        if pick.empty or passing.empty:
            continue
        best = passing.loc[passing["err_3d_m"].idxmin()]
        pick_rows.append(pick.iloc[0])
        best_pass_rows.append(best)

    pick_df = pd.DataFrame(pick_rows)
    best_df = pd.DataFrame(best_pass_rows)
    print(f"Recoverable pairs: {len(pick_df)}")

    cols = ["rms", "ratio", "abs_max", "sats", "status", "pdop",
            "candidate_vs_spp_m", "cluster_size_50cm",
            "rank_by_rms", "dist_to_median_m"]
    print(f"\n{'feature':<22} {'pick mean':>12} {'best mean':>12} {'delta':>10}")
    for c in cols:
        p_m = pick_df[c].mean()
        b_m = best_df[c].mean()
        print(f"{c:<22} {p_m:>12.3f} {b_m:>12.3f} {b_m - p_m:>+10.3f}")

    # Status mix in pick vs best
    print(f"\nstatus distribution:")
    print(f"  pick: {pick_df['status'].value_counts().to_dict()}")
    print(f"  best: {best_df['status'].value_counts().to_dict()}")

    # Cluster size distribution
    print(f"\ncluster_size_50cm distribution:")
    print(f"  pick mean={pick_df['cluster_size_50cm'].mean():.2f} median={pick_df['cluster_size_50cm'].median()}")
    print(f"  best mean={best_df['cluster_size_50cm'].mean():.2f} median={best_df['cluster_size_50cm'].median()}")
    print(f"  pick > best cluster: {(pick_df['cluster_size_50cm'].values > best_df['cluster_size_50cm'].values).mean()*100:.1f}%")
    print(f"  pick == best cluster: {(pick_df['cluster_size_50cm'].values == best_df['cluster_size_50cm'].values).mean()*100:.1f}%")
    print(f"  pick < best cluster: {(pick_df['cluster_size_50cm'].values < best_df['cluster_size_50cm'].values).mean()*100:.1f}%")

    # rank_by_rms — is the pick the lowest rms? Or different?
    print(f"\nrank_by_rms (1=lowest rms):")
    print(f"  pick mean rank: {pick_df['rank_by_rms'].mean():.2f}")
    print(f"  best mean rank: {best_df['rank_by_rms'].mean():.2f}")


if __name__ == "__main__":
    main()
