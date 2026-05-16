#!/usr/bin/env python3
"""Characterize n/r2 truly-lost epochs: those where NO candidate in pool passes 50cm.

Output: per-epoch summary with:
- # candidates available
- best (smallest) err_3d_m
- pool composition at that epoch (how many xd_ vs base)
- min residual_rms across pool
- pdop / status mix

Goal: identify what kind of new candidate (libgnss++ config) might recover these.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")


def main():
    feat = pd.read_csv(REPO / "experiments/results/selector_training_features.csv")
    nr2 = feat[feat["run_id"] == "nagoya_run2"].copy()
    print(f"nagoya_run2 rows: {len(nr2)}, epochs: {nr2['tow'].nunique()}")

    # Per-epoch: any pass?
    epoch_summary = nr2.groupby("tow").agg(
        n_candidates=("label", "count"),
        any_pass=("is_pass_50cm", "max"),
        best_err=("err_3d_m", "min"),
        best_rms=("rms", "min"),
        mean_pdop=("pdop", "mean"),
        any_status_5=("status", lambda x: int((x == 5).any())),
        any_status_4=("status", lambda x: int((x == 4).any())),
        path_weight=("path_weight", "first"),
    ).reset_index()

    pass_count = (epoch_summary["any_pass"] == 1).sum()
    truly_lost = epoch_summary[epoch_summary["any_pass"] == 0]
    print(f"  Epochs with at least one passing candidate: {pass_count} / {len(epoch_summary)} ({100*pass_count/len(epoch_summary):.1f}%)")
    print(f"  Truly-lost epochs: {len(truly_lost)} ({100*len(truly_lost)/len(epoch_summary):.1f}%)")

    print(f"\n=== Truly-lost epochs characteristics ===")
    print(f"  best_err_3d_m distribution:")
    print(truly_lost["best_err"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

    print(f"\n  best_rms (RTK residual) distribution:")
    print(truly_lost["best_rms"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

    print(f"\n  PDOP distribution:")
    print(truly_lost["mean_pdop"].describe(percentiles=[0.5, 0.9, 0.99]))

    print(f"\n  Status presence in truly-lost epochs:")
    print(f"    any status=5 (fix): {truly_lost['any_status_5'].sum()} / {len(truly_lost)}")
    print(f"    any status=4 (float): {truly_lost['any_status_4'].sum()} / {len(truly_lost)}")

    # How does best_err scale with rms? Are truly-lost epochs all bad RTK or some are surprisingly OK?
    print(f"\n=== best_err vs best_rms bins (truly-lost) ===")
    truly_lost["err_bin"] = pd.cut(truly_lost["best_err"], bins=[0.5, 1.0, 2.0, 5.0, 20.0, 1000.0])
    truly_lost["rms_bin"] = pd.cut(truly_lost["best_rms"], bins=[0, 0.1, 0.3, 1.0, 5.0, 100.0])
    print(pd.crosstab(truly_lost["err_bin"], truly_lost["rms_bin"]))

    # Are best_err >= 1m or close to 0.5m? (close = chance of recovery with small improvement)
    near_miss = truly_lost[truly_lost["best_err"] < 1.0]
    print(f"\n  Near-miss truly-lost (best_err < 1.0m): {len(near_miss)} ({100*len(near_miss)/len(truly_lost):.1f}%)")

    # Path-weighted impact
    pw_lost = (truly_lost["path_weight"]).sum()
    pw_total = (epoch_summary["path_weight"]).sum()
    pw_pass = epoch_summary[epoch_summary["any_pass"]==1]["path_weight"].sum()
    print(f"\n=== Path-weighted impact ===")
    print(f"  Truly-lost path: {pw_lost:.1f}m ({100*pw_lost/pw_total:.1f}% of total path {pw_total:.1f}m)")
    print(f"  Oracle ceiling = pw_pass / pw_total = {100*pw_pass/pw_total:.2f}%")
    print(f"  Phase 26 conditional = 63.13% → gap to oracle = {100*pw_pass/pw_total - 63.13:.2f}pp")

    # Save
    out = REPO / "experiments/results/nr2_truly_lost_summary.csv"
    truly_lost.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
