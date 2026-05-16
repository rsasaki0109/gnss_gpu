#!/usr/bin/env python3
"""Evaluate hybrid PU (libgnss_rtk_pos_v5) per-epoch err on n/r2.

Question: in the 1296 truly-lost epochs (all 47 candidates fail 50cm),
how good is hybrid PU itself? If hybrid PU passes some, then adding it
as a labeled candidate could recover them.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
REF_BASE = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")


def load_reference(city: str, run: str):
    out = {}
    path = REF_BASE / city / run / "reference.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                tow = round(float(r["GPS TOW (s)"]), 1)
                xyz = np.array([
                    float(r["ECEF X (m)"]),
                    float(r["ECEF Y (m)"]),
                    float(r["ECEF Z (m)"]),
                ])
                out[tow] = xyz
            except (ValueError, KeyError):
                continue
    return out


def load_pos(path):
    out = {}
    with open(path) as f:
        for ln in f:
            if ln.startswith("%") or not ln.strip():
                continue
            pp = ln.split()
            if len(pp) < 5:
                continue
            try:
                tow = round(float(pp[1]), 1)
                xyz = np.array([float(pp[2]), float(pp[3]), float(pp[4])])
                out[tow] = xyz
            except ValueError:
                continue
    return out


def main():
    ref = load_reference("nagoya", "run2")
    hybrid = load_pos(REPO / "experiments/results/libgnss_rtk_pos_v5/nagoya_run2_full.pos")
    print(f"reference epochs: {len(ref)}, hybrid epochs: {len(hybrid)}")

    truly_lost = pd.read_csv(REPO / "experiments/results/nr2_truly_lost_summary.csv")
    print(f"truly-lost epochs: {len(truly_lost)}")

    # Compute hybrid PU err for all reference epochs
    hyb_rows = []
    for tow, ref_xyz in ref.items():
        if tow not in hybrid:
            continue
        err = float(np.linalg.norm(hybrid[tow] - ref_xyz))
        hyb_rows.append({"tow": tow, "hybrid_err": err})
    hyb_df = pd.DataFrame(hyb_rows)
    print(f"hybrid epochs aligned to ref: {len(hyb_df)}")
    print(f"  hybrid pass@50cm overall: {(hyb_df['hybrid_err'] < 0.5).sum()} ({100*(hyb_df['hybrid_err']<0.5).mean():.1f}%)")
    print(f"  hybrid median err: {hyb_df['hybrid_err'].median():.3f}m")
    print(f"  hybrid 90%ile err: {hyb_df['hybrid_err'].quantile(0.9):.3f}m")

    # Merge with truly-lost
    merged = truly_lost.merge(hyb_df, on="tow", how="left")
    print(f"\n=== Hybrid PU on truly-lost epochs ===")
    print(f"  epochs covered: {merged['hybrid_err'].notna().sum()} / {len(merged)}")
    print(f"  hybrid passes 50cm: {(merged['hybrid_err'] < 0.5).sum()} (would recover)")
    print(f"  hybrid err distribution on truly-lost:")
    print(merged["hybrid_err"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

    # Compare hybrid vs best candidate
    print(f"\n=== Hybrid better than best candidate? (truly-lost only) ===")
    merged["hybrid_better"] = merged["hybrid_err"] < merged["best_err"]
    print(f"  hybrid better: {merged['hybrid_better'].sum()} / {len(merged)} ({100*merged['hybrid_better'].mean():.1f}%)")

    # Path-weighted impact if we recovered all hybrid-passing truly-lost epochs
    recover_path = merged[merged["hybrid_err"] < 0.5]["path_weight"].sum()
    total_path = 4699.7  # n/r2 total path
    print(f"\n=== Recovery potential ===")
    print(f"  Hybrid would recover {recover_path:.1f}m of truly-lost path")
    print(f"  As % of n/r2 total: +{100*recover_path/total_path:.2f}pp on n/r2")
    print(f"  OFFICIAL impact: +{100*recover_path/total_path/6:.3f}pp")


if __name__ == "__main__":
    main()
