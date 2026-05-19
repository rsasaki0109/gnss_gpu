#!/usr/bin/env python3
"""Evaluate v3 ranker on each run, decide per-run best (v1 vs v3) and emit conditional CSV.

Path-weighted pass@50cm per run, comparing v1 predictions vs v3 predictions.
Picks the better one per run, then concatenates predictions.

Output: experiments/results/selector_ranker_predictions_best_of_v1v3.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
V1 = REPO / "experiments/results/selector_ranker_predictions.csv"
V3 = REPO / "experiments/results/selector_ranker_predictions_v3.csv"
OUT = REPO / "experiments/results/selector_ranker_predictions_best_of_v1v3.csv"


def per_run_pw_pass(pred: pd.DataFrame) -> dict[str, float]:
    out = {}
    for run_id, g in pred.groupby("run_id"):
        pick = g.loc[g.groupby("tow")["p_pass"].idxmax(),
                     ["tow", "is_pass_50cm", "path_weight"]]
        pw_pass = (pick["is_pass_50cm"] * pick["path_weight"]).sum()
        pw_total = pick["path_weight"].sum()
        out[run_id] = pw_pass / pw_total if pw_total > 0 else 0.0
    return out


def main():
    v1 = pd.read_csv(V1)
    v3 = pd.read_csv(V3)

    p1 = per_run_pw_pass(v1)
    p3 = per_run_pw_pass(v3)

    print(f"{'run':<14} {'v1':>10} {'v3':>10} {'delta':>10} {'pick':>6}")
    pick_v3 = []
    for run_id in sorted(p1.keys()):
        a, b = p1[run_id], p3[run_id]
        choice = "v3" if b > a else "v1"
        if b > a:
            pick_v3.append(run_id)
        print(f"{run_id:<14} {a*100:>9.4f}% {b*100:>9.4f}% {(b-a)*100:>+9.4f}pp {choice:>6}")

    avg_v1 = np.mean(list(p1.values()))
    avg_v3 = np.mean(list(p3.values()))
    avg_best = np.mean([max(p1[r], p3[r]) for r in p1])
    print(f"\nOFFICIAL: v1 {avg_v1*100:.4f}% | v3 {avg_v3*100:.4f}% | best-of {avg_best*100:.4f}%")
    print(f"          v3-v1 = {(avg_v3-avg_v1)*100:+.4f}pp | best-v1 = {(avg_best-avg_v1)*100:+.4f}pp")

    # Build conditional CSV using v3 for pick_v3 runs, v1 otherwise
    v1_part = v1[~v1["run_id"].isin(pick_v3)]
    v3_part = v3[v3["run_id"].isin(pick_v3)]
    merged = pd.concat([v1_part, v3_part], ignore_index=True)
    merged = merged.sort_values(["run_id", "tow", "label"]).reset_index(drop=True)
    merged.to_csv(OUT, index=False)
    print(f"\nSaved best-of: {OUT}")
    print(f"  v3 used for: {pick_v3}")


if __name__ == "__main__":
    main()
