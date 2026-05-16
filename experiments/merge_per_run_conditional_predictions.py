#!/usr/bin/env python3
"""Per-run-conditional merge of ranker predictions.

For runs in CONDITIONAL_RUNS, use viterbi predictions (sequence smoothed).
For other runs, use the v1 baseline (per-epoch argmax).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
BASE = REPO / "experiments/results/selector_ranker_predictions.csv"
VITERBI = REPO / "experiments/results/selector_ranker_predictions_viterbi_a05.csv"
OUT = REPO / "experiments/results/selector_ranker_predictions_conditional_nr2vit.csv"

CONDITIONAL_RUNS = {"nagoya_run2"}


def main():
    print(f"Loading base {BASE}")
    base = pd.read_csv(BASE)
    print(f"  rows: {len(base)}")
    print(f"Loading viterbi {VITERBI}")
    vit = pd.read_csv(VITERBI)
    print(f"  rows: {len(vit)}")

    base_other = base[~base["run_id"].isin(CONDITIONAL_RUNS)]
    vit_cond = vit[vit["run_id"].isin(CONDITIONAL_RUNS)]
    print(f"  base (non-conditional): {len(base_other)} rows")
    print(f"  viterbi (conditional):  {len(vit_cond)} rows")

    merged = pd.concat([base_other, vit_cond], ignore_index=True)
    merged = merged.sort_values(["run_id", "tow", "label"]).reset_index(drop=True)
    merged.to_csv(OUT, index=False)
    print(f"Saved: {OUT}")
    print(f"  total rows: {len(merged)}")
    for run_id in sorted(merged["run_id"].unique()):
        n = (merged["run_id"] == run_id).sum()
        src = "VITERBI" if run_id in CONDITIONAL_RUNS else "base"
        print(f"  {run_id}: {n} rows ({src})")


if __name__ == "__main__":
    main()
