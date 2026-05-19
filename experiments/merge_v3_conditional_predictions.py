#!/usr/bin/env python3
"""Per-run-conditional merge of v3 ranker predictions.

For runs in CONDITIONAL_RUNS (typically nagoya_run2), use v3 predictions.
For other runs, use the v1 baseline (per-epoch argmax).

Output: experiments/results/selector_ranker_predictions_conditional_nr2v3.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
BASE = REPO / "experiments/results/selector_ranker_predictions.csv"
V3 = REPO / "experiments/results/selector_ranker_predictions_v3.csv"
OUT = REPO / "experiments/results/selector_ranker_predictions_conditional_nr2v3.csv"

CONDITIONAL_RUNS = {"nagoya_run2"}


def main():
    print(f"Loading base {BASE}")
    base = pd.read_csv(BASE)
    print(f"  rows: {len(base)}")
    print(f"Loading v3 {V3}")
    v3 = pd.read_csv(V3)
    print(f"  rows: {len(v3)}")

    base_other = base[~base["run_id"].isin(CONDITIONAL_RUNS)]
    v3_cond = v3[v3["run_id"].isin(CONDITIONAL_RUNS)]
    print(f"  base (non-conditional): {len(base_other)} rows")
    print(f"  v3 (conditional):  {len(v3_cond)} rows")

    merged = pd.concat([base_other, v3_cond], ignore_index=True)
    merged = merged.sort_values(["run_id", "tow", "label"]).reset_index(drop=True)
    merged.to_csv(OUT, index=False)
    print(f"Saved: {OUT}")
    print(f"  total rows: {len(merged)}")
    for run_id in sorted(merged["run_id"].unique()):
        n = (merged["run_id"] == run_id).sum()
        src = "V3" if run_id in CONDITIONAL_RUNS else "base"
        print(f"  {run_id}: {n} rows ({src})")


if __name__ == "__main__":
    main()
