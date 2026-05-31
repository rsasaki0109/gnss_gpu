#!/usr/bin/env python3
"""Build a per-run selector prediction blend: v3 default, v5_nlos for n/r3."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
RESULTS = REPO / "experiments/results"
V3 = RESULTS / "selector_ranker_predictions_v3.csv"
V5_NLOS = RESULTS / "selector_ranker_predictions_v5_nlos.csv"
OUT = RESULTS / "selector_ranker_predictions_best_v3_v5_nlos.csv"

RUN_TO_SOURCE = {
    "nagoya_run3": V5_NLOS,
}


def main() -> None:
    v3 = pd.read_csv(V3)
    v5 = pd.read_csv(V5_NLOS)
    frames = []
    for run_id in sorted(v3["run_id"].unique()):
        source = RUN_TO_SOURCE.get(run_id, V3)
        frame = v5 if source == V5_NLOS else v3
        frames.append(frame[frame["run_id"] == run_id].copy())
        print(f"{run_id}: {source.name}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} rows={len(out)}", flush=True)


if __name__ == "__main__":
    main()
