#!/usr/bin/env python3
"""Build v3 ranker predictions with a nagoya/run2-only label boost."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
IN_CSV = REPO / "experiments/results/selector_ranker_predictions_v3.csv"
OUT_CSV = REPO / "experiments/results/selector_ranker_predictions_v3_nr2_labelboost.csv"

NR2_FACTORS = {
    "xd_gici_hs": 1.25,
    "xd_gici_z": 0.03,
    "xd_gici_c4": 5.0,
    "xd_gici_oa": 3.0,
    "rtkout5mlc1c005oG": 5.0,
    "rtkout5c005em3": 5.0,
    "xd_fgo_v17_el25": 3.0,
    "mlc1oGc005": 5.0,
    "xd_gici_ir": 0.03,
}


def main() -> None:
    df = pd.read_csv(IN_CSV)
    mask_run = df["run_id"] == "nagoya_run2"
    for label, factor in NR2_FACTORS.items():
        mask = mask_run & (df["label"] == label)
        count = int(mask.sum())
        if count:
            df.loc[mask, "p_pass"] = df.loc[mask, "p_pass"] * float(factor)
        print(f"{label:24s} factor={factor:g} rows={count}", flush=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} rows={len(df)}", flush=True)


if __name__ == "__main__":
    main()
