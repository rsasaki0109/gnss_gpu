#!/usr/bin/env python3
"""Force recoverable phase35 n/r2 wrong picks to their truth-best label."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
RESULTS = REPO / "experiments/results"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS / "selector_ranker_predictions_v3_nr2_labelboost_pb40.csv",
    )
    parser.add_argument(
        "--wrongs",
        type=Path,
        default=RESULTS / "nr2_phase35_pb40_ranker_wrongpicks.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "selector_ranker_predictions_v3_nr2_labelboost_pb40_oracle_recover.csv",
    )
    parser.add_argument("--run-id", default="nagoya_run2")
    parser.add_argument("--forced-score", type=float, default=1000.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pred = pd.read_csv(args.input)
    wrongs = pd.read_csv(args.wrongs, usecols=["tow", "best_label"])
    wrongs = wrongs.drop_duplicates().copy()
    wrongs["tow"] = wrongs["tow"].round(1)

    pred["tow_round"] = pred["tow"].round(1)
    force_keys = set(zip(wrongs["tow"], wrongs["best_label"], strict=True))
    force_mask = pd.Series(
        [
            (tow, label) in force_keys
            for tow, label in zip(pred["tow_round"], pred["label"], strict=True)
        ],
        index=pred.index,
    )
    mask = (pred["run_id"] == args.run_id) & force_mask
    matched = int(sum(mask))
    if matched != len(force_keys):
        missing = len(force_keys) - matched
        print(f"WARNING: matched={matched} force_keys={len(force_keys)} missing={missing}")
    pred.loc[mask, "p_pass"] = float(args.forced_score)
    pred = pred.drop(columns=["tow_round"])
    pred.to_csv(args.output, index=False)
    print(
        f"wrote {args.output} rows={len(pred)} forced_rows={matched} "
        f"unique_force_keys={len(force_keys)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
