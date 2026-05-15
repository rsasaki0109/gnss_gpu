#!/usr/bin/env python3
"""Apply Viterbi sequence smoothing to ranker predictions.

Replaces per-epoch independent argmax(p_pass) with HMM Viterbi over the
sequence: emission = log p_pass[label], transition = -alpha if label changes,
0 if stays. Output: a new predictions CSV with p_pass=1.0 on Viterbi-picked
label, 0.0 on others (so existing ranker selection picks Viterbi label).

Usage:
  python experiments/viterbi_smooth_predictions.py \
    --in experiments/results/selector_ranker_predictions.csv \
    --out experiments/results/selector_ranker_predictions_viterbi_a05.csv \
    --alpha 0.5
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def viterbi_one_run(df_run: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Per-run Viterbi: rows are (tow, label, p_pass). Returns picks (tow, label)."""
    # Build per-epoch label -> log p_pass dicts in tow order.
    df_run = df_run.sort_values(["tow"]).reset_index(drop=True)
    epochs = []
    for tow, g in df_run.groupby("tow", sort=True):
        emission = {}
        for _, row in g.iterrows():
            p = max(float(row["p_pass"]), 1e-12)
            emission[row["label"]] = math.log(p)
        epochs.append((float(tow), emission))

    if not epochs:
        return pd.DataFrame(columns=["run_id", "tow", "label"])

    # Viterbi: state space = union of labels across all epochs.
    # Forward DP: best_score[t][label] = max over prev label of
    #     best_score[t-1][prev] + emission[t][label] - alpha * (prev != label)
    best_score: list[dict[str, float]] = []
    backtrace: list[dict[str, str]] = []

    # t=0
    emi0 = epochs[0][1]
    best_score.append({lbl: e for lbl, e in emi0.items()})
    backtrace.append({lbl: lbl for lbl in emi0})

    for t in range(1, len(epochs)):
        emi_t = epochs[t][1]
        prev_scores = best_score[t - 1]
        cur_scores: dict[str, float] = {}
        cur_bt: dict[str, str] = {}
        # Best stay-score for each prev label = prev_scores[prev]
        # Best switch-score from any prev = max(prev_scores) - alpha
        if prev_scores:
            best_prev_lbl, best_prev_val = max(prev_scores.items(), key=lambda kv: kv[1])
            switch_score = best_prev_val - alpha
        else:
            best_prev_lbl = None
            switch_score = -float("inf")
        for lbl, e in emi_t.items():
            # Option A: stay (prev label == lbl)
            stay_val = prev_scores.get(lbl, -float("inf"))
            # Option B: switch from best prev label
            if stay_val >= switch_score:
                cur_scores[lbl] = stay_val + e
                cur_bt[lbl] = lbl
            else:
                cur_scores[lbl] = switch_score + e
                cur_bt[lbl] = best_prev_lbl if best_prev_lbl is not None else lbl
        best_score.append(cur_scores)
        backtrace.append(cur_bt)

    # Backtrace from argmax(best_score[T-1])
    T = len(epochs)
    final = best_score[-1]
    cur_lbl, _ = max(final.items(), key=lambda kv: kv[1])
    picks = [None] * T
    picks[-1] = cur_lbl
    for t in range(T - 1, 0, -1):
        cur_lbl = backtrace[t].get(cur_lbl, cur_lbl)
        picks[t - 1] = cur_lbl

    out_rows = []
    run_id = df_run["run_id"].iloc[0]
    for t, (tow, _) in enumerate(epochs):
        out_rows.append({"run_id": run_id, "tow": tow, "label": picks[t]})
    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="transition penalty (log-prob units). 0 = no smoothing.")
    args = ap.parse_args()

    print(f"Loading {args.inp} ...")
    df = pd.read_csv(args.inp)
    print(f"  rows: {len(df)}, runs: {df['run_id'].nunique()}")

    picks_all = []
    for run_id in sorted(df["run_id"].unique()):
        df_run = df[df["run_id"] == run_id]
        picks = viterbi_one_run(df_run, alpha=args.alpha)
        n_switch = (picks["label"].shift() != picks["label"]).sum() - 1
        print(f"  {run_id}: {len(picks)} epochs, {n_switch} switches "
              f"({100 * n_switch / max(len(picks), 1):.1f}%)")
        picks_all.append(picks)
    picks_df = pd.concat(picks_all, ignore_index=True)

    # Now produce a new predictions CSV with p_pass=1.0 on viterbi-picked, 0.0 else.
    # Keep is_pass_50cm/path_weight unchanged.
    print("Merging viterbi picks back into predictions ...")
    df["tow"] = df["tow"].astype(float).round(1)
    picks_df["tow"] = picks_df["tow"].astype(float).round(1)
    picks_df = picks_df.rename(columns={"label": "viterbi_label"})
    merged = df.merge(picks_df, on=["run_id", "tow"], how="left")
    merged["p_pass"] = np.where(merged["label"] == merged["viterbi_label"], 1.0, 0.0)
    merged = merged.drop(columns=["viterbi_label"])

    out_path = Path(args.out)
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
