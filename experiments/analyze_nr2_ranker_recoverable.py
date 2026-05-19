#!/usr/bin/env python3
"""Quantify the n/r2 ranker-recoverable subset.

For each epoch on n/r2:
- Look at all candidates in pool (any_pass=1 means at least one passes 50cm).
- Look at the ranker pick (highest p_pass via predictions CSV).
- If any_pass=1 BUT ranker_pick fails 50cm => recoverable by better ranker.

This is the target for wrong-fix feature engineering — we cannot recover
truly-lost epochs (any_pass=0), but we CAN recover where ranker picked wrong.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")


def main():
    feat = pd.read_csv(REPO / "experiments/results/selector_training_features.csv")
    nr2 = feat[feat["run_id"] == "nagoya_run2"].copy()

    # Per-epoch any_pass and best_err
    epoch_agg = nr2.groupby("tow").agg(
        any_pass=("is_pass_50cm", "max"),
        best_err=("err_3d_m", "min"),
        path_weight=("path_weight", "first"),
        n_cand=("label", "count"),
    ).reset_index()

    print(f"n/r2 total epochs: {len(epoch_agg)}")
    print(f"  any_pass=1 (pool has passing candidate): {epoch_agg['any_pass'].sum()} "
          f"({100*epoch_agg['any_pass'].mean():.1f}%)")
    print(f"  truly-lost (any_pass=0): {(epoch_agg['any_pass']==0).sum()} "
          f"({100*(epoch_agg['any_pass']==0).mean():.1f}%)")

    # Ranker pick: load predictions CSV, find argmax label per epoch
    # Use both v1 baseline and conditional (n/r2 viterbi) for comparison
    for label_name, pred_csv in [
        ("v1 base", "selector_ranker_predictions.csv"),
        ("conditional (nr2 viterbi)", "selector_ranker_predictions_conditional_nr2vit.csv"),
    ]:
        path = REPO / "experiments/results" / pred_csv
        pred = pd.read_csv(path)
        pred_nr2 = pred[pred["run_id"] == "nagoya_run2"].copy()
        # argmax per tow
        pick = pred_nr2.loc[pred_nr2.groupby("tow")["p_pass"].idxmax(),
                            ["tow", "label", "p_pass"]].rename(
            columns={"label": "pick_label", "p_pass": "pick_p_pass"})

        # Merge with the actual err_3d_m of that pick from features
        truth = nr2[["tow", "label", "err_3d_m"]].rename(columns={"label": "pick_label"})
        pick_truth = pick.merge(truth, on=["tow", "pick_label"], how="left")
        pick_truth["pick_pass"] = (pick_truth["err_3d_m"] < 0.5).astype(int)

        merged = epoch_agg.merge(pick_truth, on="tow", how="left")
        recoverable = merged[(merged["any_pass"] == 1) & (merged["pick_pass"] == 0)]
        nrecov_w = recoverable["path_weight"].sum()
        ntotal_w = epoch_agg["path_weight"].sum()
        npick_pass = merged["pick_pass"].sum()
        ranker_pp = 100 * merged[merged["pick_pass"] == 1]["path_weight"].sum() / ntotal_w

        print(f"\n=== {label_name} ===")
        print(f"  ranker pick passes 50cm: {npick_pass} / {len(merged)} "
              f"({100*npick_pass/len(merged):.1f}%) — path-weighted {ranker_pp:.2f}%")
        print(f"  RECOVERABLE (any_pass=1 & pick fails): {len(recoverable)} epochs, "
              f"path weight {nrecov_w:.1f}m ({100*nrecov_w/ntotal_w:.2f}% of n/r2 path)")
        print(f"  As OFFICIAL: +{100*nrecov_w/ntotal_w/6:.3f}pp ceiling on this lever")

        # What err do the failing picks have? Are they close miss or way-off?
        print(f"  failing-pick err distribution (recoverable epochs):")
        print(recoverable["err_3d_m"].describe(percentiles=[0.25, 0.5, 0.75, 0.9]))

        # Save the recoverable epoch list for v1
        if label_name == "v1 base":
            out = REPO / "experiments/results/nr2_ranker_recoverable_v1.csv"
            recoverable_save = recoverable[[
                "tow", "any_pass", "best_err", "path_weight", "n_cand",
                "pick_label", "pick_p_pass", "err_3d_m"
            ]].rename(columns={"err_3d_m": "pick_err"})
            recoverable_save.to_csv(out, index=False)
            print(f"\n  Saved: {out} ({len(recoverable_save)} rows)")


if __name__ == "__main__":
    main()
