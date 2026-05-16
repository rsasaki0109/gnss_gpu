#!/usr/bin/env python3
"""Add LORO-safe per-label prior pass rate feature for ranker retrain.

For each row (run_id=R, label=L), compute:
  label_prior_loro       = mean(is_pass_50cm) over rows with label=L, run_id!=R
  label_prior_same_city  = mean(is_pass_50cm) over rows with label=L, same city, run_id!=R
  label_count_loro       = number of rows used in label_prior_loro

These are computed offline from the full features CSV (no test-time leak as long
as we keep LORO at training: row in fold R only sees prior computed from !=R).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
SRC = REPO / "experiments/results/selector_training_features.csv"
DST = REPO / "experiments/results/selector_training_features_v2.csv"


def main():
    print(f"loading {SRC}")
    df = pd.read_csv(SRC)
    print(f"  rows: {len(df)}, labels: {df['label'].nunique()}, runs: {df['run_id'].nunique()}")
    df["city"] = df["run_id"].str.split("_").str[0]

    # Pre-aggregate per (run_id, label): sum and count of is_pass_50cm
    agg = (
        df.groupby(["run_id", "label", "city"])["is_pass_50cm"]
        .agg(["sum", "count"])
        .reset_index()
    )

    # For each (target_run, label): LORO prior = sum(other run sums) / sum(other counts)
    total_by_label = agg.groupby("label")[["sum", "count"]].sum().rename(
        columns={"sum": "tot_sum", "count": "tot_cnt"}
    )
    total_by_city_label = (
        agg.groupby(["city", "label"])[["sum", "count"]].sum()
        .rename(columns={"sum": "city_sum", "count": "city_cnt"})
    )

    # per-row priors via merge: subtract this row's run contribution
    merged = agg.merge(total_by_label, on="label", how="left")
    merged = merged.merge(total_by_city_label, on=["city", "label"], how="left")
    merged["loro_sum"] = merged["tot_sum"] - merged["sum"]
    merged["loro_cnt"] = merged["tot_cnt"] - merged["count"]
    merged["loro_city_sum"] = merged["city_sum"] - merged["sum"]
    merged["loro_city_cnt"] = merged["city_cnt"] - merged["count"]
    merged["label_prior_loro"] = (
        merged["loro_sum"] / merged["loro_cnt"].clip(lower=1)
    ).where(merged["loro_cnt"] > 0, 0.5)
    merged["label_prior_same_city"] = (
        merged["loro_city_sum"] / merged["loro_city_cnt"].clip(lower=1)
    ).where(merged["loro_city_cnt"] > 0, merged["label_prior_loro"])
    merged["label_count_loro"] = merged["loro_cnt"].astype(int)

    keep = merged[[
        "run_id", "label",
        "label_prior_loro", "label_prior_same_city", "label_count_loro",
    ]]
    df2 = df.merge(keep, on=["run_id", "label"], how="left")

    # sanity: per (run, label), prior is a constant
    sample = df2.groupby(["run_id", "label"])[["label_prior_loro", "label_prior_same_city"]].nunique()
    assert (sample <= 1).all().all(), "prior should be constant within (run,label)"

    print(f"saving {DST}")
    df2.drop(columns=["city"], inplace=True)
    df2.to_csv(DST, index=False)

    print("\n=== PER-LABEL LORO PRIOR (subset, sorted) ===")
    s = df2.groupby("label")["label_prior_loro"].first().sort_values(ascending=False)
    print(s.head(15))
    print("...")
    print(s.tail(10))

    print("\n=== PER-RUN/LABEL prior_same_city sample (nagoya_run2 picks) ===")
    nr2 = df2[df2["run_id"] == "nagoya_run2"].groupby("label").agg(
        prior=("label_prior_loro", "first"),
        prior_city=("label_prior_same_city", "first"),
        count=("label_count_loro", "first"),
    ).sort_values("prior_city", ascending=False)
    print(nr2)


if __name__ == "__main__":
    main()
