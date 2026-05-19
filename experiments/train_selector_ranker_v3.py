#!/usr/bin/env python3
"""Train LightGBM path-weighted selector ranker v3 (cluster-topology + temporal).

New features over v1:
- cluster_size_25cm, cluster_size_10cm
- max_cluster_size_50cm, is_in_max_cluster_50cm
- n_clusters_50cm, n_clusters_50cm_ge3
- cluster_min_rms_50cm, cluster_min_abs_max_50cm
- dist_to_max_cluster_centroid_m
- delta_pos_norm_m, delta_pos_vs_median_m

Goal: discriminate wrong-fix on n/r2 (1930 recoverable epochs = +3.1pp OFFICIAL ceiling).
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")

FEATURE_COLS = [
    "rms", "ratio", "abs_max", "update_rows", "sats", "status",
    "pdop", "baseline_m", "spp_valid", "spp_sats", "spp_pdop",
    "candidate_vs_spp_m", "candidate_jump_m",
    "cluster_size_50cm", "rank_by_rms", "n_candidates_in_epoch",
    "dist_to_median_m",
    # NEW v3 features:
    "cluster_size_25cm", "cluster_size_10cm",
    "max_cluster_size_50cm", "is_in_max_cluster_50cm",
    "n_clusters_50cm", "n_clusters_50cm_ge3",
    "cluster_min_rms_50cm", "cluster_min_abs_max_50cm",
    "dist_to_max_cluster_centroid_m",
    "delta_pos_norm_m", "delta_pos_vs_median_m",
]
LABEL_FEATURES = ["label"]
TARGET = "is_pass_50cm"
SAMPLE_WEIGHT = "path_weight"

LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.1,
    num_leaves=31,
    max_depth=-1,
    min_data_in_leaf=500,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    lambda_l2=1.0,
    verbosity=-1,
    force_col_wise=True,
    num_threads=8,
)


def fit_one(train_df, valid_df, label_categories):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_df["label"] = pd.Categorical(train_df["label"], categories=label_categories)
    valid_df["label"] = pd.Categorical(valid_df["label"], categories=label_categories)

    X_train = train_df[FEATURE_COLS + LABEL_FEATURES]
    y_train = train_df[TARGET].astype(int)
    w_train = train_df[SAMPLE_WEIGHT].clip(lower=1e-6).values
    X_valid = valid_df[FEATURE_COLS + LABEL_FEATURES]
    y_valid = valid_df[TARGET].astype(int)
    w_valid = valid_df[SAMPLE_WEIGHT].clip(lower=1e-6).values

    train_set = lgb.Dataset(X_train, label=y_train, weight=w_train, categorical_feature=["label"])
    valid_set = lgb.Dataset(X_valid, label=y_valid, weight=w_valid, categorical_feature=["label"], reference=train_set)

    model = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=200,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(25)],
    )
    return model


def evaluate_official(df, label_col="p_pass"):
    per_run_pw = []
    for run_id, run_df in df.groupby("run_id"):
        pw_pass = 0.0
        pw_total = 0.0
        for (_, tow), g in run_df.groupby(["run_id", "tow"]):
            idx = g[label_col].idxmax()
            row = g.loc[idx]
            w = float(row[SAMPLE_WEIGHT])
            passed = int(row[TARGET])
            pw_total += w
            pw_pass += w * passed
        per_run_pw.append((run_id, pw_pass / pw_total if pw_total > 0 else 0.0))
    official = float(np.mean([p for _, p in per_run_pw]))
    return {"official": official, "per_run_pw": per_run_pw}


def main():
    feat_path = REPO / "experiments/results/selector_training_features_v3.csv"
    print(f"Loading {feat_path} ...", flush=True)
    df = pd.read_csv(feat_path)
    print(f"  rows: {len(df)}, epochs: {df.groupby(['run_id', 'tow']).ngroups}", flush=True)

    label_categories = sorted(df["label"].unique().tolist())
    print(f"  candidate labels: {len(label_categories)}", flush=True)

    runs = sorted(df["run_id"].unique().tolist())
    print(f"\n=== LORO across {len(runs)} runs ===", flush=True)
    preds = []
    for held_run in runs:
        print(f"\n--- Held-out: {held_run} ---", flush=True)
        train_df = df[df["run_id"] != held_run]
        valid_df = df[df["run_id"] == held_run]
        rng = np.random.RandomState(42)
        n_train = len(train_df)
        train_idx = rng.permutation(n_train)
        cut = int(0.9 * n_train)
        train_split = train_df.iloc[train_idx[:cut]]
        es_split = train_df.iloc[train_idx[cut:]]

        model = fit_one(train_split, es_split, label_categories)
        X_test = valid_df[FEATURE_COLS + LABEL_FEATURES].copy()
        X_test["label"] = pd.Categorical(X_test["label"], categories=label_categories)
        p = model.predict(X_test, num_iteration=model.best_iteration)
        pred_df = valid_df[["run_id", "tow", "label", TARGET, SAMPLE_WEIGHT]].copy()
        pred_df["p_pass"] = p
        preds.append(pred_df)

    pred_all = pd.concat(preds, ignore_index=True)
    out_pred = REPO / "experiments/results/selector_ranker_predictions_v3.csv"
    pred_all.to_csv(out_pred, index=False)
    print(f"\nSaved: {out_pred}", flush=True)

    result = evaluate_official(pred_all, label_col="p_pass")
    print(f"\n=== Ranker v3 LORO OFFICIAL: {result['official'] * 100:.2f}% ===", flush=True)
    for run_id, pw in result["per_run_pw"]:
        print(f"  {run_id}: {pw * 100:.2f}%", flush=True)

    # Final model
    print(f"\n=== Final model on all data ===", flush=True)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df))
    cut = int(0.9 * len(df))
    final_model = fit_one(df.iloc[idx[:cut]], df.iloc[idx[cut:]], label_categories)
    out_model = REPO / "experiments/results/selector_ranker_model_v3.txt"
    final_model.save_model(str(out_model))
    print(f"Saved: {out_model}", flush=True)

    print(f"\n=== Feature importances (gain) ===", flush=True)
    importances = final_model.feature_importance(importance_type="gain")
    fnames = final_model.feature_name()
    for fn, imp in sorted(zip(fnames, importances), key=lambda x: -x[1]):
        print(f"  {fn:>32s}: {imp:.0f}", flush=True)


if __name__ == "__main__":
    main()
