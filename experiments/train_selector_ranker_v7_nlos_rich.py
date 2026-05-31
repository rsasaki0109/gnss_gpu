#!/usr/bin/env python3
"""Train selector ranker v7 with richer NLOS features (per-system + elev bands).

v5 = v3 + 6 NLOS basic. v7 = v3 + 19 NLOS (6 basic + 13 new):
  per-system count (4) + per-system frac visible (4) + n_systems_affected (1)
  + low_elev_count_10/20 (2) + high_elev_count_45 (1) + max_elev_deg (1)

Input: experiments/results/selector_training_features_v7_nlos_rich.csv
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
    "cluster_size_25cm", "cluster_size_10cm",
    "max_cluster_size_50cm", "is_in_max_cluster_50cm",
    "n_clusters_50cm", "n_clusters_50cm_ge3",
    "cluster_min_rms_50cm", "cluster_min_abs_max_50cm",
    "dist_to_max_cluster_centroid_m",
    "delta_pos_norm_m", "delta_pos_vs_median_m",
    # v5 NLOS features
    "nlos_n_sats", "nlos_count", "nlos_los_count", "nlos_frac",
    "nlos_min_elev_deg", "nlos_mean_elev_deg",
    # v7 NEW NLOS features
    "nlos_count_G", "nlos_count_E", "nlos_count_J", "nlos_count_C",
    "nlos_frac_G_visible", "nlos_frac_E_visible",
    "nlos_frac_J_visible", "nlos_frac_C_visible",
    "nlos_n_systems_affected",
    "nlos_low_elev_count_10", "nlos_low_elev_count_20",
    "nlos_high_elev_count_45", "nlos_max_elev_deg",
]
LABEL_FEATURES = ["label"]
TARGET = "is_pass_50cm"
SAMPLE_WEIGHT = "path_weight"

LGB_PARAMS = dict(
    objective="binary", metric="binary_logloss",
    learning_rate=0.1, num_leaves=31, max_depth=-1, min_data_in_leaf=500,
    feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
    lambda_l2=1.0, verbosity=-1, force_col_wise=True, num_threads=8,
)


def fit_one(train_df, valid_df, label_categories):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_df["label"] = pd.Categorical(train_df["label"], categories=label_categories)
    valid_df["label"] = pd.Categorical(valid_df["label"], categories=label_categories)

    x_train = train_df[FEATURE_COLS + LABEL_FEATURES]
    y_train = train_df[TARGET].astype(int)
    w_train = train_df[SAMPLE_WEIGHT].clip(lower=1e-6).values
    x_valid = valid_df[FEATURE_COLS + LABEL_FEATURES]
    y_valid = valid_df[TARGET].astype(int)
    w_valid = valid_df[SAMPLE_WEIGHT].clip(lower=1e-6).values

    train_set = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=["label"])
    valid_set = lgb.Dataset(x_valid, label=y_valid, weight=w_valid,
                            categorical_feature=["label"], reference=train_set)
    return lgb.train(LGB_PARAMS, train_set, num_boost_round=200,
                     valid_sets=[valid_set],
                     callbacks=[lgb.early_stopping(20), lgb.log_evaluation(25)])


def evaluate_official(df, label_col="p_pass"):
    per_run_pw = []
    for run_id, run_df in df.groupby("run_id"):
        pass_w = 0.0
        total_w = 0.0
        for (_run_id, _tow), group in run_df.groupby(["run_id", "tow"]):
            row = group.loc[group[label_col].idxmax()]
            weight = float(row[SAMPLE_WEIGHT])
            total_w += weight
            pass_w += weight * int(row[TARGET])
        per_run_pw.append((run_id, pass_w / total_w if total_w > 0 else 0.0))
    return {"official": float(np.mean([s for _r, s in per_run_pw])), "per_run_pw": per_run_pw}


def main():
    feat_path = REPO / "experiments/results/selector_training_features_v7_nlos_rich.csv"
    print(f"Loading {feat_path} ...", flush=True)
    df = pd.read_csv(feat_path)
    print(f"  rows: {len(df)}, epochs: {df.groupby(['run_id', 'tow']).ngroups}", flush=True)

    label_categories = sorted(df["label"].unique().tolist())
    runs = sorted(df["run_id"].unique().tolist())
    preds = []
    for held_run in runs:
        print(f"\n--- Held-out: {held_run} ---", flush=True)
        train_df = df[df["run_id"] != held_run]
        valid_df = df[df["run_id"] == held_run]
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(train_df))
        split = int(0.9 * len(train_df))
        model = fit_one(train_df.iloc[idx[:split]], train_df.iloc[idx[split:]], label_categories)

        x_test = valid_df[FEATURE_COLS + LABEL_FEATURES].copy()
        x_test["label"] = pd.Categorical(x_test["label"], categories=label_categories)
        pred_df = valid_df[["run_id", "tow", "label", TARGET, SAMPLE_WEIGHT]].copy()
        pred_df["p_pass"] = model.predict(x_test, num_iteration=model.best_iteration)
        preds.append(pred_df)

    pred_all = pd.concat(preds, ignore_index=True)
    out_pred = REPO / "experiments/results/selector_ranker_predictions_v7_nlos_rich.csv"
    pred_all.to_csv(out_pred, index=False)
    print(f"\nSaved: {out_pred}", flush=True)

    result = evaluate_official(pred_all)
    print(f"\n=== Ranker v7_nlos_rich LORO OFFICIAL: {result['official'] * 100:.4f}% ===", flush=True)
    for run_id, score in result["per_run_pw"]:
        print(f"  {run_id}: {score * 100:.4f}%", flush=True)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df))
    split = int(0.9 * len(df))
    final_model = fit_one(df.iloc[idx[:split]], df.iloc[idx[split:]], label_categories)
    out_model = REPO / "experiments/results/selector_ranker_model_v7_nlos_rich.txt"
    final_model.save_model(str(out_model))
    print(f"Saved: {out_model}", flush=True)

    print("\n=== Feature importances (gain) ===", flush=True)
    importances = final_model.feature_importance(importance_type="gain")
    for name, importance in sorted(zip(final_model.feature_name(), importances), key=lambda i: -i[1]):
        print(f"  {name:>32s}: {importance:.0f}", flush=True)


if __name__ == "__main__":
    main()
