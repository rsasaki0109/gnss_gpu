#!/usr/bin/env python3
"""Train a guarded family-local second-stage ranker for n/r2.

This is a focused follow-up to the Phase43 hard rule.  Earlier full rerankers
were negative because they replaced too many good picks.  Here the model may
only override when the base pick is one of the high-risk GICI labels and the
alternative is in the same candidate family.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

RESULTS = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/results")

HIGH_RISK = {
    "xd_gici_c4",
    "xd_gici_oa",
    "xd_gici_combo",
    "xd_gici_z",
    "xd_gici_hs",
}

FEATURES = [
    "base_p_pass",
    "rms",
    "ratio",
    "abs_max",
    "update_rows",
    "sats",
    "status",
    "pdop",
    "rank_by_rms",
    "n_options",
    "cluster_size_50cm",
    "cluster_size_25cm",
    "cluster_size_10cm",
    "max_cluster_size_50cm",
    "is_in_max_cluster_50cm",
    "dist_to_median_m",
    "dist_to_max_cluster_centroid_m",
    "delta_pos_norm_m",
    "delta_pos_vs_median_m",
    "dist_to_hybrid_m",
    "tow_phase",
]

PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 3,
    "lambda_l2": 8.0,
    "verbosity": -1,
    "force_col_wise": True,
    "num_threads": 8,
    "seed": 43,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=RESULTS / "nr2_pb40_exact_nonref_features.csv",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=160)
    parser.add_argument(
        "--out-pred",
        type=Path,
        default=RESULTS / "nr2_phase43_family_override_timefold_predictions.csv",
    )
    parser.add_argument(
        "--out-sweep",
        type=Path,
        default=RESULTS / "nr2_phase43_family_override_sweep.csv",
    )
    return parser.parse_args()


def _family(label: str) -> str:
    if str(label).startswith("xd_gici_"):
        return "xd_gici"
    if str(label).startswith("xd_fgo_"):
        return "xd_fgo"
    if str(label).startswith("rtkout"):
        return "rtkout"
    if str(label).startswith("mlc"):
        return "mlc"
    if str(label).startswith("c"):
        return "csig"
    return str(label).split("_", 1)[0]


def _score(df: pd.DataFrame, pick_col: str) -> tuple[float, float, int]:
    rows = df.loc[df.groupby("tow")[pick_col].idxmax()]
    pass_w = float(rows.loc[rows["is_pass_50cm"] == 1, "path_weight"].sum())
    total_w = float(rows["path_weight"].sum())
    pass_epochs = int((rows["is_pass_50cm"] == 1).sum())
    return (100.0 * pass_w / total_w if total_w else 0.0, pass_w, pass_epochs)


def _train_timefold(df: pd.DataFrame, folds: int, rounds: int) -> np.ndarray:
    pred = np.full(len(df), np.nan, dtype=np.float64)
    epochs = np.asarray(sorted(df["tow"].unique()))
    fold_epochs = [np.asarray(x) for x in np.array_split(epochs, int(folds))]
    categories = sorted(df["label"].unique())
    cols = FEATURES + ["label"]
    for fold_i, test_epochs in enumerate(fold_epochs, start=1):
        test_mask = df["tow"].isin(test_epochs).to_numpy()
        train = df.loc[~test_mask].copy()
        test = df.loc[test_mask].copy()
        train_x = train[cols].copy()
        test_x = test[cols].copy()
        train_x["label"] = pd.Categorical(train_x["label"], categories=categories)
        test_x["label"] = pd.Categorical(test_x["label"], categories=categories)
        dataset = lgb.Dataset(
            train_x,
            label=train["is_pass_50cm"].astype(int),
            weight=train["path_weight"].clip(lower=1e-6),
            categorical_feature=["label"],
        )
        model = lgb.train(PARAMS, dataset, num_boost_round=int(rounds))
        pred[test_mask] = model.predict(test_x)
        print(f"fold {fold_i}/{folds} rows={int(test_mask.sum())}", flush=True)
    return pred


def _apply_guarded_override(
    df: pd.DataFrame,
    *,
    margin: float,
    min_model: float,
    max_rms_rank: int,
    min_cluster50: int,
    max_base_p: float,
) -> pd.Series:
    out = df["base_p_pass"].copy()
    for _tow, group in df.groupby("tow", sort=False):
        base_idx = group["base_p_pass"].idxmax()
        base = df.loc[base_idx]
        base_label = str(base["label"])
        if base_label not in HIGH_RISK:
            continue
        base_family = _family(base_label)
        cand = group.copy()
        cand = cand.loc[cand.index != base_idx]
        cand = cand.loc[cand["label"].map(_family) == base_family]
        cand = cand.loc[cand["rank_by_rms"] <= int(max_rms_rank)]
        cand = cand.loc[cand["cluster_size_50cm"] >= int(min_cluster50)]
        cand = cand.loc[cand["base_p_pass"] <= float(max_base_p)]
        if cand.empty:
            continue
        best_idx = cand["p_stage2"].idxmax()
        best = df.loc[best_idx]
        base_stage2 = float(base["p_stage2"])
        best_stage2 = float(best["p_stage2"])
        if best_stage2 < float(min_model):
            continue
        if best_stage2 < base_stage2 + float(margin):
            continue
        out.loc[best_idx] = 1000.0
    return out


def _precompute_epoch_groups(df: pd.DataFrame) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    labels = df["label"].astype(str).to_numpy()
    families = df["label_family"].astype(str).to_numpy()
    base_p = df["base_p_pass"].to_numpy(dtype=np.float64)
    for _tow, group in df.groupby("tow", sort=False):
        idx = group.index.to_numpy(dtype=np.int64)
        base_idx = int(idx[np.argmax(base_p[idx])])
        base_label = labels[base_idx]
        cand_idx = np.array([], dtype=np.int64)
        if base_label in HIGH_RISK:
            same_family = families[idx] == families[base_idx]
            not_base = idx != base_idx
            cand_idx = idx[same_family & not_base]
        groups.append(
            {
                "base_idx": base_idx,
                "cand_idx": cand_idx,
            }
        )
    return groups


def _score_indices(
    chosen: np.ndarray,
    pass_flag: np.ndarray,
    path_weight: np.ndarray,
    total_w: float,
) -> tuple[float, float, int]:
    passed = pass_flag[chosen] == 1
    pass_w = float(path_weight[chosen][passed].sum())
    return (100.0 * pass_w / total_w if total_w else 0.0, pass_w, int(passed.sum()))


def _choose_with_guard(
    groups: list[dict[str, object]],
    *,
    p_stage2: np.ndarray,
    base_p: np.ndarray,
    rank_by_rms: np.ndarray,
    cluster50: np.ndarray,
    margin: float,
    min_model: float,
    max_rms_rank: int,
    min_cluster50: int,
    max_base_p: float,
) -> np.ndarray:
    chosen = np.empty(len(groups), dtype=np.int64)
    for gi, group in enumerate(groups):
        base_idx = int(group["base_idx"])
        chosen_idx = base_idx
        cand_idx = np.asarray(group["cand_idx"], dtype=np.int64)
        if cand_idx.size:
            mask = (
                (rank_by_rms[cand_idx] <= int(max_rms_rank))
                & (cluster50[cand_idx] >= int(min_cluster50))
                & (base_p[cand_idx] <= float(max_base_p))
            )
            cand_idx = cand_idx[mask]
            if cand_idx.size:
                best_idx = int(cand_idx[np.argmax(p_stage2[cand_idx])])
                if (
                    p_stage2[best_idx] >= float(min_model)
                    and p_stage2[best_idx] >= p_stage2[base_idx] + float(margin)
                ):
                    chosen_idx = best_idx
        chosen[gi] = chosen_idx
    return chosen


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.features)
    df = df.loc[df["run_id"] == "nagoya_run2"].copy()
    df = df.reset_index(drop=True)
    df["label_family"] = df["label"].map(_family)
    df["p_stage2"] = _train_timefold(df, int(args.folds), int(args.rounds))
    args.out_pred.parent.mkdir(parents=True, exist_ok=True)
    df[["run_id", "tow", "label", "base_p_pass", "p_stage2", "is_pass_50cm", "path_weight"]].to_csv(
        args.out_pred,
        index=False,
    )

    groups = _precompute_epoch_groups(df)
    pass_flag = df["is_pass_50cm"].to_numpy(dtype=np.int8)
    path_weight = df["path_weight"].to_numpy(dtype=np.float64)
    total_w = float(path_weight[[int(g["base_idx"]) for g in groups]].sum())
    base_chosen = np.asarray([int(g["base_idx"]) for g in groups], dtype=np.int64)
    base_score, base_w, base_epochs = _score_indices(
        base_chosen,
        pass_flag,
        path_weight,
        total_w,
    )
    rows = [
        {
            "variant": "base",
            "score": base_score,
            "delta_score": 0.0,
            "pass_w": base_w,
            "delta_pass_w": 0.0,
            "pass_epochs": base_epochs,
            "margin": np.nan,
            "min_model": np.nan,
            "max_rms_rank": np.nan,
            "min_cluster50": np.nan,
            "max_base_p": np.nan,
        }
    ]
    for margin in [0.0, 0.06, 0.12]:
        for min_model in [0.50, 0.65, 0.80]:
            for max_rms_rank in [8, 12]:
                for min_cluster50 in [3, 6]:
                    for max_base_p in [1000.0, 2.0]:
                        col = (
                            f"override_m{margin}_p{min_model}_r{max_rms_rank}"
                            f"_c{min_cluster50}_bp{max_base_p}"
                        )
                        chosen = _choose_with_guard(
                            groups,
                            p_stage2=df["p_stage2"].to_numpy(dtype=np.float64),
                            base_p=df["base_p_pass"].to_numpy(dtype=np.float64),
                            rank_by_rms=df["rank_by_rms"].to_numpy(dtype=np.float64),
                            cluster50=df["cluster_size_50cm"].to_numpy(dtype=np.float64),
                            margin=margin,
                            min_model=min_model,
                            max_rms_rank=max_rms_rank,
                            min_cluster50=min_cluster50,
                            max_base_p=max_base_p,
                        )
                        score, pass_w, pass_epochs = _score_indices(
                            chosen,
                            pass_flag,
                            path_weight,
                            total_w,
                        )
                        rows.append(
                            {
                                "variant": col,
                                "score": score,
                                "delta_score": score - base_score,
                                "pass_w": pass_w,
                                "delta_pass_w": pass_w - base_w,
                                "pass_epochs": pass_epochs,
                                "margin": margin,
                                "min_model": min_model,
                                "max_rms_rank": max_rms_rank,
                                "min_cluster50": min_cluster50,
                                "max_base_p": max_base_p,
                            }
                        )
    out = pd.DataFrame(rows).sort_values(["score", "delta_pass_w"], ascending=False)
    out.to_csv(args.out_sweep, index=False)
    print(
        f"base score={base_score:.6f} pass_w={base_w:.3f} pass_epochs={base_epochs}",
        flush=True,
    )
    print(out.head(15).to_string(index=False), flush=True)
    print(f"wrote {args.out_pred}")
    print(f"wrote {args.out_sweep}")


if __name__ == "__main__":
    main()
