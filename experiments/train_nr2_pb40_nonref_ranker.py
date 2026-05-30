#!/usr/bin/env python3
"""Train a nagoya/run2-only non-reference ranker on the exact pb40 pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from diagnose_nr2_ranker_with_extra_candidate import (  # noqa: E402
    DATA_ROOT,
    RESULTS,
    _candidate_options,
    _default_candidates,
    _effective_config,
    _load_candidates,
)
from exp_ppc_ctrbpf_fgo import (  # noqa: E402
    _diag_float,
    _filter_rtkdiag_candidates_by_policy,
    _load_full_reference,
    _load_hybrid_pos_file,
    _load_ranker_predictions,
)
from gnss_gpu.ppc_score import ppc_segment_distances  # noqa: E402

FEATURE_COLS = [
    "base_p_pass",
    "rms",
    "ratio",
    "abs_max",
    "update_rows",
    "sats",
    "status",
    "pdop",
    "baseline_m",
    "candidate_jump_m",
    "spp_valid",
    "spp_sats",
    "spp_pdop",
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
CAT_COLS = ["label"]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 3,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "force_col_wise": True,
    "num_threads": 8,
    "seed": 42,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS / "libgnss_rtk_pos_v5")
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=RESULTS / "selector_ranker_predictions_v3_nr2_labelboost_pb40.csv",
    )
    parser.add_argument(
        "--out-features",
        type=Path,
        default=RESULTS / "nr2_pb40_exact_nonref_features.csv",
    )
    parser.add_argument(
        "--out-predictions",
        type=Path,
        default=RESULTS / "selector_ranker_predictions_v3_nr2_labelboost_pb40_lgb_nr2.csv",
    )
    parser.add_argument(
        "--out-model",
        type=Path,
        default=RESULTS / "nr2_pb40_nonref_ranker_lgb.txt",
    )
    parser.add_argument("--policy", default="phase11ep")
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--rms-prefilter-k", type=int, default=99)
    parser.add_argument(
        "--base-labels-file",
        type=Path,
        default=Path("/tmp/nagoya_run2_phase11fa_labels.txt"),
    )
    parser.add_argument(
        "--base-dirs-file",
        type=Path,
        default=Path("/tmp/nagoya_run2_phase11fa_dirs.txt"),
    )
    parser.add_argument("--extra-label", action="append", default=["xd_nr2_hs_pb40"])
    parser.add_argument(
        "--extra-dir",
        action="append",
        type=Path,
        default=[
            RESULTS / "libgnss_diag_phase34/nr2_hs_piecebias40_oracle_556184_556337"
        ],
    )
    return parser.parse_args()


def _diag(row: dict[str, str], key: str) -> float:
    return _diag_float(row, key)


def _cluster_features(positions: np.ndarray) -> dict[str, np.ndarray]:
    n = positions.shape[0]
    if n == 0:
        z = np.zeros(0, dtype=float)
        return {
            "cluster_size_50cm": z,
            "cluster_size_25cm": z,
            "cluster_size_10cm": z,
            "max_cluster_size_50cm": z,
            "is_in_max_cluster_50cm": z,
            "dist_to_median_m": z,
            "dist_to_max_cluster_centroid_m": z,
            "delta_pos_norm_m": z,
            "delta_pos_vs_median_m": z,
        }
    diffs = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    c50 = (dist <= 0.5).sum(axis=1).astype(float)
    c25 = (dist <= 0.25).sum(axis=1).astype(float)
    c10 = (dist <= 0.10).sum(axis=1).astype(float)
    max_i = int(np.argmax(c50))
    max_members = dist[max_i] <= 0.5
    centroid = positions[max_members].mean(axis=0)
    median = np.median(positions, axis=0)
    return {
        "cluster_size_50cm": c50,
        "cluster_size_25cm": c25,
        "cluster_size_10cm": c10,
        "max_cluster_size_50cm": np.full(n, float(c50[max_i])),
        "is_in_max_cluster_50cm": max_members.astype(float),
        "dist_to_median_m": np.linalg.norm(positions - median, axis=1),
        "dist_to_max_cluster_centroid_m": np.linalg.norm(positions - centroid, axis=1),
        "delta_pos_norm_m": np.linalg.norm(positions - positions[max_i], axis=1),
        "delta_pos_vs_median_m": np.full(n, float(np.linalg.norm(positions[max_i] - median))),
    }


def _score_prediction(
    df: pd.DataFrame,
    pred_col: str,
    fallback: pd.DataFrame,
) -> tuple[float, int, float, float]:
    pass_w = 0.0
    total_w = 0.0
    pass_epochs = 0
    for tow, group in df.groupby("tow", sort=False):
        row = group.loc[group[pred_col].idxmax()]
        weight = float(row["path_weight"])
        total_w += weight
        if int(row["is_pass_50cm"]) == 1:
            pass_w += weight
            pass_epochs += 1
    covered = set(df["tow"].round(1))
    for row in fallback.itertuples(index=False):
        if round(float(row.tow), 1) in covered:
            continue
        total_w += float(row.path_weight)
        if int(row.is_pass_50cm) == 1:
            pass_w += float(row.path_weight)
            pass_epochs += 1
    return 100.0 * pass_w / total_w if total_w > 0 else 0.0, pass_epochs, pass_w, total_w


def _fit_predict(train: pd.DataFrame, pred: pd.DataFrame, categories: list[str]) -> np.ndarray:
    train_x = train[FEATURE_COLS + CAT_COLS].copy()
    pred_x = pred[FEATURE_COLS + CAT_COLS].copy()
    train_x["label"] = pd.Categorical(train_x["label"], categories=categories)
    pred_x["label"] = pd.Categorical(pred_x["label"], categories=categories)
    dataset = lgb.Dataset(
        train_x,
        label=train["is_pass_50cm"].astype(int),
        weight=train["path_weight"].clip(lower=1e-6),
        categorical_feature=CAT_COLS,
    )
    model = lgb.train(LGB_PARAMS, dataset, num_boost_round=260)
    return model.predict(pred_x)


def _build_features(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    city = str(args.city)
    run = str(args.run)
    run_id = f"{city}_{run}"
    labels, dirs = _default_candidates(args)
    candidates_all = _load_candidates(labels, dirs, city=city, run=run)
    candidates = _filter_rtkdiag_candidates_by_policy(
        candidates_all,
        city=city,
        run=run,
        policy=str(args.policy),
    )
    cfg = _effective_config(args)
    pred = _load_ranker_predictions(str(args.base_predictions))
    pred_run = {(tow, label): p for (rid, tow, label), p in pred.items() if rid == run_id}

    ref = _load_full_reference(args.data_root / city / run / "reference.csv")
    truth = np.asarray([ecef for _, ecef in ref], dtype=np.float64)
    weights = ppc_segment_distances(truth)
    hybrid_pos, _hybrid_status = _load_hybrid_pos_file(
        args.hybrid_pos_dir / f"{city}_{run}_full.pos"
    )

    rows: list[dict[str, object]] = []
    fallback_rows: list[dict[str, object]] = []
    for i, (tow_raw, true_pos) in enumerate(ref):
        tow = round(float(tow_raw), 1)
        hp = hybrid_pos.get(tow)
        hp_pass = 0
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(np.asarray(hp) == 0.0):
            hp_pass = int(np.linalg.norm(np.asarray(hp) - true_pos) < 0.5)
        fallback_rows.append(
            {
                "tow": tow,
                "path_weight": float(weights[i]),
                "is_pass_50cm": hp_pass,
            }
        )
        options = _candidate_options(candidates, tow=tow, cfg=cfg)
        if not options:
            continue
        positions = np.asarray([pos for _label, pos, _row in options], dtype=np.float64)
        cfeat = _cluster_features(positions)
        rms_order = np.argsort([_diag(row, "final_residual_rms") for _label, _pos, row in options])
        rank_by_rms = np.empty(len(options), dtype=float)
        rank_by_rms[rms_order] = np.arange(1, len(options) + 1, dtype=float)
        for j, (label, pos, row) in enumerate(options):
            dist_to_hybrid = np.nan
            if hp is not None and np.all(np.isfinite(hp)) and not np.all(np.asarray(hp) == 0.0):
                dist_to_hybrid = float(np.linalg.norm(pos - np.asarray(hp)))
            rows.append(
                {
                    "run_id": run_id,
                    "tow": tow,
                    "label": label,
                    "base_p_pass": float(pred_run.get((tow, label), 0.0)),
                    "rms": _diag(row, "final_residual_rms"),
                    "ratio": _diag(row, "final_ratio"),
                    "abs_max": _diag(row, "final_residual_abs_max"),
                    "update_rows": _diag(row, "final_update_rows"),
                    "sats": _diag(row, "final_sats"),
                    "status": _diag(row, "final_status"),
                    "pdop": _diag(row, "final_pdop"),
                    "baseline_m": _diag(row, "final_baseline_m"),
                    "candidate_jump_m": _diag(row, "candidate_jump_m"),
                    "spp_valid": _diag(row, "spp_valid"),
                    "spp_sats": _diag(row, "spp_sats"),
                    "spp_pdop": _diag(row, "spp_pdop"),
                    "rank_by_rms": float(rank_by_rms[j]),
                    "n_options": float(len(options)),
                    "dist_to_hybrid_m": dist_to_hybrid,
                    "tow_phase": float((tow - ref[0][0]) / max(ref[-1][0] - ref[0][0], 1.0)),
                    "err_3d_m": float(np.linalg.norm(pos - true_pos)),
                    "is_pass_50cm": int(np.linalg.norm(pos - true_pos) < 0.5),
                    "path_weight": float(weights[i]),
                    **{key: float(value[j]) for key, value in cfeat.items()},
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(fallback_rows)


def main() -> None:
    args = _parse_args()
    df, fallback = _build_features(args)
    args.out_features.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_features, index=False)
    print(f"wrote {args.out_features} rows={len(df)} epochs={df['tow'].nunique()}")

    base_score = _score_prediction(df, "base_p_pass", fallback)
    print(
        f"base score={base_score[0]:.6f} pass_epochs={base_score[1]} "
        f"pass_w={base_score[2]:.3f}/{base_score[3]:.3f}"
    )

    categories = sorted(df["label"].unique())
    rng = np.random.RandomState(42)
    train_idx = rng.permutation(len(df))
    split = int(0.9 * len(df))
    train = df.iloc[train_idx[:split]]
    valid = df.iloc[train_idx[split:]]
    train_x = train[FEATURE_COLS + CAT_COLS].copy()
    valid_x = valid[FEATURE_COLS + CAT_COLS].copy()
    train_x["label"] = pd.Categorical(train_x["label"], categories=categories)
    valid_x["label"] = pd.Categorical(valid_x["label"], categories=categories)
    train_set = lgb.Dataset(
        train_x,
        label=train["is_pass_50cm"].astype(int),
        weight=train["path_weight"].clip(lower=1e-6),
        categorical_feature=CAT_COLS,
    )
    valid_set = lgb.Dataset(
        valid_x,
        label=valid["is_pass_50cm"].astype(int),
        weight=valid["path_weight"].clip(lower=1e-6),
        categorical_feature=CAT_COLS,
        reference=train_set,
    )
    model = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=400,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    all_x = df[FEATURE_COLS + CAT_COLS].copy()
    all_x["label"] = pd.Categorical(all_x["label"], categories=categories)
    df["p_pass_lgb_insample"] = model.predict(all_x, num_iteration=model.best_iteration)
    ins = _score_prediction(df, "p_pass_lgb_insample", fallback)
    print(f"lgb insample score={ins[0]:.6f} pass_epochs={ins[1]} pass_w={ins[2]:.3f}/{ins[3]:.3f}")

    epochs = np.array(sorted(df["tow"].unique()))
    fold_pred = np.full(len(df), np.nan, dtype=float)
    for fold, epoch_fold in enumerate(np.array_split(epochs, 5), start=1):
        test_mask = df["tow"].isin(epoch_fold).to_numpy()
        train_fold = df.loc[~test_mask]
        test_fold = df.loc[test_mask]
        fold_pred[test_mask] = _fit_predict(train_fold, test_fold, categories)
        fold_df = df.copy()
        fold_df["p_fold"] = fold_pred
        known = fold_df["p_fold"].notna()
        fold_score = _score_prediction(fold_df.loc[known], "p_fold", fallback)
        print(
            f"fold={fold} covered_epochs={fold_df.loc[known, 'tow'].nunique()} "
            f"partial_score={fold_score[0]:.6f}"
        )
    df["p_pass_lgb_timefold"] = fold_pred
    cv = _score_prediction(df, "p_pass_lgb_timefold", fallback)
    print(f"lgb timefold score={cv[0]:.6f} pass_epochs={cv[1]} pass_w={cv[2]:.3f}/{cv[3]:.3f}")

    model.save_model(str(args.out_model))
    pred_out = pd.read_csv(args.base_predictions)
    run_id = f"{args.city}_{args.run}"
    lgb_scores = df[["tow", "label", "p_pass_lgb_insample"]].copy()
    lgb_scores["tow_round"] = lgb_scores["tow"].round(1)
    pred_out["tow_round"] = pred_out["tow"].round(1)
    pred_out = pred_out.merge(
        lgb_scores[["tow_round", "label", "p_pass_lgb_insample"]],
        on=["tow_round", "label"],
        how="left",
    )
    mask = (pred_out["run_id"] == run_id) & pred_out["p_pass_lgb_insample"].notna()
    pred_out.loc[mask, "p_pass"] = pred_out.loc[mask, "p_pass_lgb_insample"]
    pred_out = pred_out.drop(columns=["tow_round", "p_pass_lgb_insample"])
    pred_out.to_csv(args.out_predictions, index=False)
    print(f"wrote {args.out_predictions} rows={len(pred_out)} replaced={int(mask.sum())}")
    print(f"saved model {args.out_model}")

    print("\nfeature importance:")
    for name, importance in sorted(
        zip(model.feature_name(), model.feature_importance(importance_type="gain"), strict=True),
        key=lambda item: -item[1],
    )[:20]:
        print(f"  {name:32s} {importance:.0f}")


if __name__ == "__main__":
    main()
