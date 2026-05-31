#!/usr/bin/env python3
"""Evaluate feature ablations for the n/r2 pb40 non-reference ranker."""

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

from exp_ppc_ctrbpf_fgo import _load_full_reference, _load_hybrid_pos_file  # noqa: E402
from gnss_gpu.ppc_score import ppc_segment_distances  # noqa: E402

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
RESULTS = REPO / "experiments/results"
DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")

BASE_FEATURES = [
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

VARIANTS: dict[str, tuple[list[str], bool]] = {
    "all": (BASE_FEATURES, True),
    "no_tow": ([c for c in BASE_FEATURES if c != "tow_phase"], True),
    "no_label": (BASE_FEATURES, False),
    "no_tow_no_label": ([c for c in BASE_FEATURES if c != "tow_phase"], False),
    "compact": (
        [
            "base_p_pass",
            "rms",
            "ratio",
            "abs_max",
            "sats",
            "status",
            "rank_by_rms",
            "cluster_size_50cm",
            "cluster_size_25cm",
            "max_cluster_size_50cm",
            "is_in_max_cluster_50cm",
            "dist_to_median_m",
            "dist_to_hybrid_m",
        ],
        True,
    ),
    "compact_no_label": (
        [
            "base_p_pass",
            "rms",
            "ratio",
            "abs_max",
            "sats",
            "status",
            "rank_by_rms",
            "cluster_size_50cm",
            "cluster_size_25cm",
            "max_cluster_size_50cm",
            "is_in_max_cluster_50cm",
            "dist_to_median_m",
            "dist_to_hybrid_m",
        ],
        False,
    ),
    "base_cluster": (
        [
            "base_p_pass",
            "rank_by_rms",
            "cluster_size_50cm",
            "cluster_size_25cm",
            "max_cluster_size_50cm",
            "is_in_max_cluster_50cm",
            "dist_to_median_m",
            "dist_to_hybrid_m",
        ],
        True,
    ),
    "base_cluster_no_label": (
        [
            "base_p_pass",
            "rank_by_rms",
            "cluster_size_50cm",
            "cluster_size_25cm",
            "max_cluster_size_50cm",
            "is_in_max_cluster_50cm",
            "dist_to_median_m",
            "dist_to_hybrid_m",
        ],
        False,
    ),
}

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 300,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 3,
    "lambda_l2": 3.0,
    "verbosity": -1,
    "force_col_wise": True,
    "num_threads": 8,
    "seed": 42,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=RESULTS / "nr2_pb40_exact_nonref_features.csv",
    )
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS / "libgnss_rtk_pos_v5")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=220)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=RESULTS / "nr2_pb40_feature_variant_scores.csv",
    )
    return parser.parse_args()


def _fallback(args: argparse.Namespace) -> pd.DataFrame:
    ref = _load_full_reference(args.data_root / args.city / args.run / "reference.csv")
    truth = np.asarray([ecef for _, ecef in ref], dtype=np.float64)
    weights = ppc_segment_distances(truth)
    hybrid, _status = _load_hybrid_pos_file(
        args.hybrid_pos_dir / f"{args.city}_{args.run}_full.pos"
    )
    rows = []
    for i, (tow_raw, true_pos) in enumerate(ref):
        tow = round(float(tow_raw), 1)
        hp = hybrid.get(tow)
        is_pass = 0
        if hp is not None and np.all(np.isfinite(hp)) and not np.all(np.asarray(hp) == 0.0):
            is_pass = int(np.linalg.norm(np.asarray(hp) - true_pos) < 0.5)
        rows.append({"tow": tow, "path_weight": float(weights[i]), "is_pass_50cm": is_pass})
    return pd.DataFrame(rows)


def _score(df: pd.DataFrame, pred_col: str, fallback: pd.DataFrame) -> tuple[float, float, int]:
    covered: set[float] = set()
    pass_w = 0.0
    total_w = 0.0
    pass_epochs = 0
    for tow, group in df.groupby("tow", sort=False):
        row = group.loc[group[pred_col].idxmax()]
        w = float(row["path_weight"])
        total_w += w
        covered.add(round(float(tow), 1))
        if int(row["is_pass_50cm"]) == 1:
            pass_w += w
            pass_epochs += 1
    for row in fallback.itertuples(index=False):
        tow = round(float(row.tow), 1)
        if tow in covered:
            continue
        w = float(row.path_weight)
        total_w += w
        if int(row.is_pass_50cm) == 1:
            pass_w += w
            pass_epochs += 1
    return 100.0 * pass_w / total_w if total_w else 0.0, pass_w, pass_epochs


def _train_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    use_label: bool,
    rounds: int,
    categories: list[str],
) -> np.ndarray:
    cols = features + (["label"] if use_label else [])
    train_x = train[cols].copy()
    test_x = test[cols].copy()
    cat_cols = ["label"] if use_label else []
    if use_label:
        train_x["label"] = pd.Categorical(train_x["label"], categories=categories)
        test_x["label"] = pd.Categorical(test_x["label"], categories=categories)
    dataset = lgb.Dataset(
        train_x,
        label=train["is_pass_50cm"].astype(int),
        weight=train["path_weight"].clip(lower=1e-6),
        categorical_feature=cat_cols,
    )
    model = lgb.train(LGB_PARAMS, dataset, num_boost_round=rounds)
    return model.predict(test_x)


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.features)
    fb = _fallback(args)
    categories = sorted(df["label"].unique())
    epochs = np.array(sorted(df["tow"].unique()))
    folds = [np.asarray(fold) for fold in np.array_split(epochs, int(args.folds))]

    rows = []
    base_score, base_w, base_pass = _score(df, "base_p_pass", fb)
    rows.append(
        {
            "variant": "base_p_pass",
            "mode": "base",
            "score": base_score,
            "pass_w": base_w,
            "pass_epochs": base_pass,
        }
    )
    print(f"base_p_pass score={base_score:.6f} pass_w={base_w:.3f}")

    for name, (features, use_label) in VARIANTS.items():
        pred = np.full(len(df), np.nan, dtype=float)
        for fold_i, test_epochs in enumerate(folds, start=1):
            test_mask = df["tow"].isin(test_epochs).to_numpy()
            train = df.loc[~test_mask]
            test = df.loc[test_mask]
            pred[test_mask] = _train_predict(
                train,
                test,
                features,
                use_label,
                int(args.rounds),
                categories,
            )
            print(f"variant={name} fold={fold_i} done", flush=True)
        df[f"pred_{name}"] = pred
        score, pass_w, pass_epochs = _score(df, f"pred_{name}", fb)
        print(f"variant={name:22s} timefold_score={score:.6f} pass_w={pass_w:.3f}")
        rows.append(
            {
                "variant": name,
                "mode": "timefold",
                "score": score,
                "pass_w": pass_w,
                "pass_epochs": pass_epochs,
            }
        )

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    out.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
