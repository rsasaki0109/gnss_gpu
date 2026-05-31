#!/usr/bin/env python3
"""Sweep base/LGB blend policies using time-fold predictions."""

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

from eval_nr2_pb40_feature_variants import (  # noqa: E402
    LGB_PARAMS,
    _fallback,
    _score,
)

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")
RESULTS = REPO / "experiments/results"

FEATURES = [
    "base_p_pass",
    "rank_by_rms",
    "cluster_size_50cm",
    "cluster_size_25cm",
    "max_cluster_size_50cm",
    "is_in_max_cluster_50cm",
    "dist_to_median_m",
    "dist_to_hybrid_m",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=RESULTS / "nr2_pb40_exact_nonref_features.csv",
    )
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data"),
    )
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS / "libgnss_rtk_pos_v5")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=160)
    parser.add_argument(
        "--out-pred",
        type=Path,
        default=RESULTS / "nr2_pb40_base_cluster_no_label_timefold_predictions.csv",
    )
    parser.add_argument(
        "--out-sweep",
        type=Path,
        default=RESULTS / "nr2_pb40_timefold_blend_sweep.csv",
    )
    return parser.parse_args()


def _train_predict(train: pd.DataFrame, test: pd.DataFrame, rounds: int) -> np.ndarray:
    train_x = train[FEATURES].copy()
    test_x = test[FEATURES].copy()
    dataset = lgb.Dataset(
        train_x,
        label=train["is_pass_50cm"].astype(int),
        weight=train["path_weight"].clip(lower=1e-6),
    )
    model = lgb.train(LGB_PARAMS, dataset, num_boost_round=rounds)
    return model.predict(test_x)


def _rank01(values: pd.Series, higher_better: bool = True) -> pd.Series:
    rank = values.rank(method="first", ascending=not higher_better)
    n = len(values)
    if n <= 1:
        return pd.Series(np.ones(n), index=values.index)
    return 1.0 - (rank - 1.0) / (n - 1.0)


def _add_epoch_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["base_rank01"] = out.groupby("tow", group_keys=False)["base_p_pass"].apply(_rank01)
    out["lgb_rank01"] = out.groupby("tow", group_keys=False)["p_lgb_timefold"].apply(_rank01)
    return out


def _base_top_frame(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("tow")["base_p_pass"].idxmax()
    base = df.loc[idx, ["tow", "label", "base_p_pass", "p_lgb_timefold"]].copy()
    base = base.rename(
        columns={
            "label": "base_label",
            "base_p_pass": "base_top_score",
            "p_lgb_timefold": "base_lgb",
        }
    )
    return base


def _top2_lgb_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tow, group in df.groupby("tow", sort=False):
        ordered = group.sort_values("p_lgb_timefold", ascending=False)
        top = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else top
        rows.append(
            {
                "tow": tow,
                "lgb_label": top["label"],
                "lgb_top": float(top["p_lgb_timefold"]),
                "lgb_second": float(second["p_lgb_timefold"]),
                "lgb_margin": float(top["p_lgb_timefold"] - second["p_lgb_timefold"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.features)
    fb = _fallback(args)
    epochs = np.array(sorted(df["tow"].unique()))

    pred = np.full(len(df), np.nan, dtype=float)
    for fold_i, fold_epochs in enumerate(np.array_split(epochs, int(args.folds)), start=1):
        test_mask = df["tow"].isin(fold_epochs).to_numpy()
        train = df.loc[~test_mask]
        test = df.loc[test_mask]
        pred[test_mask] = _train_predict(train, test, int(args.rounds))
        print(f"fold={fold_i} done", flush=True)
    df["p_lgb_timefold"] = pred
    df = _add_epoch_scores(df)
    df.to_csv(args.out_pred, index=False)

    base_score, base_w, base_epochs = _score(df, "base_p_pass", fb)
    raw_score, raw_w, raw_epochs = _score(df, "p_lgb_timefold", fb)
    rows = [
        {
            "policy": "base",
            "param": "",
            "score": base_score,
            "pass_w": base_w,
            "pass_epochs": base_epochs,
        },
        {
            "policy": "raw_lgb_timefold",
            "param": "",
            "score": raw_score,
            "pass_w": raw_w,
            "pass_epochs": raw_epochs,
        },
    ]

    for alpha in [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]:
        col = f"blend_rank_{alpha:g}"
        df[col] = (1.0 - alpha) * df["base_rank01"] + alpha * df["lgb_rank01"]
        score, pass_w, pass_epochs = _score(df, col, fb)
        rows.append(
            {
                "policy": "blend_rank",
                "param": f"alpha={alpha:g}",
                "score": score,
                "pass_w": pass_w,
                "pass_epochs": pass_epochs,
            }
        )

    base = _base_top_frame(df)
    lgb = _top2_lgb_frame(df)
    meta = base.merge(lgb, on="tow", how="inner")
    meta["same"] = meta["base_label"] == meta["lgb_label"]
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        for margin in [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
            override_tows = set(
                meta.loc[
                    (~meta["same"])
                    & (meta["lgb_top"] >= threshold)
                    & (meta["lgb_margin"] >= margin),
                    "tow",
                ].round(1)
            )
            col = f"guard_t{threshold:g}_m{margin:g}"
            df[col] = df["base_p_pass"]
            mask = df["tow"].round(1).isin(override_tows)
            df.loc[mask, col] = df.loc[mask, "p_lgb_timefold"]
            score, pass_w, pass_epochs = _score(df, col, fb)
            rows.append(
                {
                    "policy": "guard_override",
                    "param": f"top>={threshold:g},margin>={margin:g},n={len(override_tows)}",
                    "score": score,
                    "pass_w": pass_w,
                    "pass_epochs": pass_epochs,
                }
            )

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    out.to_csv(args.out_sweep, index=False)
    print(f"wrote {args.out_pred}")
    print(f"wrote {args.out_sweep}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
