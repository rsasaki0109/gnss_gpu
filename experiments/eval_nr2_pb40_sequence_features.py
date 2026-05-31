#!/usr/bin/env python3
"""Evaluate past-only sequence features for the n/r2 pb40 ranker."""

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
    _default_candidates,
    _load_candidates,
)
from eval_nr2_pb40_feature_variants import _fallback, _score  # noqa: E402

REPO = Path("/media/sasaki/aiueo/ai_coding_ws/gnss_gpu")

BASE_CLUSTER_NO_LABEL = [
    "base_p_pass",
    "rank_by_rms",
    "cluster_size_50cm",
    "cluster_size_25cm",
    "max_cluster_size_50cm",
    "is_in_max_cluster_50cm",
    "dist_to_median_m",
    "dist_to_hybrid_m",
]

SEQ_COLS = [
    "prev_same_dt_s",
    "prev_same_disp_m",
    "prev_same_speed_mps",
    "prev_same_valid",
    "prev_same_hyb_vel_diff_mps",
    "prev_same_hyb_pred_err_m",
    "prev_same_dist_to_hybrid_delta_m",
    "same_label_seen_1s",
    "same_label_seen_3s",
    "same_label_seen_10s",
]

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
    parser.add_argument(
        "--out-features",
        type=Path,
        default=RESULTS / "nr2_pb40_exact_sequence_features.csv",
    )
    parser.add_argument(
        "--out-scores",
        type=Path,
        default=RESULTS / "nr2_pb40_sequence_feature_scores.csv",
    )
    parser.add_argument(
        "--out-pred",
        type=Path,
        default=RESULTS / "nr2_pb40_sequence_timefold_predictions.csv",
    )
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--hybrid-pos-dir", type=Path, default=RESULTS / "libgnss_rtk_pos_v5")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=180)
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


def _hybrid_velocity(hybrid: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    last_t: float | None = None
    last_p: np.ndarray | None = None
    for tow in sorted(hybrid):
        pos = np.asarray(hybrid[tow], dtype=float)
        if last_t is not None and last_p is not None:
            dt = float(tow - last_t)
            if 0.0 < dt <= 0.6:
                out[tow] = (pos - last_p) / dt
        last_t = tow
        last_p = pos
    return out


def _load_position_rows(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    labels, dirs = _default_candidates(args)
    candidates = _load_candidates(labels, dirs, city=str(args.city), run=str(args.run))
    needed = {(round(float(row.tow), 1), str(row.label)) for row in df[["tow", "label"]].itertuples(index=False)}

    pos_rows: list[dict[str, object]] = []
    for label, pos_lookup, _diag in candidates:
        label_needed = {tow for tow, needed_label in needed if needed_label == label}
        if not label_needed:
            continue
        for tow in label_needed:
            pos = pos_lookup.get(tow)
            if pos is None:
                continue
            pos_rows.append(
                {
                    "tow": tow,
                    "label": label,
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                }
            )
    return pd.DataFrame(pos_rows)


def _add_sequence_features(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    pos_rows = _load_position_rows(args, df)
    work = df.merge(pos_rows, on=["tow", "label"], how="left")
    hybrid, _status = __import__(
        "exp_ppc_ctrbpf_fgo", fromlist=["_load_hybrid_pos_file"]
    )._load_hybrid_pos_file(args.hybrid_pos_dir / f"{args.city}_{args.run}_full.pos")
    hyb_vel = _hybrid_velocity(hybrid)
    hybrid_rows = []
    for tow, pos in hybrid.items():
        vel = hyb_vel.get(tow)
        hybrid_rows.append(
            {
                "tow": tow,
                "hyb_x": float(pos[0]),
                "hyb_y": float(pos[1]),
                "hyb_z": float(pos[2]),
                "hyb_vx": float(vel[0]) if vel is not None else np.nan,
                "hyb_vy": float(vel[1]) if vel is not None else np.nan,
                "hyb_vz": float(vel[2]) if vel is not None else np.nan,
            }
        )
    work = work.merge(pd.DataFrame(hybrid_rows), on="tow", how="left")

    for col in SEQ_COLS:
        work[col] = np.nan
    work["prev_same_valid"] = 0.0
    work["same_label_seen_1s"] = 0.0
    work["same_label_seen_3s"] = 0.0
    work["same_label_seen_10s"] = 0.0

    updates: list[pd.DataFrame] = []
    for _label, group in work.sort_values("tow").groupby("label", sort=False):
        g = group.copy()
        prev = g[["tow", "x", "y", "z", "dist_to_hybrid_m"]].shift(1)
        dt = g["tow"] - prev["tow"]
        disp = np.linalg.norm(
            g[["x", "y", "z"]].to_numpy(dtype=float)
            - prev[["x", "y", "z"]].to_numpy(dtype=float),
            axis=1,
        )
        valid = np.isfinite(disp) & (dt > 0.0) & (dt <= 1.0)
        speed = np.where(valid, disp / dt.clip(lower=1e-6), np.nan)
        cand_vel = (
            g[["x", "y", "z"]].to_numpy(dtype=float)
            - prev[["x", "y", "z"]].to_numpy(dtype=float)
        ) / dt.to_numpy(dtype=float).reshape(-1, 1)
        hyb_vel_arr = g[["hyb_vx", "hyb_vy", "hyb_vz"]].to_numpy(dtype=float)
        hyb_diff = np.linalg.norm(cand_vel - hyb_vel_arr, axis=1)
        hyb_pred = prev[["x", "y", "z"]].to_numpy(dtype=float) + hyb_vel_arr * dt.to_numpy(dtype=float).reshape(-1, 1)
        hyb_pred_err = np.linalg.norm(g[["x", "y", "z"]].to_numpy(dtype=float) - hyb_pred, axis=1)

        g["prev_same_dt_s"] = np.where(np.isfinite(dt), dt, np.nan)
        g["prev_same_disp_m"] = np.where(valid, disp, np.nan)
        g["prev_same_speed_mps"] = speed
        g["prev_same_valid"] = valid.astype(float)
        g["prev_same_hyb_vel_diff_mps"] = np.where(valid, hyb_diff, np.nan)
        g["prev_same_hyb_pred_err_m"] = np.where(valid, hyb_pred_err, np.nan)
        g["prev_same_dist_to_hybrid_delta_m"] = np.where(
            valid,
            g["dist_to_hybrid_m"].to_numpy(dtype=float) - prev["dist_to_hybrid_m"].to_numpy(dtype=float),
            np.nan,
        )
        g["same_label_seen_1s"] = ((dt > 0.0) & (dt <= 1.0)).astype(float)
        g["same_label_seen_3s"] = ((dt > 0.0) & (dt <= 3.0)).astype(float)
        g["same_label_seen_10s"] = ((dt > 0.0) & (dt <= 10.0)).astype(float)
        updates.append(g)

    out = pd.concat(updates, ignore_index=True)
    return out.drop(columns=["x", "y", "z", "hyb_x", "hyb_y", "hyb_z", "hyb_vx", "hyb_vy", "hyb_vz"])


def _train_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    rounds: int,
) -> np.ndarray:
    dataset = lgb.Dataset(
        train[features],
        label=train["is_pass_50cm"].astype(int),
        weight=train["path_weight"].clip(lower=1e-6),
    )
    model = lgb.train(LGB_PARAMS, dataset, num_boost_round=rounds)
    return model.predict(test[features])


def _timefold(df: pd.DataFrame, features: list[str], args: argparse.Namespace) -> np.ndarray:
    epochs = np.array(sorted(df["tow"].unique()))
    pred = np.full(len(df), np.nan, dtype=float)
    for fold_i, fold_epochs in enumerate(np.array_split(epochs, int(args.folds)), start=1):
        test_mask = df["tow"].isin(fold_epochs).to_numpy()
        pred[test_mask] = _train_predict(
            df.loc[~test_mask],
            df.loc[test_mask],
            features,
            int(args.rounds),
        )
        print(f"fold={fold_i} done features={len(features)}", flush=True)
    return pred


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.features)
    seq = _add_sequence_features(args, df)
    seq.to_csv(args.out_features, index=False)
    fb = _fallback(args)

    rows = []
    base_score, base_w, base_epochs = _score(seq, "base_p_pass", fb)
    rows.append({"variant": "base", "score": base_score, "pass_w": base_w, "pass_epochs": base_epochs})

    variants = {
        "base_cluster_no_label": BASE_CLUSTER_NO_LABEL,
        "base_cluster_seq": BASE_CLUSTER_NO_LABEL + SEQ_COLS,
        "seq_only": ["base_p_pass"] + SEQ_COLS,
        "cluster_seq_no_base": [c for c in BASE_CLUSTER_NO_LABEL if c != "base_p_pass"] + SEQ_COLS,
    }
    for name, features in variants.items():
        pred_col = f"pred_{name}"
        seq[pred_col] = _timefold(seq, features, args)
        score, pass_w, pass_epochs = _score(seq, pred_col, fb)
        rows.append(
            {
                "variant": name,
                "score": score,
                "pass_w": pass_w,
                "pass_epochs": pass_epochs,
            }
        )
        print(f"variant={name} score={score:.6f} pass_w={pass_w:.3f}", flush=True)

    seq.to_csv(args.out_pred, index=False)
    scores = pd.DataFrame(rows).sort_values("score", ascending=False)
    scores.to_csv(args.out_scores, index=False)
    print(f"wrote {args.out_features}")
    print(f"wrote {args.out_pred}")
    print(f"wrote {args.out_scores}")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
