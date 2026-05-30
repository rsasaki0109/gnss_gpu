#!/usr/bin/env python3
"""Evaluate stronger past-only motion features for n/r2 pb40 ranker."""

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
from exp_ppc_ctrbpf_fgo import _load_hybrid_pos_file  # noqa: E402

BASE_COLS = [
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
    "same_label_seen_1s",
    "same_label_seen_3s",
]

MOTION_COLS = [
    "prev2_same_valid",
    "prev_same_speed_delta_mps",
    "prev_same_accel_mps2",
    "prev_base_label_same",
    "prev_base_pred_err_m",
    "prev_base_disp_m",
    "prev_cluster_pred_err_m",
    "prev_cluster_disp_delta_m",
    "prev_cluster_motion_diff_mps",
    "prev_centroid_pred_err_m",
    "prev_centroid_motion_diff_mps",
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
    "lambda_l2": 4.0,
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
    parser.add_argument("--rounds", type=int, default=160)
    parser.add_argument(
        "--out-features",
        type=Path,
        default=RESULTS / "nr2_pb40_exact_motion_features.csv",
    )
    parser.add_argument(
        "--out-pred",
        type=Path,
        default=RESULTS / "nr2_pb40_motion_timefold_predictions.csv",
    )
    parser.add_argument(
        "--out-scores",
        type=Path,
        default=RESULTS / "nr2_pb40_motion_feature_scores.csv",
    )
    parser.add_argument(
        "--out-sweep",
        type=Path,
        default=RESULTS / "nr2_pb40_motion_guard_sweep.csv",
    )
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


def _load_positions(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    labels, dirs = _default_candidates(args)
    candidates = _load_candidates(labels, dirs, city=str(args.city), run=str(args.run))
    needed = {
        (round(float(row.tow), 1), str(row.label))
        for row in df[["tow", "label"]].itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for label, pos_lookup, _diag in candidates:
        for tow, needed_label in needed:
            if needed_label != label:
                continue
            pos = pos_lookup.get(tow)
            if pos is None:
                continue
            rows.append(
                {
                    "tow": tow,
                    "label": label,
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                }
            )
    return pd.DataFrame(rows)


def _cluster_motion_tables(work: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for tow, group in work.groupby("tow", sort=True):
        pos = group[["x", "y", "z"]].to_numpy(dtype=float)
        valid = np.all(np.isfinite(pos), axis=1)
        pos = pos[valid]
        if pos.size == 0:
            continue
        diffs = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diffs, axis=2)
        c50 = (dist <= 0.5).sum(axis=1)
        max_i = int(np.argmax(c50))
        members = pos[dist[max_i] <= 0.5]
        rows.append(
            {
                "tow": float(tow),
                "cluster_x": float(members[:, 0].mean()),
                "cluster_y": float(members[:, 1].mean()),
                "cluster_z": float(members[:, 2].mean()),
                "centroid_x": float(pos[:, 0].mean()),
                "centroid_y": float(pos[:, 1].mean()),
                "centroid_z": float(pos[:, 2].mean()),
            }
        )
    table = pd.DataFrame(rows).sort_values("tow")
    prev = table.shift(1)
    dt = table["tow"] - prev["tow"]
    valid = (dt > 0.0) & (dt <= 1.0)
    for prefix in ["cluster", "centroid"]:
        cur = table[[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]].to_numpy(dtype=float)
        old = prev[[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]].to_numpy(dtype=float)
        vel = (cur - old) / dt.to_numpy(dtype=float).reshape(-1, 1)
        table[f"{prefix}_prev_dt_s"] = np.where(valid, dt, np.nan)
        table[f"{prefix}_prev_disp_m"] = np.where(valid, np.linalg.norm(cur - old, axis=1), np.nan)
        table[f"{prefix}_vx"] = np.where(valid, vel[:, 0], np.nan)
        table[f"{prefix}_vy"] = np.where(valid, vel[:, 1], np.nan)
        table[f"{prefix}_vz"] = np.where(valid, vel[:, 2], np.nan)
    return table, prev


def _add_features(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    pos = _load_positions(args, df)
    work = df.merge(pos, on=["tow", "label"], how="left")
    hybrid, _status = _load_hybrid_pos_file(args.hybrid_pos_dir / f"{args.city}_{args.run}_full.pos")
    hyb_vel = _hybrid_velocity(hybrid)
    hyb_rows = []
    for tow, hpos in hybrid.items():
        vel = hyb_vel.get(tow)
        hyb_rows.append(
            {
                "tow": tow,
                "hyb_x": float(hpos[0]),
                "hyb_y": float(hpos[1]),
                "hyb_z": float(hpos[2]),
                "hyb_vx": float(vel[0]) if vel is not None else np.nan,
                "hyb_vy": float(vel[1]) if vel is not None else np.nan,
                "hyb_vz": float(vel[2]) if vel is not None else np.nan,
            }
        )
    work = work.merge(pd.DataFrame(hyb_rows), on="tow", how="left")

    cluster, _prev_cluster = _cluster_motion_tables(work)
    work = work.merge(cluster, on="tow", how="left")

    base_idx = work.groupby("tow")["base_p_pass"].idxmax()
    base = work.loc[base_idx, ["tow", "label", "x", "y", "z"]].sort_values("tow").copy()
    base_prev = base.shift(1)
    base_prev = pd.DataFrame(
        {
            "tow": base["tow"],
            "prev_base_label": base_prev["label"],
            "prev_base_x": base_prev["x"],
            "prev_base_y": base_prev["y"],
            "prev_base_z": base_prev["z"],
            "prev_base_tow": base_prev["tow"],
        }
    )
    work = work.merge(base_prev, on="tow", how="left")

    updates: list[pd.DataFrame] = []
    for _label, group in work.sort_values("tow").groupby("label", sort=False):
        g = group.copy()
        prev1 = g[["tow", "x", "y", "z", "dist_to_hybrid_m"]].shift(1)
        prev2 = g[["tow", "x", "y", "z"]].shift(2)
        dt1 = g["tow"] - prev1["tow"]
        dt2 = prev1["tow"] - prev2["tow"]
        cur = g[["x", "y", "z"]].to_numpy(dtype=float)
        p1 = prev1[["x", "y", "z"]].to_numpy(dtype=float)
        p2 = prev2[["x", "y", "z"]].to_numpy(dtype=float)
        valid1 = np.isfinite(cur).all(axis=1) & np.isfinite(p1).all(axis=1) & (dt1 > 0.0) & (dt1 <= 1.0)
        valid2 = valid1 & np.isfinite(p2).all(axis=1) & (dt2 > 0.0) & (dt2 <= 1.0)
        v1 = (cur - p1) / dt1.to_numpy(dtype=float).reshape(-1, 1)
        v0 = (p1 - p2) / dt2.to_numpy(dtype=float).reshape(-1, 1)
        disp1 = np.linalg.norm(cur - p1, axis=1)
        speed1 = np.linalg.norm(v1, axis=1)
        hvel = g[["hyb_vx", "hyb_vy", "hyb_vz"]].to_numpy(dtype=float)
        pred_h = p1 + hvel * dt1.to_numpy(dtype=float).reshape(-1, 1)
        g["prev_same_dt_s"] = np.where(np.isfinite(dt1), dt1, np.nan)
        g["prev_same_disp_m"] = np.where(valid1, disp1, np.nan)
        g["prev_same_speed_mps"] = np.where(valid1, speed1, np.nan)
        g["prev_same_valid"] = valid1.astype(float)
        g["prev_same_hyb_vel_diff_mps"] = np.where(valid1, np.linalg.norm(v1 - hvel, axis=1), np.nan)
        g["prev_same_hyb_pred_err_m"] = np.where(valid1, np.linalg.norm(cur - pred_h, axis=1), np.nan)
        g["same_label_seen_1s"] = ((dt1 > 0.0) & (dt1 <= 1.0)).astype(float)
        g["same_label_seen_3s"] = ((dt1 > 0.0) & (dt1 <= 3.0)).astype(float)
        g["prev2_same_valid"] = valid2.astype(float)
        g["prev_same_speed_delta_mps"] = np.where(valid2, speed1 - np.linalg.norm(v0, axis=1), np.nan)
        g["prev_same_accel_mps2"] = np.where(valid2, np.linalg.norm(v1 - v0, axis=1) / dt1.clip(lower=1e-6), np.nan)
        updates.append(g)
    out = pd.concat(updates, ignore_index=True)

    cur = out[["x", "y", "z"]].to_numpy(dtype=float)
    prev_base = out[["prev_base_x", "prev_base_y", "prev_base_z"]].to_numpy(dtype=float)
    hvel = out[["hyb_vx", "hyb_vy", "hyb_vz"]].to_numpy(dtype=float)
    dt_base = out["tow"].to_numpy(dtype=float) - out["prev_base_tow"].to_numpy(dtype=float)
    valid_base = np.isfinite(prev_base).all(axis=1) & (dt_base > 0.0) & (dt_base <= 1.0)
    base_pred = prev_base + hvel * dt_base.reshape(-1, 1)
    out["prev_base_label_same"] = (out["label"] == out["prev_base_label"]).astype(float)
    out["prev_base_pred_err_m"] = np.where(valid_base, np.linalg.norm(cur - base_pred, axis=1), np.nan)
    out["prev_base_disp_m"] = np.where(valid_base, np.linalg.norm(cur - prev_base, axis=1), np.nan)

    for prefix in ["cluster", "centroid"]:
        center = out[[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]].to_numpy(dtype=float)
        vel = out[[f"{prefix}_vx", f"{prefix}_vy", f"{prefix}_vz"]].to_numpy(dtype=float)
        dt = out[f"{prefix}_prev_dt_s"].to_numpy(dtype=float)
        pred = center - vel * dt.reshape(-1, 1)
        candidate_motion = cur - pred
        center_motion = vel * dt.reshape(-1, 1)
        out[f"prev_{prefix}_pred_err_m"] = np.linalg.norm(cur - (pred + center_motion), axis=1)
        out[f"prev_{prefix}_motion_diff_mps"] = np.linalg.norm(candidate_motion / dt.reshape(-1, 1) - vel, axis=1)
    out["prev_cluster_disp_delta_m"] = out["prev_same_disp_m"] - out["cluster_prev_disp_m"]

    drop_cols = [
        "x",
        "y",
        "z",
        "hyb_x",
        "hyb_y",
        "hyb_z",
        "hyb_vx",
        "hyb_vy",
        "hyb_vz",
        "prev_base_x",
        "prev_base_y",
        "prev_base_z",
        "prev_base_tow",
        "prev_base_label",
    ]
    generated_prefix_cols = []
    for prefix in ["cluster", "centroid"]:
        generated_prefix_cols.extend(
            [
                f"{prefix}_x",
                f"{prefix}_y",
                f"{prefix}_z",
                f"{prefix}_vx",
                f"{prefix}_vy",
                f"{prefix}_vz",
                f"{prefix}_prev_dt_s",
                f"{prefix}_prev_disp_m",
            ]
        )
    drop_cols.extend(generated_prefix_cols)
    return out.drop(columns=[c for c in drop_cols if c in out.columns])


def _train_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], rounds: int) -> np.ndarray:
    data = lgb.Dataset(
        train[features],
        label=train["is_pass_50cm"].astype(int),
        weight=train["path_weight"].clip(lower=1e-6),
    )
    model = lgb.train(LGB_PARAMS, data, num_boost_round=rounds)
    return model.predict(test[features])


def _timefold(df: pd.DataFrame, features: list[str], args: argparse.Namespace) -> np.ndarray:
    pred = np.full(len(df), np.nan, dtype=float)
    epochs = np.array(sorted(df["tow"].unique()))
    for fold_i, fold_epochs in enumerate(np.array_split(epochs, int(args.folds)), start=1):
        mask = df["tow"].isin(fold_epochs).to_numpy()
        pred[mask] = _train_predict(df.loc[~mask], df.loc[mask], features, int(args.rounds))
        print(f"fold={fold_i} features={len(features)}", flush=True)
    return pred


def _guard_sweep(df: pd.DataFrame, pred_col: str, fb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_score, base_w, base_epochs = _score(df, "base_p_pass", fb)
    raw_score, raw_w, raw_epochs = _score(df, pred_col, fb)
    rows.append(("base", "", base_score, base_w, base_epochs))
    rows.append(("raw", pred_col, raw_score, raw_w, raw_epochs))
    meta = []
    for tow, group in df.groupby("tow", sort=False):
        base = group.loc[group["base_p_pass"].idxmax()]
        ordered = group.sort_values(pred_col, ascending=False)
        top = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else top
        meta.append(
            {
                "tow": round(float(tow), 1),
                "base_label": str(base["label"]),
                "top_label": str(top["label"]),
                "top": float(top[pred_col]),
                "margin": float(top[pred_col] - second[pred_col]),
            }
        )
    meta_df = pd.DataFrame(meta)
    for th in [0.80, 0.85, 0.90, 0.93, 0.95, 0.97]:
        for margin in [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
            tows = set(
                meta_df.loc[
                    (meta_df["base_label"] != meta_df["top_label"])
                    & (meta_df["top"] >= th)
                    & (meta_df["margin"] >= margin),
                    "tow",
                ]
            )
            col = f"guard_{th:g}_{margin:g}"
            df[col] = df["base_p_pass"]
            mask = df["tow"].round(1).isin(tows)
            df.loc[mask, col] = df.loc[mask, pred_col]
            score, pass_w, pass_epochs = _score(df, col, fb)
            rows.append(("guard", f"th={th:g},margin={margin:g},n={len(tows)}", score, pass_w, pass_epochs))
    return pd.DataFrame(rows, columns=["policy", "param", "score", "pass_w", "pass_epochs"]).sort_values("score", ascending=False)


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.features)
    motion = _add_features(args, df)
    motion.to_csv(args.out_features, index=False)
    fb = _fallback(args)
    variants = {
        "base_cluster": BASE_COLS,
        "seq_only": ["base_p_pass"] + SEQ_COLS,
        "motion_only": ["base_p_pass"] + MOTION_COLS,
        "seq_motion": ["base_p_pass"] + SEQ_COLS + MOTION_COLS,
        "cluster_motion": BASE_COLS + MOTION_COLS,
        "cluster_seq_motion": BASE_COLS + SEQ_COLS + MOTION_COLS,
    }
    rows = []
    base_score, base_w, base_epochs = _score(motion, "base_p_pass", fb)
    rows.append({"variant": "base", "score": base_score, "pass_w": base_w, "pass_epochs": base_epochs})
    for name, features in variants.items():
        col = f"pred_{name}"
        motion[col] = _timefold(motion, features, args)
        score, pass_w, pass_epochs = _score(motion, col, fb)
        rows.append({"variant": name, "score": score, "pass_w": pass_w, "pass_epochs": pass_epochs})
        print(f"{name}: {score:.6f}", flush=True)
    motion.to_csv(args.out_pred, index=False)
    scores = pd.DataFrame(rows).sort_values("score", ascending=False)
    scores.to_csv(args.out_scores, index=False)
    best_pred = scores[scores["variant"] != "base"].iloc[0]["variant"]
    sweep = _guard_sweep(motion, f"pred_{best_pred}", fb)
    sweep.to_csv(args.out_sweep, index=False)
    print(scores.to_string(index=False))
    print()
    print(sweep.head(25).to_string(index=False))
    print(f"wrote {args.out_features}")
    print(f"wrote {args.out_pred}")
    print(f"wrote {args.out_scores}")
    print(f"wrote {args.out_sweep}")


if __name__ == "__main__":
    main()
