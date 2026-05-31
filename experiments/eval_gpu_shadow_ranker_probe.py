#!/usr/bin/env python3
"""Cross-validated ranker probe for GPU urban-shadow selector features.

This is a small, dependency-light check before wiring GPU shadow columns into
the full LightGBM selector training scripts.  It joins epoch-level GPU features
onto a real candidate-level selector CSV, derives candidate-dependent shadow
interaction coefficients, and compares:

* `rms_selector`: current simple residual ordering.
* `base_ranker`: ridge regressor using existing selector features.
* `gpu_ranker`: same ridge regressor plus GPU shadow interaction features.
* `oracle`: lowest truth error in the candidate pool.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_gpu_shadow_selector_probe import (  # noqa: E402
    DEFAULT_RISK_COL,
    RESULTS_DIR,
    _epoch_key,
    _finite_float,
    _read_candidate_csv,
    _read_csv,
    _to_float,
    _write_csv,
    _write_json,
    derive_shadow_coefficients,
    feature_join_keys,
    merge_epoch_gpu_features,
)


DEFAULT_INPUT_CSV = RESULTS_DIR / "selector_training_features_v6_tdcp.csv"
DEFAULT_FEATURE_CSV = RESULTS_DIR / "ppc_gpu_urban_shadow_selector_features_tokyo_run1_nav_smoke.csv"
DEFAULT_OUT_DIR = RESULTS_DIR / "gpu_shadow_ranker_probe_tokyo_run1"

BASE_FEATURE_COLS = [
    "rms",
    "ratio",
    "abs_max",
    "update_rows",
    "sats",
    "status",
    "pdop",
    "baseline_m",
    "spp_valid",
    "spp_sats",
    "spp_pdop",
    "candidate_vs_spp_m",
    "candidate_jump_m",
    "cluster_size_50cm",
    "cluster_size_25cm",
    "cluster_size_10cm",
    "max_cluster_size_50cm",
    "is_in_max_cluster_50cm",
    "n_clusters_50cm",
    "n_clusters_50cm_ge3",
    "cluster_min_rms_50cm",
    "cluster_min_abs_max_50cm",
    "dist_to_max_cluster_centroid_m",
    "delta_pos_norm_m",
    "delta_pos_vs_median_m",
    "delta_pos_2step_m",
    "delta_pos_3step_m",
    "delta_pos_vertical_m",
    "delta_pos_horizontal_m",
    "delta_pos_accel_m",
    "rank_by_rms",
    "n_candidates_in_epoch",
    "dist_to_median_m",
]

GPU_FEATURE_COLS = [
    "gpu_urban_shadow_risk_score",
    "gpu_urban_mean_blocked_ratio",
    "gpu_urban_max_blocked_ratio",
    "gpu_urban_low_elev_blocked_ratio",
    "gpu_urban_expected_nlos_bias_m",
    "gpu_urban_route_weight_delta_log",
    "gpu_urban_particle_blocked_mean",
    "gpu_urban_particle_blocked_std",
    "gpu_urban_particle_shadow_contrast",
    "gpu_urban_n_sat",
    "gpu_urban_n_nlos",
    "gpu_shadow_penalty_coeff",
    "gpu_shadow_rescue_coeff",
    "gpu_shadow_risk_x_penalty",
    "gpu_shadow_risk_x_rescue",
    "gpu_shadow_risk_x_not_max_cluster",
    "gpu_shadow_risk_x_rank",
    "gpu_shadow_risk_x_dist_median",
    "gpu_shadow_risk_x_jump",
    "gpu_shadow_risk_x_vertical",
    "gpu_shadow_risk_x_rms",
    "gpu_shadow_risk_x_abs_max",
]


def _sanitize(value: float, default: float = 0.0) -> float:
    return value if math.isfinite(value) else default


def add_gpu_interaction_features(rows: list[dict], *, risk_col: str = DEFAULT_RISK_COL) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        risk = max(0.0, _finite_float(row.get(risk_col), 0.0))
        penalty = _finite_float(row.get("gpu_shadow_penalty_coeff"), 0.0)
        rescue = _finite_float(row.get("gpu_shadow_rescue_coeff"), 0.0)
        item["gpu_shadow_risk_x_penalty"] = risk * penalty
        item["gpu_shadow_risk_x_rescue"] = risk * rescue
        item["gpu_shadow_risk_x_not_max_cluster"] = risk * (
            1.0 - max(0.0, min(1.0, _finite_float(row.get("is_in_max_cluster_50cm"), 0.0)))
        )
        item["gpu_shadow_risk_x_rank"] = risk * max(_finite_float(row.get("rank_by_rms"), 1.0) - 1.0, 0.0)
        item["gpu_shadow_risk_x_dist_median"] = risk * max(_finite_float(row.get("dist_to_median_m"), 0.0), 0.0)
        item["gpu_shadow_risk_x_jump"] = risk * max(_finite_float(row.get("candidate_jump_m"), 0.0), 0.0)
        item["gpu_shadow_risk_x_vertical"] = risk * max(_finite_float(row.get("delta_pos_vertical_m"), 0.0), 0.0)
        item["gpu_shadow_risk_x_rms"] = risk * max(_finite_float(row.get("rms"), 0.0), 0.0)
        item["gpu_shadow_risk_x_abs_max"] = risk * max(_finite_float(row.get("abs_max"), 0.0), 0.0)
        out.append(item)
    return out


def load_joined_rows(args: argparse.Namespace) -> tuple[list[dict], dict[str, int]]:
    feature_rows = _read_csv(args.gpu_feature_csv)
    allowed_keys = feature_join_keys(
        feature_rows,
        feature_source_run_id=args.feature_source_run_id,
        target_run_id=args.feature_target_run_id or args.candidate_run_id,
    )
    candidate_rows = _read_candidate_csv(
        args.input_csv,
        candidate_run_id=args.candidate_run_id,
        allowed_keys=allowed_keys,
    )
    rows, matched, missing = merge_epoch_gpu_features(
        candidate_rows,
        feature_rows,
        feature_source_run_id=args.feature_source_run_id,
        target_run_id=args.feature_target_run_id or args.candidate_run_id,
        keep_only_matched=True,
    )
    rows = derive_shadow_coefficients(rows)
    rows = add_gpu_interaction_features(rows, risk_col=args.risk_col)
    return rows, {
        "source_rows": len(rows),
        "feature_matched_rows": matched,
        "feature_missing_rows": missing,
    }


def _row_features(row: dict, columns: list[str]) -> list[float]:
    return [_sanitize(_to_float(row.get(col), 0.0), 0.0) for col in columns]


def _ridge_fit_predict(
    train_rows: list[dict],
    test_rows: list[dict],
    *,
    feature_cols: list[str],
    target_col: str,
    weight_col: str,
    ridge_lambda: float,
) -> np.ndarray:
    x_train = np.asarray([_row_features(row, feature_cols) for row in train_rows], dtype=np.float64)
    y_train = np.log1p(
        np.asarray([max(_finite_float(row.get(target_col), 0.0), 0.0) for row in train_rows], dtype=np.float64)
    )
    w_train = np.asarray([max(_finite_float(row.get(weight_col), 1.0), 1e-6) for row in train_rows], dtype=np.float64)
    x_test = np.asarray([_row_features(row, feature_cols) for row in test_rows], dtype=np.float64)
    if x_train.size == 0 or x_test.size == 0:
        return np.zeros(len(test_rows), dtype=np.float64)

    mu = np.mean(x_train, axis=0)
    sigma = np.std(x_train, axis=0)
    sigma[sigma < 1e-9] = 1.0
    x_train = (x_train - mu) / sigma
    x_test = (x_test - mu) / sigma
    x_train = np.column_stack([np.ones(len(x_train), dtype=np.float64), x_train])
    x_test = np.column_stack([np.ones(len(x_test), dtype=np.float64), x_test])

    sqrt_w = np.sqrt(w_train)
    xw = x_train * sqrt_w[:, None]
    yw = y_train * sqrt_w
    reg = np.eye(xw.shape[1], dtype=np.float64) * float(ridge_lambda)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)
    return np.expm1(x_test @ beta)


def _fold_for_epochs(rows: list[dict], n_folds: int) -> dict[tuple[str, float], int]:
    keys = sorted({_epoch_key(row) for row in rows})
    return {key: idx % max(1, n_folds) for idx, key in enumerate(keys)}


def _select_min(group: list[dict], score_col: str) -> dict:
    return min(
        group,
        key=lambda row: (
            _finite_float(row.get(score_col), float("inf")),
            str(row.get("label", "")),
        ),
    )


def _score_selection(rows: list[dict], method_col: str, *, target_col: str, weight_col: str) -> dict:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_epoch_key(row)].append(row)
    selected = [_select_min(group, method_col) for group in grouped.values()]
    errors = [_finite_float(row.get(target_col), 0.0) for row in selected]
    weights = [max(_finite_float(row.get(weight_col), 1.0), 1e-6) for row in selected]
    pass50 = [1.0 if err <= 0.5 else 0.0 for err in errors]
    total_w = sum(weights) or 1.0
    return {
        "epochs": len(selected),
        "mean_error_m": float(np.mean(errors)) if errors else 0.0,
        "p50_error_m": float(np.percentile(errors, 50.0)) if errors else 0.0,
        "p95_error_m": float(np.percentile(errors, 95.0)) if errors else 0.0,
        "weighted_pass_50cm": float(sum(w * ok for w, ok in zip(weights, pass50)) / total_w),
    }


def cross_validated_ranker_probe(
    rows: list[dict],
    *,
    base_feature_cols: list[str],
    gpu_feature_cols: list[str],
    target_col: str = "err_3d_m",
    weight_col: str = "path_weight",
    n_folds: int = 5,
    ridge_lambda: float = 10.0,
) -> tuple[list[dict], dict]:
    fold_by_key = _fold_for_epochs(rows, n_folds)
    eval_rows: list[dict] = []

    for fold in range(max(1, n_folds)):
        train_rows = [row for row in rows if fold_by_key[_epoch_key(row)] != fold]
        test_rows = [row for row in rows if fold_by_key[_epoch_key(row)] == fold]
        base_pred = _ridge_fit_predict(
            train_rows,
            test_rows,
            feature_cols=base_feature_cols,
            target_col=target_col,
            weight_col=weight_col,
            ridge_lambda=ridge_lambda,
        )
        gpu_pred = _ridge_fit_predict(
            train_rows,
            test_rows,
            feature_cols=base_feature_cols + gpu_feature_cols,
            target_col=target_col,
            weight_col=weight_col,
            ridge_lambda=ridge_lambda,
        )
        for row, base_score, gpu_score in zip(test_rows, base_pred, gpu_pred):
            out = dict(row)
            out["cv_fold"] = fold
            out["base_ranker_score"] = float(base_score)
            out["gpu_ranker_score"] = float(gpu_score)
            out["rms_selector_score"] = _finite_float(row.get("rms"), float("inf"))
            out["oracle_score"] = _finite_float(row.get(target_col), float("inf"))
            eval_rows.append(out)

    selected_rows = _selection_rows(eval_rows, target_col=target_col, weight_col=weight_col)
    summary = summarize_ranker_probe(eval_rows, selected_rows, target_col=target_col, weight_col=weight_col)
    return selected_rows, summary


def _selection_rows(rows: list[dict], *, target_col: str, weight_col: str) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_epoch_key(row)].append(row)

    out = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        rms = _select_min(group, "rms_selector_score")
        base = _select_min(group, "base_ranker_score")
        gpu = _select_min(group, "gpu_ranker_score")
        oracle = _select_min(group, "oracle_score")
        risk = max(_finite_float(row.get("gpu_urban_shadow_risk_score"), 0.0) for row in group)
        out.append(
            {
                "run_id": key[0],
                "tow": key[1],
                "gpu_urban_shadow_risk_score": risk,
                "n_candidates": len(group),
                "rms_label": rms.get("label", ""),
                "base_label": base.get("label", ""),
                "gpu_label": gpu.get("label", ""),
                "oracle_label": oracle.get("label", ""),
                "rms_error_m": _finite_float(rms.get(target_col), 0.0),
                "base_error_m": _finite_float(base.get(target_col), 0.0),
                "gpu_error_m": _finite_float(gpu.get(target_col), 0.0),
                "oracle_error_m": _finite_float(oracle.get(target_col), 0.0),
                "path_weight": max(_finite_float(oracle.get(weight_col), 1.0), 1e-6),
                "gpu_changed_vs_base": float(gpu.get("label", "") != base.get("label", "")),
                "gpu_changed_vs_rms": float(gpu.get("label", "") != rms.get("label", "")),
                "gpu_minus_base_error_m": _finite_float(gpu.get(target_col), 0.0)
                - _finite_float(base.get(target_col), 0.0),
                "gpu_minus_rms_error_m": _finite_float(gpu.get(target_col), 0.0)
                - _finite_float(rms.get(target_col), 0.0),
            }
        )
    return out


def summarize_ranker_probe(
    eval_rows: list[dict],
    selected_rows: list[dict],
    *,
    target_col: str,
    weight_col: str,
) -> dict:
    method_cols = {
        "rms_selector": "rms_selector_score",
        "base_ranker": "base_ranker_score",
        "gpu_ranker": "gpu_ranker_score",
        "oracle": "oracle_score",
    }
    method_summary = {
        name: _score_selection(eval_rows, col, target_col=target_col, weight_col=weight_col)
        for name, col in method_cols.items()
    }
    deltas = [_finite_float(row.get("gpu_minus_base_error_m"), 0.0) for row in selected_rows]
    risk_changed = [
        _finite_float(row.get("gpu_urban_shadow_risk_score"), 0.0)
        for row in selected_rows
        if _finite_float(row.get("gpu_changed_vs_base"), 0.0) > 0.5
    ]
    return {
        "epochs": len(selected_rows),
        "candidate_rows": len(eval_rows),
        "method_summary": method_summary,
        "gpu_vs_base_changed_epochs": int(sum(_finite_float(row.get("gpu_changed_vs_base"), 0.0) > 0.5 for row in selected_rows)),
        "gpu_vs_rms_changed_epochs": int(sum(_finite_float(row.get("gpu_changed_vs_rms"), 0.0) > 0.5 for row in selected_rows)),
        "gpu_vs_base_improved_epochs": int(sum(value < -1e-12 for value in deltas)),
        "gpu_vs_base_worse_epochs": int(sum(value > 1e-12 for value in deltas)),
        "gpu_minus_base_mean_error_m": float(np.mean(deltas)) if deltas else 0.0,
        "gpu_changed_mean_risk": float(np.mean(risk_changed)) if risk_changed else 0.0,
    }


def run_probe(args: argparse.Namespace) -> dict:
    rows, join_meta = load_joined_rows(args)
    selected_rows, summary = cross_validated_ranker_probe(
        rows,
        base_feature_cols=BASE_FEATURE_COLS,
        gpu_feature_cols=GPU_FEATURE_COLS,
        target_col=args.target_col,
        weight_col=args.weight_col,
        n_folds=args.n_folds,
        ridge_lambda=args.ridge_lambda,
    )
    summary.update(
        {
            **join_meta,
            "input_csv": str(args.input_csv),
            "gpu_feature_csv": str(args.gpu_feature_csv),
            "candidate_run_id": args.candidate_run_id,
            "feature_source_run_id": args.feature_source_run_id,
            "feature_target_run_id": args.feature_target_run_id,
            "target_col": args.target_col,
            "weight_col": args.weight_col,
            "n_folds": args.n_folds,
            "ridge_lambda": args.ridge_lambda,
            "base_feature_count": len(BASE_FEATURE_COLS),
            "gpu_feature_count": len(GPU_FEATURE_COLS),
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = args.out_dir / "gpu_shadow_ranker_probe_selected_rows.csv"
    summary_json = args.out_dir / "gpu_shadow_ranker_probe_summary.json"
    _write_csv(rows_csv, selected_rows)
    _write_json(summary_json, summary)
    return {
        "summary": summary,
        "rows_csv": str(rows_csv),
        "summary_json": str(summary_json),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--gpu-feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--candidate-run-id", default="tokyo_run1")
    parser.add_argument("--feature-source-run-id", default="tokyo_run1_nav")
    parser.add_argument("--feature-target-run-id", default="tokyo_run1")
    parser.add_argument("--risk-col", default=DEFAULT_RISK_COL)
    parser.add_argument("--target-col", default="err_3d_m")
    parser.add_argument("--weight-col", default="path_weight")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--ridge-lambda", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = run_probe(args)
    summary = result["summary"]
    base = summary["method_summary"]["base_ranker"]["mean_error_m"]
    gpu = summary["method_summary"]["gpu_ranker"]["mean_error_m"]
    print(
        "[gpu-shadow-ranker-probe] "
        f"epochs={summary['epochs']} candidates={summary['candidate_rows']} "
        f"base_mean={base:.6f} gpu_mean={gpu:.6f} "
        f"delta={summary['gpu_minus_base_mean_error_m']:.6f}"
    )
    print(f"[gpu-shadow-ranker-probe] wrote {result['summary_json']}")
    print(f"[gpu-shadow-ranker-probe] wrote {result['rows_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
