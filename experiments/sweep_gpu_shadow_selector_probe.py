#!/usr/bin/env python3
"""Sweep GPU shadow selector penalty/rescue weights on a candidate pool."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_gpu_shadow_selector_probe import (
    DEFAULT_RISK_COL,
    RESULTS_DIR,
    _read_candidate_csv,
    _read_csv,
    _write_csv,
    _write_json,
    derive_shadow_coefficients,
    evaluate_rows,
    feature_join_keys,
    merge_epoch_gpu_features,
    prepare_probe_rows,
)


DEFAULT_INPUT_CSV = RESULTS_DIR / "selector_training_features_v6_tdcp.csv"
DEFAULT_FEATURE_CSV = RESULTS_DIR / "ppc_gpu_urban_shadow_selector_features_tokyo_run1_nav_smoke.csv"
DEFAULT_OUT_DIR = RESULTS_DIR / "gpu_shadow_selector_probe_real_candidates_tokyo_run1_sweep"


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("grid list must not be empty")
    return values


def load_prepared_rows(args: argparse.Namespace) -> tuple[list[dict], dict[str, int]]:
    feature_rows = _read_csv(args.gpu_feature_csv) if args.gpu_feature_csv is not None else []
    allowed_keys = None
    if args.keep_only_feature_epochs and feature_rows:
        allowed_keys = feature_join_keys(
            feature_rows,
            feature_source_run_id=args.feature_source_run_id,
            target_run_id=args.feature_target_run_id or args.candidate_run_id,
        )
    rows = _read_candidate_csv(
        args.input_csv,
        candidate_run_id=args.candidate_run_id,
        allowed_keys=allowed_keys,
    )
    matched = 0
    missing = 0
    if feature_rows:
        rows, matched, missing = merge_epoch_gpu_features(
            rows,
            feature_rows,
            feature_source_run_id=args.feature_source_run_id,
            target_run_id=args.feature_target_run_id or args.candidate_run_id,
            keep_only_matched=args.keep_only_feature_epochs,
        )
    if args.derive_shadow_coeffs:
        rows = derive_shadow_coefficients(rows)
    rows, synthesized = prepare_probe_rows(
        rows,
        input_mode=args.input_mode,
        risk_col=args.risk_col,
        robust_base_penalty=args.robust_base_penalty,
        robust_score_gain=args.robust_score_gain,
    )
    meta = {
        "source_rows": len(rows),
        "feature_matched_rows": matched,
        "feature_missing_rows": missing,
        "synthesized_candidates": int(synthesized),
    }
    return rows, meta


def sweep_rows(
    rows: list[dict],
    *,
    risk_col: str,
    base_score_col: str,
    truth_cost_col: str,
    penalty_weights: list[float],
    rescue_weights: list[float],
) -> tuple[list[dict], dict]:
    out_rows: list[dict] = []
    best = None
    for penalty_weight in penalty_weights:
        for rescue_weight in rescue_weights:
            _selected, summary, _buckets = evaluate_rows(
                rows,
                risk_col=risk_col,
                base_score_col=base_score_col,
                truth_cost_col=truth_cost_col,
                penalty_weight=penalty_weight,
                rescue_weight=rescue_weight,
            )
            row = {
                "penalty_weight": penalty_weight,
                "rescue_weight": rescue_weight,
                "epochs": summary["epochs"],
                "changed_epochs": summary["changed_epochs"],
                "change_rate": summary["change_rate"],
                "mean_risk": summary["mean_risk"],
                "mean_changed_risk": summary["mean_changed_risk"],
                "mean_baseline_truth_cost": summary.get("mean_baseline_truth_cost", 0.0),
                "mean_gpu_truth_cost": summary.get("mean_gpu_truth_cost", 0.0),
                "mean_gpu_minus_baseline_truth_cost": summary.get(
                    "mean_gpu_minus_baseline_truth_cost",
                    0.0,
                ),
                "improved_epochs": summary.get("improved_epochs", 0),
                "worse_epochs": summary.get("worse_epochs", 0),
                "mean_baseline_regret": summary.get("mean_baseline_regret", 0.0),
                "mean_gpu_regret": summary.get("mean_gpu_regret", 0.0),
            }
            out_rows.append(row)
            if best is None or (
                row["mean_gpu_minus_baseline_truth_cost"],
                row["worse_epochs"],
                -row["improved_epochs"],
            ) < (
                best["mean_gpu_minus_baseline_truth_cost"],
                best["worse_epochs"],
                -best["improved_epochs"],
            ):
                best = row
    summary = {
        "grid_rows": len(out_rows),
        "best": best or {},
    }
    return out_rows, summary


def run_sweep(args: argparse.Namespace) -> dict:
    rows, meta = load_prepared_rows(args)
    penalty_weights = _parse_float_list(args.penalty_weights)
    rescue_weights = _parse_float_list(args.rescue_weights)
    sweep, summary = sweep_rows(
        rows,
        risk_col=args.risk_col,
        base_score_col=args.base_score_col,
        truth_cost_col=args.truth_cost_col,
        penalty_weights=penalty_weights,
        rescue_weights=rescue_weights,
    )
    summary.update(
        {
            **meta,
            "input_csv": str(args.input_csv),
            "gpu_feature_csv": str(args.gpu_feature_csv) if args.gpu_feature_csv is not None else "",
            "candidate_run_id": args.candidate_run_id,
            "feature_source_run_id": args.feature_source_run_id,
            "feature_target_run_id": args.feature_target_run_id,
            "risk_col": args.risk_col,
            "base_score_col": args.base_score_col,
            "truth_cost_col": args.truth_cost_col,
            "penalty_weights": penalty_weights,
            "rescue_weights": rescue_weights,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = args.out_dir / "gpu_shadow_selector_probe_weight_sweep.csv"
    summary_json = args.out_dir / "gpu_shadow_selector_probe_weight_sweep_summary.json"
    _write_csv(sweep_csv, sweep)
    _write_json(summary_json, summary)
    return {
        "summary": summary,
        "sweep_csv": str(sweep_csv),
        "summary_json": str(summary_json),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--gpu-feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--input-mode", choices=("auto", "candidates", "epoch"), default="candidates")
    parser.add_argument("--candidate-run-id", default="tokyo_run1")
    parser.add_argument("--feature-source-run-id", default="tokyo_run1_nav")
    parser.add_argument("--feature-target-run-id", default="tokyo_run1")
    parser.add_argument("--keep-only-feature-epochs", action="store_true", default=True)
    parser.add_argument("--derive-shadow-coeffs", action="store_true", default=True)
    parser.add_argument("--risk-col", default=DEFAULT_RISK_COL)
    parser.add_argument("--base-score-col", default="rms")
    parser.add_argument("--truth-cost-col", default="err_3d_m")
    parser.add_argument("--penalty-weights", default="0,0.5,1,2,4")
    parser.add_argument("--rescue-weights", default="0,0.5,1,2,4")
    parser.add_argument("--robust-base-penalty", type=float, default=0.06)
    parser.add_argument("--robust-score-gain", type=float, default=0.80)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = run_sweep(args)
    best = result["summary"]["best"]
    print(
        "[gpu-shadow-selector-sweep] "
        f"grid={result['summary']['grid_rows']} "
        f"best_penalty={best.get('penalty_weight')} "
        f"best_rescue={best.get('rescue_weight')} "
        f"best_delta={best.get('mean_gpu_minus_baseline_truth_cost'):.6f}"
    )
    print(f"[gpu-shadow-selector-sweep] wrote {result['sweep_csv']}")
    print(f"[gpu-shadow-selector-sweep] wrote {result['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
