#!/usr/bin/env python3
"""Rank moving trajectory offsets with fixed wide-lane DD ranges."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT / "experiments"))

from analyze_wp29_static_grid_widelane_shadow import widelane_residual_scores  # noqa: E402
from exp_wp23b_basin_ar import _build_dd_measurements  # noqa: E402
from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.stop_segment_static import _dd_expected_and_jacobian_m  # noqa: E402
from gnss_gpu.widelane import WidelaneDDPseudorangeComputer  # noqa: E402


def _trajectory(path: Path) -> dict[int, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return {
            int(row["epoch"]): np.asarray(
                [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
                dtype=np.float64,
            )
            for row in csv.DictReader(fh)
        }


def candidate_offsets_from_artifact(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept the native offset artifact or WP31 block-ambiguity hypotheses."""
    if "candidates" in source:
        return list(source["candidates"])
    output = []
    for row in source.get("hypotheses", []):
        output.append(
            {
                "candidate_id": int(row["seed_id"]),
                "offset_ecef_m": list(row["offset_ecef_m"]),
                "audit_sub50cm_epochs": int(row.get("audit_sub50cm_epochs", 0)),
                "audit_rms_m": float(row.get("audit_median_error_m", float("nan"))),
            }
        )
    return output


def fit_dynamic_offset(
    rows: list[tuple[float, np.ndarray, float, float]],
    *,
    degree: int,
    ridge: float,
    huber_k: float,
    iterations: int = 10,
    prior_coefficients: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Fit polynomial ECEF offsets from ``(t, jacobian, residual, weight)`` rows."""
    if not rows:
        raise ValueError("dynamic offset fit needs evidence rows")
    design = []
    target = []
    base_weight = []
    for t, jacobian, residual, weight in rows:
        basis = np.asarray([float(t) ** power for power in range(degree + 1)])
        design.append(np.outer(basis, np.asarray(jacobian, dtype=np.float64)).ravel())
        target.append(float(residual))
        base_weight.append(max(float(weight), 1.0e-6))
    a = np.asarray(design, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    base = np.asarray(base_weight, dtype=np.float64)
    robust = np.ones_like(y)
    prior = (
        np.zeros(a.shape[1], dtype=np.float64)
        if prior_coefficients is None
        else np.asarray(prior_coefficients, dtype=np.float64).reshape(-1)
    )
    if prior.size != a.shape[1]:
        raise ValueError("dynamic offset prior has the wrong shape")
    coefficients = prior.copy()
    ridge_diag = np.full(a.shape[1], max(float(ridge), 0.0) * len(y))
    for _ in range(max(int(iterations), 1)):
        weight = base * robust
        hessian = a.T @ (weight[:, None] * a) + np.diag(ridge_diag)
        gradient = a.T @ (weight * (y - a @ prior))
        coefficients = prior + np.linalg.solve(
            hessian + 1.0e-9 * np.eye(hessian.shape[0]), gradient
        )
        residuals = a @ coefficients - y
        scale = max(1.4826 * float(np.median(np.abs(residuals - np.median(residuals)))), 0.05)
        threshold = max(float(huber_k), 0.1) * scale
        robust = np.minimum(1.0, threshold / np.maximum(np.abs(residuals), 1.0e-12))
    objective = float(np.average(np.square(a @ coefficients - y), weights=base))
    return coefficients.reshape(degree + 1, 3), objective


def evaluate_dynamic_offset(coefficients: np.ndarray, t: float) -> np.ndarray:
    basis = np.asarray([float(t) ** power for power in range(len(coefficients))])
    return basis @ np.asarray(coefficients, dtype=np.float64)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.candidates_json.read_text(encoding="utf-8"))
    candidates = candidate_offsets_from_artifact(source)
    base_positions = (
        _trajectory(args.warmup_trajectory) if args.warmup_trajectory else {}
    )
    base_positions.update(_trajectory(args.trajectory))
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=int(args.end), systems=("G", "R", "E", "C", "J")
    )
    computer = WidelaneDDPseudorangeComputer(
        args.data_dir / "base.obs",
        args.data_dir / "rover.obs",
        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
        allowed_systems=("G", "J"),
        min_epochs=int(args.min_epochs),
        max_std_cycles=float(args.max_std_cycles),
        ratio_threshold=float(args.ratio_threshold),
        min_fix_rate=float(args.min_fix_rate),
    )
    residuals: list[list[float]] = [[] for _ in candidates]
    block_residuals: list[list[list[float]]] = [
        [[] for _ in range(int(args.bootstrap_blocks))] for _ in candidates
    ]
    evidence_epochs = candidate_pairs = fixed_pairs = 0
    dynamic_rows: list[tuple[float, np.ndarray, float, float]] = []
    for epoch in range(int(args.end)):
        approximate = base_positions[epoch]
        measurements = _build_dd_measurements(
            np.asarray(data["sat_ecef"][epoch], dtype=np.float64),
            np.asarray(data["system_ids"][epoch], dtype=np.int32),
            list(data["used_prns"][epoch]),
            np.asarray(data["weights"][epoch], dtype=np.float64),
            approximate,
            ("G", "E", "J", "C"),
            min_elevation_deg=-90.0,
            min_snr=0.0,
            keep_best=0,
        )
        dd, stats = computer.compute_dd(
            float(data["times"][epoch]),
            measurements,
            rover_position_approx=approximate,
            min_common_sats=4,
            rover_weights=np.asarray(data["weights"][epoch], dtype=np.float64),
        )
        if epoch < int(args.start) or (epoch - int(args.stride_origin)) % int(args.stride):
            continue
        candidate_pairs += int(stats.n_candidate_pairs)
        fixed_pairs += int(stats.n_fixed_pairs)
        if dd is None:
            continue
        evidence_epochs += 1
        block = min(
            int(args.bootstrap_blocks) - 1,
            int(args.bootstrap_blocks)
            * (epoch - int(args.start))
            // max(int(args.end) - int(args.start), 1),
        )
        measured = np.asarray(dd.dd_pseudorange_m, dtype=np.float64).ravel()
        time_normalized = (
            2.0 * (epoch - int(args.start)) / max(int(args.end) - int(args.start) - 1, 1)
            - 1.0
        )
        dd_weights = np.asarray(dd.dd_weights, dtype=np.float64).ravel()
        for row_index in range(len(measured)):
            expected, jacobian = _dd_expected_and_jacobian_m(
                approximate,
                np.asarray(dd.sat_ecef_k[row_index], dtype=np.float64),
                np.asarray(dd.sat_ecef_ref[row_index], dtype=np.float64),
                float(dd.base_range_k[row_index]),
                float(dd.base_range_ref[row_index]),
            )
            dynamic_rows.append(
                (
                    time_normalized,
                    jacobian,
                    float(measured[row_index] - expected),
                    float(dd_weights[row_index]) if row_index < len(dd_weights) else 1.0,
                )
            )
        for candidate_index, candidate in enumerate(candidates):
            position = approximate + np.asarray(candidate["offset_ecef_m"], dtype=np.float64)
            for row_index in range(len(measured)):
                expected, _ = _dd_expected_and_jacobian_m(
                    position,
                    np.asarray(dd.sat_ecef_k[row_index], dtype=np.float64),
                    np.asarray(dd.sat_ecef_ref[row_index], dtype=np.float64),
                    float(dd.base_range_k[row_index]),
                    float(dd.base_range_ref[row_index]),
                )
                residuals[candidate_index].append(float(measured[row_index] - expected))
                block_residuals[candidate_index][block].append(
                    float(measured[row_index] - expected)
                )
    rows = []
    for candidate_index, (candidate, values) in enumerate(zip(candidates, residuals)):
        offset = np.asarray(candidate["offset_ecef_m"], dtype=np.float64)
        block_scores = [
            widelane_residual_scores(block_values, sigma_m=float(args.residual_sigma_m))[
                "widelane_rms_m"
            ]
            + float(args.offset_prior_weight) * float(np.dot(offset, offset))
            for block_values in block_residuals[candidate_index]
        ]
        full_scores = widelane_residual_scores(
            values, sigma_m=float(args.residual_sigma_m)
        )
        rows.append(
            {
                "candidate_id": int(candidate["candidate_id"]),
                "offset_ecef_m": offset.tolist(),
                "audit_sub50cm_epochs": int(candidate.get("audit_sub50cm_epochs", 0)),
                "audit_rms_m": float(candidate.get("audit_rms_m", float("nan"))),
                "n_widelane_rows": len(values),
                **full_scores,
                "offset_norm_m": float(np.linalg.norm(offset)),
                "regularized_widelane_score": full_scores["widelane_rms_m"]
                + float(args.offset_prior_weight) * float(np.dot(offset, offset)),
                "bootstrap_scores": block_scores,
                "bootstrap_wins": 0,
            }
        )
    for score_name in ("widelane_rms_m", "widelane_median_abs_m", "widelane_cauchy_mean"):
        finite = [
            index for index, row in enumerate(rows) if np.isfinite(row[score_name])
        ]
        for row in rows:
            row[f"{score_name}_rank"] = None
        for rank, index in enumerate(
            sorted(finite, key=lambda value: rows[value][score_name]), start=1
        ):
            rows[index][f"{score_name}_rank"] = rank
    finite_regularized = [
        index
        for index, row in enumerate(rows)
        if np.isfinite(row["regularized_widelane_score"])
    ]
    for row in rows:
        row["regularized_widelane_rank"] = None
    for rank, index in enumerate(
        sorted(finite_regularized, key=lambda value: rows[value]["regularized_widelane_score"]),
        start=1,
    ):
        rows[index]["regularized_widelane_rank"] = rank
    for block in range(int(args.bootstrap_blocks)):
        finite_block = [
            index
            for index, row in enumerate(rows)
            if np.isfinite(row["bootstrap_scores"][block])
        ]
        if finite_block:
            winner = min(finite_block, key=lambda index: rows[index]["bootstrap_scores"][block])
            rows[winner]["bootstrap_wins"] += 1
    ranked_regularized = sorted(
        finite_regularized, key=lambda value: rows[value]["regularized_widelane_score"]
    )
    selected_candidate_id = None
    selection_reason = "no_finite_candidate"
    score_gap = None
    if len(ranked_regularized) >= 2:
        best, runner = ranked_regularized[:2]
        score_gap = float(
            rows[runner]["regularized_widelane_score"]
            - rows[best]["regularized_widelane_score"]
        )
        if evidence_epochs < int(args.min_evidence_epochs):
            selection_reason = "insufficient_evidence_epochs"
        elif fixed_pairs < int(args.min_fixed_pairs):
            selection_reason = "insufficient_fixed_pairs"
        elif rows[best]["bootstrap_wins"] < int(args.min_bootstrap_wins):
            selection_reason = "insufficient_bootstrap_wins"
        elif score_gap < float(args.min_score_gap):
            selection_reason = "insufficient_score_gap"
        else:
            selected_candidate_id = int(rows[best]["candidate_id"])
            selection_reason = "regularized_widelane_consensus"
    dynamic_result: dict[str, Any] | None = None
    dynamic_sweep: list[dict[str, Any]] = []
    if dynamic_rows:
        prior_coefficients = np.zeros((int(args.dynamic_degree) + 1, 3))
        if selected_candidate_id is not None:
            selected_row = next(
                row for row in rows if int(row["candidate_id"]) == selected_candidate_id
            )
            prior_coefficients[0] = np.asarray(
                selected_row["offset_ecef_m"], dtype=np.float64
            )
        ridge_values = [
            float(value)
            for value in str(args.dynamic_ridges).split(",")
            if value.strip()
        ]
        if float(args.dynamic_ridge) not in ridge_values:
            ridge_values.insert(0, float(args.dynamic_ridge))
        for ridge_value in ridge_values:
            coefficients, objective = fit_dynamic_offset(
                dynamic_rows,
                degree=int(args.dynamic_degree),
                ridge=ridge_value,
                huber_k=float(args.dynamic_huber_k),
                prior_coefficients=prior_coefficients,
            )
            dynamic_errors = []
            dynamic_offset_norms = []
            for epoch in range(int(args.start), int(args.end)):
                t = (
                    2.0
                    * (epoch - int(args.start))
                    / max(int(args.end) - int(args.start) - 1, 1)
                    - 1.0
                )
                offset = evaluate_dynamic_offset(coefficients, t)
                dynamic_offset_norms.append(float(np.linalg.norm(offset)))
                dynamic_errors.append(
                    float(
                        np.linalg.norm(
                            base_positions[epoch]
                            + offset
                            - np.asarray(data["ground_truth"][epoch], dtype=np.float64)
                        )
                    )
                )
            row = {
                "degree": int(args.dynamic_degree),
                "ridge": ridge_value,
                "huber_k": float(args.dynamic_huber_k),
                "coefficients_ecef": coefficients.tolist(),
                "prior_candidate_id": selected_candidate_id,
                "prior_coefficients_ecef": prior_coefficients.tolist(),
                "measurement_rows": len(dynamic_rows),
                "weighted_residual_mse": objective,
                "max_offset_norm_m": max(dynamic_offset_norms),
                "audit_sub50cm_epochs": sum(error < 0.5 for error in dynamic_errors),
                "audit_rms_m": float(np.sqrt(np.mean(np.square(dynamic_errors)))),
            }
            dynamic_sweep.append(row)
            if ridge_value == float(args.dynamic_ridge):
                dynamic_result = row
    return {
        "segment": [int(args.start), int(args.end)],
        "stride": int(args.stride),
        "stride_origin": int(args.stride_origin),
        "evidence_epochs": evidence_epochs,
        "candidate_pairs": candidate_pairs,
        "fixed_pairs": fixed_pairs,
        "offset_prior_weight": float(args.offset_prior_weight),
        "bootstrap_blocks": int(args.bootstrap_blocks),
        "accepted": bool(evidence_epochs and fixed_pairs),
        "reason": (
            "ranked" if evidence_epochs and fixed_pairs else "no_fixed_widelane_evidence"
        ),
        "selected_candidate_id": selected_candidate_id,
        "selection_reason": selection_reason,
        "regularized_score_gap": score_gap,
        "min_bootstrap_wins": int(args.min_bootstrap_wins),
        "min_score_gap": float(args.min_score_gap),
        "min_evidence_epochs": int(args.min_evidence_epochs),
        "min_fixed_pairs": int(args.min_fixed_pairs),
        "dynamic_offset": dynamic_result,
        "dynamic_offset_sweep": dynamic_sweep,
        "candidates": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_json", type=Path)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--warmup-trajectory", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--stride-origin", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--max-std-cycles", type=float, default=0.75)
    parser.add_argument("--ratio-threshold", type=float, default=3.0)
    parser.add_argument("--min-fix-rate", type=float, default=0.3)
    parser.add_argument("--residual-sigma-m", type=float, default=1.0)
    parser.add_argument("--offset-prior-weight", type=float, default=0.15)
    parser.add_argument("--bootstrap-blocks", type=int, default=4)
    parser.add_argument("--min-bootstrap-wins", type=int, default=2)
    parser.add_argument("--min-score-gap", type=float, default=0.01)
    parser.add_argument("--min-evidence-epochs", type=int, default=20)
    parser.add_argument("--min-fixed-pairs", type=int, default=100)
    parser.add_argument("--dynamic-degree", type=int, default=1)
    parser.add_argument("--dynamic-ridge", type=float, default=0.15)
    parser.add_argument(
        "--dynamic-ridges", default="0.15,0.3,0.5,1,2,3,5,10,20,30,50,100,200"
    )
    parser.add_argument("--dynamic-huber-k", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.bootstrap_blocks < 1:
        parser.error("--bootstrap-blocks must be at least 1")
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
