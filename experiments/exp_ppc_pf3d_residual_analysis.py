#!/usr/bin/env python3
"""Residual diagnostics for PPC PF3D / PF3D-BVH experiments.

This script reruns WLS, standard PF, and PF3D-BVH on one PPC segment and dumps
per-satellite pseudorange residuals together with ray-traced LOS/NLOS labels.
It is intended to explain cases where PF3D-BVH degrades despite non-zero NLOS.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

RESULTS_DIR = _SCRIPT_DIR / "results"

from evaluate import save_results
from exp_urbannav_baseline import load_or_generate_data, run_wls
from exp_urbannav_pf3d import (
    PF_NLOS_BIAS,
    PF_SIGMA_LOS,
    PF_SIGMA_NLOS,
    load_plateau_model,
    run_pf3d_variant,
    run_pf_standard,
)


def _compute_residuals(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    weights: np.ndarray,
    state_xyzcb: np.ndarray,
) -> np.ndarray:
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
    state = np.asarray(state_xyzcb, dtype=np.float64).ravel()
    ranges = np.linalg.norm(sat - state[:3], axis=1)
    pred_pr = ranges + float(state[3])
    return pr - pred_pr


def _reference_state_at_truth(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    weights: np.ndarray,
    ground_truth_xyz: np.ndarray,
) -> np.ndarray:
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    xyz = np.asarray(ground_truth_xyz, dtype=np.float64).ravel()

    ranges = np.linalg.norm(sat - xyz, axis=1)
    valid = np.isfinite(pr) & np.isfinite(w) & (w > 0.0)
    if np.any(valid):
        cb = float(np.sum(w[valid] * (pr[valid] - ranges[valid])) / np.sum(w[valid]))
    else:
        cb = float(np.mean(pr - ranges))
    return np.array([xyz[0], xyz[1], xyz[2], cb], dtype=np.float64)


def _compute_los_flags(accelerator, rx_xyz: np.ndarray, sat_ecef: np.ndarray) -> np.ndarray:
    if accelerator is None:
        return np.ones(len(sat_ecef), dtype=bool)
    return np.asarray(accelerator.check_los(rx_xyz, sat_ecef), dtype=bool)


def _add_summary_rows(
    summary_rows: list[dict[str, object]],
    residuals: np.ndarray,
    los_gt: np.ndarray,
    method: str,
) -> None:
    res = np.asarray(residuals, dtype=np.float64)
    masks = {
        "all": np.ones(len(res), dtype=bool),
        "los_gt": np.asarray(los_gt, dtype=bool),
        "nlos_gt": ~np.asarray(los_gt, dtype=bool),
    }
    for bucket, mask in masks.items():
        if not np.any(mask):
            continue
        vals = res[mask]
        summary_rows.append(
            {
                "method": method,
                "bucket": bucket,
                "count": int(len(vals)),
                "mean_residual": float(np.mean(vals)),
                "mean_abs_residual": float(np.mean(np.abs(vals))),
                "p05": float(np.percentile(vals, 5)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "positive_frac": float(np.mean(vals > 0.0)),
                "negative_frac": float(np.mean(vals < 0.0)),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual analysis for PPC PF3D-BVH segments")
    parser.add_argument("--data-dir", type=Path, required=True, help="PPC run directory")
    parser.add_argument("--model-dir", type=Path, required=True, help="PLATEAU CityGML subset directory")
    parser.add_argument("--plateau-zone", type=int, default=9, help="PLATEAU plane-rect zone")
    parser.add_argument("--start-epoch", type=int, default=0, help="Usable epoch offset")
    parser.add_argument("--max-epochs", type=int, default=100, help="Epoch count")
    parser.add_argument("--systems", type=str, default="G", help="Comma-separated constellations")
    parser.add_argument("--n-particles", type=int, default=10_000, help="Particle count")
    parser.add_argument("--sigma-pr", type=float, default=PF_SIGMA_LOS, help="PF pseudorange sigma")
    parser.add_argument("--sigma-los", type=float, default=PF_SIGMA_LOS, help="PF3D LOS sigma")
    parser.add_argument("--sigma-nlos", type=float, default=PF_SIGMA_NLOS, help="PF3D NLOS sigma")
    parser.add_argument("--nlos-bias", type=float, default=PF_NLOS_BIAS, help="PF3D NLOS bias")
    parser.add_argument("--blocked-nlos-prob", type=float, default=1.0, help="P(NLOS | ray blocked)")
    parser.add_argument("--clear-nlos-prob", type=float, default=0.0, help="P(NLOS | ray clear)")
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="ppc_pf3d_residual_analysis",
        help="Output filename prefix under experiments/results/",
    )
    args = parser.parse_args()

    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  PPC PF3D Residual Analysis")
    print("=" * 72)

    print("\n[1] Loading data ...")
    data = load_or_generate_data(
        args.data_dir,
        max_real_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        systems=systems,
    )
    print(f"    Dataset: {data.get('dataset_name', 'synthetic')}")

    print("\n[2] Loading building model ...")
    building_model = load_plateau_model(args.model_dir, zone=args.plateau_zone)
    accelerator = None
    if building_model is not None:
        try:
            from gnss_gpu.bvh import BVHAccelerator

            accelerator = BVHAccelerator.from_building_model(building_model)
            print(f"    BVH accelerator: {accelerator.n_nodes} nodes, {accelerator.n_triangles} triangles")
        except Exception as exc:
            print(f"    BVH accelerator unavailable ({type(exc).__name__}: {exc}), using linear model")
            accelerator = building_model

    print("\n[3] Running WLS / PF / PF3D-BVH ...")
    wls_states, _ = run_wls(data)
    pf_states, _, _ = run_pf_standard(
        data,
        args.n_particles,
        wls_states,
        sigma_pr=args.sigma_pr,
        return_states=True,
    )
    pf3d_bvh_states, _, _ = run_pf3d_variant(
        data,
        building_model,
        args.n_particles,
        wls_states,
        "pf3d_bvh",
        sigma_pr=args.sigma_pr,
        sigma_los=args.sigma_los,
        sigma_nlos=args.sigma_nlos,
        nlos_bias=args.nlos_bias,
        blocked_nlos_prob=args.blocked_nlos_prob,
        clear_nlos_prob=args.clear_nlos_prob,
        return_states=True,
    )

    print("\n[4] Building residual tables ...")
    state_rows: list[dict[str, object]] = []
    epoch_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for i in range(data["n_epochs"]):
        sat_ecef = np.asarray(data["sat_ecef"][i], dtype=np.float64)
        pseudoranges = np.asarray(data["pseudoranges"][i], dtype=np.float64)
        weights = np.asarray(data["weights"][i], dtype=np.float64)
        sat_ids = list(data.get("used_prns", [])[i])
        system_ids = np.asarray(data.get("system_ids", [])[i], dtype=np.int32)
        truth_state = _reference_state_at_truth(
            sat_ecef,
            pseudoranges,
            weights,
            np.asarray(data["ground_truth"][i], dtype=np.float64),
        )

        los_gt = _compute_los_flags(accelerator, truth_state[:3], sat_ecef)
        los_wls = _compute_los_flags(accelerator, wls_states[i, :3], sat_ecef)
        los_pf = _compute_los_flags(accelerator, pf_states[i, :3], sat_ecef)
        los_pf3d_bvh = _compute_los_flags(accelerator, pf3d_bvh_states[i, :3], sat_ecef)

        residual_gt = _compute_residuals(sat_ecef, pseudoranges, weights, truth_state)
        residual_wls = _compute_residuals(sat_ecef, pseudoranges, weights, wls_states[i])
        residual_pf = _compute_residuals(sat_ecef, pseudoranges, weights, pf_states[i])
        residual_pf3d_bvh = _compute_residuals(sat_ecef, pseudoranges, weights, pf3d_bvh_states[i])

        state_rows.append(
            {
                "epoch": i,
                "gps_tow": float(data["times"][i]),
                "satellite_count": int(len(sat_ids)),
                "truth_x": float(truth_state[0]),
                "truth_y": float(truth_state[1]),
                "truth_z": float(truth_state[2]),
                "truth_cb": float(truth_state[3]),
                "wls_x": float(wls_states[i, 0]),
                "wls_y": float(wls_states[i, 1]),
                "wls_z": float(wls_states[i, 2]),
                "wls_cb": float(wls_states[i, 3]),
                "pf_x": float(pf_states[i, 0]),
                "pf_y": float(pf_states[i, 1]),
                "pf_z": float(pf_states[i, 2]),
                "pf_cb": float(pf_states[i, 3]),
                "pf3d_bvh_x": float(pf3d_bvh_states[i, 0]),
                "pf3d_bvh_y": float(pf3d_bvh_states[i, 1]),
                "pf3d_bvh_z": float(pf3d_bvh_states[i, 2]),
                "pf3d_bvh_cb": float(pf3d_bvh_states[i, 3]),
            }
        )

        for s, sat_id in enumerate(sat_ids):
            epoch_rows.append(
                {
                    "epoch": i,
                    "gps_tow": float(data["times"][i]),
                    "sat_id": sat_id,
                    "system_id": int(system_ids[s]) if s < len(system_ids) else -1,
                    "weight": float(weights[s]),
                    "los_gt": bool(los_gt[s]),
                    "los_wls": bool(los_wls[s]),
                    "los_pf": bool(los_pf[s]),
                    "los_pf3d_bvh": bool(los_pf3d_bvh[s]),
                    "residual_truth_ref": float(residual_gt[s]),
                    "residual_wls": float(residual_wls[s]),
                    "residual_pf": float(residual_pf[s]),
                    "residual_pf3d_bvh": float(residual_pf3d_bvh[s]),
                }
            )

    los_gt_all = np.array([bool(row["los_gt"]) for row in epoch_rows], dtype=bool)
    _add_summary_rows(
        summary_rows,
        np.array([row["residual_truth_ref"] for row in epoch_rows], dtype=np.float64),
        los_gt_all,
        "truth_ref",
    )
    _add_summary_rows(
        summary_rows,
        np.array([row["residual_wls"] for row in epoch_rows], dtype=np.float64),
        los_gt_all,
        "wls",
    )
    _add_summary_rows(
        summary_rows,
        np.array([row["residual_pf"] for row in epoch_rows], dtype=np.float64),
        los_gt_all,
        "pf",
    )
    _add_summary_rows(
        summary_rows,
        np.array([row["residual_pf3d_bvh"] for row in epoch_rows], dtype=np.float64),
        los_gt_all,
        "pf3d_bvh",
    )

    state_path = RESULTS_DIR / f"{args.results_prefix}_states.csv"
    epoch_path = RESULTS_DIR / f"{args.results_prefix}_epochs.csv"
    summary_path = RESULTS_DIR / f"{args.results_prefix}_summary.csv"

    save_results({key: [row[key] for row in state_rows] for key in state_rows[0]}, state_path)
    save_results({key: [row[key] for row in epoch_rows] for key in epoch_rows[0]}, epoch_path)
    save_results({key: [row[key] for row in summary_rows] for key in summary_rows[0]}, summary_path)

    print(f"    States: {state_path}")
    print(f"    Epoch residuals: {epoch_path}")
    print(f"    Summary: {summary_path}")


if __name__ == "__main__":
    main()
