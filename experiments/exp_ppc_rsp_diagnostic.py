#!/usr/bin/env python3
# ruff: noqa: E402
"""Diagnose DD-gradient reservoir Stein corrections on PPC realtime fusion."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from evaluate import ecef_errors_2d_3d
from exp_ppc_realtime_fusion import _dd_measurements, run_fusion_eval
from gnss_gpu.dd_likelihood import dd_log_likelihood_gradients
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer
from gnss_gpu.io.ppc import PPCDatasetLoader
from gnss_gpu.ppc_score import score_ppc2024
from gnss_gpu.reservoir_stein import ReservoirSteinConfig, reservoir_stein_update

RESULTS_DIR = _SCRIPT_DIR / "results"


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _base_fusion_kwargs() -> dict[str, object]:
    return {
        "tdcp_min_sats": 5,
        "tdcp_max_postfit_rms_m": 0.5,
        "tdcp_max_cycle_jump": 20000.0,
        "tdcp_max_velocity_mps": 50.0,
        "carrier_phase_sign": 1.0,
        "receiver_motion_sign": -1.0,
        "dd_huber_k_m": 1.0,
        "dd_trim_m": 1.5,
        "dd_min_kept_pairs": 5,
        "dd_max_shift_m": 200.0,
        "dd_anchor_blend_alpha": 0.3,
        "dd_interpolate_base_epochs": True,
        "widelane": True,
        "widelane_min_epochs": 5,
        "widelane_max_std_cycles": 0.75,
        "widelane_ratio_threshold": 3.0,
        "widelane_min_fix_rate": 0.3,
        "widelane_min_kept_pairs": 3,
        "widelane_max_shift_m": 5.0,
        "widelane_max_robust_rms_m": 0.8,
        "widelane_veto_rms_band_min_m": 0.15,
        "widelane_veto_rms_band_max_m": 0.35,
        "widelane_veto_min_kept_pairs": 4,
        "widelane_anchor_blend_alpha": 1.0,
        "height_hold_alpha": 1.0,
        "height_hold_release_on_last_velocity": True,
        "height_hold_release_min_dd_shift_m": 0.4,
        "rsp_correction": True,
        "rsp_n_particles": 64,
        "rsp_spread_m": 1.0,
        "rsp_sigma_m": 1.0,
        "rsp_huber_k_m": 1.0,
        "rsp_stein_steps": 1,
        "rsp_stein_step_size": 0.1,
        "rsp_repulsion_scale": 0.25,
        "rsp_min_dd_shift_m": 0.85,
        "rsp_max_dd_shift_m": 1.4,
        "rsp_min_dd_rms_m": 0.45,
        "rsp_max_dd_rms_m": 0.7,
        "rsp_random_seed": 42,
        "last_velocity_max_age_s": 8.0,
    }


def _rng_particles(center: np.ndarray, n_particles: int, spread_m: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(center, dtype=np.float64).reshape(1, 3) + rng.normal(
        0.0,
        float(spread_m),
        size=(int(n_particles), 3),
    )


def _project_to_reference_radius(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Keep radial height close to the realtime fused reference."""
    cand = np.asarray(candidate, dtype=np.float64).reshape(3)
    ref = np.asarray(reference, dtype=np.float64).reshape(3)
    ref_norm = float(np.linalg.norm(ref))
    cand_norm = float(np.linalg.norm(cand))
    if cand_norm <= 0.0 or ref_norm <= 0.0:
        return cand.copy()
    return cand * (ref_norm / cand_norm)


def main() -> None:
    parser = argparse.ArgumentParser(description="PPC DD-gradient reservoir Stein diagnostic")
    parser.add_argument("--data-dir", type=Path, required=True, help="PPC run directory")
    parser.add_argument("--start-epoch", type=int, default=1300)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--systems", type=str, default="G,E,J")
    parser.add_argument("--diagnose-start", type=int, default=153)
    parser.add_argument("--diagnose-end", type=int, default=174)
    parser.add_argument("--n-particles", type=int, default=64)
    parser.add_argument("--spread-m", type=float, default=1.0)
    parser.add_argument("--sigma-m", type=float, default=1.0)
    parser.add_argument("--huber-k-m", type=float, default=1.0)
    parser.add_argument("--stein-step-size", type=float, default=0.1)
    parser.add_argument("--stein-steps", type=int, default=1)
    parser.add_argument("--repulsion-scale", type=float, default=0.25)
    parser.add_argument(
        "--project-reference-radius",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Project corrected epoch position back to the fused radial height",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-prefix", type=str, default="ppc_rsp_diagnostic")
    args = parser.parse_args()

    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        systems=systems,
        include_sat_velocity=True,
    )
    _summary, _epochs, arrays = run_fusion_eval(
        data,
        args.data_dir,
        systems,
        **_base_fusion_kwargs(),
    )
    fused = np.asarray(arrays["fused"], dtype=np.float64)
    corrected = fused.copy()
    truth = np.asarray(data["ground_truth"], dtype=np.float64)
    dd_computer = DDPseudorangeComputer(
        args.data_dir / "base.obs",
        rover_obs_path=args.data_dir / "rover.obs",
        allowed_systems=systems,
        interpolate_base_epochs=True,
    )

    rows: list[dict[str, object]] = []
    for epoch in range(int(args.diagnose_start), int(args.diagnose_end) + 1):
        rows_for_dd = _dd_measurements(data, epoch, fused[epoch])
        dd = dd_computer.compute_dd(float(data["times"][epoch]), rows_for_dd)
        if dd is None or int(dd.n_dd) < 3:
            continue

        particles = _rng_particles(
            fused[epoch],
            args.n_particles,
            args.spread_m,
            args.seed + epoch,
        )
        gradients = dd_log_likelihood_gradients(
            dd,
            particles,
            sigma_m=args.sigma_m,
            huber_k_m=args.huber_k_m,
        )
        log_weights = -0.5 * np.sum(np.square(particles - fused[epoch]), axis=1) / (
            float(args.spread_m) ** 2
        )
        result = reservoir_stein_update(
            particles,
            log_weights,
            gradients,
            ReservoirSteinConfig(
                reservoir_size=int(args.n_particles),
                elite_fraction=0.25,
                stein_steps=int(args.stein_steps),
                stein_step_size=float(args.stein_step_size),
                repulsion_scale=float(args.repulsion_scale),
                seed=int(args.seed + epoch),
            ),
        )
        estimate = np.average(result.particles, axis=0, weights=result.weights)
        if args.project_reference_radius:
            estimate = _project_to_reference_radius(estimate, fused[epoch])
        corrected[epoch] = estimate
        before_2d, before_3d = ecef_errors_2d_3d(fused[epoch : epoch + 1], truth[epoch])
        after_2d, after_3d = ecef_errors_2d_3d(estimate.reshape(1, 3), truth[epoch])
        rows.append(
            {
                "epoch": int(epoch),
                "n_dd": int(dd.n_dd),
                "ess_before": float(result.ess_before),
                "mean_gradient_norm_m": float(np.mean(np.linalg.norm(gradients, axis=1))),
                "mean_shift_m": float(np.linalg.norm(estimate - fused[epoch])),
                "before_2d_m": float(before_2d[0]),
                "after_2d_m": float(after_2d[0]),
                "before_3d_m": float(before_3d[0]),
                "after_3d_m": float(after_3d[0]),
                "improved_3d": bool(after_3d[0] < before_3d[0]),
            }
        )

    base_score = score_ppc2024(fused, truth)
    corrected_score = score_ppc2024(corrected, truth)
    summary_rows = [
        {
            "method": "base_realtime_fusion",
            "ppc_score_pct": float(base_score.score_pct),
            "ppc_epoch_pass_pct": float(base_score.epoch_pass_pct),
            "pass_epochs": int(np.count_nonzero(base_score.pass_mask)),
        },
        {
            "method": "rsp_corrected_epochs",
            "ppc_score_pct": float(corrected_score.score_pct),
            "ppc_epoch_pass_pct": float(corrected_score.epoch_pass_pct),
            "pass_epochs": int(np.count_nonzero(corrected_score.pass_mask)),
            "diagnosed_epochs": int(len(rows)),
        },
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, RESULTS_DIR / f"{args.results_prefix}_epochs.csv")
    _write_rows(summary_rows, RESULTS_DIR / f"{args.results_prefix}_summary.csv")
    print(f"base ppc={base_score.score_pct:.3f}% pass={np.count_nonzero(base_score.pass_mask)}")
    print(
        f"rsp  ppc={corrected_score.score_pct:.3f}% "
        f"pass={np.count_nonzero(corrected_score.pass_mask)}"
    )
    print(f"saved {RESULTS_DIR / f'{args.results_prefix}_epochs.csv'}")


if __name__ == "__main__":
    main()
