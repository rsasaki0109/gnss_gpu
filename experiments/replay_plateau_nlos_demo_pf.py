#!/usr/bin/env python3
"""Replay the PLATEAU NLOS demo mask through a particle-filter consumer path.

This script consumes the exported mask CSV contract:

    tow,epoch_idx,prn,is_los

The CSV is the only LOS/NLOS input used by the mask-aware filter. Synthetic
pseudoranges are regenerated with the deterministic PLATEAU NLOS measurement
model used by the demo, so the downstream PF behavior can be tested without
real GNSS logs or CUDA kernels.

Run from the repo root:

    PYTHONPATH=python:. python3 experiments/replay_plateau_nlos_demo_pf.py \
      --mask-csv experiments/results/plateau_nlos_demo_mask.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASK_CSV = PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_demo_mask.csv"
DEFAULT_SUMMARY_JSON = (
    PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_demo_pf_replay_summary.json"
)

DEFAULT_NLOS_WEIGHT = 0.10
DEFAULT_N_PARTICLES = 12000
DEFAULT_SIGMA_PR_M = 7.5
DEFAULT_PROCESS_SIGMA_M = 0.35
DEFAULT_CLOCK_SIGMA_M = 2.0


def _load_spp_replay_module():
    module_path = PROJECT_ROOT / "experiments" / "replay_plateau_nlos_demo_spp.py"
    spec = importlib.util.spec_from_file_location("replay_plateau_nlos_demo_spp", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _p50(values: np.ndarray) -> float:
    return float(np.median(values))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def _summarize_errors(values: np.ndarray) -> dict[str, float]:
    return {
        "p50_m": _p50(values),
        "rms_m": _rms(values),
        "mean_m": float(np.mean(values)),
        "max_m": float(np.max(values)),
    }


class _ReplayParticleFilter:
    """Small deterministic PF used to verify the NLOS mask replay contract."""

    def __init__(
        self,
        *,
        n_particles: int,
        sigma_pr_m: float,
        process_sigma_m: float,
        clock_sigma_m: float,
        seed: int,
    ) -> None:
        self.n_particles = int(n_particles)
        self.sigma_pr_m = float(sigma_pr_m)
        self.process_sigma_m = float(process_sigma_m)
        self.clock_sigma_m = float(clock_sigma_m)
        self.rng = np.random.default_rng(seed)
        self.particles = np.empty((0, 4), dtype=np.float64)
        self.log_weights = np.empty(0, dtype=np.float64)

    def initialize(
        self,
        center_ecef: np.ndarray,
        *,
        clock_bias_m: float,
        spread_pos_m: float,
        spread_clock_m: float,
    ) -> None:
        center_ecef = np.asarray(center_ecef, dtype=np.float64)
        self.particles = np.column_stack(
            [
                self.rng.normal(center_ecef[0], spread_pos_m, self.n_particles),
                self.rng.normal(center_ecef[1], spread_pos_m, self.n_particles),
                self.rng.normal(center_ecef[2], spread_pos_m, self.n_particles),
                self.rng.normal(clock_bias_m, spread_clock_m, self.n_particles),
            ]
        )
        self.log_weights = np.full(self.n_particles, -np.log(self.n_particles), dtype=np.float64)

    def predict(self, delta_ecef: np.ndarray) -> None:
        self.particles[:, :3] += np.asarray(delta_ecef, dtype=np.float64)
        self.particles[:, :3] += self.rng.normal(
            0.0,
            self.process_sigma_m,
            size=(self.n_particles, 3),
        )
        self.particles[:, 3] += self.rng.normal(0.0, self.clock_sigma_m, self.n_particles)

    def update(
        self,
        sat_ecef: np.ndarray,
        pseudorange_m: np.ndarray,
        *,
        sat_weights: np.ndarray | None = None,
    ) -> float:
        pr = np.asarray(pseudorange_m, dtype=np.float64)
        if sat_weights is None:
            sat_weights = np.ones(pr.shape[0], dtype=np.float64)
        else:
            sat_weights = np.asarray(sat_weights, dtype=np.float64)

        log_likelihood = np.zeros(self.n_particles, dtype=np.float64)
        for sat, obs_pr, obs_weight in zip(sat_ecef, pr, sat_weights):
            ranges = np.linalg.norm(sat - self.particles[:, :3], axis=1)
            residual = obs_pr - (ranges + self.particles[:, 3])
            log_likelihood += -0.5 * float(obs_weight) * (residual / self.sigma_pr_m) ** 2

        self.log_weights += log_likelihood
        self.log_weights -= _logsumexp(self.log_weights)
        weights = np.exp(self.log_weights)
        ess = float(1.0 / np.sum(weights * weights))
        if ess < 0.55 * self.n_particles:
            self._resample(weights)
        return ess

    def estimate(self) -> np.ndarray:
        weights = np.exp(self.log_weights - _logsumexp(self.log_weights))
        return np.average(self.particles, axis=0, weights=weights)

    def _resample(self, weights: np.ndarray) -> None:
        positions = (self.rng.random() + np.arange(self.n_particles)) / self.n_particles
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, positions, side="left")
        self.particles = self.particles[indices].copy()
        self.log_weights.fill(-np.log(self.n_particles))


def _logsumexp(values: np.ndarray) -> float:
    vmax = float(np.max(values))
    return float(vmax + np.log(np.sum(np.exp(values - vmax))))


def replay_pf(
    mask_csv: Path = DEFAULT_MASK_CSV,
    *,
    summary_json: Path | None = DEFAULT_SUMMARY_JSON,
    gml_path: Path | None = None,
    n_particles: int = DEFAULT_N_PARTICLES,
    nlos_weight: float = DEFAULT_NLOS_WEIGHT,
    bias_scale: float = 1.0,
    sigma_pr_m: float = DEFAULT_SIGMA_PR_M,
    process_sigma_m: float = DEFAULT_PROCESS_SIGMA_M,
    clock_sigma_m: float = DEFAULT_CLOCK_SIGMA_M,
) -> dict[str, object]:
    spp_replay = _load_spp_replay_module()
    demo = spp_replay._load_demo_module()
    if gml_path is None:
        gml_path = PROJECT_ROOT / "data" / "sample_plateau.gml"

    rng = np.random.default_rng(20260606)
    triangles = demo.load_plateau_triangles(gml_path)
    origin_ecef, enu_to_ecef, verts_enu = demo.build_local_frame(triangles)
    ecef_to_enu = enu_to_ecef.T
    ground_z_m = float(verts_enu[:, 2].min() + 1.8)

    sats_azel = demo.default_satellite_az_el_deg()
    elevations_deg = np.array([el for _az, el in sats_azel], dtype=np.float64)
    sat_ecef = demo.build_satellites(origin_ecef, enu_to_ecef, sats_azel)

    n_epochs = 70
    rx_enu = np.zeros((n_epochs, 3), dtype=np.float64)
    rx_enu[:, 0] = np.linspace(-55.0, 55.0, n_epochs)
    rx_enu[:, 1] = -10.0
    rx_enu[:, 2] = ground_z_m
    rx_ecef = origin_ecef + rx_enu @ enu_to_ecef.T

    mask = spp_replay.load_mask_csv(mask_csv, n_epochs=n_epochs, n_satellites=len(sats_azel))
    los_mask = np.asarray(mask["los_mask"], dtype=bool)
    expected_bias = np.asarray(mask["expected_bias_m"], dtype=np.float64)
    missing_bias = (~los_mask) & (expected_bias <= 0.0)
    if np.any(missing_bias):
        computed = np.vstack(
            [demo.nlos_expected_bias_m(elevations_deg, los_mask[i]) for i in range(n_epochs)]
        )
        expected_bias[missing_bias] = computed[missing_bias]
    complete_epoch = np.asarray(mask["complete_epoch"], dtype=bool)

    clock_bias_m = 1432.0
    initial_offset_enu = np.array([8.0, -6.0, 4.0], dtype=np.float64)
    initial_center = rx_ecef[0] + enu_to_ecef @ initial_offset_enu

    naive_pf = _ReplayParticleFilter(
        n_particles=n_particles,
        sigma_pr_m=sigma_pr_m,
        process_sigma_m=process_sigma_m,
        clock_sigma_m=clock_sigma_m,
        seed=20260607,
    )
    mask_pf = _ReplayParticleFilter(
        n_particles=n_particles,
        sigma_pr_m=sigma_pr_m,
        process_sigma_m=process_sigma_m,
        clock_sigma_m=clock_sigma_m,
        seed=20260607,
    )
    for pf in (naive_pf, mask_pf):
        pf.initialize(
            initial_center,
            clock_bias_m=clock_bias_m + 12.0,
            spread_pos_m=22.0,
            spread_clock_m=35.0,
        )

    naive_errors: list[float] = []
    mask_soft_errors: list[float] = []
    used_epochs: list[int] = []
    naive_ess: list[float] = []
    mask_soft_ess: list[float] = []

    previous_epoch_idx: int | None = None
    for epoch_idx in range(n_epochs):
        if not complete_epoch[epoch_idx]:
            continue

        if previous_epoch_idx is not None:
            delta_ecef = rx_ecef[epoch_idx] - rx_ecef[previous_epoch_idx]
            naive_pf.predict(delta_ecef)
            mask_pf.predict(delta_ecef)
        previous_epoch_idx = epoch_idx

        is_los = los_mask[epoch_idx]
        obs = demo.simulate_pseudorange_epoch(
            rng,
            rx_ecef[epoch_idx],
            sat_ecef,
            elevations_deg,
            is_los,
            clock_bias_m,
        )
        pr = obs["pseudorange_m"]
        correction = bias_scale * expected_bias[epoch_idx]
        weights = np.where(is_los, 1.0, float(nlos_weight))

        naive_ess.append(naive_pf.update(sat_ecef, pr))
        mask_soft_ess.append(mask_pf.update(sat_ecef, pr - correction, sat_weights=weights))

        true_rx = rx_ecef[epoch_idx]
        naive_errors.append(demo.horizontal_error_m(naive_pf.estimate()[:3], true_rx, ecef_to_enu))
        mask_soft_errors.append(demo.horizontal_error_m(mask_pf.estimate()[:3], true_rx, ecef_to_enu))
        used_epochs.append(epoch_idx)

    naive_arr = np.asarray(naive_errors, dtype=np.float64)
    mask_arr = np.asarray(mask_soft_errors, dtype=np.float64)
    if naive_arr.size == 0:
        raise RuntimeError("PF replay produced no solved epochs")

    summary = {
        "mask_csv": str(mask_csv),
        "gml_path": str(gml_path),
        "n_epochs": int(n_epochs),
        "n_complete_mask_epochs": int(np.count_nonzero(complete_epoch)),
        "n_solved_epochs": int(naive_arr.size),
        "n_particles": int(n_particles),
        "nlos_weight": float(nlos_weight),
        "bias_scale": float(bias_scale),
        "sigma_pr_m": float(sigma_pr_m),
        "process_sigma_m": float(process_sigma_m),
        "clock_sigma_m": float(clock_sigma_m),
        "nlos_frac": float(np.mean(~los_mask[complete_epoch])),
        "naive_pf": _summarize_errors(naive_arr),
        "mask_soft_pf": _summarize_errors(mask_arr),
        "mask_soft_wins": int(np.sum(mask_arr < naive_arr)),
        "rms_gain_vs_naive_pct": float(100.0 * (1.0 - _rms(mask_arr) / _rms(naive_arr))),
        "used_epochs": used_epochs,
        "naive_ess_mean": float(np.mean(naive_ess)),
        "mask_soft_ess_mean": float(np.mean(mask_soft_ess)),
    }

    if summary_json is not None:
        summary_json = Path(summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary["summary_json"] = str(summary_json)
        summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return summary


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-csv", type=Path, default=DEFAULT_MASK_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--gml", type=Path, default=None)
    parser.add_argument("--particles", type=int, default=DEFAULT_N_PARTICLES)
    parser.add_argument("--nlos-weight", type=float, default=DEFAULT_NLOS_WEIGHT)
    parser.add_argument("--bias-scale", type=float, default=1.0)
    parser.add_argument("--sigma-pr-m", type=float, default=DEFAULT_SIGMA_PR_M)
    parser.add_argument("--process-sigma-m", type=float, default=DEFAULT_PROCESS_SIGMA_M)
    parser.add_argument("--clock-sigma-m", type=float, default=DEFAULT_CLOCK_SIGMA_M)
    args = parser.parse_args()
    if args.particles <= 0:
        parser.error("--particles must be positive")
    if not (0.0 < args.nlos_weight <= 1.0):
        parser.error("--nlos-weight must be in (0, 1]")
    if args.bias_scale < 0.0:
        parser.error("--bias-scale must be non-negative")
    if args.sigma_pr_m <= 0.0:
        parser.error("--sigma-pr-m must be positive")
    if args.process_sigma_m < 0.0:
        parser.error("--process-sigma-m must be non-negative")
    if args.clock_sigma_m < 0.0:
        parser.error("--clock-sigma-m must be non-negative")

    summary = replay_pf(
        args.mask_csv,
        summary_json=args.summary_json,
        gml_path=args.gml,
        n_particles=args.particles,
        nlos_weight=args.nlos_weight,
        bias_scale=args.bias_scale,
        sigma_pr_m=args.sigma_pr_m,
        process_sigma_m=args.process_sigma_m,
        clock_sigma_m=args.clock_sigma_m,
    )
    print("PLATEAU NLOS PF replay")
    print("=" * 70)
    print(
        f"mask={summary['mask_csv']} solved={summary['n_solved_epochs']}/"
        f"{summary['n_complete_mask_epochs']} nlos_frac={summary['nlos_frac']:.4f}"
    )
    print(
        f"particles={summary['n_particles']} sigma_pr={summary['sigma_pr_m']:.1f} m "
        f"nlos_weight={summary['nlos_weight']:.2f}"
    )
    print(f"{'method':<24}{'P50 err':>12}{'RMS err':>12}")
    print("-" * 48)
    for name, label in [
        ("naive_pf", "naive PF"),
        ("mask_soft_pf", "mask-soft PF"),
    ]:
        metrics = summary[name]
        print(f"{label:<24}{metrics['p50_m']:>10.2f} m{metrics['rms_m']:>10.2f} m")
    print("-" * 48)
    print(
        f"mask-soft wins {summary['mask_soft_wins']}/{summary['n_solved_epochs']} epochs; "
        f"RMS gain {summary['rms_gain_vs_naive_pct']:.0f}%"
    )
    if summary.get("summary_json"):
        print(f"summary: {summary['summary_json']}")
    return summary


if __name__ == "__main__":
    main()
