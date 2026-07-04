#!/usr/bin/env python3
"""Evaluate GMM pseudorange likelihood on the deterministic PLATEAU PF replay."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_GML = PROJECT_ROOT / "data" / "sample_plateau.gml"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _log_normal(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1.0e-6)
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma * math.sqrt(2.0 * math.pi))


def _log_gmm_residual(residual: np.ndarray, *, w_los: float, mu_nlos: float, sigma_los: float, sigma_nlos: float) -> np.ndarray:
    comp_los = w_los * np.exp(_log_normal(residual, 0.0, sigma_los))
    comp_nlos = (1.0 - w_los) * np.exp(_log_normal(residual, mu_nlos, sigma_nlos))
    return np.log(np.maximum(comp_los + comp_nlos, 1.0e-300))


class _GmmReplayParticleFilter:
    """Deterministic PF with optional GMM pseudorange likelihood."""

    def __init__(
        self,
        *,
        n_particles: int,
        sigma_pr_m: float,
        process_sigma_m: float,
        clock_sigma_m: float,
        seed: int,
        use_gmm: bool,
        w_los: float = 0.7,
        mu_nlos: float = 15.0,
        sigma_nlos: float = 30.0,
    ) -> None:
        self.n_particles = int(n_particles)
        self.sigma_pr_m = float(sigma_pr_m)
        self.process_sigma_m = float(process_sigma_m)
        self.clock_sigma_m = float(clock_sigma_m)
        self.use_gmm = bool(use_gmm)
        self.w_los = float(w_los)
        self.mu_nlos = float(mu_nlos)
        self.sigma_nlos = float(sigma_nlos)
        self.rng = np.random.default_rng(seed)
        self.particles = np.empty((0, 4), dtype=np.float64)
        self.log_weights = np.empty(0, dtype=np.float64)

    def initialize(self, center_ecef: np.ndarray, *, clock_bias_m: float, spread_pos_m: float, spread_clock_m: float) -> None:
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
        self.particles[:, :3] += self.rng.normal(0.0, self.process_sigma_m, size=(self.n_particles, 3))
        self.particles[:, 3] += self.rng.normal(0.0, self.clock_sigma_m, self.n_particles)

    def update(self, sat_ecef: np.ndarray, pseudorange_m: np.ndarray, *, sat_weights: np.ndarray | None = None) -> None:
        pr = np.asarray(pseudorange_m, dtype=np.float64)
        if sat_weights is None:
            sat_weights = np.ones(pr.shape[0], dtype=np.float64)
        else:
            sat_weights = np.asarray(sat_weights, dtype=np.float64)

        log_likelihood = np.zeros(self.n_particles, dtype=np.float64)
        for sat, obs_pr, obs_weight in zip(sat_ecef, pr, sat_weights):
            ranges = np.linalg.norm(sat - self.particles[:, :3], axis=1)
            residual = obs_pr - (ranges + self.particles[:, 3])
            if self.use_gmm:
                log_likelihood += float(obs_weight) * _log_gmm_residual(
                    residual,
                    w_los=self.w_los,
                    mu_nlos=self.mu_nlos,
                    sigma_los=self.sigma_pr_m,
                    sigma_nlos=self.sigma_nlos,
                )
            else:
                log_likelihood += -0.5 * float(obs_weight) * (residual / self.sigma_pr_m) ** 2

        vmax = float(np.max(log_likelihood))
        self.log_weights += log_likelihood
        self.log_weights -= float(vmax + np.log(np.sum(np.exp(self.log_weights - vmax))))

    def estimate(self) -> np.ndarray:
        vmax = float(np.max(self.log_weights))
        weights = np.exp(self.log_weights - vmax)
        weights /= np.sum(weights)
        return np.average(self.particles, axis=0, weights=weights)


def evaluate_gmm_configs(
    mask_csv: Path,
    *,
    gml_path: Path = SAMPLE_GML,
    n_particles: int = 1500,
    configs: tuple[tuple[float, float, float], ...] = ((0.7, 15.0, 30.0), (0.8, 10.0, 20.0)),
) -> dict[str, object]:
    replay = _load_module("replay_plateau_nlos_demo_pf", PROJECT_ROOT / "experiments" / "replay_plateau_nlos_demo_pf.py")
    spp_replay = replay._load_spp_replay_module()
    demo = spp_replay._load_demo_module()

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
    complete_epoch = np.asarray(mask["complete_epoch"], dtype=bool)

    rows: list[dict[str, object]] = []
    for w_los, mu_nlos, sigma_nlos in configs:
        pf = _GmmReplayParticleFilter(
            n_particles=n_particles,
            sigma_pr_m=replay.DEFAULT_SIGMA_PR_M,
            process_sigma_m=replay.DEFAULT_PROCESS_SIGMA_M,
            clock_sigma_m=replay.DEFAULT_CLOCK_SIGMA_M,
            seed=20260607,
            use_gmm=True,
            w_los=w_los,
            mu_nlos=mu_nlos,
            sigma_nlos=sigma_nlos,
        )
        pf.initialize(
            rx_ecef[0] + enu_to_ecef @ np.array([8.0, -6.0, 4.0]),
            clock_bias_m=1432.0 + 12.0,
            spread_pos_m=22.0,
            spread_clock_m=35.0,
        )
        errors: list[float] = []
        previous_epoch_idx: int | None = None
        for epoch_idx in range(n_epochs):
            if not complete_epoch[epoch_idx]:
                continue
            if previous_epoch_idx is not None:
                pf.predict(rx_ecef[epoch_idx] - rx_ecef[previous_epoch_idx])
            previous_epoch_idx = epoch_idx
            is_los = los_mask[epoch_idx]
            obs = demo.simulate_pseudorange_epoch(
                rng,
                rx_ecef[epoch_idx],
                sat_ecef,
                elevations_deg,
                is_los,
                1432.0,
            )
            pf.update(sat_ecef, obs["pseudorange_m"])
            errors.append(
                demo.horizontal_error_m(pf.estimate()[:3], rx_ecef[epoch_idx], ecef_to_enu)
            )
        arr = np.asarray(errors, dtype=np.float64)
        rows.append(
            {
                "w_los": w_los,
                "mu_nlos": mu_nlos,
                "sigma_nlos": sigma_nlos,
                "rms_m": float(np.sqrt(np.mean(arr * arr))),
                "p50_m": float(np.median(arr)),
            }
        )

    baseline = replay.replay_pf(mask_csv, summary_json=None, n_particles=n_particles)
    return {
        "configs": rows,
        "best": min(rows, key=lambda row: float(row["rms_m"])),
        "mask_soft_baseline_rms_m": float(baseline["mask_soft_pf"]["rms_m"]),
        "naive_baseline_rms_m": float(baseline["naive_pf"]["rms_m"]),
    }


def main() -> None:
    exporter = _load_module(
        "export_plateau_nlos_demo_mask",
        PROJECT_ROOT / "experiments" / "export_plateau_nlos_demo_mask.py",
    )
    out_dir = PROJECT_ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_csv = out_dir / "gmm_nlos_eval_mask.csv"
    exporter.export_mask_csv(mask_csv)

    summary = evaluate_gmm_configs(mask_csv)
    summary_path = out_dir / "gmm_nlos_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["best"], indent=2))


if __name__ == "__main__":
    main()
