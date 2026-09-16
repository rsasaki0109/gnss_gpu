#!/usr/bin/env python3
"""Diagnose the per-particle 3D-mapping-aided likelihood against plain PF.

This is the first prototype for the "GPU PF x 3D city physics" track. It asks a
single mechanistic question before investing in a full PPC/UrbanNav integration:

    Does moving the LOS/NLOS geometry into each particle's likelihood
    (``ParticleFilter3DBVH``) change the weight distribution and the position
    error relative to a plain pseudorange PF, and how sensitive is it to the
    assumed NLOS bias?

The scene, satellites, trajectory, and NLOS pseudorange model are reused from
``examples/demo_plateau_nlos_simulation.py`` so numbers stay comparable with the
existing PLATEAU NLOS demo suite. Every filter sees the *same* pre-generated
observations, so the only difference is the likelihood.

Reported per configuration:

* P50 / RMS horizontal error,
* mean ESS and mean particle-cloud spread,
* mean fraction of particle-satellite rays the BVH marks blocked,
* NLOS detection recall at the estimated position,
* wall-clock ms/epoch.

Run from the repo root:

    PYTHONPATH=python:. python3 experiments/diag_pf_3dma_likelihood.py

Outputs are written under ``experiments/results/`` (gitignored).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GML = PROJECT_ROOT / "data" / "sample_plateau.gml"
DEFAULT_SUMMARY_JSON = (
    PROJECT_ROOT / "experiments" / "results" / "diag_pf_3dma_likelihood_summary.json"
)


def _load_demo_module():
    module_path = PROJECT_ROOT / "examples" / "demo_plateau_nlos_simulation.py"
    spec = importlib.util.spec_from_file_location("demo_plateau_nlos_simulation", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _p50(values: np.ndarray) -> float:
    return float(np.median(values))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def _build_scene(demo, gml_path: Path, n_epochs: int):
    triangles = demo.load_plateau_triangles(gml_path)
    origin_ecef, enu_to_ecef, verts_enu = demo.build_local_frame(triangles)
    ecef_to_enu = enu_to_ecef.T
    ground_z_m = float(verts_enu[:, 2].min() + 1.8)

    sats_azel = demo.default_satellite_az_el_deg()
    elevations_deg = np.array([el for _az, el in sats_azel], dtype=np.float64)
    sat_ecef = demo.build_satellites(origin_ecef, enu_to_ecef, sats_azel)

    rx_enu = np.zeros((n_epochs, 3), dtype=np.float64)
    rx_enu[:, 0] = np.linspace(-55.0, 55.0, n_epochs)
    rx_enu[:, 1] = -10.0
    rx_enu[:, 2] = ground_z_m
    rx_ecef = origin_ecef + rx_enu @ enu_to_ecef.T
    return {
        "triangles": triangles,
        "origin_ecef": origin_ecef,
        "enu_to_ecef": enu_to_ecef,
        "ecef_to_enu": ecef_to_enu,
        "sat_ecef": sat_ecef,
        "elevations_deg": elevations_deg,
        "rx_ecef": rx_ecef,
        "sats_azel": sats_azel,
    }


def _generate_observations(demo, scene, bvh, clock_bias_m: float, seed: int):
    n_epochs = scene["rx_ecef"].shape[0]
    sat_rep = np.repeat(scene["sat_ecef"][None, :, :], n_epochs, axis=0)
    los_mask = np.asarray(bvh.check_los_batch(scene["rx_ecef"], sat_rep), dtype=bool)

    rng = np.random.default_rng(seed)
    observations = []
    for k in range(n_epochs):
        observations.append(
            demo.simulate_pseudorange_epoch(
                rng,
                scene["rx_ecef"][k],
                scene["sat_ecef"],
                scene["elevations_deg"],
                los_mask[k],
                clock_bias_m,
            )
        )
    return los_mask, observations


def _make_configs(bias_values):
    configs = [
        {
            "name": "naive",
            "kind": "plain",
            "sigma_pr": 7.5,
            "nlos_weight": 1.0,
            "bias_correct": False,
        },
        {
            "name": "oracle-corrected",
            "kind": "plain",
            "sigma_pr": 7.5,
            "nlos_weight": 0.02,
            "bias_correct": True,
        },
        {
            "name": "3dma soft p=0.5",
            "kind": "bvh",
            "sigma_los": 2.0,
            "sigma_nlos": 30.0,
            "nlos_bias": 25.0,
            "blocked_nlos_prob": 0.5,
            "clear_nlos_prob": 0.0,
        },
        {
            "name": "3dma tight nlos",
            "kind": "bvh",
            "sigma_los": 2.0,
            "sigma_nlos": 10.0,
            "nlos_bias": 25.0,
            "blocked_nlos_prob": 0.5,
            "clear_nlos_prob": 0.0,
        },
        {
            "name": "3dma soft bias=0",
            "kind": "bvh",
            "sigma_los": 2.0,
            "sigma_nlos": 30.0,
            "nlos_bias": 0.0,
            "blocked_nlos_prob": 0.5,
            "clear_nlos_prob": 0.0,
        },
    ]
    for bias in bias_values:
        configs.append(
            {
                "name": f"3dma hard bias={bias:g}",
                "kind": "bvh",
                "sigma_los": 2.0,
                "sigma_nlos": 30.0,
                "nlos_bias": float(bias),
                "blocked_nlos_prob": 1.0,
                "clear_nlos_prob": 0.0,
            }
        )
    return configs


def _build_filter(config, scene, bvh, n_particles: int, seed: int):
    from gnss_gpu.particle_filter import ParticleFilter
    from gnss_gpu.particle_filter_3d_bvh import ParticleFilter3DBVH

    common = dict(
        n_particles=n_particles,
        sigma_pos=0.5,
        sigma_cb=2.0,
        resampling="systematic",
        ess_threshold=0.5,
        seed=seed,
    )
    if config["kind"] == "bvh":
        pf = ParticleFilter3DBVH(
            bvh=bvh,
            sigma_los=config["sigma_los"],
            sigma_nlos=config["sigma_nlos"],
            nlos_bias=config["nlos_bias"],
            blocked_nlos_prob=config.get("blocked_nlos_prob", 1.0),
            clear_nlos_prob=config.get("clear_nlos_prob", 0.0),
            **common,
        )
    else:
        pf = ParticleFilter(sigma_pr=config["sigma_pr"], **common)

    initial_center = scene["rx_ecef"][0] + scene["enu_to_ecef"] @ np.array([8.0, -6.0, 4.0])
    pf.initialize(
        initial_center,
        clock_bias=1432.0 + 12.0,
        spread_pos=22.0,
        spread_cb=35.0,
    )
    return pf


def _evaluate(
    config,
    scene,
    bvh,
    los_mask,
    observations,
    n_particles: int,
    seed: int,
    collect_details: bool = False,
):
    pf = _build_filter(config, scene, bvh, n_particles, seed)

    n_epochs = scene["rx_ecef"].shape[0]
    ecef_to_enu = scene["ecef_to_enu"]
    sats_broadcast = None

    errors: list[float] = []
    enu_offsets: list[float] = []
    ess_values: list[float] = []
    spreads: list[float] = []
    blocked_fracs: list[float] = []
    detect_acc: list[float] = []
    nlos_recall: list[float] = []
    per_epoch_enu: list[list[float]] = []

    previous_epoch: int | None = None
    start = time.perf_counter()
    for k in range(n_epochs):
        if previous_epoch is not None:
            delta = scene["rx_ecef"][k] - scene["rx_ecef"][previous_epoch]
            pf.predict(velocity=delta, dt=1.0)
        previous_epoch = k

        obs = observations[k]
        truth_los = los_mask[k]

        if config["kind"] == "bvh":
            particles = pf.get_particles()[:, :3]
            if sats_broadcast is None or sats_broadcast.shape[0] != particles.shape[0]:
                sats_broadcast = np.repeat(
                    scene["sat_ecef"][None, :, :], particles.shape[0], axis=0
                )
            particle_los = np.asarray(
                bvh.check_los_batch(particles, sats_broadcast), dtype=bool
            )
            blocked_fracs.append(float(np.mean(~particle_los)))
            spreads.append(float(np.mean(np.std(particles, axis=0))))

        pseudoranges = np.asarray(obs["pseudorange_m"], dtype=np.float64).copy()
        if config.get("bias_correct", False):
            pseudoranges -= np.asarray(obs["expected_nlos_bias_m"], dtype=np.float64)

        weights = None
        nlos_weight = config.get("nlos_weight", 1.0)
        if nlos_weight != 1.0:
            weights = np.where(truth_los, 1.0, float(nlos_weight))

        pf.update(scene["sat_ecef"], pseudoranges, weights)
        ess_values.append(float(pf.get_ess()))

        estimate = pf.estimate()
        delta_enu = ecef_to_enu @ (estimate[:3] - scene["rx_ecef"][k])
        errors.append(float(np.hypot(delta_enu[0], delta_enu[1])))
        enu_offsets.append(float(delta_enu[2]))
        per_epoch_enu.append([float(delta_enu[0]), float(delta_enu[1]), float(delta_enu[2])])

        if config["kind"] == "bvh":
            est_los = np.asarray(bvh.check_los(estimate[:3], scene["sat_ecef"]), dtype=bool)
            detect_acc.append(float(np.mean(est_los == truth_los)))
            if np.any(~truth_los):
                nlos_recall.append(float(np.mean(~est_los[~truth_los])))

    elapsed = time.perf_counter() - start
    error_arr = np.asarray(errors, dtype=np.float64)
    summary = {
        "name": config["name"],
        "kind": config["kind"],
        "p50_m": _p50(error_arr),
        "rms_m": _rms(error_arr),
        "mean_m": float(np.mean(error_arr)),
        "max_m": float(np.max(error_arr)),
        "ess_mean": float(np.mean(ess_values)),
        "ess_ratio_mean": float(np.mean(ess_values) / n_particles),
        "up_offset_mean_m": float(np.mean(np.abs(enu_offsets))),
        "up_offset_final_m": float(enu_offsets[-1]),
        "ms_per_epoch": float(1000.0 * elapsed / n_epochs),
        "n_particles": int(n_particles),
    }
    if blocked_fracs:
        summary["blocked_frac_mean"] = float(np.mean(blocked_fracs))
        summary["particle_spread_m_mean"] = float(np.mean(spreads))
    if detect_acc:
        summary["nlos_detect_acc"] = float(np.mean(detect_acc))
    if nlos_recall:
        summary["nlos_recall"] = float(np.mean(nlos_recall))
    summary["config"] = {
        key: config[key]
        for key in config
        if key not in {"name", "kind"}
    }
    if collect_details:
        summary["per_epoch_error_m"] = error_arr.tolist()
        summary["per_epoch_enu_m"] = per_epoch_enu
    return summary


def run_diag(
    gml_path: Path = DEFAULT_GML,
    *,
    summary_json: Path | None = DEFAULT_SUMMARY_JSON,
    n_particles: int = 20000,
    n_epochs: int = 70,
    bias_values=(0.0, 10.0, 20.0, 30.0, 40.0),
    seed: int = 20260607,
    collect_details: bool = False,
) -> dict[str, object]:
    from gnss_gpu.bvh import BVHAccelerator

    demo = _load_demo_module()
    scene = _build_scene(demo, gml_path, n_epochs)
    bvh = BVHAccelerator(scene["triangles"])
    clock_bias_m = 1432.0
    los_mask, observations = _generate_observations(demo, scene, bvh, clock_bias_m, seed)

    configs = _make_configs(bias_values)
    results = [
        _evaluate(
            config,
            scene,
            bvh,
            los_mask,
            observations,
            n_particles,
            seed,
            collect_details=collect_details,
        )
        for config in configs
    ]

    nlos_fraction = float(np.mean(~los_mask))
    summary: dict[str, object] = {
        "gml_path": str(gml_path),
        "n_triangles": int(scene["triangles"].shape[0]),
        "n_satellites": int(scene["sat_ecef"].shape[0]),
        "n_epochs": int(n_epochs),
        "n_particles": int(n_particles),
        "nlos_fraction": nlos_fraction,
        "nlos_bias_mean_m": float(
            np.mean(
                np.vstack(
                    [np.asarray(o["true_nlos_bias_m"]) for o in observations]
                )[~los_mask]
            )
        ),
        "results": results,
    }

    if summary_json is not None:
        summary_json = Path(summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary["summary_json"] = str(summary_json)
        summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return summary


def _print_summary(summary: dict[str, object]) -> None:
    print("Per-particle 3DMA likelihood diagnostic")
    print("=" * 96)
    print(
        f"mesh={summary['n_triangles']} tris  sats={summary['n_satellites']}  "
        f"epochs={summary['n_epochs']}  particles={summary['n_particles']}  "
        f"NLOS={100.0 * summary['nlos_fraction']:.1f}%  "
        f"mean bias={summary['nlos_bias_mean_m']:.1f} m"
    )
    print("-" * 96)
    header = (
        f"{'config':<20}{'P50':>8}{'RMS':>8}{'ESS%':>7}"
        f"{'blocked':>8}{'detect':>8}{'nlrec':>7}{'upOff':>8}{'ms/ep':>8}"
    )
    print(header)
    print("-" * 96)
    for row in summary["results"]:
        print(
            f"{row['name']:<20}"
            f"{row['p50_m']:>7.2f} {row['rms_m']:>7.2f}"
            f"{100.0 * row['ess_ratio_mean']:>6.1f}%"
            f"{100.0 * row.get('blocked_frac_mean', float('nan')):>7.1f}%"
            f"{100.0 * row.get('nlos_detect_acc', float('nan')):>7.1f}%"
            f"{100.0 * row.get('nlos_recall', float('nan')):>6.1f}%"
            f"{row['up_offset_final_m']:>7.2f}"
            f"{row['ms_per_epoch']:>7.2f}"
        )
    print("-" * 96)
    if summary.get("summary_json"):
        print(f"summary: {summary['summary_json']}")


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gml", type=Path, default=DEFAULT_GML)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--particles", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument(
        "--bias-values",
        type=float,
        nargs="+",
        default=[0.0, 10.0, 20.0, 30.0, 40.0],
    )
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    if args.particles <= 0:
        parser.error("--particles must be positive")
    if args.epochs < 2:
        parser.error("--epochs must be at least 2")

    summary = run_diag(
        args.gml,
        summary_json=args.summary_json,
        n_particles=args.particles,
        n_epochs=args.epochs,
        bias_values=tuple(args.bias_values),
        seed=args.seed,
        collect_details=args.details,
    )
    _print_summary(summary)
    return summary


if __name__ == "__main__":
    main()
