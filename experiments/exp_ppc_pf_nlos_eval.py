#!/usr/bin/env python3
"""Evaluate PF-domain geometry NLOS mask soft weights on the PLATEAU demo replay.

This mirrors the PPC PF update path (``gnss_gpu.nlos_mask`` -> per-sat weights
before ``pf.update``) without requiring the PPC dataset or CUDA kernels.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "python"))

from gnss_gpu.nlos_mask import apply_mask_to_weights, load_nlos_mask_tables  # noqa: E402

SAMPLE_GML = PROJECT_ROOT / "data" / "sample_plateau.gml"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prn_list(n_satellites: int, prefix: str = "G") -> list[str]:
    return [f"{prefix}{idx + 1:02d}" for idx in range(n_satellites)]


def evaluate_pf_nlos_mask_path(
    mask_csv: Path,
    *,
    gml_path: Path = SAMPLE_GML,
    n_particles: int = 1500,
    k_weak: float = 3.0,
    k_strong: float = 3.0,
    bias_scale: float = 1.0,
) -> dict[str, object]:
    """Compare naive PF vs production mask-soft weights on the demo replay."""
    replay_pf = _load_module(
        "replay_plateau_nlos_demo_pf",
        PROJECT_ROOT / "experiments" / "replay_plateau_nlos_demo_pf.py",
    )
    spp_replay = replay_pf._load_spp_replay_module()
    demo = spp_replay._load_demo_module()

    rng = np.random.default_rng(20260606)
    triangles = demo.load_plateau_triangles(gml_path)
    origin_ecef, enu_to_ecef, verts_enu = demo.build_local_frame(triangles)
    ecef_to_enu = enu_to_ecef.T
    ground_z_m = float(verts_enu[:, 2].min() + 1.8)

    sats_azel = demo.default_satellite_az_el_deg()
    elevations_deg = np.array([el for _az, el in sats_azel], dtype=np.float64)
    sat_ecef = demo.build_satellites(origin_ecef, enu_to_ecef, sats_azel)
    prns = _prn_list(len(sats_azel))

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

    tables = load_nlos_mask_tables(mask_csv)
    clock_bias_m = 1432.0
    initial_offset_enu = np.array([8.0, -6.0, 4.0], dtype=np.float64)
    initial_center = rx_ecef[0] + enu_to_ecef @ initial_offset_enu

    naive_pf = replay_pf._ReplayParticleFilter(
        n_particles=n_particles,
        sigma_pr_m=replay_pf.DEFAULT_SIGMA_PR_M,
        process_sigma_m=replay_pf.DEFAULT_PROCESS_SIGMA_M,
        clock_sigma_m=replay_pf.DEFAULT_CLOCK_SIGMA_M,
        seed=20260607,
    )
    mask_pf = replay_pf._ReplayParticleFilter(
        n_particles=n_particles,
        sigma_pr_m=replay_pf.DEFAULT_SIGMA_PR_M,
        process_sigma_m=replay_pf.DEFAULT_PROCESS_SIGMA_M,
        clock_sigma_m=replay_pf.DEFAULT_CLOCK_SIGMA_M,
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
    mask_errors: list[float] = []
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
        base_weights = np.ones(len(prns), dtype=np.float64)
        mask_weights = np.asarray(
            apply_mask_to_weights(
                epoch_idx,
                prns,
                base_weights,
                tables,
                k_weak=k_weak,
                k_strong=k_strong,
            ),
            dtype=np.float64,
        )

        naive_pf.update(sat_ecef, pr)
        mask_pf.update(sat_ecef, pr - correction, sat_weights=mask_weights)

        true_rx = rx_ecef[epoch_idx]
        naive_errors.append(
            demo.horizontal_error_m(naive_pf.estimate()[:3], true_rx, ecef_to_enu)
        )
        mask_errors.append(
            demo.horizontal_error_m(mask_pf.estimate()[:3], true_rx, ecef_to_enu)
        )

    naive_arr = np.asarray(naive_errors, dtype=np.float64)
    mask_arr = np.asarray(mask_errors, dtype=np.float64)
    if naive_arr.size == 0:
        raise RuntimeError("PF NLOS eval produced no solved epochs")

    return {
        "mask_csv": str(mask_csv),
        "k_weak": float(k_weak),
        "k_strong": float(k_strong),
        "n_epochs": float(n_epochs),
        "n_solved_epochs": float(naive_arr.size),
        "naive_rms_m": float(np.sqrt(np.mean(naive_arr * naive_arr))),
        "mask_soft_rms_m": float(np.sqrt(np.mean(mask_arr * mask_arr))),
        "wins_mask_over_naive": float(np.sum(mask_arr < naive_arr)),
    }


def main() -> None:
    exporter = _load_module(
        "export_plateau_nlos_demo_mask",
        PROJECT_ROOT / "experiments" / "export_plateau_nlos_demo_mask.py",
    )
    out_dir = PROJECT_ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_csv = out_dir / "ppc_pf_nlos_eval_mask.csv"
    exporter.export_mask_csv(mask_csv)

    summary = evaluate_pf_nlos_mask_path(mask_csv)
    summary_path = out_dir / "ppc_pf_nlos_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
