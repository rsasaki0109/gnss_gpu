#!/usr/bin/env python3
"""Grid sweep of residual / PR-acceleration down-weighting on the PLATEAU PF replay."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "python"))

from gnss_gpu.pseudorange_weighting import apply_pseudorange_weighting  # noqa: E402

SAMPLE_GML = PROJECT_ROOT / "data" / "sample_plateau.gml"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _horizontal_error(estimate_ecef: np.ndarray, true_ecef: np.ndarray, ecef_to_enu: np.ndarray) -> float:
    delta = (estimate_ecef - true_ecef) @ ecef_to_enu.T
    return float(np.hypot(delta[0], delta[1]))


def replay_with_soft_weighting(
    mask_csv: Path,
    *,
    gml_path: Path = SAMPLE_GML,
    n_particles: int = 1500,
    residual_downweight: bool = False,
    residual_threshold: float = 10.0,
    pr_accel_downweight: bool = False,
    pr_accel_threshold: float = 5.0,
    nlos_weight: float = 0.10,
    bias_scale: float = 1.0,
) -> dict[str, float]:
    """Run mask-soft PF replay with optional 2B/3B weighting on top of geometry mask."""
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
    expected_bias = np.asarray(mask["expected_bias_m"], dtype=np.float64)
    missing_bias = (~los_mask) & (expected_bias <= 0.0)
    if np.any(missing_bias):
        computed = np.vstack(
            [demo.nlos_expected_bias_m(elevations_deg, los_mask[i]) for i in range(n_epochs)]
        )
        expected_bias[missing_bias] = computed[missing_bias]
    complete_epoch = np.asarray(mask["complete_epoch"], dtype=bool)

    pf = replay._ReplayParticleFilter(
        n_particles=n_particles,
        sigma_pr_m=replay.DEFAULT_SIGMA_PR_M,
        process_sigma_m=replay.DEFAULT_PROCESS_SIGMA_M,
        clock_sigma_m=replay.DEFAULT_CLOCK_SIGMA_M,
        seed=20260607,
    )
    pf.initialize(
        rx_ecef[0] + enu_to_ecef @ np.array([8.0, -6.0, 4.0]),
        clock_bias_m=1432.0 + 12.0,
        spread_pos_m=22.0,
        spread_clock_m=35.0,
    )

    pr_history: dict[int, list[float]] = {}
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
        pr = obs["pseudorange_m"] - bias_scale * expected_bias[epoch_idx]
        base_weights = np.where(is_los, 1.0, float(nlos_weight))
        measurements = [SimpleNamespace(prn=i + 1) for i in range(len(base_weights))]
        weights = apply_pseudorange_weighting(
            measurements,
            sat_ecef,
            pr,
            base_weights,
            pf.estimate()[:3],
            pr_history,
            residual_downweight=residual_downweight,
            residual_threshold=residual_threshold,
            pr_accel_downweight=pr_accel_downweight,
            pr_accel_threshold=pr_accel_threshold,
        )
        pf.update(sat_ecef, pr, sat_weights=weights)
        errors.append(_horizontal_error(pf.estimate()[:3], rx_ecef[epoch_idx], ecef_to_enu))

    arr = np.asarray(errors, dtype=np.float64)
    return {
        "rms_m": float(np.sqrt(np.mean(arr * arr))),
        "p50_m": float(np.median(arr)),
        "n_epochs": float(arr.size),
    }


def sweep_soft_weights(
    mask_csv: Path,
    *,
    residual_thresholds: tuple[float, ...] = (5.0, 10.0, 20.0),
    pr_accel_thresholds: tuple[float, ...] = (3.0, 5.0, 10.0),
    n_particles: int = 1500,
) -> dict[str, object]:
    """Evaluate a small grid and return the best RMS configuration."""
    rows: list[dict[str, object]] = []
    for residual_threshold in residual_thresholds:
        for pr_accel_threshold in pr_accel_thresholds:
            for residual_on, accel_on in itertools.product((False, True), repeat=2):
                metrics = replay_with_soft_weighting(
                    mask_csv,
                    n_particles=n_particles,
                    residual_downweight=residual_on,
                    residual_threshold=residual_threshold,
                    pr_accel_downweight=accel_on,
                    pr_accel_threshold=pr_accel_threshold,
                )
                rows.append(
                    {
                        "residual_downweight": residual_on,
                        "residual_threshold": residual_threshold,
                        "pr_accel_downweight": accel_on,
                        "pr_accel_threshold": pr_accel_threshold,
                        **metrics,
                    }
                )

    best = min(rows, key=lambda row: float(row["rms_m"]))
    return {"rows": rows, "best": best}


def main() -> None:
    exporter = _load_module(
        "export_plateau_nlos_demo_mask",
        PROJECT_ROOT / "experiments" / "export_plateau_nlos_demo_mask.py",
    )
    out_dir = PROJECT_ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_csv = out_dir / "nlos_soft_weight_sweep_mask.csv"
    exporter.export_mask_csv(mask_csv)

    summary = sweep_soft_weights(mask_csv)
    summary_path = out_dir / "nlos_soft_weight_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["best"], indent=2))


if __name__ == "__main__":
    main()
