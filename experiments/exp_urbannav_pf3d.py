#!/usr/bin/env python3
"""Experiment: 3D-aware Particle Filter on PPC / UrbanNav / synthetic data.

Evaluates standard PF, PF3D, and PF3D-BVH (when available) against WLS/EKF
baselines. Supports PPC-Dataset real data via the shared baseline loader. When
no PLATEAU model is available, the script can create a small synthetic building
model around the trajectory origin as a plumbing fallback.

Outputs
-------
  experiments/results/<prefix>_results.csv
  experiments/results/<prefix>_summary.csv
  experiments/results/<prefix>_cdf.png
  experiments/results/<prefix>_timeline.png
  experiments/results/<prefix>_nlos_stats.csv
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

RESULTS_DIR = _SCRIPT_DIR / "results"

from evaluate import (
    SimplePFCPU,
    compute_metrics,
    ecef_to_lla,
    plot_cdf,
    plot_error_timeline,
    print_comparison_table,
    save_results,
)
from exp_urbannav_baseline import (
    _solve_single_system_epoch,
    load_or_generate_data,
    run_ekf,
    run_wls,
)
from gnss_gpu.multi_gnss import MultiGNSSSolver
from gnss_gpu.multi_gnss_quality import (
    MultiGNSSQualityVetoConfig,
    select_multi_gnss_solution,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PARTICLES_PF3D = 100_000
PF_SIGMA_POS = 2.0
PF_SIGMA_CB = 300.0
PF_SIGMA_LOS = 3.0
PF_SIGMA_NLOS = 30.0
PF_NLOS_BIAS = 20.0
PF_RESCUE_DISTANCE_M = 80.0
PF_RESCUE_SPREAD_POS = 30.0
PF_RESCUE_SPREAD_CB = 200.0


def _epoch_dt(data: dict, i: int) -> float:
    if i <= 0:
        return float(data.get("dt", 1.0))

    times = np.asarray(data.get("times", []), dtype=np.float64)
    if len(times) > i:
        dt = float(times[i] - times[i - 1])
        if dt > 0.0:
            return dt
    return float(data.get("dt", 1.0))


def _longest_segment(mask: np.ndarray, times: np.ndarray) -> tuple[int, float]:
    longest_epochs = 0
    longest_duration = 0.0
    start = None

    for i, flagged in enumerate(mask):
        if flagged and start is None:
            start = i
        elif not flagged and start is not None:
            end = i - 1
            n_epochs = end - start + 1
            duration = float(times[end] - times[start]) if end > start else 0.0
            if n_epochs > longest_epochs:
                longest_epochs = n_epochs
                longest_duration = duration
            start = None

    if start is not None:
        end = len(mask) - 1
        n_epochs = end - start + 1
        duration = float(times[end] - times[start]) if end > start else 0.0
        if n_epochs > longest_epochs:
            longest_epochs = n_epochs
            longest_duration = duration

    return longest_epochs, longest_duration


def _augment_tail_metrics(metrics: dict, times: np.ndarray) -> dict:
    errors = np.asarray(metrics["errors_2d"], dtype=np.float64)
    outlier_mask = errors > 100.0
    catastrophic_mask = errors > 500.0
    longest_epochs, longest_duration = _longest_segment(outlier_mask, times)
    metrics["outlier_rate_pct"] = 100.0 * float(np.mean(outlier_mask))
    metrics["catastrophic_rate_pct"] = 100.0 * float(np.mean(catastrophic_mask))
    metrics["longest_outlier_segment_epochs"] = float(longest_epochs)
    metrics["longest_outlier_segment_s"] = float(longest_duration)
    return metrics


def _select_pf_epoch_measurements(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    weights: np.ndarray,
    system_ids: np.ndarray | None,
    multi_solver: MultiGNSSSolver | None,
    quality_veto_config: MultiGNSSQualityVetoConfig | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Select per-epoch measurements for PF update.

    When a quality-veto config is provided, bad multi-GNSS epochs fall back to
    the reference constellation only. Otherwise the full observation set is
    used unchanged.
    """

    sat_ecef = np.asarray(sat_ecef, dtype=np.float64)
    pseudoranges = np.asarray(pseudoranges, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        quality_veto_config is None
        or multi_solver is None
        or system_ids is None
    ):
        return sat_ecef, pseudoranges, weights, True

    system_ids = np.asarray(system_ids, dtype=np.int32)
    ref_mask = system_ids == int(quality_veto_config.reference_system)
    if int(np.count_nonzero(ref_mask)) < 4:
        return sat_ecef, pseudoranges, weights, True

    multi_position, multi_biases, _ = multi_solver.solve(
        sat_ecef,
        pseudoranges,
        system_ids,
        weights,
    )
    reference_solution = _solve_single_system_epoch(
        sat_ecef[ref_mask],
        pseudoranges[ref_mask],
        weights[ref_mask],
    )
    decision = select_multi_gnss_solution(
        reference_solution=reference_solution,
        multi_position=multi_position,
        multi_biases=multi_biases,
        sat_ecef=sat_ecef,
        pseudoranges=pseudoranges,
        system_ids=system_ids,
        config=quality_veto_config,
    )
    if decision.use_multi:
        return sat_ecef, pseudoranges, weights, True
    return sat_ecef[ref_mask], pseudoranges[ref_mask], weights[ref_mask], False


def _should_rescue_pf_epoch(
    estimate_state: np.ndarray,
    reference_position: np.ndarray | None,
    rescue_distance_m: float | None,
) -> bool:
    """Return whether the PF should be re-centered on the reference track."""

    if reference_position is None or rescue_distance_m is None:
        return False

    estimate_state = np.asarray(estimate_state, dtype=np.float64).ravel()
    reference_position = np.asarray(reference_position, dtype=np.float64).ravel()
    if estimate_state.size < 3 or reference_position.size < 3:
        return False
    return float(np.linalg.norm(estimate_state[:3] - reference_position[:3])) > float(
        rescue_distance_m
    )


def _reference_predict_velocity(
    guide_reference_positions: np.ndarray | None,
    epoch_index: int,
    dt: float,
) -> np.ndarray | None:
    """Estimate a predict-step velocity from a reference trajectory."""

    if guide_reference_positions is None or epoch_index <= 0:
        return None

    reference_positions = np.asarray(guide_reference_positions, dtype=np.float64)
    if len(reference_positions) <= epoch_index:
        return None

    denom = max(float(dt), 1e-3)
    return (reference_positions[epoch_index, :3] - reference_positions[epoch_index - 1, :3]) / denom


def _select_guide_velocity(
    guide_reference_positions: np.ndarray | None,
    epoch_index: int,
    dt: float,
    guide_mode: str,
    use_multi: bool,
    selected_satellite_count: int,
    guide_satellite_max: int | None = None,
) -> np.ndarray | None:
    """Pick a guide velocity according to an experiment-only guide policy."""

    velocity = _reference_predict_velocity(guide_reference_positions, epoch_index, dt)
    if velocity is None:
        return None
    if guide_mode == "always":
        return velocity
    if guide_mode in {"none", "init_only"}:
        return None
    if guide_mode == "fallback_only":
        return velocity if not use_multi else None
    if guide_mode == "fallback_or_low_sat":
        if not use_multi:
            return velocity
        if (
            guide_satellite_max is not None
            and int(selected_satellite_count) <= int(guide_satellite_max)
        ):
            return velocity
        return None
    raise ValueError(f"unknown guide_mode: {guide_mode}")


def _guide_backend_suffix(
    guide_reference_positions: np.ndarray | None,
    guide_mode: str,
    guide_satellite_max: int | None,
) -> str:
    if guide_reference_positions is None:
        return ""
    if guide_mode == "always":
        return "+Guide"
    if guide_mode == "init_only":
        return "+GuideInit"
    if guide_mode == "fallback_only":
        return "+GuideFallback"
    if guide_mode == "fallback_or_low_sat":
        return f"+GuideFallbackLowSat({guide_satellite_max})"
    return f"+Guide[{guide_mode}]"


def load_plateau_model(model_dir: Path, zone: int) -> object | None:
    """Try to load one PLATEAU CityGML model from *model_dir*."""
    gml_files = list(model_dir.glob("**/*.gml")) + list(model_dir.glob("**/*.xml"))
    if not gml_files:
        return None

    print(f"    Found {len(gml_files)} CityGML files in {model_dir}")
    try:
        from gnss_gpu.io.plateau import load_plateau

        model = load_plateau(model_dir, zone=zone)
        print(f"    PLATEAU model loaded: {model.triangles.shape[0]} triangles")
        return model
    except Exception as e:
        print(f"    Could not load PLATEAU model: {e}")
        return None


def create_synthetic_building_model(origin_ecef: np.ndarray) -> object | None:
    """Create a small synthetic urban model around the origin."""
    try:
        from gnss_gpu.raytrace import BuildingModel

        lat, lon, _ = ecef_to_lla(origin_ecef[0], origin_ecef[1], origin_ecef[2])
        sin_lat = math.sin(lat)
        cos_lat = math.cos(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)
        rotation = np.array(
            [
                [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
                [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
                [0.0, cos_lat, sin_lat],
            ]
        )

        def box_triangles_ecef(cx_enu: float, cy_enu: float, cz_enu: float,
                               width: float, depth: float, height: float) -> np.ndarray:
            hw, hd, hh = width / 2, depth / 2, height / 2
            corners_enu = np.array(
                [
                    [cx_enu - hw, cy_enu - hd, cz_enu - hh],
                    [cx_enu + hw, cy_enu - hd, cz_enu - hh],
                    [cx_enu + hw, cy_enu + hd, cz_enu - hh],
                    [cx_enu - hw, cy_enu + hd, cz_enu - hh],
                    [cx_enu - hw, cy_enu - hd, cz_enu + hh],
                    [cx_enu + hw, cy_enu - hd, cz_enu + hh],
                    [cx_enu + hw, cy_enu + hd, cz_enu + hh],
                    [cx_enu - hw, cy_enu + hd, cz_enu + hh],
                ],
                dtype=np.float64,
            )
            corners_ecef = np.array([origin_ecef + rotation @ corner for corner in corners_enu])
            faces = [
                (0, 1, 2), (0, 2, 3),
                (4, 5, 6), (4, 6, 7),
                (0, 1, 5), (0, 5, 4),
                (2, 3, 7), (2, 7, 6),
                (1, 2, 6), (1, 6, 5),
                (0, 3, 7), (0, 7, 4),
            ]
            return np.array(
                [[corners_ecef[a], corners_ecef[b], corners_ecef[c]] for a, b, c in faces],
                dtype=np.float32,
            )

        buildings_enu = [
            (30.0, 20.0, 25.0, 40.0, 100.0),
            (-40.0, -10.0, 30.0, 30.0, 60.0),
            (10.0, -50.0, 20.0, 20.0, 80.0),
            (-20.0, 60.0, 35.0, 25.0, 120.0),
        ]
        triangles = [
            box_triangles_ecef(east, north, height / 2, width, depth, height)
            for east, north, width, depth, height in buildings_enu
        ]
        model = BuildingModel(np.concatenate(triangles, axis=0))
        print(f"    Synthetic building model: {model.triangles.shape[0]} triangles")
        return model
    except Exception as e:
        print(f"    BuildingModel unavailable: {e}")
        return None


def run_pf_standard(
    data: dict,
    n_particles: int,
    wls_init: np.ndarray,
    sigma_pr: float = PF_SIGMA_LOS,
    quality_veto_config: MultiGNSSQualityVetoConfig | None = None,
    guide_reference_positions: np.ndarray | None = None,
    guide_initial_from_reference: bool = False,
    guide_mode: str = "always",
    guide_satellite_max: int | None = None,
    rescue_reference_positions: np.ndarray | None = None,
    rescue_distance_m: float | None = None,
    rescue_spread_pos: float = PF_RESCUE_SPREAD_POS,
    rescue_spread_cb: float = PF_RESCUE_SPREAD_CB,
    return_states: bool = False,
) -> tuple[np.ndarray, float, str]:
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    system_ids = data.get("system_ids")
    positions = np.zeros((n_epochs, 3))
    states = np.zeros((n_epochs, 4)) if return_states else None
    t0 = time.perf_counter()
    multi_solver = None
    if (
        quality_veto_config is not None
        and system_ids is not None
        and len(data.get("constellations", ())) > 1
    ):
        systems = sorted({int(sid) for epoch_ids in system_ids for sid in epoch_ids})
        multi_solver = MultiGNSSSolver(systems=systems)
    kept_multi_epochs = 0
    fallback_epochs = 0
    rescue_epochs = 0
    init_reference = (
        np.asarray(guide_reference_positions[0], dtype=np.float64)
        if guide_initial_from_reference and guide_reference_positions is not None
        else None
    )
    init_position = (
        np.asarray(init_reference[:3], dtype=np.float64)
        if init_reference is not None
        else np.asarray(wls_init[0, :3], dtype=np.float64)
    )
    init_clock_bias = (
        float(init_reference[3])
        if init_reference is not None and init_reference.size > 3
        else float(wls_init[0, 3])
    )

    try:
        from gnss_gpu import ParticleFilter

        pf = ParticleFilter(
            n_particles=n_particles,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=sigma_pr,
            resampling="megopolis",
            seed=42,
        )
        pf.initialize(
            init_position,
            clock_bias=init_clock_bias,
            spread_pos=50.0,
            spread_cb=500.0,
        )
        for i in range(n_epochs):
            dt = _epoch_dt(data, i)
            sat_i, pr_i, w_i, use_multi = _select_pf_epoch_measurements(
                sat_ecef[i],
                pseudoranges[i],
                weights[i],
                None if system_ids is None else system_ids[i],
                multi_solver,
                quality_veto_config,
            )
            pf.predict(
                velocity=_select_guide_velocity(
                    guide_reference_positions,
                    i,
                    dt,
                    guide_mode,
                    use_multi,
                    sat_i.shape[0],
                    guide_satellite_max=guide_satellite_max,
                ),
                dt=dt,
            )
            if _should_rescue_pf_epoch(
                pf.estimate(),
                None if rescue_reference_positions is None else rescue_reference_positions[i],
                rescue_distance_m,
            ):
                reference_position = np.asarray(rescue_reference_positions[i], dtype=np.float64)
                estimate_state = np.asarray(pf.estimate(), dtype=np.float64)
                rescue_clock_bias = (
                    float(reference_position[3])
                    if reference_position.size > 3
                    else float(estimate_state[3])
                )
                pf.initialize(
                    reference_position[:3],
                    clock_bias=rescue_clock_bias,
                    spread_pos=rescue_spread_pos,
                    spread_cb=rescue_spread_cb,
                )
                rescue_epochs += 1
            kept_multi_epochs += int(use_multi)
            fallback_epochs += int(not use_multi)
            pf.update(sat_i, pr_i, weights=w_i)
            estimate = np.asarray(pf.estimate(), dtype=np.float64)
            positions[i] = estimate[:3]
            if states is not None:
                states[i] = estimate
        backend = "GPU"
    except (ImportError, RuntimeError, Exception) as e:
        n_cpu = min(n_particles, 50_000)
        print(f"      Standard PF GPU failed ({type(e).__name__}: {e}), using CPU fallback")
        pf = SimplePFCPU(
            n_particles=n_cpu,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=sigma_pr,
            seed=42,
        )
        pf.initialize(
            init_position,
            clock_bias=init_clock_bias,
            spread_pos=50.0,
            spread_cb=500.0,
        )
        for i in range(n_epochs):
            dt = _epoch_dt(data, i)
            sat_i, pr_i, w_i, use_multi = _select_pf_epoch_measurements(
                sat_ecef[i],
                pseudoranges[i],
                weights[i],
                None if system_ids is None else system_ids[i],
                multi_solver,
                quality_veto_config,
            )
            pf.predict(
                velocity=_select_guide_velocity(
                    guide_reference_positions,
                    i,
                    dt,
                    guide_mode,
                    use_multi,
                    sat_i.shape[0],
                    guide_satellite_max=guide_satellite_max,
                ),
                dt=dt,
            )
            if _should_rescue_pf_epoch(
                pf.estimate(),
                None if rescue_reference_positions is None else rescue_reference_positions[i],
                rescue_distance_m,
            ):
                reference_position = np.asarray(rescue_reference_positions[i], dtype=np.float64)
                estimate_state = np.asarray(pf.estimate(), dtype=np.float64)
                rescue_clock_bias = (
                    float(reference_position[3])
                    if reference_position.size > 3
                    else float(estimate_state[3])
                )
                pf.initialize(
                    reference_position[:3],
                    clock_bias=rescue_clock_bias,
                    spread_pos=rescue_spread_pos,
                    spread_cb=rescue_spread_cb,
                )
                rescue_epochs += 1
            kept_multi_epochs += int(use_multi)
            fallback_epochs += int(not use_multi)
            pf.update(sat_i, pr_i, weights=w_i)
            estimate = np.asarray(pf.estimate(), dtype=np.float64)
            positions[i] = estimate[:3]
            if states is not None:
                states[i] = estimate
        backend = f"CPU({n_cpu})"

    ms = (time.perf_counter() - t0) * 1000.0 / max(n_epochs, 1)
    if quality_veto_config is not None and multi_solver is not None:
        backend += f"+QVGated({fallback_epochs}fb)"
    backend += _guide_backend_suffix(
        guide_reference_positions,
        guide_mode,
        guide_satellite_max,
    )
    if rescue_reference_positions is not None and rescue_distance_m is not None:
        backend += f"+Rescue({rescue_epochs})"
    return (states if states is not None else positions), ms, backend


def _build_pf3d_variant(
    building_model: object,
    n_particles: int,
    variant: str,
    sigma_pr: float,
    sigma_los: float,
    sigma_nlos: float,
    nlos_bias: float,
    blocked_nlos_prob: float,
    clear_nlos_prob: float,
):
    if variant == "pf3d":
        from gnss_gpu.particle_filter_3d import ParticleFilter3D

        return ParticleFilter3D(
            building_model=building_model,
            sigma_los=sigma_los,
            sigma_nlos=sigma_nlos,
            nlos_bias=nlos_bias,
            blocked_nlos_prob=blocked_nlos_prob,
            clear_nlos_prob=clear_nlos_prob,
            n_particles=n_particles,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=sigma_pr,
            resampling="megopolis",
            seed=42,
        )

    if variant == "pf3d_bvh":
        from gnss_gpu.bvh import BVHAccelerator
        from gnss_gpu.particle_filter_3d_bvh import ParticleFilter3DBVH

        bvh = BVHAccelerator.from_building_model(building_model)
        return ParticleFilter3DBVH(
            bvh=bvh,
            sigma_los=sigma_los,
            sigma_nlos=sigma_nlos,
            nlos_bias=nlos_bias,
            blocked_nlos_prob=blocked_nlos_prob,
            clear_nlos_prob=clear_nlos_prob,
            n_particles=n_particles,
            sigma_pos=PF_SIGMA_POS,
            sigma_cb=PF_SIGMA_CB,
            sigma_pr=sigma_pr,
            resampling="megopolis",
            seed=42,
        )

    raise ValueError(f"unknown PF3D variant: {variant}")


def run_pf3d_variant(
    data: dict,
    building_model: object | None,
    n_particles: int,
    wls_init: np.ndarray,
    variant: str,
    sigma_pr: float = PF_SIGMA_LOS,
    sigma_los: float = PF_SIGMA_LOS,
    sigma_nlos: float = PF_SIGMA_NLOS,
    nlos_bias: float = PF_NLOS_BIAS,
    blocked_nlos_prob: float = 1.0,
    clear_nlos_prob: float = 0.0,
    quality_veto_config: MultiGNSSQualityVetoConfig | None = None,
    guide_reference_positions: np.ndarray | None = None,
    guide_initial_from_reference: bool = False,
    guide_mode: str = "always",
    guide_satellite_max: int | None = None,
    rescue_reference_positions: np.ndarray | None = None,
    rescue_distance_m: float | None = None,
    rescue_spread_pos: float = PF_RESCUE_SPREAD_POS,
    rescue_spread_cb: float = PF_RESCUE_SPREAD_CB,
    return_states: bool = False,
) -> tuple[np.ndarray, float, str]:
    n_epochs = data["n_epochs"]
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    system_ids = data.get("system_ids")
    positions = np.zeros((n_epochs, 3))
    states = np.zeros((n_epochs, 4)) if return_states else None
    t0 = time.perf_counter()
    multi_solver = None
    if (
        quality_veto_config is not None
        and system_ids is not None
        and len(data.get("constellations", ())) > 1
    ):
        systems = sorted({int(sid) for epoch_ids in system_ids for sid in epoch_ids})
        multi_solver = MultiGNSSSolver(systems=systems)
    fallback_epochs = 0
    rescue_epochs = 0
    init_reference = (
        np.asarray(guide_reference_positions[0], dtype=np.float64)
        if guide_initial_from_reference and guide_reference_positions is not None
        else None
    )
    init_position = (
        np.asarray(init_reference[:3], dtype=np.float64)
        if init_reference is not None
        else np.asarray(wls_init[0, :3], dtype=np.float64)
    )
    init_clock_bias = (
        float(init_reference[3])
        if init_reference is not None and init_reference.size > 3
        else float(wls_init[0, 3])
    )

    if building_model is None:
        fallback_pos, fallback_ms, fallback_backend = run_pf_standard(
            data,
            n_particles,
            wls_init,
            sigma_pr=sigma_pr,
            quality_veto_config=quality_veto_config,
            guide_reference_positions=guide_reference_positions,
            guide_initial_from_reference=guide_initial_from_reference,
            guide_mode=guide_mode,
            guide_satellite_max=guide_satellite_max,
            rescue_reference_positions=rescue_reference_positions,
            rescue_distance_m=rescue_distance_m,
            rescue_spread_pos=rescue_spread_pos,
            rescue_spread_cb=rescue_spread_cb,
            return_states=return_states,
        )
        return fallback_pos, fallback_ms, f"{fallback_backend}-no-model"

    try:
        pf = _build_pf3d_variant(
            building_model,
            n_particles,
            variant,
            sigma_pr=sigma_pr,
            sigma_los=sigma_los,
            sigma_nlos=sigma_nlos,
            nlos_bias=nlos_bias,
            blocked_nlos_prob=blocked_nlos_prob,
            clear_nlos_prob=clear_nlos_prob,
        )
        pf.initialize(
            init_position,
            clock_bias=init_clock_bias,
            spread_pos=50.0,
            spread_cb=500.0,
        )
        for i in range(n_epochs):
            dt = _epoch_dt(data, i)
            sat_i, pr_i, w_i, use_multi = _select_pf_epoch_measurements(
                sat_ecef[i],
                pseudoranges[i],
                weights[i],
                None if system_ids is None else system_ids[i],
                multi_solver,
                quality_veto_config,
            )
            pf.predict(
                velocity=_select_guide_velocity(
                    guide_reference_positions,
                    i,
                    dt,
                    guide_mode,
                    use_multi,
                    sat_i.shape[0],
                    guide_satellite_max=guide_satellite_max,
                ),
                dt=dt,
            )
            if _should_rescue_pf_epoch(
                pf.estimate(),
                None if rescue_reference_positions is None else rescue_reference_positions[i],
                rescue_distance_m,
            ):
                reference_position = np.asarray(rescue_reference_positions[i], dtype=np.float64)
                estimate_state = np.asarray(pf.estimate(), dtype=np.float64)
                rescue_clock_bias = (
                    float(reference_position[3])
                    if reference_position.size > 3
                    else float(estimate_state[3])
                )
                pf.initialize(
                    reference_position[:3],
                    clock_bias=rescue_clock_bias,
                    spread_pos=rescue_spread_pos,
                    spread_cb=rescue_spread_cb,
                )
                rescue_epochs += 1
            fallback_epochs += int(not use_multi)
            pf.update(sat_i, pr_i, weights=w_i)
            estimate = np.asarray(pf.estimate(), dtype=np.float64)
            positions[i] = estimate[:3]
            if states is not None:
                states[i] = estimate
        backend = "GPU-3D" if variant == "pf3d" else "GPU-3D-BVH"
    except (ImportError, RuntimeError, Exception) as e:
        print(f"      {variant} failed ({type(e).__name__}: {e}), falling back to standard PF")
        fallback_pos, fallback_ms, fallback_backend = run_pf_standard(
            data,
            n_particles,
            wls_init,
            sigma_pr=sigma_pr,
            quality_veto_config=quality_veto_config,
            guide_reference_positions=guide_reference_positions,
            guide_initial_from_reference=guide_initial_from_reference,
            guide_mode=guide_mode,
            guide_satellite_max=guide_satellite_max,
            rescue_reference_positions=rescue_reference_positions,
            rescue_distance_m=rescue_distance_m,
            rescue_spread_pos=rescue_spread_pos,
            rescue_spread_cb=rescue_spread_cb,
            return_states=return_states,
        )
        return fallback_pos, fallback_ms, f"{fallback_backend}-fallback"

    ms = (time.perf_counter() - t0) * 1000.0 / max(n_epochs, 1)
    if quality_veto_config is not None and multi_solver is not None:
        backend += f"+QVGated({fallback_epochs}fb)"
    backend += _guide_backend_suffix(
        guide_reference_positions,
        guide_mode,
        guide_satellite_max,
    )
    if rescue_reference_positions is not None and rescue_distance_m is not None:
        backend += f"+Rescue({rescue_epochs})"
    return (states if states is not None else positions), ms, backend


def _compute_nlos_stats(data: dict, accelerator: object | None, ground_truth_pos: np.ndarray) -> dict:
    stats = {
        "n_total_obs": 0,
        "n_nlos_classified": 0,
        "nlos_fraction": 0.0,
    }
    if accelerator is None:
        return stats

    try:
        sat_ecef = data["sat_ecef"]
        n_epochs = data["n_epochs"]
        stride = max(1, n_epochs // 30)

        n_total = 0
        n_nlos = 0
        for i in range(0, n_epochs, stride):
            sats = np.asarray(sat_ecef[i], dtype=np.float64).reshape(-1, 3)
            if sats.size == 0:
                continue
            is_los = np.asarray(accelerator.check_los(ground_truth_pos[i], sats), dtype=bool)
            n_total += int(is_los.size)
            n_nlos += int(np.count_nonzero(~is_los))

        stats["n_total_obs"] = n_total
        stats["n_nlos_classified"] = n_nlos
        stats["nlos_fraction"] = float(n_nlos / n_total) if n_total else 0.0
    except Exception as e:
        print(f"      NLOS stats computation failed: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="3D-aware PF evaluation on PPC or synthetic data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to PPC run directory, city directory, dataset root, or none for synthetic",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to PLATEAU CityGML directory",
    )
    parser.add_argument(
        "--plateau-zone",
        type=int,
        default=9,
        help="Japanese plane rectangular zone for PLATEAU CityGML (default: 9)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=300,
        help="Number of epochs for synthetic fallback (default: 300)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Optional cap on real-data epochs",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Skip this many usable real-data epochs before evaluation",
    )
    parser.add_argument(
        "--systems",
        type=str,
        default="G",
        help="Comma-separated constellations for real data, e.g. G or G,E",
    )
    parser.add_argument(
        "--urban-rover",
        type=str,
        default="ublox",
        help="UrbanNav rover observation source, e.g. ublox or trimble",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=N_PARTICLES_PF3D,
        help=f"Particle count (default: {N_PARTICLES_PF3D})",
    )
    parser.add_argument(
        "--sigma-pr",
        type=float,
        default=PF_SIGMA_LOS,
        help=f"Standard pseudorange noise sigma for PF/PF3D variants (default: {PF_SIGMA_LOS})",
    )
    parser.add_argument(
        "--sigma-los",
        type=float,
        default=PF_SIGMA_LOS,
        help=f"LOS likelihood sigma for PF3D variants (default: {PF_SIGMA_LOS})",
    )
    parser.add_argument(
        "--sigma-nlos",
        type=float,
        default=PF_SIGMA_NLOS,
        help=f"NLOS likelihood sigma for PF3D variants (default: {PF_SIGMA_NLOS})",
    )
    parser.add_argument(
        "--nlos-bias",
        type=float,
        default=PF_NLOS_BIAS,
        help=f"Expected positive NLOS bias in metres (default: {PF_NLOS_BIAS})",
    )
    parser.add_argument(
        "--blocked-nlos-prob",
        type=float,
        default=1.0,
        help="P(NLOS | ray blocked) for PF3D variants (default: 1.0)",
    )
    parser.add_argument(
        "--clear-nlos-prob",
        type=float,
        default=0.0,
        help="P(NLOS | ray clear) for PF3D variants (default: 0.0)",
    )
    parser.add_argument(
        "--disable-synthetic-model",
        action="store_true",
        help="Do not create a synthetic fallback building model when no PLATEAU model is provided",
    )
    parser.add_argument(
        "--skip-bvh",
        action="store_true",
        help="Skip PF3D-BVH even when BVH bindings are available",
    )
    parser.add_argument(
        "--skip-pf3d",
        action="store_true",
        help="Skip the linear-scan PF3D variant and run only PF / PF3D-BVH",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="pf3d",
        help="Output filename prefix under experiments/results/",
    )
    args = parser.parse_args()
    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Experiment: 3D-aware Particle Filter on PPC / Synthetic Data")
    print("=" * 72)

    # ------------------------------------------------------------------
    # [1] Data
    # ------------------------------------------------------------------
    print("\n[1] Loading data ...")
    data = load_or_generate_data(
        args.data_dir,
        n_epochs=args.n_epochs,
        max_real_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        systems=systems,
        urban_rover=args.urban_rover,
    )
    ground_truth = data["ground_truth"]
    n_epochs = data["n_epochs"]
    times = np.asarray(data["times"], dtype=np.float64)
    if "dataset_name" in data:
        print(f"    Dataset: {data['dataset_name']}")

    # ------------------------------------------------------------------
    # [2] Building model
    # ------------------------------------------------------------------
    print("\n[2] Loading 3D building model ...")
    building_model = None
    building_source = "none"

    if args.model_dir is not None and args.model_dir.exists():
        building_model = load_plateau_model(args.model_dir, zone=args.plateau_zone)
        if building_model is not None:
            building_source = f"PLATEAU:{args.model_dir}"

    if building_model is None and not args.disable_synthetic_model:
        print("    PLATEAU model not found. Creating synthetic building model.")
        building_model = create_synthetic_building_model(np.asarray(data["origin_ecef"], dtype=np.float64))
        if building_model is not None:
            building_source = "synthetic"

    if building_model is None:
        print("    WARNING: no building model available. PF3D variants will fall back.")
    else:
        print(f"    Building model source: {building_source}")

    # ------------------------------------------------------------------
    # [3] Baselines
    # ------------------------------------------------------------------
    print("\n[3] Running baselines ...")
    wls_pos, wls_ms = run_wls(data)
    wls_metrics = _augment_tail_metrics(compute_metrics(wls_pos[:, :3], ground_truth), times)
    wls_metrics["time_ms"] = wls_ms
    print(f"    WLS: RMS 2D={wls_metrics['rms_2d']:.2f} m, P95={wls_metrics['p95']:.2f} m")

    ekf_pos, ekf_ms = run_ekf(data, wls_pos)
    ekf_metrics = _augment_tail_metrics(compute_metrics(ekf_pos, ground_truth), times)
    ekf_metrics["time_ms"] = ekf_ms
    print(f"    EKF: RMS 2D={ekf_metrics['rms_2d']:.2f} m, P95={ekf_metrics['p95']:.2f} m")

    # ------------------------------------------------------------------
    # [4] Standard PF
    # ------------------------------------------------------------------
    print(f"\n[4] Running standard PF ({args.n_particles} particles) ...")
    pf_pos, pf_ms, pf_backend = run_pf_standard(
        data,
        args.n_particles,
        wls_pos,
        sigma_pr=args.sigma_pr,
    )
    pf_metrics = _augment_tail_metrics(compute_metrics(pf_pos, ground_truth), times)
    pf_metrics["time_ms"] = pf_ms
    pf_metrics["backend"] = pf_backend
    print(f"    PF ({pf_backend}): RMS 2D={pf_metrics['rms_2d']:.2f} m, P95={pf_metrics['p95']:.2f} m")

    # ------------------------------------------------------------------
    # [5] PF3D
    # ------------------------------------------------------------------
    pf3d_metrics = None
    pf3d_pos = None
    pf3d_backend = "-"
    if args.skip_pf3d:
        print("\n[5] Skipping PF3D by request.")
    else:
        print(f"\n[5] Running PF3D ({args.n_particles} particles) ...")
        pf3d_pos, pf3d_ms, pf3d_backend = run_pf3d_variant(
            data,
            building_model,
            args.n_particles,
            wls_pos,
            "pf3d",
            sigma_pr=args.sigma_pr,
            sigma_los=args.sigma_los,
            sigma_nlos=args.sigma_nlos,
            nlos_bias=args.nlos_bias,
            blocked_nlos_prob=args.blocked_nlos_prob,
            clear_nlos_prob=args.clear_nlos_prob,
        )
        pf3d_metrics = _augment_tail_metrics(compute_metrics(pf3d_pos, ground_truth), times)
        pf3d_metrics["time_ms"] = pf3d_ms
        pf3d_metrics["backend"] = pf3d_backend
        print(
            f"    PF3D ({pf3d_backend}): RMS 2D={pf3d_metrics['rms_2d']:.2f} m, "
            f"P95={pf3d_metrics['p95']:.2f} m"
        )

    # ------------------------------------------------------------------
    # [6] PF3D-BVH
    # ------------------------------------------------------------------
    pf3d_bvh_metrics = None
    pf3d_bvh_pos = None
    pf3d_bvh_backend = "-"
    if args.skip_bvh:
        print("\n[6] Skipping PF3D-BVH by request.")
    else:
        print(f"\n[6] Running PF3D-BVH ({args.n_particles} particles) ...")
        pf3d_bvh_pos, pf3d_bvh_ms, pf3d_bvh_backend = run_pf3d_variant(
            data,
            building_model,
            args.n_particles,
            wls_pos,
            "pf3d_bvh",
            sigma_pr=args.sigma_pr,
            sigma_los=args.sigma_los,
            sigma_nlos=args.sigma_nlos,
            nlos_bias=args.nlos_bias,
            blocked_nlos_prob=args.blocked_nlos_prob,
            clear_nlos_prob=args.clear_nlos_prob,
        )
        pf3d_bvh_metrics = _augment_tail_metrics(compute_metrics(pf3d_bvh_pos, ground_truth), times)
        pf3d_bvh_metrics["time_ms"] = pf3d_bvh_ms
        pf3d_bvh_metrics["backend"] = pf3d_bvh_backend
        print(
            f"    PF3D-BVH ({pf3d_bvh_backend}): RMS 2D={pf3d_bvh_metrics['rms_2d']:.2f} m, "
            f"P95={pf3d_bvh_metrics['p95']:.2f} m"
        )

    # ------------------------------------------------------------------
    # [7] NLOS statistics
    # ------------------------------------------------------------------
    print("\n[7] NLOS statistics ...")
    nlos_stats = _compute_nlos_stats(data, building_model, ground_truth)
    if nlos_stats["n_total_obs"] > 0:
        print(
            f"    Ray-traced NLOS fraction: {100.0 * nlos_stats['nlos_fraction']:.2f}% "
            f"over {nlos_stats['n_total_obs']} sampled observations"
        )
    else:
        print("    NLOS statistics unavailable.")

    # ------------------------------------------------------------------
    # [8] Comparison
    # ------------------------------------------------------------------
    print("\n[8] Results summary:")
    all_metrics = {
        "WLS": wls_metrics,
        "EKF": ekf_metrics,
        f"PF-{args.n_particles // 1000}K": pf_metrics,
    }
    positions_by_label = {
        "WLS": wls_pos[:, :3],
        "EKF": ekf_pos,
        f"PF-{args.n_particles // 1000}K": pf_pos,
    }
    if pf3d_metrics is not None and pf3d_pos is not None:
        all_metrics[f"PF3D-{args.n_particles // 1000}K"] = pf3d_metrics
        positions_by_label[f"PF3D-{args.n_particles // 1000}K"] = pf3d_pos
    if pf3d_bvh_metrics is not None and pf3d_bvh_pos is not None:
        all_metrics[f"PF3D-BVH-{args.n_particles // 1000}K"] = pf3d_bvh_metrics
        positions_by_label[f"PF3D-BVH-{args.n_particles // 1000}K"] = pf3d_bvh_pos

    print_comparison_table(all_metrics)

    # ------------------------------------------------------------------
    # [9] Save results
    # ------------------------------------------------------------------
    print("\n[9] Saving results ...")
    epochs = np.arange(n_epochs)
    row_data: dict[str, object] = {
        "epoch": epochs,
        "gps_tow": times,
        "satellite_count": np.asarray(data.get("satellite_counts", np.full(n_epochs, data.get("n_satellites", 0))), dtype=np.int32),
        "gt_x": ground_truth[:, 0],
        "gt_y": ground_truth[:, 1],
        "gt_z": ground_truth[:, 2],
    }
    for label, pos in positions_by_label.items():
        key = label.lower().replace("-", "_")
        row_data[f"error_2d_{key}"] = all_metrics[label]["errors_2d"]
        pos_arr = np.asarray(pos, dtype=np.float64)
        row_data[f"est_x_{key}"] = pos_arr[:, 0]
        row_data[f"est_y_{key}"] = pos_arr[:, 1]
        row_data[f"est_z_{key}"] = pos_arr[:, 2]
    save_results(row_data, RESULTS_DIR / f"{args.results_prefix}_results.csv")

    method_labels = list(all_metrics.keys())
    summary = {
        "method": method_labels,
        "rms_2d": [all_metrics[m]["rms_2d"] for m in method_labels],
        "mean_2d": [all_metrics[m]["mean_2d"] for m in method_labels],
        "p50": [all_metrics[m]["p50"] for m in method_labels],
        "p95": [all_metrics[m]["p95"] for m in method_labels],
        "max_2d": [all_metrics[m]["max_2d"] for m in method_labels],
        "outlier_rate_pct": [all_metrics[m]["outlier_rate_pct"] for m in method_labels],
        "catastrophic_rate_pct": [all_metrics[m]["catastrophic_rate_pct"] for m in method_labels],
        "longest_outlier_segment_epochs": [
            all_metrics[m]["longest_outlier_segment_epochs"] for m in method_labels
        ],
        "longest_outlier_segment_s": [
            all_metrics[m]["longest_outlier_segment_s"] for m in method_labels
        ],
        "time_ms": [all_metrics[m].get("time_ms", 0.0) for m in method_labels],
        "backend": [all_metrics[m].get("backend", "-") for m in method_labels],
        "building_source": [building_source for _ in method_labels],
        "n_epochs": [n_epochs for _ in method_labels],
    }
    save_results(summary, RESULTS_DIR / f"{args.results_prefix}_summary.csv")

    save_results(
        {
            "building_source": [building_source],
            "n_total_obs": [nlos_stats["n_total_obs"]],
            "n_nlos_classified": [nlos_stats["n_nlos_classified"]],
            "nlos_fraction": [nlos_stats["nlos_fraction"]],
        },
        RESULTS_DIR / f"{args.results_prefix}_nlos_stats.csv",
    )

    # ------------------------------------------------------------------
    # [10] Plots
    # ------------------------------------------------------------------
    if not args.no_plots:
        print("\n[10] Generating plots ...")
        errors_for_cdf = {label: all_metrics[label]["errors_2d"] for label in all_metrics}
        plot_cdf(
            errors_for_cdf,
            RESULTS_DIR / f"{args.results_prefix}_cdf.png",
            title="CDF of 2D Positioning Error - PF3D Variants",
        )
        plot_error_timeline(
            times,
            errors_for_cdf,
            RESULTS_DIR / f"{args.results_prefix}_timeline.png",
            title="2D Positioning Error Over Time - PF3D Variants",
        )

    print(f"\n{'=' * 72}")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
