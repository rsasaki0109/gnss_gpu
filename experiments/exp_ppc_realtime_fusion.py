#!/usr/bin/env python3
# ruff: noqa: E402
"""Evaluate causal PPC TDCP motion with DD-pseudorange base-station anchors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from evaluate import compute_metrics, ecef_errors_2d_3d
from exp_ppc_tdcp_velocity import _epoch_measurements, _velocity_truth
from exp_urbannav_baseline import run_wls
from gnss_gpu.dd_likelihood import dd_log_likelihood_gradients
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer, DDPseudorangeResult
from gnss_gpu.io.ppc import PPCDatasetLoader
from gnss_gpu.ppc_score import ppc_3d_errors, ppc_score_dict, ppc_segment_distances
from gnss_gpu.reservoir_stein import ReservoirSteinConfig, reservoir_stein_update
from gnss_gpu.tdcp_velocity import L1_WAVELENGTH, estimate_velocity_from_tdcp_with_metrics
from gnss_gpu.widelane import WidelaneDDPseudorangeComputer

RESULTS_DIR = _SCRIPT_DIR / "results"
_SYSTEM_ID_MAP = {"G": 0, "R": 1, "E": 2, "C": 3, "J": 4}
_WGS84_A = 6_378_137.0
_WGS84_E2 = 6.694379990141316e-3


@dataclass(frozen=True)
class _DDAnchorStats:
    accepted: bool = False
    n_dd: int = 0
    kept_pairs: int = 0
    shift_m: float = float("nan")
    robust_rms_m: float = float("nan")


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _ecef_to_llh(ecef: np.ndarray) -> tuple[float, float, float]:
    x, y, z = np.asarray(ecef, dtype=np.float64).reshape(3)
    lon = math.atan2(float(y), float(x))
    p = math.hypot(float(x), float(y))
    lat = math.atan2(float(z), p * (1.0 - _WGS84_E2))
    for _ in range(6):
        sin_lat = math.sin(lat)
        n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        lat = math.atan2(float(z) + _WGS84_E2 * n * sin_lat, p)
    sin_lat = math.sin(lat)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    alt = p / max(math.cos(lat), 1.0e-12) - n
    return lat, lon, alt


def _llh_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    sin_lat = math.sin(float(lat))
    cos_lat = math.cos(float(lat))
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        [
            (n + float(alt)) * cos_lat * math.cos(float(lon)),
            (n + float(alt)) * cos_lat * math.sin(float(lon)),
            (n * (1.0 - _WGS84_E2) + float(alt)) * sin_lat,
        ],
        dtype=np.float64,
    )


def _blend_to_altitude(
    ecef: np.ndarray,
    target_alt_m: float,
    *,
    alpha: float,
) -> tuple[np.ndarray, float]:
    pos = np.asarray(ecef, dtype=np.float64).reshape(3)
    blend_alpha = float(np.clip(alpha, 0.0, 1.0))
    if blend_alpha <= 0.0:
        return pos.copy(), 0.0
    lat, lon, _alt = _ecef_to_llh(pos)
    target = _llh_to_ecef(lat, lon, float(target_alt_m))
    corrected = pos + blend_alpha * (target - pos)
    return corrected, float(np.linalg.norm(corrected - pos))


def _project_to_reference_radius(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    cand = np.asarray(candidate, dtype=np.float64).reshape(3)
    ref = np.asarray(reference, dtype=np.float64).reshape(3)
    cand_norm = float(np.linalg.norm(cand))
    ref_norm = float(np.linalg.norm(ref))
    if cand_norm <= 0.0 or ref_norm <= 0.0:
        return cand.copy()
    return cand * (ref_norm / cand_norm)


def _height_hold_effective_alpha(
    base_alpha: float,
    *,
    last_velocity_used: bool,
    anchor_stats: _DDAnchorStats,
    release_on_last_velocity: bool,
    release_on_dd_shift: bool,
    release_min_dd_shift_m: float,
) -> float:
    alpha = float(np.clip(base_alpha, 0.0, 1.0))
    if (
        alpha > 0.0
        and np.isfinite(anchor_stats.shift_m)
        and anchor_stats.shift_m >= float(release_min_dd_shift_m)
        and (release_on_dd_shift or (release_on_last_velocity and last_velocity_used))
    ):
        return 0.0
    return alpha


def _height_reference_trusted(
    anchor_stats: _DDAnchorStats,
    *,
    max_initial_dd_rms_m: float,
) -> bool:
    if not np.isfinite(max_initial_dd_rms_m):
        return True
    return (
        np.isfinite(anchor_stats.robust_rms_m)
        and anchor_stats.robust_rms_m <= float(max_initial_dd_rms_m)
    )


def _dd_anchor_effective_alpha(
    base_alpha: float,
    *,
    high_alpha: float,
    anchor_stats: _DDAnchorStats,
    high_min_shift_m: float,
    high_max_robust_rms_m: float,
) -> float:
    alpha = float(np.clip(base_alpha, 0.0, 1.0))
    elevated_alpha = float(np.clip(high_alpha, 0.0, 1.0))
    if elevated_alpha <= alpha:
        return alpha
    if (
        np.isfinite(anchor_stats.shift_m)
        and np.isfinite(anchor_stats.robust_rms_m)
        and anchor_stats.shift_m >= float(high_min_shift_m)
        and anchor_stats.robust_rms_m <= float(high_max_robust_rms_m)
    ):
        return elevated_alpha
    return alpha


def _sat_elevation(rx_ecef: np.ndarray, sat_ecef: np.ndarray) -> float:
    lat, lon, _alt = _ecef_to_llh(rx_ecef)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    dx = np.asarray(sat_ecef, dtype=np.float64).reshape(3) - np.asarray(
        rx_ecef,
        dtype=np.float64,
    ).reshape(3)
    east = -sin_lon * dx[0] + cos_lon * dx[1]
    north = -sin_lat * cos_lon * dx[0] - sin_lat * sin_lon * dx[1] + cos_lat * dx[2]
    up = cos_lat * cos_lon * dx[0] + cos_lat * sin_lon * dx[1] + sin_lat * dx[2]
    return float(math.atan2(up, math.hypot(east, north)))


def _dd_measurements(data: dict, epoch_idx: int, rx_ecef: np.ndarray) -> list[SimpleNamespace]:
    rows: list[SimpleNamespace] = []
    sat_ids = data["used_prns"][epoch_idx]
    sat_ecef = np.asarray(data["sat_ecef"][epoch_idx], dtype=np.float64)
    pseudoranges = np.asarray(data["pseudoranges"][epoch_idx], dtype=np.float64)
    weights = np.asarray(data["weights"][epoch_idx], dtype=np.float64)
    for i, sat_id in enumerate(sat_ids):
        if not sat_id or sat_id[0] not in _SYSTEM_ID_MAP:
            continue
        if i >= sat_ecef.shape[0] or i >= len(pseudoranges):
            continue
        rows.append(
            SimpleNamespace(
                system_id=_SYSTEM_ID_MAP[sat_id[0]],
                prn=int(sat_id[1:]),
                satellite_ecef=sat_ecef[i],
                corrected_pseudorange=float(pseudoranges[i]),
                elevation=_sat_elevation(rx_ecef, sat_ecef[i]),
                snr=float(weights[i]) if i < len(weights) else 1.0,
                weight=float(weights[i]) if i < len(weights) else 1.0,
            )
        )
    return rows


def _dd_residual_and_design(
    dd: DDPseudorangeResult,
    pos: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    range_k = np.linalg.norm(dd.sat_ecef_k[mask] - pos, axis=1)
    range_ref = np.linalg.norm(dd.sat_ecef_ref[mask] - pos, axis=1)
    expected = range_k - range_ref - dd.base_range_k[mask] + dd.base_range_ref[mask]
    residual = dd.dd_pseudorange_m[mask] - expected
    unit_k = (dd.sat_ecef_k[mask] - pos) / np.maximum(range_k[:, np.newaxis], 1.0)
    unit_ref = (dd.sat_ecef_ref[mask] - pos) / np.maximum(
        range_ref[:, np.newaxis],
        1.0,
    )
    design = -unit_k + unit_ref
    return residual, design


def _robust_dd_pr_anchor(
    seed_ecef: np.ndarray,
    dd: DDPseudorangeResult | None,
    *,
    huber_k_m: float,
    trim_m: float,
    min_kept_pairs: int,
    max_shift_m: float,
    max_iter: int = 8,
    tol_m: float = 1.0e-3,
) -> tuple[np.ndarray | None, _DDAnchorStats]:
    if dd is None or int(dd.n_dd) < 3:
        return None, _DDAnchorStats(n_dd=0 if dd is None else int(dd.n_dd))

    seed = np.asarray(seed_ecef, dtype=np.float64).reshape(3)
    pos = seed.copy()
    mask = np.ones(int(dd.n_dd), dtype=bool)
    for _iteration in range(int(max_iter)):
        if int(np.count_nonzero(mask)) < 3:
            return None, _DDAnchorStats(n_dd=int(dd.n_dd), kept_pairs=int(np.count_nonzero(mask)))
        residual, design = _dd_residual_and_design(dd, pos, mask)
        abs_res = np.abs(residual)
        base_w = np.clip(np.asarray(dd.dd_weights, dtype=np.float64)[mask], 1.0e-6, None)
        if huber_k_m > 0.0:
            robust_w = np.minimum(1.0, float(huber_k_m) / np.maximum(abs_res, 1.0e-6))
        else:
            robust_w = np.ones_like(abs_res)
        sw = np.sqrt(base_w * robust_w)
        try:
            delta, *_ = np.linalg.lstsq(design * sw[:, np.newaxis], residual * sw, rcond=None)
        except np.linalg.LinAlgError:
            return None, _DDAnchorStats(n_dd=int(dd.n_dd), kept_pairs=int(np.count_nonzero(mask)))
        if not np.all(np.isfinite(delta)):
            return None, _DDAnchorStats(n_dd=int(dd.n_dd), kept_pairs=int(np.count_nonzero(mask)))
        pos += delta

        full_mask = np.ones(int(dd.n_dd), dtype=bool)
        full_residual, _ = _dd_residual_and_design(dd, pos, full_mask)
        if np.isfinite(trim_m) and trim_m > 0.0:
            trimmed = np.abs(full_residual) <= float(trim_m)
            if int(np.count_nonzero(trimmed)) >= 3:
                mask = trimmed
        if float(np.linalg.norm(delta)) < float(tol_m):
            break

    full_mask = np.ones(int(dd.n_dd), dtype=bool)
    full_residual, _ = _dd_residual_and_design(dd, pos, full_mask)
    kept_pairs = (
        int(np.count_nonzero(np.abs(full_residual) <= float(trim_m)))
        if np.isfinite(trim_m) and trim_m > 0.0
        else int(dd.n_dd)
    )
    shift_m = float(np.linalg.norm(pos - seed))
    if huber_k_m > 0.0:
        robust_rms = float(np.sqrt(np.mean(np.minimum(full_residual * full_residual, huber_k_m * huber_k_m))))
    else:
        robust_rms = float(np.sqrt(np.mean(full_residual * full_residual)))
    accepted = (
        kept_pairs >= int(min_kept_pairs)
        and shift_m <= float(max_shift_m)
        and np.all(np.isfinite(pos))
    )
    stats = _DDAnchorStats(
        accepted=bool(accepted),
        n_dd=int(dd.n_dd),
        kept_pairs=kept_pairs,
        shift_m=shift_m,
        robust_rms_m=robust_rms,
    )
    return (pos if accepted else None), stats


def _try_dd_anchor(
    dd_computer: DDPseudorangeComputer,
    data: dict,
    epoch_idx: int,
    seed_ecef: np.ndarray,
    *,
    huber_k_m: float,
    trim_m: float,
    min_kept_pairs: int,
    max_shift_m: float,
) -> tuple[np.ndarray | None, _DDAnchorStats]:
    measurements = _dd_measurements(data, epoch_idx, seed_ecef)
    dd = dd_computer.compute_dd(
        float(data["times"][epoch_idx]),
        measurements,
        rover_position_approx=seed_ecef,
        min_common_sats=4,
    )
    return _robust_dd_pr_anchor(
        seed_ecef,
        dd,
        huber_k_m=huber_k_m,
        trim_m=trim_m,
        min_kept_pairs=min_kept_pairs,
        max_shift_m=max_shift_m,
    )


def _try_rsp_correction(
    dd_computer: DDPseudorangeComputer,
    data: dict,
    epoch_idx: int,
    seed_ecef: np.ndarray,
    *,
    n_particles: int,
    spread_m: float,
    sigma_m: float,
    huber_k_m: float,
    stein_steps: int,
    stein_step_size: float,
    repulsion_scale: float,
    random_seed: int,
) -> tuple[np.ndarray | None, dict[str, object]]:
    if n_particles <= 0 or spread_m <= 0.0 or sigma_m <= 0.0:
        return None, _rsp_epoch_fields(False)

    seed = np.asarray(seed_ecef, dtype=np.float64).reshape(3)
    measurements = _dd_measurements(data, epoch_idx, seed)
    dd = dd_computer.compute_dd(
        float(data["times"][epoch_idx]),
        measurements,
        rover_position_approx=seed,
        min_common_sats=4,
    )
    if dd is None or int(dd.n_dd) < 3:
        return None, _rsp_epoch_fields(False, n_dd=0 if dd is None else int(dd.n_dd))

    rng = np.random.default_rng(int(random_seed) + int(epoch_idx))
    particles = seed.reshape(1, 3) + rng.normal(
        0.0,
        float(spread_m),
        size=(int(n_particles), 3),
    )
    gradients = dd_log_likelihood_gradients(
        dd,
        particles,
        sigma_m=float(sigma_m),
        huber_k_m=float(huber_k_m),
    )
    log_weights = -0.5 * np.sum(np.square(particles - seed), axis=1) / (float(spread_m) ** 2)
    result = reservoir_stein_update(
        particles,
        log_weights,
        gradients,
        ReservoirSteinConfig(
            reservoir_size=int(n_particles),
            elite_fraction=0.25,
            stein_steps=int(stein_steps),
            stein_step_size=float(stein_step_size),
            repulsion_scale=float(repulsion_scale),
            seed=int(random_seed) + int(epoch_idx),
        ),
    )
    estimate = np.average(result.particles, axis=0, weights=result.weights)
    corrected = _project_to_reference_radius(estimate, seed)
    shift_m = float(np.linalg.norm(corrected - seed))
    return corrected, _rsp_epoch_fields(
        True,
        n_dd=int(dd.n_dd),
        shift_m=shift_m,
        ess_before=float(result.ess_before),
        mean_gradient_norm_m=float(np.mean(np.linalg.norm(gradients, axis=1))),
    )


def _rsp_epoch_fields(
    used: bool,
    *,
    n_dd: int = 0,
    shift_m: float = float("nan"),
    ess_before: float = float("nan"),
    mean_gradient_norm_m: float = float("nan"),
) -> dict[str, object]:
    return {
        "rsp_correction_used": bool(used),
        "rsp_n_dd": int(n_dd),
        "rsp_shift_m": float(shift_m) if np.isfinite(shift_m) else "",
        "rsp_ess_before": float(ess_before) if np.isfinite(ess_before) else "",
        "rsp_mean_gradient_norm_m": (
            float(mean_gradient_norm_m) if np.isfinite(mean_gradient_norm_m) else ""
        ),
    }


def _empty_widelane_stats(reason: str = "disabled") -> SimpleNamespace:
    return SimpleNamespace(
        reason=reason,
        n_candidate_pairs=0,
        n_fixed_pairs=0,
        fix_rate=0.0,
        n_dd=0,
    )


def _try_widelane_anchor(
    wl_computer: WidelaneDDPseudorangeComputer | None,
    data: dict,
    epoch_idx: int,
    seed_ecef: np.ndarray,
    *,
    huber_k_m: float,
    trim_m: float,
    min_kept_pairs: int,
    max_shift_m: float,
    max_robust_rms_m: float,
    veto_rms_band_min_m: float,
    veto_rms_band_max_m: float,
    veto_min_kept_pairs: int,
) -> tuple[np.ndarray | None, _DDAnchorStats, object]:
    if wl_computer is None:
        return None, _DDAnchorStats(), _empty_widelane_stats()

    measurements = _dd_measurements(data, epoch_idx, seed_ecef)
    wl_result, wl_stats = wl_computer.compute_dd(
        float(data["times"][epoch_idx]),
        measurements,
        rover_position_approx=seed_ecef,
        min_common_sats=max(int(min_kept_pairs) + 1, 4),
    )
    anchor, anchor_stats = _robust_dd_pr_anchor(
        seed_ecef,
        wl_result,
        huber_k_m=huber_k_m,
        trim_m=trim_m,
        min_kept_pairs=min_kept_pairs,
        max_shift_m=max_shift_m,
    )
    if (
        anchor is not None
        and np.isfinite(anchor_stats.robust_rms_m)
        and anchor_stats.robust_rms_m > float(max_robust_rms_m)
    ):
        anchor = None
    if (
        anchor is not None
        and int(anchor_stats.kept_pairs) >= int(veto_min_kept_pairs)
        and np.isfinite(anchor_stats.robust_rms_m)
        and float(veto_rms_band_min_m)
        <= anchor_stats.robust_rms_m
        <= float(veto_rms_band_max_m)
    ):
        anchor = None
    return anchor, anchor_stats, wl_stats


def _widelane_epoch_fields(
    *,
    used: bool,
    anchor_stats: _DDAnchorStats,
    wl_stats: object,
) -> dict[str, object]:
    return {
        "widelane_anchor_used": bool(used),
        "widelane_reason": str(getattr(wl_stats, "reason", "")),
        "widelane_candidate_pairs": int(getattr(wl_stats, "n_candidate_pairs", 0)),
        "widelane_fixed_pairs": int(getattr(wl_stats, "n_fixed_pairs", 0)),
        "widelane_fix_rate": float(getattr(wl_stats, "fix_rate", 0.0)),
        "widelane_n_dd": int(getattr(wl_stats, "n_dd", 0)),
        "widelane_anchor_kept": int(anchor_stats.kept_pairs),
        "widelane_anchor_shift_m": (
            float(anchor_stats.shift_m) if np.isfinite(anchor_stats.shift_m) else ""
        ),
        "widelane_anchor_robust_rms_m": (
            float(anchor_stats.robust_rms_m)
            if np.isfinite(anchor_stats.robust_rms_m)
            else ""
        ),
    }


def _attach_epoch_error_fields(
    per_epoch: list[dict[str, object]],
    wls_pos: np.ndarray,
    fused: np.ndarray,
    truth: np.ndarray,
    *,
    ppc_threshold_m: float = 0.5,
) -> None:
    wls_errors_2d, _ = ecef_errors_2d_3d(wls_pos[:, :3], truth)
    fused_errors_2d, _ = ecef_errors_2d_3d(fused, truth)
    wls_errors_3d = ppc_3d_errors(wls_pos[:, :3], truth)
    fused_errors_3d = ppc_3d_errors(fused, truth)
    segment_distances = ppc_segment_distances(truth)

    for row in per_epoch:
        idx = int(row["epoch"])
        segment_distance = float(segment_distances[idx])
        wls_pass = bool(wls_errors_3d[idx] <= ppc_threshold_m)
        fused_pass = bool(fused_errors_3d[idx] <= ppc_threshold_m)
        row["ppc_segment_distance_m"] = segment_distance
        row["wls_error_2d_m"] = float(wls_errors_2d[idx])
        row["wls_error_3d_m"] = float(wls_errors_3d[idx])
        row["wls_ppc_pass"] = wls_pass
        row["wls_ppc_pass_distance_m"] = segment_distance if wls_pass else 0.0
        row["fused_error_2d_m"] = float(fused_errors_2d[idx])
        row["fused_error_3d_m"] = float(fused_errors_3d[idx])
        row["fused_ppc_pass"] = fused_pass
        row["fused_ppc_pass_distance_m"] = segment_distance if fused_pass else 0.0


def run_fusion_eval(
    data: dict,
    data_dir: Path,
    systems: tuple[str, ...],
    *,
    tdcp_min_sats: int,
    tdcp_max_postfit_rms_m: float,
    tdcp_max_cycle_jump: float,
    tdcp_max_velocity_mps: float,
    carrier_phase_sign: float,
    receiver_motion_sign: float,
    dd_huber_k_m: float,
    dd_trim_m: float,
    dd_min_kept_pairs: int,
    dd_max_shift_m: float,
    dd_anchor_blend_alpha: float,
    dd_anchor_high_blend_alpha: float,
    dd_anchor_high_min_shift_m: float,
    dd_anchor_high_max_robust_rms_m: float,
    dd_interpolate_base_epochs: bool,
    widelane: bool,
    widelane_min_epochs: int,
    widelane_max_std_cycles: float,
    widelane_ratio_threshold: float,
    widelane_min_fix_rate: float,
    widelane_min_kept_pairs: int,
    widelane_max_shift_m: float,
    widelane_max_robust_rms_m: float,
    widelane_veto_rms_band_min_m: float,
    widelane_veto_rms_band_max_m: float,
    widelane_veto_min_kept_pairs: int,
    widelane_anchor_blend_alpha: float,
    height_hold_alpha: float,
    height_hold_release_on_last_velocity: bool,
    height_hold_release_on_dd_shift: bool,
    height_hold_release_min_dd_shift_m: float,
    height_hold_reference_max_dd_rms_m: float,
    rsp_correction: bool,
    rsp_n_particles: int,
    rsp_spread_m: float,
    rsp_sigma_m: float,
    rsp_huber_k_m: float,
    rsp_stein_steps: int,
    rsp_stein_step_size: float,
    rsp_repulsion_scale: float,
    rsp_min_dd_shift_m: float,
    rsp_max_dd_shift_m: float,
    rsp_min_dd_rms_m: float,
    rsp_max_dd_rms_m: float,
    rsp_random_seed: int,
    last_velocity_max_age_s: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    wls_pos, wls_ms = run_wls(data)
    times = np.asarray(data["times"], dtype=np.float64)
    truth = np.asarray(data["ground_truth"], dtype=np.float64)
    truth_vel = _velocity_truth(truth, times)

    dd_computer = DDPseudorangeComputer(
        data_dir / "base.obs",
        rover_obs_path=data_dir / "rover.obs",
        allowed_systems=systems,
        interpolate_base_epochs=dd_interpolate_base_epochs,
    )
    wl_computer = (
        WidelaneDDPseudorangeComputer(
            data_dir / "base.obs",
            data_dir / "rover.obs",
            allowed_systems=systems,
            interpolate_base_epochs=dd_interpolate_base_epochs,
            min_epochs=widelane_min_epochs,
            max_std_cycles=widelane_max_std_cycles,
            ratio_threshold=widelane_ratio_threshold,
            min_fix_rate=widelane_min_fix_rate,
        )
        if widelane
        else None
    )

    fused = np.zeros_like(truth)
    per_epoch: list[dict[str, object]] = []
    n_tdcp_used = 0
    n_dd_used = 0
    n_widelane_used = 0
    n_rsp_used = 0
    n_dd_high_blend_used = 0
    tdcp_errors: list[float] = []
    tdcp_rms_values: list[float] = []
    last_velocity: np.ndarray | None = None
    last_velocity_age = float("inf")
    anchor_alpha = float(np.clip(dd_anchor_blend_alpha, 0.0, 1.0))
    wl_anchor_alpha = float(np.clip(widelane_anchor_blend_alpha, 0.0, 1.0))
    height_alpha = float(np.clip(height_hold_alpha, 0.0, 1.0))

    anchor, anchor_stats = _try_dd_anchor(
        dd_computer,
        data,
        0,
        wls_pos[0, :3],
        huber_k_m=dd_huber_k_m,
        trim_m=dd_trim_m,
        min_kept_pairs=dd_min_kept_pairs,
        max_shift_m=dd_max_shift_m,
    )
    if anchor is not None:
        fused[0] = anchor
        n_dd_used += 1
    else:
        fused[0] = wls_pos[0, :3]
    dd_effective_alpha = 1.0 if anchor is not None else 0.0
    wl_anchor, wl_anchor_stats, wl_stats = _try_widelane_anchor(
        wl_computer,
        data,
        0,
        fused[0],
        huber_k_m=dd_huber_k_m,
        trim_m=dd_trim_m,
        min_kept_pairs=widelane_min_kept_pairs,
        max_shift_m=widelane_max_shift_m,
        max_robust_rms_m=widelane_max_robust_rms_m,
        veto_rms_band_min_m=widelane_veto_rms_band_min_m,
        veto_rms_band_max_m=widelane_veto_rms_band_max_m,
        veto_min_kept_pairs=widelane_veto_min_kept_pairs,
    )
    wl_used = wl_anchor is not None
    if wl_used:
        fused[0] = fused[0] + wl_anchor_alpha * (wl_anchor - fused[0])
        n_widelane_used += 1
    height_reference_alt_m = _ecef_to_llh(fused[0])[2]
    height_reference_trusted = _height_reference_trusted(
        anchor_stats,
        max_initial_dd_rms_m=height_hold_reference_max_dd_rms_m,
    )
    height_correction_m = 0.0
    height_effective_alpha = height_alpha if height_reference_trusted else 0.0
    rsp_fields = _rsp_epoch_fields(False)

    per_epoch.append(
        {
            "epoch": 0,
            "tow": float(times[0]),
            "tdcp_used": False,
            "tdcp_last_velocity_used": False,
            "dd_pr_anchor_used": bool(anchor is not None),
            "dd_pr_n": int(anchor_stats.n_dd),
            "dd_pr_kept": int(anchor_stats.kept_pairs),
            "dd_pr_shift_m": float(anchor_stats.shift_m) if np.isfinite(anchor_stats.shift_m) else "",
            "dd_pr_robust_rms_m": (
                float(anchor_stats.robust_rms_m) if np.isfinite(anchor_stats.robust_rms_m) else ""
            ),
            "dd_anchor_effective_alpha": float(dd_effective_alpha),
            "height_hold_used": bool(height_alpha > 0.0),
            "height_hold_effective_alpha": float(height_effective_alpha),
            "height_hold_reference_alt_m": float(height_reference_alt_m),
            "height_hold_reference_trusted": bool(height_reference_trusted),
            "height_hold_correction_m": float(height_correction_m),
            **rsp_fields,
            **_widelane_epoch_fields(
                used=wl_used,
                anchor_stats=wl_anchor_stats,
                wl_stats=wl_stats,
            ),
        }
    )

    for i in range(1, len(times)):
        dt = float(times[i] - times[i - 1])
        velocity, tdcp_rms = estimate_velocity_from_tdcp_with_metrics(
            fused[i - 1],
            _epoch_measurements(data, i - 1),
            _epoch_measurements(data, i),
            dt=dt,
            wavelength=L1_WAVELENGTH,
            carrier_phase_sign=carrier_phase_sign,
            receiver_motion_sign=receiver_motion_sign,
            min_sats=tdcp_min_sats,
            max_cycle_jump=tdcp_max_cycle_jump,
            max_postfit_rms_m=tdcp_max_postfit_rms_m,
            max_velocity_mps=tdcp_max_velocity_mps,
        )
        tdcp_used = velocity is not None and dt > 0.0
        last_velocity_used = False
        if tdcp_used:
            pred = fused[i - 1] + velocity * dt
            n_tdcp_used += 1
            last_velocity = velocity
            last_velocity_age = 0.0
            tdcp_rms_values.append(float(tdcp_rms))
            if np.all(np.isfinite(truth_vel[i])):
                tdcp_errors.append(float(np.linalg.norm(velocity - truth_vel[i])))
        elif (
            last_velocity is not None
            and dt > 0.0
            and last_velocity_age <= float(last_velocity_max_age_s)
        ):
            pred = fused[i - 1] + last_velocity * dt
            last_velocity_age += dt
            last_velocity_used = True
        else:
            pred = fused[i - 1]
            last_velocity_age = float("inf")

        anchor, anchor_stats = _try_dd_anchor(
            dd_computer,
            data,
            i,
            pred,
            huber_k_m=dd_huber_k_m,
            trim_m=dd_trim_m,
            min_kept_pairs=dd_min_kept_pairs,
            max_shift_m=dd_max_shift_m,
        )
        dd_used = anchor is not None
        dd_effective_alpha = 0.0
        if dd_used:
            dd_effective_alpha = _dd_anchor_effective_alpha(
                anchor_alpha,
                high_alpha=dd_anchor_high_blend_alpha,
                anchor_stats=anchor_stats,
                high_min_shift_m=dd_anchor_high_min_shift_m,
                high_max_robust_rms_m=dd_anchor_high_max_robust_rms_m,
            )
            if dd_effective_alpha > anchor_alpha:
                n_dd_high_blend_used += 1
            pred = pred + dd_effective_alpha * (anchor - pred)
            n_dd_used += 1
        wl_anchor, wl_anchor_stats, wl_stats = _try_widelane_anchor(
            wl_computer,
            data,
            i,
            pred,
            huber_k_m=dd_huber_k_m,
            trim_m=dd_trim_m,
            min_kept_pairs=widelane_min_kept_pairs,
            max_shift_m=widelane_max_shift_m,
            max_robust_rms_m=widelane_max_robust_rms_m,
            veto_rms_band_min_m=widelane_veto_rms_band_min_m,
            veto_rms_band_max_m=widelane_veto_rms_band_max_m,
            veto_min_kept_pairs=widelane_veto_min_kept_pairs,
        )
        wl_used = wl_anchor is not None
        if wl_used:
            pred = pred + wl_anchor_alpha * (wl_anchor - pred)
            n_widelane_used += 1
        height_correction_m = 0.0
        height_effective_alpha = _height_hold_effective_alpha(
            height_alpha,
            last_velocity_used=last_velocity_used,
            anchor_stats=anchor_stats,
            release_on_last_velocity=height_hold_release_on_last_velocity,
            release_on_dd_shift=height_hold_release_on_dd_shift,
            release_min_dd_shift_m=height_hold_release_min_dd_shift_m,
        )
        if not height_reference_trusted:
            height_effective_alpha = 0.0
        if height_effective_alpha > 0.0:
            pred, height_correction_m = _blend_to_altitude(
                pred,
                height_reference_alt_m,
                alpha=height_effective_alpha,
            )
        rsp_fields = _rsp_epoch_fields(False)
        rsp_gate = (
            bool(rsp_correction)
            and bool(tdcp_used)
            and not bool(last_velocity_used)
            and not bool(wl_used)
            and height_effective_alpha > 0.0
            and np.isfinite(anchor_stats.shift_m)
            and float(rsp_min_dd_shift_m) <= anchor_stats.shift_m <= float(rsp_max_dd_shift_m)
            and np.isfinite(anchor_stats.robust_rms_m)
            and float(rsp_min_dd_rms_m)
            <= anchor_stats.robust_rms_m
            <= float(rsp_max_dd_rms_m)
        )
        if rsp_gate:
            rsp_anchor, rsp_fields = _try_rsp_correction(
                dd_computer,
                data,
                i,
                pred,
                n_particles=rsp_n_particles,
                spread_m=rsp_spread_m,
                sigma_m=rsp_sigma_m,
                huber_k_m=rsp_huber_k_m,
                stein_steps=rsp_stein_steps,
                stein_step_size=rsp_stein_step_size,
                repulsion_scale=rsp_repulsion_scale,
                random_seed=rsp_random_seed,
            )
            if rsp_anchor is not None:
                pred = rsp_anchor
                n_rsp_used += 1
        fused[i] = pred

        per_epoch.append(
            {
                "epoch": i,
                "tow": float(times[i]),
                "dt": float(dt),
                "tdcp_used": bool(tdcp_used),
                "tdcp_last_velocity_used": bool(last_velocity_used),
                "tdcp_postfit_rms_m": float(tdcp_rms) if np.isfinite(tdcp_rms) else "",
                "tdcp_velocity_error_mps": (
                    float(np.linalg.norm(velocity - truth_vel[i]))
                    if tdcp_used and np.all(np.isfinite(truth_vel[i]))
                    else ""
                ),
                "dd_pr_anchor_used": bool(dd_used),
                "dd_pr_n": int(anchor_stats.n_dd),
                "dd_pr_kept": int(anchor_stats.kept_pairs),
                "dd_pr_shift_m": (
                    float(anchor_stats.shift_m) if np.isfinite(anchor_stats.shift_m) else ""
                ),
                "dd_pr_robust_rms_m": (
                    float(anchor_stats.robust_rms_m)
                    if np.isfinite(anchor_stats.robust_rms_m)
                    else ""
                ),
                "dd_anchor_effective_alpha": float(dd_effective_alpha),
                "height_hold_used": bool(height_alpha > 0.0),
                "height_hold_effective_alpha": float(height_effective_alpha),
                "height_hold_reference_alt_m": float(height_reference_alt_m),
                "height_hold_reference_trusted": bool(height_reference_trusted),
                "height_hold_correction_m": float(height_correction_m),
                **rsp_fields,
                **_widelane_epoch_fields(
                    used=wl_used,
                    anchor_stats=wl_anchor_stats,
                    wl_stats=wl_stats,
                ),
            }
        )

    wls_metrics = compute_metrics(wls_pos[:, :3], truth)
    fused_metrics = compute_metrics(fused, truth)
    _attach_epoch_error_fields(per_epoch, wls_pos[:, :3], fused, truth)

    tdcp_vel_rmse = (
        float(np.sqrt(np.mean(np.square(tdcp_errors)))) if tdcp_errors else float("nan")
    )
    summary_rows = [
        {
            "method": "WLS",
            "n_epochs": len(times),
            "time_ms_per_epoch": float(wls_ms),
            **ppc_score_dict(wls_pos[:, :3], truth),
            "rms_2d": float(wls_metrics["rms_2d"]),
            "p50": float(wls_metrics["p50"]),
            "p95": float(wls_metrics["p95"]),
            "max_2d": float(wls_metrics["max_2d"]),
        },
        {
            "method": "TDCP + DD-PR/WL anchors" if widelane else "TDCP + DD-PR anchors",
            "n_epochs": len(times),
            "tdcp_used_epochs": int(n_tdcp_used),
            "tdcp_use_rate_pct": float(100.0 * n_tdcp_used / max(len(times) - 1, 1)),
            "tdcp_velocity_rmse_mps": tdcp_vel_rmse,
            "tdcp_postfit_rms_median": (
                float(np.median(tdcp_rms_values)) if tdcp_rms_values else float("nan")
            ),
            "last_velocity_max_age_s": float(last_velocity_max_age_s),
            "dd_pr_anchor_epochs": int(n_dd_used),
            "dd_pr_anchor_rate_pct": float(100.0 * n_dd_used / max(len(times), 1)),
            "dd_anchor_blend_alpha": float(anchor_alpha),
            "dd_anchor_high_blend_alpha": float(dd_anchor_high_blend_alpha),
            "dd_anchor_high_min_shift_m": float(dd_anchor_high_min_shift_m),
            "dd_anchor_high_max_robust_rms_m": float(dd_anchor_high_max_robust_rms_m),
            "dd_anchor_high_blend_epochs": int(n_dd_high_blend_used),
            "widelane_enabled": bool(widelane),
            "widelane_anchor_epochs": int(n_widelane_used),
            "widelane_anchor_rate_pct": float(100.0 * n_widelane_used / max(len(times), 1)),
            "widelane_anchor_blend_alpha": float(wl_anchor_alpha),
            "widelane_max_shift_m": float(widelane_max_shift_m),
            "widelane_max_robust_rms_m": float(widelane_max_robust_rms_m),
            "widelane_veto_rms_band_min_m": float(widelane_veto_rms_band_min_m),
            "widelane_veto_rms_band_max_m": float(widelane_veto_rms_band_max_m),
            "widelane_veto_min_kept_pairs": int(widelane_veto_min_kept_pairs),
            "height_hold_alpha": float(height_alpha),
            "height_hold_release_on_last_velocity": bool(height_hold_release_on_last_velocity),
            "height_hold_release_on_dd_shift": bool(height_hold_release_on_dd_shift),
            "height_hold_release_min_dd_shift_m": float(height_hold_release_min_dd_shift_m),
            "height_hold_reference_max_dd_rms_m": float(height_hold_reference_max_dd_rms_m),
            "height_hold_reference_trusted": bool(height_reference_trusted),
            "height_hold_reference_alt_m": float(height_reference_alt_m),
            "rsp_correction_enabled": bool(rsp_correction),
            "rsp_correction_epochs": int(n_rsp_used),
            "rsp_correction_rate_pct": float(100.0 * n_rsp_used / max(len(times), 1)),
            "rsp_n_particles": int(rsp_n_particles),
            "rsp_spread_m": float(rsp_spread_m),
            "rsp_sigma_m": float(rsp_sigma_m),
            "rsp_huber_k_m": float(rsp_huber_k_m),
            "rsp_stein_steps": int(rsp_stein_steps),
            "rsp_stein_step_size": float(rsp_stein_step_size),
            "rsp_repulsion_scale": float(rsp_repulsion_scale),
            "rsp_min_dd_shift_m": float(rsp_min_dd_shift_m),
            "rsp_max_dd_shift_m": float(rsp_max_dd_shift_m),
            "rsp_min_dd_rms_m": float(rsp_min_dd_rms_m),
            "rsp_max_dd_rms_m": float(rsp_max_dd_rms_m),
            **ppc_score_dict(fused, truth),
            "rms_2d": float(fused_metrics["rms_2d"]),
            "p50": float(fused_metrics["p50"]),
            "p95": float(fused_metrics["p95"]),
            "max_2d": float(fused_metrics["max_2d"]),
        },
    ]
    arrays = {"wls_pos": wls_pos[:, :3], "fused": fused}
    return summary_rows, per_epoch, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal PPC TDCP + DD-PR fusion baseline")
    parser.add_argument("--data-dir", type=Path, required=True, help="PPC run directory")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--systems", type=str, default="G,E,J")
    parser.add_argument("--tdcp-min-sats", type=int, default=5)
    parser.add_argument("--tdcp-max-postfit-rms-m", type=float, default=0.5)
    parser.add_argument("--tdcp-max-cycle-jump", type=float, default=20000.0)
    parser.add_argument("--tdcp-max-velocity-mps", type=float, default=50.0)
    parser.add_argument("--carrier-phase-sign", type=float, default=1.0)
    parser.add_argument("--receiver-motion-sign", type=float, default=-1.0)
    parser.add_argument("--dd-huber-k-m", type=float, default=1.0)
    parser.add_argument("--dd-trim-m", type=float, default=1.5)
    parser.add_argument("--dd-min-kept-pairs", type=int, default=5)
    parser.add_argument("--dd-max-shift-m", type=float, default=200.0)
    parser.add_argument(
        "--dd-anchor-blend-alpha",
        type=float,
        default=0.3,
        help="Causal blend from TDCP prediction toward accepted DD-PR anchor",
    )
    parser.add_argument(
        "--dd-anchor-high-blend-alpha",
        type=float,
        default=0.3,
        help="Elevated DD-PR blend when DD shift/RMS gate accepts",
    )
    parser.add_argument("--dd-anchor-high-min-shift-m", type=float, default=float("inf"))
    parser.add_argument("--dd-anchor-high-max-robust-rms-m", type=float, default=0.7)
    parser.add_argument(
        "--dd-interpolate-base-epochs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpolate 1 Hz base RINEX observations onto PPC rover epochs",
    )
    parser.add_argument(
        "--widelane",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fixed L1-L2 wide-lane DD pseudorange anchors after DD-PR updates",
    )
    parser.add_argument("--widelane-min-epochs", type=int, default=5)
    parser.add_argument("--widelane-max-std-cycles", type=float, default=0.75)
    parser.add_argument("--widelane-ratio-threshold", type=float, default=3.0)
    parser.add_argument("--widelane-min-fix-rate", type=float, default=0.3)
    parser.add_argument("--widelane-min-kept-pairs", type=int, default=3)
    parser.add_argument("--widelane-max-shift-m", type=float, default=5.0)
    parser.add_argument("--widelane-max-robust-rms-m", type=float, default=0.8)
    parser.add_argument("--widelane-veto-rms-band-min-m", type=float, default=0.15)
    parser.add_argument("--widelane-veto-rms-band-max-m", type=float, default=0.35)
    parser.add_argument("--widelane-veto-min-kept-pairs", type=int, default=4)
    parser.add_argument("--widelane-anchor-blend-alpha", type=float, default=1.0)
    parser.add_argument(
        "--height-hold-alpha",
        type=float,
        default=1.0,
        help="Causal blend toward the first fused ellipsoidal height",
    )
    parser.add_argument(
        "--height-hold-release-on-last-velocity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Release height hold when stale TDCP velocity and DD-PR strongly disagree",
    )
    parser.add_argument(
        "--height-hold-release-on-dd-shift",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Release height hold whenever accepted DD-PR shift exceeds the release threshold",
    )
    parser.add_argument("--height-hold-release-min-dd-shift-m", type=float, default=0.4)
    parser.add_argument(
        "--height-hold-reference-max-dd-rms-m",
        type=float,
        default=float("inf"),
        help="Disable height hold when the initial DD-PR height reference RMS exceeds this value",
    )
    parser.add_argument(
        "--rsp-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply gated DD-gradient reservoir Stein horizontal corrections",
    )
    parser.add_argument("--rsp-n-particles", type=int, default=64)
    parser.add_argument("--rsp-spread-m", type=float, default=1.0)
    parser.add_argument("--rsp-sigma-m", type=float, default=1.0)
    parser.add_argument("--rsp-huber-k-m", type=float, default=1.0)
    parser.add_argument("--rsp-stein-steps", type=int, default=1)
    parser.add_argument("--rsp-stein-step-size", type=float, default=0.1)
    parser.add_argument("--rsp-repulsion-scale", type=float, default=0.25)
    parser.add_argument("--rsp-min-dd-shift-m", type=float, default=0.85)
    parser.add_argument("--rsp-max-dd-shift-m", type=float, default=1.4)
    parser.add_argument("--rsp-min-dd-rms-m", type=float, default=0.45)
    parser.add_argument("--rsp-max-dd-rms-m", type=float, default=0.7)
    parser.add_argument("--rsp-random-seed", type=int, default=42)
    parser.add_argument("--last-velocity-max-age-s", type=float, default=8.0)
    parser.add_argument("--results-prefix", type=str, default="ppc_realtime_fusion")
    args = parser.parse_args()

    if not PPCDatasetLoader.is_run_directory(args.data_dir):
        raise FileNotFoundError(f"not a PPC run directory: {args.data_dir}")
    systems = tuple(part.strip().upper() for part in args.systems.split(",") if part.strip())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  PPC Realtime TDCP + DD-PR Fusion")
    print("=" * 72)
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Systems   : {','.join(systems)}")
    print(f"  Start     : {args.start_epoch}")
    print(f"  Max epochs: {args.max_epochs}")

    data = PPCDatasetLoader(args.data_dir).load_experiment_data(
        max_epochs=args.max_epochs,
        start_epoch=args.start_epoch,
        systems=systems,
        include_sat_velocity=True,
    )
    print(f"  Loaded    : {data['dataset_name']} ({data['n_epochs']} epochs)")

    summary_rows, per_epoch_rows, _arrays = run_fusion_eval(
        data,
        args.data_dir,
        systems,
        tdcp_min_sats=args.tdcp_min_sats,
        tdcp_max_postfit_rms_m=args.tdcp_max_postfit_rms_m,
        tdcp_max_cycle_jump=args.tdcp_max_cycle_jump,
        tdcp_max_velocity_mps=args.tdcp_max_velocity_mps,
        carrier_phase_sign=args.carrier_phase_sign,
        receiver_motion_sign=args.receiver_motion_sign,
        dd_huber_k_m=args.dd_huber_k_m,
        dd_trim_m=args.dd_trim_m,
        dd_min_kept_pairs=args.dd_min_kept_pairs,
        dd_max_shift_m=args.dd_max_shift_m,
        dd_anchor_blend_alpha=args.dd_anchor_blend_alpha,
        dd_anchor_high_blend_alpha=args.dd_anchor_high_blend_alpha,
        dd_anchor_high_min_shift_m=args.dd_anchor_high_min_shift_m,
        dd_anchor_high_max_robust_rms_m=args.dd_anchor_high_max_robust_rms_m,
        dd_interpolate_base_epochs=args.dd_interpolate_base_epochs,
        widelane=args.widelane,
        widelane_min_epochs=args.widelane_min_epochs,
        widelane_max_std_cycles=args.widelane_max_std_cycles,
        widelane_ratio_threshold=args.widelane_ratio_threshold,
        widelane_min_fix_rate=args.widelane_min_fix_rate,
        widelane_min_kept_pairs=args.widelane_min_kept_pairs,
        widelane_max_shift_m=args.widelane_max_shift_m,
        widelane_max_robust_rms_m=args.widelane_max_robust_rms_m,
        widelane_veto_rms_band_min_m=args.widelane_veto_rms_band_min_m,
        widelane_veto_rms_band_max_m=args.widelane_veto_rms_band_max_m,
        widelane_veto_min_kept_pairs=args.widelane_veto_min_kept_pairs,
        widelane_anchor_blend_alpha=args.widelane_anchor_blend_alpha,
        height_hold_alpha=args.height_hold_alpha,
        height_hold_release_on_last_velocity=args.height_hold_release_on_last_velocity,
        height_hold_release_on_dd_shift=args.height_hold_release_on_dd_shift,
        height_hold_release_min_dd_shift_m=args.height_hold_release_min_dd_shift_m,
        height_hold_reference_max_dd_rms_m=args.height_hold_reference_max_dd_rms_m,
        rsp_correction=args.rsp_correction,
        rsp_n_particles=args.rsp_n_particles,
        rsp_spread_m=args.rsp_spread_m,
        rsp_sigma_m=args.rsp_sigma_m,
        rsp_huber_k_m=args.rsp_huber_k_m,
        rsp_stein_steps=args.rsp_stein_steps,
        rsp_stein_step_size=args.rsp_stein_step_size,
        rsp_repulsion_scale=args.rsp_repulsion_scale,
        rsp_min_dd_shift_m=args.rsp_min_dd_shift_m,
        rsp_max_dd_shift_m=args.rsp_max_dd_shift_m,
        rsp_min_dd_rms_m=args.rsp_min_dd_rms_m,
        rsp_max_dd_rms_m=args.rsp_max_dd_rms_m,
        rsp_random_seed=args.rsp_random_seed,
        last_velocity_max_age_s=args.last_velocity_max_age_s,
    )

    summary_path = RESULTS_DIR / f"{args.results_prefix}_summary.csv"
    epoch_path = RESULTS_DIR / f"{args.results_prefix}_epochs.csv"
    _write_rows(summary_rows, summary_path)
    _write_rows(per_epoch_rows, epoch_path)

    print()
    for row in summary_rows:
        print(
            f"  {row['method']:<24} "
            f"ppc={row['ppc_score_pct']:.2f}% "
            f"rms={row['rms_2d']:.2f}m p95={row['p95']:.2f}m"
        )
    print(f"  Saved: {summary_path}")
    print(f"  Saved: {epoch_path}")


if __name__ == "__main__":
    main()
