"""Sparse DD-carrier candidate helpers for the GSDC2023 bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.gsdc2023_base_correction import (
    GPS_WEEK_SECONDS,
    base_setting,
    course_base_obs_path,
    read_base_station_xyz,
    trip_course_phone,
    unix_ms_to_gps_abs_seconds,
)
from experiments.gsdc2023_tdcp import valid_adr_state
from gnss_gpu.dd_carrier import (
    BEIDOU_B1I_WAVELENGTH,
    BEIDOU_B2A_WAVELENGTH,
    DDCarrierComputer,
    DDResult,
    GALILEO_E1_WAVELENGTH,
    GALILEO_E5A_WAVELENGTH,
    GPS_L1_WAVELENGTH,
    GPS_L5_WAVELENGTH,
    QZSS_L1_WAVELENGTH,
    QZSS_L5_WAVELENGTH,
)
from gnss_gpu.spp import _elevation_azimuth


DD_CARRIER_FGO_SOURCE = "fgo_dd_carrier"

ANDROID_TO_DD_SYSTEM_ID = {
    1: 0,  # GPS
    3: 1,  # GLONASS
    6: 2,  # Galileo
    5: 3,  # BeiDou
    4: 4,  # QZSS
}


@dataclass(frozen=True)
class DDCarrierAnchorConfig:
    min_dd_pairs: int = 4
    sigma_cycles: float = 0.12
    prior_sigma_m: float = 1.5
    max_shift_m: float = 3.0
    max_initial_rms_m: float = 0.40
    max_final_rms_m: float = 0.25
    max_iter: int = 5
    tol_m: float = 1.0e-4


@dataclass(frozen=True)
class DDCarrierBridgeConfig:
    tow_snap_tolerance_s: float = 0.6
    anchor: DDCarrierAnchorConfig = DDCarrierAnchorConfig()
    base_obs_template: str | None = None
    require_base_obs_template: bool = False
    smooth_corrections: bool = False
    anchor_correction_sigma_m: float = 0.5
    correction_smooth_sigma_m: float = 0.25
    correction_zero_sigma_m: float = 5.0


def carrier_wavelength_m(constellation: int, signal_type: str) -> float:
    signal = str(signal_type).upper()
    is_l5 = "L5" in signal or "E5" in signal
    if int(constellation) == 1:
        return GPS_L5_WAVELENGTH if is_l5 else GPS_L1_WAVELENGTH
    if int(constellation) == 6:
        return GALILEO_E5A_WAVELENGTH if is_l5 else GALILEO_E1_WAVELENGTH
    if int(constellation) == 4:
        return QZSS_L5_WAVELENGTH if is_l5 else QZSS_L1_WAVELENGTH
    if int(constellation) == 5:
        return BEIDOU_B2A_WAVELENGTH if is_l5 else BEIDOU_B1I_WAVELENGTH
    return GPS_L1_WAVELENGTH


def rover_measurements_for_epoch(batch, epoch_idx: int) -> list[SimpleNamespace]:
    if batch.adr is None or batch.adr_state is None:
        return []
    rx = np.asarray(batch.kaggle_wls[epoch_idx, :3], dtype=np.float64)
    out: list[SimpleNamespace] = []
    for slot_idx, (constellation, svid, signal_type) in enumerate(batch.slot_keys):
        adr_m = float(batch.adr[epoch_idx, slot_idx])
        state = int(batch.adr_state[epoch_idx, slot_idx])
        sat = np.asarray(batch.sat_ecef[epoch_idx, slot_idx], dtype=np.float64)
        if not (np.isfinite(adr_m) and adr_m != 0.0 and valid_adr_state(state)):
            continue
        if sat.shape != (3,) or not np.isfinite(sat).all() or np.linalg.norm(sat) < 1.0e6:
            continue
        wavelength = carrier_wavelength_m(int(constellation), str(signal_type))
        if not np.isfinite(wavelength) or wavelength <= 0.0:
            continue
        elevation = 0.3
        if np.isfinite(rx).all() and np.linalg.norm(rx) > 1.0e6:
            elevation, _ = _elevation_azimuth(rx, sat)
        out.append(
            SimpleNamespace(
                system_id=ANDROID_TO_DD_SYSTEM_ID.get(int(constellation), 0),
                prn=int(svid),
                satellite_ecef=sat,
                carrier_phase=adr_m / wavelength,
                elevation=float(elevation),
                snr=30.0,
            )
        )
    return out


def course_base_obs_path_for_template(
    data_root: Path,
    split: str,
    course: str,
    base_name: str,
    rinex_type: str | None,
    *,
    base_obs_template: str | None,
    require_template: bool = False,
) -> Path:
    if not base_obs_template:
        return course_base_obs_path(data_root, split, course, base_name, rinex_type)
    suffix = "rnx3" if rinex_type == "V3" else "rnx2" if rinex_type == "V2" else "rnx3"
    rendered = base_obs_template.format(base=base_name, rinex_type=rinex_type or "", suffix=suffix)
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = data_root / split / course / candidate
    if candidate.is_file() or require_template:
        return candidate
    return course_base_obs_path(data_root, split, course, base_name, rinex_type)


def computer_for_trip(
    data_root: Path,
    trip: str,
    cache: dict[Path, DDCarrierComputer],
    *,
    base_obs_template: str | None = None,
    require_base_obs_template: bool = False,
) -> DDCarrierComputer:
    split, course, phone = trip_course_phone(trip)
    if split is None or course is None or phone is None:
        raise ValueError(f"trip must be split/course/phone: {trip}")
    base_name, rinex_type = base_setting(data_root, split, course, phone)
    obs_path = course_base_obs_path_for_template(
        data_root,
        split,
        course,
        base_name,
        rinex_type,
        base_obs_template=base_obs_template,
        require_template=require_base_obs_template,
    )
    key = obs_path.resolve()
    if key not in cache:
        base_xyz = read_base_station_xyz(data_root, course, base_name, apply_offset=False)
        cache[key] = DDCarrierComputer(
            obs_path,
            base_position=base_xyz,
            allowed_systems=("G", "E", "J", "C"),
            interpolate_base_epochs=True,
        )
    return cache[key]


def snap_tow_to_base_epoch(computer: DDCarrierComputer, tow: float, tolerance_s: float) -> float | None:
    base_tows = np.asarray(getattr(computer, "_base_tow_keys", np.array([], dtype=np.float64)), dtype=np.float64)
    if base_tows.size == 0:
        return None
    idx = int(np.searchsorted(base_tows, float(tow)))
    candidates = []
    if idx < base_tows.size:
        candidates.append(float(base_tows[idx]))
    if idx > 0:
        candidates.append(float(base_tows[idx - 1]))
    if not candidates:
        return None
    nearest = min(candidates, key=lambda value: abs(value - float(tow)))
    if abs(nearest - float(tow)) <= float(tolerance_s):
        return nearest
    return None


def _dd_expected_m(position_ecef: np.ndarray, dd: DDResult) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(position_ecef, dtype=np.float64).reshape(3)
    sat_k = np.asarray(dd.sat_ecef_k, dtype=np.float64).reshape(-1, 3)
    sat_ref = np.asarray(dd.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)
    rho_k_vec = pos[None, :] - sat_k
    rho_ref_vec = pos[None, :] - sat_ref
    rho_k = np.linalg.norm(rho_k_vec, axis=1)
    rho_ref = np.linalg.norm(rho_ref_vec, axis=1)
    expected = rho_k - rho_ref - np.asarray(dd.base_range_k) + np.asarray(dd.base_range_ref)
    jac = rho_k_vec / np.maximum(rho_k[:, None], 1.0) - rho_ref_vec / np.maximum(rho_ref[:, None], 1.0)
    return expected, jac


def dd_carrier_fixed_ambiguity_update(
    seed_ecef: np.ndarray,
    dd: DDResult,
    config: DDCarrierAnchorConfig,
) -> tuple[np.ndarray, dict[str, float | bool | int | str]]:
    seed = np.asarray(seed_ecef, dtype=np.float64).reshape(3)
    if int(dd.n_dd) < int(config.min_dd_pairs):
        return seed, {"accepted": False, "reason": "few_pairs", "dd_pairs": int(dd.n_dd)}

    wavelengths = np.asarray(dd.wavelengths_m, dtype=np.float64).reshape(-1)
    expected_seed, _ = _dd_expected_m(seed, dd)
    expected_seed_cycles = expected_seed / wavelengths
    ambiguities = np.round(np.asarray(dd.dd_carrier_cycles, dtype=np.float64) - expected_seed_cycles)
    obs_m = (np.asarray(dd.dd_carrier_cycles, dtype=np.float64) - ambiguities) * wavelengths
    pos = seed.copy()

    def residual_at(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        expected, jac = _dd_expected_m(x, dd)
        return obs_m - expected, jac

    residual, _ = residual_at(pos)
    initial_rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else float("inf")
    if not np.isfinite(initial_rms) or initial_rms > float(config.max_initial_rms_m):
        return seed, {
            "accepted": False,
            "reason": "initial_rms",
            "dd_pairs": int(dd.n_dd),
            "initial_rms_m": initial_rms,
        }

    sigma_m = np.maximum(wavelengths * float(config.sigma_cycles), 1.0e-3)
    sqrt_w = np.sqrt(np.asarray(dd.dd_weights, dtype=np.float64).reshape(-1)) / sigma_m
    prior_sqrt_w = 1.0 / max(float(config.prior_sigma_m), 1.0e-6)
    iters = 0
    for iters in range(1, int(config.max_iter) + 1):
        residual, jac = residual_at(pos)
        a = jac * sqrt_w[:, None]
        b = residual * sqrt_w
        a = np.vstack([a, np.eye(3) * prior_sqrt_w])
        b = np.concatenate([b, (seed - pos) * prior_sqrt_w])
        try:
            delta, *_ = np.linalg.lstsq(a, b, rcond=None)
        except np.linalg.LinAlgError:
            return seed, {"accepted": False, "reason": "singular", "dd_pairs": int(dd.n_dd)}
        if not np.isfinite(delta).all():
            return seed, {"accepted": False, "reason": "nonfinite", "dd_pairs": int(dd.n_dd)}
        pos = pos + delta
        if float(np.linalg.norm(delta)) < float(config.tol_m):
            break

    final_residual, _ = residual_at(pos)
    final_rms = float(np.sqrt(np.mean(final_residual * final_residual))) if final_residual.size else float("inf")
    shift_m = float(np.linalg.norm(pos - seed))
    accepted = (
        np.isfinite(final_rms)
        and final_rms <= float(config.max_final_rms_m)
        and shift_m <= float(config.max_shift_m)
        and final_rms <= initial_rms
    )
    return (
        pos if accepted else seed,
        {
            "accepted": bool(accepted),
            "reason": "accepted" if accepted else "gate",
            "dd_pairs": int(dd.n_dd),
            "initial_rms_m": initial_rms,
            "final_rms_m": final_rms,
            "shift_m": shift_m,
            "iters": int(iters),
        },
    )


def smooth_anchor_corrections(
    n_epoch: int,
    anchor_indices: np.ndarray,
    anchor_deltas: np.ndarray,
    *,
    anchor_sigma_m: float,
    smooth_sigma_m: float,
    zero_sigma_m: float,
) -> np.ndarray:
    n = int(n_epoch)
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    idx = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
    deltas = np.asarray(anchor_deltas, dtype=np.float64).reshape(-1, 3)
    valid = (idx >= 0) & (idx < n) & np.isfinite(deltas).all(axis=1)
    idx = idx[valid]
    deltas = deltas[valid]
    if idx.size == 0:
        return np.zeros((n, 3), dtype=np.float64)

    h = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros((n, 3), dtype=np.float64)
    if zero_sigma_m > 0.0:
        h += np.eye(n, dtype=np.float64) / (float(zero_sigma_m) ** 2)
    anchor_w = 1.0 / (max(float(anchor_sigma_m), 1.0e-6) ** 2)
    for epoch_idx, delta in zip(idx, deltas):
        h[int(epoch_idx), int(epoch_idx)] += anchor_w
        rhs[int(epoch_idx)] += anchor_w * delta
    if n > 1 and smooth_sigma_m > 0.0:
        smooth_w = 1.0 / (float(smooth_sigma_m) ** 2)
        for epoch_idx in range(n - 1):
            h[epoch_idx, epoch_idx] += smooth_w
            h[epoch_idx + 1, epoch_idx + 1] += smooth_w
            h[epoch_idx, epoch_idx + 1] -= smooth_w
            h[epoch_idx + 1, epoch_idx] -= smooth_w
    try:
        return np.linalg.solve(h, rhs)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(h, rhs, rcond=None)
        return solution


def apply_sparse_dd_carrier_anchors(
    data_root: Path,
    trip: str,
    batch,
    seed_state: np.ndarray,
    config: DDCarrierBridgeConfig,
    cache: dict[Path, DDCarrierComputer] | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    cache = cache if cache is not None else {}
    computer = computer_for_trip(
        data_root,
        trip,
        cache,
        base_obs_template=config.base_obs_template,
        require_base_obs_template=config.require_base_obs_template,
    )
    tows = np.mod(unix_ms_to_gps_abs_seconds(batch.times_ms), GPS_WEEK_SECONDS)
    anchored = np.asarray(seed_state, dtype=np.float64).copy()
    accepted = 0
    dd_epochs = 0
    snapped = 0
    pair_counts: list[int] = []
    shifts: list[float] = []
    initial_rms: list[float] = []
    final_rms: list[float] = []
    anchor_indices: list[int] = []
    anchor_deltas: list[np.ndarray] = []
    for epoch_idx, tow in enumerate(tows):
        tow_for_dd = snap_tow_to_base_epoch(computer, float(tow), config.tow_snap_tolerance_s)
        if tow_for_dd is None:
            continue
        snapped += 1
        measurements = rover_measurements_for_epoch(batch, epoch_idx)
        dd = computer.compute_dd(
            tow_for_dd,
            measurements,
            rover_position_approx=anchored[epoch_idx, :3],
            min_common_sats=config.anchor.min_dd_pairs,
        )
        if dd is None:
            continue
        dd_epochs += 1
        pair_counts.append(int(dd.n_dd))
        pos, stats = dd_carrier_fixed_ambiguity_update(anchored[epoch_idx, :3], dd, config.anchor)
        if bool(stats.get("accepted", False)):
            accepted += 1
            delta = pos - anchored[epoch_idx, :3]
            anchor_indices.append(int(epoch_idx))
            anchor_deltas.append(delta)
            if not config.smooth_corrections:
                anchored[epoch_idx, :3] = pos
            shifts.append(float(stats.get("shift_m", np.nan)))
            initial_rms.append(float(stats.get("initial_rms_m", np.nan)))
            final_rms.append(float(stats.get("final_rms_m", np.nan)))

    correction_mean_norm_m = 0.0
    correction_p95_norm_m = 0.0
    if config.smooth_corrections and anchor_indices:
        corrections = smooth_anchor_corrections(
            anchored.shape[0],
            np.asarray(anchor_indices, dtype=np.int64),
            np.asarray(anchor_deltas, dtype=np.float64),
            anchor_sigma_m=config.anchor_correction_sigma_m,
            smooth_sigma_m=config.correction_smooth_sigma_m,
            zero_sigma_m=config.correction_zero_sigma_m,
        )
        correction_norm = np.linalg.norm(corrections, axis=1)
        correction_mean_norm_m = float(np.mean(correction_norm))
        correction_p95_norm_m = float(np.percentile(correction_norm, 95))
        anchored[:, :3] = anchored[:, :3] + corrections

    return anchored, {
        "base_snapped_epochs": int(snapped),
        "dd_epochs": int(dd_epochs),
        "accepted_anchor_epochs": int(accepted),
        "smooth_corrections": bool(config.smooth_corrections),
        "dd_pairs_mean": float(np.mean(pair_counts)) if pair_counts else 0.0,
        "accepted_shift_mean_m": float(np.nanmean(shifts)) if shifts else 0.0,
        "accepted_initial_rms_mean_m": float(np.nanmean(initial_rms)) if initial_rms else 0.0,
        "accepted_final_rms_mean_m": float(np.nanmean(final_rms)) if final_rms else 0.0,
        "correction_mean_norm_m": correction_mean_norm_m,
        "correction_p95_norm_m": correction_p95_norm_m,
    }


__all__ = [
    "DD_CARRIER_FGO_SOURCE",
    "DDCarrierAnchorConfig",
    "DDCarrierBridgeConfig",
    "apply_sparse_dd_carrier_anchors",
    "carrier_wavelength_m",
    "computer_for_trip",
    "course_base_obs_path_for_template",
    "dd_carrier_fixed_ambiguity_update",
    "rover_measurements_for_epoch",
    "smooth_anchor_corrections",
    "snap_tow_to_base_epoch",
]
