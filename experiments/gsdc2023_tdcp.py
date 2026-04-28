"""TDCP construction helpers for GSDC2023 raw bridge experiments."""

from __future__ import annotations

import numpy as np

from experiments.gsdc2023_residual_model import geometric_range_with_sagnac


DEFAULT_TDCP_SIGMA_M = 0.03
DEFAULT_TDCP_CONSISTENCY_THRESHOLD_M = 1.5
TDCP_WEIGHT_SCALE_IDENTITY = 1.0
DEFAULT_TDCP_WEIGHT_SCALE = 3.0e-7
DEFAULT_TDCP_GEOMETRY_CORRECTION = True
TDCP_LOFFSET_M = 1.117
TDCP_DISABLE_PHONES = {"sm-a325f", "samsunga32"}
TDCP_LOFFSET_PHONES = {"sm-a205u", "sm-a217m", "sm-a505g", "sm-a600t", "sm-a505u"}
TDCP_XXDD_PHONES = TDCP_LOFFSET_PHONES | {"samsunga325g"}

ADR_STATE_VALID = 1
ADR_STATE_RESET = 2
ADR_STATE_CYCLE_SLIP = 4


def tdcp_enabled_for_phone(phone: str, requested: bool) -> bool:
    if not requested:
        return False
    return phone.lower() not in TDCP_DISABLE_PHONES


def tdcp_loffset_m(phone: str) -> float:
    return TDCP_LOFFSET_M if phone.lower() in TDCP_LOFFSET_PHONES else 0.0


def tdcp_use_drift_for_phone(phone: str) -> bool:
    return phone.lower() in TDCP_XXDD_PHONES


def valid_adr_state(state: int) -> bool:
    return bool(state & ADR_STATE_VALID) and not bool(state & (ADR_STATE_RESET | ADR_STATE_CYCLE_SLIP))


def build_tdcp_arrays(
    adr: np.ndarray,
    adr_state: np.ndarray,
    adr_uncertainty: np.ndarray,
    doppler: np.ndarray | None,
    dt: np.ndarray,
    *,
    consistency_threshold_m: float,
    doppler_weights: np.ndarray | None = None,
    clock_jump: np.ndarray | None = None,
    loffset_m: float = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    n_epoch, n_sat = adr.shape
    if n_epoch < 2:
        return None, None, 0

    tdcp_meas = np.zeros((n_epoch - 1, n_sat), dtype=np.float64)
    tdcp_weights = np.zeros((n_epoch - 1, n_sat), dtype=np.float64)
    consistency_mask_count = 0
    valid_phase = np.zeros((n_epoch, n_sat), dtype=bool)
    rejected_pair = np.zeros((n_epoch - 1, n_sat), dtype=bool)
    positive_dt = np.asarray(dt[:-1], dtype=np.float64)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0.0)]
    matlab_dt_s = float(np.round(np.median(positive_dt), 2)) if positive_dt.size else 0.0

    for t in range(n_epoch):
        for s in range(n_sat):
            valid_phase[t, s] = np.isfinite(adr[t, s]) and valid_adr_state(int(adr_state[t, s]))

    for t in range(n_epoch - 1):
        dt_s = float(dt[t])
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            continue
        if clock_jump is not None and t + 1 < clock_jump.size and bool(clock_jump[t + 1]):
            continue
        for s in range(n_sat):
            a0 = adr[t, s]
            a1 = adr[t + 1, s]
            if not valid_phase[t, s] or not valid_phase[t + 1, s]:
                continue
            meas = float(a1 - a0 + loffset_m)
            if doppler is not None:
                d0 = float(doppler[t, s])
                d1 = float(doppler[t + 1, s])
                doppler_pair_valid = np.isfinite(d0) and np.isfinite(d1)
                if doppler_weights is not None:
                    doppler_pair_valid = (
                        doppler_pair_valid
                        and float(doppler_weights[t, s]) > 0.0
                        and float(doppler_weights[t + 1, s]) > 0.0
                    )
                if doppler_pair_valid and matlab_dt_s > 0.0:
                    doppler_tdcp = -0.5 * (d0 + d1) * matlab_dt_s
                    if abs(meas - doppler_tdcp) > consistency_threshold_m:
                        rejected_pair[t, s] = True
                        consistency_mask_count += 1
                        continue

    # MATLAB exobs_residuals masks carrier-phase observations at both endpoints
    # of a failed dDL pair, which also removes adjacent TDCP factors.
    if np.any(rejected_pair):
        valid_phase[:-1, :] &= ~rejected_pair
        valid_phase[1:, :] &= ~rejected_pair

    for t in range(n_epoch - 1):
        dt_s = float(dt[t])
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            continue
        if clock_jump is not None and t + 1 < clock_jump.size and bool(clock_jump[t + 1]):
            continue
        for s in range(n_sat):
            if not valid_phase[t, s] or not valid_phase[t + 1, s]:
                continue
            meas = float(adr[t + 1, s] - adr[t, s] + loffset_m)
            sigma = DEFAULT_TDCP_SIGMA_M
            u0 = float(adr_uncertainty[t, s])
            u1 = float(adr_uncertainty[t + 1, s])
            if np.isfinite(u0) and np.isfinite(u1) and u0 > 0.0 and u1 > 0.0:
                sigma = float(np.sqrt(u0 * u0 + u1 * u1))
            sigma = max(sigma, 1e-3)
            tdcp_meas[t, s] = meas
            tdcp_weights[t, s] = 1.0 / (sigma * sigma)

    if not np.any(tdcp_weights > 0.0):
        return None, None, consistency_mask_count
    return tdcp_meas, tdcp_weights, consistency_mask_count


def apply_tdcp_weight_scale(tdcp_weights: np.ndarray | None, scale: float) -> None:
    if tdcp_weights is None:
        return
    scale = float(scale)
    if not np.isfinite(scale):
        raise ValueError(f"tdcp_weight_scale must be finite, got {scale!r}")
    if scale <= 0.0:
        tdcp_weights[:, :] = 0.0
        return
    if scale != TDCP_WEIGHT_SCALE_IDENTITY:
        tdcp_weights *= scale


def apply_tdcp_geometry_correction(
    tdcp_meas: np.ndarray | None,
    tdcp_weights: np.ndarray | None,
    sat_ecef: np.ndarray,
    reference_xyz: np.ndarray,
) -> int:
    if tdcp_meas is None or tdcp_weights is None:
        return 0
    n_pair = min(tdcp_meas.shape[0], sat_ecef.shape[0] - 1, reference_xyz.shape[0] - 1)
    n_sat = min(tdcp_meas.shape[1], sat_ecef.shape[1])
    corrected = 0
    for t in range(n_pair):
        ref0 = reference_xyz[t]
        ref1 = reference_xyz[t + 1]
        if not np.isfinite(ref0).all() or not np.isfinite(ref1).all():
            continue
        for s in range(n_sat):
            if float(tdcp_weights[t, s]) <= 0.0 or not np.isfinite(tdcp_meas[t, s]):
                continue
            sat0 = sat_ecef[t, s]
            sat1 = sat_ecef[t + 1, s]
            if not np.isfinite(sat0).all() or not np.isfinite(sat1).all():
                continue
            rho0 = float(geometric_range_with_sagnac(sat0, ref0))
            rho1 = float(geometric_range_with_sagnac(sat1, ref1))
            if not np.isfinite(rho0) or not np.isfinite(rho1):
                continue
            tdcp_meas[t, s] -= rho1 - rho0
            corrected += 1
    return corrected
