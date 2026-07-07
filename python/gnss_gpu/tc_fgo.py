"""Tightly-coupled GNSS+IMU factor-graph optimization (WP11/12a float, WP12b AR).

Per-epoch sliding-window states are antenna position and velocity in a
base-anchored local ENU frame. WP12a optionally promotes accel/gyro biases
into the LM state (``optimize_imu_biases``) with random-walk between factors.
WP12b adds DD carrier-phase factors, per-epoch ambiguity states with
N-continuity, LAMBDA/subset-AR validation, and fix-and-hold folding.

Attitude (quaternion) is still propagated outside the window via ``INSEKF``.

GNSS factors reuse double-differenced pseudoranges from ``local_fgo`` (DD
against ``base.obs``). WP12a adds optional RTK baseline position anchors,
GNSS-quality-scaled IMU trust, quality-scaled marginalization, and DDPR
recovery helpers for the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Sequence

import numpy as np

from gnss_gpu.ins_ekf import (
    INSConfig,
    INSEKF,
    _ecef_to_enu_rotation,
    _quat_from_axis_angle,
    _quat_multiply,
    _quat_normalize,
    _quat_to_rotmat,
    _skew,
)
from gnss_gpu.local_fgo import (
    DDCarrierEpoch,
    DDPseudorangeEpoch,
    LambdaFixConfig,
    LocalFgoConfig,
    LocalFgoWindow,
    _apply_lambda_fixes_to_dd,
    _dd_expected_and_jacobian_m,
    _dd_pair_key,
    _ddpr_cross_check,
    _estimate_lambda_fixes,
    _huber_sqrt_weight,
    _valid_dd_row,
    _weighted_sigma,
    _weights,
    _wrap_cycles,
)
from gnss_gpu.lambda_ambiguity import solve_lambda

_G_ENU = np.array([0.0, 0.0, -9.81], dtype=np.float64)
_NAV_DIM = 6  # pos(3) + vel(3)
_BIAS_DIM = 6  # b_a(3) + b_g(3)
_STATE_DIM = _NAV_DIM  # legacy alias when biases are not optimized
DEFAULT_LEVER_ARM_BODY_M = np.array([0.31, 0.0, 0.55], dtype=np.float64)


@dataclass(frozen=True)
class ImuPreintSegment:
    """Collapsed IMU preintegration between two GNSS epochs."""

    delta_p_body: np.ndarray
    delta_v_body: np.ndarray
    delta_t_s: float
    delta_angle_body: np.ndarray
    dp_d_ba: np.ndarray
    dv_d_ba: np.ndarray
    dp_d_bg: np.ndarray
    dv_d_bg: np.ndarray


@dataclass
class TcFgoNavState:
    """Nominal navigation state at one GNSS epoch (antenna / ENU)."""

    p_enu: np.ndarray
    v_enu: np.ndarray
    q_body_to_enu: np.ndarray
    b_a: np.ndarray
    b_g: np.ndarray

    def copy(self) -> "TcFgoNavState":
        return TcFgoNavState(
            p_enu=np.asarray(self.p_enu, dtype=np.float64).copy(),
            v_enu=np.asarray(self.v_enu, dtype=np.float64).copy(),
            q_body_to_enu=np.asarray(self.q_body_to_enu, dtype=np.float64).copy(),
            b_a=np.asarray(self.b_a, dtype=np.float64).copy(),
            b_g=np.asarray(self.b_g, dtype=np.float64).copy(),
        )


@dataclass
class TcFgoEpochObs:
    """Per-epoch observations and motion flags."""

    dd_pseudorange: DDPseudorangeEpoch | None = None
    dd_carrier: DDCarrierEpoch | None = None
    enable_nhc: bool = False
    enable_zupt: bool = False
    anchor_pos_enu: np.ndarray | None = None
    anchor_sigma_m: float | None = None
    doppler_vel_enu: np.ndarray | None = None
    doppler_sigma_mps: float | None = None


@dataclass
class TcFgoConfig:
    """Noise and optimizer settings for the TC-FGO sliding window."""

    window_epochs: int = 5
    prior_pos_sigma_m: float = 0.5
    prior_vel_sigma_mps: float = 0.5
    marginal_pos_sigma_m: float = 0.2
    marginal_vel_sigma_mps: float = 0.3
    imu_pos_sigma_m: float = 0.15
    imu_vel_sigma_mps: float = 0.08
    dd_pr_sigma_m: float = 5.0
    pr_huber_k: float = 1.5
    pr_huber_disable_raw_rms_m: float = 15.0
    min_weight: float = 1e-3
    nhc_sigma_mps: float = 0.05
    zupt_sigma_mps: float = 0.02
    max_iterations: int = 25
    relative_error_tol: float = 1e-5
    lever_arm_body_m: np.ndarray = field(
        default_factory=lambda: DEFAULT_LEVER_ARM_BODY_M.copy()
    )
    # WP12a stabilization knobs (all off by default for WP11 bit-compat).
    optimize_imu_biases: bool = False
    bias_rw_sigma_accel: float = 0.02
    bias_rw_sigma_gyro_radps: float = 0.002
    bias_prior_sigma_accel: float = 0.15
    bias_prior_sigma_gyro_radps: float = 0.02
    enable_imu_gnss_quality_scale: bool = False
    imu_quality_rms_ref_m: float = 5.0
    marginal_quality_rms_ref_m: float = 3.0
    marginal_quality_min_dd: float = 4.0
    doppler_body_vel_sigma_mps: float = 0.0
    # WP12b carrier + AR (all off by default).
    enable_dd_carrier: bool = False
    dd_cp_sigma_cycles: float = 0.20
    dd_cp_fixed_sigma_cycles: float = 0.05
    dd_cp_huber_k: float = 1.5
    n_continuity_sigma_held_cyc: float = 0.01
    n_continuity_sigma_float_cyc: float = 0.1
    enable_lambda_ar: bool = False
    lambda_ratio_threshold: float = 3.0
    lambda_min_epochs: int = 3
    lambda_max_group_size: int = 8
    subset_ar_max_drop: int = 2
    ddpr_reject_threshold: float = 0.05
    post_ar_ddpr_degrade_threshold: float = 0.10
    # WP12d: quality-gated AR (offer LAMBDA only when float certifies sub-meter).
    enable_ar_quality_gate: bool = True
    ar_cert_max_pos_sigma_m: float = 0.5
    ar_cert_max_dd_pr_rms_m: float = 2.0
    ar_cert_max_dd_cp_rms_cyc: float = 1.0
    ar_cert_min_epochs_since_recovery: int = 10
    ar_cert_min_dd_carrier: int = 4
    ar_cert_max_epochs_since_anchor: int = 0  # 0 = disabled; WP12e anchor-proximity gate
    enable_ar_subset: bool = True
    enable_ar_ddpr_crossval: bool = True
    enable_ar_post_ar_gate: bool = True
    enable_ar_hold: bool = True
    # WP12b: carry float ambiguities across sliding windows (inuex35 BetweenFactorDouble).
    enable_persistent_ambiguities: bool = False
    persistent_ambiguity_sigma_cyc: float = 0.1
    # WP12c: Schur-complement marginalization (replaces diagonal 0.2 m hack).
    enable_schur_marginalization: bool = False
    schur_min_eigenvalue: float = 1.0e-6
    schur_max_eigenvalue: float = 0.0  # 0 = derive cap from in-window Hessian
    schur_info_cap_ratio: float = 1.0
    schur_front_nav_only: bool = True
    schur_use_bank_ambiguity_priors: bool = False


def state_dim(config: TcFgoConfig | None = None) -> int:
    """Return LM state dimension per epoch (6 or 12)."""

    config = TcFgoConfig() if config is None else config
    return _NAV_DIM + _BIAS_DIM if bool(config.optimize_imu_biases) else _NAV_DIM


@dataclass
class TcFgoWindowProblem:
    """One sliding-window solve."""

    initial_states: Sequence[TcFgoNavState]
    imu_segments: Sequence[ImuPreintSegment | None]
    observations: Sequence[TcFgoEpochObs]
    origin_ecef: np.ndarray
    origin_lat: float
    origin_lon: float
    marginal_prior: TcFgoNavState | None = None
    marginal_prior_sigmas: np.ndarray | None = None
    schur_marginal: "TcSchurMarginal | None" = None
    last_dd_pr_rms_m: float = float("inf")
    held_ambiguities: dict[tuple[int, tuple[str, str, str, str]], int] | None = None
    window_start_epoch: int = 0
    epochs_since_recovery: int = 10**9
    epochs_since_anchor: int = 10**9


@dataclass
class TcFgoWindowResult:
    states: list[TcFgoNavState]
    initial_error: float
    final_error: float
    factor_counts: dict[str, int]
    n_iterations: int = 0
    converged: bool = False
    ambiguity_values: np.ndarray | None = None
    ar_accepted: bool = False
    ar_info: dict[str, Any] | None = None
    epoch_fixed: list[bool] | None = None
    accepted_fixes: dict[tuple[int, tuple[str, str, str, str]], int] | None = None
    schur_marginal: "TcSchurMarginal | None" = None


def ecef_to_enu(
    ecef: np.ndarray,
    origin_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> np.ndarray:
    origin = np.asarray(origin_ecef, dtype=np.float64).reshape(3)
    R = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon))
    return R @ (np.asarray(ecef, dtype=np.float64).reshape(3) - origin)


def enu_to_ecef(
    enu: np.ndarray,
    origin_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> np.ndarray:
    origin = np.asarray(origin_ecef, dtype=np.float64).reshape(3)
    R = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon))
    return origin + R.T @ np.asarray(enu, dtype=np.float64).reshape(3)


def lever_arm_offset_enu(q_body_to_enu: np.ndarray, lever_arm_body_m: np.ndarray) -> np.ndarray:
    """Antenna offset from IMU/body frame origin in ENU."""
    R = _quat_to_rotmat(q_body_to_enu)
    return R @ np.asarray(lever_arm_body_m, dtype=np.float64).reshape(3)


def collapse_imu_preintegration_segment(
    delta_p: np.ndarray | None,
    delta_v: np.ndarray | None,
    delta_angle: np.ndarray | None,
    delta_t: np.ndarray | None,
    dp_d_ba: np.ndarray | None,
    dv_d_ba: np.ndarray | None,
    dp_d_bg: np.ndarray | None,
    dv_d_bg: np.ndarray | None,
) -> ImuPreintSegment | None:
    """Sum valid sub-interval rows into one body-frame preintegration segment."""

    if delta_p is None or delta_v is None or delta_t is None:
        return None
    dp = np.asarray(delta_p, dtype=np.float64)
    dv = np.asarray(delta_v, dtype=np.float64)
    dt = np.asarray(delta_t, dtype=np.float64).ravel()
    if dp.ndim != 2 or dv.ndim != 2 or dp.shape != dv.shape or dp.shape[0] != dt.size:
        return None
    valid = np.isfinite(dt) & (dt > 0.0) & np.isfinite(dp).all(axis=1) & np.isfinite(dv).all(axis=1)
    if not valid.any():
        return None
    dp_sum = np.nansum(dp[valid], axis=0)
    dv_sum = np.nansum(dv[valid], axis=0)
    dt_sum = float(np.sum(dt[valid]))
    da = (
        np.nansum(np.asarray(delta_angle, dtype=np.float64)[valid], axis=0)
        if delta_angle is not None
        else np.zeros(3, dtype=np.float64)
    )

    def _sum_jac(jac: np.ndarray | None) -> np.ndarray:
        if jac is None:
            return np.zeros((3, 3), dtype=np.float64)
        arr = np.asarray(jac, dtype=np.float64)
        if arr.ndim != 3:
            return np.zeros((3, 3), dtype=np.float64)
        return np.nansum(arr[valid], axis=0)

    return ImuPreintSegment(
        delta_p_body=dp_sum,
        delta_v_body=dv_sum,
        delta_t_s=dt_sum,
        delta_angle_body=da,
        dp_d_ba=_sum_jac(dp_d_ba),
        dv_d_ba=_sum_jac(dv_d_ba),
        dp_d_bg=_sum_jac(dp_d_bg),
        dv_d_bg=_sum_jac(dv_d_bg),
    )


def bias_corrected_preintegration(
    segment: ImuPreintSegment,
    b_a: np.ndarray,
    b_g: np.ndarray,
    b_a_lin: np.ndarray,
    b_g_lin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """First-order bias correction around the linearization biases."""

    dba = np.asarray(b_a, dtype=np.float64).reshape(3) - np.asarray(b_a_lin, dtype=np.float64).reshape(3)
    dbg = np.asarray(b_g, dtype=np.float64).reshape(3) - np.asarray(b_g_lin, dtype=np.float64).reshape(3)
    dp = segment.delta_p_body + segment.dp_d_ba @ dba + segment.dp_d_bg @ dbg
    dv = segment.delta_v_body + segment.dv_d_ba @ dba + segment.dv_d_bg @ dbg
    return dp, dv


def imu_preintegration_residual(
    p_i: np.ndarray,
    v_i: np.ndarray,
    p_j: np.ndarray,
    v_j: np.ndarray,
    q_i: np.ndarray,
    q_j: np.ndarray,
    segment: ImuPreintSegment,
    *,
    b_a: np.ndarray,
    b_g: np.ndarray,
    b_a_lin: np.ndarray,
    b_g_lin: np.ndarray,
    lever_arm_body_m: np.ndarray,
    g_enu: np.ndarray = _G_ENU,
) -> tuple[np.ndarray, np.ndarray]:
    """Position (3) and velocity (3) IMU preintegration residuals in ENU."""

    dt = float(segment.delta_t_s)
    R_i = _quat_to_rotmat(q_i)
    R_j = _quat_to_rotmat(q_j)
    l = np.asarray(lever_arm_body_m, dtype=np.float64).reshape(3)
    dp, dv = bias_corrected_preintegration(segment, b_a, b_g, b_a_lin, b_g_lin)
    dp_enu = R_i @ dp
    dv_enu = R_i @ dv
    lever_i = R_i @ l
    lever_j = R_j @ l
    r_p = (
        np.asarray(p_j, dtype=np.float64).reshape(3)
        - np.asarray(p_i, dtype=np.float64).reshape(3)
        - np.asarray(v_i, dtype=np.float64).reshape(3) * dt
        - 0.5 * np.asarray(g_enu, dtype=np.float64).reshape(3) * dt * dt
        - dp_enu
        - (lever_j - lever_i)
    )
    r_v = (
        np.asarray(v_j, dtype=np.float64).reshape(3)
        - np.asarray(v_i, dtype=np.float64).reshape(3)
        - np.asarray(g_enu, dtype=np.float64).reshape(3) * dt
        - dv_enu
    )
    return r_p, r_v


def imu_preintegration_jacobian(
    p_i: np.ndarray,
    v_i: np.ndarray,
    p_j: np.ndarray,
    v_j: np.ndarray,
    q_i: np.ndarray,
    segment: ImuPreintSegment,
    *,
    lever_arm_body_m: np.ndarray,
    g_enu: np.ndarray = _G_ENU,
    include_bias_jacobians: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Analytic Jacobians w.r.t. nav states and optionally epoch-i biases."""

    dt = float(segment.delta_t_s)
    eye3 = np.eye(3, dtype=np.float64)
    jac_pi_p = -eye3
    jac_pi_v = -eye3 * dt
    jac_pj_p = eye3
    jac_vi_v = -eye3
    jac_vj_v = eye3
    jac_ba = np.zeros((3, 3), dtype=np.float64)
    jac_bg = np.zeros((3, 3), dtype=np.float64)
    jac_v_ba = np.zeros((3, 3), dtype=np.float64)
    jac_v_bg = np.zeros((3, 3), dtype=np.float64)
    if include_bias_jacobians:
        R_i = _quat_to_rotmat(q_i)
        jac_ba = -R_i @ segment.dp_d_ba
        jac_bg = -R_i @ segment.dp_d_bg
        jac_v_ba = -R_i @ segment.dv_d_ba
        jac_v_bg = -R_i @ segment.dv_d_bg
    _ = (p_i, v_i, p_j, segment, lever_arm_body_m, g_enu)
    return jac_pi_p, jac_pi_v, jac_pj_p, jac_vi_v, jac_vj_v, jac_ba, jac_bg, jac_v_ba, jac_v_bg


def nhc_residual_and_jacobian(
    v_enu: np.ndarray,
    q_body_to_enu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-holonomic constraint: lateral and vertical body velocity ~ 0."""

    R_bn = _quat_to_rotmat(q_body_to_enu)
    v_body = R_bn.T @ np.asarray(v_enu, dtype=np.float64).reshape(3)
    residual = np.array([v_body[1], v_body[2]], dtype=np.float64)
    jac = np.zeros((2, 3), dtype=np.float64)
    jac[:, :] = (R_bn.T[1:3, :]).copy()
    return residual, jac


def zupt_residual_and_jacobian(v_enu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(v_enu, dtype=np.float64).reshape(3)
    return residual, np.eye(3, dtype=np.float64)


def marginalization_prior_residual(
    state: np.ndarray,
    prior_state: np.ndarray,
) -> np.ndarray:
    n = min(
        int(np.asarray(state, dtype=np.float64).size),
        int(np.asarray(prior_state, dtype=np.float64).size),
    )
    return np.asarray(state, dtype=np.float64).reshape(-1)[:n] - np.asarray(
        prior_state, dtype=np.float64
    ).reshape(-1)[:n]


def state_vector_from_nav(nav: TcFgoNavState, config: TcFgoConfig | None = None) -> np.ndarray:
    config = TcFgoConfig() if config is None else config
    parts = [
        np.asarray(nav.p_enu, dtype=np.float64).reshape(3),
        np.asarray(nav.v_enu, dtype=np.float64).reshape(3),
    ]
    if config.optimize_imu_biases:
        parts.extend(
            [
                np.asarray(nav.b_a, dtype=np.float64).reshape(3),
                np.asarray(nav.b_g, dtype=np.float64).reshape(3),
            ]
        )
    return np.concatenate(parts)


def nav_from_state_vector(
    x: np.ndarray,
    template: TcFgoNavState,
    config: TcFgoConfig | None = None,
) -> TcFgoNavState:
    config = TcFgoConfig() if config is None else config
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    b_a = x[6:9].copy() if config.optimize_imu_biases and x.size >= 12 else template.b_a.copy()
    b_g = x[9:12].copy() if config.optimize_imu_biases and x.size >= 12 else template.b_g.copy()
    return TcFgoNavState(
        p_enu=x[0:3].copy(),
        v_enu=x[3:6].copy(),
        q_body_to_enu=template.q_body_to_enu.copy(),
        b_a=b_a,
        b_g=b_g,
    )


def position_anchor_residual_and_jacobian(
    pos_enu: np.ndarray,
    anchor_pos_enu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute position anchor residual (ENU) and Jacobian w.r.t. position."""

    residual = np.asarray(pos_enu, dtype=np.float64).reshape(3) - np.asarray(
        anchor_pos_enu, dtype=np.float64
    ).reshape(3)
    return residual, np.eye(3, dtype=np.float64)


def bias_random_walk_residual(
    b_j: np.ndarray,
    b_i: np.ndarray,
) -> np.ndarray:
    return np.asarray(b_j, dtype=np.float64).reshape(3) - np.asarray(b_i, dtype=np.float64).reshape(3)


def imu_gnss_quality_scale(
    dd_pr_rms_m: float,
    dt_s: float,
    config: TcFgoConfig,
) -> float:
    """Inflate IMU sigmas from last DDPR residual (inuex35 pattern)."""

    if not config.enable_imu_gnss_quality_scale:
        return 1.0
    ref = max(float(config.imu_quality_rms_ref_m), 1.0e-3)
    dt = max(float(dt_s), 0.2)
    rms = max(float(dd_pr_rms_m), ref)
    return max(1.0, (rms * rms) / (ref * dt))


def _dd_pr_raw_residuals_for_epoch(
    pos_ecef: np.ndarray,
    obs: DDPseudorangeEpoch,
) -> tuple[np.ndarray, int]:
    """Unweighted DD pseudorange residuals (meters) and valid row count."""

    dd = np.asarray(obs.dd_pseudorange_m, dtype=np.float64).ravel()
    sat_k = np.asarray(obs.sat_ecef_k, dtype=np.float64).reshape(-1, 3)
    sat_ref = np.asarray(obs.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)
    base_k = np.asarray(obs.base_range_k, dtype=np.float64).ravel()
    base_ref = np.asarray(obs.base_range_ref, dtype=np.float64).ravel()
    x = np.asarray(pos_ecef, dtype=np.float64).reshape(3)
    raw: list[float] = []
    for j in range(len(dd)):
        if not _valid_dd_row(dd[j], sat_k[j], sat_ref[j], base_k[j], base_ref[j], 1.0):
            continue
        expected_m, _ = _dd_expected_and_jacobian_m(x, sat_k[j], sat_ref[j], base_k[j], base_ref[j])
        raw.append(float(expected_m - float(dd[j])))
    if not raw:
        return np.zeros(0, dtype=np.float64), 0
    arr = np.asarray(raw, dtype=np.float64)
    return arr, int(arr.size)


def compute_dd_pr_postfit_rms(
    nav: TcFgoNavState,
    obs: DDPseudorangeEpoch,
    *,
    origin_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    config: TcFgoConfig,
    fgo_config: LocalFgoConfig,
    huber_weighted: bool = True,
) -> tuple[float, int]:
    """Post-fit DD pseudorange RMS and row count for one epoch."""

    pos_ecef = enu_to_ecef(nav.p_enu, origin_ecef, origin_lat, origin_lon)
    if not huber_weighted:
        r, n = _dd_pr_raw_residuals_for_epoch(pos_ecef, obs)
        if n == 0:
            return float("inf"), 0
        return float(np.sqrt(np.mean(r * r))), n
    dd_residuals, _, _ = _dd_pr_blocks_for_epoch(pos_ecef, obs, config, fgo_config)
    if not dd_residuals:
        return float("inf"), 0
    r = np.concatenate([np.asarray(blk, dtype=np.float64).ravel() for blk in dd_residuals])
    return float(np.sqrt(np.mean(r * r))), int(r.size)


def marginal_pos_sigma_from_schur(
    schur: "TcSchurMarginal | None",
    *,
    config: TcFgoConfig | None = None,
) -> float:
    """1σ position bound (m) from the Schur front nav position block."""

    if schur is None or int(schur.precision.shape[0]) < 3:
        return float("inf")
    config = TcFgoConfig() if config is None else config
    sdim = int(schur.sdim)
    pos_block = np.asarray(schur.precision, dtype=np.float64)[:3, :3]
    pos_block = clamp_information_eigenvalues(
        pos_block,
        float(config.schur_min_eigenvalue),
    )
    try:
        cov = np.linalg.inv(pos_block)
    except np.linalg.LinAlgError:
        return float("inf")
    if not np.isfinite(cov).all():
        return float("inf")
    diag = np.diag(cov).astype(np.float64)
    diag = np.maximum(diag, 0.0)
    # Horizontal position dominates AR eligibility; ignore vertical if biases present.
    if sdim >= 6:
        return float(math.sqrt(max(float(np.max(diag[0:2])), 0.0)))
    return float(math.sqrt(max(float(np.max(diag)), 0.0)))


def compute_dd_cp_postfit_rms(
    nav: TcFgoNavState,
    obs: DDCarrierEpoch,
    *,
    origin_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    config: TcFgoConfig,
    ambiguity_values: np.ndarray | None = None,
    layout: TcAmbiguityLayout | None = None,
    local_epoch: int = 0,
) -> tuple[float, int]:
    """Post-fit DD carrier-phase RMS (cycles) for one epoch."""

    pos_ecef = enu_to_ecef(nav.p_enu, origin_ecef, origin_lat, origin_lon)
    cp = obs
    n_rows = int(cp.n)
    residuals: list[float] = []
    for row in range(n_rows):
        held_int = None
        if layout is not None:
            held_int = layout.held_map.get((local_epoch, row))
        amb_val = None
        if layout is not None and held_int is None:
            amb_idx = layout.index_map.get((local_epoch, row))
            if amb_idx is not None and ambiguity_values is not None:
                amb_val = float(ambiguity_values[int(amb_idx)])
        residual, _, _ = ambiguity_carrier_residual_and_jacobian(
            pos_ecef,
            cp,
            row,
            ambiguity_value=amb_val,
            held_integer=held_int,
        )
        if np.isfinite(residual):
            residuals.append(float(residual))
    if not residuals:
        return float("inf"), 0
    arr = np.asarray(residuals, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr))), int(arr.size)


@dataclass(frozen=True)
class FloatQualityCertificate:
    """Per-window float health certificate for AR eligibility (WP12d)."""

    marginal_pos_sigma_m: float
    dd_pr_postfit_rms_m: float
    dd_cp_postfit_rms_cyc: float
    epochs_since_recovery: int
    epochs_since_anchor: int
    n_dd_carrier: int
    passed: bool
    fail_reasons: tuple[str, ...] = ()


def evaluate_float_quality_certificate(
    *,
    config: TcFgoConfig,
    schur_marginal: "TcSchurMarginal | None",
    last_nav: TcFgoNavState,
    last_dd_pr: DDPseudorangeEpoch | None,
    last_dd_cp: DDCarrierEpoch | None,
    origin_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    fgo_config: LocalFgoConfig,
    ambiguity_values: np.ndarray | None = None,
    layout: TcAmbiguityLayout | None = None,
    local_epoch: int = 0,
    epochs_since_recovery: int = 10**9,
    epochs_since_anchor: int = 10**9,
    n_dd_carrier_factors: int = 0,
) -> FloatQualityCertificate:
    """Decide whether the current float is safe to offer LAMBDA."""

    pos_sigma = marginal_pos_sigma_from_schur(schur_marginal, config=config)
    if last_dd_pr is not None:
        dd_pr_rms, _ = compute_dd_pr_postfit_rms(
            last_nav,
            last_dd_pr,
            origin_ecef=origin_ecef,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            config=config,
            fgo_config=fgo_config,
            huber_weighted=False,
        )
    else:
        dd_pr_rms = float("inf")
    if last_dd_cp is not None:
        dd_cp_rms, n_cp = compute_dd_cp_postfit_rms(
            last_nav,
            last_dd_cp,
            origin_ecef=origin_ecef,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            config=config,
            ambiguity_values=ambiguity_values,
            layout=layout,
            local_epoch=local_epoch,
        )
    else:
        dd_cp_rms, n_cp = float("inf"), 0
    n_carrier = max(int(n_dd_carrier_factors), int(n_cp))
    reasons: list[str] = []
    if pos_sigma > float(config.ar_cert_max_pos_sigma_m):
        reasons.append(f"marginal_sigma={pos_sigma:.3f}>{config.ar_cert_max_pos_sigma_m}")
    if dd_pr_rms > float(config.ar_cert_max_dd_pr_rms_m):
        reasons.append(f"dd_pr_rms={dd_pr_rms:.3f}>{config.ar_cert_max_dd_pr_rms_m}")
    if last_dd_cp is not None and dd_cp_rms > float(config.ar_cert_max_dd_cp_rms_cyc):
        reasons.append(f"dd_cp_rms={dd_cp_rms:.3f}>{config.ar_cert_max_dd_cp_rms_cyc}")
    if int(epochs_since_recovery) < int(config.ar_cert_min_epochs_since_recovery):
        reasons.append(
            f"recovery_recency={epochs_since_recovery}<{config.ar_cert_min_epochs_since_recovery}"
        )
    if n_carrier < int(config.ar_cert_min_dd_carrier):
        reasons.append(f"n_dd_carrier={n_carrier}<{config.ar_cert_min_dd_carrier}")
    max_anchor = int(config.ar_cert_max_epochs_since_anchor)
    if max_anchor > 0 and int(epochs_since_anchor) > max_anchor:
        reasons.append(f"anchor_distance={epochs_since_anchor}>{max_anchor}")
    return FloatQualityCertificate(
        marginal_pos_sigma_m=float(pos_sigma),
        dd_pr_postfit_rms_m=float(dd_pr_rms),
        dd_cp_postfit_rms_cyc=float(dd_cp_rms),
        epochs_since_recovery=int(epochs_since_recovery),
        epochs_since_anchor=int(epochs_since_anchor),
        n_dd_carrier=int(n_carrier),
        passed=len(reasons) == 0,
        fail_reasons=tuple(reasons),
    )


def dd_pr_position_update_from_epoch(
    seed_ecef: np.ndarray,
    obs: DDPseudorangeEpoch,
    *,
    min_dd: int = 3,
    dd_sigma_m: float = 8.0,
    prior_sigma_m: float = 5.0,
    max_shift_m: float = 50.0,
    max_iter: int = 6,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """DD-only WLS position update from a ``DDPseudorangeEpoch`` (recovery helper)."""

    seed = np.asarray(seed_ecef, dtype=np.float64).reshape(3)
    stats: dict[str, float | int | bool] = {
        "accepted": False,
        "n_dd": int(obs.n),
        "shift_m": 0.0,
        "final_rms_m": float("inf"),
    }
    if obs.n < int(min_dd) or not np.all(np.isfinite(seed)):
        return seed.copy(), stats

    dd = np.asarray(obs.dd_pseudorange_m, dtype=np.float64).ravel()
    sat_k = np.asarray(obs.sat_ecef_k, dtype=np.float64).reshape(-1, 3)
    sat_ref = np.asarray(obs.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)
    base_k = np.asarray(obs.base_range_k, dtype=np.float64).ravel()
    base_ref = np.asarray(obs.base_range_ref, dtype=np.float64).ravel()
    weights = _weights(obs.weights, len(dd))

    pos = seed.copy()
    for _ in range(int(max_iter)):
        range_k = np.linalg.norm(sat_k - pos, axis=1)
        range_ref = np.linalg.norm(sat_ref - pos, axis=1)
        expected = range_k - range_ref - base_k + base_ref
        residual = dd - expected
        if not np.all(np.isfinite(residual)):
            return seed.copy(), stats
        unit_k = (sat_k - pos) / np.maximum(range_k[:, None], 1.0)
        unit_ref = (sat_ref - pos) / np.maximum(range_ref[:, None], 1.0)
        design = -unit_k + unit_ref
        w = np.clip(weights, 1.0e-6, None) / max(float(dd_sigma_m), 1.0e-6) ** 2
        lhs = design * np.sqrt(w)[:, None]
        rhs = residual * np.sqrt(w)
        if prior_sigma_m > 0.0:
            prior_w = 1.0 / (float(prior_sigma_m) ** 2)
            lhs = np.vstack([lhs, np.eye(3) * math.sqrt(prior_w)])
            rhs = np.concatenate([rhs, (seed - pos) * math.sqrt(prior_w)])
        try:
            delta, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return seed.copy(), stats
        if not np.all(np.isfinite(delta)):
            return seed.copy(), stats
        pos += delta
        if float(np.linalg.norm(delta)) < 1.0e-3:
            break

    shift = float(np.linalg.norm(pos - seed))
    range_k = np.linalg.norm(sat_k - pos, axis=1)
    range_ref = np.linalg.norm(sat_ref - pos, axis=1)
    residual = dd - (range_k - range_ref - base_k + base_ref)
    stats["shift_m"] = shift
    stats["final_rms_m"] = float(np.sqrt(np.mean(residual * residual)))
    rms_ok = math.isfinite(stats["final_rms_m"]) and float(stats["final_rms_m"]) < 15.0
    if shift <= float(max_shift_m) or rms_ok:
        stats["accepted"] = True
        return pos, stats
    return seed.copy(), stats


def quality_scaled_marginalization_prior(
    solved_front_state: TcFgoNavState,
    config: TcFgoConfig | None = None,
    *,
    dd_pr_rms_m: float = float("inf"),
    n_dd: int = 0,
) -> tuple[TcFgoNavState, np.ndarray]:
    """Diagonal prior for the sliding-window front, scaled by GNSS quality."""

    config = TcFgoConfig() if config is None else config
    sigmas = np.array(
        [
            config.marginal_pos_sigma_m,
            config.marginal_pos_sigma_m,
            config.marginal_pos_sigma_m,
            config.marginal_vel_sigma_mps,
            config.marginal_vel_sigma_mps,
            config.marginal_vel_sigma_mps,
        ],
        dtype=np.float64,
    )
    if config.optimize_imu_biases:
        sigmas = np.concatenate(
            [
                sigmas,
                np.full(3, config.bias_prior_sigma_accel, dtype=np.float64),
                np.full(3, config.bias_prior_sigma_gyro_radps, dtype=np.float64),
            ]
        )
    rms_ref = max(float(config.marginal_quality_rms_ref_m), 1.0e-3)
    min_dd = max(float(config.marginal_quality_min_dd), 1.0)
    quality_factor = max(1.0, float(dd_pr_rms_m) / rms_ref) if math.isfinite(dd_pr_rms_m) else 2.0
    dd_factor = max(1.0, min_dd / max(float(n_dd), 1.0))
    pos_scale = quality_factor * dd_factor
    sigmas[0:3] *= pos_scale
    return solved_front_state.copy(), sigmas


@dataclass
class AmbiguityEstimate:
    """One float DD ambiguity carried across TC windows."""

    value: float
    sigma: float
    generation: int
    last_epoch: int


@dataclass
class TcAmbiguityBank:
    """Cross-window float ambiguity memory keyed by DD pair (inuex35 amb_gen pattern)."""

    generation: int = 0
    estimates: dict[tuple[str, str, str, str], AmbiguityEstimate] = field(default_factory=dict)

    def bump_generation(self) -> None:
        self.generation += 1
        self.estimates.clear()

    def get(self, pair_key: tuple[str, str, str, str]) -> AmbiguityEstimate | None:
        est = self.estimates.get(pair_key)
        if est is None or int(est.generation) != int(self.generation):
            return None
        return est

    def update(
        self,
        pair_key: tuple[str, str, str, str],
        value: float,
        sigma: float,
        epoch: int,
    ) -> None:
        self.estimates[pair_key] = AmbiguityEstimate(
            value=float(value),
            sigma=float(sigma),
            generation=int(self.generation),
            last_epoch=int(epoch),
        )


@dataclass
class TcAmbiguityLayout:
    """Float ambiguity indices inside one TC window LM vector."""

    index_map: dict[tuple[int, int], int]
    pair_key_map: dict[tuple[int, int], tuple[str, str, str, str]]
    initial_values: np.ndarray
    held_map: dict[tuple[int, int], int]
    cross_window_priors: dict[int, tuple[float, float]] = field(default_factory=dict)

    @property
    def n_amb(self) -> int:
        return int(self.initial_values.size)


def tc_dd_pair_key(obs: DDCarrierEpoch, row: int) -> tuple[str, str, str, str]:
    """Public wrapper for DD pair keys (tests / runner)."""

    return _dd_pair_key(obs, row)


def build_ambiguity_layout(
    observations: Sequence[TcFgoEpochObs],
    config: TcFgoConfig,
    *,
    held_global: dict[tuple[int, tuple[str, str, str, str]], int] | None = None,
    window_start_epoch: int = 0,
    initial_nav: Sequence[TcFgoNavState] | None = None,
    origin_ecef: np.ndarray | None = None,
    origin_lat: float = 0.0,
    origin_lon: float = 0.0,
    ambiguity_bank: TcAmbiguityBank | None = None,
) -> TcAmbiguityLayout | None:
    """Build float ambiguity states for DD carrier rows in a window."""

    if not config.enable_dd_carrier:
        return None
    held_global = held_global or {}
    index_map: dict[tuple[int, int], int] = {}
    pair_key_map: dict[tuple[int, int], tuple[str, str, str, str]] = {}
    held_map: dict[tuple[int, int], int] = {}
    cross_window_priors: dict[int, tuple[float, float]] = {}
    initial: list[float] = []
    use_bank = bool(config.enable_persistent_ambiguities) and ambiguity_bank is not None
    prior_sigma = float(config.persistent_ambiguity_sigma_cyc)
    for local_i, obs in enumerate(observations):
        if obs.dd_carrier is None:
            continue
        global_epoch = int(window_start_epoch) + int(local_i)
        cp = obs.dd_carrier
        dd = np.asarray(cp.dd_carrier_cycles, dtype=np.float64).ravel()
        sat_k = np.asarray(cp.sat_ecef_k, dtype=np.float64).reshape(-1, 3)
        sat_ref = np.asarray(cp.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)
        base_k = np.asarray(cp.base_range_k, dtype=np.float64).ravel()
        base_ref = np.asarray(cp.base_range_ref, dtype=np.float64).ravel()
        wavelengths = np.asarray(cp.wavelengths_m, dtype=np.float64).ravel()
        if initial_nav is not None and origin_ecef is not None and local_i < len(initial_nav):
            pos_ecef = enu_to_ecef(initial_nav[local_i].p_enu, origin_ecef, origin_lat, origin_lon)
        else:
            pos_ecef = None
        for row in range(len(dd)):
            if not _valid_dd_row(
                dd[row], sat_k[row], sat_ref[row], base_k[row], base_ref[row], wavelengths[row]
            ):
                continue
            pair_key = _dd_pair_key(cp, row)
            held_int = held_global.get((global_epoch, pair_key))
            if held_int is None and cp.fixed_ambiguities is not None:
                fixed_row = np.asarray(cp.fixed_ambiguities, dtype=np.float64).ravel()
                if row < fixed_row.size and np.isfinite(fixed_row[row]):
                    held_int = int(round(float(fixed_row[row])))
            if held_int is not None:
                held_map[(local_i, row)] = int(held_int)
                continue
            if pos_ecef is not None:
                expected_m, _ = _dd_expected_and_jacobian_m(
                    pos_ecef, sat_k[row], sat_ref[row], base_k[row], base_ref[row]
                )
                n0 = float(dd[row] - expected_m / float(wavelengths[row]))
            else:
                n0 = float(dd[row])
            amb_idx = len(initial)
            bank_est = ambiguity_bank.get(pair_key) if use_bank else None
            if bank_est is not None:
                n0 = float(bank_est.value)
                cross_window_priors[amb_idx] = (
                    float(bank_est.value),
                    max(float(bank_est.sigma), prior_sigma),
                )
            index_map[(local_i, row)] = amb_idx
            pair_key_map[(local_i, row)] = pair_key
            initial.append(n0)
    if not index_map:
        return TcAmbiguityLayout(
            index_map={},
            pair_key_map={},
            initial_values=np.zeros(0),
            held_map=held_map,
            cross_window_priors=cross_window_priors,
        )
    return TcAmbiguityLayout(
        index_map=index_map,
        pair_key_map=pair_key_map,
        initial_values=np.asarray(initial, dtype=np.float64),
        held_map=held_map,
        cross_window_priors=cross_window_priors,
    )


def update_ambiguity_bank_from_window(
    bank: TcAmbiguityBank,
    layout: TcAmbiguityLayout | None,
    amb_values: np.ndarray | None,
    *,
    window_start_epoch: int,
    config: TcFgoConfig,
) -> None:
    """Store the latest solved float ambiguity per DD pair for the next window."""

    if layout is None or amb_values is None or layout.n_amb == 0:
        return
    amb = np.asarray(amb_values, dtype=np.float64).ravel()
    if amb.size < layout.n_amb:
        return
    sigma = float(config.persistent_ambiguity_sigma_cyc)
    last_by_pair: dict[tuple[str, str, str, str], tuple[int, int]] = {}
    for (local_i, row), amb_idx in layout.index_map.items():
        pair_key = layout.pair_key_map[(local_i, row)]
        prev = last_by_pair.get(pair_key)
        if prev is None or int(local_i) > int(prev[0]):
            last_by_pair[pair_key] = (int(local_i), int(amb_idx))
    for pair_key, (local_i, amb_idx) in last_by_pair.items():
        bank.update(
            pair_key,
            float(amb[int(amb_idx)]),
            sigma,
            int(window_start_epoch) + int(local_i),
        )


def apply_held_ambiguities_to_carrier(
    observations: Sequence[TcFgoEpochObs],
    held_global: dict[tuple[int, tuple[str, str, str, str]], int],
    *,
    window_start_epoch: int = 0,
) -> list[TcFgoEpochObs]:
    """Fold held integers into ``DDCarrierEpoch.fixed_ambiguities`` for the window."""

    out: list[TcFgoEpochObs] = []
    for local_i, obs in enumerate(observations):
        if obs.dd_carrier is None:
            out.append(obs)
            continue
        global_epoch = int(window_start_epoch) + int(local_i)
        cp = obs.dd_carrier
        n = int(cp.n)
        fixed = np.full(n, np.nan, dtype=np.float64)
        for row in range(n):
            pair_key = _dd_pair_key(cp, row)
            held = held_global.get((global_epoch, pair_key))
            if held is not None:
                fixed[row] = float(held)
        out.append(
            TcFgoEpochObs(
                dd_pseudorange=obs.dd_pseudorange,
                dd_carrier=DDCarrierEpoch(
                    dd_carrier_cycles=cp.dd_carrier_cycles,
                    sat_ecef_k=cp.sat_ecef_k,
                    sat_ecef_ref=cp.sat_ecef_ref,
                    base_range_k=cp.base_range_k,
                    base_range_ref=cp.base_range_ref,
                    wavelengths_m=cp.wavelengths_m,
                    weights=cp.weights,
                    sat_ids=cp.sat_ids,
                    ref_sat_ids=cp.ref_sat_ids,
                    fixed_ambiguities=fixed,
                ),
                enable_nhc=obs.enable_nhc,
                enable_zupt=obs.enable_zupt,
                anchor_pos_enu=obs.anchor_pos_enu,
                anchor_sigma_m=obs.anchor_sigma_m,
                doppler_vel_enu=obs.doppler_vel_enu,
                doppler_sigma_mps=obs.doppler_sigma_mps,
            )
        )
    return out


def ambiguity_carrier_residual_and_jacobian(
    pos_ecef: np.ndarray,
    obs: DDCarrierEpoch,
    row: int,
    *,
    ambiguity_value: float | None,
    held_integer: int | None,
) -> tuple[float, np.ndarray, float | None]:
    """Carrier residual (cycles) and Jacobians w.r.t. ECEF pos and ambiguity."""

    dd = float(np.asarray(obs.dd_carrier_cycles, dtype=np.float64).ravel()[row])
    sat_k = np.asarray(obs.sat_ecef_k, dtype=np.float64).reshape(-1, 3)[row]
    sat_ref = np.asarray(obs.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)[row]
    base_k = float(np.asarray(obs.base_range_k, dtype=np.float64).ravel()[row])
    base_ref = float(np.asarray(obs.base_range_ref, dtype=np.float64).ravel()[row])
    wavelength = float(np.asarray(obs.wavelengths_m, dtype=np.float64).ravel()[row])
    expected_m, jac_m = _dd_expected_and_jacobian_m(pos_ecef, sat_k, sat_ref, base_k, base_ref)
    if held_integer is not None:
        residual = dd - expected_m / wavelength - float(held_integer)
        return float(residual), -jac_m / wavelength, None
    if ambiguity_value is None:
        residual = _wrap_cycles(dd - expected_m / wavelength)
        return float(residual), jac_m, None
    residual = dd - expected_m / wavelength - float(ambiguity_value)
    return float(residual), -jac_m / wavelength, -1.0


def subset_ar_select(
    float_amb: np.ndarray,
    cov: np.ndarray,
    per_sat_residuals: np.ndarray,
    *,
    ratio_threshold: float,
    max_drop: int,
) -> tuple[np.ndarray | None, bool, float]:
    """Subset-AR: drop up to ``max_drop`` worst-residual sats; keep best ratio."""

    amb = np.asarray(float_amb, dtype=np.float64).ravel()
    q = np.asarray(cov, dtype=np.float64)
    resid = np.asarray(per_sat_residuals, dtype=np.float64).ravel()
    if amb.size == 0:
        return None, False, 0.0
    if amb.size == 1:
        fixed, ok, solution = solve_lambda(amb, q, ratio_threshold=float(ratio_threshold))
        return fixed, ok, float(solution.ratio)
    order = np.argsort(-np.abs(resid))
    best_ratio = 0.0
    best_fixed: np.ndarray | None = None
    best_ok = False
    median_abs = float(np.median(np.abs(resid))) if resid.size else 0.0
    for n_drop in range(0, min(int(max_drop), amb.size - 1) + 1):
        keep = np.sort(order[n_drop:])
        if keep.size == 0:
            continue
        sub_amb = amb[keep]
        sub_cov = q[np.ix_(keep, keep)]
        fixed_sub, ok, solution = solve_lambda(sub_amb, sub_cov, ratio_threshold=float(ratio_threshold))
        ratio = float(solution.ratio)
        if ok and fixed_sub is not None and ratio >= float(ratio_threshold):
            if n_drop == 0 and max_drop > 0:
                worst = float(np.max(np.abs(resid[keep])))
                if worst > max(3.0 * median_abs, 0.5):
                    continue
            if ratio > best_ratio or not best_ok:
                best_ratio = ratio
                best_ok = True
                full = np.full(amb.size, np.nan, dtype=np.float64)
                full[keep] = fixed_sub.astype(np.float64)
                best_fixed = full
        elif not best_ok and ratio > best_ratio:
            best_ratio = ratio
    return best_fixed, best_ok, best_ratio


def _estimate_tc_lambda_fixes(
    problem: TcFgoWindowProblem,
    float_result: TcFgoWindowResult,
    config: TcFgoConfig,
    fgo_config: LocalFgoConfig,
    *,
    layout: TcAmbiguityLayout | None = None,
) -> tuple[dict[tuple[int, tuple[str, str, str, str]], int], dict[str, Any]]:
    """LAMBDA fixes for one TC window, optionally with subset-AR on the last epoch."""

    n = len(problem.initial_states)
    win_start = int(problem.window_start_epoch)
    win = LocalFgoWindow(win_start, win_start + n - 1)
    positions_ecef = np.vstack(
        [
            enu_to_ecef(s.p_enu, problem.origin_ecef, problem.origin_lat, problem.origin_lon)
            for s in float_result.states
        ]
    )
    dd_carrier_padded: list[DDCarrierEpoch | None] = [None] * (win_start + n)
    for i, obs in enumerate(problem.observations):
        gi = win_start + i
        dd_carrier_padded[gi] = obs.dd_carrier
    lam_cfg = LambdaFixConfig(
        ratio_threshold=float(config.lambda_ratio_threshold),
        fixed_sigma_cycles=float(config.dd_cp_fixed_sigma_cycles),
        min_epochs=int(config.lambda_min_epochs),
        max_group_size=int(config.lambda_max_group_size),
        ddpr_reject_threshold=float(config.ddpr_reject_threshold),
        max_iterations=1,
    )
    if not config.enable_ar_subset:
        return _estimate_lambda_fixes(dd_carrier_padded, positions_ecef, win, lam_cfg)

    info: dict[str, Any] = {"subset_ar": True, "n_fixed": 0, "best_ratio": 0.0}
    fixes: dict[tuple[int, tuple[str, str, str, str]], int] = {}
    if layout is None or layout.n_amb == 0 or float_result.ambiguity_values is None:
        info["subset_skip"] = "no_float_ambiguities"
        return fixes, info

    last_local = n - 1
    last_obs = problem.observations[last_local].dd_carrier
    if last_obs is None:
        info["subset_skip"] = "no_carrier_last_epoch"
        return fixes, info

    pos_ecef = positions_ecef[last_local]
    amb = np.asarray(float_result.ambiguity_values, dtype=np.float64).ravel()
    pair_rows: list[tuple[tuple[str, str, str, str], float, float, int]] = []
    for (local_i, row), amb_idx in layout.index_map.items():
        if int(local_i) != int(last_local):
            continue
        if (local_i, row) in layout.held_map:
            continue
        pair_key = layout.pair_key_map[(local_i, row)]
        cp = last_obs
        residual, _, _ = ambiguity_carrier_residual_and_jacobian(
            pos_ecef,
            cp,
            row,
            ambiguity_value=float(amb[int(amb_idx)]),
            held_integer=None,
        )
        pair_rows.append((pair_key, float(amb[int(amb_idx)]), abs(float(residual)), int(row)))

    if len(pair_rows) < 2:
        plain, plain_info = _estimate_lambda_fixes(dd_carrier_padded, positions_ecef, win, lam_cfg)
        plain_info["subset_fallback"] = "too_few_pairs"
        return plain, plain_info

    float_amb = np.asarray([item[1] for item in pair_rows], dtype=np.float64)
    per_sat_resid = np.asarray([item[2] for item in pair_rows], dtype=np.float64)
    cov = np.diag(
        np.full(
            float_amb.size,
            max(float(config.dd_cp_sigma_cycles) ** 2, 1.0e-4),
            dtype=np.float64,
        )
    )
    fixed, ok, ratio = subset_ar_select(
        float_amb,
        cov,
        per_sat_resid,
        ratio_threshold=float(config.lambda_ratio_threshold),
        max_drop=int(config.subset_ar_max_drop),
    )
    info["best_ratio"] = float(ratio)
    info["n_candidates"] = int(float_amb.size)
    if not ok or fixed is None:
        info["n_ratio_rejected"] = 1
        return fixes, info

    global_epoch = win_start + last_local
    for idx, (pair_key, _amb, _res, _row) in enumerate(pair_rows):
        if idx >= fixed.size or not np.isfinite(fixed[idx]):
            continue
        integer = int(round(float(fixed[idx])))
        fixes[(global_epoch, pair_key)] = integer
        for local_i, row in layout.index_map:
            if layout.pair_key_map.get((local_i, row)) != pair_key:
                continue
            if (local_i, row) in layout.held_map:
                continue
            fixes[(win_start + int(local_i), pair_key)] = integer

    info["n_fixed"] = 1
    info["n_fixed_observations"] = int(len(fixes))
    return fixes, info


def _window_nav_dim(n_epochs: int, config: TcFgoConfig) -> int:
    return int(n_epochs) * state_dim(config)


def _unpack_window_state(
    x_flat: np.ndarray,
    n_epochs: int,
    config: TcFgoConfig,
    layout: TcAmbiguityLayout | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    sdim = state_dim(config)
    nav_flat = np.asarray(x_flat, dtype=np.float64).ravel()[: _window_nav_dim(n_epochs, config)]
    x_nav = nav_flat.reshape(n_epochs, sdim)
    if layout is None or layout.n_amb == 0:
        return x_nav, None
    amb = np.asarray(x_flat, dtype=np.float64).ravel()[_window_nav_dim(n_epochs, config) :]
    amb = amb[: layout.n_amb]
    return x_nav, amb


def _pack_window_state(
    x_nav: np.ndarray,
    amb: np.ndarray | None,
) -> np.ndarray:
    parts = [np.asarray(x_nav, dtype=np.float64).ravel()]
    if amb is not None and amb.size:
        parts.append(np.asarray(amb, dtype=np.float64).ravel())
    return np.concatenate(parts)


def _dd_pr_blocks_for_epoch(
    pos_ecef: np.ndarray,
    obs: DDPseudorangeEpoch,
    config: TcFgoConfig,
    fgo_config: LocalFgoConfig,
    *,
    last_raw_rms_m: float = float("inf"),
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    dd = np.asarray(obs.dd_pseudorange_m, dtype=np.float64).ravel()
    sat_k = np.asarray(obs.sat_ecef_k, dtype=np.float64).reshape(-1, 3)
    sat_ref = np.asarray(obs.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)
    base_k = np.asarray(obs.base_range_k, dtype=np.float64).ravel()
    base_ref = np.asarray(obs.base_range_ref, dtype=np.float64).ravel()
    weights = _weights(obs.weights, len(dd))
    residuals: list[np.ndarray] = []
    jacobians: list[np.ndarray] = []
    sigmas: list[float] = []
    x = np.asarray(pos_ecef, dtype=np.float64).reshape(3)
    huber_k = float(config.pr_huber_k)
    if (
        huber_k > 0.0
        and math.isfinite(last_raw_rms_m)
        and float(last_raw_rms_m) > float(config.pr_huber_disable_raw_rms_m)
    ):
        huber_k = 0.0
    for j in range(len(dd)):
        if not _valid_dd_row(dd[j], sat_k[j], sat_ref[j], base_k[j], base_ref[j], 1.0):
            continue
        expected_m, jac_m = _dd_expected_and_jacobian_m(x, sat_k[j], sat_ref[j], base_k[j], base_ref[j])
        residual = np.array([expected_m - float(dd[j])], dtype=np.float64)
        sigma = _weighted_sigma(config.dd_pr_sigma_m, weights[j], config.min_weight)
        scale = _huber_sqrt_weight(residual / sigma, huber_k)
        residuals.append(residual * scale)
        jacobians.append(jac_m.reshape(1, 3) * scale)
        sigmas.append(sigma)
    return residuals, jacobians, sigmas


def _linearize_window(
    x: np.ndarray,
    problem: TcFgoWindowProblem,
    config: TcFgoConfig,
    fgo_config: LocalFgoConfig,
    layout: TcAmbiguityLayout | None = None,
    ambiguity_bank: TcAmbiguityBank | None = None,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, int]]:
    n = len(problem.initial_states)
    sdim = state_dim(config)
    if layout is None and config.enable_dd_carrier:
        layout = build_ambiguity_layout(
            problem.observations,
            config,
            held_global=problem.held_ambiguities,
            window_start_epoch=int(problem.window_start_epoch),
            initial_nav=problem.initial_states,
            origin_ecef=problem.origin_ecef,
            origin_lat=problem.origin_lat,
            origin_lon=problem.origin_lon,
            ambiguity_bank=ambiguity_bank,
        )
    if layout is not None and layout.n_amb > 0:
        if x.ndim == 2:
            x_flat = _pack_window_state(x, layout.initial_values)
        else:
            x_flat = np.asarray(x, dtype=np.float64).ravel()
        x_nav, x_amb = _unpack_window_state(x_flat, n, config, layout)
    else:
        x_nav = np.asarray(x, dtype=np.float64).reshape(n, sdim) if x.ndim == 2 else np.asarray(x, dtype=np.float64).reshape(n, sdim)
        x_amb = None
        x_flat = _pack_window_state(x_nav, None)
    n_vars = int(x_flat.size)
    residual_blocks: list[np.ndarray] = []
    jacobian_blocks: list[np.ndarray] = []
    counts = {
        "imu": 0,
        "dd_pseudorange": 0,
        "dd_carrier": 0,
        "dd_carrier_fixed": 0,
        "n_continuity": 0,
        "n_cross_window_prior": 0,
        "nhc": 0,
        "zupt": 0,
        "marginal_prior": 0,
        "endpoint_prior": 0,
        "position_anchor": 0,
        "bias_rw": 0,
        "doppler_vel": 0,
    }

    def add_block(local_index: int, residual: np.ndarray, jac: np.ndarray, sigma: float) -> None:
        if sigma <= 0.0 or not np.isfinite(sigma):
            return
        r = np.asarray(residual, dtype=np.float64).ravel() / float(sigma)
        j = np.asarray(jac, dtype=np.float64) / float(sigma)
        if not (np.isfinite(r).all() and np.isfinite(j).all()):
            return
        row = np.zeros((r.size, n_vars), dtype=np.float64)
        col0 = int(local_index) * sdim
        row[:, col0 : col0 + j.shape[1]] = j
        residual_blocks.append(r)
        jacobian_blocks.append(row)

    def add_coupled(
        local_i: int,
        local_j: int,
        residual: np.ndarray,
        jac_i: np.ndarray,
        jac_j: np.ndarray,
        sigma: float,
    ) -> None:
        if sigma <= 0.0 or not np.isfinite(sigma):
            return
        r = np.asarray(residual, dtype=np.float64).ravel() / float(sigma)
        ji = np.asarray(jac_i, dtype=np.float64) / float(sigma)
        jj = np.asarray(jac_j, dtype=np.float64) / float(sigma)
        if not (np.isfinite(r).all() and np.isfinite(ji).all() and np.isfinite(jj).all()):
            return
        row = np.zeros((r.size, n_vars), dtype=np.float64)
        row[:, local_i * sdim : local_i * sdim + ji.shape[1]] = ji
        row[:, local_j * sdim : local_j * sdim + jj.shape[1]] = jj
        residual_blocks.append(r)
        jacobian_blocks.append(row)

    if problem.schur_marginal is not None:
        schur = problem.schur_marginal
        x_idx, prior_idx = _schur_overlap_indices(schur, n, layout, config)
        if x_idx.size > 0 and prior_idx.size == x_idx.size:
            sub_precision = schur.precision[np.ix_(prior_idx, prior_idx)]
            sub_mean = schur.mean[prior_idx]
            sub_precision = clamp_information_eigenvalues(
                sub_precision,
                float(config.schur_min_eigenvalue),
                max_eigenvalue=float(config.schur_max_eigenvalue) if config.schur_max_eigenvalue > 0.0 else None,
            )
            try:
                chol = np.linalg.cholesky(sub_precision)
            except np.linalg.LinAlgError:
                chol = None
            if chol is not None:
                x_flat_vec = np.asarray(x_flat, dtype=np.float64).ravel()
                delta_overlap = x_flat_vec[x_idx] - sub_mean
                residual = chol.T @ delta_overlap
                row_jac = np.zeros((residual.size, n_vars), dtype=np.float64)
                for r_i in range(residual.size):
                    for c_i, col in enumerate(x_idx):
                        row_jac[r_i, int(col)] = chol[c_i, r_i]
                if np.isfinite(residual).all() and np.isfinite(row_jac).all():
                    residual_blocks.append(residual)
                    jacobian_blocks.append(row_jac)
                    counts["marginal_prior"] += int(residual.size)
    elif problem.marginal_prior is not None:
        prior_x = state_vector_from_nav(problem.marginal_prior, config)
        sigmas = (
            np.asarray(problem.marginal_prior_sigmas, dtype=np.float64).reshape(-1)
            if problem.marginal_prior_sigmas is not None
            else np.array(
                [
                    config.marginal_pos_sigma_m,
                    config.marginal_pos_sigma_m,
                    config.marginal_pos_sigma_m,
                    config.marginal_vel_sigma_mps,
                    config.marginal_vel_sigma_mps,
                    config.marginal_vel_sigma_mps,
                ],
                dtype=np.float64,
            )
        )
        for dim in range(min(int(sigmas.size), int(prior_x.size))):
            sigma = float(sigmas[dim])
            if sigma <= 0.0:
                continue
            jac_row = np.zeros((1, sdim), dtype=np.float64)
            jac_row[0, dim] = 1.0
            add_block(0, np.array([x_nav[0, dim] - prior_x[dim]]), jac_row, sigma)
            counts["marginal_prior"] += 1

    endpoint_sigmas = [
        config.prior_pos_sigma_m,
        config.prior_pos_sigma_m,
        config.prior_pos_sigma_m,
        config.prior_vel_sigma_mps,
        config.prior_vel_sigma_mps,
        config.prior_vel_sigma_mps,
    ]
    if config.optimize_imu_biases:
        endpoint_sigmas.extend(
            [
                config.bias_prior_sigma_accel,
                config.bias_prior_sigma_accel,
                config.bias_prior_sigma_accel,
                config.bias_prior_sigma_gyro_radps,
                config.bias_prior_sigma_gyro_radps,
                config.bias_prior_sigma_gyro_radps,
            ]
        )
    init_last = state_vector_from_nav(problem.initial_states[-1], config)
    for dim, sigma in enumerate(endpoint_sigmas):
        if dim >= init_last.size:
            break
        jac_row = np.zeros((1, sdim), dtype=np.float64)
        jac_row[0, dim] = 1.0
        add_block(n - 1, np.array([x_nav[-1, dim] - init_last[dim]]), jac_row, float(sigma))
        counts["endpoint_prior"] += 1

    R_ecef_enu = _ecef_to_enu_rotation(problem.origin_lat, problem.origin_lon).T
    lever = np.asarray(config.lever_arm_body_m, dtype=np.float64).reshape(3)

    for i in range(n):
        nav_i = nav_from_state_vector(x_nav[i], problem.initial_states[i], config)
        obs = problem.observations[i]

        if obs.dd_pseudorange is not None:
            pos_ecef = enu_to_ecef(nav_i.p_enu, problem.origin_ecef, problem.origin_lat, problem.origin_lon)
            dd_residuals, dd_jacs, dd_sigmas = _dd_pr_blocks_for_epoch(
                pos_ecef,
                obs.dd_pseudorange,
                config,
                fgo_config,
                last_raw_rms_m=float(problem.last_dd_pr_rms_m),
            )
            for r_blk, j_blk, sigma in zip(dd_residuals, dd_jacs, dd_sigmas, strict=True):
                jac_enu = np.zeros((1, sdim), dtype=np.float64)
                jac_enu[0, 0:3] = j_blk[0] @ R_ecef_enu
                add_block(i, r_blk, jac_enu, sigma)
                counts["dd_pseudorange"] += int(r_blk.size)

        if config.enable_dd_carrier and obs.dd_carrier is not None:
            pos_ecef = enu_to_ecef(nav_i.p_enu, problem.origin_ecef, problem.origin_lat, problem.origin_lon)
            cp = obs.dd_carrier
            n_rows = int(cp.n)
            for row in range(n_rows):
                held_int = None
                if layout is not None:
                    held_int = layout.held_map.get((i, row))
                amb_val = None
                amb_idx = None
                if layout is not None and held_int is None:
                    amb_idx = layout.index_map.get((i, row))
                    if amb_idx is not None and x_amb is not None:
                        amb_val = float(x_amb[int(amb_idx)])
                residual, jac_ecef, jac_n = ambiguity_carrier_residual_and_jacobian(
                    pos_ecef,
                    cp,
                    row,
                    ambiguity_value=amb_val,
                    held_integer=held_int,
                )
                sigma = (
                    float(config.dd_cp_fixed_sigma_cycles)
                    if held_int is not None
                    else float(config.dd_cp_sigma_cycles)
                )
                weights = _weights(cp.weights, n_rows)
                sigma = _weighted_sigma(sigma, weights[row], config.min_weight)
                scale = _huber_sqrt_weight(np.array([residual]) / sigma, float(config.dd_cp_huber_k))
                r_scaled = np.array([residual * scale], dtype=np.float64)
                row_jac = np.zeros((1, n_vars), dtype=np.float64)
                col0 = i * sdim
                row_jac[0, col0 : col0 + 3] = (jac_ecef @ R_ecef_enu) * scale
                if jac_n is not None and amb_idx is not None:
                    row_jac[0, _window_nav_dim(n, config) + int(amb_idx)] = float(jac_n) * scale
                if np.isfinite(r_scaled).all() and np.isfinite(row_jac).all():
                    residual_blocks.append(r_scaled / sigma)
                    jacobian_blocks.append(row_jac / sigma)
                    counts["dd_carrier"] += 1
                    if held_int is not None:
                        counts["dd_carrier_fixed"] += 1

        if obs.anchor_pos_enu is not None and obs.anchor_sigma_m is not None:
            r_anchor, j_anchor = position_anchor_residual_and_jacobian(nav_i.p_enu, obs.anchor_pos_enu)
            sigma = float(obs.anchor_sigma_m)
            for rr in range(3):
                jac_pos = np.zeros((1, sdim), dtype=np.float64)
                jac_pos[0, 0:3] = j_anchor[rr]
                add_block(i, np.array([r_anchor[rr]]), jac_pos, sigma)
                counts["position_anchor"] += 1

        dop_sigma = (
            float(obs.doppler_sigma_mps)
            if obs.doppler_sigma_mps is not None and obs.doppler_sigma_mps > 0.0
            else float(config.doppler_body_vel_sigma_mps)
        )
        if obs.doppler_vel_enu is not None and dop_sigma > 0.0:
            r_dop = nav_i.v_enu - np.asarray(obs.doppler_vel_enu, dtype=np.float64).reshape(3)
            for rr in range(3):
                jac_vel = np.zeros((1, sdim), dtype=np.float64)
                jac_vel[0, 3:6] = np.eye(3, dtype=np.float64)[rr]
                add_block(i, np.array([r_dop[rr]]), jac_vel, dop_sigma)
                counts["doppler_vel"] += 1

        if obs.enable_nhc:
            r_nhc, j_nhc = nhc_residual_and_jacobian(nav_i.v_enu, nav_i.q_body_to_enu)
            for rr in range(r_nhc.size):
                jac_vel = np.zeros((1, sdim), dtype=np.float64)
                jac_vel[0, 3:6] = j_nhc[rr]
                add_block(i, np.array([r_nhc[rr]]), jac_vel, config.nhc_sigma_mps)
            counts["nhc"] += int(r_nhc.size)

        if obs.enable_zupt:
            r_z, j_z = zupt_residual_and_jacobian(nav_i.v_enu)
            for rr in range(3):
                jac_vel = np.zeros((1, sdim), dtype=np.float64)
                jac_vel[0, 3:6] = j_z[rr]
                add_block(i, np.array([r_z[rr]]), jac_vel, config.zupt_sigma_mps)
            counts["zupt"] += 3

    if layout is not None and layout.n_amb > 0 and x_amb is not None:
        nav_base = _window_nav_dim(n, config)
        pair_to_entries: dict[tuple[str, str, str, str], list[tuple[int, int, int]]] = {}
        for (local_i, row), amb_idx in layout.index_map.items():
            pair_key = layout.pair_key_map[(local_i, row)]
            pair_to_entries.setdefault(pair_key, []).append((local_i, row, amb_idx))
        for _pair_key, entries in pair_to_entries.items():
            entries_sorted = sorted(entries, key=lambda item: item[0])
            for k in range(len(entries_sorted) - 1):
                i0, _row0, idx0 = entries_sorted[k]
                i1, _row1, idx1 = entries_sorted[k + 1]
                if i1 != i0 + 1:
                    continue
                sigma = float(config.n_continuity_sigma_float_cyc)
                residual = np.array([float(x_amb[idx1] - x_amb[idx0])], dtype=np.float64)
                row_jac = np.zeros((1, n_vars), dtype=np.float64)
                row_jac[0, nav_base + idx1] = 1.0
                row_jac[0, nav_base + idx0] = -1.0
                if sigma > 0.0 and np.isfinite(residual).all():
                    residual_blocks.append(residual / sigma)
                    jacobian_blocks.append(row_jac / sigma)
                    counts["n_continuity"] += 1

    if (
        layout is not None
        and x_amb is not None
        and layout.cross_window_priors
        and (not config.enable_schur_marginalization or config.schur_use_bank_ambiguity_priors)
    ):
        nav_base = _window_nav_dim(n, config)
        for amb_idx, (prior_val, prior_sigma) in layout.cross_window_priors.items():
            if int(amb_idx) >= int(x_amb.size):
                continue
            sigma = max(float(prior_sigma), 1.0e-6)
            residual = np.array([float(x_amb[int(amb_idx)] - prior_val)], dtype=np.float64)
            row_jac = np.zeros((1, n_vars), dtype=np.float64)
            row_jac[0, nav_base + int(amb_idx)] = 1.0
            if sigma > 0.0 and np.isfinite(residual).all():
                residual_blocks.append(residual / sigma)
                jacobian_blocks.append(row_jac / sigma)
                counts["n_cross_window_prior"] += 1

    for i in range(n - 1):
        seg = problem.imu_segments[i]
        if seg is None:
            continue
        nav_i = nav_from_state_vector(x_nav[i], problem.initial_states[i], config)
        nav_j = nav_from_state_vector(x_nav[i + 1], problem.initial_states[i + 1], config)
        q_j = _quat_normalize(
            _quat_multiply(nav_i.q_body_to_enu, _quat_from_axis_angle(seg.delta_angle_body))
        )
        r_p, r_v = imu_preintegration_residual(
            nav_i.p_enu,
            nav_i.v_enu,
            nav_j.p_enu,
            nav_j.v_enu,
            nav_i.q_body_to_enu,
            q_j,
            seg,
            b_a=nav_i.b_a,
            b_g=nav_i.b_g,
            b_a_lin=problem.initial_states[i].b_a,
            b_g_lin=problem.initial_states[i].b_g,
            lever_arm_body_m=lever,
        )
        (
            jac_pi_p,
            jac_pi_v,
            jac_pj_p,
            jac_vi_v,
            jac_vj_v,
            jac_ba,
            jac_bg,
            jac_v_ba,
            jac_v_bg,
        ) = imu_preintegration_jacobian(
            nav_i.p_enu,
            nav_i.v_enu,
            nav_j.p_enu,
            nav_j.v_enu,
            nav_i.q_body_to_enu,
            seg,
            lever_arm_body_m=lever,
            include_bias_jacobians=config.optimize_imu_biases,
        )
        imu_scale = imu_gnss_quality_scale(problem.last_dd_pr_rms_m, seg.delta_t_s, config)
        imu_pos_sigma = config.imu_pos_sigma_m * imu_scale
        imu_vel_sigma = config.imu_vel_sigma_mps * imu_scale

        if config.optimize_imu_biases:
            jac_state_i = np.hstack([jac_pi_p, jac_pi_v, jac_ba, jac_bg])
            jac_state_j_p = np.hstack([jac_pj_p, np.zeros((3, 3), dtype=np.float64), np.zeros((3, 6), dtype=np.float64)])
            jac_state_i_v = np.hstack([np.zeros((3, 3), dtype=np.float64), jac_vi_v, jac_v_ba, jac_v_bg])
            jac_state_j_v = np.hstack([np.zeros((3, 3), dtype=np.float64), jac_vj_v, np.zeros((3, 6), dtype=np.float64)])
        else:
            jac_state_i = np.hstack([jac_pi_p, jac_pi_v])
            jac_state_j_p = np.hstack([jac_pj_p, np.zeros((3, 3), dtype=np.float64)])
            jac_state_i_v = np.hstack([np.zeros((3, 3), dtype=np.float64), jac_vi_v])
            jac_state_j_v = np.hstack([np.zeros((3, 3), dtype=np.float64), jac_vj_v])

        for rr in range(3):
            add_coupled(
                i,
                i + 1,
                np.array([r_p[rr]]),
                jac_state_i[rr : rr + 1],
                jac_state_j_p[rr : rr + 1],
                imu_pos_sigma,
            )
            counts["imu"] += 1
        for rr in range(3):
            add_coupled(
                i,
                i + 1,
                np.array([r_v[rr]]),
                jac_state_i_v[rr : rr + 1],
                jac_state_j_v[rr : rr + 1],
                imu_vel_sigma,
            )
            counts["imu"] += 1

        if config.optimize_imu_biases:
            r_ba = bias_random_walk_residual(nav_j.b_a, nav_i.b_a)
            r_bg = bias_random_walk_residual(nav_j.b_g, nav_i.b_g)
            for rr in range(3):
                jac_i = np.zeros((1, sdim), dtype=np.float64)
                jac_j = np.zeros((1, sdim), dtype=np.float64)
                jac_i[0, 6 + rr] = -1.0
                jac_j[0, 6 + rr] = 1.0
                add_coupled(i, i + 1, np.array([r_ba[rr]]), jac_i, jac_j, config.bias_rw_sigma_accel)
                counts["bias_rw"] += 1
            for rr in range(3):
                jac_i = np.zeros((1, sdim), dtype=np.float64)
                jac_j = np.zeros((1, sdim), dtype=np.float64)
                jac_i[0, 9 + rr] = -1.0
                jac_j[0, 9 + rr] = 1.0
                add_coupled(i, i + 1, np.array([r_bg[rr]]), jac_i, jac_j, config.bias_rw_sigma_gyro_radps)
                counts["bias_rw"] += 1

    if not residual_blocks:
        return 0.0, np.zeros(0, dtype=np.float64), np.zeros((0, n_vars), dtype=np.float64), counts
    residuals = np.concatenate(residual_blocks)
    jacobian = np.vstack(jacobian_blocks)
    return float(0.5 * np.dot(residuals, residuals)), residuals, jacobian, counts


def solve_tc_fgo_window(
    problem: TcFgoWindowProblem,
    config: TcFgoConfig | None = None,
    fgo_config: LocalFgoConfig | None = None,
    ambiguity_bank: TcAmbiguityBank | None = None,
) -> TcFgoWindowResult:
    """Levenberg-Marquardt on a single TC-FGO window (numpy path)."""

    config = TcFgoConfig() if config is None else config
    fgo_config = LocalFgoConfig() if fgo_config is None else fgo_config
    n = len(problem.initial_states)
    if n == 0:
        raise ValueError("window must contain at least one epoch")
    x_nav = np.vstack([state_vector_from_nav(s, config) for s in problem.initial_states])
    layout = build_ambiguity_layout(
        problem.observations,
        config,
        held_global=problem.held_ambiguities,
        window_start_epoch=int(problem.window_start_epoch),
        initial_nav=problem.initial_states,
        origin_ecef=problem.origin_ecef,
        origin_lat=problem.origin_lat,
        origin_lon=problem.origin_lon,
        ambiguity_bank=ambiguity_bank,
    )
    amb_init = layout.initial_values if layout is not None and layout.n_amb > 0 else None
    x_flat = _pack_window_state(x_nav, amb_init)
    initial_error, _, _, _ = _linearize_window(
        x_flat, problem, config, fgo_config, layout=layout, ambiguity_bank=ambiguity_bank
    )
    current_error = initial_error
    damping = 1e-3
    counts: dict[str, int] = {}
    n_iterations = 0
    converged = False

    for _ in range(max(0, int(config.max_iterations))):
        n_iterations += 1
        _cost, residuals, jacobian, counts = _linearize_window(
            x_flat, problem, config, fgo_config, layout=layout, ambiguity_bank=ambiguity_bank
        )
        if residuals.size == 0:
            break
        hessian = jacobian.T @ jacobian
        gradient = np.asarray(jacobian.T @ residuals, dtype=np.float64).ravel()
        diag = np.maximum(np.diag(hessian), 1e-12)
        accepted = False
        for _attempt in range(8):
            try:
                step = np.linalg.solve(hessian + np.diag(float(damping) * diag), -gradient)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            if not np.isfinite(step).all():
                damping *= 10.0
                continue
            trial_flat = x_flat + step
            trial_error, _, _, _ = _linearize_window(
                trial_flat, problem, config, fgo_config, layout=layout, ambiguity_bank=ambiguity_bank
            )
            if np.isfinite(trial_error) and trial_error < current_error:
                rel = (current_error - trial_error) / max(current_error, 1.0)
                x_flat = trial_flat
                current_error = trial_error
                damping = max(damping * 0.3, 1e-9)
                accepted = True
                if rel < float(config.relative_error_tol):
                    converged = True
                    break
                break
            damping *= 10.0
        if not accepted:
            break
        if converged:
            break

    _, _, _, counts = _linearize_window(
        x_flat, problem, config, fgo_config, layout=layout, ambiguity_bank=ambiguity_bank
    )
    x_nav_final, amb_final = _unpack_window_state(x_flat, n, config, layout)
    states = [nav_from_state_vector(x_nav_final[i], problem.initial_states[i], config) for i in range(n)]
    if (
        ambiguity_bank is not None
        and config.enable_persistent_ambiguities
        and config.enable_dd_carrier
        and (not config.enable_schur_marginalization or config.schur_use_bank_ambiguity_priors)
    ):
        update_ambiguity_bank_from_window(
            ambiguity_bank,
            layout,
            amb_final,
            window_start_epoch=int(problem.window_start_epoch),
            config=config,
        )
    result = TcFgoWindowResult(
        states=states,
        initial_error=float(initial_error),
        final_error=float(current_error),
        factor_counts=counts,
        n_iterations=int(n_iterations),
        converged=bool(converged),
        ambiguity_values=amb_final.copy() if amb_final is not None else None,
    )
    if config.enable_schur_marginalization and n >= 2:
        schur = compute_schur_marginal_from_window(
            x_flat,
            problem,
            config,
            fgo_config,
            layout=layout,
            ambiguity_bank=ambiguity_bank,
        )
        if schur is not None:
            result.schur_marginal = schur
    if config.enable_lambda_ar:
        if config.enable_ar_quality_gate:
            last_local = n - 1
            last_obs = problem.observations[last_local]
            cert = evaluate_float_quality_certificate(
                config=config,
                schur_marginal=problem.schur_marginal,
                last_nav=result.states[last_local],
                last_dd_pr=last_obs.dd_pseudorange,
                last_dd_cp=last_obs.dd_carrier,
                origin_ecef=problem.origin_ecef,
                origin_lat=problem.origin_lat,
                origin_lon=problem.origin_lon,
                fgo_config=fgo_config,
                ambiguity_values=result.ambiguity_values,
                layout=layout,
                local_epoch=last_local,
                epochs_since_recovery=int(problem.epochs_since_recovery),
                epochs_since_anchor=int(problem.epochs_since_anchor),
                n_dd_carrier_factors=int(result.factor_counts.get("dd_carrier", 0)),
            )
            if not cert.passed:
                result.ar_info = {
                    "float_error": float(result.final_error),
                    "certificate": {
                        "passed": False,
                        "marginal_pos_sigma_m": cert.marginal_pos_sigma_m,
                        "dd_pr_postfit_rms_m": cert.dd_pr_postfit_rms_m,
                        "dd_cp_postfit_rms_cyc": cert.dd_cp_postfit_rms_cyc,
                        "epochs_since_recovery": cert.epochs_since_recovery,
                        "epochs_since_anchor": cert.epochs_since_anchor,
                        "fail_reasons": list(cert.fail_reasons),
                    },
                }
                result.epoch_fixed = [False] * n
                return result
        return run_tc_window_ar(
            problem,
            result,
            config,
            fgo_config,
            layout=layout,
        )
    return result


def run_tc_window_ar(
    problem: TcFgoWindowProblem,
    float_result: TcFgoWindowResult,
    config: TcFgoConfig,
    fgo_config: LocalFgoConfig,
    *,
    layout: TcAmbiguityLayout | None = None,
) -> TcFgoWindowResult:
    """LAMBDA + subset-AR + DDPR cross-check + post-AR cost gate on one window."""

    n = len(problem.initial_states)
    win_start = int(problem.window_start_epoch)
    win = LocalFgoWindow(win_start, win_start + n - 1)
    positions_ecef = np.vstack(
        [
            enu_to_ecef(s.p_enu, problem.origin_ecef, problem.origin_lat, problem.origin_lon)
            for s in float_result.states
        ]
    )
    dd_pr_padded: list[DDPseudorangeEpoch | None] = [None] * (win_start + n)
    for i, obs in enumerate(problem.observations):
        gi = win_start + i
        dd_pr_padded[gi] = obs.dd_pseudorange

    fixes, ar_info = _estimate_tc_lambda_fixes(
        problem,
        float_result,
        config,
        fgo_config,
        layout=layout,
    )
    ar_summary: dict[str, Any] = {"float_error": float(float_result.final_error), "ar_info": ar_info}
    if config.enable_ar_quality_gate and float_result.ar_info is None:
        cert = evaluate_float_quality_certificate(
            config=config,
            schur_marginal=problem.schur_marginal,
            last_nav=float_result.states[-1],
            last_dd_pr=problem.observations[-1].dd_pseudorange,
            last_dd_cp=problem.observations[-1].dd_carrier,
            origin_ecef=problem.origin_ecef,
            origin_lat=problem.origin_lat,
            origin_lon=problem.origin_lon,
            fgo_config=fgo_config,
            ambiguity_values=float_result.ambiguity_values,
            layout=layout,
            local_epoch=n - 1,
            epochs_since_recovery=int(problem.epochs_since_recovery),
            epochs_since_anchor=int(problem.epochs_since_anchor),
            n_dd_carrier_factors=int(float_result.factor_counts.get("dd_carrier", 0)),
        )
        ar_summary["certificate"] = {
            "passed": cert.passed,
            "marginal_pos_sigma_m": cert.marginal_pos_sigma_m,
            "dd_pr_postfit_rms_m": cert.dd_pr_postfit_rms_m,
            "dd_cp_postfit_rms_cyc": cert.dd_cp_postfit_rms_cyc,
            "epochs_since_recovery": cert.epochs_since_recovery,
            "epochs_since_anchor": cert.epochs_since_anchor,
            "fail_reasons": list(cert.fail_reasons),
        }

    if not fixes:
        float_result.ar_info = ar_summary
        float_result.epoch_fixed = [False] * n
        return float_result

    held = dict(problem.held_ambiguities or {})
    for (epoch_idx, pair_key), integer in fixes.items():
        rel = int(epoch_idx) - win_start
        if 0 <= rel < n:
            held[(int(epoch_idx), pair_key)] = int(integer)

    obs_with_held = apply_held_ambiguities_to_carrier(problem.observations, held, window_start_epoch=win_start)
    ar_problem = TcFgoWindowProblem(
        initial_states=float_result.states,
        imu_segments=problem.imu_segments,
        observations=obs_with_held,
        origin_ecef=problem.origin_ecef,
        origin_lat=problem.origin_lat,
        origin_lon=problem.origin_lon,
        marginal_prior=problem.marginal_prior,
        marginal_prior_sigmas=problem.marginal_prior_sigmas,
        schur_marginal=problem.schur_marginal,
        last_dd_pr_rms_m=problem.last_dd_pr_rms_m,
        held_ambiguities=held,
        window_start_epoch=win_start,
        epochs_since_recovery=int(problem.epochs_since_recovery),
        epochs_since_anchor=int(problem.epochs_since_anchor),
    )
    fixed_config = replace(config, enable_lambda_ar=False)
    candidate = solve_tc_fgo_window(ar_problem, config=fixed_config, fgo_config=fgo_config)

    touched = sorted({int(ep) for ep, _ in fixes})
    pos_before = positions_ecef
    pos_after = np.vstack(
        [
            enu_to_ecef(s.p_enu, problem.origin_ecef, problem.origin_lat, problem.origin_lon)
            for s in candidate.states
        ]
    )
    ddpr_threshold = float(config.ddpr_reject_threshold) if config.enable_ar_ddpr_crossval else 0.0
    accepted, rms_before, rms_after = _ddpr_cross_check(
        dd_pr_padded,
        win,
        touched,
        pos_before,
        pos_after,
        ddpr_threshold,
    )
    ar_summary["ddpr_rms_before"] = rms_before
    ar_summary["ddpr_rms_after"] = rms_after
    ar_summary["ddpr_gate_accepted"] = bool(accepted)
    if not accepted:
        float_result.ar_info = ar_summary
        float_result.epoch_fixed = [False] * n
        return float_result

    if (
        config.enable_ar_post_ar_gate
        and rms_before is not None
        and rms_after is not None
        and float(config.post_ar_ddpr_degrade_threshold) > 0.0
        and rms_after > rms_before * (1.0 + float(config.post_ar_ddpr_degrade_threshold))
    ):
        ar_summary["post_ar_gate_accepted"] = False
        float_result.ar_info = ar_summary
        float_result.epoch_fixed = [False] * n
        return float_result

    ar_summary["post_ar_gate_accepted"] = True
    ar_summary["n_fixed_observations"] = int(len(fixes))
    candidate.ar_accepted = True
    candidate.ar_info = ar_summary
    fixed_epochs = {int(ep) for ep, _ in fixes}
    candidate.epoch_fixed = [(win_start + i) in fixed_epochs for i in range(n)]
    if config.enable_ar_hold:
        candidate.accepted_fixes = dict(fixes)
    return candidate


@dataclass
class PhaseInitConfig:
    """Two-phase initialization (mirrors inuex35 runner thresholds)."""

    vel_thresh_mps: float = 1.0
    n_collect_fixes: int = 5
    static_speed_max_mps: float = 1.0


def collect_static_imu_samples(
    imu_times_s: np.ndarray,
    imu_acc: np.ndarray,
    imu_gyro_dps: np.ndarray,
    t_min: float,
    t_max: float,
) -> np.ndarray:
    """Return IMU rows ``[t, ax, ay, az, gx, gy, gz]`` in the static window."""

    times = np.asarray(imu_times_s, dtype=np.float64).ravel()
    mask = (times >= float(t_min)) & (times <= float(t_max))
    if not mask.any():
        return np.zeros((0, 7), dtype=np.float64)
    acc = np.asarray(imu_acc, dtype=np.float64)[mask]
    gyro = np.asarray(imu_gyro_dps, dtype=np.float64)[mask]
    return np.column_stack([times[mask], acc, gyro])


def run_two_phase_initialization(
    ins: INSEKF,
    *,
    epoch_times_s: np.ndarray,
    rtk_fix_positions_ecef: Sequence[tuple[float, np.ndarray]],
    imu_samples_static: np.ndarray,
    origin_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
    static_fix_positions_ecef: Sequence[tuple[float, np.ndarray]] | None = None,
    phase_cfg: PhaseInitConfig | None = None,
) -> tuple[TcFgoNavState, int]:
    """Phase 1 static alignment + Phase 2 yaw from velocity; returns state and phase-2 index."""

    phase_cfg = PhaseInitConfig() if phase_cfg is None else phase_cfg
    for row in np.asarray(imu_samples_static, dtype=np.float64):
        if row.size < 7:
            continue
        ins.feed_imu_for_alignment(float(row[0]), row[1:4], row[4:7])
    if not ins.aligned:
        ins.feed_imu_for_alignment(0.0, np.array([0.0, 0.0, 9.81]), np.zeros(3))

    static_fixes = (
        list(static_fix_positions_ecef)
        if static_fix_positions_ecef is not None
        else list(rtk_fix_positions_ecef)
    )
    fixes = list(rtk_fix_positions_ecef)
    if len(static_fixes) < int(phase_cfg.n_collect_fixes):
        raise ValueError(
            f"need at least {phase_cfg.n_collect_fixes} static RTK FIX epochs for phase-1 init, "
            f"got {len(static_fixes)}"
        )
    fix_pos_ecef = np.mean(np.vstack([p for _, p in static_fixes[: phase_cfg.n_collect_fixes]]), axis=0)
    p_enu = ecef_to_enu(fix_pos_ecef, origin_ecef, origin_lat, origin_lon)
    ins.initialize_position(p_enu)

    times = np.asarray(epoch_times_s, dtype=np.float64).ravel()
    phase2_idx = int(times.size - 1) if times.size else 0
    for j in range(1, len(fixes)):
        t0, p0 = fixes[j - 1]
        t1, p1 = fixes[j]
        dt_fix = float(t1 - t0)
        if dt_fix <= 0.0:
            continue
        e0 = ecef_to_enu(p0, origin_ecef, origin_lat, origin_lon)
        e1 = ecef_to_enu(p1, origin_ecef, origin_lat, origin_lon)
        v_est = (e1 - e0) / dt_fix
        speed = float(math.hypot(float(v_est[0]), float(v_est[1])))
        if speed > float(phase_cfg.vel_thresh_mps):
            ins.initialize_yaw_from_velocity(v_est)
            phase2_idx = int(np.searchsorted(times, float(t1), side="left"))
            phase2_idx = min(max(phase2_idx, 0), int(times.size - 1))
            break

    state = TcFgoNavState(
        p_enu=ins.position_enu(),
        v_enu=ins.velocity_enu(),
        q_body_to_enu=ins.q.copy(),
        b_a=ins.accel_bias_body(),
        b_g=ins.gyro_bias_body_radps(),
    )
    return state, phase2_idx


def propagate_nav_state_with_imu(
    state: TcFgoNavState,
    imu_rows: np.ndarray,
    ins_cfg: INSConfig | None = None,
) -> TcFgoNavState:
    """Propagate a nominal state through raw IMU rows using INSEKF mechanization."""

    ins = INSEKF(ins_cfg)
    ins.aligned = True
    ins.yaw_initialized = True
    ins.p = np.asarray(state.p_enu, dtype=np.float64).copy()
    ins.v = np.asarray(state.v_enu, dtype=np.float64).copy()
    ins.q = np.asarray(state.q_body_to_enu, dtype=np.float64).copy()
    ins.b_a = np.asarray(state.b_a, dtype=np.float64).copy()
    ins.b_g = np.asarray(state.b_g, dtype=np.float64).copy()
    ins.last_t = None
    ins.propagate(np.asarray(imu_rows, dtype=np.float64))
    return TcFgoNavState(
        p_enu=ins.position_enu(),
        v_enu=ins.velocity_enu(),
        q_body_to_enu=ins.q.copy(),
        b_a=ins.accel_bias_body(),
        b_g=ins.gyro_bias_body_radps(),
    )


def naive_marginalization_prior(
    solved_front_state: TcFgoNavState,
    config: TcFgoConfig | None = None,
) -> tuple[TcFgoNavState, np.ndarray]:
    """Build a diagonal prior for the sliding-window front after dropping one epoch."""

    config = TcFgoConfig() if config is None else config
    sigmas = np.array(
        [
            config.marginal_pos_sigma_m,
            config.marginal_pos_sigma_m,
            config.marginal_pos_sigma_m,
            config.marginal_vel_sigma_mps,
            config.marginal_vel_sigma_mps,
            config.marginal_vel_sigma_mps,
        ],
        dtype=np.float64,
    )
    if config.optimize_imu_biases:
        sigmas = np.concatenate(
            [
                sigmas,
                np.full(3, config.bias_prior_sigma_accel, dtype=np.float64),
                np.full(3, config.bias_prior_sigma_gyro_radps, dtype=np.float64),
            ]
        )
    return solved_front_state.copy(), sigmas


@dataclass
class TcSchurMarginal:
    """Dense information prior carried across sliding-window steps (WP12c)."""

    mean: np.ndarray
    precision: np.ndarray
    n_nav_epochs: int
    sdim: int
    amb_pair_keys: list[tuple[str, str, str, str]] = field(default_factory=list)


def clamp_information_eigenvalues(
    precision: np.ndarray,
    min_eigenvalue: float,
    *,
    max_eigenvalue: float | None = None,
) -> np.ndarray:
    """Floor/cap eigenvalues of a symmetric information matrix for stability."""

    mat = np.asarray(precision, dtype=np.float64)
    sym = 0.5 * (mat + mat.T)
    floor = max(float(min_eigenvalue), 1.0e-15)
    try:
        eigvals, eigvecs = np.linalg.eigh(sym)
    except np.linalg.LinAlgError:
        return np.eye(sym.shape[0], dtype=np.float64) * floor
    eigvals = np.maximum(eigvals, floor)
    if max_eigenvalue is not None and float(max_eigenvalue) > 0.0:
        eigvals = np.minimum(eigvals, float(max_eigenvalue))
    return (eigvecs * eigvals) @ eigvecs.T


@dataclass
class SchurMarginalBlocks:
    """Information-form Schur marginal on remaining variables."""

    precision: np.ndarray
    linear: np.ndarray


def schur_complement_marginalize(
    hessian: np.ndarray,
    gradient: np.ndarray,
    n_margin: int,
    *,
    min_eigenvalue: float = 1.0e-6,
    max_eigenvalue: float | None = None,
) -> SchurMarginalBlocks | None:
    """Marginalize the first ``n_margin`` variables from (H, g)."""

    m = int(n_margin)
    d = int(np.asarray(hessian, dtype=np.float64).shape[0])
    if m <= 0 or m >= d:
        return None
    H = np.asarray(hessian, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64).reshape(-1)
    Hmm = H[:m, :m]
    Hmr = H[:m, m:]
    Hrm = H[m:, :m]
    Hrr = H[m:, m:]
    gm = g[:m]
    gr = g[m:]
    floor = max(float(min_eigenvalue), 1.0e-15)
    Hmm_reg = Hmm + np.eye(m, dtype=np.float64) * floor
    try:
        inv_Hmm_Hmr = np.linalg.solve(Hmm_reg, Hmr)
        inv_Hmm_gm = np.linalg.solve(Hmm_reg, gm)
    except np.linalg.LinAlgError:
        return None
    precision = Hrr - Hrm @ inv_Hmm_Hmr
    linear = gr - Hrm @ inv_Hmm_gm
    precision = clamp_information_eigenvalues(
        precision,
        floor,
        max_eigenvalue=max_eigenvalue,
    )
    if not np.isfinite(precision).all() or not np.isfinite(linear).all():
        return None
    return SchurMarginalBlocks(precision=precision, linear=np.asarray(linear, dtype=np.float64))


def schur_front_block_marginalize(
    blocks: SchurMarginalBlocks,
    keep_dim: int,
    *,
    min_eigenvalue: float = 1.0e-6,
) -> SchurMarginalBlocks | None:
    """Keep the leading ``keep_dim`` states; integrate out all trailing variables."""

    m = int(keep_dim)
    P = np.asarray(blocks.precision, dtype=np.float64)
    eta = np.asarray(blocks.linear, dtype=np.float64).ravel()
    d = int(P.shape[0])
    if m <= 0 or m >= d:
        return None
    P_ff = P[:m, :m]
    P_fr = P[:m, m:]
    P_rf = P[m:, :m]
    P_rr = P[m:, m:]
    eta_f = eta[:m]
    eta_r = eta[m:]
    floor = max(float(min_eigenvalue), 1.0e-15)
    P_rr_reg = P_rr + np.eye(P_rr.shape[0], dtype=np.float64) * floor
    try:
        inv_Prr_Prf = np.linalg.solve(P_rr_reg, P_rf)
        inv_Prr_eta_r = np.linalg.solve(P_rr_reg, eta_r)
    except np.linalg.LinAlgError:
        return None
    precision = P_ff - P_fr @ inv_Prr_Prf
    linear = eta_f - P_fr @ inv_Prr_eta_r
    precision = clamp_information_eigenvalues(precision, floor)
    if not np.isfinite(precision).all() or not np.isfinite(linear).all():
        return None
    return SchurMarginalBlocks(precision=precision, linear=np.asarray(linear, dtype=np.float64))


def cap_schur_information_from_hessian(
    precision: np.ndarray,
    hessian: np.ndarray,
    n_margin: int,
    config: TcFgoConfig,
) -> np.ndarray:
    """Cap marginal eigenvalues so they cannot exceed in-window measured information."""

    m = int(n_margin)
    H = np.asarray(hessian, dtype=np.float64)
    if m <= 0 or m >= H.shape[0]:
        return np.asarray(precision, dtype=np.float64)
    Hrr = H[m:, m:]
    sym = 0.5 * (Hrr + Hrr.T)
    try:
        ref_max = float(np.max(np.linalg.eigvalsh(sym)))
    except np.linalg.LinAlgError:
        ref_max = float(np.max(np.abs(np.diag(sym))))
    cap = max(
        float(config.schur_min_eigenvalue),
        ref_max * max(float(config.schur_info_cap_ratio), 1.0e-6),
    )
    if float(config.schur_max_eigenvalue) > 0.0:
        cap = min(cap, float(config.schur_max_eigenvalue))
    return clamp_information_eigenvalues(
        precision,
        float(config.schur_min_eigenvalue),
        max_eigenvalue=cap,
    )


def schur_marginal_mean_from_state(
    x_flat: np.ndarray,
    n_margin: int,
) -> np.ndarray:
    """Remaining-state linearization point after dropping the first nav block."""

    x = np.asarray(x_flat, dtype=np.float64).ravel()
    m = int(n_margin)
    if m <= 0 or m >= x.size:
        return np.zeros(max(0, x.size - m), dtype=np.float64)
    return x[m:].copy()


def build_window_hessian_gradient(
    x_flat: np.ndarray,
    problem: TcFgoWindowProblem,
    config: TcFgoConfig,
    fgo_config: LocalFgoConfig,
    layout: TcAmbiguityLayout | None = None,
    ambiguity_bank: TcAmbiguityBank | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Normal-equation blocks at the current linearization point."""

    _cost, residuals, jacobian, _counts = _linearize_window(
        x_flat,
        problem,
        config,
        fgo_config,
        layout=layout,
        ambiguity_bank=ambiguity_bank,
    )
    if residuals.size == 0 or jacobian.size == 0:
        return None
    hessian = np.asarray(jacobian.T @ jacobian, dtype=np.float64)
    gradient = np.asarray(jacobian.T @ residuals, dtype=np.float64).ravel()
    if not (np.isfinite(hessian).all() and np.isfinite(gradient).all()):
        return None
    return hessian, gradient


def compute_schur_marginal_from_window(
    x_flat: np.ndarray,
    problem: TcFgoWindowProblem,
    config: TcFgoConfig,
    fgo_config: LocalFgoConfig,
    layout: TcAmbiguityLayout | None = None,
    ambiguity_bank: TcAmbiguityBank | None = None,
) -> TcSchurMarginal | None:
    """Schur-complement marginal after dropping the oldest epoch nav block."""

    n = len(problem.initial_states)
    if n < 2:
        return None
    sdim = state_dim(config)
    blocks = build_window_hessian_gradient(
        x_flat,
        problem,
        config,
        fgo_config,
        layout=layout,
        ambiguity_bank=ambiguity_bank,
    )
    if blocks is None:
        return None
    hessian, gradient = blocks
    margin = schur_complement_marginalize(
        hessian,
        gradient,
        sdim,
        min_eigenvalue=float(config.schur_min_eigenvalue),
    )
    if margin is None:
        return None
    mean = schur_marginal_mean_from_state(x_flat, sdim)
    precision = cap_schur_information_from_hessian(
        margin.precision,
        hessian,
        sdim,
        config,
    )
    amb_keys: list[tuple[str, str, str, str]] = []
    n_nav_epochs = int(n - 1)
    if config.schur_front_nav_only and precision.shape[0] > sdim:
        front = schur_front_block_marginalize(
            margin,
            sdim,
            min_eigenvalue=float(config.schur_min_eigenvalue),
        )
        if front is None:
            return None
        precision = cap_schur_information_from_hessian(
            front.precision,
            hessian,
            sdim,
            config,
        )
        mean = mean[:sdim]
        amb_keys = []
        n_nav_epochs = 1
    elif layout is not None and layout.n_amb > 0 and not config.schur_front_nav_only:
        seen: set[tuple[str, str, str, str]] = set()
        for _key, pair_key in sorted(layout.pair_key_map.items(), key=lambda item: item[0]):
            if pair_key in seen:
                continue
            seen.add(pair_key)
            amb_keys.append(pair_key)
    return TcSchurMarginal(
        mean=np.asarray(mean, dtype=np.float64).copy(),
        precision=np.asarray(precision, dtype=np.float64).copy(),
        n_nav_epochs=int(n_nav_epochs),
        sdim=int(sdim),
        amb_pair_keys=amb_keys,
    )


def _schur_overlap_indices(
    schur: TcSchurMarginal,
    n_epochs: int,
    layout: TcAmbiguityLayout | None,
    config: TcFgoConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Map prior vector indices -> current flat LM indices for overlapping states."""

    sdim = int(schur.sdim)
    nav_overlap = min(int(schur.n_nav_epochs), max(0, int(n_epochs) - 1))
    prior_idx: list[int] = []
    x_idx: list[int] = []
    for ep in range(nav_overlap):
        for dim in range(sdim):
            prior_idx.append(ep * sdim + dim)
            x_idx.append(ep * sdim + dim)
    nav_base = _window_nav_dim(n_epochs, config)
    if layout is not None and schur.amb_pair_keys and not config.schur_front_nav_only:
        amb_offset = int(schur.n_nav_epochs) * sdim
        for amb_i, pair_key in enumerate(schur.amb_pair_keys):
            prior_amb = amb_offset + amb_i
            if prior_amb >= int(schur.mean.size):
                break
            for (local_i, row), amb_idx in layout.index_map.items():
                if layout.pair_key_map.get((local_i, row)) == pair_key:
                    x_idx.append(nav_base + int(amb_idx))
                    prior_idx.append(prior_amb)
                    break
    return (
        np.asarray(x_idx, dtype=np.int64),
        np.asarray(prior_idx, dtype=np.int64),
    )


def is_static_epoch(speed_mps: float, thresh: float = 1.0) -> bool:
    return float(speed_mps) < float(thresh)
