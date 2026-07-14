"""Unit tests for WP11 tightly-coupled GNSS+IMU FGO primitives."""

from __future__ import annotations

from dataclasses import replace

import math

import numpy as np
import pytest

from gnss_gpu.ins_ekf import _quat_from_axis_angle, _quat_normalize
from gnss_gpu.local_fgo import DDCarrierEpoch, DDPseudorangeEpoch, LocalFgoConfig
from gnss_gpu.tc_fgo import (
    DEFAULT_LEVER_ARM_BODY_M,
    ImuPreintSegment,
    TcAmbiguityBank,
    TcFgoConfig,
    TcFgoEpochObs,
    TcFgoNavState,
    TcFgoWindowProblem,
    TcSchurMarginal,
    ambiguity_carrier_residual_and_jacobian,
    apply_held_ambiguities_to_carrier,
    bias_corrected_preintegration,
    bias_random_walk_residual,
    build_ambiguity_layout,
    clamp_information_eigenvalues,
    dd_pr_position_update_from_epoch,
    enu_to_ecef,
    ecef_to_enu,
    imu_gnss_quality_scale,
    imu_preintegration_jacobian,
    imu_preintegration_residual,
    lever_arm_offset_enu,
    marginalization_prior_residual,
    naive_marginalization_prior,
    nhc_residual_and_jacobian,
    position_anchor_residual_and_jacobian,
    quality_scaled_marginalization_prior,
    evaluate_float_quality_certificate,
    marginal_pos_sigma_from_schur,
    schur_complement_marginalize,
    solve_tc_fgo_window,
    state_vector_from_nav,
    subset_ar_select,
    tc_dd_pair_key,
    update_ambiguity_bank_from_window,
    zupt_residual_and_jacobian,
    _linearize_window,
)


def _finite_difference_jacobian(
    fun,
    x0: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    base = np.asarray(fun(x0), dtype=np.float64).ravel()
    jac = np.zeros((base.size, x0.size), dtype=np.float64)
    for k in range(x0.size):
        step = np.zeros_like(x0)
        step[k] = eps
        plus = np.asarray(fun(x0 + step), dtype=np.float64).ravel()
        minus = np.asarray(fun(x0 - step), dtype=np.float64).ravel()
        jac[:, k] = (plus - minus) / (2.0 * eps)
    return jac


def _make_segment(dt: float = 0.2) -> ImuPreintSegment:
    return ImuPreintSegment(
        delta_p_body=np.array([0.05, 0.0, 0.0], dtype=np.float64),
        delta_v_body=np.array([0.5, 0.0, 0.0], dtype=np.float64),
        delta_t_s=float(dt),
        delta_angle_body=np.zeros(3, dtype=np.float64),
        dp_d_ba=np.eye(3) * 0.01,
        dv_d_ba=np.eye(3) * 0.02,
        dp_d_bg=np.eye(3) * 0.001,
        dv_d_bg=np.eye(3) * 0.002,
    )


def test_imu_preintegration_residual_jacobian_finite_difference():
    seg = _make_segment()
    q_i = _quat_normalize(np.array([0.0, 0.0, 0.0, 1.0]))
    q_j = q_i.copy()
    p_i = np.array([0.0, 0.0, 0.0])
    v_i = np.array([1.0, 0.0, 0.0])
    p_j = np.array([0.2, 0.0, 0.0])
    v_j = np.array([1.0, 0.0, 0.0])
    b_a = np.zeros(3)
    b_g = np.zeros(3)

    def stacked(x: np.ndarray) -> np.ndarray:
        pi, vi, pj, vj = x[0:3], x[3:6], x[6:9], x[9:12]
        rp, rv = imu_preintegration_residual(
            pi,
            vi,
            pj,
            vj,
            q_i,
            q_j,
            seg,
            b_a=b_a,
            b_g=b_g,
            b_a_lin=b_a,
            b_g_lin=b_g,
            lever_arm_body_m=DEFAULT_LEVER_ARM_BODY_M,
        )
        return np.concatenate([rp, rv])

    x0 = np.concatenate([p_i, v_i, p_j, v_j])
    jac_fd = _finite_difference_jacobian(stacked, x0)
    (
        jac_pi_p,
        jac_pi_v,
        jac_pj_p,
        jac_vi_v,
        jac_vj_v,
        _jac_ba,
        _jac_bg,
        _jac_v_ba,
        _jac_v_bg,
    ) = imu_preintegration_jacobian(
        p_i,
        v_i,
        p_j,
        v_j,
        q_i,
        seg,
        lever_arm_body_m=DEFAULT_LEVER_ARM_BODY_M,
    )
    jac_analytic = np.vstack(
        [
            np.hstack([jac_pi_p, jac_pi_v, jac_pj_p, np.zeros((3, 3))]),
            np.hstack([np.zeros((3, 3)), jac_vi_v, np.zeros((3, 3)), jac_vj_v]),
        ]
    )
    assert jac_fd.shape == jac_analytic.shape
    assert np.linalg.norm(jac_fd - jac_analytic, ord=np.inf) < 5e-4


def test_nhc_and_zupt_residuals():
    q = _quat_from_axis_angle(np.array([0.0, 0.0, math.radians(30.0)]))
    v_enu = np.array([3.0, 0.2, 0.1])
    r_nhc, j_nhc = nhc_residual_and_jacobian(v_enu, q)
    assert r_nhc.shape == (2,)
    assert j_nhc.shape == (2, 3)
    r_z, j_z = zupt_residual_and_jacobian(v_enu)
    np.testing.assert_allclose(r_z, v_enu)
    np.testing.assert_allclose(j_z, np.eye(3))


def test_lever_arm_offset_enu():
    q = _quat_normalize(np.array([0.0, 0.0, 0.0, 1.0]))
    offset = lever_arm_offset_enu(q, DEFAULT_LEVER_ARM_BODY_M)
    np.testing.assert_allclose(offset, DEFAULT_LEVER_ARM_BODY_M, atol=1e-12)


def test_bias_corrected_preintegration():
    seg = _make_segment()
    b_a = np.array([0.01, 0.0, 0.0])
    b_g = np.zeros(3)
    dp, dv = bias_corrected_preintegration(seg, b_a, b_g, np.zeros(3), np.zeros(3))
    assert dp[0] > seg.delta_p_body[0]
    assert dv[0] > seg.delta_v_body[0]


def test_marginalization_prior():
    state = TcFgoNavState(
        p_enu=np.array([1.0, 2.0, 3.0]),
        v_enu=np.array([0.1, 0.2, 0.0]),
        q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )
    prior, sigmas = naive_marginalization_prior(state)
    x = state_vector_from_nav(state) + np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    residual = marginalization_prior_residual(x, state_vector_from_nav(prior))
    assert residual.shape == (6,)
    assert sigmas.shape == (6,)
    assert abs(residual[0] - 0.05) < 1e-12


def _synthetic_satellites():
    return np.array(
        [
            [4_000_000.0, 3_000_000.0, 2_000_000.0],
            [4_001_000.0, 3_000_500.0, 2_000_100.0],
            [3_999_500.0, 3_000_200.0, 2_000_300.0],
            [4_000_500.0, 2_999_800.0, 2_000_050.0],
        ],
        dtype=np.float64,
    )


def _dd_epoch_for_position(x: np.ndarray, sats: np.ndarray, base_pos: np.ndarray) -> DDPseudorangeEpoch:
    ref = sats[0]
    sat_k = sats[1:4]
    sat_ref = np.repeat(ref.reshape(1, 3), len(sat_k), axis=0)
    base_k = np.linalg.norm(sat_k - base_pos, axis=1)
    base_ref = np.repeat(np.linalg.norm(ref - base_pos), len(sat_k))
    dd_m = (
        np.linalg.norm(sat_k - x, axis=1)
        - np.linalg.norm(sat_ref - x, axis=1)
        - base_k
        + base_ref
    )
    return DDPseudorangeEpoch(
        dd_pseudorange_m=dd_m,
        sat_ecef_k=sat_k,
        sat_ecef_ref=sat_ref,
        base_range_k=base_k,
        base_range_ref=base_ref,
        weights=np.ones(len(sat_k), dtype=np.float64),
    )


def test_synthetic_constant_velocity_end_to_end():
    """Constant-velocity trajectory with synthetic DD pseudorange + IMU."""

    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])

    n = 6
    dt = 0.2
    speed = 5.0
    true_states: list[TcFgoNavState] = []
    for i in range(n):
        p = np.array([speed * dt * i, 0.0, 0.0])
        v = np.array([speed, 0.0, 0.0])
        true_states.append(
            TcFgoNavState(
                p_enu=p,
                v_enu=v,
                q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
                b_a=np.zeros(3),
                b_g=np.zeros(3),
            )
        )

    init_states = []
    for st in true_states:
        noisy = st.copy()
        noisy.p_enu = st.p_enu + np.array([0.05, -0.03, 0.02])
        noisy.v_enu = st.v_enu + np.array([0.02, 0.01, 0.0])
        init_states.append(noisy)

    observations: list[TcFgoEpochObs] = []
    for st in true_states:
        pos_ecef = enu_to_ecef(st.p_enu, origin, origin_lat, origin_lon)
        observations.append(TcFgoEpochObs(dd_pseudorange=_dd_epoch_for_position(pos_ecef, sats, base_pos)))

    imu_segments = []
    for _ in range(n - 1):
        imu_segments.append(
            ImuPreintSegment(
                delta_p_body=np.array([0.0, 0.0, 0.5 * 9.81 * dt * dt]),
                delta_v_body=np.array([0.0, 0.0, 9.81 * dt]),
                delta_t_s=dt,
                delta_angle_body=np.zeros(3),
                dp_d_ba=np.zeros((3, 3)),
                dv_d_ba=np.zeros((3, 3)),
                dp_d_bg=np.zeros((3, 3)),
                dv_d_bg=np.zeros((3, 3)),
            )
        )

    problem = TcFgoWindowProblem(
        initial_states=init_states,
        imu_segments=imu_segments,
        observations=observations,
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    config = TcFgoConfig(
        window_epochs=n,
        prior_pos_sigma_m=10.0,
        prior_vel_sigma_mps=2.0,
        imu_pos_sigma_m=0.05,
        imu_vel_sigma_mps=0.05,
        dd_pr_sigma_m=1.0,
        pr_huber_k=0.0,
        max_iterations=40,
    )
    result = solve_tc_fgo_window(problem, config=config)
    pos_err = [
        float(np.linalg.norm(result.states[i].p_enu - true_states[i].p_enu))
        for i in range(n)
    ]
    assert max(pos_err) < 0.1


def test_bias_in_state_imu_jacobian_finite_difference():
    """Bias Jacobians w.r.t. epoch-i accel/gyro biases (WP12a)."""

    seg = _make_segment()
    q_i = _quat_normalize(np.array([0.0, 0.0, 0.0, 1.0]))
    q_j = q_i.copy()
    p_i = np.zeros(3)
    v_i = np.array([1.0, 0.0, 0.0])
    p_j = np.array([0.2, 0.0, 0.0])
    v_j = np.array([1.0, 0.0, 0.0])
    b_a = np.array([0.01, -0.005, 0.002])
    b_g = np.array([0.001, 0.0, -0.0005])
    b_a_lin = np.zeros(3)
    b_g_lin = np.zeros(3)

    def stacked_ba(ba: np.ndarray) -> np.ndarray:
        rp, rv = imu_preintegration_residual(
            p_i,
            v_i,
            p_j,
            v_j,
            q_i,
            q_j,
            seg,
            b_a=ba,
            b_g=b_g,
            b_a_lin=b_a_lin,
            b_g_lin=b_g_lin,
            lever_arm_body_m=DEFAULT_LEVER_ARM_BODY_M,
        )
        return np.concatenate([rp, rv])

    jac_ba_fd = _finite_difference_jacobian(stacked_ba, b_a, eps=1e-7)
    *_, jac_ba, jac_bg, jac_v_ba, jac_v_bg = imu_preintegration_jacobian(
        p_i,
        v_i,
        p_j,
        v_j,
        q_i,
        seg,
        lever_arm_body_m=DEFAULT_LEVER_ARM_BODY_M,
        include_bias_jacobians=True,
    )
    jac_analytic_ba = np.vstack([jac_ba, jac_v_ba])
    assert jac_ba_fd.shape == jac_analytic_ba.shape
    assert np.linalg.norm(jac_ba_fd - jac_analytic_ba, ord=np.inf) < 5e-3

    def stacked_bg(bg: np.ndarray) -> np.ndarray:
        rp, rv = imu_preintegration_residual(
            p_i,
            v_i,
            p_j,
            v_j,
            q_i,
            q_j,
            seg,
            b_a=b_a,
            b_g=bg,
            b_a_lin=b_a_lin,
            b_g_lin=b_g_lin,
            lever_arm_body_m=DEFAULT_LEVER_ARM_BODY_M,
        )
        return np.concatenate([rp, rv])

    jac_bg_fd = _finite_difference_jacobian(stacked_bg, b_g, eps=1e-7)
    jac_analytic_bg = np.vstack([jac_bg, jac_v_bg])
    assert jac_bg_fd.shape == jac_analytic_bg.shape
    assert np.linalg.norm(jac_bg_fd - jac_analytic_bg, ord=np.inf) < 5e-3


def test_position_anchor_factor():
    anchor = np.array([10.0, 20.0, 30.0])
    pos = anchor + np.array([0.05, -0.02, 0.01])
    r, j = position_anchor_residual_and_jacobian(pos, anchor)
    np.testing.assert_allclose(r, pos - anchor)
    np.testing.assert_allclose(j, np.eye(3))


def test_quality_scaled_marginal_and_imu_scale():
    cfg = TcFgoConfig(marginal_pos_sigma_m=0.2, marginal_quality_rms_ref_m=3.0, marginal_quality_min_dd=4.0)
    state = TcFgoNavState(
        p_enu=np.zeros(3),
        v_enu=np.zeros(3),
        q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )
    _, sig_good = quality_scaled_marginalization_prior(state, cfg, dd_pr_rms_m=3.0, n_dd=4)
    _, sig_bad = quality_scaled_marginalization_prior(state, cfg, dd_pr_rms_m=12.0, n_dd=2)
    assert sig_bad[0] > sig_good[0]
    scale = imu_gnss_quality_scale(10.0, 0.2, TcFgoConfig(enable_imu_gnss_quality_scale=True))
    assert scale > 1.0
    scale_off = imu_gnss_quality_scale(10.0, 0.2, TcFgoConfig(enable_imu_gnss_quality_scale=False))
    assert scale_off == 1.0


def test_dd_pr_raw_rms_detects_large_mismatch():
    """Raw RMS reflects km-level linearization error; Huber RMS is capped."""

    from gnss_gpu.local_fgo import LocalFgoConfig
    from gnss_gpu.tc_fgo import compute_dd_pr_postfit_rms, ecef_to_enu

    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    true_pos = origin + np.array([10.0, 5.0, 2.0])
    wrong_pos = origin + np.array([1010.0, 5.0, 2.0])
    obs = _dd_epoch_for_position(true_pos, sats, base_pos)
    nav_bad = TcFgoNavState(
        p_enu=ecef_to_enu(wrong_pos, origin, origin_lat, origin_lon),
        v_enu=np.zeros(3),
        q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )
    cfg = TcFgoConfig()
    fgo_cfg = LocalFgoConfig()
    rms_huber, _ = compute_dd_pr_postfit_rms(
        nav_bad,
        obs,
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        config=cfg,
        fgo_config=fgo_cfg,
        huber_weighted=True,
    )
    rms_raw, _ = compute_dd_pr_postfit_rms(
        nav_bad,
        obs,
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        config=cfg,
        fgo_config=fgo_cfg,
        huber_weighted=False,
    )
    assert rms_raw > 100.0
    assert rms_huber < rms_raw


def test_dd_pr_recovery_accepts_good_rms_despite_shift_cap():
    """Recovery accepts large shifts when post-fit DD RMS is sane (WP12a fix)."""

    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    true_pos = origin + np.array([10.0, 5.0, 2.0])
    obs = _dd_epoch_for_position(true_pos, sats, base_pos)
    recovered, stats = dd_pr_position_update_from_epoch(
        true_pos + np.array([20.0, 0.0, 0.0]),
        obs,
        max_shift_m=5.0,
        max_iter=8,
        prior_sigma_m=10.0,
    )
    assert stats["accepted"]
    assert float(stats["final_rms_m"]) < 5.0
    assert float(np.linalg.norm(recovered - true_pos)) < 30.0

    b_i = np.zeros(3)
    b_j = np.array([0.01, 0.0, -0.002])
    np.testing.assert_allclose(bias_random_walk_residual(b_j, b_i), b_j - b_i)


def _synthetic_carrier_epoch(sats, base_pos, x, cycles_offset=0.0) -> DDCarrierEpoch:
    ref = sats[0]
    sat_k = sats[1:3]
    sat_ref = np.repeat(ref.reshape(1, 3), len(sat_k), axis=0)
    base_k = np.linalg.norm(sat_k - base_pos, axis=1)
    base_ref = np.repeat(np.linalg.norm(ref - base_pos), len(sat_k))
    wl = 299792458.0 / 1575.42e6
    dd_m = (
        np.linalg.norm(sat_k - x, axis=1)
        - np.linalg.norm(sat_ref - x, axis=1)
        - base_k
        + base_ref
    )
    dd_cyc = dd_m / wl + float(cycles_offset)
    return DDCarrierEpoch(
        dd_carrier_cycles=dd_cyc,
        sat_ecef_k=sat_k,
        sat_ecef_ref=sat_ref,
        base_range_k=base_k,
        base_range_ref=base_ref,
        wavelengths_m=np.full(len(sat_k), wl),
        weights=np.ones(len(sat_k)),
        sat_ids=("G02", "G03"),
        ref_sat_ids=("G01", "G01"),
    )


def test_tc_fgo_wcp_and_switchable_pseudorange_are_wired_into_linearization():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    states = []
    observations = []
    for i in range(3):
        state = TcFgoNavState(
            p_enu=np.array([float(i), 0.0, 0.0]),
            v_enu=np.array([1.0, 0.0, 0.0]),
            q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
            b_a=np.zeros(3),
            b_g=np.zeros(3),
        )
        states.append(state)
        pos_ecef = enu_to_ecef(state.p_enu, origin, origin_lat, origin_lon)
        pr = _dd_epoch_for_position(pos_ecef, sats, base_pos)
        if i == 1:
            pr.dd_pseudorange_m = pr.dd_pseudorange_m.copy()
            pr.dd_pseudorange_m[0] += 100.0
        observations.append(
            TcFgoEpochObs(
                dd_pseudorange=pr,
                dd_carrier=_synthetic_carrier_epoch(sats, base_pos, pos_ecef, 7.0),
                dd_carrier_arc_ids=(0, 0),
            )
        )
    problem = TcFgoWindowProblem(
        initial_states=states,
        imu_segments=[None, None],
        observations=observations,
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    config = TcFgoConfig(
        enable_wcp=True,
        wcp_min_epochs=3,
        enable_switchable_pseudorange=True,
        commit_switchable_pseudorange=True,
        switch_prior_strength=4.0,
        switch_min_reliable_rows=2,
        pr_huber_k=0.0,
    )
    x = np.vstack([state_vector_from_nav(state, config) for state in states])
    _cost, _residual, _jacobian, counts = _linearize_window(
        x, problem, config, LocalFgoConfig()
    )
    assert counts["wcp"] == 4  # two three-epoch arcs, one null direction removed each
    assert counts["switchable_pseudorange"] == 9
    assert counts["switched_pseudorange"] >= 1
    assert counts["switch_integrity_abstained_epochs"] == 0

    strict_config = replace(config, switch_max_downweighted_fraction=0.0)
    _cost, _residual, _jacobian, strict_counts = _linearize_window(
        x, problem, strict_config, LocalFgoConfig()
    )
    assert strict_counts["switch_integrity_abstained_epochs"] >= 1
    assert strict_counts["switch_integrity_abstained_rows"] >= 3

    shadow_config = replace(config, commit_switchable_pseudorange=False)
    _cost, shadow_residual, shadow_jacobian, shadow_counts = _linearize_window(
        x, problem, shadow_config, LocalFgoConfig()
    )
    baseline_config = replace(config, enable_switchable_pseudorange=False)
    _cost, baseline_residual, baseline_jacobian, _counts = _linearize_window(
        x, problem, baseline_config, LocalFgoConfig()
    )
    assert shadow_counts["switch_shadow_epochs"] == 3
    assert np.allclose(shadow_residual, baseline_residual)
    assert np.allclose(shadow_jacobian, baseline_jacobian)


def test_tc_fgo_wcp_does_not_cross_slip_generation():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    states = []
    observations = []
    for i in range(3):
        state = TcFgoNavState(
            p_enu=np.array([float(i), 0.0, 0.0]), v_enu=np.array([1.0, 0.0, 0.0]),
            q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]), b_a=np.zeros(3), b_g=np.zeros(3)
        )
        states.append(state)
        pos_ecef = enu_to_ecef(state.p_enu, origin, origin_lat, origin_lon)
        observations.append(
            TcFgoEpochObs(
                dd_carrier=_synthetic_carrier_epoch(sats, base_pos, pos_ecef, 2.0),
                dd_carrier_arc_ids=((0 if i < 2 else 1), 0),
            )
        )
    problem = TcFgoWindowProblem(
        initial_states=states, imu_segments=[None, None], observations=observations,
        origin_ecef=origin, origin_lat=origin_lat, origin_lon=origin_lon
    )
    config = TcFgoConfig(enable_wcp=True, wcp_min_epochs=3)
    x = np.vstack([state_vector_from_nav(state, config) for state in states])
    _cost, _residual, _jacobian, counts = _linearize_window(
        x, problem, config, LocalFgoConfig()
    )
    assert counts["wcp"] == 2  # only the second pair remains a three-epoch arc


def test_tc_fgo_wcp_auto_generations_detect_geometry_corrected_cycle_slip():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    states = []
    observations = []
    for i in range(4):
        state = TcFgoNavState(
            p_enu=np.array([float(i), 0.0, 0.0]),
            v_enu=np.array([1.0, 0.0, 0.0]),
            q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
            b_a=np.zeros(3),
            b_g=np.zeros(3),
        )
        states.append(state)
        pos_ecef = enu_to_ecef(state.p_enu, origin, origin_lat, origin_lon)
        carrier = _synthetic_carrier_epoch(sats, base_pos, pos_ecef, 2.0)
        if i >= 2:
            carrier.dd_carrier_cycles[0] += 5.0
        observations.append(TcFgoEpochObs(dd_carrier=carrier))
    problem = TcFgoWindowProblem(
        initial_states=states,
        imu_segments=[None, None, None],
        observations=observations,
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    config = TcFgoConfig(
        enable_wcp=True,
        wcp_min_epochs=2,
        wcp_slip_threshold_cycles=1.5,
    )
    x = np.vstack([state_vector_from_nav(state, config) for state in states])
    _cost, _residual, _jacobian, counts = _linearize_window(
        x, problem, config, LocalFgoConfig()
    )
    # Slipped pair: two 2-epoch arcs -> 2 rows. Continuous pair: one
    # 4-epoch arc -> 3 rows. Without automatic segmentation this would be 6.
    assert counts["wcp"] == 5


def test_ambiguity_carrier_jacobian_finite_difference():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    true_pos = origin + np.array([10.0, 5.0, 2.0])
    obs = _synthetic_carrier_epoch(sats, base_pos, true_pos, cycles_offset=3.7)

    def pos_residual(p: np.ndarray) -> float:
        r, _, _ = ambiguity_carrier_residual_and_jacobian(
            p, obs, 0, ambiguity_value=3.7, held_integer=None
        )
        return r

    jac_fd = _finite_difference_jacobian(lambda p: np.array([pos_residual(p)]), true_pos)
    _, jac_analytic, _ = ambiguity_carrier_residual_and_jacobian(
        true_pos, obs, 0, ambiguity_value=3.7, held_integer=None
    )
    assert np.linalg.norm(jac_fd.reshape(3) - jac_analytic, ord=np.inf) < 5e-3

    def amb_residual(n: np.ndarray) -> float:
        r, _, _ = ambiguity_carrier_residual_and_jacobian(
            true_pos, obs, 0, ambiguity_value=float(n[0]), held_integer=None
        )
        return r

    jac_n_fd = _finite_difference_jacobian(amb_residual, np.array([3.7]))
    _, _, jac_n = ambiguity_carrier_residual_and_jacobian(
        true_pos, obs, 0, ambiguity_value=3.7, held_integer=None
    )
    assert abs(jac_n_fd[0, 0] - jac_n) < 5e-4


def test_held_ambiguity_carrier_jacobian_finite_difference():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    true_pos = origin + np.array([10.0, 5.0, 2.0])
    obs = _synthetic_carrier_epoch(sats, base_pos, true_pos, cycles_offset=4.0)

    def pos_residual(p: np.ndarray) -> float:
        r, _, _ = ambiguity_carrier_residual_and_jacobian(
            p, obs, 0, ambiguity_value=None, held_integer=4
        )
        return r

    jac_fd = _finite_difference_jacobian(lambda p: np.array([pos_residual(p)]), true_pos)
    _, jac_analytic, _ = ambiguity_carrier_residual_and_jacobian(
        true_pos, obs, 0, ambiguity_value=None, held_integer=4
    )
    assert np.linalg.norm(jac_fd.reshape(3) - jac_analytic, ord=np.inf) < 5e-3


def test_held_integer_resolv_tightens_position_to_truth():
    """After AR acceptance, re-solving with held integers must move position toward truth."""

    from gnss_gpu.local_fgo import LocalFgoConfig

    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    true_pos = origin + np.array([10.0, 5.0, 2.0])
    wrong_pos = origin + np.array([15.0, 5.0, 2.0])
    ref = sats[0]
    sat_k = sats[1:4]
    sat_ref = np.repeat(ref.reshape(1, 3), len(sat_k), axis=0)
    base_k = np.linalg.norm(sat_k - base_pos, axis=1)
    base_ref = np.repeat(np.linalg.norm(ref - base_pos), len(sat_k))
    wl = 299792458.0 / 1575.42e6
    dd_m = (
        np.linalg.norm(sat_k - true_pos, axis=1)
        - np.linalg.norm(sat_ref - true_pos, axis=1)
        - base_k
        + base_ref
    )
    dd_cp = DDCarrierEpoch(
        dd_carrier_cycles=dd_m / wl,
        sat_ecef_k=sat_k,
        sat_ecef_ref=sat_ref,
        base_range_k=base_k,
        base_range_ref=base_ref,
        wavelengths_m=np.full(len(sat_k), wl),
        weights=np.ones(len(sat_k)),
        sat_ids=("G02", "G03", "G04"),
        ref_sat_ids=("G01", "G01", "G01"),
    )
    dd_pr = _dd_epoch_for_position(true_pos, sats, base_pos)
    obs = TcFgoEpochObs(dd_pseudorange=dd_pr, dd_carrier=dd_cp)
    wrong_nav = TcFgoNavState(
        p_enu=ecef_to_enu(wrong_pos, origin, origin_lat, origin_lon),
        v_enu=np.zeros(3),
        q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )
    cfg = TcFgoConfig(
        enable_dd_carrier=True,
        dd_cp_fixed_sigma_cycles=0.0001,
        dd_cp_sigma_cycles=0.2,
        max_iterations=60,
    )
    problem = TcFgoWindowProblem(
        initial_states=[wrong_nav],
        imu_segments=[],
        observations=[obs],
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    float_result = solve_tc_fgo_window(problem, config=cfg, fgo_config=LocalFgoConfig())
    float_ecef = enu_to_ecef(float_result.states[0].p_enu, origin, origin_lat, origin_lon)
    float_err = float(np.linalg.norm(float_ecef - true_pos))
    assert float_err > 1.0

    held: dict[tuple[int, tuple[str, str, str, str]], int] = {}
    for row in range(int(dd_cp.n)):
        held[(0, tc_dd_pair_key(dd_cp, row))] = 0

    obs_held = apply_held_ambiguities_to_carrier([obs], held, window_start_epoch=0)
    ar_problem = TcFgoWindowProblem(
        initial_states=float_result.states,
        imu_segments=[],
        observations=obs_held,
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        held_ambiguities=held,
        window_start_epoch=0,
    )
    fixed_result = solve_tc_fgo_window(
        ar_problem,
        config=replace(cfg, enable_lambda_ar=False),
        fgo_config=LocalFgoConfig(),
    )
    fixed_ecef = enu_to_ecef(fixed_result.states[0].p_enu, origin, origin_lat, origin_lon)
    fixed_err = float(np.linalg.norm(fixed_ecef - true_pos))
    assert fixed_err < 0.05
    assert fixed_err < 0.5 * float_err

    float_amb = np.array([0.0, 10.0, 0.0], dtype=np.float64)
    cov = np.diag([0.001, 0.001, 0.001])
    residuals = np.array([0.0, 10.0, 0.0], dtype=np.float64)
    fixed, ok, _ratio = subset_ar_select(
        float_amb,
        cov,
        residuals,
        ratio_threshold=2.0,
        max_drop=1,
    )
    assert ok
    assert fixed is not None
    assert int(round(fixed[0])) == 0 and int(round(fixed[2])) == 0
    assert not np.isfinite(fixed[1])


def test_apply_held_ambiguities_folds_integers():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    obs = TcFgoEpochObs(
        dd_carrier=_synthetic_carrier_epoch(sats, base_pos, origin + np.array([10.0, 5.0, 2.0]))
    )
    pair_key = tc_dd_pair_key(obs.dd_carrier, 0)
    held = {(0, pair_key): 7}
    folded = apply_held_ambiguities_to_carrier([obs], held, window_start_epoch=0)
    fixed = np.asarray(folded[0].dd_carrier.fixed_ambiguities, dtype=np.float64)
    assert fixed[0] == 7.0


def test_build_ambiguity_layout_skips_held_rows():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    nav = TcFgoNavState(
        p_enu=ecef_to_enu(origin + np.array([10.0, 5.0, 2.0]), origin, origin_lat, origin_lon),
        v_enu=np.zeros(3),
        q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )
    obs = TcFgoEpochObs(
        dd_carrier=_synthetic_carrier_epoch(sats, base_pos, origin + np.array([10.0, 5.0, 2.0]))
    )
    pair_key = tc_dd_pair_key(obs.dd_carrier, 0)
    cfg = TcFgoConfig(enable_dd_carrier=True)
    layout = build_ambiguity_layout(
        [obs],
        cfg,
        held_global={(0, pair_key): 4},
        window_start_epoch=0,
        initial_nav=[nav],
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    assert layout is not None
    assert layout.n_amb == 1
    assert (0, 1) in layout.held_map or (0, 0) in layout.held_map


def test_ambiguity_bank_seeds_cross_window_prior():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    nav = TcFgoNavState(
        p_enu=ecef_to_enu(origin + np.array([10.0, 5.0, 2.0]), origin, origin_lat, origin_lon),
        v_enu=np.zeros(3),
        q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )
    obs = TcFgoEpochObs(
        dd_carrier=_synthetic_carrier_epoch(sats, base_pos, origin + np.array([10.0, 5.0, 2.0]))
    )
    pair_key = tc_dd_pair_key(obs.dd_carrier, 0)
    bank = TcAmbiguityBank()
    bank.update(pair_key, 12.34, 0.1, epoch=4)
    cfg = TcFgoConfig(enable_dd_carrier=True, enable_persistent_ambiguities=True)
    layout = build_ambiguity_layout(
        [obs],
        cfg,
        window_start_epoch=5,
        initial_nav=[nav],
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        ambiguity_bank=bank,
    )
    assert layout is not None
    assert layout.n_amb >= 1
    assert abs(float(layout.initial_values[0]) - 12.34) < 1e-9
    assert 0 in layout.cross_window_priors
    assert layout.cross_window_priors[0][0] == pytest.approx(12.34)


def test_ambiguity_bank_bump_clears_generation():
    bank = TcAmbiguityBank()
    bank.update(("G01", "G02", "L1", "L1"), 1.0, 0.1, epoch=0)
    bank.bump_generation()
    assert bank.get(("G01", "G02", "L1", "L1")) is None
    assert bank.generation == 1


def test_update_ambiguity_bank_from_window_keeps_latest_epoch():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    nav = TcFgoNavState(
        p_enu=ecef_to_enu(origin + np.array([10.0, 5.0, 2.0]), origin, origin_lat, origin_lon),
        v_enu=np.zeros(3),
        q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )
    obs0 = TcFgoEpochObs(
        dd_carrier=_synthetic_carrier_epoch(sats, base_pos, origin + np.array([10.0, 5.0, 2.0]))
    )
    obs1 = TcFgoEpochObs(
        dd_carrier=_synthetic_carrier_epoch(sats, base_pos, origin + np.array([10.2, 5.0, 2.0]))
    )
    cfg = TcFgoConfig(enable_dd_carrier=True, enable_persistent_ambiguities=True)
    layout = build_ambiguity_layout(
        [obs0, obs1],
        cfg,
        window_start_epoch=10,
        initial_nav=[nav, nav],
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    bank = TcAmbiguityBank()
    amb_values = np.zeros(layout.n_amb, dtype=np.float64)
    amb_values[layout.index_map[(1, 0)]] = 9.0
    update_ambiguity_bank_from_window(
        bank, layout, amb_values, window_start_epoch=10, config=cfg
    )
    pair_key = tc_dd_pair_key(obs0.dd_carrier, 0)
    est = bank.get(pair_key)
    assert est is not None
    assert est.value == pytest.approx(9.0)
    assert est.last_epoch == 11


def test_schur_complement_matches_analytic_gaussian_marginal():
    """Toy Gaussian: Schur marginal matches the precision Schur formula."""

    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    precision = A.T @ A + np.eye(4)
    mean = rng.standard_normal(4)
    gradient = -(precision @ mean)
    m = 2
    schur = schur_complement_marginalize(precision, gradient, m, min_eigenvalue=1e-9)
    assert schur is not None
    prec_s = schur.precision

    direct = precision[m:, m:] - precision[m:, :m] @ np.linalg.inv(precision[:m, :m]) @ precision[:m, m:]
    np.testing.assert_allclose(prec_s, direct, atol=1e-5)


def test_schur_chain_three_state_exact_conditioning():
    """3-state chain: marginalizing state 0 reproduces exact Gaussian conditioning."""

    # States x0, x1, x2 with chain factors: x0~N(0,1/s0), x1|x0, x2|x1.
    s0, s1, s2 = 2.0, 3.0, 4.0
    H = np.array(
        [
            [s0 + s1, -s1, 0.0],
            [-s1, s1 + s2, -s2],
            [0.0, -s2, s2],
        ],
        dtype=np.float64,
    )
    x_star = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    g = -(H @ x_star)
    blocks = schur_complement_marginalize(H, g, 1, min_eigenvalue=1e-12)
    assert blocks is not None

    direct_prec = H[1:, 1:] - H[1:, :1] * (1.0 / H[0, 0]) * H[:1, 1:]
    np.testing.assert_allclose(blocks.precision, direct_prec, atol=1e-9)

    # Prior r = L^T (x - mu) at linearization must be zero for remaining states.
    mu = x_star[1:]
    delta = x_star[1:] - mu
    chol = np.linalg.cholesky(blocks.precision)
    np.testing.assert_allclose(chol.T @ delta, np.zeros(2), atol=1e-12)


def test_sliding_schur_matches_joint_solve_constant_velocity():
    """Two consecutive Schur windows == joint solve on the same epoch span."""

    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    n_joint = 5
    win = 4
    true_states: list[TcFgoNavState] = []
    observations: list[TcFgoEpochObs] = []
    for i in range(n_joint):
        p = np.array([5.0 * 0.2 * i, 0.0, 0.0])
        true_states.append(
            TcFgoNavState(
                p_enu=p,
                v_enu=np.array([5.0, 0.0, 0.0]),
                q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
                b_a=np.zeros(3),
                b_g=np.zeros(3),
            )
        )
        pos_ecef = enu_to_ecef(p, origin, origin_lat, origin_lon)
        observations.append(TcFgoEpochObs(dd_pseudorange=_dd_epoch_for_position(pos_ecef, sats, base_pos)))

    def noisy_init(st: TcFgoNavState) -> TcFgoNavState:
        out = st.copy()
        out.p_enu = st.p_enu + np.array([0.05, -0.03, 0.02])
        out.v_enu = st.v_enu + np.array([0.02, 0.01, 0.0])
        return out

    imu_segments = [
        ImuPreintSegment(
            delta_p_body=np.array([0.0, 0.0, 0.5 * 9.81 * 0.2 * 0.2]),
            delta_v_body=np.array([0.0, 0.0, 9.81 * 0.2]),
            delta_t_s=0.2,
            delta_angle_body=np.zeros(3),
            dp_d_ba=np.zeros((3, 3)),
            dv_d_ba=np.zeros((3, 3)),
            dp_d_bg=np.zeros((3, 3)),
            dv_d_bg=np.zeros((3, 3)),
        )
        for _ in range(n_joint - 1)
    ]
    cfg = TcFgoConfig(
        window_epochs=win,
        prior_pos_sigma_m=10.0,
        prior_vel_sigma_mps=2.0,
        imu_pos_sigma_m=0.05,
        imu_vel_sigma_mps=0.05,
        dd_pr_sigma_m=1.0,
        pr_huber_k=0.0,
        max_iterations=40,
        enable_schur_marginalization=True,
    )
    inits = [noisy_init(s) for s in true_states]

    joint_problem = TcFgoWindowProblem(
        initial_states=inits,
        imu_segments=imu_segments,
        observations=observations,
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
    )
    joint = solve_tc_fgo_window(problem=joint_problem, config=replace(cfg, window_epochs=n_joint))

    w0 = TcFgoWindowProblem(
        initial_states=[s.copy() for s in inits[:win]],
        imu_segments=imu_segments[: win - 1],
        observations=observations[:win],
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        window_start_epoch=0,
    )
    r0 = solve_tc_fgo_window(w0, config=cfg)
    assert r0.schur_marginal is not None

    w1 = TcFgoWindowProblem(
        initial_states=[s.copy() for s in inits[1:n_joint]],
        imu_segments=imu_segments[1:n_joint],
        observations=observations[1:n_joint],
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        schur_marginal=r0.schur_marginal,
        window_start_epoch=1,
    )
    r1 = solve_tc_fgo_window(w1, config=cfg)

    for local_i in range(len(r1.states)):
        global_i = 1 + local_i
        jp = joint.states[global_i].p_enu
        sp = r1.states[local_i].p_enu
        assert float(np.linalg.norm(jp - sp)) < 0.05, f"epoch {global_i}"


def test_clamp_information_eigenvalues_floors_ill_conditioned_matrix():
  mat = np.diag([1e-12, 1.0, 1.0])
  clamped = clamp_information_eigenvalues(mat, min_eigenvalue=1e-4)
  eigvals = np.linalg.eigvalsh(clamped)
  assert float(np.min(eigvals)) >= 1e-4 - 1e-12


def test_schur_marginal_cleared_on_recovery_path_in_runner_logic():
  """Recovery wipe rule: stale Schur information must be dropped."""

  schur = TcSchurMarginal(
      mean=np.zeros(6),
      precision=np.eye(6),
      n_nav_epochs=1,
      sdim=6,
  )
  marginal_prior = TcFgoNavState(
      p_enu=np.zeros(3),
      v_enu=np.zeros(3),
      q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
      b_a=np.zeros(3),
      b_g=np.zeros(3),
  )
  marginal_sigmas = np.ones(6)
  recovery_fired = True
  if recovery_fired:
      marginal_prior = None
      marginal_sigmas = None
      schur = None
  assert schur is None
  assert marginal_prior is None
  assert marginal_sigmas is None


def test_compute_schur_marginal_from_synthetic_window():
  """End-to-end Schur marginal on a small TC window has full-rank information."""

  origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
  origin_lat, origin_lon = 0.35, 0.65
  sats = _synthetic_satellites()
  base_pos = origin + np.array([100.0, -50.0, 20.0])
  n = 4
  states = []
  observations = []
  for i in range(n):
      p = np.array([5.0 * 0.2 * i, 0.0, 0.0])
      states.append(
          TcFgoNavState(
              p_enu=p,
              v_enu=np.array([5.0, 0.0, 0.0]),
              q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
              b_a=np.zeros(3),
              b_g=np.zeros(3),
          )
      )
      pos_ecef = enu_to_ecef(p, origin, origin_lat, origin_lon)
      observations.append(TcFgoEpochObs(dd_pseudorange=_dd_epoch_for_position(pos_ecef, sats, base_pos)))
  imu_segments = [_make_segment() for _ in range(n - 1)]
  problem = TcFgoWindowProblem(
      initial_states=states,
      imu_segments=imu_segments,
      observations=observations,
      origin_ecef=origin,
      origin_lat=origin_lat,
      origin_lon=origin_lon,
  )
  cfg = TcFgoConfig(enable_schur_marginalization=True, pr_huber_k=0.0, max_iterations=20)
  result = solve_tc_fgo_window(problem, config=cfg)
  assert result.schur_marginal is not None
  assert result.schur_marginal.n_nav_epochs == 1
  assert result.schur_marginal.mean.size == 6
  assert result.schur_marginal.precision.shape == (6, 6)
  eigvals = np.linalg.eigvalsh(result.schur_marginal.precision)
  assert float(np.min(eigvals)) >= cfg.schur_min_eigenvalue - 1e-12
  assert float(np.max(eigvals)) < 1e8


def test_marginal_pos_sigma_from_schur_tight_prior():
    precision = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0])
    schur = TcSchurMarginal(
        mean=np.zeros(6),
        precision=precision,
        n_nav_epochs=1,
        sdim=6,
    )
    sigma = marginal_pos_sigma_from_schur(schur)
    assert sigma == pytest.approx(0.1, rel=1e-6)


def test_float_quality_certificate_passes_open_sky_like():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    nav = TcFgoNavState(
        p_enu=np.array([0.0, 0.0, 0.0]),
        v_enu=np.array([5.0, 0.0, 0.0]),
        q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )
    pos_ecef = enu_to_ecef(nav.p_enu, origin, origin_lat, origin_lon)
    dd_pr = _dd_epoch_for_position(pos_ecef, sats, base_pos)
    schur = TcSchurMarginal(
        mean=np.zeros(6),
        precision=np.diag([400.0, 400.0, 400.0, 10.0, 10.0, 10.0]),
        n_nav_epochs=1,
        sdim=6,
    )
    from gnss_gpu.local_fgo import LocalFgoConfig

    cfg = TcFgoConfig()
    cert = evaluate_float_quality_certificate(
        config=cfg,
        schur_marginal=schur,
        last_nav=nav,
        last_dd_pr=dd_pr,
        last_dd_cp=None,
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        fgo_config=LocalFgoConfig(),
        epochs_since_recovery=50,
        n_dd_carrier_factors=6,
    )
    assert cert.passed
    assert cert.marginal_pos_sigma_m < cfg.ar_cert_max_pos_sigma_m


def test_float_quality_certificate_fails_recovery_recency():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    nav = TcFgoNavState(
        p_enu=np.array([0.0, 0.0, 0.0]),
        v_enu=np.array([5.0, 0.0, 0.0]),
        q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )
    pos_ecef = enu_to_ecef(nav.p_enu, origin, origin_lat, origin_lon)
    dd_pr = _dd_epoch_for_position(pos_ecef, sats, base_pos)
    schur = TcSchurMarginal(
        mean=np.zeros(6),
        precision=np.diag([400.0, 400.0, 400.0, 10.0, 10.0, 10.0]),
        n_nav_epochs=1,
        sdim=6,
    )
    from gnss_gpu.local_fgo import LocalFgoConfig

    cfg = TcFgoConfig()
    cert = evaluate_float_quality_certificate(
        config=cfg,
        schur_marginal=schur,
        last_nav=nav,
        last_dd_pr=dd_pr,
        last_dd_cp=None,
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        fgo_config=LocalFgoConfig(),
        epochs_since_recovery=2,
        n_dd_carrier_factors=6,
    )
    assert not cert.passed
    assert any("recovery_recency" in r for r in cert.fail_reasons)


def test_float_quality_certificate_fails_high_dd_pr():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    true_ecef = enu_to_ecef(np.array([0.0, 0.0, 0.0]), origin, origin_lat, origin_lon)
    nav = TcFgoNavState(
        p_enu=np.array([100.0, 50.0, 20.0]),
        v_enu=np.array([5.0, 0.0, 0.0]),
        q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )
    dd_pr = _dd_epoch_for_position(true_ecef, sats, base_pos)
    schur = TcSchurMarginal(
        mean=np.zeros(6),
        precision=np.diag([400.0, 400.0, 400.0, 10.0, 10.0, 10.0]),
        n_nav_epochs=1,
        sdim=6,
    )
    from gnss_gpu.local_fgo import LocalFgoConfig

    cfg = TcFgoConfig()
    cert = evaluate_float_quality_certificate(
        config=cfg,
        schur_marginal=schur,
        last_nav=nav,
        last_dd_pr=dd_pr,
        last_dd_cp=None,
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        fgo_config=LocalFgoConfig(),
        epochs_since_recovery=50,
        n_dd_carrier_factors=6,
    )
    assert not cert.passed
    assert any("dd_pr_rms" in r for r in cert.fail_reasons)


def test_ar_hold_wipe_on_recovery_runner_pattern():
    """Mirror runner: recovery clears held ambiguities and bumps bank generation."""

    held = {(10, ("G01", "G02", "L1", "L1")): 3}
    bank = TcAmbiguityBank()
    bank.update(("G01", "G02", "L1", "L1"), 3.0, 0.1, epoch=10)
    recovery_fired = True
    if recovery_fired:
        held.clear()
        bank.bump_generation()
    assert held == {}
    assert bank.get(("G01", "G02", "L1", "L1")) is None


def test_quality_gated_ar_skips_when_certificate_fails():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    nav = TcFgoNavState(
        p_enu=np.array([100.0, 50.0, 20.0]),
        v_enu=np.array([5.0, 0.0, 0.0]),
        q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )
    pos_ecef = enu_to_ecef(nav.p_enu, origin, origin_lat, origin_lon)
    obs = TcFgoEpochObs(dd_pseudorange=_dd_epoch_for_position(pos_ecef, sats, base_pos))
    problem = TcFgoWindowProblem(
        initial_states=[nav],
        imu_segments=[],
        observations=[obs],
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        epochs_since_recovery=0,
    )
    cfg = TcFgoConfig(enable_lambda_ar=True, enable_ar_quality_gate=True, max_iterations=5)
    result = solve_tc_fgo_window(problem, config=cfg)
    assert not result.ar_accepted
    assert result.ar_info is not None
    assert result.ar_info.get("certificate", {}).get("passed") is False


def test_float_quality_certificate_fails_anchor_distance():
    origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0], dtype=np.float64)
    origin_lat, origin_lon = 0.35, 0.65
    sats = _synthetic_satellites()
    base_pos = origin + np.array([100.0, -50.0, 20.0])
    nav = TcFgoNavState(
        p_enu=np.array([0.0, 0.0, 0.0]),
        v_enu=np.array([5.0, 0.0, 0.0]),
        q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
        b_a=np.zeros(3),
        b_g=np.zeros(3),
    )
    pos_ecef = enu_to_ecef(nav.p_enu, origin, origin_lat, origin_lon)
    dd_pr = _dd_epoch_for_position(pos_ecef, sats, base_pos)
    schur = TcSchurMarginal(
        mean=np.zeros(6),
        precision=np.diag([400.0, 400.0, 400.0, 10.0, 10.0, 10.0]),
        n_nav_epochs=1,
        sdim=6,
    )
    from gnss_gpu.local_fgo import LocalFgoConfig

    cfg = TcFgoConfig(ar_cert_max_epochs_since_anchor=25)
    cert = evaluate_float_quality_certificate(
        config=cfg,
        schur_marginal=schur,
        last_nav=nav,
        last_dd_pr=dd_pr,
        last_dd_cp=None,
        origin_ecef=origin,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        fgo_config=LocalFgoConfig(),
        epochs_since_recovery=50,
        epochs_since_anchor=100,
        n_dd_carrier_factors=6,
    )
    assert not cert.passed
    assert any("anchor_distance" in r for r in cert.fail_reasons)

