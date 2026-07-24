"""Unit tests for the standalone, FGO-free IMU preintegration module (WP21).

Covers:
- synthetic trajectories (constant acceleration, pure rotation, circle) with
  analytic/high-order-numeric ground truth;
- G1: cross-check against the on-manifold preintegration recursion that the
  tc_fgo stack actually consumes (``experiments/gsdc2023_imu.py``'s
  ``preintegrate_processed_imu`` with ``sample_dt_mode="taroz"``, the engine
  behind ``tc_fgo.collapse_imu_preintegration_segment`` /
  ``imu_preintegration_segment_with_bias_jacobians``), on both a synthetic
  sample stream and a real PPC Tokyo run2 IMU slice;
- bias-correction Jacobian check via finite differences.
"""

from __future__ import annotations

import numpy as np
import pytest

from gnss_gpu.imu_preintegration import (
    GRAVITY_ENU,
    PreintegratedIMU,
    preintegrate_raw,
    right_jacobian_inverse_so3,
    right_jacobian_so3,
    rotm_to_rotvec,
    rotvec_to_rotm,
)


# ---------------------------------------------------------------------------
# SO(3) helper sanity
# ---------------------------------------------------------------------------


def test_rotvec_rotm_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        phi = rng.normal(scale=0.5, size=3)
        R = rotvec_to_rotm(phi)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10
        phi_back = rotm_to_rotvec(R)
        assert np.allclose(phi_back, phi, atol=1e-8)


def test_right_jacobian_inverse_is_inverse():
    rng = np.random.default_rng(1)
    for _ in range(10):
        phi = rng.normal(scale=0.3, size=3)
        Jr = right_jacobian_so3(phi)
        Jr_inv = right_jacobian_inverse_so3(phi)
        assert np.allclose(Jr @ Jr_inv, np.eye(3), atol=1e-8)


# ---------------------------------------------------------------------------
# Synthetic trajectory 1: constant acceleration, zero rotation
# ---------------------------------------------------------------------------


def test_constant_acceleration_matches_analytic():
    n = 500
    dt = 0.01
    accel = np.tile(np.array([2.0, -1.0, 0.5]), (n, 1))
    gyro = np.zeros((n, 3))
    dts = np.full(n, dt)

    result = preintegrate_raw(accel, gyro, dts)
    T = n * dt

    # Discrete recursion sums an exact quadratic for constant acceleration
    # regardless of dt granularity -> near machine-precision agreement.
    expected_v = accel[0] * T
    expected_p = 0.5 * accel[0] * T * T
    assert np.allclose(result.delta_v, expected_v, rtol=1e-9, atol=1e-9)
    assert np.allclose(result.delta_p, expected_p, rtol=1e-9, atol=1e-9)
    assert np.allclose(result.delta_R, np.eye(3), atol=1e-12)
    assert np.allclose(result.delta_angle, 0.0, atol=1e-12)
    assert result.n_samples == n
    assert abs(result.delta_t - T) < 1e-9


# ---------------------------------------------------------------------------
# Synthetic trajectory 2: pure rotation about a fixed axis, zero acceleration
# ---------------------------------------------------------------------------


def test_pure_rotation_matches_analytic():
    n = 1000
    dt = 0.01
    w = 0.3  # rad/s about z
    gyro = np.tile(np.array([0.0, 0.0, w]), (n, 1))
    accel = np.zeros((n, 3))
    dts = np.full(n, dt)

    result = preintegrate_raw(accel, gyro, dts)
    T = n * dt
    theta = w * T

    expected_R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    # Rotations about a fixed axis compose additively -> exact (fp-precision).
    assert np.allclose(result.delta_R, expected_R, atol=1e-9)
    assert np.allclose(result.delta_angle, [0.0, 0.0, theta], atol=1e-9)
    assert np.allclose(result.delta_v, 0.0, atol=1e-12)
    assert np.allclose(result.delta_p, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Synthetic trajectory 3: constant-speed circular motion (combined
# accel + gyro), checked against the closed-form analytic solution of the
# continuous kinematic ODE.
# ---------------------------------------------------------------------------


def test_circle_trajectory_matches_analytic_within_numeric_tolerance():
    speed = 5.0  # m/s
    radius = 20.0  # m
    w = speed / radius  # rad/s, turning left (+z)
    centripetal = speed * speed / radius

    n = 4000
    dt = 0.001
    # Body frame: x=forward, y=left, z=up. Constant-speed circular motion
    # has purely lateral (body +y) centripetal acceleration.
    accel = np.tile(np.array([0.0, centripetal, 0.0]), (n, 1))
    gyro = np.tile(np.array([0.0, 0.0, w]), (n, 1))
    dts = np.full(n, dt)

    result = preintegrate_raw(accel, gyro, dts)
    T = n * dt
    theta = w * T

    # A preintegration segment's delta_v/delta_p are the *pure specific-force
    # integral* relative to zero initial velocity (the actual v_i=(speed,0,0)
    # forward velocity is external state, added back in
    # predict_position_velocity). For constant-speed circular motion, the
    # absolute frame-i-relative velocity is v(t) = speed*(cos(wt), sin(wt), 0)
    # (initial condition v(0)=(speed,0,0)); subtracting the uniform-motion
    # baseline speed*(1,0,0) gives the closed-form delta_v below, and
    # delta_p = integral_0^T delta_v(t) dt gives delta_p.
    expected_v = np.array([speed * (np.cos(theta) - 1.0), speed * np.sin(theta), 0.0])
    expected_p = np.array(
        [radius * (np.sin(theta) - theta), radius * (1.0 - np.cos(theta)), 0.0]
    )

    p_err = np.linalg.norm(result.delta_p - expected_p)
    v_err = np.linalg.norm(result.delta_v - expected_v)
    # Discrete double-integration of a curving path has O(dt) truncation
    # error; dt=1ms over ~2.5s arc gives sub-cm/sub-mm agreement.
    assert p_err < 5e-3, f"circle delta_p error {p_err:.6f} m"
    assert v_err < 5e-3, f"circle delta_v error {v_err:.6f} m/s"

    R_err = np.linalg.norm(result.delta_R - rotvec_to_rotm(np.array([0.0, 0.0, theta])))
    assert R_err < 1e-8


# ---------------------------------------------------------------------------
# PreintegratedIMU stateful accumulator wraps preintegrate_raw correctly
# ---------------------------------------------------------------------------


def test_stateful_accumulator_matches_functional_form():
    rng = np.random.default_rng(2)
    n = 200
    accel = rng.normal(scale=1.0, size=(n, 3)) + np.array([0.0, 0.0, 9.81])
    gyro = rng.normal(scale=0.05, size=(n, 3))
    dts = np.full(n, 0.01)

    preint = PreintegratedIMU()
    preint.add_samples(accel, gyro, dts)

    direct = preintegrate_raw(accel, gyro, dts)
    assert np.allclose(preint.delta_p, direct.delta_p)
    assert np.allclose(preint.delta_v, direct.delta_v)
    assert np.allclose(preint.delta_R, direct.delta_R)
    assert preint.n_samples == n

    preint.reset()
    assert preint.n_samples == 0
    assert np.allclose(preint.delta_p, 0.0)
    assert np.allclose(preint.delta_R, np.eye(3))


def test_predict_position_velocity_matches_tc_fgo_residual_formula():
    """Exercises the same zero-residual condition as
    ``tc_fgo.imu_preintegration_residual`` (ported in the module docstring)."""

    rng = np.random.default_rng(3)
    n = 300
    accel = rng.normal(scale=0.5, size=(n, 3))
    gyro = rng.normal(scale=0.02, size=(n, 3))
    dts = np.full(n, 0.01)
    preint = PreintegratedIMU()
    preint.add_samples(accel, gyro, dts)

    p_i = np.array([100.0, -50.0, 10.0])
    v_i = np.array([3.0, 1.0, 0.0])
    R_i = rotvec_to_rotm(np.array([0.01, -0.02, 0.3]))

    p_j, v_j = preint.predict_position_velocity(p_i, v_i, R_i)

    dt = preint.delta_t
    dp, dv = preint.bias_corrected_delta_p_v()
    expected_p_j = p_i + v_i * dt + 0.5 * GRAVITY_ENU * dt * dt + R_i @ dp
    expected_v_j = v_i + GRAVITY_ENU * dt + R_i @ dv
    assert np.allclose(p_j, expected_p_j)
    assert np.allclose(v_j, expected_v_j)


# ---------------------------------------------------------------------------
# Bias-correction Jacobian check via finite differences.
#
# dp_d_ba / dv_d_ba / dp_d_bg / dv_d_bg / dR_d_bg are each verified to equal
# d(delta_p, delta_v, delta_angle) / d(constant additive offset applied to
# every accel or gyro sample in the segment) -- see the module docstring for
# why this is the well-defined, sign-unambiguous quantity these Jacobians
# represent.
# ---------------------------------------------------------------------------


def test_accel_bias_jacobian_finite_difference():
    rng = np.random.default_rng(4)
    n = 150
    accel = rng.normal(scale=1.0, size=(n, 3)) + np.array([1.0, 0.5, 9.81])
    gyro = rng.normal(scale=0.05, size=(n, 3))
    dts = np.full(n, 0.01)

    base = preintegrate_raw(accel, gyro, dts)
    eps = 1e-6
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = eps
        pert = preintegrate_raw(accel + shift, gyro, dts)
        fd_p = (pert.delta_p - base.delta_p) / eps
        fd_v = (pert.delta_v - base.delta_v) / eps
        assert np.allclose(fd_p, base.dp_d_ba[:, axis], atol=1e-4), axis
        assert np.allclose(fd_v, base.dv_d_ba[:, axis], atol=1e-4), axis


def test_gyro_bias_jacobian_finite_difference():
    rng = np.random.default_rng(5)
    n = 150
    accel = rng.normal(scale=1.0, size=(n, 3)) + np.array([0.3, -0.2, 9.81])
    gyro = rng.normal(scale=0.1, size=(n, 3))
    dts = np.full(n, 0.01)

    base = preintegrate_raw(accel, gyro, dts)
    eps = 1e-7
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = eps
        pert = preintegrate_raw(accel, gyro + shift, dts)
        fd_p = (pert.delta_p - base.delta_p) / eps
        fd_v = (pert.delta_v - base.delta_v) / eps
        fd_angle = (pert.delta_angle - base.delta_angle) / eps
        assert np.allclose(fd_p, base.dp_d_bg[:, axis], atol=1e-3), axis
        assert np.allclose(fd_v, base.dv_d_bg[:, axis], atol=1e-3), axis
        assert np.allclose(fd_angle, base.dR_d_bg[:, axis], atol=1e-3), axis


def test_bias_corrected_delta_p_v_matches_taylor_formula():
    rng = np.random.default_rng(6)
    n = 100
    accel = rng.normal(scale=0.5, size=(n, 3))
    gyro = rng.normal(scale=0.03, size=(n, 3))
    dts = np.full(n, 0.01)
    preint = PreintegratedIMU()
    preint.add_samples(accel, gyro, dts)

    ba = np.array([0.01, -0.02, 0.005])
    bg = np.array([0.001, 0.0, -0.002])
    dp, dv = preint.bias_corrected_delta_p_v(ba, bg)
    r = preint._compute()  # noqa: SLF001 - white-box check of the documented formula
    expected_dp = r.delta_p + r.dp_d_ba @ ba + r.dp_d_bg @ bg
    expected_dv = r.delta_v + r.dv_d_ba @ ba + r.dv_d_bg @ bg
    assert np.allclose(dp, expected_dp)
    assert np.allclose(dv, expected_dv)


# ---------------------------------------------------------------------------
# G1: cross-check against the actual preintegration engine the tc_fgo stack
# consumes (experiments/gsdc2023_imu.py). tc_fgo.py's own
# collapse_imu_preintegration_segment only *sums* pre-built per-interval
# segments; the on-manifold recursion that builds those segments (Delta_p,
# Delta_v, Delta_R, bias Jacobians, covariance) lives in
# gsdc2023_imu.preintegrate_processed_imu with sample_dt_mode="taroz" --
# ported into gnss_gpu.imu_preintegration verbatim. Both tc_fgo.py and this
# cross-check are therefore fair game per the task spec ("Tests MAY import
# tc_fgo to cross-check numbers").
# ---------------------------------------------------------------------------


def _cross_check_against_gsdc2023_imu(accel, gyro, dts):
    gsdc2023_imu = pytest.importorskip("gsdc2023_imu")

    times_ms = np.concatenate([[0.0], np.cumsum(dts) * 1000.0])
    imu_times_ms = times_ms[:-1]  # one sample "at" the start of each interval

    acc_processed = gsdc2023_imu.ProcessedIMU(
        times_ms=imu_times_ms,
        xyz=accel,
        dt_s=dts,
        norm_3d=np.linalg.norm(accel, axis=1),
        norm_std=np.zeros(len(dts)),
        sync_coefficient=1.0,
        bias=np.zeros_like(accel),
    )
    gyro_processed = gsdc2023_imu.ProcessedIMU(
        times_ms=imu_times_ms,
        xyz=gyro,
        dt_s=dts,
        norm_3d=np.linalg.norm(gyro, axis=1),
        norm_std=np.zeros(len(dts)),
        sync_coefficient=1.0,
        bias=np.zeros_like(gyro),
    )
    epoch_times_ms = np.array([times_ms[0], times_ms[-1]])
    preint = gsdc2023_imu.preintegrate_processed_imu(
        acc_processed,
        gyro_processed,
        epoch_times_ms,
        delta_frame="body",
        sample_dt_mode="taroz",
    )
    return preint


def test_g1_cross_check_against_gsdc2023_imu_synthetic():
    rng = np.random.default_rng(42)
    n = 400
    accel = rng.normal(scale=1.0, size=(n, 3)) + np.array([0.2, -0.1, 9.81])
    gyro = rng.normal(scale=0.05, size=(n, 3))
    dts = np.full(n, 0.01)

    ours = preintegrate_raw(accel, gyro, dts)
    theirs = _cross_check_against_gsdc2023_imu(accel, gyro, dts)

    dp_theirs = theirs.delta_p_body[0]
    dv_theirs = theirs.delta_v_body[0]
    dangle_theirs = theirs.delta_angle_rad[0]

    def rel_diff(a, b):
        denom = max(np.linalg.norm(b), 1e-9)
        return np.linalg.norm(a - b) / denom

    p_rel = rel_diff(ours.delta_p, dp_theirs)
    v_rel = rel_diff(ours.delta_v, dv_theirs)
    angle_rel = rel_diff(ours.delta_angle, dangle_theirs)

    print(
        f"\n[G1 synthetic] max rel diff: delta_p={p_rel:.3e} "
        f"delta_v={v_rel:.3e} delta_angle={angle_rel:.3e}"
    )
    assert p_rel < 1e-9
    assert v_rel < 1e-9
    assert angle_rel < 1e-9

    dp_ba_theirs = theirs.delta_p_bias_accel_jac[0]
    dv_ba_theirs = theirs.delta_v_bias_accel_jac[0]
    dp_bg_theirs = theirs.delta_p_bias_gyro_jac[0]
    dv_bg_theirs = theirs.delta_v_bias_gyro_jac[0]
    dangle_bg_theirs = theirs.delta_angle_bias_gyro_jac[0]

    assert np.allclose(ours.dp_d_ba, dp_ba_theirs, atol=1e-9)
    assert np.allclose(ours.dv_d_ba, dv_ba_theirs, atol=1e-9)
    assert np.allclose(ours.dp_d_bg, dp_bg_theirs, atol=1e-9)
    assert np.allclose(ours.dv_d_bg, dv_bg_theirs, atol=1e-9)
    assert np.allclose(ours.dR_d_bg, dangle_bg_theirs, atol=1e-9)


_PPC_RUN2_IMU = None
try:  # pragma: no cover - resolved once at import; skipped cleanly if absent
    from pathlib import Path

    _CANDIDATE = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "PPC-Dataset-data"
        / "tokyo"
        / "run2"
        / "imu.csv"
    )
    if _CANDIDATE.exists():
        _PPC_RUN2_IMU = _CANDIDATE
except Exception:  # noqa: BLE001
    _PPC_RUN2_IMU = None


@pytest.mark.skipif(_PPC_RUN2_IMU is None, reason="PPC Tokyo run2 imu.csv not present")
def test_g1_cross_check_against_gsdc2023_imu_real_ppc_data():
    """G1: preintegration matches tc_fgo's engine on real PPC IMU data."""

    from gnss_gpu.io.ppc import PPCDatasetLoader

    imu_data = PPCDatasetLoader(_PPC_RUN2_IMU.parent).load_imu()
    deg2rad = np.pi / 180.0
    n = min(2000, imu_data["time"].size)
    accel = np.column_stack(
        [imu_data["acc_x"][:n], imu_data["acc_y"][:n], imu_data["acc_z"][:n]]
    )
    gyro = (
        np.column_stack(
            [imu_data["gyro_x"][:n], imu_data["gyro_y"][:n], imu_data["gyro_z"][:n]]
        )
        * deg2rad
    )
    time_s = imu_data["time"][:n]
    dts = np.diff(time_s)
    accel = accel[:-1]
    gyro = gyro[:-1]
    finite = np.isfinite(dts) & (dts > 0.0) & np.isfinite(accel).all(axis=1) & np.isfinite(gyro).all(axis=1)
    accel = accel[finite]
    gyro = gyro[finite]
    dts = dts[finite]
    assert dts.size > 100, "expected a substantial real IMU sample slice"

    ours = preintegrate_raw(accel, gyro, dts)
    theirs = _cross_check_against_gsdc2023_imu(accel, gyro, dts)

    dp_theirs = theirs.delta_p_body[0]
    dv_theirs = theirs.delta_v_body[0]
    dangle_theirs = theirs.delta_angle_rad[0]

    def rel_diff(a, b):
        denom = max(np.linalg.norm(b), 1e-9)
        return np.linalg.norm(a - b) / denom

    p_rel = rel_diff(ours.delta_p, dp_theirs)
    v_rel = rel_diff(ours.delta_v, dv_theirs)
    angle_rel = rel_diff(ours.delta_angle, dangle_theirs)
    print(
        f"\n[G1 real PPC tokyo/run2, n={dts.size} samples, "
        f"T={float(np.sum(dts)):.2f}s] max rel diff: "
        f"delta_p={p_rel:.3e} delta_v={v_rel:.3e} delta_angle={angle_rel:.3e}"
    )
    assert p_rel < 1e-8
    assert v_rel < 1e-8
    assert angle_rel < 1e-8
