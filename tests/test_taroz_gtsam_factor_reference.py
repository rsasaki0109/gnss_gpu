"""Pinned intermediate residuals from Taro Suzuki's gtsam_gnss factors.

These tests mirror the ``evaluateError`` formulas in
``ref/matlab_local/include/gtsam_gnss``.  They provide a small reference layer
for the GSDC2023 Taroz FGO port, independent of the CUDA extension.
"""

from __future__ import annotations

import numpy as np

from experiments.gsdc2023_imu import (
    ProcessedIMU,
    preintegrate_processed_imu,
    rotm_to_rotvec,
    rotvec_to_rotm,
)


def _rot_z(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _preintegrated_rotvec_with_gyro_bias(
    gyro_xyz: np.ndarray,
    sample_dt_s: np.ndarray,
    gyro_bias: np.ndarray,
) -> np.ndarray:
    delta_rot = np.eye(3, dtype=np.float64)
    for omega, dt_s in zip(
        np.asarray(gyro_xyz, dtype=np.float64).reshape(-1, 3),
        np.asarray(sample_dt_s, dtype=np.float64).reshape(-1),
    ):
        delta_rot = delta_rot @ rotvec_to_rotm((omega - gyro_bias) * dt_s)
    return rotm_to_rotvec(delta_rot)


def _preintegrated_pv_with_gyro_bias(
    acc_xyz: np.ndarray,
    gyro_xyz: np.ndarray,
    sample_dt_s: np.ndarray,
    gyro_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta_rot = np.eye(3, dtype=np.float64)
    vel = np.zeros(3, dtype=np.float64)
    pos = np.zeros(3, dtype=np.float64)
    for acc_sample, omega, dt_s in zip(
        np.asarray(acc_xyz, dtype=np.float64).reshape(-1, 3),
        np.asarray(gyro_xyz, dtype=np.float64).reshape(-1, 3),
        np.asarray(sample_dt_s, dtype=np.float64).reshape(-1),
    ):
        acc_delta = delta_rot @ acc_sample
        pos += vel * dt_s + 0.5 * acc_delta * dt_s * dt_s
        vel += acc_delta * dt_s
        delta_rot = delta_rot @ rotvec_to_rotm((omega - gyro_bias) * dt_s)
    return pos, vel


def _positive_gyro_bias_jacobian_finite_difference(
    gyro_xyz: np.ndarray,
    sample_dt_s: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    jac = np.zeros((3, 3), dtype=np.float64)
    for axis in range(3):
        bias_step = np.zeros(3, dtype=np.float64)
        bias_step[axis] = eps
        plus = _preintegrated_rotvec_with_gyro_bias(gyro_xyz, sample_dt_s, bias_step)
        minus = _preintegrated_rotvec_with_gyro_bias(gyro_xyz, sample_dt_s, -bias_step)
        jac[:, axis] = -(plus - minus) / (2.0 * eps)
    return jac


def _positive_pv_gyro_bias_jacobian_finite_difference(
    acc_xyz: np.ndarray,
    gyro_xyz: np.ndarray,
    sample_dt_s: np.ndarray,
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    jac_p = np.zeros((3, 3), dtype=np.float64)
    jac_v = np.zeros((3, 3), dtype=np.float64)
    for axis in range(3):
        bias_step = np.zeros(3, dtype=np.float64)
        bias_step[axis] = eps
        plus_p, plus_v = _preintegrated_pv_with_gyro_bias(acc_xyz, gyro_xyz, sample_dt_s, bias_step)
        minus_p, minus_v = _preintegrated_pv_with_gyro_bias(acc_xyz, gyro_xyz, sample_dt_s, -bias_step)
        jac_p[:, axis] = -(plus_p - minus_p) / (2.0 * eps)
        jac_v[:, axis] = -(plus_v - minus_v) / (2.0 * eps)
    return jac_p, jac_v


def _clock_selector(n_clock: int, sys_idx: int) -> np.ndarray:
    h = np.zeros(n_clock, dtype=np.float64)
    h[0] = 1.0
    if 0 <= sys_idx < n_clock:
        h[sys_idx] = 1.0
    return h


def test_pseudorange_factor_xc_intermediate_residual_and_jacobians():
    los = np.array([0.6, -0.8, 0.0], dtype=np.float64)
    x0 = np.array([10.0, 20.0, -5.0], dtype=np.float64)
    x = np.array([11.5, 18.0, -3.0], dtype=np.float64)
    c = np.array([4.0, -1.0, 2.5], dtype=np.float64)
    pr_residual_at_initial = 3.2
    sys_idx = 2

    h_clock = _clock_selector(c.size, sys_idx)
    residual = los @ (x - x0) + h_clock @ c - pr_residual_at_initial

    np.testing.assert_allclose(residual, np.float64(5.8))
    np.testing.assert_allclose(los.reshape(1, 3), np.array([[0.6, -0.8, 0.0]]))
    np.testing.assert_allclose(h_clock.reshape(1, 3), np.array([[1.0, 0.0, 1.0]]))


def test_doppler_factor_vd_intermediate_residual_and_jacobians():
    los = np.array([0.2, -0.3, 0.9327379053], dtype=np.float64)
    v0 = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    v = np.array([3.0, -1.0, -0.5], dtype=np.float64)
    drift = np.array([0.7], dtype=np.float64)
    doppler_residual_at_initial = 0.25

    residual = los @ (v - v0) + drift[0] - doppler_residual_at_initial

    np.testing.assert_allclose(residual, np.float64(-0.3827379053))
    np.testing.assert_allclose(los.reshape(1, 3), np.array([[0.2, -0.3, 0.9327379053]]))
    np.testing.assert_allclose(np.ones((1, 1)), np.array([[1.0]]))


def test_tdcp_factor_xxcc_intermediate_residual_and_jacobians():
    los = np.array([0.5, 0.4, -0.766], dtype=np.float64)
    x1_0 = np.array([2.0, 1.0, 0.0], dtype=np.float64)
    x2_0 = np.array([2.5, 1.5, 0.2], dtype=np.float64)
    x1 = np.array([2.2, 0.6, 0.1], dtype=np.float64)
    x2 = np.array([3.0, 1.8, 0.4], dtype=np.float64)
    c1 = np.array([5.0, 0.2], dtype=np.float64)
    c2 = np.array([5.3, -0.1], dtype=np.float64)
    tdcp_m = 0.42

    dx = (x2 - x2_0) - (x1 - x1_0)
    dc = c2 - c1
    h_clock = np.array([1.0, 0.0], dtype=np.float64)
    residual = los @ dx + h_clock @ dc - tdcp_m

    np.testing.assert_allclose(residual, np.float64(0.2334), atol=1e-12)
    np.testing.assert_allclose(-los.reshape(1, 3), np.array([[-0.5, -0.4, 0.766]]))
    np.testing.assert_allclose(los.reshape(1, 3), np.array([[0.5, 0.4, -0.766]]))
    np.testing.assert_allclose(-h_clock.reshape(1, 2), np.array([[-1.0, -0.0]]))
    np.testing.assert_allclose(h_clock.reshape(1, 2), np.array([[1.0, 0.0]]))


def test_tdcp_factor_xxdd_intermediate_residual_and_jacobians():
    los = np.array([-0.1, 0.9, 0.4242640687], dtype=np.float64)
    x1_0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    x2_0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x1 = np.array([0.2, -0.1, 0.3], dtype=np.float64)
    x2 = np.array([1.4, 0.4, 0.5], dtype=np.float64)
    d1 = np.array([0.8], dtype=np.float64)
    d2 = np.array([1.2], dtype=np.float64)
    dt = 0.5
    tdcp_m = 0.1

    dx = (x2 - x2_0) - (x1 - x1_0)
    residual = los @ dx + dt * (d1[0] + d2[0]) / 2.0 - tdcp_m

    np.testing.assert_allclose(residual, np.float64(0.91485281374), atol=1e-12)
    np.testing.assert_allclose(-los.reshape(1, 3), np.array([[0.1, -0.9, -0.4242640687]]))
    np.testing.assert_allclose(los.reshape(1, 3), np.array([[-0.1, 0.9, 0.4242640687]]))
    np.testing.assert_allclose(np.array([[dt / 2.0]]), np.array([[0.25]]))


def test_clock_factor_ccdd_intermediate_residual_and_jacobians():
    c1 = np.array([3.0, 1.5], dtype=np.float64)
    c2 = np.array([3.7, 1.2], dtype=np.float64)
    d1 = np.array([0.4], dtype=np.float64)
    d2 = np.array([0.8], dtype=np.float64)
    dt = 2.0

    residual = c2 - c1
    residual[0] -= (d1[0] + d2[0]) * dt / 2.0
    hd = np.zeros((2, 1), dtype=np.float64)
    hd[0, 0] = -dt / 2.0

    np.testing.assert_allclose(residual, np.array([-0.5, -0.3]), atol=1e-12)
    np.testing.assert_allclose(-np.eye(2), np.array([[-1.0, -0.0], [-0.0, -1.0]]))
    np.testing.assert_allclose(np.eye(2), np.array([[1.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(hd, np.array([[-1.0], [0.0]]))


def test_motion_factor_xxvv_intermediate_residual_and_jacobians():
    x1 = np.array([12.0, -4.0, 8.0], dtype=np.float64)
    x2 = np.array([14.4, -7.2, 8.9], dtype=np.float64)
    v1 = np.array([3.0, -1.0, 0.5], dtype=np.float64)
    v2 = np.array([1.0, -3.0, 2.5], dtype=np.float64)
    dt = 1.2

    residual = x2 - x1 - (v1 + v2) * dt / 2.0
    hv = -dt / 2.0 * np.eye(3)

    np.testing.assert_allclose(residual, np.array([0.0, -0.8, -0.9]), atol=1e-12)
    np.testing.assert_allclose(-np.eye(3), np.diag([-1.0, -1.0, -1.0]))
    np.testing.assert_allclose(np.eye(3), np.diag([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(hv, np.diag([-0.6, -0.6, -0.6]))


def _zero_rotation_preintegration_bias_jacobians(sample_dt_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acc_jac_v = np.zeros((3, 3), dtype=np.float64)
    acc_jac_p = np.zeros((3, 3), dtype=np.float64)
    gyro_jac_angle = np.zeros((3, 3), dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)
    for dt_s in np.asarray(sample_dt_s, dtype=np.float64):
        acc_jac_p += acc_jac_v * dt_s + 0.5 * eye3 * dt_s * dt_s
        acc_jac_v += eye3 * dt_s
        gyro_jac_angle += eye3 * dt_s
    return acc_jac_p, acc_jac_v, gyro_jac_angle


def test_taroz_imu_preintegration_zero_rotation_bias_jacobian_intermediates():
    """Pin the GTSAM zero-rotation IMU bias Jacobian inputs used by Taroz.

    ``fgo_gnss_imu.m`` collects ``IMUindices`` in the GNSS interval and calls
    ``integrateMeasurement(..., acc.dt(imuIndex))`` for each sample.  With zero
    rotation and identity body/sensor orientation, GTSAM's preintegrated bias
    Jacobians reduce to this discrete sample-dt recursion.
    """

    sample_times_ms = np.array([0.0, 200.0, 500.0], dtype=np.float64)
    sample_dt_s = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    acc = ProcessedIMU(
        times_ms=sample_times_ms,
        xyz=np.zeros((sample_times_ms.size, 3), dtype=np.float64),
        dt_s=sample_dt_s,
        norm_3d=np.zeros(sample_times_ms.size, dtype=np.float64),
        norm_std=np.zeros(sample_times_ms.size, dtype=np.float64),
        sync_coefficient=1.0,
    )
    gyro = ProcessedIMU(
        times_ms=sample_times_ms,
        xyz=np.zeros((sample_times_ms.size, 3), dtype=np.float64),
        dt_s=sample_dt_s,
        norm_3d=np.zeros(sample_times_ms.size, dtype=np.float64),
        norm_std=np.zeros(sample_times_ms.size, dtype=np.float64),
        sync_coefficient=1.0,
    )

    preint = preintegrate_processed_imu(
        acc,
        gyro,
        np.array([0.0, 500.0], dtype=np.float64),
        sample_dt_mode="taroz",
    )
    expected_p, expected_v, expected_angle = _zero_rotation_preintegration_bias_jacobians(sample_dt_s)

    np.testing.assert_allclose(preint.delta_t_s, np.array([0.9]), atol=1e-12)
    assert preint.delta_p_bias_accel_jac is not None
    assert preint.delta_v_bias_accel_jac is not None
    assert preint.delta_angle_bias_gyro_jac is not None
    np.testing.assert_allclose(preint.delta_p_bias_accel_jac[0], expected_p, atol=1e-12)
    np.testing.assert_allclose(preint.delta_v_bias_accel_jac[0], expected_v, atol=1e-12)
    np.testing.assert_allclose(preint.delta_angle_bias_gyro_jac[0], expected_angle, atol=1e-12)
    np.testing.assert_allclose(expected_p, np.eye(3, dtype=np.float64) * 0.405, atol=1e-12)
    np.testing.assert_allclose(expected_v, np.eye(3, dtype=np.float64) * 0.9, atol=1e-12)
    np.testing.assert_allclose(expected_angle, np.eye(3, dtype=np.float64) * 0.9, atol=1e-12)


def test_taroz_imu_preintegration_unit_noise_covariance_matches_gtsam_zero_rotation_probe():
    sample_times_ms = np.array([0.0, 500.0], dtype=np.float64)
    sample_dt_s = np.full(sample_times_ms.size, 0.5, dtype=np.float64)
    acc = ProcessedIMU(
        times_ms=sample_times_ms,
        xyz=np.zeros((sample_times_ms.size, 3), dtype=np.float64),
        dt_s=sample_dt_s,
        norm_3d=np.zeros(sample_times_ms.size, dtype=np.float64),
        norm_std=np.zeros(sample_times_ms.size, dtype=np.float64),
        sync_coefficient=1.0,
    )
    gyro = ProcessedIMU(
        times_ms=sample_times_ms,
        xyz=np.zeros((sample_times_ms.size, 3), dtype=np.float64),
        dt_s=sample_dt_s,
        norm_3d=np.zeros(sample_times_ms.size, dtype=np.float64),
        norm_std=np.zeros(sample_times_ms.size, dtype=np.float64),
        sync_coefficient=1.0,
    )

    preint = preintegrate_processed_imu(
        acc,
        gyro,
        np.array([0.0, 1000.0], dtype=np.float64),
        sample_dt_mode="taroz",
    )

    expected_accel = np.zeros((9, 9), dtype=np.float64)
    expected_gyro = np.zeros((9, 9), dtype=np.float64)
    expected_integration = np.zeros((9, 9), dtype=np.float64)
    for axis in range(3):
        p = axis
        v = 3 + axis
        a = 6 + axis
        expected_accel[p, p] = 0.3125
        expected_accel[p, v] = 0.5
        expected_accel[v, p] = 0.5
        expected_accel[v, v] = 1.0
        expected_gyro[a, a] = 1.0
        expected_integration[p, p] = 1.0

    assert preint.pva_accel_noise_cov is not None
    assert preint.pva_gyro_noise_cov is not None
    assert preint.pva_integration_noise_cov is not None
    np.testing.assert_allclose(preint.pva_accel_noise_cov[0], expected_accel, atol=1e-12)
    np.testing.assert_allclose(preint.pva_gyro_noise_cov[0], expected_gyro, atol=1e-12)
    np.testing.assert_allclose(preint.pva_integration_noise_cov[0], expected_integration, atol=1e-12)


def test_taroz_imu_preintegration_rotates_accel_by_cumulative_gyro_intermediates():
    """Pin the nonzero-rotation sample-dt inputs before native IMU factors."""

    sample_times_ms = np.array([0.0, 500.0, 1000.0], dtype=np.float64)
    sample_dt_s = np.full(sample_times_ms.size, 0.5, dtype=np.float64)
    acc = ProcessedIMU(
        times_ms=sample_times_ms,
        xyz=np.tile(np.array([2.0, 0.0, 0.0], dtype=np.float64), (sample_times_ms.size, 1)),
        dt_s=sample_dt_s,
        norm_3d=np.full(sample_times_ms.size, 2.0, dtype=np.float64),
        norm_std=np.zeros(sample_times_ms.size, dtype=np.float64),
        sync_coefficient=1.0,
    )
    gyro = ProcessedIMU(
        times_ms=sample_times_ms,
        xyz=np.tile(np.array([0.0, 0.0, 0.1], dtype=np.float64), (sample_times_ms.size, 1)),
        dt_s=sample_dt_s,
        norm_3d=np.full(sample_times_ms.size, 0.1, dtype=np.float64),
        norm_std=np.zeros(sample_times_ms.size, dtype=np.float64),
        sync_coefficient=1.0,
    )

    preint = preintegrate_processed_imu(
        acc,
        gyro,
        np.array([0.0, 1000.0], dtype=np.float64),
        sample_dt_mode="taroz",
    )

    eye3 = np.eye(3, dtype=np.float64)
    r1 = _rot_z(0.05)
    r2 = _rot_z(0.10)
    acc_vec = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    expected_v = 0.5 * (eye3 + r1 + r2) @ acc_vec
    expected_p = (0.625 * eye3 + 0.375 * r1 + 0.125 * r2) @ acc_vec
    expected_v_jac = 0.5 * (eye3 + r1 + r2)
    expected_p_jac = 0.625 * eye3 + 0.375 * r1 + 0.125 * r2
    expected_p_gyro_jac, expected_v_gyro_jac = _positive_pv_gyro_bias_jacobian_finite_difference(
        acc.xyz,
        gyro.xyz,
        sample_dt_s,
    )

    np.testing.assert_allclose(preint.delta_t_s, np.array([1.5]), atol=1e-12)
    np.testing.assert_array_equal(preint.sample_count, np.array([3], dtype=np.int32))
    np.testing.assert_allclose(preint.delta_v_body[0], expected_v, atol=1e-12)
    np.testing.assert_allclose(preint.delta_p_body[0], expected_p, atol=1e-12)
    np.testing.assert_allclose(preint.delta_angle_rad[0], np.array([0.0, 0.0, 0.15]), atol=1e-12)
    assert preint.delta_p_bias_accel_jac is not None
    assert preint.delta_v_bias_accel_jac is not None
    assert preint.delta_p_bias_gyro_jac is not None
    assert preint.delta_v_bias_gyro_jac is not None
    assert preint.delta_angle_bias_gyro_jac is not None
    np.testing.assert_allclose(preint.delta_v_bias_accel_jac[0], expected_v_jac, atol=1e-12)
    np.testing.assert_allclose(preint.delta_p_bias_accel_jac[0], expected_p_jac, atol=1e-12)
    np.testing.assert_allclose(preint.delta_p_bias_gyro_jac[0], expected_p_gyro_jac, atol=1e-8)
    np.testing.assert_allclose(preint.delta_v_bias_gyro_jac[0], expected_v_gyro_jac, atol=1e-8)
    np.testing.assert_allclose(
        preint.delta_angle_bias_gyro_jac[0],
        _positive_gyro_bias_jacobian_finite_difference(gyro.xyz, sample_dt_s),
        atol=1e-8,
    )
    np.testing.assert_allclose(preint.delta_angle_bias_gyro_jac[0, 2, 2], 1.5, atol=1e-12)


def test_taroz_imu_preintegration_noncommuting_gyro_uses_so3_log_and_bias_jacobian():
    """Pin SO(3) rotation accumulation and gyro-bias Jacobian direction."""

    sample_times_ms = np.array([0.0, 400.0, 900.0], dtype=np.float64)
    sample_dt_s = np.array([0.4, 0.5, 0.3], dtype=np.float64)
    gyro_xyz = np.array(
        [
            [0.30, 0.10, 0.00],
            [0.00, 0.20, 0.40],
            [-0.20, 0.00, 0.10],
        ],
        dtype=np.float64,
    )
    acc = ProcessedIMU(
        times_ms=sample_times_ms,
        xyz=np.zeros((sample_times_ms.size, 3), dtype=np.float64),
        dt_s=sample_dt_s,
        norm_3d=np.zeros(sample_times_ms.size, dtype=np.float64),
        norm_std=np.zeros(sample_times_ms.size, dtype=np.float64),
        sync_coefficient=1.0,
    )
    gyro = ProcessedIMU(
        times_ms=sample_times_ms,
        xyz=gyro_xyz,
        dt_s=sample_dt_s,
        norm_3d=np.linalg.norm(gyro_xyz, axis=1),
        norm_std=np.zeros(sample_times_ms.size, dtype=np.float64),
        sync_coefficient=1.0,
    )

    preint = preintegrate_processed_imu(
        acc,
        gyro,
        np.array([0.0, 900.0], dtype=np.float64),
        sample_dt_mode="taroz",
    )

    expected_angle = _preintegrated_rotvec_with_gyro_bias(gyro_xyz, sample_dt_s, np.zeros(3, dtype=np.float64))
    expected_jac = _positive_gyro_bias_jacobian_finite_difference(gyro_xyz, sample_dt_s)

    np.testing.assert_allclose(preint.delta_t_s, np.array([1.2]), atol=1e-12)
    np.testing.assert_array_equal(preint.sample_count, np.array([3], dtype=np.int32))
    np.testing.assert_allclose(preint.delta_angle_rad[0], expected_angle, atol=1e-12)
    assert not np.allclose(expected_angle, np.sum(gyro_xyz * sample_dt_s[:, None], axis=0), atol=1e-5)
    assert preint.delta_angle_bias_gyro_jac is not None
    np.testing.assert_allclose(preint.delta_angle_bias_gyro_jac[0], expected_jac, atol=1e-8)


def test_taroz_imu_factor_body_gravity_residual_intermediates():
    """Pin the GTSAM ImuFactor p/v residual topology used by native imu_gravity."""

    p_i = np.array([10.0, -4.0, 2.0], dtype=np.float64)
    v_i = np.array([1.5, -0.25, 0.5], dtype=np.float64)
    att_i = np.array([0.15, -0.05, 0.20], dtype=np.float64)
    att_j = np.array([0.16, -0.06, 0.19], dtype=np.float64)
    dt = 0.8
    gravity = np.array([0.2, -0.1, -9.7], dtype=np.float64)
    delta_p = np.array([0.35, -0.12, 3.05], dtype=np.float64)
    delta_v = np.array([0.8, -0.2, 8.0], dtype=np.float64)
    delta_angle = np.array([0.02, -0.01, 0.03], dtype=np.float64)
    injected_p_residual = np.array([0.10, -0.20, 0.30], dtype=np.float64)
    injected_v_residual = np.array([-0.30, 0.20, 0.10], dtype=np.float64)
    rot_i = rotvec_to_rotm(att_i)
    rot_j = rotvec_to_rotm(att_j)
    predicted_rot_j = rot_i @ rotvec_to_rotm(delta_angle)
    predicted_p_j = p_i + v_i * dt + 0.5 * gravity * dt * dt + rot_i @ delta_p
    predicted_v_j = v_i + gravity * dt + rot_i @ delta_v

    p_j = predicted_p_j - rot_j @ injected_p_residual
    v_j = predicted_v_j - rot_j @ injected_v_residual

    res_p = rot_j.T @ (predicted_p_j - p_j)
    res_v = rot_j.T @ (predicted_v_j - v_j)
    res_r = rotm_to_rotvec(rot_j.T @ predicted_rot_j)

    np.testing.assert_allclose(res_p, injected_p_residual, atol=1e-12)
    np.testing.assert_allclose(res_v, injected_v_residual, atol=1e-12)
    np.testing.assert_allclose(res_r, np.array([0.011569448965, 0.001856379742, 0.039314793922]), atol=1e-12)
