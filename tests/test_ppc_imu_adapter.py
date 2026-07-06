"""Unit tests for the TASK_D D3.1 PPC IMU preintegration adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from ppc_imu_adapter import (  # noqa: E402
    build_ppc_imu_preintegration,
    ppc_imu_to_processed,
)


def _synthetic_imu_dict(
    times_s: np.ndarray,
    acc_xyz: np.ndarray,
    gyro_xyz_dps: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "time": times_s,
        "acc_x": acc_xyz[:, 0],
        "acc_y": acc_xyz[:, 1],
        "acc_z": acc_xyz[:, 2],
        "gyro_x": gyro_xyz_dps[:, 0],
        "gyro_y": gyro_xyz_dps[:, 1],
        "gyro_z": gyro_xyz_dps[:, 2],
    }


def test_ppc_imu_to_processed_converts_gyro_deg_to_rad():
    n = 5
    times_s = np.arange(n, dtype=np.float64) * 0.01
    acc_xyz = np.zeros((n, 3), dtype=np.float64)
    gyro_xyz_dps = np.tile(np.array([180.0, -90.0, 0.0]), (n, 1))
    imu_data = _synthetic_imu_dict(times_s, acc_xyz, gyro_xyz_dps)

    acc, gyro = ppc_imu_to_processed(imu_data)

    assert acc.times_ms.size == n
    np.testing.assert_allclose(acc.times_ms, times_s * 1000.0)
    np.testing.assert_allclose(gyro.xyz[:, 0], np.pi, rtol=1e-12)
    np.testing.assert_allclose(gyro.xyz[:, 1], -np.pi / 2.0, rtol=1e-12)
    np.testing.assert_allclose(gyro.xyz[:, 2], 0.0, atol=1e-12)


def test_ppc_imu_to_processed_drops_non_finite_samples():
    times_s = np.array([0.0, 0.01, 0.02, 0.03], dtype=np.float64)
    acc_xyz = np.array(
        [[1.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    gyro_xyz_dps = np.zeros((4, 3), dtype=np.float64)
    imu_data = _synthetic_imu_dict(times_s, acc_xyz, gyro_xyz_dps)

    acc, gyro = ppc_imu_to_processed(imu_data)

    assert acc.times_ms.size == 3
    assert gyro.times_ms.size == 3
    np.testing.assert_allclose(acc.times_ms, np.array([0.0, 20.0, 30.0]))


def test_build_ppc_imu_preintegration_body_frame_constant_accel():
    """Constant body-frame acceleration with zero rotation integrates analytically."""
    dt = 0.01
    duration = 1.0
    n = int(duration / dt) + 1
    times_s = np.arange(n, dtype=np.float64) * dt
    acc_xyz = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    gyro_xyz_dps = np.zeros((n, 3), dtype=np.float64)
    imu_data = _synthetic_imu_dict(times_s, acc_xyz, gyro_xyz_dps)

    epoch_times_s = np.array([0.0, 1.0], dtype=np.float64)
    reference_ecef = np.tile(np.array([-3958080.0, 3350070.0, 3700660.0]), (2, 1))

    preint = build_ppc_imu_preintegration(
        imu_data,
        epoch_times_s,
        reference_ecef,
        delta_frame="body",
    )

    assert preint.delta_t_s.shape == (1,)
    assert preint.delta_t_s[0] == pytest.approx(1.0, abs=1e-3)
    # a=1 m/s^2 for 1s (body frame, no rotation/gravity compensation applied):
    # delta_v = a*T = 1.0 m/s; delta_p = 0.5*a*T^2 = 0.5 m.
    np.testing.assert_allclose(preint.delta_v_body[0], [1.0, 0.0, 0.0], atol=2e-2)
    np.testing.assert_allclose(preint.delta_p_body[0], [0.5, 0.0, 0.0], atol=2e-2)


def test_build_ppc_imu_preintegration_ecef_frame_gravity_compensated_when_stationary():
    """Stationary accelerometer (+g up) should preintegrate to ~zero ECEF delta."""
    dt = 0.01
    duration = 1.0
    n = int(duration / dt) + 1
    times_s = np.arange(n, dtype=np.float64) * dt
    # Sensor at rest: measures +g "up" (reaction to gravity), zero rotation.
    acc_xyz = np.tile(np.array([0.0, 0.0, 9.80665]), (n, 1))
    gyro_xyz_dps = np.zeros((n, 3), dtype=np.float64)
    imu_data = _synthetic_imu_dict(times_s, acc_xyz, gyro_xyz_dps)

    epoch_times_s = np.array([0.0, 1.0], dtype=np.float64)
    # A fixed ECEF point (roughly Tokyo); receiver does not move -> yaw undefined -> 0.
    reference_ecef = np.tile(np.array([-3958080.0, 3350070.0, 3700660.0]), (2, 1))

    preint = build_ppc_imu_preintegration(
        imu_data,
        epoch_times_s,
        reference_ecef,
        delta_frame="ecef",
    )

    assert preint.delta_frame == "ecef"
    np.testing.assert_allclose(preint.delta_v_body[0], [0.0, 0.0, 0.0], atol=1e-2)
    np.testing.assert_allclose(preint.delta_p_body[0], [0.0, 0.0, 0.0], atol=1e-2)


def test_build_ppc_imu_preintegration_empty_imu_returns_zero_intervals():
    epoch_times_s = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    reference_ecef = np.tile(np.array([-3958080.0, 3350070.0, 3700660.0]), (3, 1))
    imu_data = _synthetic_imu_dict(
        np.zeros(0, dtype=np.float64),
        np.zeros((0, 3), dtype=np.float64),
        np.zeros((0, 3), dtype=np.float64),
    )

    preint = build_ppc_imu_preintegration(imu_data, epoch_times_s, reference_ecef)

    assert preint.delta_t_s.shape == (2,)
    np.testing.assert_allclose(preint.delta_v_body, 0.0)
    np.testing.assert_allclose(preint.delta_p_body, 0.0)
