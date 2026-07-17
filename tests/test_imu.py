"""Unit tests for gnss_gpu.imu (WP21b): IMUPredictor gravity-sign fix and
ComplementaryHeadingFilter heading-variance tracking.
"""

from __future__ import annotations

import math

import numpy as np

from gnss_gpu.imu import ComplementaryHeadingFilter, IMUPredictor


def _imu_dict(accel: np.ndarray, gyro: np.ndarray, dt: float = 0.02) -> dict:
    n = accel.shape[0]
    return {
        "tow": np.arange(n, dtype=np.float64) * dt,
        "accel": accel,
        "gyro": gyro,
        "wheel_vel": np.full(n, np.nan, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# IMUPredictor gravity-sign fix
# ---------------------------------------------------------------------------


def test_gravity_autodetect_positive_at_rest_matches_ppc_convention():
    """PPC-style data: static accel_z ~ +9.81. A static segment must not
    accumulate a large phantom vertical velocity (the pre-fix bug: adding a
    second +g every sample -> unbounded climb)."""

    n = 200
    accel = np.tile(np.array([0.0, 0.0, 9.81]), (n, 1))
    gyro = np.zeros((n, 3))
    imu_dict = _imu_dict(accel, gyro)

    predictor = IMUPredictor(imu_dict)
    assert predictor.gravity_convention == "positive_at_rest"

    vel = predictor.get_velocity_enu(imu_dict["tow"][0], imu_dict["tow"][-1] + 0.02)
    assert vel is not None
    # Static segment: velocity should stay near zero, not blow up to ~2g*T.
    assert np.linalg.norm(vel) < 0.1, vel


def test_gravity_autodetect_negative_at_rest_preserves_legacy_behavior():
    """Legacy-convention data: static accel_z ~ -9.81. Must reproduce the
    original (pre-WP21b) formula exactly: az_body = accel_z + 9.81."""

    n = 200
    accel = np.tile(np.array([0.0, 0.0, -9.81]), (n, 1))
    gyro = np.zeros((n, 3))
    imu_dict = _imu_dict(accel, gyro)

    predictor = IMUPredictor(imu_dict)
    assert predictor.gravity_convention == "negative_at_rest"

    vel = predictor.get_velocity_enu(imu_dict["tow"][0], imu_dict["tow"][-1] + 0.02)
    assert vel is not None
    assert np.linalg.norm(vel) < 0.1, vel


def test_gravity_convention_explicit_override():
    n = 50
    accel = np.tile(np.array([0.0, 0.0, 9.81]), (n, 1))
    gyro = np.zeros((n, 3))
    imu_dict = _imu_dict(accel, gyro)

    # Force the "wrong" (legacy) convention on PPC-style (+9.81-at-rest) data:
    # this should reproduce the pre-fix bug (adds a second g).
    predictor = IMUPredictor(imu_dict, gravity_convention="negative_at_rest")
    assert predictor._gravity_removal_bias == 9.81
    vel = predictor.get_velocity_enu(imu_dict["tow"][0], imu_dict["tow"][-1] + 0.02)
    # ~2g * T upward velocity: T = n*dt = 50*0.02 = 1.0s -> ~19.6 m/s climb.
    assert vel[2] > 15.0, vel

    predictor2 = IMUPredictor(imu_dict, gravity_convention="positive_at_rest")
    assert predictor2._gravity_removal_bias == -9.81
    vel2 = predictor2.get_velocity_enu(imu_dict["tow"][0], imu_dict["tow"][-1] + 0.02)
    assert abs(vel2[2]) < 0.1, vel2


def test_gravity_convention_rejects_invalid_value():
    n = 10
    accel = np.tile(np.array([0.0, 0.0, 9.81]), (n, 1))
    gyro = np.zeros((n, 3))
    imu_dict = _imu_dict(accel, gyro)
    try:
        IMUPredictor(imu_dict, gravity_convention="sideways")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid gravity_convention")


# ---------------------------------------------------------------------------
# ComplementaryHeadingFilter heading-variance tracking
# ---------------------------------------------------------------------------


def test_heading_variance_defaults_to_zero_and_is_backward_compatible():
    imu_dict = _imu_dict(np.zeros((10, 3)), np.zeros((10, 3)))
    f = ComplementaryHeadingFilter(imu_dict)
    assert f.heading_variance_rad2 == 0.0
    f.update_heading_gyro(imu_dict["tow"][0], imu_dict["tow"][-1])
    # Zero gyro but nonzero dt still grows variance via the ARW model.
    assert f.heading_variance_rad2 > 0.0
    # Old call pattern (no sigma) must not touch variance beyond gyro growth.
    var_before = f.heading_variance_rad2
    f.correct_heading_spp(0.5)
    assert f.heading_variance_rad2 == var_before


def test_heading_variance_grows_with_gyro_integration_time():
    n = 500
    dt = 0.01
    imu_dict = _imu_dict(np.zeros((n, 3)), np.zeros((n, 3)), dt=dt)
    f = ComplementaryHeadingFilter(imu_dict, sigma_gyro_radps_sqrthz=0.01)
    f.update_heading_gyro(imu_dict["tow"][0], imu_dict["tow"][-1])
    T = float(imu_dict["tow"][-1] - imu_dict["tow"][0])
    expected = 0.01 ** 2 * T
    assert math.isclose(f.heading_variance_rad2, expected, rel_tol=1e-6)


def test_heading_variance_shrinks_on_confident_spp_correction():
    imu_dict = _imu_dict(np.zeros((10, 3)), np.zeros((10, 3)))
    f = ComplementaryHeadingFilter(imu_dict, alpha=0.3)
    f.heading_variance_rad2 = 1.0  # large prior uncertainty
    # A very confident SPP heading measurement should pull variance down.
    f.correct_heading_spp(0.1, sigma_spp_heading_rad=0.001)
    expected = (1.0 - 0.3) ** 2 * 1.0 + (0.3 ** 2) * (0.001 ** 2)
    assert math.isclose(f.heading_variance_rad2, expected, rel_tol=1e-9)
    assert f.heading_variance_rad2 < 1.0


def test_heading_variance_ignores_nonfinite_sigma():
    imu_dict = _imu_dict(np.zeros((10, 3)), np.zeros((10, 3)))
    f = ComplementaryHeadingFilter(imu_dict)
    f.heading_variance_rad2 = 0.5
    f.correct_heading_spp(0.1, sigma_spp_heading_rad=float("nan"))
    assert f.heading_variance_rad2 == 0.5
    f.correct_heading_spp(0.1, sigma_spp_heading_rad=float("inf"))
    assert f.heading_variance_rad2 == 0.5
