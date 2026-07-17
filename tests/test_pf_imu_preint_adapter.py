"""Unit tests for the WP21 PF <-> IMU-preintegration adapter (no GPU required)."""

from __future__ import annotations

import numpy as np

from gnss_gpu.pf_imu_preint_adapter import (
    ImuPreintPfGuide,
    body_to_ecef_frame,
    ecef_to_enu_rotation,
    ecef_to_lla_rad,
    imu_preint_predict,
)

_TOKYO_ECEF = np.array([-3959955.0, 3348757.0, 3699287.0])  # roughly Tokyo


class _StubHeadingFilter:
    """Minimal stand-in for gnss_gpu.imu.ComplementaryHeadingFilter."""

    def __init__(self, heading0: float = 0.0):
        self.heading = heading0
        self.corrections = []

    def correct_heading_spp(self, spp_heading_rad: float) -> None:
        self.corrections.append(spp_heading_rad)
        self.heading = spp_heading_rad  # full snap for test determinism


class _StubPF:
    """Minimal stand-in for ParticleFilterDeviceRuntime, records predict() calls."""

    def __init__(self, position_ecef: np.ndarray):
        self._pos = np.asarray(position_ecef, dtype=np.float64)
        self.calls: list[dict] = []

    def estimate(self) -> np.ndarray:
        return np.concatenate([self._pos, [0.0]])

    def predict(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if kwargs.get("velocity") is not None:
            self._pos = self._pos + np.asarray(kwargs["velocity"]) * kwargs["dt"]


def test_ecef_to_lla_roundtrip_reasonable():
    lat, lon = ecef_to_lla_rad(_TOKYO_ECEF)
    # Tokyo: ~35.6N, ~139.7E
    assert 0.5 < lat < 0.7
    assert 2.3 < lon < 2.5


def test_body_to_ecef_frame_is_orthonormal():
    R_be, R_enu_ecef = body_to_ecef_frame(0.7, _TOKYO_ECEF)
    assert np.allclose(R_be @ R_be.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R_be) - 1.0) < 1e-9
    assert np.allclose(R_enu_ecef @ ecef_to_enu_rotation(*ecef_to_lla_rad(_TOKYO_ECEF)), np.eye(3), atol=1e-9)


def test_body_forward_maps_to_expected_enu_direction_north_heading():
    # heading = 0 (north): body-x forward should map (in ENU) to (0, 1, 0).
    lat, lon = ecef_to_lla_rad(_TOKYO_ECEF)
    R_be, R_enu_ecef = body_to_ecef_frame(0.0, _TOKYO_ECEF)
    R_be_enu = R_enu_ecef.T @ R_be  # body -> ENU directly
    forward_enu = R_be_enu @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(forward_enu, [0.0, 1.0, 0.0], atol=1e-9)
    up_enu = R_be_enu @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(up_enu, [0.0, 0.0, 1.0], atol=1e-9)


def test_static_segment_gravity_only_yields_near_zero_displacement():
    """A perfectly static IMU (accel=(0,0,g), gyro=0) should not produce a
    large phantom displacement guide once gravity is compensated."""

    guide = ImuPreintPfGuide(_StubHeadingFilter(heading0=0.3))
    dt_sample = 0.01
    n = 20  # 0.2s epoch at 100 Hz
    for _ in range(n):
        guide.add_sample(np.array([0.0, 0.0, 9.81]), np.zeros(3), dt_sample)

    p_i = _TOKYO_ECEF
    velocity_guide, sigma_pos_eff = guide.close_segment(p_i, dt=n * dt_sample, v_gnss_ecef=np.zeros(3))
    assert velocity_guide is not None
    assert sigma_pos_eff is not None and sigma_pos_eff > 0.0
    # Should be near zero, not the ~4.9 m/s that an uncompensated 1g bias
    # would otherwise inject via 0.5*g*dt^2.
    assert np.linalg.norm(velocity_guide) < 0.05, velocity_guide


def test_close_segment_returns_none_when_no_samples():
    guide = ImuPreintPfGuide(_StubHeadingFilter())
    velocity_guide, sigma_pos_eff = guide.close_segment(_TOKYO_ECEF, dt=0.2)
    assert velocity_guide is None
    assert sigma_pos_eff is None


def test_imu_preint_predict_falls_back_to_cv_when_segment_empty():
    pf = _StubPF(_TOKYO_ECEF)
    guide = ImuPreintPfGuide(_StubHeadingFilter())
    imu_preint_predict(pf, guide, dt=0.2)
    assert len(pf.calls) == 1
    assert pf.calls[0]["velocity"] is None


def test_imu_preint_predict_uses_guide_and_resets_segment():
    pf = _StubPF(_TOKYO_ECEF)
    guide = ImuPreintPfGuide(_StubHeadingFilter(heading0=0.0))
    dt_sample = 0.01
    for _ in range(20):
        # Small constant forward-ish acceleration on top of gravity.
        guide.add_sample(np.array([0.5, 0.0, 9.81]), np.zeros(3), dt_sample)
    assert guide.n_samples == 20

    imu_preint_predict(pf, guide, dt=0.2, v_gnss_ecef=np.zeros(3))
    assert len(pf.calls) == 1
    call = pf.calls[0]
    assert call["velocity"] is not None
    assert call["sigma_pos"] > 0.0
    assert call["rbpf_velocity_kf"] is False
    # Segment must be reset after closing.
    assert guide.n_samples == 0
