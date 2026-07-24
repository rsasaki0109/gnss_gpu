"""Unit tests for the WP21 PF <-> IMU-preintegration adapter (no GPU required)."""

from __future__ import annotations

import math

import numpy as np

from gnss_gpu.imu import ComplementaryHeadingFilter
from gnss_gpu.pf_imu_preint_adapter import (
    ImuPreintPfGuide,
    body_to_ecef_frame,
    ecef_to_enu_rotation,
    ecef_to_lla_rad,
    imu_preint_predict,
    imu_preint_predict_velocity_kf,
)

_TOKYO_ECEF = np.array([-3959955.0, 3348757.0, 3699287.0])  # roughly Tokyo


class _StubHeadingFilter:
    """Minimal stand-in for gnss_gpu.imu.ComplementaryHeadingFilter.

    Deliberately has no `heading_variance_rad2` attribute and a
    single-argument `correct_heading_spp`, to exercise
    `ImuPreintPfGuide`'s backward-compatible (use_heading_uncertainty=False)
    default path.
    """

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
        self.velocity_cov_calls: list[np.ndarray] = []

    def estimate(self) -> np.ndarray:
        return np.concatenate([self._pos, [0.0]])

    def predict(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if kwargs.get("velocity") is not None:
            self._pos = self._pos + np.asarray(kwargs["velocity"]) * kwargs["dt"]

    def set_velocity_covariance(self, cov_3x3) -> None:
        self.velocity_cov_calls.append(np.asarray(cov_3x3, dtype=np.float64).copy())


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


# ---------------------------------------------------------------------------
# WP21b item 1: heading-uncertainty propagation into sigma_pos
# ---------------------------------------------------------------------------


def _feed_forward_segment(guide: ImuPreintPfGuide, n: int = 20, dt_sample: float = 0.01,
                           forward_accel: float = 1.0) -> None:
    for _ in range(n):
        guide.add_sample(np.array([forward_accel, 0.0, 9.81]), np.zeros(3), dt_sample)


def test_heading_uncertainty_disabled_by_default_matches_v1_exactly():
    """use_heading_uncertainty defaults False: sigma_pos must be identical to
    the accel/gyro-covariance-only (WP21 Phase A) formula, independent of
    heading_variance_rad2."""

    heading_filter = ComplementaryHeadingFilter({"tow": np.zeros(1), "gyro": np.zeros((1, 3)),
                                                  "wheel_vel": np.full(1, np.nan)})
    heading_filter.heading_variance_rad2 = 10.0  # large, must be ignored
    guide = ImuPreintPfGuide(heading_filter, sigma_pos_floor=0.01)
    _feed_forward_segment(guide)
    _, sigma_pos_with_var = guide.close_segment(_TOKYO_ECEF, dt=0.2, v_gnss_ecef=np.zeros(3))

    heading_filter2 = ComplementaryHeadingFilter({"tow": np.zeros(1), "gyro": np.zeros((1, 3)),
                                                   "wheel_vel": np.full(1, np.nan)})
    guide2 = ImuPreintPfGuide(heading_filter2, sigma_pos_floor=0.01)
    _feed_forward_segment(guide2)
    _, sigma_pos_zero_var = guide2.close_segment(_TOKYO_ECEF, dt=0.2, v_gnss_ecef=np.zeros(3))

    assert math.isclose(sigma_pos_with_var, sigma_pos_zero_var, rel_tol=1e-12)


def test_heading_uncertainty_enabled_grows_sigma_pos_with_heading_variance():
    """With use_heading_uncertainty=True, a larger heading_variance_rad2
    must strictly increase the resulting sigma_pos (cross-track lever)."""

    def _sigma_pos_for_variance(var: float) -> float:
        heading_filter = ComplementaryHeadingFilter(
            {"tow": np.zeros(1), "gyro": np.zeros((1, 3)), "wheel_vel": np.full(1, np.nan)}
        )
        heading_filter.heading_variance_rad2 = var
        guide = ImuPreintPfGuide(
            heading_filter, sigma_pos_floor=1e-6, use_heading_uncertainty=True
        )
        _feed_forward_segment(guide, forward_accel=20.0)  # sizable displacement
        _, sigma_pos = guide.close_segment(_TOKYO_ECEF, dt=0.2, v_gnss_ecef=np.zeros(3))
        return sigma_pos

    sigma_low = _sigma_pos_for_variance(0.0)
    sigma_high = _sigma_pos_for_variance((math.radians(5.0)) ** 2)
    assert sigma_high > sigma_low


def test_heading_uncertainty_matches_cross_track_lever_formula():
    """sigma_pos_heading term should equal |displacement| * sigma_heading_rad
    combined in quadrature with the accel/gyro covariance term."""

    heading_filter = ComplementaryHeadingFilter(
        {"tow": np.zeros(1), "gyro": np.zeros((1, 3)), "wheel_vel": np.full(1, np.nan)}
    )
    heading_var = (math.radians(3.0)) ** 2
    heading_filter.heading_variance_rad2 = heading_var
    guide = ImuPreintPfGuide(heading_filter, sigma_pos_floor=1e-6, use_heading_uncertainty=True)
    _feed_forward_segment(guide, forward_accel=5.0)
    velocity_guide, sigma_pos = guide.close_segment(_TOKYO_ECEF, dt=0.2, v_gnss_ecef=np.zeros(3))

    displacement_m = float(np.linalg.norm(velocity_guide) * 0.2)
    sigma_pos_heading_expected = displacement_m * math.sqrt(heading_var)
    # sigma_pos >= the heading-only term (quadrature sum with a nonnegative
    # accel/gyro covariance term can only add, never subtract).
    assert sigma_pos >= sigma_pos_heading_expected - 1e-9


def test_sigma_spp_heading_rad_uninformative_when_displacement_small():
    heading_filter = ComplementaryHeadingFilter(
        {"tow": np.zeros(1), "gyro": np.zeros((1, 3)), "wheel_vel": np.full(1, np.nan)}
    )
    guide = ImuPreintPfGuide(heading_filter, use_heading_uncertainty=True,
                              min_heading_fix_disp_m=2.0)
    assert guide._sigma_spp_heading_rad(None) == math.pi
    assert guide._sigma_spp_heading_rad(0.5) == math.pi
    assert guide._sigma_spp_heading_rad(float("nan")) == math.pi
    informative = guide._sigma_spp_heading_rad(50.0)
    assert 0.0 < informative < math.pi


def test_close_segment_updates_heading_filter_variance_when_enabled():
    heading_filter = ComplementaryHeadingFilter(
        {"tow": np.zeros(1), "gyro": np.zeros((1, 3)), "wheel_vel": np.full(1, np.nan)}
    )
    heading_filter.heading_variance_rad2 = 1.0
    guide = ImuPreintPfGuide(heading_filter, use_heading_uncertainty=True)
    _feed_forward_segment(guide)
    guide.close_segment(
        _TOKYO_ECEF, dt=0.2, v_gnss_ecef=np.zeros(3),
        spp_heading_rad=0.1, spp_displacement_m=50.0,
    )
    # A confident SPP correction (large displacement -> small sigma) should
    # shrink the prior heading variance.
    assert heading_filter.heading_variance_rad2 < 1.0


# ---------------------------------------------------------------------------
# WP21b item 2: per-particle velocity-KF feeding
# ---------------------------------------------------------------------------


def test_velocity_covariance_ecef_none_before_first_close_segment():
    guide = ImuPreintPfGuide(_StubHeadingFilter())
    assert guide.velocity_covariance_ecef is None


def test_velocity_covariance_ecef_populated_and_psd_after_close_segment():
    guide = ImuPreintPfGuide(_StubHeadingFilter(heading0=0.2))
    _feed_forward_segment(guide)
    guide.close_segment(_TOKYO_ECEF, dt=0.2, v_gnss_ecef=np.zeros(3))
    cov = guide.velocity_covariance_ecef
    assert cov is not None
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-12)


def test_imu_preint_predict_velocity_kf_sets_covariance_before_predict():
    pf = _StubPF(_TOKYO_ECEF)
    guide = ImuPreintPfGuide(_StubHeadingFilter(heading0=0.0))
    _feed_forward_segment(guide)

    imu_preint_predict_velocity_kf(pf, guide, dt=0.2, v_gnss_ecef=np.zeros(3))

    assert len(pf.velocity_cov_calls) == 1
    assert pf.velocity_cov_calls[0].shape == (3, 3)
    assert len(pf.calls) == 1
    call = pf.calls[0]
    assert call["rbpf_velocity_kf"] is True
    assert call["velocity_guide_alpha"] == 1.0
    assert guide.n_samples == 0


def test_imu_preint_predict_velocity_kf_falls_back_to_cv_when_segment_empty():
    pf = _StubPF(_TOKYO_ECEF)
    guide = ImuPreintPfGuide(_StubHeadingFilter())
    imu_preint_predict_velocity_kf(pf, guide, dt=0.2)
    assert len(pf.calls) == 1
    assert pf.calls[0]["velocity"] is None
    assert len(pf.velocity_cov_calls) == 0
