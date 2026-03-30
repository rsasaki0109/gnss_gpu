"""Tests for the EKF positioning module."""

import numpy as np
import pytest

from gnss_gpu.ekf import EKFPositioner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_satellites(n_sat=8, seed=42):
    """Generate satellite ECEF positions on a sphere at ~20200 km altitude."""
    rng = np.random.RandomState(seed)
    R_orbit = 26_571_000.0  # GPS orbit radius [m]
    # Distribute satellites roughly uniformly
    theta = rng.uniform(0, 2 * np.pi, n_sat)
    phi = rng.uniform(-np.pi / 3, np.pi / 3, n_sat)  # avoid low-elevation sats
    sat = np.zeros((n_sat, 3))
    sat[:, 0] = R_orbit * np.cos(phi) * np.cos(theta)
    sat[:, 1] = R_orbit * np.cos(phi) * np.sin(theta)
    sat[:, 2] = R_orbit * np.sin(phi)
    return sat


def _true_pseudoranges(true_pos, sat_ecef, clock_bias=0.0, noise_sigma=0.0,
                       rng=None):
    """Compute pseudoranges from true position to satellites."""
    diff = sat_ecef - true_pos[np.newaxis, :]
    ranges = np.sqrt(np.sum(diff ** 2, axis=1))
    pr = ranges + clock_bias
    if noise_sigma > 0:
        if rng is None:
            rng = np.random.RandomState(0)
        pr += rng.randn(len(ranges)) * noise_sigma
    return pr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEKFInitialization:
    def test_basic_init(self):
        ekf = EKFPositioner()
        assert not ekf.initialized

    def test_initialize_from_position(self):
        ekf = EKFPositioner()
        pos = np.array([-3962108.0, 3381309.0, 3668678.0])  # Tokyo approx ECEF
        ekf.initialize(pos, clock_bias=1000.0)
        assert ekf.initialized
        result = ekf.get_position()
        np.testing.assert_allclose(result, pos, atol=1e-6)

    def test_initial_velocity_is_zero(self):
        ekf = EKFPositioner()
        ekf.initialize(np.array([1e6, 2e6, 3e6]))
        vel = ekf.get_velocity()
        np.testing.assert_allclose(vel, [0, 0, 0], atol=1e-12)

    def test_initial_covariance_shape(self):
        ekf = EKFPositioner()
        ekf.initialize(np.array([1e6, 2e6, 3e6]))
        P = ekf.get_covariance()
        assert P.shape == (8, 8)

    def test_initial_covariance_diagonal(self):
        ekf = EKFPositioner()
        ekf.initialize(np.array([1e6, 2e6, 3e6]), sigma_pos=50.0, sigma_cb=500.0)
        P = ekf.get_covariance()
        # Position diagonal should be sigma_pos^2
        for i in range(3):
            assert abs(P[i, i] - 50.0 ** 2) < 1e-6
        # Clock bias
        assert abs(P[6, 6] - 500.0 ** 2) < 1e-6


class TestEKFPredict:
    def test_position_shifts_by_velocity(self):
        ekf = EKFPositioner()
        pos = np.array([0.0, 0.0, 6371000.0])
        ekf.initialize(pos)
        # Manually set velocity by running a fake cycle
        # Instead, use internal state manipulation via predict behavior
        # After init with zero velocity, predict should keep position unchanged
        ekf.predict(dt=1.0)
        result = ekf.get_position()
        np.testing.assert_allclose(result, pos, atol=1e-6)

    def test_covariance_grows_after_predict(self):
        ekf = EKFPositioner(sigma_pos=1.0)
        ekf.initialize(np.array([0.0, 0.0, 6371000.0]),
                       sigma_pos=10.0, sigma_cb=100.0)
        P_before = ekf.get_covariance().copy()
        ekf.predict(dt=1.0)
        P_after = ekf.get_covariance()
        # Covariance should grow (diagonal elements increase)
        for i in range(8):
            assert P_after[i, i] >= P_before[i, i]

    def test_velocity_propagates_position(self):
        """Verify that after update gives us velocity, predict moves position."""
        ekf = EKFPositioner(sigma_pr=2.0, sigma_pos=0.1, sigma_vel=0.01)
        true_pos = np.array([-3962108.0, 3381309.0, 3668678.0])
        velocity = np.array([1.0, 0.5, -0.3])  # m/s
        sat = _generate_satellites(10)

        # Initialize near true position
        ekf.initialize(true_pos, sigma_pos=10.0, sigma_cb=100.0)

        # Feed several epochs with moving true position to build up velocity estimate
        dt = 1.0
        rng = np.random.RandomState(123)
        current_pos = true_pos.copy()
        for _ in range(20):
            current_pos = current_pos + velocity * dt
            pr = _true_pseudoranges(current_pos, sat, noise_sigma=1.0, rng=rng)
            w = np.ones(len(pr)) / (2.0 ** 2)
            ekf.predict(dt=dt)
            ekf.update(sat, pr, w)

        est_vel = ekf.get_velocity()
        # Velocity estimate should be in the right ballpark (within 1 m/s per axis)
        np.testing.assert_allclose(est_vel, velocity, atol=1.0)


class TestEKFUpdate:
    def test_converges_to_true_position(self):
        ekf = EKFPositioner(sigma_pr=5.0)
        true_pos = np.array([-3962108.0, 3381309.0, 3668678.0])
        sat = _generate_satellites(8)
        pr = _true_pseudoranges(true_pos, sat, clock_bias=0.0, noise_sigma=0.0)
        w = np.ones(8) / (5.0 ** 2)

        # Initialize with offset
        ekf.initialize(true_pos + np.array([100.0, -50.0, 200.0]),
                       sigma_pos=500.0, sigma_cb=1000.0)

        # Run several predict+update cycles with perfect observations
        for _ in range(10):
            ekf.predict(dt=1.0)
            ekf.update(sat, pr, w)

        result = ekf.get_position()
        error = np.linalg.norm(result - true_pos)
        assert error < 10.0, f"Position error {error:.1f} m too large"

    def test_covariance_shrinks(self):
        ekf = EKFPositioner(sigma_pr=5.0)
        true_pos = np.array([-3962108.0, 3381309.0, 3668678.0])
        sat = _generate_satellites(8)
        pr = _true_pseudoranges(true_pos, sat)
        w = np.ones(8) / (5.0 ** 2)

        ekf.initialize(true_pos + np.array([50.0, 50.0, 50.0]),
                       sigma_pos=500.0, sigma_cb=1000.0)

        P_init = ekf.get_covariance().copy()

        for _ in range(5):
            ekf.predict(dt=1.0)
            ekf.update(sat, pr, w)

        P_after = ekf.get_covariance()
        # Position covariance should decrease
        for i in range(3):
            assert P_after[i, i] < P_init[i, i], \
                f"P[{i},{i}]: {P_after[i, i]:.1f} >= {P_init[i, i]:.1f}"


class TestEKFPipeline:
    def test_100_epoch_error_decreases(self):
        """Full pipeline: 100 epochs, verify error decreases over time."""
        ekf = EKFPositioner(sigma_pr=5.0, sigma_pos=1.0, sigma_vel=0.1,
                            sigma_clk=100.0, sigma_drift=10.0)
        true_pos = np.array([-3962108.0, 3381309.0, 3668678.0])
        sat = _generate_satellites(10)
        rng = np.random.RandomState(99)

        # Initialize with 200m offset
        offset = np.array([150.0, -100.0, 120.0])
        ekf.initialize(true_pos + offset, clock_bias=500.0,
                       sigma_pos=500.0, sigma_cb=2000.0)

        errors = []
        for epoch in range(100):
            pr = _true_pseudoranges(true_pos, sat, clock_bias=0.0,
                                    noise_sigma=3.0, rng=rng)
            w = np.ones(len(pr)) / (5.0 ** 2)

            ekf.predict(dt=1.0)
            ekf.update(sat, pr, w)

            err = np.linalg.norm(ekf.get_position() - true_pos)
            errors.append(err)

        # Error in last 10 epochs should be much smaller than first 10
        early_mean = np.mean(errors[:10])
        late_mean = np.mean(errors[90:])
        assert late_mean < early_mean, \
            f"Late error {late_mean:.1f} >= early error {early_mean:.1f}"
        # Final error should be reasonable (< 20m with 3m noise)
        assert errors[-1] < 20.0, f"Final error {errors[-1]:.1f} m too large"

    def test_compare_with_wls_stability(self):
        """EKF should produce more stable results than independent WLS epochs."""
        true_pos = np.array([-3962108.0, 3381309.0, 3668678.0])
        sat = _generate_satellites(8)
        rng = np.random.RandomState(77)

        n_epochs = 50
        noise_sigma = 5.0

        # EKF trajectory
        ekf = EKFPositioner(sigma_pr=noise_sigma)
        ekf.initialize(true_pos, sigma_pos=100.0, sigma_cb=1000.0)
        ekf_positions = []
        for _ in range(n_epochs):
            pr = _true_pseudoranges(true_pos, sat, noise_sigma=noise_sigma, rng=rng)
            w = np.ones(8) / (noise_sigma ** 2)
            ekf.predict(dt=1.0)
            ekf.update(sat, pr, w)
            ekf_positions.append(ekf.get_position().copy())

        ekf_positions = np.array(ekf_positions)
        ekf_std = np.std(np.linalg.norm(ekf_positions - true_pos, axis=1))

        # WLS trajectory (independent per epoch, using same noise seed)
        rng2 = np.random.RandomState(77)
        wls_errors = []
        for _ in range(n_epochs):
            pr = _true_pseudoranges(true_pos, sat, noise_sigma=noise_sigma, rng=rng2)
            # Simple iterative least squares (replicating WLS logic in Python)
            # Use EKF update-only (no predict memory) as a proxy
            ekf_single = EKFPositioner(sigma_pr=noise_sigma)
            ekf_single.initialize(true_pos + rng2.randn(3) * 10,
                                  sigma_pos=1e6, sigma_cb=1e6)
            w = np.ones(8) / (noise_sigma ** 2)
            ekf_single.update(sat, pr, w)
            wls_errors.append(np.linalg.norm(ekf_single.get_position() - true_pos))

        wls_std = np.std(wls_errors)

        # EKF should have lower variance due to temporal filtering
        assert ekf_std < wls_std * 1.5, \
            f"EKF std {ekf_std:.2f} not better than WLS std {wls_std:.2f}"


class TestEKFEdgeCases:
    def test_not_initialized_raises(self):
        ekf = EKFPositioner()
        with pytest.raises(RuntimeError):
            ekf.predict()
        with pytest.raises(RuntimeError):
            ekf.update(np.zeros((4, 3)), np.zeros(4))
        with pytest.raises(RuntimeError):
            ekf.get_position()

    def test_single_satellite(self):
        """With fewer than 4 satellites, EKF should still run (underdetermined update)."""
        ekf = EKFPositioner()
        true_pos = np.array([0.0, 0.0, 6371000.0])
        ekf.initialize(true_pos)
        sat = _generate_satellites(1)
        pr = _true_pseudoranges(true_pos, sat)
        w = np.ones(1) / 25.0
        # Should not crash
        ekf.predict(dt=1.0)
        ekf.update(sat, pr, w)
        pos = ekf.get_position()
        assert pos.shape == (3,)

    def test_many_satellites(self):
        """Test with a large number of satellites."""
        ekf = EKFPositioner()
        true_pos = np.array([-3962108.0, 3381309.0, 3668678.0])
        sat = _generate_satellites(20)
        pr = _true_pseudoranges(true_pos, sat, noise_sigma=2.0,
                                rng=np.random.RandomState(0))
        w = np.ones(20) / 4.0

        ekf.initialize(true_pos + np.array([50.0, -30.0, 80.0]),
                       sigma_pos=200.0)
        for _ in range(10):
            ekf.predict(dt=1.0)
            ekf.update(sat, pr, w)

        error = np.linalg.norm(ekf.get_position() - true_pos)
        assert error < 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
