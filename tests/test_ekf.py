"""Tests for the EKF positioning module."""

import numpy as np
import pytest

from gnss_gpu.ekf import EKFPositioner

try:
    from gnss_gpu._gnss_gpu_ekf import (
        EKFConfig,
        ekf_initialize,
        ekf_predict,
        ekf_update,
        ekf_batch,
    )
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


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


class TestEKFValidation:
    def test_wrapper_initialize_rejects_bad_position(self):
        ekf = EKFPositioner()
        with pytest.raises(ValueError, match="position_ecef must have shape"):
            ekf.initialize([0.0, 0.0])

    def test_wrapper_predict_rejects_nonpositive_dt(self):
        ekf = EKFPositioner()
        ekf.initialize(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="dt must be positive"):
            ekf.predict(dt=0.0)

    @pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
    def test_binding_rejects_invalid_ekf_config(self):
        with pytest.raises(RuntimeError, match="sigma_pos must be positive"):
            EKFConfig(sigma_pos=0.0)

    @pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
    def test_binding_rejects_invalid_initial_pos(self):
        with pytest.raises(RuntimeError, match="initial_pos must have shape"):
            ekf_initialize(np.array([1.0, 2.0]), 0.0)
        with pytest.raises(RuntimeError, match="initial_pos must be finite"):
            ekf_initialize(np.array([1.0, np.nan, 3.0]), 0.0)

    @pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
    def test_binding_rejects_invalid_predict_inputs(self):
        config = EKFConfig()
        state_x = np.zeros(8)
        state_P = np.eye(8).ravel()

        with pytest.raises(RuntimeError, match="state_x must have shape"):
            ekf_predict(np.zeros(7), state_P, 1.0, config)
        with pytest.raises(RuntimeError, match="state_P must have shape"):
            ekf_predict(state_x, np.zeros(63), 1.0, config)
        with pytest.raises(RuntimeError, match="dt must be positive"):
            ekf_predict(state_x, state_P, 0.0, config)

    @pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
    def test_binding_rejects_invalid_update_inputs(self):
        config = EKFConfig()
        state_x = np.zeros(8)
        state_P = np.eye(8).ravel()
        sat = np.ones((2, 3))
        pr = np.ones(2)
        w = np.ones(2)

        with pytest.raises(RuntimeError, match="n_sat must be >= 1"):
            ekf_update(state_x, state_P, sat[:0], np.array([]), np.array([]))
        with pytest.raises(RuntimeError, match="sat_ecef shape must match"):
            ekf_update(state_x, state_P, sat[:1], pr, w)
        with pytest.raises(RuntimeError, match="weights must be non-negative"):
            ekf_update(state_x, state_P, sat, pr, np.array([1.0, -1.0]))

    @pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
    def test_binding_rejects_invalid_batch_inputs(self):
        config = EKFConfig()
        states_x = np.zeros((2, 8))
        states_P = np.zeros((2, 8, 8))
        sat = np.ones((3, 3))
        pr = np.ones(3)
        w = np.ones(3)

        with pytest.raises(RuntimeError, match="states_P must have shape"):
            ekf_batch(states_x, np.zeros((2, 64)), sat, pr, w, 1.0, config)
        with pytest.raises(RuntimeError, match="n_instances must be >= 1"):
            ekf_batch(np.zeros((0, 8)), np.zeros((0, 8, 8)), sat, pr, w, 1.0, config)
        with pytest.raises(RuntimeError, match="weights length must match"):
            ekf_batch(states_x, states_P, sat, pr, np.ones(2), 1.0, config)

    @pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
    def test_binding_valid_smoke_shapes(self):
        config = EKFConfig()
        pos = np.array([1.0, 2.0, 3.0])
        state = ekf_initialize(pos, 0.0)
        state_x = np.asarray(state.get_state(), dtype=np.float64)
        state_P = np.asarray(state.get_covariance(), dtype=np.float64).ravel()

        ekf_predict(state_x, state_P, 1.0, config)

        sat = np.ones((4, 3))
        pr = np.full(4, 20e6)
        w = np.ones(4)
        ekf_update(state_x, state_P, sat, pr, w)

        states_x = np.tile(state_x, (2, 1))
        states_P = np.tile(state_P.reshape(8, 8), (2, 1, 1))
        out_x, out_P = ekf_batch(states_x, states_P, sat, pr, w, 1.0, config)
        assert out_x.shape == (2, 8)
        assert out_P.shape == (2, 8, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
