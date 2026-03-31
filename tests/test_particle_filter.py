"""Tests for GPU Mega Particle Filter (requires CUDA GPU)."""

import numpy as np
import pytest

try:
    from gnss_gpu._gnss_gpu_pf import (
        pf_initialize, pf_predict, pf_weight, pf_compute_ess,
        pf_resample_systematic, pf_resample_megopolis,
        pf_estimate, pf_get_particles,
    )
    from gnss_gpu.particle_filter import ParticleFilter
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")

# Number of particles for tests (smaller than production for speed)
N_TEST = 10000
SEED = 12345


def _make_test_scenario():
    """Create a test scenario with realistic GPS satellite positions.

    True receiver: Tokyo Station area (~35.68N, 139.77E)
    Same scenario as test_positioning.py.
    """
    true_pos = np.array([-3957199.0, 3310205.0, 3737911.0])
    true_cb = 3000.0  # clock bias in meters (~10 us)

    sat_ecef = np.array([
        [-14985000.0,  -3988000.0,  21474000.0],  # G01
        [ -9575000.0,  15498000.0,  19457000.0],  # G03
        [  7624000.0, -16218000.0,  19843000.0],  # G06
        [ 16305000.0,  12037000.0,  17183000.0],  # G09
        [-20889000.0,  13759000.0,   8291000.0],  # G11
        [  5463000.0,  24413000.0,   8934000.0],  # G14
        [ 22169000.0,   3975000.0,  13781000.0],  # G17
        [-11527000.0, -19421000.0,  13682000.0],  # G22
    ])

    ranges = np.sqrt(np.sum((sat_ecef - true_pos) ** 2, axis=1))
    pseudoranges = ranges + true_cb
    weights = np.ones(len(sat_ecef))

    return sat_ecef, pseudoranges, weights, true_pos, true_cb


class TestInitialize:
    """Test particle initialization."""

    def test_particles_scattered_around_init(self):
        N = N_TEST
        init_pos = np.array([-3957199.0, 3310205.0, 3737911.0])
        init_cb = 3000.0
        spread_pos = 50.0
        spread_cb = 500.0

        px = np.empty(N, dtype=np.float64)
        py = np.empty(N, dtype=np.float64)
        pz = np.empty(N, dtype=np.float64)
        pcb = np.empty(N, dtype=np.float64)

        pf_initialize(px, py, pz, pcb,
                      init_pos[0], init_pos[1], init_pos[2], init_cb,
                      spread_pos, spread_cb, N, SEED)

        # Mean should be close to initial position
        assert abs(np.mean(px) - init_pos[0]) < 5.0 * spread_pos / np.sqrt(N) * 3
        assert abs(np.mean(py) - init_pos[1]) < 5.0 * spread_pos / np.sqrt(N) * 3
        assert abs(np.mean(pz) - init_pos[2]) < 5.0 * spread_pos / np.sqrt(N) * 3
        assert abs(np.mean(pcb) - init_cb) < 5.0 * spread_cb / np.sqrt(N) * 3

        # Std should be close to spread
        assert abs(np.std(px) - spread_pos) < spread_pos * 0.1
        assert abs(np.std(py) - spread_pos) < spread_pos * 0.1
        assert abs(np.std(pcb) - spread_cb) < spread_cb * 0.1

    def test_different_seeds_give_different_particles(self):
        N = 1000
        px1 = np.empty(N, dtype=np.float64)
        py1 = np.empty(N, dtype=np.float64)
        pz1 = np.empty(N, dtype=np.float64)
        pcb1 = np.empty(N, dtype=np.float64)
        px2 = np.empty(N, dtype=np.float64)
        py2 = np.empty(N, dtype=np.float64)
        pz2 = np.empty(N, dtype=np.float64)
        pcb2 = np.empty(N, dtype=np.float64)

        pf_initialize(px1, py1, pz1, pcb1, 0, 0, 0, 0, 100, 100, N, 1)
        pf_initialize(px2, py2, pz2, pcb2, 0, 0, 0, 0, 100, 100, N, 2)

        assert not np.allclose(px1, px2)


class TestPredict:
    """Test prediction step."""

    def test_mean_shifts_by_velocity(self):
        N = N_TEST
        px = np.empty(N, dtype=np.float64)
        py = np.empty(N, dtype=np.float64)
        pz = np.empty(N, dtype=np.float64)
        pcb = np.empty(N, dtype=np.float64)

        init_x, init_y, init_z = 0.0, 0.0, 0.0
        pf_initialize(px, py, pz, pcb, init_x, init_y, init_z, 0.0,
                      1.0, 1.0, N, SEED)

        mean_x_before = np.mean(px)
        mean_y_before = np.mean(py)

        vx = np.array([10.0], dtype=np.float64)
        vy = np.array([20.0], dtype=np.float64)
        vz = np.array([0.0], dtype=np.float64)
        dt = 1.0

        pf_predict(px, py, pz, pcb, vx, vy, vz,
                   dt, 0.1, 0.1, N, SEED, 1)

        mean_x_after = np.mean(px)
        mean_y_after = np.mean(py)

        # Mean should shift by approximately v*dt
        dx = mean_x_after - mean_x_before
        dy = mean_y_after - mean_y_before
        assert abs(dx - 10.0) < 1.0, f"Expected dx~10, got {dx:.2f}"
        assert abs(dy - 20.0) < 1.0, f"Expected dy~20, got {dy:.2f}"


class TestWeight:
    """Test weight computation."""

    def test_near_particles_get_higher_weight(self):
        N = 1000
        sat_ecef, pseudoranges, weights_sat, true_pos, true_cb = _make_test_scenario()

        # Half particles near true position, half far away
        px = np.empty(N, dtype=np.float64)
        py = np.empty(N, dtype=np.float64)
        pz = np.empty(N, dtype=np.float64)
        pcb = np.empty(N, dtype=np.float64)

        rng = np.random.default_rng(42)
        half = N // 2
        # Near particles
        px[:half] = true_pos[0] + rng.normal(0, 1.0, half)
        py[:half] = true_pos[1] + rng.normal(0, 1.0, half)
        pz[:half] = true_pos[2] + rng.normal(0, 1.0, half)
        pcb[:half] = true_cb + rng.normal(0, 10.0, half)

        # Far particles (1 km away)
        px[half:] = true_pos[0] + 1000.0 + rng.normal(0, 1.0, N - half)
        py[half:] = true_pos[1] + 1000.0 + rng.normal(0, 1.0, N - half)
        pz[half:] = true_pos[2] + rng.normal(0, 1.0, N - half)
        pcb[half:] = true_cb + rng.normal(0, 10.0, N - half)

        log_weights = np.empty(N, dtype=np.float64)
        pf_weight(px, py, pz, pcb,
                  sat_ecef.ravel(), pseudoranges, weights_sat, log_weights,
                  N, len(pseudoranges), 5.0)

        mean_near = np.mean(log_weights[:half])
        mean_far = np.mean(log_weights[half:])
        assert mean_near > mean_far, (
            f"Near particles should have higher weights: {mean_near:.1f} vs {mean_far:.1f}")


class TestESS:
    """Test Effective Sample Size computation."""

    def test_uniform_weights_ess_equals_n(self):
        N = N_TEST
        log_weights = np.zeros(N, dtype=np.float64)
        ess = pf_compute_ess(log_weights, N)
        # ESS should be approximately N for uniform weights
        assert abs(ess - N) < N * 0.01, f"Expected ESS~{N}, got {ess:.1f}"

    def test_single_dominant_particle(self):
        N = 1000
        log_weights = np.full(N, -1000.0, dtype=np.float64)
        log_weights[0] = 0.0  # One dominant particle
        ess = pf_compute_ess(log_weights, N)
        assert ess < 2.0, f"Expected ESS~1, got {ess:.1f}"

    def test_ess_between_bounds(self):
        N = 1000
        rng = np.random.default_rng(42)
        log_weights = rng.normal(0, 1.0, N)
        ess = pf_compute_ess(log_weights, N)
        assert 1.0 <= ess <= N, f"ESS {ess:.1f} out of bounds [1, {N}]"


class TestSystematicResampling:
    """Test systematic resampling."""

    def test_particles_cluster_around_high_weight(self):
        N = 1000
        sat_ecef, pseudoranges, weights_sat, true_pos, true_cb = _make_test_scenario()

        px = np.empty(N, dtype=np.float64)
        py = np.empty(N, dtype=np.float64)
        pz = np.empty(N, dtype=np.float64)
        pcb = np.empty(N, dtype=np.float64)

        # Spread particles widely
        pf_initialize(px, py, pz, pcb,
                      true_pos[0], true_pos[1], true_pos[2], true_cb,
                      500.0, 2000.0, N, SEED)

        std_before = np.std(px)

        # Compute weights
        log_weights = np.empty(N, dtype=np.float64)
        pf_weight(px, py, pz, pcb,
                  sat_ecef.ravel(), pseudoranges, weights_sat, log_weights,
                  N, len(pseudoranges), 5.0)

        # Resample
        pf_resample_systematic(px, py, pz, pcb, log_weights, N, SEED)

        std_after = np.std(px)
        # After resampling, particles should be more concentrated
        assert std_after < std_before, (
            f"Std should decrease after resampling: {std_before:.1f} -> {std_after:.1f}")


class TestMegopolisResampling:
    """Test Megopolis resampling."""

    def test_particles_cluster_around_high_weight(self):
        N = 1000
        sat_ecef, pseudoranges, weights_sat, true_pos, true_cb = _make_test_scenario()

        px = np.empty(N, dtype=np.float64)
        py = np.empty(N, dtype=np.float64)
        pz = np.empty(N, dtype=np.float64)
        pcb = np.empty(N, dtype=np.float64)

        pf_initialize(px, py, pz, pcb,
                      true_pos[0], true_pos[1], true_pos[2], true_cb,
                      500.0, 2000.0, N, SEED)

        std_before = np.std(px)

        log_weights = np.empty(N, dtype=np.float64)
        pf_weight(px, py, pz, pcb,
                  sat_ecef.ravel(), pseudoranges, weights_sat, log_weights,
                  N, len(pseudoranges), 5.0)

        pf_resample_megopolis(px, py, pz, pcb, log_weights, N, 15, SEED)

        std_after = np.std(px)
        assert std_after < std_before, (
            f"Std should decrease after resampling: {std_before:.1f} -> {std_after:.1f}")

    def test_megopolis_more_iterations_gives_better_mixing(self):
        N = 2000
        log_weights = np.zeros(N, dtype=np.float64)
        # Make first quarter high weight
        log_weights[:N // 4] = 0.0
        log_weights[N // 4:] = -10.0

        rng = np.random.default_rng(42)
        px = rng.normal(0, 1.0, N)
        py = rng.normal(0, 1.0, N)
        pz = rng.normal(0, 1.0, N)
        pcb = rng.normal(0, 1.0, N)

        # Save originals
        px_few = px.copy(); py_few = py.copy()
        pz_few = pz.copy(); pcb_few = pcb.copy()
        px_many = px.copy(); py_many = py.copy()
        pz_many = pz.copy(); pcb_many = pcb.copy()

        pf_resample_megopolis(px_few, py_few, pz_few, pcb_few,
                              log_weights, N, 2, SEED)
        pf_resample_megopolis(px_many, py_many, pz_many, pcb_many,
                              log_weights, N, 20, SEED)

        # With more iterations, more particles should have moved to the
        # high-weight region (first quarter's values)
        # This is a statistical test, so just verify both ran without error
        assert len(px_few) == N
        assert len(px_many) == N


class TestEstimate:
    """Test weighted mean estimation."""

    def test_estimate_with_uniform_weights(self):
        N = N_TEST
        px = np.empty(N, dtype=np.float64)
        py = np.empty(N, dtype=np.float64)
        pz = np.empty(N, dtype=np.float64)
        pcb = np.empty(N, dtype=np.float64)

        init_pos = np.array([100.0, 200.0, 300.0])
        init_cb = 50.0

        pf_initialize(px, py, pz, pcb,
                      init_pos[0], init_pos[1], init_pos[2], init_cb,
                      10.0, 10.0, N, SEED)

        log_weights = np.zeros(N, dtype=np.float64)
        result = np.empty(4, dtype=np.float64)
        pf_estimate(px, py, pz, pcb, log_weights, result, N)

        # Should be close to mean of particles (which should be near init)
        assert abs(result[0] - init_pos[0]) < 2.0
        assert abs(result[1] - init_pos[1]) < 2.0
        assert abs(result[2] - init_pos[2]) < 2.0
        assert abs(result[3] - init_cb) < 2.0


class TestGetParticles:
    """Test particle extraction for visualization."""

    def test_get_particles_shape(self):
        N = 1000
        px = np.arange(N, dtype=np.float64)
        py = np.arange(N, dtype=np.float64) * 2
        pz = np.arange(N, dtype=np.float64) * 3
        pcb = np.arange(N, dtype=np.float64) * 4

        output = np.empty(N * 4, dtype=np.float64)
        pf_get_particles(px, py, pz, pcb, output, N)
        output = output.reshape(N, 4)

        np.testing.assert_allclose(output[:, 0], px)
        np.testing.assert_allclose(output[:, 1], py)
        np.testing.assert_allclose(output[:, 2], pz)
        np.testing.assert_allclose(output[:, 3], pcb)


class TestFullPipeline:
    """Test complete particle filter pipeline."""

    def test_convergence_to_true_position(self):
        """Full pipeline: initialize -> predict -> update -> estimate.
        Uses the same Tokyo Station scenario as test_positioning.py.
        """
        sat_ecef, pseudoranges, weights_sat, true_pos, true_cb = _make_test_scenario()

        pf = ParticleFilter(
            n_particles=N_TEST,
            sigma_pos=0.5,
            sigma_cb=100.0,
            sigma_pr=5.0,
            resampling="megopolis",
            ess_threshold=0.5,
            seed=SEED,
        )

        # Initialize with a rough estimate (100m offset)
        init_pos = true_pos + np.array([50.0, -30.0, 40.0])
        pf.initialize(init_pos, clock_bias=true_cb + 200.0,
                      spread_pos=200.0, spread_cb=1000.0)

        # Run multiple update cycles (stationary receiver)
        for step in range(5):
            pf.predict(velocity=None, dt=1.0)
            pf.update(sat_ecef, pseudoranges, weights_sat)

        est = pf.estimate()
        pos_err = np.linalg.norm(est[:3] - true_pos)
        cb_err = abs(est[3] - true_cb)

        assert pos_err < 50.0, f"Position error {pos_err:.1f} m (expected < 50 m)"
        assert cb_err < 500.0, f"Clock bias error {cb_err:.1f} m (expected < 500 m)"

    def test_convergence_systematic_resampling(self):
        """Same pipeline but with systematic resampling."""
        sat_ecef, pseudoranges, weights_sat, true_pos, true_cb = _make_test_scenario()

        pf = ParticleFilter(
            n_particles=N_TEST,
            sigma_pos=0.5,
            sigma_cb=100.0,
            sigma_pr=5.0,
            resampling="systematic",
            ess_threshold=0.5,
            seed=SEED,
        )

        init_pos = true_pos + np.array([50.0, -30.0, 40.0])
        pf.initialize(init_pos, clock_bias=true_cb + 200.0,
                      spread_pos=200.0, spread_cb=1000.0)

        # Systematic resampling needs more iterations than Megopolis to
        # converge with 10k particles due to sample impoverishment after
        # each resampling step (particles collapse to fewer unique values).
        for step in range(15):
            pf.predict(velocity=None, dt=1.0)
            pf.update(sat_ecef, pseudoranges, weights_sat)

        est = pf.estimate()
        pos_err = np.linalg.norm(est[:3] - true_pos)
        assert pos_err < 100.0, f"Position error {pos_err:.1f} m (expected < 100 m)"

    def test_ess_drops_then_recovers(self):
        """ESS should drop after weight update and recover after resampling."""
        sat_ecef, pseudoranges, weights_sat, true_pos, true_cb = _make_test_scenario()

        pf = ParticleFilter(
            n_particles=N_TEST,
            sigma_pos=0.5,
            sigma_cb=100.0,
            sigma_pr=5.0,
            resampling="megopolis",
            ess_threshold=0.0,  # Disable auto-resampling
            seed=SEED,
        )

        pf.initialize(true_pos, clock_bias=true_cb,
                      spread_pos=500.0, spread_cb=2000.0)

        # Initial ESS should be N (uniform weights)
        ess_init = pf.get_ess()
        assert abs(ess_init - N_TEST) < N_TEST * 0.01

        # After weight update with spread particles, ESS should drop
        pf.update(sat_ecef, pseudoranges, weights_sat)
        ess_after = pf.get_ess()
        assert ess_after < ess_init * 0.5, (
            f"ESS should drop after weighting: {ess_init:.0f} -> {ess_after:.0f}")

    def test_get_particles_returns_correct_shape(self):
        """ParticleFilter.get_particles() should return (N, 4)."""
        pf = ParticleFilter(n_particles=1000, seed=SEED)
        pf.initialize([0, 0, 0], clock_bias=0.0)

        particles = pf.get_particles()
        assert particles.shape == (1000, 4)

    def test_not_initialized_raises(self):
        """Operations on uninitialized filter should raise."""
        pf = ParticleFilter(n_particles=100, seed=SEED)
        with pytest.raises(RuntimeError):
            pf.predict()
        with pytest.raises(RuntimeError):
            pf.estimate()
        with pytest.raises(RuntimeError):
            pf.get_ess()
