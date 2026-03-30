"""Tests for GPU SVGD Particle Filter (requires CUDA GPU)."""

import numpy as np
import pytest

try:
    from gnss_gpu._gnss_gpu_pf import pf_initialize, pf_get_particles
    from gnss_gpu._gnss_gpu_svgd import (
        pf_estimate_bandwidth, pf_svgd_step, pf_svgd_estimate,
    )
    from gnss_gpu.svgd import SVGDParticleFilter
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
    Same scenario as test_particle_filter.py.
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


class TestBandwidthEstimation:
    """Test bandwidth estimation via median heuristic."""

    def test_bandwidth_positive(self):
        """Bandwidth should always be positive."""
        N = N_TEST
        px = np.empty(N, dtype=np.float64)
        py = np.empty(N, dtype=np.float64)
        pz = np.empty(N, dtype=np.float64)
        pcb = np.empty(N, dtype=np.float64)

        pf_initialize(px, py, pz, pcb,
                      0.0, 0.0, 0.0, 0.0,
                      100.0, 100.0, N, SEED)

        bw = pf_estimate_bandwidth(px, py, pz, pcb, N, 1000, SEED)
        assert bw > 0.0, f"Bandwidth should be positive, got {bw}"

    def test_bandwidth_scales_with_spread(self):
        """Larger particle spread should give larger bandwidth."""
        N = N_TEST
        px1 = np.empty(N, dtype=np.float64)
        py1 = np.empty(N, dtype=np.float64)
        pz1 = np.empty(N, dtype=np.float64)
        pcb1 = np.empty(N, dtype=np.float64)
        px2 = np.empty(N, dtype=np.float64)
        py2 = np.empty(N, dtype=np.float64)
        pz2 = np.empty(N, dtype=np.float64)
        pcb2 = np.empty(N, dtype=np.float64)

        pf_initialize(px1, py1, pz1, pcb1,
                      0.0, 0.0, 0.0, 0.0,
                      10.0, 10.0, N, SEED)
        pf_initialize(px2, py2, pz2, pcb2,
                      0.0, 0.0, 0.0, 0.0,
                      1000.0, 1000.0, N, SEED + 1)

        bw1 = pf_estimate_bandwidth(px1, py1, pz1, pcb1, N, 1000, SEED)
        bw2 = pf_estimate_bandwidth(px2, py2, pz2, pcb2, N, 1000, SEED)

        assert bw2 > bw1, (
            f"Wider spread should give larger bandwidth: {bw1:.2f} vs {bw2:.2f}")

    def test_bandwidth_reasonable_range(self):
        """Bandwidth should be in a reasonable range for typical GNSS spread."""
        N = N_TEST
        px = np.empty(N, dtype=np.float64)
        py = np.empty(N, dtype=np.float64)
        pz = np.empty(N, dtype=np.float64)
        pcb = np.empty(N, dtype=np.float64)

        pf_initialize(px, py, pz, pcb,
                      -3957199.0, 3310205.0, 3737911.0, 3000.0,
                      100.0, 500.0, N, SEED)

        bw = pf_estimate_bandwidth(px, py, pz, pcb, N, 1000, SEED)
        # With spread ~100m in 3D + 500m clock bias, median distance should be
        # order of hundreds of meters, bandwidth should be positive finite
        assert 1.0 < bw < 1e6, f"Bandwidth {bw:.2f} seems unreasonable"


class TestSVGDGradient:
    """Test SVGD gradient computation."""

    def test_particles_move_toward_true_position(self):
        """After one SVGD step, particle mean should be closer to truth."""
        sat_ecef, pseudoranges, weights_sat, true_pos, true_cb = _make_test_scenario()

        N = N_TEST
        px = np.empty(N, dtype=np.float64)
        py = np.empty(N, dtype=np.float64)
        pz = np.empty(N, dtype=np.float64)
        pcb = np.empty(N, dtype=np.float64)

        # Initialize with offset from true position
        init_pos = true_pos + np.array([100.0, -50.0, 80.0])
        pf_initialize(px, py, pz, pcb,
                      init_pos[0], init_pos[1], init_pos[2], true_cb + 500.0,
                      50.0, 200.0, N, SEED)

        # Record mean before SVGD
        mean_before = np.array([np.mean(px), np.mean(py), np.mean(pz)])
        err_before = np.linalg.norm(mean_before - true_pos)

        # Estimate bandwidth and perform one SVGD step
        bw = pf_estimate_bandwidth(px, py, pz, pcb, N, 1000, SEED)
        pf_svgd_step(px, py, pz, pcb,
                     sat_ecef.ravel(), pseudoranges, weights_sat,
                     N, len(pseudoranges),
                     5.0, 0.5, 32, bw, SEED, 0)

        mean_after = np.array([np.mean(px), np.mean(py), np.mean(pz)])
        err_after = np.linalg.norm(mean_after - true_pos)

        assert err_after < err_before, (
            f"Particles should move toward truth: "
            f"{err_before:.1f}m -> {err_after:.1f}m")

    def test_svgd_maintains_diversity(self):
        """SVGD should maintain particle diversity (unlike resampling)."""
        sat_ecef, pseudoranges, weights_sat, true_pos, true_cb = _make_test_scenario()

        N = N_TEST
        px = np.empty(N, dtype=np.float64)
        py = np.empty(N, dtype=np.float64)
        pz = np.empty(N, dtype=np.float64)
        pcb = np.empty(N, dtype=np.float64)

        pf_initialize(px, py, pz, pcb,
                      true_pos[0], true_pos[1], true_pos[2], true_cb,
                      100.0, 500.0, N, SEED)

        # After SVGD, all particles should still be unique
        bw = pf_estimate_bandwidth(px, py, pz, pcb, N, 1000, SEED)
        pf_svgd_step(px, py, pz, pcb,
                     sat_ecef.ravel(), pseudoranges, weights_sat,
                     N, len(pseudoranges),
                     5.0, 0.5, 32, bw, SEED, 0)

        n_unique = len(np.unique(px))
        assert n_unique == N, (
            f"SVGD should preserve particle uniqueness: "
            f"{n_unique}/{N} unique particles")


class TestSVGDEstimate:
    """Test SVGD mean estimate."""

    def test_estimate_equals_mean(self):
        """SVGD estimate should be the simple mean of particles."""
        N = 1000
        px = np.empty(N, dtype=np.float64)
        py = np.empty(N, dtype=np.float64)
        pz = np.empty(N, dtype=np.float64)
        pcb = np.empty(N, dtype=np.float64)

        pf_initialize(px, py, pz, pcb,
                      100.0, 200.0, 300.0, 50.0,
                      10.0, 10.0, N, SEED)

        result = np.empty(4, dtype=np.float64)
        pf_svgd_estimate(px, py, pz, pcb, result, N)

        # Should match numpy mean
        np.testing.assert_allclose(result[0], np.mean(px), rtol=1e-6)
        np.testing.assert_allclose(result[1], np.mean(py), rtol=1e-6)
        np.testing.assert_allclose(result[2], np.mean(pz), rtol=1e-6)
        np.testing.assert_allclose(result[3], np.mean(pcb), rtol=1e-6)


class TestSVGDConvergence:
    """Test SVGD convergence on realistic GNSS scenario."""

    def test_svgd_converges_to_true_position(self):
        """Full SVGD pipeline should converge to true position."""
        sat_ecef, pseudoranges, weights_sat, true_pos, true_cb = _make_test_scenario()

        pf = SVGDParticleFilter(
            n_particles=N_TEST,
            sigma_pos=0.5,
            sigma_cb=100.0,
            sigma_pr=5.0,
            svgd_steps=5,
            step_size=0.5,
            n_neighbors=32,
            seed=SEED,
        )

        # Initialize with offset from true position
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

        assert pos_err < 100.0, f"Position error {pos_err:.1f} m (expected < 100 m)"
        assert cb_err < 1000.0, f"Clock bias error {cb_err:.1f} m (expected < 1000 m)"

    def test_svgd_improves_over_iterations(self):
        """Position error should decrease over update iterations."""
        sat_ecef, pseudoranges, weights_sat, true_pos, true_cb = _make_test_scenario()

        pf = SVGDParticleFilter(
            n_particles=N_TEST,
            sigma_pos=0.5,
            sigma_cb=100.0,
            sigma_pr=5.0,
            svgd_steps=3,
            step_size=0.5,
            n_neighbors=32,
            seed=SEED,
        )

        init_pos = true_pos + np.array([100.0, -80.0, 60.0])
        pf.initialize(init_pos, clock_bias=true_cb + 500.0,
                      spread_pos=300.0, spread_cb=2000.0)

        errors = []
        for step in range(8):
            pf.predict(velocity=None, dt=1.0)
            pf.update(sat_ecef, pseudoranges, weights_sat)
            est = pf.estimate()
            errors.append(np.linalg.norm(est[:3] - true_pos))

        # Error at the end should be less than error at the beginning
        assert errors[-1] < errors[0], (
            f"Error should decrease: {errors[0]:.1f} -> {errors[-1]:.1f}")

    def test_get_particles_shape(self):
        """SVGDParticleFilter.get_particles() should return (N, 4)."""
        pf = SVGDParticleFilter(n_particles=1000, seed=SEED)
        pf.initialize([0, 0, 0], clock_bias=0.0)

        particles = pf.get_particles()
        assert particles.shape == (1000, 4)

    def test_not_initialized_raises(self):
        """Operations on uninitialized filter should raise."""
        pf = SVGDParticleFilter(n_particles=100, seed=SEED)
        with pytest.raises(RuntimeError):
            pf.predict()
        with pytest.raises(RuntimeError):
            pf.estimate()
        with pytest.raises(RuntimeError):
            pf.update(np.zeros((4, 3)), np.zeros(4))


class TestSVGDLargeScale:
    """Test SVGD with large particle counts (GPU stress test)."""

    @pytest.mark.slow
    def test_100k_particles(self):
        """Test SVGD with 100K particles."""
        sat_ecef, pseudoranges, weights_sat, true_pos, true_cb = _make_test_scenario()

        pf = SVGDParticleFilter(
            n_particles=100_000,
            sigma_pos=0.5,
            sigma_cb=100.0,
            sigma_pr=5.0,
            svgd_steps=3,
            step_size=0.5,
            n_neighbors=32,
            seed=SEED,
        )

        pf.initialize(true_pos, clock_bias=true_cb,
                      spread_pos=200.0, spread_cb=1000.0)

        for step in range(3):
            pf.predict(velocity=None, dt=1.0)
            pf.update(sat_ecef, pseudoranges, weights_sat)

        est = pf.estimate()
        pos_err = np.linalg.norm(est[:3] - true_pos)
        assert pos_err < 200.0, f"Position error {pos_err:.1f} m with 100K particles"

    @pytest.mark.slow
    def test_1m_particles(self):
        """Test SVGD with 1M particles (full scale)."""
        sat_ecef, pseudoranges, weights_sat, true_pos, true_cb = _make_test_scenario()

        pf = SVGDParticleFilter(
            n_particles=1_000_000,
            sigma_pos=0.5,
            sigma_cb=100.0,
            sigma_pr=5.0,
            svgd_steps=3,
            step_size=0.5,
            n_neighbors=32,
            seed=SEED,
        )

        pf.initialize(true_pos, clock_bias=true_cb,
                      spread_pos=200.0, spread_cb=1000.0)

        # Just one cycle to verify it runs without OOM or crash
        pf.predict(velocity=None, dt=1.0)
        pf.update(sat_ecef, pseudoranges, weights_sat)

        est = pf.estimate()
        assert est.shape == (4,), f"Expected shape (4,), got {est.shape}"
        # Sanity check: result should be finite
        assert np.all(np.isfinite(est)), f"Result contains non-finite values: {est}"
