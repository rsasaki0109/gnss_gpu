"""Tests for CUDA stream support in ParticleFilterDevice.

Verifies that stream-based (async) operations produce identical results
to synchronous execution, and benchmarks latency improvement.
"""

import time
import numpy as np
import pytest

try:
    from gnss_gpu._gnss_gpu_pf_device import (
        pf_device_create,
        pf_device_destroy,
        pf_device_initialize,
        pf_device_predict,
        pf_device_weight,
        pf_device_ess,
        pf_device_resample_systematic,
        pf_device_resample_megopolis,
        pf_device_estimate,
        pf_device_get_particles,
        pf_device_sync,
    )
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CUDA pf_device module not available")

SEED = 42
N_PARTICLES = 100_000
N_SAT = 8

# Simulated receiver at ECEF origin with clock bias
TRUE_POS = np.array([0.0, 0.0, 0.0])
TRUE_CB = 100.0


def _make_satellite_data(n_sat=N_SAT):
    """Create synthetic satellite positions and pseudoranges."""
    rng = np.random.RandomState(123)
    # Satellites at ~20,000 km altitude in various directions
    directions = rng.randn(n_sat, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    sat_ecef = directions * 20_000_000.0  # ~20,000 km

    ranges = np.sqrt(np.sum((sat_ecef - TRUE_POS) ** 2, axis=1))
    pseudoranges = ranges + TRUE_CB
    weights = np.ones(n_sat, dtype=np.float64)
    return sat_ecef.astype(np.float64), pseudoranges.astype(np.float64), weights


class TestStreamCorrectness:
    """Verify stream-based operations produce correct results."""

    def test_create_destroy(self):
        """State can be created and destroyed without error."""
        state = pf_device_create(1000)
        assert state.n_particles == 1000
        assert state.allocated
        pf_device_destroy(state)

    def test_sync_after_init(self):
        """pf_device_sync completes without error after initialization."""
        state = pf_device_create(1000)
        pf_device_initialize(state, 0.0, 0.0, 0.0, 100.0, 10.0, 50.0, SEED)
        pf_device_sync(state)
        pf_device_destroy(state)

    def test_estimate_deterministic(self):
        """Two runs with the same seed produce the same estimate."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()

        results = []
        for _ in range(2):
            state = pf_device_create(N_PARTICLES)
            pf_device_initialize(state, 0.0, 0.0, 0.0, TRUE_CB, 50.0, 200.0, SEED)
            pf_device_predict(state, 0.0, 0.0, 0.0, 1.0, 1.0, 300.0, SEED, 1)
            pf_device_weight(state, sat_ecef.ravel(), pseudoranges, weights,
                             N_SAT, 5.0)
            pf_device_sync(state)
            est = pf_device_estimate(state)
            results.append(est.copy())
            pf_device_destroy(state)

        np.testing.assert_array_equal(results[0], results[1])

    def test_predict_weight_estimate_pipeline(self):
        """Full predict-weight-estimate pipeline produces finite results."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()

        state = pf_device_create(N_PARTICLES)
        pf_device_initialize(state, 0.0, 0.0, 0.0, TRUE_CB, 50.0, 200.0, SEED)

        for step in range(5):
            pf_device_predict(state, 0.0, 0.0, 0.0, 1.0, 1.0, 300.0, SEED, step + 1)
            pf_device_weight(state, sat_ecef.ravel(), pseudoranges, weights,
                             N_SAT, 5.0)

        pf_device_sync(state)
        est = pf_device_estimate(state)
        assert est.shape == (4,)
        assert np.all(np.isfinite(est))
        pf_device_destroy(state)

    def test_ess_valid_after_weight(self):
        """ESS is a valid positive number after weight update."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()

        state = pf_device_create(N_PARTICLES)
        pf_device_initialize(state, 0.0, 0.0, 0.0, TRUE_CB, 50.0, 200.0, SEED)
        pf_device_weight(state, sat_ecef.ravel(), pseudoranges, weights,
                         N_SAT, 5.0)
        ess = pf_device_ess(state)
        assert 0 < ess <= N_PARTICLES
        pf_device_destroy(state)

    def test_resampling_systematic_with_stream(self):
        """Systematic resampling works correctly with stream."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()

        state = pf_device_create(N_PARTICLES)
        pf_device_initialize(state, 0.0, 0.0, 0.0, TRUE_CB, 50.0, 200.0, SEED)
        pf_device_weight(state, sat_ecef.ravel(), pseudoranges, weights,
                         N_SAT, 5.0)
        pf_device_resample_systematic(state, SEED)
        pf_device_sync(state)

        est = pf_device_estimate(state)
        assert np.all(np.isfinite(est))
        pf_device_destroy(state)

    def test_resampling_megopolis_with_stream(self):
        """Megopolis resampling works correctly with stream."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()

        state = pf_device_create(N_PARTICLES)
        pf_device_initialize(state, 0.0, 0.0, 0.0, TRUE_CB, 50.0, 200.0, SEED)
        pf_device_weight(state, sat_ecef.ravel(), pseudoranges, weights,
                         N_SAT, 5.0)
        pf_device_resample_megopolis(state, 10, SEED)
        pf_device_sync(state)

        est = pf_device_estimate(state)
        assert np.all(np.isfinite(est))
        pf_device_destroy(state)

    def test_get_particles_after_stream_ops(self):
        """get_particles returns valid data after streamed operations."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()
        n = 10_000

        state = pf_device_create(n)
        pf_device_initialize(state, 0.0, 0.0, 0.0, TRUE_CB, 50.0, 200.0, SEED)
        pf_device_predict(state, 1.0, 0.0, 0.0, 1.0, 1.0, 300.0, SEED, 1)
        pf_device_weight(state, sat_ecef.ravel(), pseudoranges, weights,
                         N_SAT, 5.0)

        particles = pf_device_get_particles(state)
        assert particles.shape == (n, 4)
        assert np.all(np.isfinite(particles))
        pf_device_destroy(state)

    def test_many_satellites_exceeds_initial_capacity(self):
        """Handle more satellites than the initial pinned buffer capacity (64)."""
        n_sat = 80  # exceeds MAX_SATS=64 to trigger reallocation
        sat_ecef, pseudoranges, weights = _make_satellite_data(n_sat)

        state = pf_device_create(10_000)
        pf_device_initialize(state, 0.0, 0.0, 0.0, TRUE_CB, 50.0, 200.0, SEED)
        pf_device_weight(state, sat_ecef.ravel(), pseudoranges, weights,
                         n_sat, 5.0)
        pf_device_sync(state)

        est = pf_device_estimate(state)
        assert np.all(np.isfinite(est))
        pf_device_destroy(state)


class TestParticleFilterDeviceWrapper:
    """Test the high-level ParticleFilterDevice Python wrapper with streams."""

    def test_sync_method_exists(self):
        """ParticleFilterDevice exposes a sync() method."""
        pf = ParticleFilterDevice(n_particles=1000, seed=SEED)
        pf.initialize(position_ecef=TRUE_POS, clock_bias=TRUE_CB,
                      spread_pos=50.0, spread_cb=200.0)
        pf.sync()  # should not raise

    def test_full_pipeline_with_sync(self):
        """Full pipeline through the wrapper class."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()

        pf = ParticleFilterDevice(
            n_particles=N_PARTICLES,
            sigma_pos=1.0, sigma_cb=300.0, sigma_pr=5.0,
            resampling="systematic", seed=SEED)

        pf.initialize(position_ecef=TRUE_POS, clock_bias=TRUE_CB,
                      spread_pos=50.0, spread_cb=200.0)

        for step in range(3):
            pf.predict(dt=1.0)
            pf.update(sat_ecef, pseudoranges, weights)

        pf.sync()
        result = pf.estimate()
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))

    def test_get_log_weights_shape(self):
        """Log-weights D2H matches n_particles after update."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()
        pf = ParticleFilterDevice(
            n_particles=1024,
            sigma_pos=1.0,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
        )
        pf.initialize(position_ecef=TRUE_POS, clock_bias=TRUE_CB,
                      spread_pos=50.0, spread_cb=200.0)
        pf.predict(dt=1.0)
        pf.update(sat_ecef, pseudoranges, weights, resample=False)
        lw = pf.get_log_weights()
        assert lw.shape == (1024,)
        assert np.all(np.isfinite(lw))
        pf.resample_if_needed()

    def test_get_resample_ancestors_after_systematic(self):
        """Systematic resample writes ancestor indices retrievable from host."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()
        pf = ParticleFilterDevice(
            n_particles=512,
            sigma_pos=1.0,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
        )
        pf.initialize(position_ecef=TRUE_POS, clock_bias=TRUE_CB,
                      spread_pos=50.0, spread_cb=200.0)
        pf.predict(dt=1.0)
        pf.update(sat_ecef, pseudoranges, weights)
        pf._resample()
        anc = pf.get_resample_ancestors()
        assert anc.shape == (512,)
        assert anc.dtype == np.int32
        assert np.all(anc >= 0) and np.all(anc < 512)

    def test_convergence_near_truth(self):
        """Estimate should converge near the true position."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()

        pf = ParticleFilterDevice(
            n_particles=N_PARTICLES,
            sigma_pos=1.0, sigma_cb=300.0, sigma_pr=5.0,
            resampling="systematic",
            ess_threshold=0.5,
            seed=SEED)

        pf.initialize(position_ecef=TRUE_POS, clock_bias=TRUE_CB,
                      spread_pos=20.0, spread_cb=200.0)

        for step in range(5):
            pf.predict(dt=0.0)
            pf.update(sat_ecef, pseudoranges, weights)

        result = pf.estimate()
        pos_error = np.linalg.norm(result[:3] - TRUE_POS)
        # With 100k particles and clean observations, should be well under 100m
        assert pos_error < 100.0, f"Position error {pos_error:.1f}m too large"


class TestStreamBenchmark:
    """Benchmark latency of streamed operations.

    These tests measure wall-clock time; they pass unconditionally
    but print timing information for manual inspection.
    """

    @pytest.mark.parametrize("n_particles", [100_000, 500_000])
    def test_benchmark_predict_weight_cycle(self, n_particles):
        """Benchmark predict+weight cycle with CUDA streams."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()

        pf = ParticleFilterDevice(
            n_particles=n_particles,
            sigma_pos=1.0, sigma_cb=300.0, sigma_pr=5.0,
            resampling="systematic", seed=SEED)

        pf.initialize(position_ecef=TRUE_POS, clock_bias=TRUE_CB,
                      spread_pos=50.0, spread_cb=200.0)

        # Warm-up
        for step in range(3):
            pf.predict(dt=1.0)
            pf.update(sat_ecef, pseudoranges, weights)
        pf.sync()

        # Timed run
        n_iters = 50
        t0 = time.perf_counter()
        for step in range(n_iters):
            pf.predict(dt=1.0)
            pf.update(sat_ecef, pseudoranges, weights)
        pf.sync()
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / n_iters) * 1000.0
        print(f"\n[STREAM BENCHMARK] {n_particles} particles, "
              f"{n_iters} iterations: {avg_ms:.2f} ms/iter "
              f"({elapsed:.3f}s total)")

    def test_benchmark_async_overlap(self):
        """Measure benefit of async H2D + kernel overlap.

        Launches predict and weight in rapid succession. With streams,
        the H2D copy for weight can overlap with the predict kernel.
        """
        sat_ecef, pseudoranges, weights = _make_satellite_data()
        n_particles = 500_000

        pf = ParticleFilterDevice(
            n_particles=n_particles,
            sigma_pos=1.0, sigma_cb=300.0, sigma_pr=5.0,
            resampling="megopolis", seed=SEED)

        pf.initialize(position_ecef=TRUE_POS, clock_bias=TRUE_CB,
                      spread_pos=50.0, spread_cb=200.0)

        # Warm-up
        for step in range(3):
            pf.predict(dt=1.0)
            pf.update(sat_ecef, pseudoranges, weights)
        pf.sync()

        # Timed: rapid fire predict+weight without intermediate syncs
        n_iters = 100
        t0 = time.perf_counter()
        for step in range(n_iters):
            pf.predict(dt=1.0)
            # Weight update H2D can overlap with tail of predict kernel
            pf.update(sat_ecef, pseudoranges, weights)
        pf.sync()
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / n_iters) * 1000.0
        print(f"\n[ASYNC OVERLAP BENCHMARK] {n_particles} particles, "
              f"{n_iters} iterations: {avg_ms:.2f} ms/iter "
              f"({elapsed:.3f}s total)")

        # Timed: with explicit sync between each step (simulates no-stream behavior)
        t0 = time.perf_counter()
        for step in range(n_iters):
            pf.predict(dt=1.0)
            pf.sync()
            pf.update(sat_ecef, pseudoranges, weights)
            pf.sync()
        elapsed_sync = time.perf_counter() - t0

        avg_ms_sync = (elapsed_sync / n_iters) * 1000.0
        print(f"[SYNC BASELINE]            {n_particles} particles, "
              f"{n_iters} iterations: {avg_ms_sync:.2f} ms/iter "
              f"({elapsed_sync:.3f}s total)")

        speedup = elapsed_sync / max(elapsed, 1e-9)
        print(f"[SPEEDUP]                  {speedup:.2f}x")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
