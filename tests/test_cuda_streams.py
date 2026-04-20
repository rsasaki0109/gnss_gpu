"""Tests for CUDA stream support in ParticleFilterDevice.

Verifies that stream-based (async) operations produce identical results
to synchronous execution, and benchmarks latency improvement.
"""

import time
from types import SimpleNamespace
import numpy as np
import pytest

try:
    from gnss_gpu._gnss_gpu_pf_device import (
        pf_device_create,
        pf_device_destroy,
        pf_device_initialize,
        pf_device_predict,
        pf_device_weight,
        pf_device_weight_doppler,
        pf_device_ess,
        pf_device_position_spread,
        pf_device_resample_systematic,
        pf_device_resample_megopolis,
        pf_device_estimate,
        pf_device_get_particles,
        pf_device_get_particle_states,
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
L1_WAVELENGTH = 0.19029367279836488

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


def _make_doppler_data():
    """Create RINEX-style Doppler rows with a known receiver velocity."""
    rx_pos = np.array([-3957199.0, 3310205.0, 3737911.0], dtype=np.float64)
    true_vel = np.array([0.3, -0.5, 0.8], dtype=np.float64)
    true_cd = 5.0
    sat_ecef = np.array([
        [-14985000.0, -3988000.0, 21474000.0],
        [-9575000.0, 15498000.0, 19457000.0],
        [7624000.0, -16218000.0, 19843000.0],
        [16305000.0, 12037000.0, 17183000.0],
        [-20889000.0, 13759000.0, 8291000.0],
        [5463000.0, 24413000.0, 8934000.0],
    ], dtype=np.float64)
    sat_vel = np.array([
        [1200.0, -2800.0, 500.0],
        [-800.0, 1500.0, -2700.0],
        [2500.0, 1800.0, -900.0],
        [-1100.0, -2200.0, 2100.0],
        [600.0, 2900.0, 1300.0],
        [-2600.0, 400.0, -1800.0],
    ], dtype=np.float64)
    doppler = np.zeros(sat_ecef.shape[0], dtype=np.float64)
    for i in range(sat_ecef.shape[0]):
        los = sat_ecef[i] - rx_pos
        los /= np.linalg.norm(los)
        range_rate = float(np.dot(sat_vel[i] - true_vel, los) + true_cd)
        doppler[i] = -range_rate / L1_WAVELENGTH
    weights = np.ones(sat_ecef.shape[0], dtype=np.float64)
    return rx_pos, sat_ecef, sat_vel, doppler, weights, true_vel


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

    def test_position_spread_valid_after_weight(self):
        """Weighted RMS particle spread is finite and positive after weight update."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()

        state = pf_device_create(N_PARTICLES)
        pf_device_initialize(state, 0.0, 0.0, 0.0, TRUE_CB, 50.0, 200.0, SEED)
        pf_device_weight(state, sat_ecef.ravel(), pseudoranges, weights, N_SAT, 5.0)
        est = pf_device_estimate(state)
        spread = pf_device_position_spread(state, float(est[0]), float(est[1]), float(est[2]))
        assert np.isfinite(spread)
        assert spread > 0.0
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

    def test_full_particle_states_include_velocity_kf_state(self):
        """Full device state exposes sampled position plus velocity KF state."""
        n = 2048
        state = pf_device_create(n)
        pf_device_initialize(
            state,
            10.0, 20.0, 30.0, TRUE_CB,
            0.0, 0.0, SEED,
            1.5, -2.0, 0.25, 0.0, 2.0,
        )

        states = pf_device_get_particle_states(state)
        assert states.shape == (n, 16)
        np.testing.assert_allclose(states[:, :3], np.tile([10.0, 20.0, 30.0], (n, 1)))
        np.testing.assert_allclose(states[:, 3], TRUE_CB)
        np.testing.assert_allclose(states[:, 4:7], np.tile([1.5, -2.0, 0.25], (n, 1)))
        np.testing.assert_allclose(states[:, 7], 4.0)
        np.testing.assert_allclose(states[:, 11], 4.0)
        np.testing.assert_allclose(states[:, 15], 4.0)
        off_diag = states[:, [8, 9, 10, 12, 13, 14]]
        np.testing.assert_allclose(off_diag, 0.0)
        pf_device_destroy(state)

    def test_predict_uses_per_particle_velocity_state(self):
        """Predict stores the velocity guide per particle before propagation."""
        n = 2048
        state = pf_device_create(n)
        pf_device_initialize(state, 0.0, 0.0, 0.0, TRUE_CB, 0.0, 0.0, SEED)
        pf_device_predict(
            state,
            2.0, -1.0, 0.5,
            0.25, 0.0, 0.0,
            SEED, 1,
        )

        states = pf_device_get_particle_states(state)
        np.testing.assert_allclose(states[:, :3], np.tile([0.5, -0.25, 0.125], (n, 1)))
        np.testing.assert_allclose(states[:, 3], TRUE_CB)
        np.testing.assert_allclose(states[:, 4:7], np.tile([2.0, -1.0, 0.5], (n, 1)))
        pf_device_destroy(state)

    def test_doppler_update_moves_particle_velocity(self):
        """Per-particle Doppler update nudges velocity toward the WLS solution."""
        rx_pos, sat_ecef, sat_vel, doppler, weights, true_vel = _make_doppler_data()
        n = 2048
        state = pf_device_create(n)
        pf_device_initialize(
            state,
            float(rx_pos[0]), float(rx_pos[1]), float(rx_pos[2]), TRUE_CB,
            0.0, 0.0, SEED,
            0.0, 0.0, 0.0, 0.0,
        )
        pf_device_weight_doppler(
            state,
            sat_ecef.ravel(), sat_vel.ravel(), doppler, weights,
            len(doppler),
            L1_WAVELENGTH, 0.5, 1.0, 100.0,
        )

        states = pf_device_get_particle_states(state)
        np.testing.assert_allclose(
            states[:, 4:7],
            np.tile(true_vel, (n, 1)),
            atol=1e-3,
        )
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

    def test_get_particle_states_wrapper_shape(self):
        """Wrapper exposes the 16D RBPF state without changing estimate()."""
        pf = ParticleFilterDevice(n_particles=2048, seed=SEED)
        pf.initialize(
            position_ecef=TRUE_POS,
            clock_bias=TRUE_CB,
            spread_pos=0.0,
            spread_cb=0.0,
            velocity=np.array([1.0, 0.5, -0.25]),
            spread_vel=0.0,
            velocity_init_sigma=3.0,
        )
        states = pf.get_particle_states()
        assert states.shape == (2048, 16)
        np.testing.assert_allclose(
            states[:, 4:7],
            np.tile([1.0, 0.5, -0.25], (2048, 1)),
        )
        np.testing.assert_allclose(states[:, 7], 9.0)
        np.testing.assert_allclose(states[:, 11], 9.0)
        np.testing.assert_allclose(states[:, 15], 9.0)
        assert pf.estimate().shape == (4,)

    def test_update_doppler_wrapper_updates_velocity(self):
        """High-level wrapper exposes Doppler per-particle velocity updates."""
        rx_pos, sat_ecef, sat_vel, doppler, weights, true_vel = _make_doppler_data()
        pf = ParticleFilterDevice(n_particles=2048, seed=SEED, resampling="systematic")
        pf.initialize(
            position_ecef=rx_pos,
            clock_bias=TRUE_CB,
            spread_pos=0.0,
            spread_cb=0.0,
            velocity=np.zeros(3),
            spread_vel=0.0,
        )
        pf.update_doppler(
            sat_ecef,
            sat_vel,
            doppler,
            weights=weights,
            sigma_mps=0.5,
            velocity_update_gain=1.0,
            max_velocity_update_mps=100.0,
            resample=False,
        )
        states = pf.get_particle_states()
        np.testing.assert_allclose(
            states[:, 4:7],
            np.tile(true_vel, (2048, 1)),
            atol=1e-3,
        )

    def test_get_position_spread_shrinks_after_update(self):
        """Position spread decreases after incorporating informative measurements."""
        sat_ecef, pseudoranges, weights = _make_satellite_data()
        pf = ParticleFilterDevice(
            n_particles=50_000,
            sigma_pos=1.0,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
        )
        pf.initialize(position_ecef=TRUE_POS, clock_bias=TRUE_CB,
                      spread_pos=80.0, spread_cb=200.0)
        initial_spread = pf.get_position_spread(center=TRUE_POS)
        pf.update(sat_ecef, pseudoranges, weights, resample=False)
        spread_after = pf.get_position_spread()
        assert np.isfinite(initial_spread)
        assert np.isfinite(spread_after)
        assert spread_after < initial_spread

    def test_update_dd_pseudorange_smoke(self):
        """DD pseudorange update runs end-to-end and keeps estimates finite."""
        rng = np.random.RandomState(321)
        n_sat = 5
        directions = rng.randn(n_sat, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        sat_ecef = directions * 20_000_000.0

        rover_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        base_pos = np.array([120.0, -35.0, 18.0], dtype=np.float64)
        rover_ranges = np.linalg.norm(sat_ecef - rover_pos, axis=1)
        base_ranges = np.linalg.norm(sat_ecef - base_pos, axis=1)

        ref = 0
        dd_result = SimpleNamespace(
            dd_pseudorange_m=(rover_ranges[1:] - rover_ranges[ref]) - (base_ranges[1:] - base_ranges[ref]),
            sat_ecef_k=sat_ecef[1:].astype(np.float64),
            sat_ecef_ref=np.repeat(sat_ecef[[ref]], n_sat - 1, axis=0).astype(np.float64),
            base_range_k=base_ranges[1:].astype(np.float64),
            base_range_ref=np.repeat(base_ranges[ref], n_sat - 1).astype(np.float64),
            dd_weights=np.ones(n_sat - 1, dtype=np.float64),
            ref_sat_ids=tuple(["G01"] * (n_sat - 1)),
            n_dd=n_sat - 1,
        )

        pf = ParticleFilterDevice(
            n_particles=50_000,
            sigma_pos=1.0,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
        )
        pf.initialize(position_ecef=rover_pos, clock_bias=250.0, spread_pos=30.0, spread_cb=200.0)
        pf.update_dd_pseudorange(dd_result, sigma_pr=0.5)
        pf.sync()

        est = pf.estimate()
        assert est.shape == (4,)
        assert np.all(np.isfinite(est))
        assert np.linalg.norm(est[:3] - rover_pos) < 10.0

    def test_per_particle_nlos_gate_skips_outlier_residuals(self):
        """Per-particle NLOS gate ignores outlier rows in the device kernels."""
        sat_ecef, pseudoranges, weights = _make_satellite_data(n_sat=5)
        pseudoranges_outlier = pseudoranges.copy()
        pseudoranges_outlier[0] += 100.0

        pf = ParticleFilterDevice(
            n_particles=1024,
            sigma_pos=1.0,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
            per_particle_nlos_gate=True,
            per_particle_nlos_undiff_pr_threshold_m=30.0,
        )
        pf.initialize(TRUE_POS, clock_bias=TRUE_CB, spread_pos=0.0, spread_cb=0.0)
        pf.update(sat_ecef, pseudoranges_outlier, weights, resample=False)
        undiff_lw = pf.get_log_weights()
        assert float(np.max(np.abs(undiff_lw))) < 1e-6

        pf_reject_all = ParticleFilterDevice(
            n_particles=1024,
            sigma_pos=1.0,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
            per_particle_nlos_gate=True,
            per_particle_nlos_undiff_pr_threshold_m=30.0,
        )
        pf_reject_all.initialize(TRUE_POS, clock_bias=TRUE_CB, spread_pos=0.0, spread_cb=0.0)
        pf_reject_all.update(sat_ecef, pseudoranges + 100.0, weights, resample=False)
        reject_all_lw = pf_reject_all.get_log_weights()
        assert float(np.max(reject_all_lw)) < -900.0

        rover_pos = TRUE_POS.copy()
        base_pos = np.array([120.0, -35.0, 18.0], dtype=np.float64)
        rover_ranges = np.linalg.norm(sat_ecef - rover_pos, axis=1)
        base_ranges = np.linalg.norm(sat_ecef - base_pos, axis=1)
        ref = 0
        dd_pr = (rover_ranges[1:] - rover_ranges[ref]) - (base_ranges[1:] - base_ranges[ref])
        dd_pr[0] += 50.0
        dd_pr_result = SimpleNamespace(
            dd_pseudorange_m=dd_pr.astype(np.float64),
            sat_ecef_k=sat_ecef[1:].astype(np.float64),
            sat_ecef_ref=np.repeat(sat_ecef[[ref]], len(dd_pr), axis=0).astype(np.float64),
            base_range_k=base_ranges[1:].astype(np.float64),
            base_range_ref=np.repeat(base_ranges[ref], len(dd_pr)).astype(np.float64),
            dd_weights=np.ones(len(dd_pr), dtype=np.float64),
            ref_sat_ids=tuple(["G01"] * len(dd_pr)),
            n_dd=len(dd_pr),
        )
        pf_dd_pr = ParticleFilterDevice(
            n_particles=1024,
            sigma_pos=1.0,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
            per_particle_nlos_gate=True,
            per_particle_nlos_dd_pr_threshold_m=10.0,
        )
        pf_dd_pr.initialize(rover_pos, clock_bias=0.0, spread_pos=0.0, spread_cb=0.0)
        pf_dd_pr.update_dd_pseudorange(dd_pr_result, sigma_pr=0.5, resample=False)
        dd_pr_lw = pf_dd_pr.get_log_weights()
        assert float(np.max(np.abs(dd_pr_lw))) < 1e-6

        wavelengths = np.full(len(dd_pr), 0.190293673, dtype=np.float64)
        dd_carrier = (
            (rover_ranges[1:] - rover_ranges[ref]) - (base_ranges[1:] - base_ranges[ref])
        ) / wavelengths
        dd_carrier[0] += 0.45
        dd_cp_result = SimpleNamespace(
            dd_carrier_cycles=dd_carrier.astype(np.float64),
            sat_ecef_k=sat_ecef[1:].astype(np.float64),
            sat_ecef_ref=np.repeat(sat_ecef[[ref]], len(dd_carrier), axis=0).astype(np.float64),
            base_range_k=base_ranges[1:].astype(np.float64),
            base_range_ref=np.repeat(base_ranges[ref], len(dd_carrier)).astype(np.float64),
            dd_weights=np.ones(len(dd_carrier), dtype=np.float64),
            wavelengths_m=wavelengths,
            ref_sat_ids=tuple(["G01"] * len(dd_carrier)),
            n_dd=len(dd_carrier),
        )
        pf_dd_cp = ParticleFilterDevice(
            n_particles=1024,
            sigma_pos=1.0,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
            per_particle_nlos_gate=True,
            per_particle_nlos_dd_carrier_threshold_cycles=0.3,
        )
        pf_dd_cp.initialize(rover_pos, clock_bias=0.0, spread_pos=0.0, spread_cb=0.0)
        pf_dd_cp.update_dd_carrier_afv(dd_cp_result, sigma_cycles=0.05, resample=False)
        dd_cp_lw = pf_dd_cp.get_log_weights()
        assert float(np.max(np.abs(dd_cp_lw))) < 1e-6

    def test_per_particle_huber_softens_outlier_cost_without_rejecting(self):
        """Per-particle Huber keeps outliers finite but less dominant than Gaussian."""
        sat_ecef, pseudoranges, weights = _make_satellite_data(n_sat=5)
        pseudoranges_outlier = pseudoranges.copy()
        pseudoranges_outlier[0] += 100.0

        def undiff_pr_lw(huber):
            pf = ParticleFilterDevice(
                n_particles=1024,
                sigma_pos=1.0,
                sigma_cb=300.0,
                sigma_pr=5.0,
                resampling="systematic",
                seed=SEED,
                per_particle_huber=huber,
                per_particle_huber_undiff_pr_k=1.5,
            )
            pf.initialize(TRUE_POS, clock_bias=TRUE_CB, spread_pos=0.0, spread_cb=0.0)
            pf.update(sat_ecef, pseudoranges_outlier, weights, resample=False)
            return float(pf.get_log_weights()[0])

        gaussian_undiff = undiff_pr_lw(False)
        huber_undiff = undiff_pr_lw(True)
        assert gaussian_undiff == pytest.approx(-200.0)
        assert huber_undiff == pytest.approx(-28.875)
        assert gaussian_undiff < huber_undiff < 0.0
        assert np.exp(huber_undiff) > 0.0

        rover_pos = TRUE_POS.copy()
        base_pos = np.array([120.0, -35.0, 18.0], dtype=np.float64)
        rover_ranges = np.linalg.norm(sat_ecef - rover_pos, axis=1)
        base_ranges = np.linalg.norm(sat_ecef - base_pos, axis=1)
        ref = 0
        dd_pr = (rover_ranges[1:] - rover_ranges[ref]) - (base_ranges[1:] - base_ranges[ref])
        dd_pr_outlier = dd_pr.copy()
        dd_pr_outlier[0] += 50.0
        dd_pr_result = SimpleNamespace(
            dd_pseudorange_m=dd_pr_outlier.astype(np.float64),
            sat_ecef_k=sat_ecef[1:].astype(np.float64),
            sat_ecef_ref=np.repeat(sat_ecef[[ref]], len(dd_pr), axis=0).astype(np.float64),
            base_range_k=base_ranges[1:].astype(np.float64),
            base_range_ref=np.repeat(base_ranges[ref], len(dd_pr)).astype(np.float64),
            dd_weights=np.ones(len(dd_pr), dtype=np.float64),
            ref_sat_ids=tuple(["G01"] * len(dd_pr)),
            n_dd=len(dd_pr),
        )

        def dd_pr_lw(huber):
            pf = ParticleFilterDevice(
                n_particles=1024,
                sigma_pos=1.0,
                sigma_cb=300.0,
                sigma_pr=5.0,
                resampling="systematic",
                seed=SEED,
                per_particle_huber=huber,
                per_particle_huber_dd_pr_k=1.5,
            )
            pf.initialize(rover_pos, clock_bias=0.0, spread_pos=0.0, spread_cb=0.0)
            pf.update_dd_pseudorange(dd_pr_result, sigma_pr=0.5, resample=False)
            return float(pf.get_log_weights()[0])

        gaussian_dd_pr = dd_pr_lw(False)
        huber_dd_pr = dd_pr_lw(True)
        assert gaussian_dd_pr == pytest.approx(-5000.0)
        assert huber_dd_pr == pytest.approx(-148.875)
        assert gaussian_dd_pr < huber_dd_pr < 0.0
        assert np.exp(huber_dd_pr) > 0.0

        wavelengths = np.full(len(dd_pr), 0.190293673, dtype=np.float64)
        dd_carrier = dd_pr / wavelengths
        dd_carrier_outlier = dd_carrier.copy()
        dd_carrier_outlier[0] += 0.45
        dd_cp_result = SimpleNamespace(
            dd_carrier_cycles=dd_carrier_outlier.astype(np.float64),
            sat_ecef_k=sat_ecef[1:].astype(np.float64),
            sat_ecef_ref=np.repeat(sat_ecef[[ref]], len(dd_carrier), axis=0).astype(np.float64),
            base_range_k=base_ranges[1:].astype(np.float64),
            base_range_ref=np.repeat(base_ranges[ref], len(dd_carrier)).astype(np.float64),
            dd_weights=np.ones(len(dd_carrier), dtype=np.float64),
            wavelengths_m=wavelengths,
            ref_sat_ids=tuple(["G01"] * len(dd_carrier)),
            n_dd=len(dd_carrier),
        )

        def dd_cp_lw(huber):
            pf = ParticleFilterDevice(
                n_particles=1024,
                sigma_pos=1.0,
                sigma_cb=300.0,
                sigma_pr=5.0,
                resampling="systematic",
                seed=SEED,
                per_particle_huber=huber,
                per_particle_huber_dd_carrier_k=1.5,
            )
            pf.initialize(rover_pos, clock_bias=0.0, spread_pos=0.0, spread_cb=0.0)
            pf.update_dd_carrier_afv(dd_cp_result, sigma_cycles=0.05, resample=False)
            return float(pf.get_log_weights()[0])

        gaussian_dd_cp = dd_cp_lw(False)
        huber_dd_cp = dd_cp_lw(True)
        assert gaussian_dd_cp == pytest.approx(-40.5, abs=1e-9)
        assert huber_dd_cp == pytest.approx(-12.375, abs=1e-9)
        assert gaussian_dd_cp < huber_dd_cp < 0.0
        assert np.exp(huber_dd_cp) > 0.0

    def test_smooth_with_dd_pseudorange_smoke(self):
        """Forward-backward smoothing replays stored DD pseudorange observations."""
        rng = np.random.RandomState(987)
        n_sat = 5
        directions = rng.randn(n_sat, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        sat_ecef = directions * 20_000_000.0

        rover_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        base_pos = np.array([120.0, -35.0, 18.0], dtype=np.float64)
        rover_ranges = np.linalg.norm(sat_ecef - rover_pos, axis=1)
        base_ranges = np.linalg.norm(sat_ecef - base_pos, axis=1)

        ref = 0
        dd_template = SimpleNamespace(
            dd_pseudorange_m=(rover_ranges[1:] - rover_ranges[ref]) - (base_ranges[1:] - base_ranges[ref]),
            sat_ecef_k=sat_ecef[1:].astype(np.float64),
            sat_ecef_ref=np.repeat(sat_ecef[[ref]], n_sat - 1, axis=0).astype(np.float64),
            base_range_k=base_ranges[1:].astype(np.float64),
            base_range_ref=np.repeat(base_ranges[ref], n_sat - 1).astype(np.float64),
            dd_weights=np.ones(n_sat - 1, dtype=np.float64),
            ref_sat_ids=tuple(["G01"] * (n_sat - 1)),
            n_dd=n_sat - 1,
        )

        pf = ParticleFilterDevice(
            n_particles=20_000,
            sigma_pos=1.0,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
        )
        pf.initialize(position_ecef=rover_pos, clock_bias=250.0, spread_pos=20.0, spread_cb=150.0)
        pf.enable_smoothing()

        pr = rover_ranges + 250.0
        weights = np.ones(n_sat, dtype=np.float64)
        for _ in range(3):
            pf.predict(dt=1.0)
            pf.update_dd_pseudorange(dd_template, sigma_pr=0.5)
            pf.store_epoch(
                sat_ecef,
                pr,
                weights,
                None,
                1.0,
                spp_ref=None,
                dd_pseudorange=dd_template,
                dd_pseudorange_sigma=0.5,
            )

        smoothed, forward = pf.smooth(position_update_sigma=None)
        assert smoothed.shape == (3, 3)
        assert forward.shape == (3, 3)
        assert np.all(np.isfinite(smoothed))
        assert np.all(np.isfinite(forward))

    def test_smooth_with_undiff_carrier_afv_smoke(self):
        """Forward-backward smoothing replays stored undifferenced carrier AFV updates."""
        rng = np.random.RandomState(4321)
        n_sat = 6
        directions = rng.randn(n_sat, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        sat_ecef = directions * 20_000_000.0

        rover_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        wavelength = 0.190293673
        rover_ranges = np.linalg.norm(sat_ecef - rover_pos, axis=1)
        carrier_cycles = (rover_ranges / wavelength).astype(np.float64)
        weights = np.ones(n_sat, dtype=np.float64)

        carrier_afv = {
            "sat_ecef": sat_ecef.astype(np.float64),
            "carrier_phase_cycles": carrier_cycles.astype(np.float64),
            "weights": weights.astype(np.float64),
        }

        pf = ParticleFilterDevice(
            n_particles=20_000,
            sigma_pos=0.5,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
        )
        pf.initialize(position_ecef=rover_pos, clock_bias=0.0, spread_pos=5.0, spread_cb=10.0)
        pf.enable_smoothing()

        for _ in range(3):
            pf.predict(dt=1.0)
            pf.update(sat_ecef, rover_ranges, weights)
            pf.resample_if_needed()
            pf.update_carrier_afv(
                sat_ecef,
                carrier_cycles,
                weights=weights,
                wavelength=wavelength,
                sigma_cycles=0.10,
            )
            pf.store_epoch(
                sat_ecef,
                rover_ranges,
                weights,
                None,
                1.0,
                spp_ref=None,
                carrier_afv=carrier_afv,
                carrier_afv_sigma=0.10,
                carrier_afv_wavelength=wavelength,
            )

        smoothed, forward = pf.smooth(position_update_sigma=None)
        assert smoothed.shape == (3, 3)
        assert forward.shape == (3, 3)
        assert np.all(np.isfinite(smoothed))
        assert np.all(np.isfinite(forward))

    def test_smooth_with_carrier_anchor_pseudorange_smoke(self):
        """Forward-backward smoothing replays stored carrier-anchor pseudorange updates."""
        rng = np.random.RandomState(2468)
        n_sat = 6
        directions = rng.randn(n_sat, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        sat_ecef = directions * 20_000_000.0

        rover_pos = np.array([5.0, -3.0, 2.0], dtype=np.float64)
        rover_cb = 120.0
        wavelength = 0.190293673
        rover_ranges = np.linalg.norm(sat_ecef - rover_pos, axis=1)
        bias_cycles = np.linspace(1000.0, 1005.0, n_sat, dtype=np.float64)
        carrier_cycles = (rover_ranges + rover_cb) / wavelength + bias_cycles
        carrier_anchor = {
            "sat_ecef": sat_ecef.astype(np.float64),
            "pseudoranges": (wavelength * (carrier_cycles - bias_cycles)).astype(np.float64),
            "weights": np.ones(n_sat, dtype=np.float64),
        }

        pf = ParticleFilterDevice(
            n_particles=20_000,
            sigma_pos=1.0,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
        )
        pf.initialize(position_ecef=rover_pos, clock_bias=rover_cb, spread_pos=5.0, spread_cb=20.0)
        pf.enable_smoothing()

        for _ in range(3):
            pf.predict(dt=1.0)
            pf.update(
                carrier_anchor["sat_ecef"],
                carrier_anchor["pseudoranges"],
                weights=carrier_anchor["weights"],
                sigma_pr=0.25,
            )
            pf.store_epoch(
                carrier_anchor["sat_ecef"],
                carrier_anchor["pseudoranges"],
                carrier_anchor["weights"],
                None,
                1.0,
                spp_ref=None,
                carrier_anchor_pseudorange=carrier_anchor,
                carrier_anchor_sigma=0.25,
            )

        smoothed, forward = pf.smooth(position_update_sigma=None)
        assert smoothed.shape == (3, 3)
        assert forward.shape == (3, 3)
        assert np.all(np.isfinite(smoothed))
        assert np.all(np.isfinite(forward))

    def test_update_dd_carrier_afv_smoke(self):
        """DD carrier AFV update accepts per-pair reference geometry and wavelengths."""
        sat_ecef = np.array(
            [
                [20_200_000.0, 0.0, 0.0],
                [0.0, 21_000_000.0, 0.0],
                [0.0, 0.0, 20_600_000.0],
                [-20_800_000.0, 2_000_000.0, 1_000_000.0],
            ],
            dtype=np.float64,
        )
        rover_pos = np.array([0.3, -0.2, 0.1], dtype=np.float64)
        base_pos = np.array([120.0, -35.0, 18.0], dtype=np.float64)

        rover_ranges = np.linalg.norm(sat_ecef - rover_pos, axis=1)
        base_ranges = np.linalg.norm(sat_ecef - base_pos, axis=1)
        wavelengths = np.array([0.190293673, 0.192039486, 0.190293673], dtype=np.float64)
        refs = np.array([0, 2, 2], dtype=np.int64)
        sats = np.array([1, 3, 1], dtype=np.int64)
        dd_carrier = (
            (rover_ranges[sats] - rover_ranges[refs])
            - (base_ranges[sats] - base_ranges[refs])
        ) / wavelengths

        dd_result = SimpleNamespace(
            dd_carrier_cycles=dd_carrier.astype(np.float64),
            sat_ecef_k=sat_ecef[sats].astype(np.float64),
            sat_ecef_ref=sat_ecef[refs].astype(np.float64),
            base_range_k=base_ranges[sats].astype(np.float64),
            base_range_ref=base_ranges[refs].astype(np.float64),
            dd_weights=np.ones(len(sats), dtype=np.float64),
            wavelengths_m=wavelengths.astype(np.float64),
            ref_sat_ids=("G01", "E03", "G03"),
            n_dd=len(sats),
        )

        pf = ParticleFilterDevice(
            n_particles=30_000,
            sigma_pos=0.1,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
        )
        pf.initialize(position_ecef=rover_pos, clock_bias=0.0, spread_pos=0.2, spread_cb=10.0)
        pf.update_dd_carrier_afv(dd_result, sigma_cycles=0.05)
        pf.sync()

        est = pf.estimate()
        assert est.shape == (4,)
        assert np.all(np.isfinite(est))
        assert np.linalg.norm(est[:3] - rover_pos) < 2.0

    def test_smooth_with_dd_pseudorange_and_dd_carrier_smoke(self):
        """Backward smoothing replays both DD pseudorange and DD carrier updates."""
        sat_ecef = np.array(
            [
                [20_200_000.0, 0.0, 0.0],
                [0.0, 21_000_000.0, 0.0],
                [0.0, 0.0, 20_600_000.0],
                [-20_800_000.0, 2_000_000.0, 1_000_000.0],
            ],
            dtype=np.float64,
        )
        rover_pos = np.array([0.3, -0.2, 0.1], dtype=np.float64)
        base_pos = np.array([120.0, -35.0, 18.0], dtype=np.float64)
        rover_ranges = np.linalg.norm(sat_ecef - rover_pos, axis=1)
        base_ranges = np.linalg.norm(sat_ecef - base_pos, axis=1)

        dd_pr = SimpleNamespace(
            dd_pseudorange_m=((rover_ranges[1:] - rover_ranges[0]) - (base_ranges[1:] - base_ranges[0])).astype(np.float64),
            sat_ecef_k=sat_ecef[1:].astype(np.float64),
            sat_ecef_ref=np.repeat(sat_ecef[[0]], 3, axis=0).astype(np.float64),
            base_range_k=base_ranges[1:].astype(np.float64),
            base_range_ref=np.repeat(base_ranges[0], 3).astype(np.float64),
            dd_weights=np.ones(3, dtype=np.float64),
            ref_sat_ids=("G01", "G01", "G01"),
            n_dd=3,
        )
        dd_cp = SimpleNamespace(
            dd_carrier_cycles=(
                ((rover_ranges[[1, 3, 1]] - rover_ranges[[0, 2, 2]])
                 - (base_ranges[[1, 3, 1]] - base_ranges[[0, 2, 2]]))
                / np.array([0.190293673, 0.192039486, 0.190293673], dtype=np.float64)
            ).astype(np.float64),
            sat_ecef_k=sat_ecef[[1, 3, 1]].astype(np.float64),
            sat_ecef_ref=sat_ecef[[0, 2, 2]].astype(np.float64),
            base_range_k=base_ranges[[1, 3, 1]].astype(np.float64),
            base_range_ref=base_ranges[[0, 2, 2]].astype(np.float64),
            dd_weights=np.ones(3, dtype=np.float64),
            wavelengths_m=np.array([0.190293673, 0.192039486, 0.190293673], dtype=np.float64),
            ref_sat_ids=("G01", "E03", "G03"),
            n_dd=3,
        )

        pf = ParticleFilterDevice(
            n_particles=20_000,
            sigma_pos=0.1,
            sigma_cb=300.0,
            sigma_pr=5.0,
            resampling="systematic",
            seed=SEED,
        )
        pf.initialize(position_ecef=rover_pos, clock_bias=0.0, spread_pos=0.2, spread_cb=10.0)
        pf.enable_smoothing()

        pr = rover_ranges.astype(np.float64)
        weights = np.ones(len(sat_ecef), dtype=np.float64)
        for _ in range(3):
            pf.predict(dt=1.0)
            pf.update_dd_pseudorange(dd_pr, sigma_pr=0.5)
            pf.resample_if_needed()
            pf.update_dd_carrier_afv(dd_cp, sigma_cycles=0.05)
            pf.store_epoch(
                sat_ecef,
                pr,
                weights,
                None,
                1.0,
                spp_ref=None,
                dd_pseudorange=dd_pr,
                dd_pseudorange_sigma=0.5,
                dd_carrier=dd_cp,
                dd_carrier_sigma=0.05,
            )

        smoothed, forward = pf.smooth(position_update_sigma=None)
        assert smoothed.shape == (3, 3)
        assert forward.shape == (3, 3)
        assert np.all(np.isfinite(smoothed))
        assert np.all(np.isfinite(forward))

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
