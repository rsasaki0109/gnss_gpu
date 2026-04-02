"""Tests for 3D-aware particle filter with building ray tracing.

Requires CUDA GPU and compiled pf3d module.
"""

import numpy as np
import pytest

try:
    from gnss_gpu._gnss_gpu_pf3d import pf_weight_3d
    from gnss_gpu._gnss_gpu_pf import (
        pf_initialize, pf_weight, pf_compute_ess, pf_estimate,
    )
    from gnss_gpu.particle_filter import ParticleFilter
    from gnss_gpu.particle_filter_3d import ParticleFilter3D
    from gnss_gpu.raytrace import BuildingModel
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")

# Smaller particle count for testing
N_TEST = 10000
SEED = 12345


def _make_box_building():
    """Create a simple box building for testing.

    Building: center at (100, 0, 25), width=20 (x), depth=20 (y), height=50 (z).
    The building spans x=[90,110], y=[-10,10], z=[0,50].
    """
    return BuildingModel.create_box(
        center=[100.0, 0.0, 25.0], width=20.0, depth=20.0, height=50.0)


def _make_test_scenario():
    """Create a simple test scenario in local-frame coordinates.

    Receiver at origin.  8 satellites at varying directions, some blocked
    by a building at x~100.

    Returns
    -------
    sat_ecef, pseudoranges, weights, true_pos, true_cb, building
    """
    true_pos = np.array([0.0, 0.0, 0.0])
    true_cb = 100.0  # clock bias [m]

    building = _make_box_building()

    # Satellites -- some behind the building (NLOS), some clear (LOS)
    sat_ecef = np.array([
        [0.0,     0.0,   20000.0],   # S0: straight up  -> LOS
        [200.0,   0.0,      25.0],   # S1: behind building -> NLOS
        [0.0,   200.0,    5000.0],   # S2: to the side  -> LOS
        [150.0,   0.0,      10.0],   # S3: behind building low -> NLOS
        [-200.0,  0.0,    5000.0],   # S4: opposite side -> LOS
        [0.0,  -200.0,    5000.0],   # S5: to the side  -> LOS
        [200.0, 200.0,   10000.0],   # S6: diagonal, above building -> LOS
        [105.0,   0.0,      25.0],   # S7: inside/behind building -> NLOS
    ], dtype=np.float64)

    ranges = np.sqrt(np.sum((sat_ecef - true_pos) ** 2, axis=1))
    pseudoranges = ranges + true_cb
    weights = np.ones(len(sat_ecef), dtype=np.float64)

    return sat_ecef, pseudoranges, weights, true_pos, true_cb, building


class TestWeight3DKernel:
    """Low-level tests for the pf_weight_3d kernel."""

    def test_los_particles_get_tight_sigma(self):
        """Particles with clear LOS should use sigma_los (tight)."""
        building = _make_box_building()
        n = 100
        n_sat = 1

        # All particles at origin, satellite straight up -> LOS
        px = np.zeros(n, dtype=np.float64)
        py = np.zeros(n, dtype=np.float64)
        pz = np.zeros(n, dtype=np.float64)
        pcb = np.full(n, 100.0, dtype=np.float64)

        sat_ecef = np.array([0.0, 0.0, 20000.0], dtype=np.float64)
        dist = np.sqrt(np.sum(sat_ecef ** 2))
        pseudoranges = np.array([dist + 100.0], dtype=np.float64)
        weights_sat = np.ones(1, dtype=np.float64)

        tri = building.triangles.reshape(-1, 3, 3)
        log_w = np.zeros(n, dtype=np.float64)

        sigma_los = 3.0
        sigma_nlos = 30.0
        nlos_bias = 20.0

        pf_weight_3d(px, py, pz, pcb,
                     sat_ecef, pseudoranges, weights_sat, tri,
                     log_w, n, n_sat,
                     sigma_los, sigma_nlos, nlos_bias)

        # Residual is zero for all particles (perfect match), so log_w ~ 0
        assert np.allclose(log_w, 0.0, atol=1e-10)

    def test_nlos_particles_get_loose_sigma(self):
        """Particles behind building should use sigma_nlos (loose)."""
        building = _make_box_building()
        n = 100
        n_sat = 1

        # Particles at origin, satellite behind building -> NLOS
        px = np.zeros(n, dtype=np.float64)
        py = np.zeros(n, dtype=np.float64)
        pz = np.zeros(n, dtype=np.float64)
        pcb = np.full(n, 100.0, dtype=np.float64)

        # Satellite behind building
        sat_pos = np.array([200.0, 0.0, 25.0], dtype=np.float64)
        dist = np.sqrt(np.sum(sat_pos ** 2))
        # Pseudorange with NLOS bias added (simulating multipath delay)
        nlos_bias = 20.0
        pseudoranges = np.array([dist + 100.0 + nlos_bias], dtype=np.float64)
        weights_sat = np.ones(1, dtype=np.float64)

        tri = building.triangles.reshape(-1, 3, 3)
        log_w = np.zeros(n, dtype=np.float64)

        sigma_los = 3.0
        sigma_nlos = 30.0

        pf_weight_3d(px, py, pz, pcb,
                     sat_pos, pseudoranges, weights_sat, tri,
                     log_w, n, n_sat,
                     sigma_los, sigma_nlos, nlos_bias)

        # After subtracting nlos_bias, residual ~ 0.  Log weights should be ~ 0.
        assert np.allclose(log_w, 0.0, atol=1e-10)

    def test_nlos_has_lower_penalty_than_los_sigma(self):
        """NLOS satellite with large residual should penalize less than LOS sigma would."""
        building = _make_box_building()
        n = 100

        px = np.zeros(n, dtype=np.float64)
        py = np.zeros(n, dtype=np.float64)
        pz = np.zeros(n, dtype=np.float64)
        pcb = np.full(n, 100.0, dtype=np.float64)

        # Satellite behind building -> NLOS
        sat_pos = np.array([200.0, 0.0, 25.0], dtype=np.float64)
        dist = np.sqrt(np.sum(sat_pos ** 2))
        # Pseudorange has 50m extra (large error beyond bias)
        extra_error = 50.0
        pseudoranges = np.array([dist + 100.0 + extra_error], dtype=np.float64)
        weights_sat = np.ones(1, dtype=np.float64)
        tri = building.triangles.reshape(-1, 3, 3)

        sigma_los = 3.0
        sigma_nlos = 30.0
        nlos_bias = 20.0

        # 3D weight (uses sigma_nlos for NLOS)
        log_w_3d = np.zeros(n, dtype=np.float64)
        pf_weight_3d(px, py, pz, pcb,
                     sat_pos, pseudoranges, weights_sat, tri,
                     log_w_3d, n, 1,
                     sigma_los, sigma_nlos, nlos_bias)

        # Standard weight with sigma_los (would harshly penalize the large residual)
        log_w_std = np.zeros(n, dtype=np.float64)
        pf_weight(px, py, pz, pcb,
                  sat_pos, pseudoranges, weights_sat, log_w_std,
                  n, 1, sigma_los)

        # 3D should penalize much less (log_w closer to 0)
        assert np.all(log_w_3d > log_w_std)

    def test_nlos_negative_residual_does_not_apply_positive_bias(self):
        """Negative NLOS residuals should not be shifted by a positive bias term."""
        building = _make_box_building()
        n = 32

        px = np.zeros(n, dtype=np.float64)
        py = np.zeros(n, dtype=np.float64)
        pz = np.zeros(n, dtype=np.float64)
        pcb = np.full(n, 100.0, dtype=np.float64)

        sat_pos = np.array([200.0, 0.0, 25.0], dtype=np.float64)
        dist = np.sqrt(np.sum(sat_pos ** 2))
        nlos_bias = 20.0
        pseudoranges = np.array([dist + 100.0 - nlos_bias], dtype=np.float64)
        weights_sat = np.ones(1, dtype=np.float64)

        tri = building.triangles.reshape(-1, 3, 3)
        log_w = np.zeros(n, dtype=np.float64)
        sigma_nlos = 30.0

        pf_weight_3d(
            px, py, pz, pcb,
            sat_pos, pseudoranges, weights_sat, tri,
            log_w, n, 1,
            3.0, sigma_nlos, nlos_bias,
        )

        expected = -0.5 * (nlos_bias ** 2) / (sigma_nlos ** 2)
        assert np.allclose(log_w, expected, atol=1e-10)

    def test_different_particles_different_los(self):
        """Particles at different positions should get different LOS classification."""
        building = _make_box_building()

        # Particle 0: at origin (LOS to satellite above)
        # Particle 1: at (50,0,0) -- satellite behind building at (200,0,25) is NLOS for both,
        #             but satellite above at (0,0,20000) is LOS for both
        n = 2
        px = np.array([0.0, 50.0], dtype=np.float64)
        py = np.array([0.0, 0.0], dtype=np.float64)
        pz = np.array([0.0, 0.0], dtype=np.float64)
        pcb = np.array([100.0, 100.0], dtype=np.float64)

        # 2 satellites
        sat_ecef = np.array([
            [0.0, 0.0, 20000.0],   # above -> LOS for both
            [200.0, 0.0, 25.0],    # behind building -> NLOS for both
        ], dtype=np.float64).ravel()

        # Compute true ranges for particle 0
        ranges_p0 = np.array([20000.0, np.sqrt(200**2 + 25**2)])
        pseudoranges = (ranges_p0 + 100.0).astype(np.float64)
        weights_sat = np.ones(2, dtype=np.float64)
        tri = building.triangles.reshape(-1, 3, 3)

        log_w = np.zeros(n, dtype=np.float64)
        pf_weight_3d(px, py, pz, pcb,
                     sat_ecef, pseudoranges, weights_sat, tri,
                     log_w, n, 2,
                     3.0, 30.0, 20.0)

        # Particle at origin with matching pseudoranges should have higher weight
        # than particle at (50,0,0) which has geometric mismatch
        assert log_w[0] > log_w[1]


class TestParticleFilter3D:
    """High-level tests for the ParticleFilter3D class."""

    def test_construction(self):
        """Test that ParticleFilter3D can be constructed."""
        building = _make_box_building()
        pf = ParticleFilter3D(
            building_model=building,
            sigma_los=3.0,
            sigma_nlos=30.0,
            nlos_bias=20.0,
            n_particles=1000,
            sigma_pos=1.0,
            sigma_cb=300.0,
        )
        assert pf.sigma_los == 3.0
        assert pf.sigma_nlos == 30.0
        assert pf.nlos_bias == 20.0
        assert pf.n_particles == 1000

    def test_invalid_building_model(self):
        """Test that non-BuildingModel argument raises TypeError."""
        with pytest.raises(TypeError):
            ParticleFilter3D(building_model="not a model", n_particles=100)

    def test_update_runs(self):
        """Test that update() completes without error."""
        building = _make_box_building()
        pf = ParticleFilter3D(
            building_model=building,
            n_particles=N_TEST,
            sigma_pos=1.0, sigma_cb=300.0,
            sigma_los=3.0, sigma_nlos=30.0, nlos_bias=20.0,
            seed=SEED)

        pf.initialize(position_ecef=[0.0, 0.0, 0.0], clock_bias=100.0,
                      spread_pos=10.0, spread_cb=100.0)

        sat_ecef = np.array([
            [0.0, 0.0, 20000.0],
            [200.0, 0.0, 25.0],
        ], dtype=np.float64)
        dist0 = 20000.0
        dist1 = np.sqrt(200**2 + 25**2)
        pseudoranges = np.array([dist0 + 100.0, dist1 + 100.0])

        pf.update(sat_ecef, pseudoranges)
        result = pf.estimate()
        assert result.shape == (4,)

    def test_estimate_near_truth(self):
        """Test that PF3D estimate converges near the true position."""
        sat_ecef, pseudoranges, weights, true_pos, true_cb, building = \
            _make_test_scenario()

        pf = ParticleFilter3D(
            building_model=building,
            n_particles=N_TEST,
            sigma_pos=1.0, sigma_cb=300.0,
            sigma_los=3.0, sigma_nlos=30.0, nlos_bias=20.0,
            resampling="systematic",
            seed=SEED)

        pf.initialize(position_ecef=true_pos, clock_bias=true_cb,
                      spread_pos=20.0, spread_cb=200.0)

        # Run multiple update steps to converge
        for _ in range(3):
            pf.predict(dt=0.0)
            pf.update(sat_ecef, pseudoranges, weights)

        result = pf.estimate()
        pos_error = np.linalg.norm(result[:3] - true_pos)
        # With clean observations the position should converge within ~30m
        assert pos_error < 30.0, f"Position error {pos_error:.1f}m too large"

    def test_pf3d_vs_pf_with_nlos(self):
        """PF3D should outperform standard PF when NLOS satellites have multipath bias.

        Synthetic scenario: some satellites are NLOS and their pseudoranges
        carry a positive multipath bias.  The standard PF treats all satellites
        equally, while PF3D detects NLOS and adjusts the likelihood.
        """
        true_pos = np.array([0.0, 0.0, 0.0])
        true_cb = 100.0
        building = _make_box_building()

        # 4 LOS + 2 NLOS satellites
        sat_ecef = np.array([
            [0.0,     0.0,   20000.0],   # LOS
            [0.0,   200.0,    5000.0],   # LOS
            [-200.0,  0.0,    5000.0],   # LOS
            [0.0,  -200.0,    5000.0],   # LOS
            [200.0,   0.0,      25.0],   # NLOS (behind building)
            [150.0,   0.0,      10.0],   # NLOS (behind building)
        ], dtype=np.float64)

        ranges = np.sqrt(np.sum((sat_ecef - true_pos) ** 2, axis=1))
        # Add multipath bias to NLOS satellites
        nlos_bias = 30.0
        pseudoranges = ranges + true_cb
        pseudoranges[4] += nlos_bias  # NLOS satellite 4
        pseudoranges[5] += nlos_bias  # NLOS satellite 5
        weights = np.ones(len(sat_ecef), dtype=np.float64)

        # --- Standard PF ---
        pf_std = ParticleFilter(
            n_particles=N_TEST, sigma_pos=1.0, sigma_cb=300.0,
            sigma_pr=5.0, resampling="systematic", seed=SEED)
        pf_std.initialize(position_ecef=true_pos, clock_bias=true_cb,
                          spread_pos=20.0, spread_cb=200.0)
        for _ in range(3):
            pf_std.predict(dt=0.0)
            pf_std.update(sat_ecef, pseudoranges, weights)
        result_std = pf_std.estimate()
        err_std = np.linalg.norm(result_std[:3] - true_pos)

        # --- PF3D ---
        pf_3d = ParticleFilter3D(
            building_model=building,
            n_particles=N_TEST, sigma_pos=1.0, sigma_cb=300.0,
            sigma_los=3.0, sigma_nlos=30.0, nlos_bias=20.0,
            resampling="systematic", seed=SEED)
        pf_3d.initialize(position_ecef=true_pos, clock_bias=true_cb,
                         spread_pos=20.0, spread_cb=200.0)
        for _ in range(3):
            pf_3d.predict(dt=0.0)
            pf_3d.update(sat_ecef, pseudoranges, weights)
        result_3d = pf_3d.estimate()
        err_3d = np.linalg.norm(result_3d[:3] - true_pos)

        # PF3D should produce a smaller position error
        assert err_3d < err_std, (
            f"PF3D error ({err_3d:.1f}m) should be less than "
            f"standard PF error ({err_std:.1f}m)")

    def test_no_building_equivalent_to_los(self):
        """With zero triangles, PF3D should behave like all-LOS weighting."""
        # Empty building model (no triangles)
        empty_building = BuildingModel(np.zeros((0, 3, 3), dtype=np.float64))

        n = 100
        px = np.zeros(n, dtype=np.float64)
        py = np.zeros(n, dtype=np.float64)
        pz = np.zeros(n, dtype=np.float64)
        pcb = np.full(n, 100.0, dtype=np.float64)

        sat_pos = np.array([0.0, 0.0, 20000.0], dtype=np.float64)
        dist = 20000.0
        pseudoranges = np.array([dist + 100.0], dtype=np.float64)
        weights_sat = np.ones(1, dtype=np.float64)

        sigma_los = 5.0

        # Standard weight
        log_w_std = np.zeros(n, dtype=np.float64)
        pf_weight(px, py, pz, pcb,
                  sat_pos, pseudoranges, weights_sat, log_w_std,
                  n, 1, sigma_los)

        # 3D weight with no triangles (all LOS)
        log_w_3d = np.zeros(n, dtype=np.float64)
        tri = empty_building.triangles.reshape(-1, 3, 3)
        pf_weight_3d(px, py, pz, pcb,
                     sat_pos, pseudoranges, weights_sat, tri,
                     log_w_3d, n, 1,
                     sigma_los, 30.0, 20.0)

        # Should produce identical weights (all LOS with same sigma)
        np.testing.assert_allclose(log_w_3d, log_w_std, atol=1e-10)


class TestPF3DFullPipeline:
    """Integration tests for the full PF3D pipeline."""

    def test_predict_update_cycle(self):
        """Test multiple predict-update cycles complete without error."""
        building = _make_box_building()
        pf = ParticleFilter3D(
            building_model=building,
            n_particles=1000,
            sigma_pos=1.0, sigma_cb=300.0,
            sigma_los=3.0, sigma_nlos=30.0, nlos_bias=20.0,
            seed=SEED)

        pf.initialize(position_ecef=[0.0, 0.0, 0.0], clock_bias=100.0,
                      spread_pos=10.0, spread_cb=100.0)

        sat_ecef = np.array([
            [0.0, 0.0, 20000.0],
            [200.0, 0.0, 25.0],
            [0.0, 200.0, 5000.0],
        ], dtype=np.float64)
        ranges = np.sqrt(np.sum(sat_ecef ** 2, axis=1))
        pseudoranges = ranges + 100.0

        for step in range(5):
            pf.predict(dt=1.0)
            pf.update(sat_ecef, pseudoranges)
            result = pf.estimate()
            assert result.shape == (4,)
            assert np.all(np.isfinite(result))

    def test_get_particles_works(self):
        """Test that get_particles returns valid data after 3D update."""
        building = _make_box_building()
        pf = ParticleFilter3D(
            building_model=building,
            n_particles=500,
            sigma_pos=1.0, sigma_cb=300.0,
            sigma_los=3.0, sigma_nlos=30.0, nlos_bias=20.0,
            seed=SEED)

        pf.initialize(position_ecef=[0.0, 0.0, 0.0], clock_bias=100.0,
                      spread_pos=10.0, spread_cb=100.0)

        sat_ecef = np.array([[0.0, 0.0, 20000.0]], dtype=np.float64)
        pseudoranges = np.array([20000.0 + 100.0])
        pf.update(sat_ecef, pseudoranges)

        particles = pf.get_particles()
        assert particles.shape == (500, 4)
        assert np.all(np.isfinite(particles))

    def test_ess_valid_after_3d_update(self):
        """ESS should be a valid positive number after 3D weight update."""
        building = _make_box_building()
        pf = ParticleFilter3D(
            building_model=building,
            n_particles=1000,
            sigma_pos=1.0, sigma_cb=300.0,
            sigma_los=3.0, sigma_nlos=30.0, nlos_bias=20.0,
            seed=SEED)

        pf.initialize(position_ecef=[0.0, 0.0, 0.0], clock_bias=100.0,
                      spread_pos=5.0, spread_cb=50.0)

        sat_ecef = np.array([
            [0.0, 0.0, 20000.0],
            [200.0, 0.0, 25.0],
        ], dtype=np.float64)
        ranges = np.sqrt(np.sum(sat_ecef ** 2, axis=1))
        pseudoranges = ranges + 100.0

        # Do update without resampling to check ESS
        pf._pf_weight_3d(
            pf._px, pf._py, pf._pz, pf._pcb,
            sat_ecef.ravel(), pseudoranges,
            np.ones(2, dtype=np.float64),
            building.triangles.reshape(-1, 3, 3),
            pf._log_weights,
            pf.n_particles, 2,
            3.0, 30.0, 20.0)

        ess = pf.get_ess()
        assert 0 < ess <= pf.n_particles


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
