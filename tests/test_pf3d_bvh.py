"""Tests for BVH-accelerated 3D-aware particle filter.

Requires CUDA GPU and compiled pf3d_bvh module.

Key test: verify that the BVH kernel produces log-weights that are
numerically identical (or within floating-point tolerance) to the
linear-scan kernel for the same inputs, including a scenario with
1000+ triangles.
"""

import numpy as np
import pytest

try:
    from gnss_gpu._gnss_gpu_pf3d_bvh import pf_weight_3d_bvh
    from gnss_gpu._gnss_gpu_pf3d import pf_weight_3d
    from gnss_gpu.particle_filter_3d_bvh import ParticleFilter3DBVH
    from gnss_gpu.particle_filter_3d import ParticleFilter3D
    from gnss_gpu.bvh import BVHAccelerator
    from gnss_gpu.raytrace import BuildingModel
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")

N_TEST = 10000
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_box_building():
    """Single box building: center (100, 0, 25), 20x20x50."""
    return BuildingModel.create_box(
        center=[100.0, 0.0, 25.0], width=20.0, depth=20.0, height=50.0)


def _make_multi_building_mesh(n_boxes=100, seed=7):
    """Generate a dense urban mesh from n_boxes random boxes (12 tris each)."""
    rng = np.random.RandomState(seed)
    all_tris = []
    for _ in range(n_boxes):
        cx = rng.uniform(-500, 500)
        cy = rng.uniform(-500, 500)
        cz = rng.uniform(0, 100)
        w  = rng.uniform(5, 30)
        d  = rng.uniform(5, 30)
        h  = rng.uniform(10, 60)
        m  = BuildingModel.create_box([cx, cy, cz], w, d, h)
        all_tris.append(m.triangles)
    return np.concatenate(all_tris, axis=0)


def _run_linear_scan(px, py, pz, pcb, sat_ecef, pr, ws, tris,
                     sigma_los, sigma_nlos, nlos_bias):
    """Run linear-scan kernel and return log_weights."""
    n = len(px)
    n_sat = len(pr)
    log_w = np.zeros(n, dtype=np.float64)
    pf_weight_3d(px, py, pz, pcb,
                 sat_ecef.ravel(), pr, ws,
                 tris.reshape(-1, 3, 3),
                 log_w, n, n_sat,
                 sigma_los, sigma_nlos, nlos_bias)
    return log_w


def _run_bvh(px, py, pz, pcb, sat_ecef, pr, ws, bvh_acc,
             sigma_los, sigma_nlos, nlos_bias):
    """Run BVH kernel and return log_weights."""
    n = len(px)
    n_sat = len(pr)
    log_w = np.zeros(n, dtype=np.float64)
    pf_weight_3d_bvh(px, py, pz, pcb,
                     sat_ecef.ravel(), pr, ws,
                     bvh_acc._nodes_flat,
                     bvh_acc._sorted_tris,
                     log_w, n, n_sat,
                     sigma_los, sigma_nlos, nlos_bias)
    return log_w


# ---------------------------------------------------------------------------
# Unit tests: kernel correctness
# ---------------------------------------------------------------------------

class TestWeight3DBVHKernel:
    """Low-level correctness tests for pf_weight_3d_bvh kernel."""

    def test_los_particles_zero_residual(self):
        """LOS particles with perfect pseudoranges should give log_w ~ 0."""
        building = _make_box_building()
        bvh = BVHAccelerator.from_building_model(building)
        n = 100
        px = np.zeros(n, dtype=np.float64)
        py = np.zeros(n, dtype=np.float64)
        pz = np.zeros(n, dtype=np.float64)
        pcb = np.full(n, 100.0, dtype=np.float64)

        sat = np.array([0.0, 0.0, 20000.0], dtype=np.float64)
        dist = float(np.linalg.norm(sat))
        pr = np.array([dist + 100.0], dtype=np.float64)
        ws = np.ones(1, dtype=np.float64)

        log_w = _run_bvh(px, py, pz, pcb,
                         sat.reshape(1, 3), pr, ws, bvh,
                         sigma_los=3.0, sigma_nlos=30.0, nlos_bias=20.0)
        assert np.allclose(log_w, 0.0, atol=1e-10)

    def test_nlos_particles_zero_residual_after_bias(self):
        """NLOS particles whose obs_pr includes nlos_bias exactly get log_w ~ 0."""
        building = _make_box_building()
        bvh = BVHAccelerator.from_building_model(building)
        n = 100
        nlos_bias = 20.0
        px = np.zeros(n, dtype=np.float64)
        py = np.zeros(n, dtype=np.float64)
        pz = np.zeros(n, dtype=np.float64)
        pcb = np.full(n, 100.0, dtype=np.float64)

        # Satellite behind building -> NLOS
        sat = np.array([200.0, 0.0, 25.0], dtype=np.float64)
        dist = float(np.linalg.norm(sat))
        pr = np.array([dist + 100.0 + nlos_bias], dtype=np.float64)
        ws = np.ones(1, dtype=np.float64)

        log_w = _run_bvh(px, py, pz, pcb,
                         sat.reshape(1, 3), pr, ws, bvh,
                         sigma_los=3.0, sigma_nlos=30.0, nlos_bias=nlos_bias)
        assert np.allclose(log_w, 0.0, atol=1e-10)

    def test_nlos_penalizes_less_than_los_sigma(self):
        """NLOS satellite with large residual should penalize less than LOS sigma."""
        building = _make_box_building()
        bvh = BVHAccelerator.from_building_model(building)
        n = 50
        px = np.zeros(n, dtype=np.float64)
        py = np.zeros(n, dtype=np.float64)
        pz = np.zeros(n, dtype=np.float64)
        pcb = np.full(n, 100.0, dtype=np.float64)

        # Large extra error beyond nlos_bias
        sat = np.array([200.0, 0.0, 25.0], dtype=np.float64)
        dist = float(np.linalg.norm(sat))
        pr = np.array([dist + 100.0 + 50.0], dtype=np.float64)
        ws = np.ones(1, dtype=np.float64)

        sigma_los  = 3.0
        sigma_nlos = 30.0
        nlos_bias  = 20.0

        # BVH (uses sigma_nlos for NLOS)
        log_w_bvh = _run_bvh(px, py, pz, pcb,
                              sat.reshape(1, 3), pr, ws, bvh,
                              sigma_los, sigma_nlos, nlos_bias)

        # Compute expected penalty using sigma_los (harsher)
        residual_los = pr[0] - (dist + 100.0)
        log_w_los_sigma = -0.5 * (residual_los ** 2) / (sigma_los ** 2)
        assert np.all(log_w_bvh > log_w_los_sigma)

    def test_nlos_negative_residual_does_not_apply_positive_bias(self):
        """Negative NLOS residuals should not be shifted by a positive bias term."""
        building = _make_box_building()
        bvh = BVHAccelerator.from_building_model(building)
        n = 32
        px = np.zeros(n, dtype=np.float64)
        py = np.zeros(n, dtype=np.float64)
        pz = np.zeros(n, dtype=np.float64)
        pcb = np.full(n, 100.0, dtype=np.float64)

        sat = np.array([200.0, 0.0, 25.0], dtype=np.float64)
        dist = float(np.linalg.norm(sat))
        nlos_bias = 20.0
        pr = np.array([dist + 100.0 - nlos_bias], dtype=np.float64)
        ws = np.ones(1, dtype=np.float64)
        sigma_nlos = 30.0

        log_w = _run_bvh(
            px, py, pz, pcb,
            sat.reshape(1, 3), pr, ws, bvh,
            sigma_los=3.0, sigma_nlos=sigma_nlos, nlos_bias=nlos_bias,
        )

        expected = -0.5 * (nlos_bias ** 2) / (sigma_nlos ** 2)
        assert np.allclose(log_w, expected, atol=1e-10)

    def test_empty_mesh_all_los(self):
        """With zero triangles every satellite is LOS; kernel must not crash."""
        empty = BVHAccelerator(np.zeros((0, 3, 3), dtype=np.float64))
        n = 50
        px = np.zeros(n, dtype=np.float64)
        py = np.zeros(n, dtype=np.float64)
        pz = np.zeros(n, dtype=np.float64)
        pcb = np.full(n, 0.0, dtype=np.float64)

        sat = np.array([0.0, 0.0, 20000.0], dtype=np.float64)
        dist = 20000.0
        pr = np.array([dist], dtype=np.float64)
        ws = np.ones(1, dtype=np.float64)

        log_w = _run_bvh(px, py, pz, pcb,
                         sat.reshape(1, 3), pr, ws, empty,
                         sigma_los=5.0, sigma_nlos=30.0, nlos_bias=20.0)
        # Perfect residual -> all zero
        assert np.allclose(log_w, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Consistency tests: BVH vs linear scan must agree exactly
# ---------------------------------------------------------------------------

class TestBVHMatchesLinearScan:
    """BVH and linear-scan kernels must produce identical log-weights."""

    def _common_particles(self, n):
        rng = np.random.RandomState(99)
        px  = rng.uniform(-10, 10, n).astype(np.float64)
        py  = rng.uniform(-10, 10, n).astype(np.float64)
        pz  = rng.uniform(-10, 10, n).astype(np.float64)
        pcb = rng.uniform(50, 150, n).astype(np.float64)
        return px, py, pz, pcb

    def test_simple_box_building(self):
        """Single box building (12 triangles): BVH == linear scan."""
        building = _make_box_building()
        bvh = BVHAccelerator.from_building_model(building)
        tris = building.triangles  # shape (12, 3, 3)

        n = 200
        px, py, pz, pcb = self._common_particles(n)

        sat = np.array([
            [0.0,   0.0,  20000.0],
            [200.0, 0.0,     25.0],
            [0.0, 200.0,   5000.0],
            [-200.0, 0.0,  5000.0],
        ], dtype=np.float64)
        n_sat = len(sat)
        pr = np.sqrt(np.sum(sat**2, axis=1)) + 100.0
        ws = np.ones(n_sat, dtype=np.float64)

        sigma_los  = 3.0
        sigma_nlos = 30.0
        nlos_bias  = 20.0

        log_w_lin = _run_linear_scan(px, py, pz, pcb, sat, pr, ws, tris,
                                     sigma_los, sigma_nlos, nlos_bias)
        log_w_bvh = _run_bvh(px, py, pz, pcb, sat, pr, ws, bvh,
                              sigma_los, sigma_nlos, nlos_bias)

        np.testing.assert_allclose(log_w_bvh, log_w_lin, atol=1e-10,
                                   err_msg="BVH and linear scan differ for simple box")

    def test_1000_triangles(self):
        """Urban mesh with 1200+ triangles: BVH must match linear scan exactly."""
        tris = _make_multi_building_mesh(n_boxes=105, seed=17)
        assert tris.shape[0] >= 1000, f"Expected >=1000 tris, got {tris.shape[0]}"

        building = BuildingModel(tris)
        bvh = BVHAccelerator(tris)

        n = 500
        px, py, pz, pcb = self._common_particles(n)

        rng = np.random.RandomState(55)
        n_sat = 8
        sat = np.zeros((n_sat, 3), dtype=np.float64)
        for i in range(n_sat):
            theta = rng.uniform(0, 2 * np.pi)
            phi   = rng.uniform(np.radians(15), np.radians(80))
            r     = rng.uniform(500, 30_000_000)
            sat[i, 0] = r * np.cos(phi) * np.cos(theta)
            sat[i, 1] = r * np.cos(phi) * np.sin(theta)
            sat[i, 2] = r * np.sin(phi)

        pr = np.sqrt(np.sum(sat**2, axis=1)) + 100.0
        ws = np.ones(n_sat, dtype=np.float64)

        sigma_los  = 3.0
        sigma_nlos = 30.0
        nlos_bias  = 20.0

        log_w_lin = _run_linear_scan(px, py, pz, pcb, sat, pr, ws, tris,
                                     sigma_los, sigma_nlos, nlos_bias)
        log_w_bvh = _run_bvh(px, py, pz, pcb, sat, pr, ws, bvh,
                              sigma_los, sigma_nlos, nlos_bias)

        np.testing.assert_allclose(
            log_w_bvh, log_w_lin, atol=1e-10,
            err_msg=f"BVH/linear mismatch for {tris.shape[0]}-triangle mesh")

    def test_varied_particle_positions(self):
        """Particles scattered widely: BVH == linear scan for complex occlusion patterns."""
        building = _make_box_building()
        bvh  = BVHAccelerator.from_building_model(building)
        tris = building.triangles

        # Place particles around the building at different z heights
        n = 300
        px  = np.tile([0.0, 50.0, 110.0, -50.0, 0.0], 60)[:n].astype(np.float64)
        py  = np.tile([0.0, 0.0, 0.0, 0.0, 10.0], 60)[:n].astype(np.float64)
        pz  = np.linspace(0.0, 60.0, n, dtype=np.float64)
        pcb = np.full(n, 100.0, dtype=np.float64)

        sat = np.array([
            [0.0,   0.0,  20000.0],
            [200.0, 0.0,     25.0],
            [200.0, 0.0,    100.0],
            [0.0, -200.0,  5000.0],
        ], dtype=np.float64)
        pr = np.sqrt(np.sum(sat**2, axis=1)) + 100.0
        ws = np.ones(len(sat), dtype=np.float64)

        log_w_lin = _run_linear_scan(px, py, pz, pcb, sat, pr, ws, tris,
                                     3.0, 30.0, 20.0)
        log_w_bvh = _run_bvh(px, py, pz, pcb, sat, pr, ws, bvh,
                              3.0, 30.0, 20.0)

        np.testing.assert_allclose(log_w_bvh, log_w_lin, atol=1e-10)


# ---------------------------------------------------------------------------
# ParticleFilter3DBVH class tests
# ---------------------------------------------------------------------------

class TestParticleFilter3DBVH:
    """High-level tests for the ParticleFilter3DBVH wrapper class."""

    def _make_pf(self, building=None, n_particles=1000):
        if building is None:
            building = _make_box_building()
        bvh = BVHAccelerator.from_building_model(building)
        return ParticleFilter3DBVH(
            bvh=bvh, n_particles=n_particles,
            sigma_pos=1.0, sigma_cb=300.0,
            sigma_los=3.0, sigma_nlos=30.0, nlos_bias=20.0,
            seed=SEED)

    def test_construction(self):
        """Constructor stores parameters correctly."""
        bvh = BVHAccelerator.from_building_model(_make_box_building())
        pf = ParticleFilter3DBVH(
            bvh=bvh, n_particles=500,
            sigma_pos=1.0, sigma_cb=300.0,
            sigma_los=4.0, sigma_nlos=25.0, nlos_bias=15.0)
        assert pf.sigma_los   == 4.0
        assert pf.sigma_nlos  == 25.0
        assert pf.nlos_bias   == 15.0
        assert pf.n_particles == 500

    def test_invalid_bvh_type(self):
        """Passing a non-BVHAccelerator raises TypeError."""
        with pytest.raises(TypeError):
            ParticleFilter3DBVH(bvh="not_a_bvh", n_particles=100)

    def test_update_runs(self):
        """update() completes without error."""
        pf = self._make_pf()
        pf.initialize(position_ecef=[0.0, 0.0, 0.0], clock_bias=100.0,
                      spread_pos=10.0, spread_cb=100.0)

        sat = np.array([[0.0, 0.0, 20000.0], [200.0, 0.0, 25.0]],
                       dtype=np.float64)
        pr = np.sqrt(np.sum(sat**2, axis=1)) + 100.0
        pf.update(sat, pr)
        result = pf.estimate()
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))

    def test_estimate_shape(self):
        """estimate() always returns shape (4,)."""
        pf = self._make_pf(n_particles=500)
        pf.initialize(position_ecef=[0.0, 0.0, 0.0], clock_bias=100.0,
                      spread_pos=5.0, spread_cb=50.0)
        sat = np.array([[0.0, 0.0, 20000.0]], dtype=np.float64)
        pr = np.array([20000.0 + 100.0])
        pf.update(sat, pr)
        assert pf.estimate().shape == (4,)

    def test_predict_update_cycle(self):
        """Multiple predict-update cycles produce finite estimates."""
        pf = self._make_pf()
        pf.initialize(position_ecef=[0.0, 0.0, 0.0], clock_bias=100.0,
                      spread_pos=5.0, spread_cb=50.0)

        sat = np.array([
            [0.0,   0.0,  20000.0],
            [200.0, 0.0,     25.0],
            [0.0, 200.0,   5000.0],
        ], dtype=np.float64)
        pr = np.sqrt(np.sum(sat**2, axis=1)) + 100.0

        for _ in range(5):
            pf.predict(dt=1.0)
            pf.update(sat, pr)
            result = pf.estimate()
            assert np.all(np.isfinite(result))

    def test_get_particles_shape(self):
        """get_particles() returns (n_particles, 4) array."""
        pf = self._make_pf(n_particles=200)
        pf.initialize(position_ecef=[0.0, 0.0, 0.0], clock_bias=100.0,
                      spread_pos=5.0, spread_cb=50.0)
        sat = np.array([[0.0, 0.0, 20000.0]], dtype=np.float64)
        pf.update(sat, np.array([20000.0 + 100.0]))
        parts = pf.get_particles()
        assert parts.shape == (200, 4)
        assert np.all(np.isfinite(parts))

    def test_ess_positive(self):
        """ESS is in (0, n_particles] after an update."""
        pf = self._make_pf(n_particles=500)
        pf.initialize(position_ecef=[0.0, 0.0, 0.0], clock_bias=100.0,
                      spread_pos=5.0, spread_cb=50.0)
        sat = np.array([[0.0, 0.0, 20000.0]], dtype=np.float64)
        pr  = np.array([20000.0 + 100.0])
        pf.update(sat, pr)
        ess = pf.get_ess()
        assert 0 < ess <= pf.n_particles

    def test_estimate_near_truth(self):
        """BVH-PF converges near the true position over several updates."""
        true_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        true_cb  = 100.0
        building = _make_box_building()

        sat = np.array([
            [0.0,    0.0,  20000.0],
            [0.0,  200.0,   5000.0],
            [-200.0, 0.0,   5000.0],
            [0.0, -200.0,   5000.0],
            [200.0,  0.0,     25.0],  # NLOS
        ], dtype=np.float64)

        ranges = np.sqrt(np.sum((sat - true_pos)**2, axis=1))
        pr = ranges + true_cb
        pr[4] += 20.0  # NLOS bias
        ws = np.ones(len(sat), dtype=np.float64)

        bvh = BVHAccelerator.from_building_model(building)
        pf = ParticleFilter3DBVH(
            bvh=bvh, n_particles=N_TEST,
            sigma_pos=1.0, sigma_cb=300.0,
            sigma_los=3.0, sigma_nlos=30.0, nlos_bias=20.0,
            resampling="systematic", seed=SEED)
        pf.initialize(position_ecef=true_pos, clock_bias=true_cb,
                      spread_pos=20.0, spread_cb=200.0)

        for _ in range(3):
            pf.predict(dt=0.0)
            pf.update(sat, pr, ws)

        result = pf.estimate()
        pos_error = np.linalg.norm(result[:3] - true_pos)
        assert pos_error < 30.0, f"Position error {pos_error:.1f} m too large"


# ---------------------------------------------------------------------------
# Equivalence: ParticleFilter3DBVH vs ParticleFilter3D must give same result
# ---------------------------------------------------------------------------

class TestBVHVsLinearPF:
    """The BVH PF and linear-scan PF should produce the same log-weights."""

    def test_log_weights_match(self):
        """ParticleFilter3DBVH and ParticleFilter3D produce identical log-weights
        when initialized with the same particles and observations."""
        building = _make_box_building()
        bvh = BVHAccelerator.from_building_model(building)

        n = 1000
        rng = np.random.RandomState(13)
        px  = rng.uniform(-5, 5, n).astype(np.float64)
        py  = rng.uniform(-5, 5, n).astype(np.float64)
        pz  = rng.uniform(-5, 5, n).astype(np.float64)
        pcb = rng.uniform(90, 110, n).astype(np.float64)

        sat = np.array([
            [0.0,   0.0,  20000.0],
            [200.0, 0.0,     25.0],
            [0.0, 200.0,   5000.0],
        ], dtype=np.float64)
        pr = np.sqrt(np.sum(sat**2, axis=1)) + 100.0
        ws = np.ones(len(sat), dtype=np.float64)

        sigma_los  = 3.0
        sigma_nlos = 30.0
        nlos_bias  = 20.0

        # Linear scan
        log_w_lin = np.zeros(n, dtype=np.float64)
        pf_weight_3d(px, py, pz, pcb,
                     sat.ravel(), pr, ws,
                     building.triangles.reshape(-1, 3, 3),
                     log_w_lin, n, len(sat),
                     sigma_los, sigma_nlos, nlos_bias)

        # BVH
        log_w_bvh = np.zeros(n, dtype=np.float64)
        pf_weight_3d_bvh(px, py, pz, pcb,
                         sat.ravel(), pr, ws,
                         bvh._nodes_flat,
                         bvh._sorted_tris,
                         log_w_bvh, n, len(sat),
                         sigma_los, sigma_nlos, nlos_bias)

        np.testing.assert_allclose(
            log_w_bvh, log_w_lin, atol=1e-10,
            err_msg="ParticleFilter3DBVH log-weights differ from linear-scan")

    def test_1200_triangles_log_weights_match(self):
        """1200-triangle urban mesh: BVH kernel matches linear-scan kernel."""
        tris = _make_multi_building_mesh(n_boxes=100, seed=3)
        assert tris.shape[0] >= 1000

        building = BuildingModel(tris)
        bvh = BVHAccelerator(tris)

        n = 300
        rng = np.random.RandomState(77)
        px  = rng.uniform(-10, 10, n).astype(np.float64)
        py  = rng.uniform(-10, 10, n).astype(np.float64)
        pz  = rng.uniform(-10, 10, n).astype(np.float64)
        pcb = rng.uniform(90, 110, n).astype(np.float64)

        n_sat = 6
        sat = np.zeros((n_sat, 3), dtype=np.float64)
        for i in range(n_sat):
            theta = i * (2 * np.pi / n_sat)
            phi   = np.radians(30 + i * 5)
            r     = 20_000_000.0
            sat[i] = [r*np.cos(phi)*np.cos(theta),
                      r*np.cos(phi)*np.sin(theta),
                      r*np.sin(phi)]
        pr = np.sqrt(np.sum(sat**2, axis=1)) + 100.0
        ws = np.ones(n_sat, dtype=np.float64)

        log_w_lin = _run_linear_scan(px, py, pz, pcb, sat, pr, ws, tris,
                                     3.0, 30.0, 20.0)
        log_w_bvh = _run_bvh(px, py, pz, pcb, sat, pr, ws, bvh,
                              3.0, 30.0, 20.0)

        np.testing.assert_allclose(
            log_w_bvh, log_w_lin, atol=1e-10,
            err_msg=f"Mismatch for {tris.shape[0]}-triangle mesh")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
