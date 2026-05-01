import numpy as np
import pytest

from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.bvh import BVHAccelerator


class TestBVHBasic:
    """Basic BVH functionality tests."""

    def setup_method(self):
        self.building = BuildingModel.create_box(
            center=[100.0, 0.0, 25.0], width=20.0, depth=20.0, height=50.0
        )
        self.bvh = BVHAccelerator.from_building_model(self.building)
        self.rx_ecef = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def test_bvh_node_count(self):
        """BVH should have at least 1 node."""
        assert self.bvh.n_nodes >= 1

    def test_bvh_triangle_count(self):
        """BVH should preserve triangle count."""
        assert self.bvh.n_triangles == 12

    def test_los_clear_sky(self):
        """Satellite above with no building obstruction should be LOS."""
        sat = np.array([[0.0, 0.0, 20000000.0]], dtype=np.float64)
        is_los = self.bvh.check_los(self.rx_ecef, sat)
        assert is_los[0]

    def test_los_blocked(self):
        """Satellite behind building should be NLOS."""
        sat = np.array([[200.0, 0.0, 25.0]], dtype=np.float64)
        is_los = self.bvh.check_los(self.rx_ecef, sat)
        assert not is_los[0]

    def test_los_mixed(self):
        """Batch with both LOS and NLOS satellites."""
        sat_clear = np.array([[0.0, 0.0, 20000000.0]], dtype=np.float64)
        sat_blocked = np.array([[200.0, 0.0, 25.0]], dtype=np.float64)
        assert self.bvh.check_los(self.rx_ecef, sat_clear)[0]
        assert not self.bvh.check_los(self.rx_ecef, sat_blocked)[0]

    def test_from_building_model(self):
        """Test construction from BuildingModel."""
        bvh = BVHAccelerator.from_building_model(self.building)
        assert bvh.n_triangles == 12

    def test_create_box(self):
        """Test create_box class method."""
        bvh = BVHAccelerator.create_box([0, 0, 0], 10, 10, 10)
        assert bvh.n_triangles == 12

    def test_invalid_shape(self):
        """Invalid triangle shape should raise ValueError."""
        with pytest.raises(ValueError):
            BVHAccelerator(np.zeros((10, 2, 3)))


class TestBVHConsistency:
    """Verify BVH gives same results as linear scan for many triangles."""

    def _random_box_triangles(self, n_boxes, rng):
        """Generate triangle mesh from random boxes."""
        all_tris = []
        for _ in range(n_boxes):
            cx = rng.uniform(-500, 500)
            cy = rng.uniform(-500, 500)
            cz = rng.uniform(0, 100)
            w = rng.uniform(5, 30)
            d = rng.uniform(5, 30)
            h = rng.uniform(10, 60)
            model = BuildingModel.create_box([cx, cy, cz], w, d, h)
            all_tris.append(model.triangles)
        return np.concatenate(all_tris, axis=0)

    def test_consistency_1000_triangles(self):
        """BVH and linear scan should agree on 1000+ triangles."""
        rng = np.random.RandomState(42)
        tris = self._random_box_triangles(100, rng)  # 100 boxes = 1200 triangles
        assert tris.shape[0] >= 1000

        linear_model = BuildingModel(tris)
        bvh_model = BVHAccelerator(tris)

        rx = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Generate random satellite positions
        n_sat = 50
        sat_ecef = np.zeros((n_sat, 3), dtype=np.float64)
        for i in range(n_sat):
            # Random direction, far away
            theta = rng.uniform(0, 2 * np.pi)
            phi = rng.uniform(-np.pi / 2, np.pi / 2)
            r = rng.uniform(100, 30000000)
            sat_ecef[i, 0] = r * np.cos(phi) * np.cos(theta)
            sat_ecef[i, 1] = r * np.cos(phi) * np.sin(theta)
            sat_ecef[i, 2] = r * np.sin(phi)

        # Compare results one at a time to match test patterns
        for i in range(n_sat):
            sat_single = sat_ecef[i:i+1]
            los_linear = linear_model.check_los(rx, sat_single)
            los_bvh = bvh_model.check_los(rx, sat_single)
            assert los_linear[0] == los_bvh[0], (
                f"Mismatch at satellite {i}: linear={los_linear[0]}, bvh={los_bvh[0]}"
            )

    def test_consistency_batch(self):
        """BVH and linear scan batch results should match."""
        rng = np.random.RandomState(123)
        tris = self._random_box_triangles(90, rng)  # 1080 triangles

        linear_model = BuildingModel(tris)
        bvh_model = BVHAccelerator(tris)

        rx = np.array([0.0, 0.0, 50.0], dtype=np.float64)

        # Satellites at known directions
        sat_ecef = np.array([
            [0.0, 0.0, 20000000.0],   # straight up
            [1000.0, 0.0, 0.0],       # along x, ground level
            [0.0, 1000.0, 50.0],      # along y, same height
            [-500.0, 300.0, 10000.0],  # some angle
        ], dtype=np.float64)

        for i in range(len(sat_ecef)):
            sat_single = sat_ecef[i:i+1]
            los_linear = linear_model.check_los(rx, sat_single)
            los_bvh = bvh_model.check_los(rx, sat_single)
            assert los_linear[0] == los_bvh[0], (
                f"Mismatch at satellite {i}: linear={los_linear[0]}, bvh={los_bvh[0]}"
            )


class TestBVHBatch:
    """Batched check_los_batch / compute_multipath_batch tests."""

    def setup_method(self):
        self.building = BuildingModel.create_box(
            center=[100.0, 0.0, 25.0], width=20.0, depth=20.0, height=50.0
        )
        self.bvh = BVHAccelerator.from_building_model(self.building)

    def test_check_los_batch_matches_per_epoch(self):
        rng = np.random.RandomState(7)
        n_epoch = 16
        n_sat = 8

        rx = rng.uniform(-50, 50, size=(n_epoch, 3))
        sat = np.zeros((n_epoch, n_sat, 3), dtype=np.float64)
        for e in range(n_epoch):
            for s in range(n_sat):
                # half above (LOS), half behind the box (NLOS)
                if s % 2 == 0:
                    sat[e, s] = [0.0, 0.0, 2e7 + s]
                else:
                    sat[e, s] = [200.0, float(s), 25.0]

        batch_los = self.bvh.check_los_batch(rx, sat)

        single_los = np.zeros((n_epoch, n_sat), dtype=bool)
        for e in range(n_epoch):
            single_los[e] = self.bvh.check_los(rx[e], sat[e])

        np.testing.assert_array_equal(batch_los, single_los)

    def test_check_los_batch_handles_nan_pad(self):
        rx = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        sat = np.array(
            [
                [[0.0, 0.0, 2e7], [200.0, 0.0, 25.0]],
                [[0.0, 0.0, 2e7], [np.nan, np.nan, np.nan]],
            ],
            dtype=np.float64,
        )

        result = self.bvh.check_los_batch(rx, sat)
        assert result[0, 0]
        assert not result[0, 1]
        assert result[1, 0]
        assert not result[1, 1]

    def test_compute_multipath_batch_matches_per_epoch(self):
        # Reference is per-epoch compute_multipath called with n_sat=1 to bypass
        # an unrelated bug in the existing per-epoch multipath kernel that
        # broadcasts thread-0 results across threads when n_sat > 1.  The
        # batched kernel does not have this problem (it uses unique tid per
        # (epoch, sat) pair).
        rng = np.random.RandomState(11)
        n_epoch = 6
        n_sat = 4

        rx = rng.uniform(-50, 50, size=(n_epoch, 3))
        rx[:, 2] = rng.uniform(0, 30, size=n_epoch)
        sat = np.zeros((n_epoch, n_sat, 3), dtype=np.float64)
        for e in range(n_epoch):
            for s in range(n_sat):
                sat[e, s] = [200.0 + 10.0 * s, 5.0 * (e - 3), 25.0 + s]

        delay_batch, refl_batch = self.bvh.compute_multipath_batch(rx, sat)

        delay_ref = np.zeros((n_epoch, n_sat), dtype=np.float64)
        refl_ref = np.zeros((n_epoch, n_sat, 3), dtype=np.float64)
        for e in range(n_epoch):
            for s in range(n_sat):
                d, r = self.bvh.compute_multipath(rx[e], sat[e, s].reshape(1, 3))
                delay_ref[e, s] = d[0]
                refl_ref[e, s] = r[0]

        np.testing.assert_allclose(delay_batch, delay_ref, atol=1e-9)
        np.testing.assert_allclose(refl_batch, refl_ref, atol=1e-9)

    def test_check_los_batch_rejects_bad_shape(self):
        rx = np.zeros((4, 3), dtype=np.float64)
        bad_sat = np.zeros((4, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            self.bvh.check_los_batch(rx, bad_sat)

        rx_bad = np.zeros((4, 2), dtype=np.float64)
        sat = np.zeros((4, 2, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            self.bvh.check_los_batch(rx_bad, sat)

        rx_short = np.zeros((3, 3), dtype=np.float64)
        sat = np.zeros((4, 2, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            self.bvh.check_los_batch(rx_short, sat)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
