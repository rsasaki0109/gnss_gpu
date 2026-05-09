"""Tests for multi-GNSS WLS positioning with ISB estimation."""

import numpy as np
import pytest

from gnss_gpu.range_model import geometric_ranges_sagnac
from gnss_gpu.multi_gnss import (
    MultiGNSSSolver,
    SYSTEM_GPS,
    SYSTEM_GLONASS,
    SYSTEM_GALILEO,
    SYSTEM_BEIDOU,
    SYSTEM_QZSS,
)

try:
    from gnss_gpu._gnss_gpu_multi_gnss import wls_multi_gnss, wls_multi_gnss_batch
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def _make_gps_scenario():
    """Create a GPS-only test scenario (same as test_positioning)."""
    true_pos = np.array([-3957199.0, 3310205.0, 3737911.0])
    true_cb = 3000.0

    sat_ecef = np.array([
        [-14985000.0,  -3988000.0,  21474000.0],
        [ -9575000.0,  15498000.0,  19457000.0],
        [  7624000.0, -16218000.0,  19843000.0],
        [ 16305000.0,  12037000.0,  17183000.0],
        [-20889000.0,  13759000.0,   8291000.0],
        [  5463000.0,  24413000.0,   8934000.0],
        [ 22169000.0,   3975000.0,  13781000.0],
        [-11527000.0, -19421000.0,  13682000.0],
    ])

    ranges = geometric_ranges_sagnac(true_pos, sat_ecef)
    pseudoranges = ranges + true_cb
    system_ids = np.zeros(len(sat_ecef), dtype=np.int32)  # all GPS

    return sat_ecef, pseudoranges, system_ids, true_pos, true_cb


def _make_multi_gnss_scenario(isb_galileo=0.0, isb_glonass=0.0):
    """Create a multi-GNSS scenario with GPS + Galileo + GLONASS satellites.

    Args:
        isb_galileo: Inter-system bias for Galileo relative to GPS [m].
        isb_glonass: Inter-system bias for GLONASS relative to GPS [m].
    """
    true_pos = np.array([-3957199.0, 3310205.0, 3737911.0])
    true_cb_gps = 3000.0

    # GPS satellites
    gps_sats = np.array([
        [-14985000.0,  -3988000.0,  21474000.0],
        [ -9575000.0,  15498000.0,  19457000.0],
        [  7624000.0, -16218000.0,  19843000.0],
        [ 16305000.0,  12037000.0,  17183000.0],
    ])

    # Galileo satellites (different orbital altitude ~23222 km)
    gal_sats = np.array([
        [-12500000.0,   8700000.0,  22100000.0],
        [ 18300000.0,  -5200000.0,  20400000.0],
        [  3100000.0,  21800000.0,  15600000.0],
        [-17800000.0, -12100000.0,  16900000.0],
    ])

    # GLONASS satellites (altitude ~19100 km)
    glo_sats = np.array([
        [ -8900000.0,  18200000.0,  16500000.0],
        [ 14600000.0,  -9300000.0,  18700000.0],
        [-19200000.0,  -7100000.0,  14200000.0],
        [ 11200000.0,  16700000.0,  13800000.0],
    ])

    sat_ecef = np.vstack([gps_sats, gal_sats, glo_sats])
    system_ids = np.array([SYSTEM_GPS] * 4 + [SYSTEM_GALILEO] * 4 + [SYSTEM_GLONASS] * 4,
                          dtype=np.int32)

    # Compute pseudoranges with per-system clock biases
    ranges = geometric_ranges_sagnac(true_pos, sat_ecef)
    pseudoranges = ranges.copy()
    pseudoranges[:4] += true_cb_gps                          # GPS clock
    pseudoranges[4:8] += true_cb_gps + isb_galileo           # Galileo clock
    pseudoranges[8:12] += true_cb_gps + isb_glonass          # GLONASS clock

    return (sat_ecef, pseudoranges, system_ids, true_pos,
            true_cb_gps, isb_galileo, isb_glonass)


# --- CPU-only tests (always run) ---

class TestMultiGNSSCPU:
    """Test multi-GNSS solver using CPU fallback."""

    def test_single_system_matches_gps(self):
        """GPS-only multi-GNSS should give same result as standard WLS."""
        sat_ecef, pseudoranges, system_ids, true_pos, true_cb = _make_gps_scenario()
        solver = MultiGNSSSolver(systems=[SYSTEM_GPS])
        pos, biases, n_iter = solver.solve(sat_ecef, pseudoranges, system_ids)

        err = np.linalg.norm(pos - true_pos)
        assert err < 0.01, f"Position error {err:.4f} m"
        assert abs(biases[SYSTEM_GPS] - true_cb) < 0.01
        assert n_iter <= 10

    def test_dual_system_isb_zero(self):
        """GPS + Galileo with zero ISB should recover position and zero ISB."""
        (sat_ecef, pseudoranges, system_ids, true_pos,
         true_cb_gps, isb_gal, _) = _make_multi_gnss_scenario(isb_galileo=0.0)

        solver = MultiGNSSSolver(systems=[SYSTEM_GPS, SYSTEM_GALILEO])
        pos, biases, n_iter = solver.solve(sat_ecef[:8], pseudoranges[:8],
                                           system_ids[:8])

        err = np.linalg.norm(pos - true_pos)
        assert err < 0.1, f"Position error {err:.4f} m"
        # ISB should be near zero
        isb_estimated = biases[SYSTEM_GALILEO] - biases[SYSTEM_GPS]
        assert abs(isb_estimated) < 0.1, f"ISB error {isb_estimated:.4f} m"

    def test_dual_system_isb_nonzero(self):
        """GPS + Galileo with 10m ISB should recover the bias offset."""
        (sat_ecef, pseudoranges, system_ids, true_pos,
         true_cb_gps, isb_gal, _) = _make_multi_gnss_scenario(isb_galileo=10.0)

        solver = MultiGNSSSolver(systems=[SYSTEM_GPS, SYSTEM_GALILEO])
        pos, biases, n_iter = solver.solve(sat_ecef[:8], pseudoranges[:8],
                                           system_ids[:8])

        err = np.linalg.norm(pos - true_pos)
        assert err < 0.1, f"Position error {err:.4f} m"
        # ISB = cb_GAL - cb_GPS should be ~10m
        isb_estimated = biases[SYSTEM_GALILEO] - biases[SYSTEM_GPS]
        assert abs(isb_estimated - 10.0) < 0.1, \
            f"ISB error: expected 10.0, got {isb_estimated:.4f} m"

    def test_three_systems(self):
        """GPS + Galileo + GLONASS with different ISBs."""
        (sat_ecef, pseudoranges, system_ids, true_pos,
         true_cb_gps, isb_gal, isb_glo) = _make_multi_gnss_scenario(
             isb_galileo=15.0, isb_glonass=-8.0)

        solver = MultiGNSSSolver(systems=[SYSTEM_GPS, SYSTEM_GLONASS, SYSTEM_GALILEO])
        pos, biases, n_iter = solver.solve(sat_ecef, pseudoranges, system_ids)

        err = np.linalg.norm(pos - true_pos)
        assert err < 0.1, f"Position error {err:.4f} m"

        isb_gal_est = biases[SYSTEM_GALILEO] - biases[SYSTEM_GPS]
        isb_glo_est = biases[SYSTEM_GLONASS] - biases[SYSTEM_GPS]
        assert abs(isb_gal_est - 15.0) < 0.1, \
            f"Galileo ISB: expected 15.0, got {isb_gal_est:.4f}"
        assert abs(isb_glo_est - (-8.0)) < 0.1, \
            f"GLONASS ISB: expected -8.0, got {isb_glo_est:.4f}"

    def test_batch_cpu(self):
        """Batch processing via CPU fallback."""
        (sat_ecef, pseudoranges, system_ids, true_pos,
         true_cb_gps, _, _) = _make_multi_gnss_scenario(isb_galileo=5.0)

        n_epoch = 10
        rng = np.random.default_rng(42)
        sat_batch = np.tile(sat_ecef, (n_epoch, 1, 1))
        pr_batch = np.tile(pseudoranges, (n_epoch, 1))
        pr_batch += rng.normal(0, 2.0, pr_batch.shape)
        sys_batch = np.tile(system_ids, (n_epoch, 1))

        solver = MultiGNSSSolver(systems=[SYSTEM_GPS, SYSTEM_GLONASS, SYSTEM_GALILEO])
        positions, cb_arr, n_iters = solver.solve_batch(
            sat_batch, pr_batch, sys_batch)

        for i in range(n_epoch):
            err = np.linalg.norm(positions[i] - true_pos)
            assert err < 20.0, f"Epoch {i}: position error {err:.2f} m"

    def test_prn_to_system(self):
        """PRN string parsing."""
        assert MultiGNSSSolver.prn_to_system("G01") == (SYSTEM_GPS, 1)
        assert MultiGNSSSolver.prn_to_system("R05") == (SYSTEM_GLONASS, 5)
        assert MultiGNSSSolver.prn_to_system("E12") == (SYSTEM_GALILEO, 12)
        assert MultiGNSSSolver.prn_to_system("C03") == (SYSTEM_BEIDOU, 3)
        assert MultiGNSSSolver.prn_to_system("J07") == (SYSTEM_QZSS, 7)

    def test_system_name(self):
        """System name lookup."""
        assert MultiGNSSSolver.system_name(SYSTEM_GPS) == "GPS"
        assert MultiGNSSSolver.system_name(SYSTEM_GALILEO) == "Galileo"

    def test_insufficient_satellites(self):
        """Should return failure with fewer sats than unknowns."""
        solver = MultiGNSSSolver(systems=[SYSTEM_GPS, SYSTEM_GALILEO])
        sat_ecef = np.array([[20000000.0, 0, 0], [0, 20000000.0, 0]])
        pr = np.array([22000000.0, 22000000.0])
        sys_ids = np.array([SYSTEM_GPS, SYSTEM_GALILEO])

        pos, biases, n_iter = solver.solve(sat_ecef, pr, sys_ids)
        assert n_iter == -1


# --- GPU tests (require CUDA module) ---

@pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
class TestMultiGNSSGPU:
    """Test multi-GNSS solver using GPU backend."""

    def test_single_system_gpu(self):
        sat_ecef, pseudoranges, system_ids, true_pos, true_cb = _make_gps_scenario()
        weights = np.ones(len(pseudoranges))
        result, n_iter = wls_multi_gnss(
            sat_ecef.ravel(), pseudoranges, weights, system_ids, 1, 10, 1e-4)

        pos = result[:3]
        err = np.linalg.norm(pos - true_pos)
        assert err < 0.01, f"Position error {err:.4f} m"

    def test_dual_system_gpu(self):
        (sat_ecef, pseudoranges, system_ids, true_pos,
         true_cb_gps, isb_gal, _) = _make_multi_gnss_scenario(isb_galileo=10.0)

        # Remap: GPS=0, Galileo=1 (contiguous)
        mapped = np.where(system_ids == SYSTEM_GALILEO, 1, 0).astype(np.int32)
        weights = np.ones(8)
        result, n_iter = wls_multi_gnss(
            sat_ecef[:8].ravel(), pseudoranges[:8], weights, mapped[:8], 2, 10, 1e-4)

        pos = result[:3]
        err = np.linalg.norm(pos - true_pos)
        assert err < 0.1, f"Position error {err:.4f} m"
        isb_est = result[4] - result[3]  # cb_GAL - cb_GPS
        assert abs(isb_est - 10.0) < 0.1

    def test_batch_gpu(self):
        (sat_ecef, pseudoranges, system_ids, true_pos,
         true_cb_gps, _, _) = _make_multi_gnss_scenario(isb_galileo=5.0)

        n_epoch = 50
        rng = np.random.default_rng(42)
        sat_batch = np.tile(sat_ecef, (n_epoch, 1, 1))
        pr_batch = np.tile(pseudoranges, (n_epoch, 1))
        pr_batch += rng.normal(0, 2.0, pr_batch.shape)

        # Remap to contiguous: GPS=0, GLO=1, GAL=2
        sys_map = {SYSTEM_GPS: 0, SYSTEM_GLONASS: 1, SYSTEM_GALILEO: 2}
        mapped = np.array([sys_map[s] for s in system_ids], dtype=np.int32)
        sys_batch = np.tile(mapped, (n_epoch, 1))
        w_batch = np.ones_like(pr_batch)

        results, iters = wls_multi_gnss_batch(
            sat_batch, pr_batch, w_batch,
            np.ascontiguousarray(sys_batch),
            3, 10, 1e-4)

        for i in range(n_epoch):
            err = np.linalg.norm(results[i, :3] - true_pos)
            assert err < 20.0, f"Epoch {i}: position error {err:.2f} m"
