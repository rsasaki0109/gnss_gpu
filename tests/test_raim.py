"""Tests for RAIM and FDE integrity monitoring (requires CUDA GPU)."""

import numpy as np
import pytest

from gnss_gpu.range_model import geometric_ranges_sagnac

try:
    from gnss_gpu._gnss_gpu import wls_position
    from gnss_gpu._gnss_gpu_raim import raim_check, raim_fde
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")


def _make_test_scenario():
    """Create a test scenario with 8 GPS satellites and clean pseudoranges.

    True receiver: Tokyo Station area (~35.68N, 139.77E)
    """
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

    weights = np.ones(len(sat_ecef))

    return sat_ecef, pseudoranges, weights, true_pos, true_cb


def test_raim_clean_data():
    """Clean pseudoranges should pass RAIM check."""
    sat_ecef, pseudoranges, weights, true_pos, true_cb = _make_test_scenario()

    result, _ = wls_position(sat_ecef.flatten(), pseudoranges, weights)

    raim = raim_check(sat_ecef.flatten(), pseudoranges, weights, result)

    assert raim.integrity_ok, (
        f"RAIM should pass for clean data: "
        f"test_stat={raim.test_statistic:.6f}, threshold={raim.threshold:.2f}"
    )
    assert raim.excluded_sat == -1
    assert raim.hpl > 0
    assert raim.vpl > 0
    assert raim.hpl < 1e6  # Should be a reasonable value, not infinity


def test_raim_detects_faulty_satellite():
    """A large pseudorange error on one satellite should be detected."""
    sat_ecef, pseudoranges, weights, true_pos, true_cb = _make_test_scenario()

    # Inject a large error (500m) on satellite 3
    faulty_pr = pseudoranges.copy()
    faulty_pr[3] += 500.0

    result, _ = wls_position(sat_ecef.flatten(), faulty_pr, weights)

    raim = raim_check(sat_ecef.flatten(), faulty_pr, weights, result)

    assert not raim.integrity_ok, (
        f"RAIM should detect fault: "
        f"test_stat={raim.test_statistic:.2f}, threshold={raim.threshold:.2f}"
    )
    assert raim.test_statistic > raim.threshold


def test_fde_excludes_faulty_satellite():
    """FDE should correctly identify and exclude the faulty satellite."""
    sat_ecef, pseudoranges, weights, true_pos, true_cb = _make_test_scenario()

    # Inject a large error (500m) on satellite 3
    faulty_pr = pseudoranges.copy()
    faulty_pr[3] += 500.0

    result, _ = wls_position(sat_ecef.flatten(), faulty_pr, weights)

    raim, corrected_pos = raim_fde(sat_ecef.flatten(), faulty_pr, weights, result)

    assert raim.integrity_ok, (
        f"FDE should recover integrity: "
        f"test_stat={raim.test_statistic:.6f}, threshold={raim.threshold:.2f}"
    )
    assert raim.excluded_sat == 3, (
        f"FDE should exclude satellite 3, got {raim.excluded_sat}"
    )

    # Corrected position should be close to truth
    pos_err = np.linalg.norm(corrected_pos[:3] - true_pos)
    assert pos_err < 1.0, f"Corrected position error {pos_err:.4f} m (expected < 1 m)"


def test_raim_with_noise():
    """Moderate noise should still pass RAIM with clean measurements."""
    sat_ecef, pseudoranges, weights, true_pos, true_cb = _make_test_scenario()

    rng = np.random.default_rng(42)
    noisy_pr = pseudoranges + rng.normal(0, 2.0, len(pseudoranges))

    result, _ = wls_position(sat_ecef.flatten(), noisy_pr, weights)

    raim = raim_check(sat_ecef.flatten(), noisy_pr, weights, result)

    # With 2m noise and 8 sats, this should generally pass
    # (chi-squared threshold is generous at p_fa=1e-5)
    assert raim.integrity_ok, (
        f"RAIM with moderate noise should pass: "
        f"test_stat={raim.test_statistic:.2f}, threshold={raim.threshold:.2f}"
    )


def test_fde_with_multiple_errors():
    """FDE with errors on two satellites should still detect anomaly."""
    sat_ecef, pseudoranges, weights, true_pos, true_cb = _make_test_scenario()

    # Inject errors on two satellites
    faulty_pr = pseudoranges.copy()
    faulty_pr[2] += 300.0
    faulty_pr[5] += 400.0

    result, _ = wls_position(sat_ecef.flatten(), faulty_pr, weights)

    raim_initial = raim_check(sat_ecef.flatten(), faulty_pr, weights, result)
    assert not raim_initial.integrity_ok, "Should detect dual-fault anomaly"


def test_raim_insufficient_satellites():
    """With exactly 4 satellites, RAIM has no redundancy (dof=0)."""
    sat_ecef, pseudoranges, weights, true_pos, true_cb = _make_test_scenario()

    # Use only first 4 satellites
    sat_4 = sat_ecef[:4]
    pr_4 = pseudoranges[:4]
    w_4 = weights[:4]

    result, _ = wls_position(sat_4.flatten(), pr_4, w_4)

    raim = raim_check(sat_4.flatten(), pr_4, w_4, result)

    # With 4 sats, dof=0, no test possible but integrity_ok should be True
    # (we trust the solution since we can't test it)
    assert raim.integrity_ok
    assert raim.test_statistic == 0.0
