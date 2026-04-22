"""Tests for RTK carrier phase positioning (requires CUDA GPU)."""

import numpy as np
import pytest

try:
    from gnss_gpu._gnss_gpu_rtk import rtk_float, rtk_float_batch, lambda_integer
    HAS_RTK = True
except ImportError:
    HAS_RTK = False

pytestmark = pytest.mark.skipif(not HAS_RTK, reason="RTK CUDA module not available")

# GPS L1 wavelength [m]
L1_WAVELENGTH = 0.19029


def _make_rtk_scenario(baseline=(100.0, 50.0, -20.0), n_sat=8, seed=42):
    """Create a synthetic short-baseline RTK scenario.

    Base station at Tokyo Station area, rover offset by `baseline` in ECEF [m].
    Satellites at ~20200 km altitude. Observations are noise-free.

    Returns
    -------
    base_ecef, rover_ecef_true, sat_ecef, rover_pr, base_pr,
    rover_carrier, base_carrier, true_ambiguities
    """
    rng = np.random.default_rng(seed)

    # Base station ECEF (Tokyo Station area)
    base_ecef = np.array([-3957199.0, 3310205.0, 3737911.0])

    # True rover position
    rover_ecef_true = base_ecef + np.array(baseline)

    # Realistic GPS satellite ECEF positions
    sat_ecef = np.array([
        [-14985000.0, -3988000.0, 21474000.0],
        [-9575000.0, 15498000.0, 19457000.0],
        [7624000.0, -16218000.0, 19843000.0],
        [16305000.0, 12037000.0, 17183000.0],
        [-20889000.0, 13759000.0, 8291000.0],
        [5463000.0, 24413000.0, 8934000.0],
        [22169000.0, 3975000.0, 13781000.0],
        [-11527000.0, -19421000.0, 13682000.0],
    ])[:n_sat]

    # Geometric ranges
    r_base = np.sqrt(np.sum((sat_ecef - base_ecef) ** 2, axis=1))
    r_rover = np.sqrt(np.sum((sat_ecef - rover_ecef_true) ** 2, axis=1))

    # True integer DD ambiguities (arbitrary but fixed)
    true_amb_sd = rng.integers(-10, 11, size=n_sat)  # single-difference ambiguities

    # Pseudoranges (no clock bias in DD, but include common bias that cancels)
    common_bias = 5000.0  # common receiver clock bias [m] — cancels in DD
    base_pr = r_base + common_bias
    rover_pr = r_rover + common_bias + 50.0  # rover has different clock

    # Carrier phase [cycles]
    # phi = range / wavelength + N (ambiguity) + clock / wavelength
    base_carrier = r_base / L1_WAVELENGTH + true_amb_sd.astype(float)
    rover_carrier = r_rover / L1_WAVELENGTH + true_amb_sd.astype(float) + 50.0 / L1_WAVELENGTH

    # Compute true DD ambiguities for reference
    # Pick ref satellite (index 0 for simplicity in test verification)
    # DD_N_i = (N_rover_i - N_base_i) - (N_rover_ref - N_base_ref)
    # With our construction: SD ambiguities are the same for base and rover,
    # so DD ambiguities = 0. Let's add distinct rover ambiguities.
    rover_amb = rng.integers(-5, 6, size=n_sat).astype(float)
    base_amb = rng.integers(-5, 6, size=n_sat).astype(float)

    # Recompute carrier with distinct ambiguities
    base_carrier = r_base / L1_WAVELENGTH + base_amb
    rover_carrier = r_rover / L1_WAVELENGTH + rover_amb + 50.0 / L1_WAVELENGTH

    # The DD ambiguities (relative to highest-elevation ref sat) are integers
    # We don't know which ref sat the solver picks, so we'll verify position accuracy

    return (base_ecef, rover_ecef_true, sat_ecef,
            rover_pr, base_pr, rover_carrier, base_carrier)


def test_double_difference_formation():
    """Test that DD observations properly cancel common-mode errors."""
    base_ecef, rover_true, sat_ecef, rover_pr, base_pr, rcp, bcp = _make_rtk_scenario()

    # Single differences should cancel common clock bias
    sd_pr = rover_pr - base_pr
    # DD should further cancel satellite-dependent errors
    ref = 0
    dd_pr = sd_pr[1:] - sd_pr[ref]

    # DD pseudorange should approximate DD geometric range
    r_base = np.sqrt(np.sum((sat_ecef - base_ecef) ** 2, axis=1))
    r_rover = np.sqrt(np.sum((sat_ecef - rover_true) ** 2, axis=1))
    dd_geo = (r_rover[1:] - r_base[1:]) - (r_rover[ref] - r_base[ref])

    # Should match closely (no atmosphere in simulation)
    np.testing.assert_allclose(dd_pr, dd_geo, atol=1e-6)


def test_rtk_float_single():
    """Test single-epoch RTK float solution on short baseline."""
    base_ecef, rover_true, sat_ecef, rover_pr, base_pr, rcp, bcp = _make_rtk_scenario()

    np.zeros(3)
    n_sat = len(rover_pr)
    n_dd = n_sat - 1
    np.zeros(n_dd)
    np.zeros(2 * n_dd)

    pos, amb, res, iters = rtk_float(
        base_ecef, rover_pr, base_pr, rcp, bcp,
        sat_ecef.flatten(), L1_WAVELENGTH, 20, 1e-6)

    err = np.linalg.norm(pos - rover_true)
    assert err < 1.0, f"RTK float position error {err:.4f} m (should be < 1m)"
    assert iters <= 20, f"Did not converge in {iters} iterations"


def test_rtk_float_noise():
    """Test RTK float with pseudorange noise (carrier still clean)."""
    rng = np.random.default_rng(123)
    base_ecef, rover_true, sat_ecef, rover_pr, base_pr, rcp, bcp = _make_rtk_scenario()

    # Add pseudorange noise (3m sigma)
    rover_pr_noisy = rover_pr + rng.normal(0, 3.0, rover_pr.shape)
    base_pr_noisy = base_pr + rng.normal(0, 3.0, base_pr.shape)

    # Carrier phase noise (3mm sigma ~ 0.016 cycles at L1)
    rcp_noisy = rcp + rng.normal(0, 0.003 / L1_WAVELENGTH, rcp.shape)
    bcp_noisy = bcp + rng.normal(0, 0.003 / L1_WAVELENGTH, bcp.shape)

    pos, amb, res, iters = rtk_float(
        base_ecef, rover_pr_noisy, base_pr_noisy, rcp_noisy, bcp_noisy,
        sat_ecef.flatten(), L1_WAVELENGTH, 20, 1e-6)

    err = np.linalg.norm(pos - rover_true)
    # Float solution with noise: should be within ~1m
    assert err < 2.0, f"RTK float position error with noise {err:.4f} m"


def test_lambda_known_integers():
    """Test LAMBDA on float ambiguities close to known integers."""
    rng = np.random.default_rng(99)
    n = 3
    true_int = np.array([3, -2, 1], dtype=np.int32)
    float_amb = true_int.astype(np.float64) + rng.normal(0, 0.05, n)
    Q_amb = np.eye(n, dtype=np.float64) * 0.02
    fixed, ratio = lambda_integer(float_amb, Q_amb.ravel(), 200)
    np.testing.assert_array_equal(fixed, true_int)


def test_lambda_larger_noise():
    """Test LAMBDA with larger noise — should still fix with good geometry."""
    rng = np.random.default_rng(77)
    n = 3
    true_int = np.array([1, -1, 2], dtype=np.int32)
    float_amb = true_int.astype(np.float64) + rng.normal(0, 0.1, n)
    Q_amb = np.eye(n, dtype=np.float64) * 0.05
    fixed, ratio = lambda_integer(float_amb, Q_amb.ravel(), 200)
    np.testing.assert_array_equal(fixed, true_int)


def test_rtk_float_batch():
    """Test batch RTK float processing."""
    base_ecef, rover_true, sat_ecef, rover_pr, base_pr, rcp, bcp = _make_rtk_scenario()
    n_epoch = 50
    n_sat = len(rover_pr)

    rng = np.random.default_rng(55)

    # Tile observations with small noise variations
    rover_pr_batch = np.tile(rover_pr, (n_epoch, 1))
    base_pr_batch = np.tile(base_pr, (n_epoch, 1))
    rcp_batch = np.tile(rcp, (n_epoch, 1))
    bcp_batch = np.tile(bcp, (n_epoch, 1))
    sat_batch = np.tile(sat_ecef, (n_epoch, 1, 1))

    # Add small noise
    rover_pr_batch += rng.normal(0, 1.0, rover_pr_batch.shape)
    rcp_batch += rng.normal(0, 0.002 / L1_WAVELENGTH, rcp_batch.shape)

    results, ambiguities, iters = rtk_float_batch(
        base_ecef, rover_pr_batch, base_pr_batch, rcp_batch, bcp_batch,
        sat_batch, L1_WAVELENGTH, 20, 1e-6)

    assert results.shape == (n_epoch, 3)
    assert ambiguities.shape == (n_epoch, n_sat - 1)

    for i in range(n_epoch):
        err = np.linalg.norm(results[i] - rover_true)
        assert err < 10.0, f"Epoch {i}: RTK batch error {err:.4f} m"


def test_rtk_solver_class():
    """Test the RTKSolver Python wrapper class."""
    from gnss_gpu.rtk import RTKSolver

    base_ecef, rover_true, sat_ecef, rover_pr, base_pr, rcp, bcp = _make_rtk_scenario()

    solver = RTKSolver(base_ecef, wavelength=L1_WAVELENGTH)

    # Float solution
    pos, amb, res = solver.solve_float(rover_pr, base_pr, rcp, bcp, sat_ecef)
    err = np.linalg.norm(pos - rover_true)
    assert err < 1.0, f"RTKSolver float error {err:.4f} m"

    # Fixed solution
    pos_fix, fix_flag, ratio = solver.solve_fixed(rover_pr, base_pr, rcp, bcp, sat_ecef)
    err_fix = np.linalg.norm(pos_fix - rover_true)
    assert err_fix < 1.0, f"RTKSolver fixed error {err_fix:.4f} m"
