"""CPU-side validation tests for RTK wrappers."""

import numpy as np
import pytest

try:
    from gnss_gpu._gnss_gpu_rtk import rtk_float  # noqa: F401
    HAS_RTK = True
except ImportError:
    HAS_RTK = False

from gnss_gpu.rtk import RTKSolver

L1_WAVELENGTH = 0.19029
BASE_ECEF = np.array([-3957199.0, 3310205.0, 3737911.0], dtype=np.float64)
SAT_ECEF = np.array([
    [-14985000.0, -3988000.0, 21474000.0],
    [-9575000.0, 15498000.0, 19457000.0],
    [7624000.0, -16218000.0, 19843000.0],
], dtype=np.float64)
ROVER_PR = np.array([2.5e7, 2.4e7, 2.6e7], dtype=np.float64)
BASE_PR = np.array([2.4e7, 2.3e7, 2.5e7], dtype=np.float64)
ROVER_CARRIER = np.array([100.0, 101.0, 99.0], dtype=np.float64)
BASE_CARRIER = np.array([99.0, 100.0, 98.0], dtype=np.float64)


def test_rtk_init_rejects_invalid_base_ecef_before_native_call():
    with pytest.raises(ValueError, match="base_ecef must have shape"):
        RTKSolver([0.0, 0.0])
    with pytest.raises(ValueError, match="base_ecef must be finite"):
        RTKSolver([0.0, np.nan, 0.0])


def test_rtk_init_rejects_invalid_params_before_native_call():
    with pytest.raises(ValueError, match="wavelength must be positive"):
        RTKSolver(BASE_ECEF, wavelength=0.0)
    with pytest.raises(ValueError, match="max_iter must be positive"):
        RTKSolver(BASE_ECEF, max_iter=0)
    with pytest.raises(ValueError, match="tol must be finite"):
        RTKSolver(BASE_ECEF, tol=np.inf)


@pytest.mark.skipif(not HAS_RTK, reason="RTK CUDA module not available")
def test_rtk_solve_float_rejects_invalid_inputs_before_native_call():
    solver = RTKSolver(BASE_ECEF, wavelength=L1_WAVELENGTH)

    with pytest.raises(ValueError, match="rover_pr must contain at least"):
        solver.solve_float([], BASE_PR, ROVER_CARRIER, BASE_CARRIER, SAT_ECEF)
    with pytest.raises(ValueError, match="base_pr must contain at least"):
        solver.solve_float(ROVER_PR, ROVER_PR[:2], ROVER_CARRIER, BASE_CARRIER, SAT_ECEF)
    with pytest.raises(ValueError, match="sat_ecef shape must match"):
        solver.solve_float(ROVER_PR, BASE_PR, ROVER_CARRIER, BASE_CARRIER, SAT_ECEF[:2])
    with pytest.raises(ValueError, match="rover_pr must be finite"):
        bad_pr = ROVER_PR.copy()
        bad_pr[0] = np.nan
        solver.solve_float(bad_pr, BASE_PR, ROVER_CARRIER, BASE_CARRIER, SAT_ECEF)


@pytest.mark.skipif(not HAS_RTK, reason="RTK CUDA module not available")
def test_rtk_solve_fixed_rejects_invalid_n_candidates_before_native_call():
    solver = RTKSolver(BASE_ECEF, wavelength=L1_WAVELENGTH)

    with pytest.raises(ValueError, match="n_candidates must be positive"):
        solver.solve_fixed(
            ROVER_PR, BASE_PR, ROVER_CARRIER, BASE_CARRIER, SAT_ECEF,
            n_candidates=0,
        )


@pytest.mark.skipif(not HAS_RTK, reason="RTK CUDA module not available")
def test_rtk_solve_batch_rejects_invalid_inputs_before_native_call():
    solver = RTKSolver(BASE_ECEF, wavelength=L1_WAVELENGTH)
    rover_batch = np.tile(ROVER_PR, (2, 1))
    base_batch = np.tile(BASE_PR, (2, 1))
    rcp_batch = np.tile(ROVER_CARRIER, (2, 1))
    bcp_batch = np.tile(BASE_CARRIER, (2, 1))
    sat_batch = np.tile(SAT_ECEF, (2, 1, 1))

    with pytest.raises(ValueError, match="rover_pr must have shape"):
        solver.solve_batch(ROVER_PR, base_batch, rcp_batch, bcp_batch, sat_batch)
    with pytest.raises(ValueError, match="base_pr must have shape"):
        solver.solve_batch(rover_batch, BASE_PR, rcp_batch, bcp_batch, sat_batch)
    with pytest.raises(ValueError, match="sat_ecef must have shape"):
        solver.solve_batch(rover_batch, base_batch, rcp_batch, bcp_batch, SAT_ECEF)
