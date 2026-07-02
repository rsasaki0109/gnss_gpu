"""CPU-side validation tests for the multipath wrapper."""

import numpy as np
import pytest

from gnss_gpu.multipath import MultipathSimulator

PLANE = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


def test_multipath_init_rejects_invalid_config():
    with pytest.raises(ValueError, match="reflector_planes must have shape"):
        MultipathSimulator(np.zeros((2, 5)))
    with pytest.raises(ValueError, match="reflector_planes must contain at least one plane"):
        MultipathSimulator(np.zeros((0, 6)))
    with pytest.raises(ValueError, match="reflector_planes must be finite"):
        MultipathSimulator(np.array([[np.nan, 0, 0, 0, 0, 1]]))
    with pytest.raises(ValueError, match="carrier_freq must be positive"):
        MultipathSimulator(PLANE, carrier_freq=0.0)
    with pytest.raises(ValueError, match="chip_rate must be positive"):
        MultipathSimulator(PLANE, chip_rate=-1.0)
    with pytest.raises(ValueError, match="correlator_spacing must be positive"):
        MultipathSimulator(PLANE, correlator_spacing=np.inf)


def test_simulate_rejects_invalid_geometry_before_gpu_import():
    sim = MultipathSimulator(PLANE)

    with pytest.raises(ValueError, match="rx_ecef must have shape"):
        sim.simulate(np.zeros((2, 2)), [[0, 0, 1e7]])
    with pytest.raises(ValueError, match="sat_ecef must have shape"):
        sim.simulate([0, 0, 0], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="sat_ecef must contain at least one satellite"):
        sim.simulate([0, 0, 0], np.zeros((0, 3)))
    with pytest.raises(ValueError, match="rx_ecef and sat_ecef must be finite"):
        sim.simulate([np.nan, 0, 0], [[0, 0, 1e7]])


def test_corrupt_pseudoranges_rejects_invalid_inputs_before_gpu_import():
    sim = MultipathSimulator(PLANE)
    clean_pr = np.array([[1.0e7]])
    rx = np.array([[0.0, 0.0, 2.0]])
    sat = np.array([[[0.0, 0.0, 2.0e7]]])

    with pytest.raises(ValueError, match="clean_pr must have shape"):
        sim.corrupt_pseudoranges(np.zeros((2, 2, 1)), rx, sat)
    with pytest.raises(ValueError, match="rx_ecef must have the same number of epochs"):
        sim.corrupt_pseudoranges(clean_pr, np.zeros((2, 3)), sat)
    with pytest.raises(ValueError, match="sat_ecef must have shape"):
        sim.corrupt_pseudoranges(clean_pr, rx, np.zeros((1, 2)))
    with pytest.raises(ValueError, match="clean_pr must be finite"):
        sim.corrupt_pseudoranges(np.array([[np.nan]]), rx, sat)
