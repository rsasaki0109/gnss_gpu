"""CPU-side validation tests for the diffraction wrapper."""

from types import SimpleNamespace

import numpy as np
import pytest

from gnss_gpu.diffraction import compute_diffraction_paths, compute_diffraction_paths_gpu


def _make_edges():
    start = np.array([[100.0, 0.0, 0.0]])
    end = np.array([[100.0, 0.0, 50.0]])
    return SimpleNamespace(
        start=start,
        end=end,
        midpoint=0.5 * (start + end),
        size=1,
    )


def _make_scenario():
    rx = np.array([-3957199.0, 3310205.0, 3737911.0])
    sat = np.array([[-14985000.0, -3988000.0, 21474000.0]])
    edges = _make_edges()
    return rx, sat, edges


def test_compute_diffraction_paths_rejects_invalid_inputs():
    rx, sat, edges = _make_scenario()

    with pytest.raises(ValueError, match="rx_ecef must have shape"):
        compute_diffraction_paths(rx.reshape(1, 3), sat, edges)

    with pytest.raises(ValueError, match="sat_ecef must have shape"):
        compute_diffraction_paths(rx, sat.ravel()[:-1], edges)

    with pytest.raises(ValueError, match="max_edge_range_m must be positive"):
        compute_diffraction_paths(rx, sat, edges, max_edge_range_m=0.0)

    with pytest.raises(ValueError, match="wavelength_m must be positive"):
        compute_diffraction_paths(rx, sat, edges, wavelength_m=-1.0)


def test_compute_diffraction_paths_rejects_nonfinite_inputs():
    rx, sat, edges = _make_scenario()

    bad_rx = rx.copy()
    bad_rx[0] = np.nan
    with pytest.raises(ValueError, match="rx_ecef must be finite"):
        compute_diffraction_paths(bad_rx, sat, edges)

    bad_sat = sat.copy()
    bad_sat[0, 0] = np.inf
    with pytest.raises(ValueError, match="sat_ecef must be finite"):
        compute_diffraction_paths(rx, bad_sat, edges)


def test_compute_diffraction_paths_gpu_rejects_before_extension_import():
    rx, sat, edges = _make_scenario()

    with pytest.raises(ValueError, match="max_paths must be a non-negative integer"):
        compute_diffraction_paths_gpu(rx, sat, edges, max_paths=-1)

    bad_rx = rx.copy()
    bad_rx[2] = np.nan
    with pytest.raises(ValueError, match="rx_ecef must be finite"):
        compute_diffraction_paths_gpu(bad_rx, sat, edges)
