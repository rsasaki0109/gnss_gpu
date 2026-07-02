"""CPU-side validation tests for the skyplot wrapper."""

import numpy as np
import pytest

from gnss_gpu.skyplot import VulnerabilityMap

SAT_ECEF = np.array([[0.0, 0.0, 2.0e7]], dtype=np.float64)


def test_init_rejects_invalid_grid_config():
    origin = (35.68, 139.77, 30.0)

    with pytest.raises(ValueError, match="origin_lla must be finite"):
        VulnerabilityMap((35.68, np.nan, 30.0))
    with pytest.raises(ValueError, match="grid_size_m must be positive"):
        VulnerabilityMap(origin, grid_size_m=0.0)
    with pytest.raises(ValueError, match="resolution_m must be positive"):
        VulnerabilityMap(origin, resolution_m=-1.0)


def test_evaluate_rejects_invalid_satellites_before_gpu_import():
    vm = VulnerabilityMap(
        origin_lla=(35.68, 139.77, 30.0),
        grid_size_m=50,
        resolution_m=25,
    )

    with pytest.raises(ValueError, match="sat_ecef must have shape"):
        vm.evaluate(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="sat_ecef must contain at least one satellite"):
        vm.evaluate(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="sat_ecef must be finite"):
        vm.evaluate(np.array([[np.nan, 0.0, 1.0e7]]))
    with pytest.raises(ValueError, match="elevation_mask_deg must be finite"):
        vm.evaluate(SAT_ECEF, elevation_mask_deg=float("inf"))
