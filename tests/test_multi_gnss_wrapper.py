"""CPU-side validation tests for the MultiGNSSSolver wrapper."""

import pytest

from gnss_gpu.multi_gnss import MultiGNSSSolver, SYSTEM_GPS


def test_solver_rejects_invalid_configuration():
    with pytest.raises(RuntimeError, match="systems must contain"):
        MultiGNSSSolver(systems=[])

    with pytest.raises(RuntimeError, match="systems must be unique"):
        MultiGNSSSolver(systems=[SYSTEM_GPS, SYSTEM_GPS])

    with pytest.raises(RuntimeError, match="supported GNSS system IDs"):
        MultiGNSSSolver(systems=[99])

    with pytest.raises(RuntimeError, match="max_iter must be >= 1"):
        MultiGNSSSolver(max_iter=0)

    with pytest.raises(RuntimeError, match="tol must be positive"):
        MultiGNSSSolver(tol=0.0)
