from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from experiments.diagnose_gsdc2023_vd_factor_residuals import _tdcp_residuals


def _batch() -> SimpleNamespace:
    sat = np.array(
        [
            [[20_200_000.0, 0.0, 0.0], [0.0, 20_200_000.0, 0.0]],
            [[20_200_000.0, 0.0, 0.0], [0.0, 20_200_000.0, 0.0]],
        ],
        dtype=np.float64,
    )
    return SimpleNamespace(
        sat_ecef=sat,
        times_ms=np.array([0.0, 1000.0], dtype=np.float64),
        dt=np.array([1.0], dtype=np.float64),
        weights=np.ones((2, 2), dtype=np.float64),
        tdcp_meas=np.array([[10.0, 20.0]], dtype=np.float64),
        tdcp_weights=np.ones((1, 2), dtype=np.float64),
        sys_kind=np.zeros((2, 2), dtype=np.int32),
        n_clock=1,
    )


def test_tdcp_residuals_separate_position_and_clock_delta_components() -> None:
    state = np.zeros((2, 8), dtype=np.float64)
    state[1, 0] = 2.0
    state[0, 6] = 100.0
    state[1, 6] = 107.0

    residual, predicted, observed, weights, position_component, clock_component = _tdcp_residuals(
        _batch(),
        state,
        tdcp_use_drift=False,
    )

    np.testing.assert_allclose(observed, [[10.0, 20.0]])
    np.testing.assert_allclose(weights, [[1.0, 1.0]])
    np.testing.assert_allclose(clock_component, [[7.0, 7.0]])
    np.testing.assert_allclose(predicted, position_component + clock_component)
    np.testing.assert_allclose(residual, observed - predicted)


def test_tdcp_residuals_broadcast_average_drift_component_to_all_slots() -> None:
    state = np.zeros((2, 8), dtype=np.float64)
    state[0, 7] = 4.0
    state[1, 7] = 6.0

    _residual, predicted, _observed, _weights, position_component, drift_component = _tdcp_residuals(
        _batch(),
        state,
        tdcp_use_drift=True,
    )

    np.testing.assert_allclose(drift_component, [[5.0, 5.0]])
    np.testing.assert_allclose(predicted, position_component + drift_component)
