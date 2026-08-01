from __future__ import annotations

import numpy as np

from experiments.audit_ppc_imu_contract import _best_time_shift, _correlation, _linear_fit


def test_correlation_and_linear_fit_recover_positive_axis_contract() -> None:
    expected = np.linspace(-3.0, 3.0, 101)
    measured = 1.02 * expected + 0.3
    assert np.isclose(_correlation(measured, expected), 1.0)
    fit = _linear_fit(measured, expected)
    assert fit is not None
    assert np.isclose(fit["scale"], 1.02)
    assert np.isclose(fit["bias"], 0.3)


def test_best_time_shift_recovers_timestamp_delay() -> None:
    imu_tow = np.arange(0.0, 20.0, 0.01)
    reference_tow = np.arange(1.0, 19.0, 0.2)
    signal = np.sin(0.7 * imu_tow) + 0.3 * np.sin(2.3 * imu_tow)
    target = np.sin(0.7 * (reference_tow - 0.12)) + 0.3 * np.sin(
        2.3 * (reference_tow - 0.12)
    )
    result = _best_time_shift(
        imu_tow,
        signal,
        reference_tow,
        target,
        np.ones(reference_tow.size, dtype=bool),
        maximum_shift_s=0.3,
        step_s=0.01,
    )
    assert np.isclose(result["best_shift_s"], 0.12, atol=0.01)
    assert result["best_correlation"] > 0.999
