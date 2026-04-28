from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.gsdc2023_diagnostics_mask import (
    apply_matlab_residual_diagnostics_mask,
    diagnostics_bool,
)


def test_diagnostics_bool_accepts_matlab_csv_values() -> None:
    assert diagnostics_bool(True)
    assert diagnostics_bool(np.bool_(True))
    assert diagnostics_bool("1")
    assert diagnostics_bool("true")
    assert diagnostics_bool("Y")
    assert diagnostics_bool(2.0)
    assert not diagnostics_bool(False)
    assert not diagnostics_bool(np.nan)
    assert not diagnostics_bool("")
    assert not diagnostics_bool("false")
    assert not diagnostics_bool("0")
    assert not diagnostics_bool(0.0)


def test_apply_matlab_residual_diagnostics_mask_restores_signal_weights(tmp_path) -> None:
    diagnostics_path = tmp_path / "phone_data_residual_diagnostics.csv"
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 1,
                "p_factor_finite": "1",
                "d_factor_finite": "1",
                "l_factor_finite": "1",
            },
            {
                "freq": "L1",
                "utcTimeMillis": 2000,
                "sys": 1,
                "svid": 1,
                "p_factor_finite": "0",
                "d_factor_finite": "0",
                "l_factor_finite": "1",
            },
            {
                "freq": "L5",
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 2,
                "p_factor_finite": "1",
                "d_factor_finite": "1",
                "l_factor_finite": "0",
            },
            {
                "freq": "L1",
                "utcTimeMillis": 1000,
                "sys": 8,
                "svid": 3,
                "p_factor_finite": "1",
                "d_factor_finite": "0",
                "l_factor_finite": "0",
            },
            {
                "freq": "L1",
                "utcTimeMillis": 9999,
                "sys": 1,
                "svid": 1,
                "p_factor_finite": "1",
                "d_factor_finite": "1",
                "l_factor_finite": "1",
            },
        ],
    ).to_csv(diagnostics_path, index=False)

    times_ms = np.array([1000.0, 2000.0], dtype=np.float64)
    slot_keys = (
        (1, 1, "GPS_L1_CA"),
        (1, 2, "GPS_L5_Q"),
        (6, 3, "GAL_E1_C_P"),
    )
    weights = np.full((2, 3), 9.0, dtype=np.float64)
    signal_weights = np.array([[0.25, 0.0, 0.75], [0.5, 0.0, 0.0]], dtype=np.float64)
    doppler_weights = np.full((2, 3), 8.0, dtype=np.float64)
    signal_doppler_weights = np.array([[4.0, 0.0, 2.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    tdcp_meas = np.array([[12.0, 13.0, 14.0]], dtype=np.float64)
    tdcp_weights = np.full((1, 3), 7.0, dtype=np.float64)
    signal_tdcp_weights = np.array([[42.0, 0.0, 11.0]], dtype=np.float64)

    p_count, d_count, l_pair_count = apply_matlab_residual_diagnostics_mask(
        diagnostics_path=diagnostics_path,
        times_ms=times_ms,
        slot_keys=slot_keys,
        weights=weights,
        signal_weights=signal_weights,
        doppler_weights=doppler_weights,
        signal_doppler_weights=signal_doppler_weights,
        tdcp_meas=tdcp_meas,
        tdcp_weights=tdcp_weights,
        signal_tdcp_weights=signal_tdcp_weights,
    )

    assert (p_count, d_count, l_pair_count) == (3, 2, 1)
    np.testing.assert_allclose(
        weights,
        np.array(
            [
                [0.25, 1.0, 0.75],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(
        doppler_weights,
        np.array(
            [
                [4.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(tdcp_weights, np.array([[42.0, 0.0, 0.0]], dtype=np.float64))


def test_apply_matlab_residual_diagnostics_mask_requires_columns(tmp_path) -> None:
    diagnostics_path = tmp_path / "bad.csv"
    pd.DataFrame([{"freq": "L1"}]).to_csv(diagnostics_path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        apply_matlab_residual_diagnostics_mask(
            diagnostics_path=diagnostics_path,
            times_ms=np.array([1000.0], dtype=np.float64),
            slot_keys=((1, 1, "GPS_L1_CA"),),
            weights=np.zeros((1, 1), dtype=np.float64),
            signal_weights=np.ones((1, 1), dtype=np.float64),
            doppler_weights=None,
            signal_doppler_weights=None,
            tdcp_meas=None,
            tdcp_weights=None,
            signal_tdcp_weights=None,
        )
