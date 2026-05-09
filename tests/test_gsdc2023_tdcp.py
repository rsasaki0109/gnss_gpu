from __future__ import annotations

import numpy as np
import pytest

from experiments.gsdc2023_residual_model import geometric_range_with_sagnac
from experiments.gsdc2023_tdcp import (
    ADR_STATE_CYCLE_SLIP,
    ADR_STATE_RESET,
    ADR_STATE_VALID,
    DEFAULT_TDCP_WEIGHT_SCALE,
    TDCP_LOFFSET_M,
    TDCP_WEIGHT_SCALE_IDENTITY,
    apply_tdcp_geometry_correction,
    apply_tdcp_weight_scale,
    build_tdcp_arrays,
    tdcp_enabled_for_phone,
    tdcp_loffset_m,
    tdcp_use_drift_for_phone,
    valid_adr_state,
)


def test_tdcp_phone_family_policy_direct():
    assert tdcp_enabled_for_phone("pixel5", requested=True)
    assert not tdcp_enabled_for_phone("pixel5", requested=False)
    assert not tdcp_enabled_for_phone("sm-a325f", requested=True)

    assert tdcp_loffset_m("sm-a205u") == TDCP_LOFFSET_M
    assert tdcp_loffset_m("pixel5") == 0.0
    assert tdcp_use_drift_for_phone("sm-a205u")
    assert not tdcp_use_drift_for_phone("pixel5")


def test_valid_adr_state_direct():
    assert valid_adr_state(ADR_STATE_VALID)
    assert not valid_adr_state(0)
    assert not valid_adr_state(ADR_STATE_VALID | ADR_STATE_RESET)
    assert not valid_adr_state(ADR_STATE_VALID | ADR_STATE_CYCLE_SLIP)


def test_build_tdcp_arrays_counts_consistency_rejects_and_ignores_masked_doppler_direct():
    adr = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float64)
    adr_state = np.ones_like(adr, dtype=np.int32)
    adr_uncertainty = np.full_like(adr, 0.02)
    doppler = np.full_like(adr, -20.0)
    doppler_weights = np.ones_like(adr)
    doppler_weights[0, 1] = 0.0

    tdcp_meas, tdcp_weights, mask_count = build_tdcp_arrays(
        adr,
        adr_state,
        adr_uncertainty,
        doppler,
        np.array([1.0, 0.0], dtype=np.float64),
        consistency_threshold_m=1.5,
        doppler_weights=doppler_weights,
    )

    assert mask_count == 1
    assert tdcp_meas is not None
    assert tdcp_weights is not None
    assert tdcp_weights[0, 0] == 0.0
    assert tdcp_weights[0, 1] > 0.0


def test_build_tdcp_arrays_propagates_consistency_rejects_to_adjacent_pairs_direct():
    adr = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [100.0, 2.0],
            [101.0, 3.0],
        ],
        dtype=np.float64,
    )
    adr_state = np.ones_like(adr, dtype=np.int32)
    adr_uncertainty = np.full_like(adr, 0.02)
    doppler = np.full_like(adr, -1.0)

    tdcp_meas, tdcp_weights, mask_count = build_tdcp_arrays(
        adr,
        adr_state,
        adr_uncertainty,
        doppler,
        np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64),
        consistency_threshold_m=1.5,
    )

    assert mask_count == 1
    assert tdcp_meas is not None
    assert tdcp_weights is not None
    assert np.all(tdcp_weights[:, 0] == 0.0)
    assert np.all(tdcp_weights[:, 1] > 0.0)


def test_build_tdcp_arrays_uses_matlab_scalar_interval_for_consistency_direct():
    adr = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    adr_state = np.ones_like(adr, dtype=np.int32)
    adr_uncertainty = np.full_like(adr, 0.02)
    doppler = np.full_like(adr, -1.0)

    tdcp_meas, tdcp_weights, mask_count = build_tdcp_arrays(
        adr,
        adr_state,
        adr_uncertainty,
        doppler,
        np.array([1.0, 3.0, 0.0], dtype=np.float64),
        consistency_threshold_m=1.5,
    )

    assert mask_count == 0
    assert tdcp_meas is not None
    assert tdcp_weights is not None
    assert np.all(tdcp_weights[:, 0] > 0.0)


def test_apply_tdcp_weight_scale_direct():
    weights = np.array([[2.0, 3.0]], dtype=np.float64)

    apply_tdcp_weight_scale(weights, TDCP_WEIGHT_SCALE_IDENTITY)
    np.testing.assert_allclose(weights, np.array([[2.0, 3.0]]))

    apply_tdcp_weight_scale(weights, DEFAULT_TDCP_WEIGHT_SCALE)
    np.testing.assert_allclose(weights, np.array([[2.0, 3.0]]) * DEFAULT_TDCP_WEIGHT_SCALE)

    weights = np.array([[2.0, 3.0]], dtype=np.float64)
    apply_tdcp_weight_scale(weights, 0.5)
    np.testing.assert_allclose(weights, np.array([[1.0, 1.5]]))

    apply_tdcp_weight_scale(weights, 0.0)
    np.testing.assert_array_equal(weights, np.zeros_like(weights))

    with pytest.raises(ValueError):
        apply_tdcp_weight_scale(weights, float("nan"))


def test_apply_tdcp_geometry_correction_direct():
    tdcp_meas = np.array([[10.0, 20.0]], dtype=np.float64)
    tdcp_weights = np.array([[1.0, 0.0]], dtype=np.float64)
    sat_ecef = np.array(
        [
            [[22_000_000.0, 14_000_000.0, 21_000_000.0], [23_000_000.0, 14_000_000.0, 21_000_000.0]],
            [[22_000_050.0, 14_000_000.0, 21_000_000.0], [23_000_100.0, 14_000_000.0, 21_000_000.0]],
        ],
        dtype=np.float64,
    )
    reference_xyz = np.array(
        [
            [-3_947_460.0, 3_431_490.0, 3_637_870.0],
            [-3_947_459.0, 3_431_490.0, 3_637_870.0],
        ],
        dtype=np.float64,
    )
    expected_delta = geometric_range_with_sagnac(sat_ecef[1, 0], reference_xyz[1]) - geometric_range_with_sagnac(
        sat_ecef[0, 0],
        reference_xyz[0],
    )

    corrected = apply_tdcp_geometry_correction(tdcp_meas, tdcp_weights, sat_ecef, reference_xyz)

    assert corrected == 1
    np.testing.assert_allclose(tdcp_meas[0, 0], 10.0 - expected_delta, rtol=1e-12)
    assert tdcp_meas[0, 1] == 20.0
