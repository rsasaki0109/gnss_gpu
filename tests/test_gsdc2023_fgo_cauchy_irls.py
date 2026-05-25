"""Tests for ``experiments.gsdc2023_fgo_cauchy_irls``.

Exercises the Cauchy IRLS reweight without invoking the native CUDA
solver — the inner Sagnac-residual machinery is shared with
``pr_cost_host`` in ``src/positioning/fgo.cu`` so the unit-level test
covers parity of that geometry.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.gsdc2023_fgo_cauchy_irls import (
    CAUCHY_C_DEFAULT_M,
    CAUCHY_WEIGHT_FLOOR,
    _apply_cauchy_weights,
    _pr_residuals_vd,
)


def test_apply_cauchy_weights_downweights_outliers():
    w = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    # Mahalanobis distances of 0, 4, 40 metres -> w_eff = 1, 1/2, 1/(1+100)
    # for c=4: z/c = 0, 1, 10.
    res = np.array([[0.0, 4.0, 40.0]], dtype=np.float64)
    eff, ratio = _apply_cauchy_weights(w, res, cauchy_c=4.0)
    assert eff[0, 0] == pytest.approx(1.0)
    assert eff[0, 1] == pytest.approx(0.5)
    assert eff[0, 2] == pytest.approx(1.0 / (1.0 + 100.0))
    assert 0.0 < ratio < 1.0


def test_apply_cauchy_weights_floor_zeros_tiny_weights():
    w = np.array([[1.0]], dtype=np.float64)
    res = np.array([[1.0e6]], dtype=np.float64)  # Massive residual.
    eff, _ = _apply_cauchy_weights(w, res, cauchy_c=4.0)
    assert eff[0, 0] == 0.0  # Below CAUCHY_WEIGHT_FLOOR -> clamped to 0.


def test_apply_cauchy_weights_preserves_zero_weight_rows():
    w = np.array([[1.0, 0.0, 1.0]], dtype=np.float64)
    res = np.array([[1.0, 999.0, 1.0]], dtype=np.float64)
    eff, _ = _apply_cauchy_weights(w, res, cauchy_c=4.0)
    assert eff[0, 1] == 0.0  # Was zero, stays zero.
    assert eff[0, 0] > 0.0
    assert eff[0, 2] > 0.0


def test_pr_residuals_vd_matches_expected_geometry():
    n_epoch = 1
    n_sat = 1
    n_clock = 1
    sat_ecef = np.array([[[1.5e7, 0.0, 1.5e7]]], dtype=np.float64)
    rx_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    expected_geometric = np.linalg.norm(sat_ecef[0, 0] - rx_xyz)
    # The Sagnac rotation is tiny but non-zero; allow ~1 m tolerance
    # against a pure Euclidean range.
    measured_pr = expected_geometric  # No clock bias, no error.
    pseudorange = np.array([[measured_pr]], dtype=np.float64)
    state = np.zeros((n_epoch, 7 + n_clock), dtype=np.float64)  # x,v,c all zero.
    residuals = _pr_residuals_vd(
        sat_ecef, pseudorange, sys_kind=np.zeros((1, 1), dtype=np.int32),
        state=state, n_clock=n_clock,
    )
    # residual = pr - (r_geom + clk); pr ≈ r_geom and clk = 0, so |residual| < 1 m.
    assert abs(residuals[0, 0]) < 1.0


def test_apply_cauchy_weights_handles_nan_residual():
    w = np.array([[1.0, 1.0]], dtype=np.float64)
    res = np.array([[1.0, float("nan")]], dtype=np.float64)
    eff, _ = _apply_cauchy_weights(w, res, cauchy_c=4.0)
    assert eff[0, 0] > 0.0
    assert eff[0, 1] == 0.0


def test_apply_cauchy_weights_uses_provided_c():
    w = np.array([[1.0]], dtype=np.float64)
    res = np.array([[4.0]], dtype=np.float64)
    # c=4 -> 1/2; c=8 -> 1/(1 + (4/8)^2) = 1/(1.25) = 0.8.
    eff_c4, _ = _apply_cauchy_weights(w, res, cauchy_c=4.0)
    eff_c8, _ = _apply_cauchy_weights(w, res, cauchy_c=8.0)
    assert eff_c4[0, 0] == pytest.approx(0.5)
    assert eff_c8[0, 0] == pytest.approx(0.8)
    assert eff_c8[0, 0] > eff_c4[0, 0]
