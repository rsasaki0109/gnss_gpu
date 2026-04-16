"""Tests for DD observation quality gates."""

from __future__ import annotations

import numpy as np

from gnss_gpu.dd_carrier import DDResult, GPS_L1_WAVELENGTH
from gnss_gpu.dd_pseudorange import DDPseudorangeResult
from gnss_gpu.dd_quality import (
    combine_sigma_scales,
    ess_gate_scale,
    gate_dd_carrier,
    gate_dd_pseudorange,
    metric_sigma_scale,
    pair_count_sigma_scale,
    spread_gate_scale,
)


def _fake_dd_geometry(n_dd: int):
    sat = np.repeat(np.array([[2.0, 0.0, 0.0]], dtype=np.float64), n_dd, axis=0)
    base_range = np.full(n_dd, 2.0, dtype=np.float64)
    return sat, base_range


def test_gate_dd_pseudorange_rejects_large_residual_pairs():
    sat, base_range = _fake_dd_geometry(3)
    dd = DDPseudorangeResult(
        dd_pseudorange_m=np.array([0.2, 4.1, -0.3], dtype=np.float64),
        sat_ecef_k=sat.copy(),
        sat_ecef_ref=sat.copy(),
        base_range_k=base_range.copy(),
        base_range_ref=base_range.copy(),
        dd_weights=np.ones(3, dtype=np.float64),
        ref_sat_ids=("G01", "G01", "G01"),
        n_dd=3,
    )

    filtered, stats = gate_dd_pseudorange(
        dd,
        np.zeros(3, dtype=np.float64),
        pair_residual_max_m=1.0,
        min_pairs=2,
    )

    assert filtered is not None
    assert filtered.n_dd == 2
    np.testing.assert_allclose(filtered.dd_pseudorange_m, np.array([0.2, -0.3], dtype=np.float64))
    assert stats.n_pair_rejected == 1
    assert not stats.rejected_by_epoch


def test_gate_dd_pseudorange_rejects_epoch_on_large_median():
    sat, base_range = _fake_dd_geometry(3)
    dd = DDPseudorangeResult(
        dd_pseudorange_m=np.array([2.1, 2.3, 1.9], dtype=np.float64),
        sat_ecef_k=sat.copy(),
        sat_ecef_ref=sat.copy(),
        base_range_k=base_range.copy(),
        base_range_ref=base_range.copy(),
        dd_weights=np.ones(3, dtype=np.float64),
        ref_sat_ids=("G01", "G01", "G01"),
        n_dd=3,
    )

    filtered, stats = gate_dd_pseudorange(
        dd,
        np.zeros(3, dtype=np.float64),
        pair_residual_max_m=3.0,
        epoch_median_residual_max_m=2.0,
        min_pairs=3,
    )

    assert filtered is None
    assert stats.rejected_by_epoch
    assert stats.n_pair_rejected == 0


def test_gate_dd_pseudorange_supports_adaptive_pair_threshold():
    sat, base_range = _fake_dd_geometry(4)
    dd = DDPseudorangeResult(
        dd_pseudorange_m=np.array([0.20, 0.25, 0.30, 4.00], dtype=np.float64),
        sat_ecef_k=sat.copy(),
        sat_ecef_ref=sat.copy(),
        base_range_k=base_range.copy(),
        base_range_ref=base_range.copy(),
        dd_weights=np.ones(4, dtype=np.float64),
        ref_sat_ids=("G01", "G01", "G01", "G01"),
        n_dd=4,
    )

    filtered, stats = gate_dd_pseudorange(
        dd,
        np.zeros(3, dtype=np.float64),
        adaptive_pair_floor_m=0.1,
        adaptive_pair_mad_mult=3.0,
        min_pairs=3,
    )

    assert filtered is not None
    assert filtered.n_dd == 3
    np.testing.assert_allclose(filtered.dd_pseudorange_m, np.array([0.20, 0.25, 0.30]))
    assert stats.n_pair_rejected == 1
    assert 0.30 < stats.pair_threshold < 1.0


def test_gate_dd_pseudorange_adaptive_floor_prevents_over_tight_gate():
    sat, base_range = _fake_dd_geometry(3)
    dd = DDPseudorangeResult(
        dd_pseudorange_m=np.array([0.20, 0.24, 0.40], dtype=np.float64),
        sat_ecef_k=sat.copy(),
        sat_ecef_ref=sat.copy(),
        base_range_k=base_range.copy(),
        base_range_ref=base_range.copy(),
        dd_weights=np.ones(3, dtype=np.float64),
        ref_sat_ids=("G01", "G01", "G01"),
        n_dd=3,
    )

    filtered, stats = gate_dd_pseudorange(
        dd,
        np.zeros(3, dtype=np.float64),
        adaptive_pair_floor_m=0.5,
        adaptive_pair_mad_mult=1.0,
        min_pairs=3,
    )

    assert filtered is not None
    assert filtered.n_dd == 3
    assert stats.n_pair_rejected == 0
    assert stats.pair_threshold == 0.5


def test_ess_gate_scale_tightens_low_ess_and_relaxes_high_ess():
    assert ess_gate_scale(0.05, min_scale=0.8, max_scale=1.2) == 0.8
    assert ess_gate_scale(0.95, min_scale=0.8, max_scale=1.2) == 1.2
    mid = ess_gate_scale(0.45, min_scale=0.8, max_scale=1.2)
    assert 0.8 < mid < 1.2


def test_spread_gate_scale_tightens_low_spread_and_relaxes_high_spread():
    assert spread_gate_scale(0.5, min_scale=0.85, max_scale=1.15) == 0.85
    assert spread_gate_scale(20.0, min_scale=0.85, max_scale=1.15) == 1.15
    mid = spread_gate_scale(4.0, min_scale=0.85, max_scale=1.15)
    assert 0.85 < mid < 1.15


def test_pair_count_sigma_scale_relaxes_sparse_support():
    assert pair_count_sigma_scale(4, low_pairs=4, high_pairs=8, max_scale=1.8) == 1.8
    assert pair_count_sigma_scale(8, low_pairs=4, high_pairs=8, max_scale=1.8) == 1.0
    mid = pair_count_sigma_scale(6, low_pairs=4, high_pairs=8, max_scale=1.8)
    assert 1.0 < mid < 1.8


def test_metric_sigma_scale_relaxes_large_metric():
    assert metric_sigma_scale(0.10, good_value=0.10, bad_value=0.20, max_scale=1.6) == 1.0
    assert metric_sigma_scale(0.20, good_value=0.10, bad_value=0.20, max_scale=1.6) == 1.6
    mid = metric_sigma_scale(0.15, good_value=0.10, bad_value=0.20, max_scale=1.6)
    assert 1.0 < mid < 1.6


def test_combine_sigma_scales_multiplies_and_clips():
    assert combine_sigma_scales(1.2, 1.1) == 1.32
    assert combine_sigma_scales(1.5, 1.4, max_scale=1.8) == 1.8


def test_gate_dd_pseudorange_applies_threshold_scale():
    sat, base_range = _fake_dd_geometry(3)
    dd = DDPseudorangeResult(
        dd_pseudorange_m=np.array([0.20, 0.40, 0.60], dtype=np.float64),
        sat_ecef_k=sat.copy(),
        sat_ecef_ref=sat.copy(),
        base_range_k=base_range.copy(),
        base_range_ref=base_range.copy(),
        dd_weights=np.ones(3, dtype=np.float64),
        ref_sat_ids=("G01", "G01", "G01"),
        n_dd=3,
    )

    filtered, stats = gate_dd_pseudorange(
        dd,
        np.zeros(3, dtype=np.float64),
        pair_residual_max_m=0.5,
        threshold_scale=0.9,
        min_pairs=2,
    )

    assert filtered is not None
    assert filtered.n_dd == 2
    assert stats.pair_threshold == 0.45
    np.testing.assert_allclose(filtered.dd_pseudorange_m, np.array([0.20, 0.40]))


def test_gate_dd_carrier_rejects_large_afv_pairs():
    sat, base_range = _fake_dd_geometry(3)
    dd = DDResult(
        dd_carrier_cycles=np.array([0.05, 0.42, -0.10], dtype=np.float64),
        sat_ecef_k=sat.copy(),
        sat_ecef_ref=sat.copy(),
        base_range_k=base_range.copy(),
        base_range_ref=base_range.copy(),
        dd_weights=np.ones(3, dtype=np.float64),
        wavelengths_m=np.full(3, GPS_L1_WAVELENGTH, dtype=np.float64),
        ref_sat_ids=("G01", "G01", "G01"),
        n_dd=3,
    )

    filtered, stats = gate_dd_carrier(
        dd,
        np.zeros(3, dtype=np.float64),
        pair_afv_max_cycles=0.2,
        min_pairs=2,
    )

    assert filtered is not None
    assert filtered.n_dd == 2
    np.testing.assert_allclose(filtered.dd_carrier_cycles, np.array([0.05, -0.10], dtype=np.float64))
    assert stats.n_pair_rejected == 1
    assert not stats.rejected_by_epoch


def test_gate_dd_carrier_rejects_epoch_on_large_median_afv():
    sat, base_range = _fake_dd_geometry(3)
    dd = DDResult(
        dd_carrier_cycles=np.array([0.24, -0.26, 0.23], dtype=np.float64),
        sat_ecef_k=sat.copy(),
        sat_ecef_ref=sat.copy(),
        base_range_k=base_range.copy(),
        base_range_ref=base_range.copy(),
        dd_weights=np.ones(3, dtype=np.float64),
        wavelengths_m=np.full(3, GPS_L1_WAVELENGTH, dtype=np.float64),
        ref_sat_ids=("G01", "G01", "G01"),
        n_dd=3,
    )

    filtered, stats = gate_dd_carrier(
        dd,
        np.zeros(3, dtype=np.float64),
        pair_afv_max_cycles=0.4,
        epoch_median_afv_max_cycles=0.2,
        min_pairs=3,
    )

    assert filtered is None
    assert stats.rejected_by_epoch
    assert stats.n_pair_rejected == 0


def test_gate_dd_carrier_supports_adaptive_pair_threshold():
    sat, base_range = _fake_dd_geometry(4)
    dd = DDResult(
        dd_carrier_cycles=np.array([0.05, 0.08, 0.09, 0.44], dtype=np.float64),
        sat_ecef_k=sat.copy(),
        sat_ecef_ref=sat.copy(),
        base_range_k=base_range.copy(),
        base_range_ref=base_range.copy(),
        dd_weights=np.ones(4, dtype=np.float64),
        wavelengths_m=np.full(4, GPS_L1_WAVELENGTH, dtype=np.float64),
        ref_sat_ids=("G01", "G01", "G01", "G01"),
        n_dd=4,
    )

    filtered, stats = gate_dd_carrier(
        dd,
        np.zeros(3, dtype=np.float64),
        adaptive_pair_floor_cycles=0.05,
        adaptive_pair_mad_mult=3.0,
        min_pairs=3,
    )

    assert filtered is not None
    assert filtered.n_dd == 3
    np.testing.assert_allclose(filtered.dd_carrier_cycles, np.array([0.05, 0.08, 0.09]))
    assert stats.n_pair_rejected == 1
    assert 0.09 < stats.pair_threshold < 0.3


def test_gate_dd_carrier_applies_threshold_scale():
    sat, base_range = _fake_dd_geometry(3)
    dd = DDResult(
        dd_carrier_cycles=np.array([0.05, 0.15, 0.24], dtype=np.float64),
        sat_ecef_k=sat.copy(),
        sat_ecef_ref=sat.copy(),
        base_range_k=base_range.copy(),
        base_range_ref=base_range.copy(),
        dd_weights=np.ones(3, dtype=np.float64),
        wavelengths_m=np.full(3, GPS_L1_WAVELENGTH, dtype=np.float64),
        ref_sat_ids=("G01", "G01", "G01"),
        n_dd=3,
    )

    filtered, stats = gate_dd_carrier(
        dd,
        np.zeros(3, dtype=np.float64),
        pair_afv_max_cycles=0.2,
        threshold_scale=0.8,
        min_pairs=2,
    )

    assert filtered is not None
    assert filtered.n_dd == 2
    assert stats.pair_threshold == 0.16000000000000003
    np.testing.assert_allclose(filtered.dd_carrier_cycles, np.array([0.05, 0.15]))
