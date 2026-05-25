"""Tests for ``experiments.gsdc2023_taroz_weighting``.

Pins the obserrmodel.m SN-branch formula against hand-computed scalars and
exercises the constellation/L1-L5 -> sigtype lookup table.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.gsdc2023_taroz_weighting import (
    CONSTELLATION_BEIDOU,
    CONSTELLATION_GALILEO,
    CONSTELLATION_GLONASS,
    CONSTELLATION_GPS,
    CONSTELLATION_QZSS,
    D_SN_RATIO_DEFAULT,
    L_SN_RATIO_DEFAULT,
    P_SN_RATIO_DEFAULT,
    SIGTYPE_BDS_B1,
    SIGTYPE_BDS_B2A,
    SIGTYPE_FACTOR,
    SIGTYPE_GAL_E1,
    SIGTYPE_GAL_E5,
    SIGTYPE_GLO_G1,
    SIGTYPE_GPS_L1,
    SIGTYPE_GPS_L5,
    SIGTYPE_OTHER,
    SN_DENOMINATOR_DEFAULT,
    SN_PERCENTILE_DEFAULT,
    TripCN0Percentiles,
    compute_carrier_sigma_sn,
    compute_doppler_sigma_sn,
    compute_pr_sigma_sn,
    compute_sn_base,
    compute_trip_cn0_percentiles,
    constellation_signal_to_sigtype,
    sigma_to_weight,
    sigtype_factor,
)


def test_sigtype_factor_table_matches_taroz_parameters_m():
    # parameters.m: prm.sigtype_factor = [0.8, 1.5, 0.8, 0.8, 0.5, 0.5, 0.5, NaN]
    assert SIGTYPE_FACTOR[:7] == (0.8, 1.5, 0.8, 0.8, 0.5, 0.5, 0.5)
    assert math.isnan(SIGTYPE_FACTOR[7])


@pytest.mark.parametrize(
    "constellation, is_l5, expected",
    [
        (CONSTELLATION_GPS, False, SIGTYPE_GPS_L1),
        (CONSTELLATION_GLONASS, False, SIGTYPE_GLO_G1),
        (CONSTELLATION_GALILEO, False, SIGTYPE_GAL_E1),
        (CONSTELLATION_BEIDOU, False, SIGTYPE_BDS_B1),
        (CONSTELLATION_GPS, True, SIGTYPE_GPS_L5),
        (CONSTELLATION_GALILEO, True, SIGTYPE_GAL_E5),
        (CONSTELLATION_BEIDOU, True, SIGTYPE_BDS_B2A),
        (CONSTELLATION_GLONASS, True, SIGTYPE_OTHER),
        (CONSTELLATION_QZSS, False, SIGTYPE_GPS_L1),  # QZSS folded into GPS family
        (CONSTELLATION_QZSS, True, SIGTYPE_GPS_L5),
        (99, False, SIGTYPE_OTHER),
    ],
)
def test_constellation_signal_to_sigtype_lookup(constellation: int, is_l5: bool, expected: int) -> None:
    assert constellation_signal_to_sigtype(constellation, is_l5) == expected


def test_sn_base_zero_at_percentile():
    # When CN0 equals the percentile, sn_base = 10^0 = 1.0 exactly.
    assert compute_sn_base(45.0, 45.0) == pytest.approx(1.0, rel=0, abs=0)


def test_sn_base_decade_at_one_sn_den_below_percentile():
    # CN0 = percentile - sn_den -> sn_base = 10^1 = 10.
    assert compute_sn_base(25.0, 45.0, sn_den=SN_DENOMINATOR_DEFAULT) == pytest.approx(10.0)


def test_sn_base_decade_at_one_sn_den_above_percentile():
    # CN0 = percentile + sn_den -> sn_base = 10^-1 = 0.1.
    assert compute_sn_base(65.0, 45.0, sn_den=SN_DENOMINATOR_DEFAULT) == pytest.approx(0.1)


def test_sn_base_nan_on_invalid_inputs():
    assert math.isnan(compute_sn_base(float("nan"), 45.0))
    assert math.isnan(compute_sn_base(45.0, float("nan")))
    assert math.isnan(compute_sn_base(45.0, 45.0, sn_den=0.0))


def test_pr_sigma_at_percentile_gps_l1():
    # σ_P = P_sn_ratio * 1.0 * sigfactor[GPS_L1] = 1.0 * 1.0 * 0.8 = 0.8
    s = compute_pr_sigma_sn(45.0, 45.0, SIGTYPE_GPS_L1)
    assert s == pytest.approx(P_SN_RATIO_DEFAULT * 0.8)


def test_pr_sigma_glonass_g1_is_1p5x_gps():
    # GLONASS gets the highest sigtype factor (1.5x) per taroz parameters.m.
    s_gps = compute_pr_sigma_sn(45.0, 45.0, SIGTYPE_GPS_L1)
    s_glo = compute_pr_sigma_sn(45.0, 45.0, SIGTYPE_GLO_G1)
    assert s_glo / s_gps == pytest.approx(1.5 / 0.8)


def test_pr_sigma_l5_is_half_of_l1():
    s_l1 = compute_pr_sigma_sn(45.0, 45.0, SIGTYPE_GPS_L1)
    s_l5 = compute_pr_sigma_sn(45.0, 45.0, SIGTYPE_GPS_L5)
    assert s_l5 / s_l1 == pytest.approx(0.5 / 0.8)


def test_pr_sigma_nan_for_unknown_signal():
    assert math.isnan(compute_pr_sigma_sn(45.0, 45.0, SIGTYPE_OTHER))


def test_doppler_sigma_has_no_sigtype_factor():
    # σ_D = D_sn_ratio * sn_base; intentionally factor-free per obserrmodel.m.
    s = compute_doppler_sigma_sn(45.0, 45.0)
    assert s == pytest.approx(D_SN_RATIO_DEFAULT)


def test_carrier_sigma_applies_sigtype_factor():
    s_gps = compute_carrier_sigma_sn(45.0, 45.0, SIGTYPE_GPS_L1)
    assert s_gps == pytest.approx(L_SN_RATIO_DEFAULT * 0.8)


def test_compute_trip_cn0_percentiles_uses_finite_samples():
    l1 = np.array([20.0, 30.0, 40.0, 50.0, 60.0, float("nan")])
    l5 = np.array([25.0, 35.0, 45.0])
    ps = compute_trip_cn0_percentiles((l1, l5), percentile=SN_PERCENTILE_DEFAULT)
    expected_l1 = float(np.percentile([20.0, 30.0, 40.0, 50.0, 60.0], 85))
    expected_l5 = float(np.percentile([25.0, 35.0, 45.0], 85))
    assert ps.l1 == pytest.approx(expected_l1)
    assert ps.l5 == pytest.approx(expected_l5)
    assert ps.for_frequency(is_l5=False) == ps.l1
    assert ps.for_frequency(is_l5=True) == ps.l5


def test_compute_trip_cn0_percentiles_handles_empty_frequency_buckets():
    ps = compute_trip_cn0_percentiles((np.array([]), np.array([45.0])), percentile=85.0)
    assert math.isnan(ps.l1)
    assert ps.l5 == pytest.approx(45.0)


def test_sigma_to_weight_round_trip():
    assert sigma_to_weight(0.5) == pytest.approx(1.0 / 0.25)
    assert sigma_to_weight(float("nan")) == 0.0
    assert sigma_to_weight(0.0) == 0.0
    assert sigma_to_weight(-1.0) == 0.0


def test_sigtype_factor_out_of_range_returns_nan():
    assert math.isnan(sigtype_factor(-1))
    assert math.isnan(sigtype_factor(len(SIGTYPE_FACTOR)))


def test_relative_pr_weights_match_obserrmodel_layout():
    """Pin per-sigtype σ ratios at the percentile to obserrmodel.m's
    ``P_sn_ratio * sn_base * sigfactor`` formula."""
    expected = {
        SIGTYPE_GPS_L1: 0.8,
        SIGTYPE_GLO_G1: 1.5,
        SIGTYPE_GAL_E1: 0.8,
        SIGTYPE_BDS_B1: 0.8,
        SIGTYPE_GPS_L5: 0.5,
        SIGTYPE_GAL_E5: 0.5,
        SIGTYPE_BDS_B2A: 0.5,
    }
    for sigtype, expected_sigma in expected.items():
        actual = compute_pr_sigma_sn(45.0, 45.0, sigtype)
        assert actual == pytest.approx(P_SN_RATIO_DEFAULT * 1.0 * expected_sigma)
