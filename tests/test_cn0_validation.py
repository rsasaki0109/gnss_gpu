"""Tests for gnss_gpu.validation.cn0_validation (no GPU/network/data)."""

import numpy as np
import pytest

from gnss_gpu.validation.cn0_validation import (
    attenuation_deficit_correlation,
    baseline_at_elevation,
    cn0_deficit,
    cn0_los_nlos_separation,
    elevation_binned_los_baseline,
)


# ---------------------------------------------------------------------------
# cn0_los_nlos_separation
# ---------------------------------------------------------------------------

def test_separation_perfectly_separable():
    rng = np.random.default_rng(0)
    los_cn0 = 45.0 + rng.normal(0.0, 0.5, 200)
    nlos_cn0 = 25.0 + rng.normal(0.0, 0.5, 200)
    cn0 = np.concatenate([los_cn0, nlos_cn0])
    is_los = np.concatenate([np.ones(200, bool), np.zeros(200, bool)])

    out = cn0_los_nlos_separation(cn0, is_los)
    assert out["n_los"] == 200
    assert out["n_nlos"] == 200
    assert out["mean_gap_dbhz"] > 15.0
    assert out["auc"] == pytest.approx(1.0, abs=1e-6)


def test_separation_inseparable_distributions():
    rng = np.random.default_rng(1)
    a = rng.normal(35.0, 3.0, 300)
    b = rng.normal(35.0, 3.0, 300)
    cn0 = np.concatenate([a, b])
    is_los = np.concatenate([np.ones(300, bool), np.zeros(300, bool)])

    out = cn0_los_nlos_separation(cn0, is_los)
    assert out["auc"] == pytest.approx(0.5, abs=0.1)


def test_separation_drops_nonfinite():
    cn0 = np.array([40.0, np.nan, 20.0, np.inf, 38.0, 22.0])
    is_los = np.array([True, True, False, False, True, False])
    out = cn0_los_nlos_separation(cn0, is_los)
    assert out["n_los"] == 2
    assert out["n_nlos"] == 2


def test_separation_length_mismatch_raises():
    with pytest.raises(ValueError):
        cn0_los_nlos_separation([1.0, 2.0], [True, False, True])


def test_separation_empty_class_gives_nan():
    out = cn0_los_nlos_separation([40.0, 41.0], [True, True])
    assert out["n_nlos"] == 0
    assert np.isnan(out["auc"])
    assert np.isnan(out["mean_nlos_dbhz"])


def test_separation_empty_input():
    out = cn0_los_nlos_separation(np.array([]), np.array([], dtype=bool))
    assert out["n_los"] == 0
    assert out["n_nlos"] == 0
    assert np.isnan(out["auc"])


# ---------------------------------------------------------------------------
# elevation_binned_los_baseline / baseline_at_elevation
# ---------------------------------------------------------------------------

def test_baseline_synthetic_linear_trend():
    # Clear-sky C/N0 rises linearly with elevation; only LOS samples feed the
    # baseline, and it should recover that trend per bin.
    rng = np.random.default_rng(2)
    elev = rng.uniform(0.0, 90.0, 2000)
    cn0_clear = 30.0 + 0.15 * elev
    is_los = rng.uniform(0, 1, 2000) > 0.3
    # Give NLOS samples an obviously different (lower, noisy) level so a bug
    # that accidentally includes them would be visible.
    cn0 = np.where(is_los, cn0_clear, cn0_clear - 15.0)

    baseline = elevation_binned_los_baseline(elev, cn0, is_los)
    edges = baseline["bin_edges_deg"]
    assert edges[0] == 0.0 and edges[-1] == 90.0
    mid = 0.5 * (edges[:-1] + edges[1:])
    expected = 30.0 + 0.15 * mid
    finite = np.isfinite(baseline["median_cn0_dbhz"])
    assert finite.all()
    np.testing.assert_allclose(baseline["median_cn0_dbhz"][finite], expected[finite], atol=1.5)


def test_baseline_empty_bin_is_nan():
    elev = np.array([5.0, 6.0, 85.0, 86.0])
    cn0 = np.array([30.0, 31.0, 48.0, 49.0])
    is_los = np.array([True, True, True, True])
    baseline = elevation_binned_los_baseline(elev, cn0, is_los, bin_edges_deg=[0, 10, 40, 70, 90])
    # bins: [0,10) has data, [10,40) empty, [40,70) empty, [70,90] has data
    assert np.isfinite(baseline["median_cn0_dbhz"][0])
    assert np.isnan(baseline["median_cn0_dbhz"][1])
    assert np.isnan(baseline["median_cn0_dbhz"][2])
    assert np.isfinite(baseline["median_cn0_dbhz"][3])
    assert baseline["count"][1] == 0


def test_baseline_ignores_nlos_and_nonfinite():
    elev = np.array([10.0, 10.0, 10.0])
    cn0 = np.array([40.0, np.nan, 20.0])
    is_los = np.array([True, True, False])
    baseline = elevation_binned_los_baseline(elev, cn0, is_los, bin_edges_deg=[0, 20])
    assert baseline["count"][0] == 1
    assert baseline["median_cn0_dbhz"][0] == pytest.approx(40.0)


def test_baseline_rejects_bad_edges():
    with pytest.raises(ValueError):
        elevation_binned_los_baseline([10.0], [40.0], [True], bin_edges_deg=[0.0])
    with pytest.raises(ValueError):
        elevation_binned_los_baseline([10.0], [40.0], [True], bin_edges_deg=[10.0, 0.0])


def test_baseline_at_elevation_right_edge_closed():
    baseline = elevation_binned_los_baseline(
        [5.0, 85.0], [30.0, 48.0], [True, True], bin_edges_deg=[0, 10, 90])
    looked_up = baseline_at_elevation(baseline, np.array([5.0, 90.0, -5.0, 150.0]))
    assert looked_up[0] == pytest.approx(30.0)
    assert looked_up[1] == pytest.approx(48.0)  # elevation==90 falls in last bin
    assert np.isnan(looked_up[2])  # below range
    assert np.isnan(looked_up[3])  # above range


def test_cn0_deficit_matches_baseline_difference():
    baseline = elevation_binned_los_baseline(
        [5.0], [30.0], [True], bin_edges_deg=[0, 10])
    deficit = cn0_deficit([5.0], [22.0], baseline)
    assert deficit[0] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# attenuation_deficit_correlation
# ---------------------------------------------------------------------------

def test_correlation_perfect_linear():
    atten = np.linspace(1.0, 20.0, 50)
    deficit = 2.0 * atten + 1.0
    out = attenuation_deficit_correlation(atten, deficit)
    assert out["n"] == 50
    assert out["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    assert out["spearman_r"] == pytest.approx(1.0, abs=1e-6)


def test_correlation_linear_with_noise_is_strong_positive():
    rng = np.random.default_rng(3)
    atten = rng.uniform(0.0, 25.0, 400)
    deficit = 1.5 * atten + rng.normal(0.0, 2.0, 400)
    out = attenuation_deficit_correlation(atten, deficit)
    assert out["pearson_r"] > 0.8
    assert out["spearman_r"] > 0.8


def test_correlation_uncorrelated_near_zero():
    rng = np.random.default_rng(4)
    atten = rng.uniform(0.0, 25.0, 500)
    deficit = rng.normal(0.0, 5.0, 500)
    out = attenuation_deficit_correlation(atten, deficit)
    assert abs(out["pearson_r"]) < 0.2
    assert abs(out["spearman_r"]) < 0.2


def test_correlation_drops_nonfinite_pairs():
    atten = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    deficit = np.array([2.0, 4.0, 6.0, np.inf, 10.0])
    out = attenuation_deficit_correlation(atten, deficit)
    assert out["n"] == 3  # (1,2), (2,4), (5,10) are the only finite pairs
    assert out["pearson_r"] == pytest.approx(1.0, abs=1e-6)


def test_correlation_too_few_points_is_nan():
    out = attenuation_deficit_correlation([1.0], [2.0])
    assert out["n"] == 1
    assert np.isnan(out["pearson_r"])
    assert np.isnan(out["spearman_r"])

    out_empty = attenuation_deficit_correlation([], [])
    assert out_empty["n"] == 0
    assert np.isnan(out_empty["pearson_r"])


def test_correlation_constant_input_is_nan():
    atten = np.full(10, 5.0)
    deficit = np.linspace(0.0, 10.0, 10)
    out = attenuation_deficit_correlation(atten, deficit)
    assert np.isnan(out["pearson_r"])


def test_correlation_length_mismatch_raises():
    with pytest.raises(ValueError):
        attenuation_deficit_correlation([1.0, 2.0], [1.0])
