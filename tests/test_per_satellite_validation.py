"""Tests for the per-satellite/per-epoch validation metrics.

Pure-numpy, no GPU / network / data dependency: exercises
:mod:`gnss_gpu.validation.per_satellite` on small synthetic arrays whose
correlation/sign/gain structure is known by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from gnss_gpu.validation.per_satellite import (
    correction_gain,
    evaluate_predictions,
    pearson_correlation,
    per_satellite_table,
    sign_agreement_rate,
    spearman_correlation,
)


def test_pearson_perfect_prediction():
    measured = np.array([1.0, 3.0, -2.0, 5.0, 0.5])
    predicted = measured.copy()
    assert pearson_correlation(predicted, measured) == pytest.approx(1.0, abs=1e-9)


def test_spearman_perfect_prediction():
    measured = np.array([1.0, 3.0, -2.0, 5.0, 0.5])
    predicted = 2.0 * measured + 1.0  # monotone, not identical -> Spearman still 1
    assert spearman_correlation(predicted, measured) == pytest.approx(1.0, abs=1e-9)


def test_pearson_anti_correlated():
    measured = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = -measured
    assert pearson_correlation(predicted, measured) == pytest.approx(-1.0, abs=1e-9)


def test_correction_gain_perfect_prediction_zeroes_residual():
    measured = np.array([10.0, -8.0, 6.0, -4.0, 2.0])
    predicted = measured.copy()
    g = correction_gain(predicted, measured)
    assert g["rms_corrected_m"] == pytest.approx(0.0, abs=1e-9)
    assert g["gain_m"] == pytest.approx(g["rms_raw_m"], abs=1e-9)
    assert g["gain_pct"] == pytest.approx(100.0, abs=1e-6)


def test_correction_gain_zero_prediction_is_zero_gain():
    measured = np.array([10.0, -8.0, 6.0, -4.0, 2.0])
    predicted = np.zeros_like(measured)
    g = correction_gain(predicted, measured)
    assert g["rms_corrected_m"] == pytest.approx(g["rms_raw_m"], abs=1e-9)
    assert g["gain_m"] == pytest.approx(0.0, abs=1e-9)
    assert g["gain_pct"] == pytest.approx(0.0, abs=1e-9)


def test_correction_gain_bad_prediction_is_negative_gain():
    # A prediction with the wrong sign makes things worse, not better.
    measured = np.array([10.0, -8.0, 6.0, -4.0, 2.0])
    predicted = -2.0 * measured
    g = correction_gain(predicted, measured)
    assert g["gain_m"] < 0.0
    assert g["gain_pct"] < 0.0


def test_sign_agreement_perfect():
    measured = np.array([5.0, -6.0, 7.0, -8.0])
    predicted = np.array([2.0, -3.0, 1.0, -1.0])
    is_nlos = np.ones(measured.shape, dtype=bool)
    s = sign_agreement_rate(predicted, measured, is_nlos=is_nlos, threshold_m=1.0)
    assert s["n"] == 4
    assert s["rate"] == pytest.approx(1.0)


def test_sign_agreement_anti_correlated_is_near_zero():
    measured = np.array([5.0, -6.0, 7.0, -8.0, 9.0, -10.0])
    predicted = -measured  # always the wrong sign
    is_nlos = np.ones(measured.shape, dtype=bool)
    s = sign_agreement_rate(predicted, measured, is_nlos=is_nlos, threshold_m=1.0)
    assert s["n"] == 6
    assert s["rate"] == pytest.approx(0.0)


def test_sign_agreement_respects_threshold_and_nlos_mask():
    measured = np.array([0.5, -0.5, 5.0, -5.0])  # first two below threshold
    predicted = np.array([1.0, -1.0, 1.0, -1.0])
    is_nlos = np.array([True, True, True, False])  # last one masked out (not NLOS)
    s = sign_agreement_rate(predicted, measured, is_nlos=is_nlos, threshold_m=1.0)
    # Only index 2 qualifies: |measured|>1 and is_nlos.
    assert s["n"] == 1
    assert s["rate"] == pytest.approx(1.0)


def test_sign_agreement_no_qualifying_samples_is_nan():
    measured = np.array([0.1, -0.2, 0.3])
    predicted = np.array([1.0, -1.0, 1.0])
    is_nlos = np.ones(measured.shape, dtype=bool)
    s = sign_agreement_rate(predicted, measured, is_nlos=is_nlos, threshold_m=1.0)
    assert s["n"] == 0
    assert np.isnan(s["rate"])


def test_nan_pairs_are_dropped_pairwise():
    measured = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    predicted = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    p = pearson_correlation(predicted, measured)
    # Only indices 0, 3, 4 are jointly finite; identical there -> corr 1.
    assert p == pytest.approx(1.0, abs=1e-9)
    g = correction_gain(predicted, measured)
    assert g["n"] == 3


def test_evaluate_predictions_perfect_and_zero_models():
    rng = np.random.default_rng(0)
    n = 40
    measured = rng.normal(scale=8.0, size=n)
    is_nlos = np.zeros(n, dtype=bool)
    is_nlos[: n // 2] = True
    sat_ids = np.array([f"G{(i % 6) + 1:02d}" for i in range(n)])

    perfect = evaluate_predictions(
        measured.copy(), measured, sat_ids=sat_ids, is_nlos=is_nlos,
        nlos_threshold_m=0.5)
    assert perfect["pearson_all"] == pytest.approx(1.0, abs=1e-9)
    assert perfect["spearman_all"] == pytest.approx(1.0, abs=1e-9)
    assert perfect["pearson_nlos"] == pytest.approx(1.0, abs=1e-9)
    assert perfect["sign_agreement_nlos"] == pytest.approx(1.0, abs=1e-6)
    assert perfect["correction_gain_m"] > 0.0
    assert perfect["correction_gain_pct"] == pytest.approx(100.0, abs=1e-6)

    zero_model = evaluate_predictions(
        np.zeros(n), measured, sat_ids=sat_ids, is_nlos=is_nlos,
        nlos_threshold_m=0.5)
    assert zero_model["correction_gain_m"] == pytest.approx(0.0, abs=1e-9)
    assert zero_model["correction_gain_pct"] == pytest.approx(0.0, abs=1e-9)
    # An always-zero prediction has zero variance -> correlation undefined.
    assert np.isnan(zero_model["pearson_all"])


def test_evaluate_predictions_anti_correlated_has_low_sign_agreement():
    rng = np.random.default_rng(1)
    n = 30
    measured = rng.normal(scale=10.0, size=n)
    predicted = -measured
    is_nlos = np.ones(n, dtype=bool)

    res = evaluate_predictions(
        predicted, measured, is_nlos=is_nlos, nlos_threshold_m=0.5)
    assert res["pearson_all"] == pytest.approx(-1.0, abs=1e-6)
    assert res["sign_agreement_nlos"] == pytest.approx(0.0, abs=1e-6)
    # Doubling-and-flipping the error can only inflate RMS -> negative gain.
    assert res["correction_gain_m"] < 0.0


def test_per_satellite_table_shape_and_content():
    sat_ids = np.array(["G01", "G01", "G02", "G02", "G02"])
    measured = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = measured.copy()
    is_nlos = np.array([True, True, False, True, True])

    rows = per_satellite_table(sat_ids, predicted, measured, is_nlos=is_nlos,
                                threshold_m=0.5)
    assert len(rows) == 2  # one row per unique satellite id
    by_id = {r["sat_id"]: r for r in rows}
    assert set(by_id) == {"G01", "G02"}
    assert by_id["G01"]["n"] == 2
    assert by_id["G02"]["n"] == 3
    assert by_id["G02"]["n_nlos"] == 2
    for r in rows:
        assert r["pearson"] == pytest.approx(1.0, abs=1e-9)
        assert r["gain_pct"] == pytest.approx(100.0, abs=1e-6)
        for key in ("n", "n_nlos", "pearson", "spearman", "sign_agreement",
                    "sign_agreement_n", "rms_raw_m", "rms_corrected_m",
                    "gain_m", "gain_pct", "sat_id"):
            assert key in r


def test_per_satellite_table_empty_is_empty_list():
    assert per_satellite_table(np.array([]), np.array([]), np.array([])) == []


def test_evaluate_predictions_without_optional_args():
    measured = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 3.0])
    res = evaluate_predictions(predicted, measured)
    assert "per_satellite" not in res
    assert res["n_nlos"] == 0
    assert np.isnan(res["pearson_nlos"])
