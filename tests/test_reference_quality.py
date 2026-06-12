"""Tests for the residual-reference quality guard."""

import numpy as np
import pytest

from gnss_gpu.validation.reference_quality import (
    auc_abs_residual_vs_nlos,
    format_reference_quality,
    residual_reference_quality,
)


def test_auc_perfect_separation():
    # LOS small, NLOS large -> AUC = 1.
    abs_res = np.array([1.0, 2.0, 3.0, 20.0, 30.0, 40.0])
    is_nlos = np.array([False, False, False, True, True, True])
    assert auc_abs_residual_vs_nlos(abs_res, is_nlos) == pytest.approx(1.0)


def test_auc_random_is_half():
    rng = np.random.default_rng(0)
    abs_res = rng.normal(20.0, 5.0, size=400)
    is_nlos = rng.random(400) < 0.4  # label independent of residual
    auc = auc_abs_residual_vs_nlos(abs_res, is_nlos)
    assert 0.4 < auc < 0.6  # ~0.5, uninformative


def test_auc_handles_ties():
    abs_res = np.array([5.0, 5.0, 5.0, 5.0])
    is_nlos = np.array([True, True, False, False])
    assert auc_abs_residual_vs_nlos(abs_res, is_nlos) == pytest.approx(0.5)


def test_auc_empty_class_is_nan():
    assert np.isnan(auc_abs_residual_vs_nlos([1.0, 2.0], [False, False]))


def test_clean_reference_flagged_clean():
    los = np.array([True] * 20 + [False] * 20)
    abs_res = np.concatenate([
        np.linspace(0.5, 3.0, 20),    # LOS small
        np.linspace(20.0, 60.0, 20),  # NLOS large
    ])
    q = residual_reference_quality(abs_res, los)
    assert q["is_clean_reference"] is True
    assert q["los_median_m"] < 10.0
    assert q["auc"] > 0.9


def test_contaminated_reference_flagged_dirty():
    # Both classes have large, overlapping residuals -> not a clean NLOS truth.
    rng = np.random.default_rng(1)
    los = np.array([True] * 200 + [False] * 100)
    abs_res = np.abs(np.concatenate([
        rng.normal(0.0, 40.0, 200),  # LOS already tens of metres
        rng.normal(0.0, 40.0, 100),  # NLOS indistinguishable
    ]))
    q = residual_reference_quality(abs_res, los)
    assert q["is_clean_reference"] is False
    assert q["los_median_m"] > 10.0
    assert "CONTAMINATED" in format_reference_quality(q)
