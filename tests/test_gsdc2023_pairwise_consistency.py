"""Tests for ``experiments.gsdc2023_pairwise_consistency``.

Synthetic per-epoch scenarios: one obvious outlier among inliers, all
inliers, and a degenerate few-sat case where the filter should revert.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.gsdc2023_pairwise_consistency import (
    MAD_THRESHOLD_DEFAULT,
    MIN_OBS_AFTER_FILTER_DEFAULT,
    _sagnac_geometric_range,
    apply_pairwise_consistency_pre_filter,
)


def _synthetic_epoch(rx, pseudoranges_extra=None, n_sat=8, residual_extra=None):
    """Build a (1, n_sat, 3) sat ECEF + (1, n_sat) pr/weights for a single rx.

    Pseudorange equals the Sagnac-corrected geometric range so the
    inlier residual is zero exactly.
    """
    rng = np.random.default_rng(0)
    sat = np.zeros((1, n_sat, 3))
    pr = np.zeros((1, n_sat))
    weights = np.ones((1, n_sat))
    for i in range(n_sat):
        sat[0, i] = rx + rng.normal(scale=2e7, size=3)
        pr[0, i] = _sagnac_geometric_range(rx, sat[0, i])
    if residual_extra is not None:
        for sat_idx, extra in residual_extra.items():
            pr[0, sat_idx] += extra
    return sat, pr, weights


def test_filter_passes_through_all_inliers():
    rx = np.array([1.0e6, -4.0e6, 4.0e6])
    sat, pr, w = _synthetic_epoch(rx)
    new_w, stats = apply_pairwise_consistency_pre_filter(
        sat, pr, w, reference_xyz=rx[None, :], sys_kind=None, n_clock=1,
        mad_threshold_m=MAD_THRESHOLD_DEFAULT,
    )
    np.testing.assert_allclose(new_w[0], 1.0)
    assert stats.obs_masked == 0
    assert stats.epochs_filtered == 0


def test_filter_masks_clear_outlier():
    rx = np.array([1.0e6, -4.0e6, 4.0e6])
    sat, pr, w = _synthetic_epoch(rx, residual_extra={3: 60.0})  # 60 m bias on sat 3
    new_w, stats = apply_pairwise_consistency_pre_filter(
        sat, pr, w, reference_xyz=rx[None, :], sys_kind=None, n_clock=1,
        mad_threshold_m=MAD_THRESHOLD_DEFAULT,
    )
    assert new_w[0, 3] == 0.0  # outlier masked
    assert (new_w[0, [0, 1, 2, 4, 5, 6, 7]] == 1.0).all()  # inliers preserved
    assert stats.obs_masked == 1
    assert stats.epochs_filtered == 1


def test_filter_reverts_when_too_few_inliers():
    rx = np.array([1.0e6, -4.0e6, 4.0e6])
    sat, pr, w = _synthetic_epoch(rx, n_sat=5, residual_extra={0: 100.0, 1: 100.0})
    new_w, stats = apply_pairwise_consistency_pre_filter(
        sat, pr, w, reference_xyz=rx[None, :], sys_kind=None, n_clock=1,
        mad_threshold_m=MAD_THRESHOLD_DEFAULT,
        min_obs_after_filter=MIN_OBS_AFTER_FILTER_DEFAULT,
    )
    # Only 3 inliers remain (5 - 2 outliers), below min=5 -> revert.
    np.testing.assert_allclose(new_w[0], w[0])
    assert stats.obs_masked == 0


def test_filter_handles_nan_reference_xyz_by_skipping():
    rx = np.array([1.0e6, -4.0e6, 4.0e6])
    sat, pr, w = _synthetic_epoch(rx, residual_extra={3: 60.0})
    bad_ref = np.array([[np.nan, np.nan, np.nan]])
    new_w, stats = apply_pairwise_consistency_pre_filter(
        sat, pr, w, reference_xyz=bad_ref, sys_kind=None, n_clock=1,
    )
    # No filtering done because ref is NaN.
    np.testing.assert_allclose(new_w, w)
    assert stats.epochs_filtered == 0


def test_filter_uses_per_system_clock_baseline():
    """Two clock groups with different baseline biases -- outlier detection
    must subtract per-group medians, not a global median."""
    rx = np.array([1.0e6, -4.0e6, 4.0e6])
    sat, pr, w = _synthetic_epoch(rx, n_sat=10)
    # Group 0 has +20 m bias on all obs, group 1 has -30 m bias on all obs.
    # No real outliers in either group.
    sys_kind = np.zeros((1, 10), dtype=np.int32)
    sys_kind[0, :5] = 0
    sys_kind[0, 5:] = 1
    pr[0, :5] += 20.0
    pr[0, 5:] -= 30.0
    new_w, stats = apply_pairwise_consistency_pre_filter(
        sat, pr, w, reference_xyz=rx[None, :], sys_kind=sys_kind, n_clock=2,
        mad_threshold_m=MAD_THRESHOLD_DEFAULT,
    )
    # All survive because per-group median absorbs the clock bias.
    assert stats.obs_masked == 0
    np.testing.assert_allclose(new_w, w)
