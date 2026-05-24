"""Tests for ``experiments.gsdc2023_max_clique_filter``.

Synthetic per-epoch scenarios verifying that the max-clique consensus
filter correctly handles:

  * a single-bias outlier among inliers (must be masked)
  * all inliers (no observations dropped)
  * two-cluster bias pattern where MAD-around-median can fail but
    max-clique still picks the larger consistent set
  * a degenerate few-sat case where the filter should revert
"""
from __future__ import annotations

import numpy as np

from experiments.gsdc2023_max_clique_filter import (
    MIN_CLIQUE_SIZE_DEFAULT,
    PAIR_THRESHOLD_DEFAULT_M,
    _greedy_max_clique,
    _sagnac_geometric_range,
    apply_max_clique_consensus_filter,
)


def _synthetic_epoch(rx, residual_extra=None, n_sat=8):
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


def test_greedy_clique_simple_complete_graph():
    adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    clique = _greedy_max_clique(adj)
    assert clique.tolist() == [0, 1, 2]


def test_greedy_clique_picks_larger_cluster():
    """A 5-clique and a 3-clique connected by no edges; expect the 5-clique."""
    adj = np.zeros((8, 8), dtype=np.uint8)
    for i in range(5):
        for j in range(5):
            if i != j:
                adj[i, j] = 1
    for i in range(5, 8):
        for j in range(5, 8):
            if i != j:
                adj[i, j] = 1
    clique = _greedy_max_clique(adj)
    assert clique.size == 5
    assert set(clique.tolist()) == {0, 1, 2, 3, 4}


def test_filter_passes_through_all_inliers():
    rx = np.array([1.0e6, -4.0e6, 4.0e6])
    sat, pr, w = _synthetic_epoch(rx)
    new_w, stats = apply_max_clique_consensus_filter(
        sat, pr, w, reference_xyz=rx[None, :], sys_kind=None, n_clock=1,
        pair_threshold_m=PAIR_THRESHOLD_DEFAULT_M,
    )
    np.testing.assert_allclose(new_w[0], 1.0)
    assert stats.obs_masked == 0


def test_filter_masks_single_bias_outlier():
    rx = np.array([1.0e6, -4.0e6, 4.0e6])
    sat, pr, w = _synthetic_epoch(rx, residual_extra={3: 60.0}, n_sat=8)
    new_w, stats = apply_max_clique_consensus_filter(
        sat, pr, w, reference_xyz=rx[None, :], sys_kind=None, n_clock=1,
        pair_threshold_m=PAIR_THRESHOLD_DEFAULT_M,
    )
    assert new_w[0, 3] == 0.0
    assert (new_w[0, [0, 1, 2, 4, 5, 6, 7]] == 1.0).all()
    assert stats.obs_masked == 1


def test_filter_picks_larger_cluster_under_two_bias_split():
    """5 obs with bias +0, 3 obs with bias +40. max-clique should keep the
    5-cluster while a pure median-MAD method could be split.
    """
    rx = np.array([1.0e6, -4.0e6, 4.0e6])
    sat, pr, w = _synthetic_epoch(rx, n_sat=10)
    # First 7 obs are inliers (0 m), last 3 are biased +40 m.
    for s in (7, 8, 9):
        pr[0, s] += 40.0
    new_w, stats = apply_max_clique_consensus_filter(
        sat, pr, w, reference_xyz=rx[None, :], sys_kind=None, n_clock=1,
        pair_threshold_m=PAIR_THRESHOLD_DEFAULT_M,
        min_clique_size=5,
    )
    # The 7-clique should be kept, 3 outliers masked.
    assert stats.obs_masked == 3
    assert (new_w[0, :7] == 1.0).all()
    assert (new_w[0, 7:] == 0.0).all()


def test_filter_reverts_when_clique_too_small():
    rx = np.array([1.0e6, -4.0e6, 4.0e6])
    # Only 4 obs total; min_clique_size=5 -> revert.
    sat, pr, w = _synthetic_epoch(rx, n_sat=4)
    new_w, stats = apply_max_clique_consensus_filter(
        sat, pr, w, reference_xyz=rx[None, :], sys_kind=None, n_clock=1,
        min_clique_size=5,
    )
    np.testing.assert_allclose(new_w, w)
    assert stats.obs_masked == 0


def test_filter_handles_nan_reference_xyz_by_skipping():
    rx = np.array([1.0e6, -4.0e6, 4.0e6])
    sat, pr, w = _synthetic_epoch(rx, residual_extra={3: 60.0})
    bad_ref = np.array([[np.nan, np.nan, np.nan]])
    new_w, stats = apply_max_clique_consensus_filter(
        sat, pr, w, reference_xyz=bad_ref, sys_kind=None, n_clock=1,
    )
    np.testing.assert_allclose(new_w, w)
    assert stats.epochs_filtered == 0


def test_filter_per_clock_group_isolation():
    """Two clock groups with independent bias baselines; outlier in group 0
    must not affect group 1.
    """
    rx = np.array([1.0e6, -4.0e6, 4.0e6])
    sat, pr, w = _synthetic_epoch(rx, n_sat=14)
    sys_kind = np.zeros((1, 14), dtype=np.int32)
    sys_kind[0, :7] = 0
    sys_kind[0, 7:] = 1
    # Group 0 has +25 m bias on all obs, group 1 has -15 m bias.
    pr[0, :7] += 25.0
    pr[0, 7:] -= 15.0
    # One real outlier in group 0 at index 2 with extra +50 m.
    pr[0, 2] += 50.0
    new_w, stats = apply_max_clique_consensus_filter(
        sat, pr, w, reference_xyz=rx[None, :], sys_kind=sys_kind, n_clock=2,
        pair_threshold_m=PAIR_THRESHOLD_DEFAULT_M, min_clique_size=5,
    )
    # Index 2 should be masked; group 1 untouched.
    assert new_w[0, 2] == 0.0
    assert (new_w[0, 7:] == 1.0).all()
