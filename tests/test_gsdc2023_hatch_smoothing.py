"""Tests for ``experiments.gsdc2023_hatch_smoothing``.

Synthetic single-satellite arcs verify the Hatch recursion and arc reset
on cycle-slip / NaN gaps.
"""
from __future__ import annotations

import numpy as np

from experiments.gsdc2023_hatch_smoothing import (
    ADR_STATE_CYCLE_SLIP,
    ADR_STATE_VALID,
    HATCH_SMOOTHING_N_DEFAULT,
    apply_hatch_smoothing,
)


def _build_arc(n_epochs: int, true_range: float, pr_noise_sigma: float, seed: int = 0):
    """Return (pseudorange, adr, state) for a stationary single sat arc.

    True range is constant; pseudorange has white noise; ADR is perfect
    (no cycle slip).  This is the textbook Hatch test: smoothing N steps
    reduces noise variance by 1/N.
    """
    rng = np.random.default_rng(seed)
    pr = np.full((n_epochs, 1), true_range) + rng.normal(scale=pr_noise_sigma, size=(n_epochs, 1))
    adr = np.full((n_epochs, 1), true_range)  # noise-free carrier
    state = np.full((n_epochs, 1), ADR_STATE_VALID, dtype=np.int32)
    return pr, adr, state


def test_smoothing_reduces_noise_on_clean_arc():
    pr, adr, state = _build_arc(n_epochs=400, true_range=2.0e7, pr_noise_sigma=3.0, seed=0)
    smoothed, stats = apply_hatch_smoothing(pr, adr, state, smoothing_n=100)
    raw_std = float(np.std(pr[:, 0]))
    smoothed_std = float(np.std(smoothed[100:, 0]))  # after window warmup
    # With N=100, theoretical reduction = sqrt(1/100) = 0.1; allow 4x slack.
    assert smoothed_std < 0.4 * raw_std, (raw_std, smoothed_std)
    assert stats.arcs_total == 1
    assert stats.obs_smoothed == 399  # first obs of arc is not smoothed
    assert stats.mean_arc_length == 400.0


def test_smoothing_preserves_mean_on_clean_arc():
    pr, adr, state = _build_arc(n_epochs=400, true_range=2.0e7, pr_noise_sigma=5.0, seed=1)
    smoothed, _ = apply_hatch_smoothing(pr, adr, state, smoothing_n=100)
    # The mean should converge to the true range; tolerate up to 3x raw stderr.
    raw_mean_err = abs(float(np.mean(pr[:, 0])) - 2.0e7)
    smoothed_mean_err = abs(float(np.mean(smoothed[100:, 0])) - 2.0e7)
    assert smoothed_mean_err <= 3.0 * raw_mean_err + 0.5


def test_cycle_slip_resets_arc():
    pr, adr, state = _build_arc(n_epochs=50, true_range=1.5e7, pr_noise_sigma=2.0, seed=2)
    # Insert cycle slip flag at epoch 25 -> new arc starts at epoch 26.
    state[25, 0] = ADR_STATE_VALID | ADR_STATE_CYCLE_SLIP
    smoothed, stats = apply_hatch_smoothing(pr, adr, state, smoothing_n=100)
    # Epoch 25 is masked out (state has CYCLE_SLIP) -> two arcs.
    assert stats.arcs_total == 2
    # At new arc start (epoch 26), smoothed = raw.
    assert smoothed[26, 0] == pr[26, 0]


def test_nan_gap_breaks_arc():
    pr, adr, state = _build_arc(n_epochs=30, true_range=1.0e7, pr_noise_sigma=1.0, seed=3)
    # Gap from epoch 10 to 12 inclusive.
    pr[10:13, 0] = np.nan
    smoothed, stats = apply_hatch_smoothing(pr, adr, state, smoothing_n=50)
    assert stats.arcs_total == 2
    # Epoch 13 (= new arc start) should equal raw.
    assert smoothed[13, 0] == pr[13, 0]
    # Epoch 10 (NaN PR) stays NaN.
    assert np.isnan(smoothed[10, 0])


def test_no_adr_returns_identity():
    pr = np.array([[1.0, 2.0], [3.0, 4.0]])
    smoothed, stats = apply_hatch_smoothing(pr, None, None)
    np.testing.assert_array_equal(smoothed, pr)
    assert stats.obs_smoothed == 0


def test_invalid_adr_state_breaks_arc():
    pr, adr, state = _build_arc(n_epochs=20, true_range=1.0e7, pr_noise_sigma=1.0, seed=4)
    # Drop VALID bit at epochs 5-7.
    state[5:8, 0] = 0
    smoothed, stats = apply_hatch_smoothing(pr, adr, state, smoothing_n=50)
    assert stats.arcs_total == 2
    # Epoch 8 starts new arc.
    assert smoothed[8, 0] == pr[8, 0]


def test_default_n_constant():
    assert HATCH_SMOOTHING_N_DEFAULT == 100
