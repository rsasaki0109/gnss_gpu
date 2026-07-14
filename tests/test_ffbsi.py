"""Tests for FFBSi index sampling (numpy / no CUDA required)."""

import numpy as np
import pytest

from gnss_gpu.particle_ffbsi import (
    ffbsi_sample_indices,
    genealogy_smooth_indices,
    transition_logpdf,
)


def test_transition_logpdf_shape():
    N = 7
    x_next = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float64)
    x_t = np.random.randn(N, 4).astype(np.float64)
    vel = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    lf = transition_logpdf(x_next, x_t, vel, 0.5, sigma_pos=1.2, sigma_cb=30.0)
    assert lf.shape == (N,)
    assert np.all(np.isfinite(lf))


def test_transition_logpdf_accepts_per_particle_velocity():
    x_t = np.zeros((3, 4), dtype=np.float64)
    velocities = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    values = transition_logpdf(
        np.array([2.0, 0.0, 0.0, 0.0]),
        x_t,
        velocities,
        1.0,
        sigma_pos=0.1,
        sigma_cb=10.0,
    )
    assert int(np.argmax(values)) == 2


def test_genealogy_smooth_indices_trace():
    rng = np.random.default_rng(1)
    T, N = 4, 16
    lw = rng.standard_normal((T, N)).astype(np.float64) * 2.0
    anc = np.zeros((T, N), dtype=np.int64)
    for t in range(T - 1):
        anc[t] = rng.integers(0, N, size=N)
    anc[T - 1] = np.arange(N)
    indices = genealogy_smooth_indices(lw, anc, rng)
    assert indices.shape == (T,)
    assert np.all(indices >= 0) and np.all(indices < N)
    for t in range(T - 2, -1, -1):
        assert indices[t] == anc[t, indices[t + 1]]


def test_genealogy_terminal_mask_conditions_terminal_lineage():
    rng = np.random.default_rng(2)
    T, N = 3, 10
    lw = np.zeros((T, N), dtype=np.float64)
    anc = np.tile(np.arange(N, dtype=np.int64), (T, 1))
    terminal_mask = np.arange(N) >= 7
    for _ in range(20):
        indices = genealogy_smooth_indices(
            lw, anc, rng, terminal_mask=terminal_mask
        )
        assert indices[-1] >= 7
        assert np.all(indices == indices[-1])


def test_genealogy_terminal_mask_validates_length():
    with pytest.raises(ValueError):
        genealogy_smooth_indices(
            np.zeros((2, 4)),
            np.tile(np.arange(4), (2, 1)),
            np.random.default_rng(0),
            terminal_mask=np.ones(3, dtype=bool),
        )


def test_ffbsi_indices_length_and_bounds():
    rng = np.random.default_rng(0)
    T, N = 5, 32
    X = rng.standard_normal((T, N, 4)).astype(np.float64)
    log_weights = rng.standard_normal((T, N)).astype(np.float64) * 2.0
    V = rng.standard_normal((T, 3)).astype(np.float64) * 0.01
    dt = np.full(T, 0.1, dtype=np.float64)
    sig_pos = np.full(T, 1.0, dtype=np.float64)
    indices = ffbsi_sample_indices(
        log_weights, X, V, dt, sig_pos, sigma_cb=10.0, rng=rng
    )
    assert indices.shape == (T,)
    assert np.all(indices >= 0) and np.all(indices < N)


def test_marginal_ffbsi_terminal_mask_and_particle_velocities():
    rng = np.random.default_rng(10)
    T, N = 3, 12
    X = np.zeros((T, N, 4), dtype=np.float64)
    X[:, N // 2 :, 0] = 20.0
    velocities = np.zeros((T, N, 3), dtype=np.float64)
    mask = np.arange(N) >= N // 2
    indices = ffbsi_sample_indices(
        np.zeros((T, N)),
        X,
        velocities,
        np.ones(T),
        np.ones(T),
        10.0,
        rng,
        terminal_mask=mask,
    )
    assert np.all(indices >= N // 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
