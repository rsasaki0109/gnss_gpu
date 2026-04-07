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


def test_genealogy_smooth_indices_trace():
    rng = np.random.default_rng(1)
    T, N = 4, 16
    lw = rng.standard_normal((T, N)).astype(np.float64) * 2.0
    anc = np.zeros((T, N), dtype=np.int64)
    for t in range(T - 1):
        anc[t] = rng.integers(0, N, size=N)
    anc[T - 1] = np.arange(N)
    I = genealogy_smooth_indices(lw, anc, rng)
    assert I.shape == (T,)
    assert np.all(I >= 0) and np.all(I < N)
    for t in range(T - 2, -1, -1):
        assert I[t] == anc[t, I[t + 1]]


def test_ffbsi_indices_length_and_bounds():
    rng = np.random.default_rng(0)
    T, N = 5, 32
    X = rng.standard_normal((T, N, 4)).astype(np.float64)
    log_weights = rng.standard_normal((T, N)).astype(np.float64) * 2.0
    V = rng.standard_normal((T, 3)).astype(np.float64) * 0.01
    dt = np.full(T, 0.1, dtype=np.float64)
    sig_pos = np.full(T, 1.0, dtype=np.float64)
    I = ffbsi_sample_indices(log_weights, X, V, dt, sig_pos, sigma_cb=10.0, rng=rng)
    assert I.shape == (T,)
    assert np.all(I >= 0) and np.all(I < N)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
