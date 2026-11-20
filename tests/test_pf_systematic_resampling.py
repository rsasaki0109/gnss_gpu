"""Systematic resampling must be unbiased: E[offspring_i] = N * w_i.

The offset u0 has to be uniform on [0, 1) before the kernel forms
(u0 + i) / N. Dividing u0 by N twice (the pre-2026-11 code) pinned the
pointers to i / N, so offspring counts became deterministic and biased.
The Python wrappers must also pass a fresh seed per resampling call.

Requires the compiled CUDA extensions; each block skips independently.
"""

import numpy as np
import pytest

try:
    from gnss_gpu import _gnss_gpu_pf_device as device_native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    HAS_DEVICE = True
except ImportError:
    HAS_DEVICE = False

try:
    from gnss_gpu import _gnss_gpu_pf as host_native
    from gnss_gpu.particle_filter import ParticleFilter
    HAS_HOST = True
except ImportError:
    HAS_HOST = False

needs_device = pytest.mark.skipif(not HAS_DEVICE, reason="CUDA pf_device module not available")
needs_host = pytest.mark.skipif(not HAS_HOST, reason="CUDA pf module not available")

# Weights whose N * w are far from integers, so a pinned offset is biased.
SMALL_WEIGHTS = np.array([0.1, 0.4, 0.2, 0.3])
SMALL_REPS = 4000
LARGE_N = 2048
LARGE_REPS = 200


def _large_log_weights():
    return np.random.default_rng(0).normal(0.0, 1.5, LARGE_N)


def _normalized(log_weights):
    w = np.exp(log_weights - log_weights.max())
    return w / w.sum()


def _device_offspring_mean(log_weights, reps):
    """Mean offspring count per particle over consecutive seeds 0..reps-1."""
    count = len(log_weights)
    pf = ParticleFilterDevice(n_particles=count, seed=1, resampling="systematic")
    pf.initialize([0.0, 0.0, 0.0], clock_bias=0.0, spread_pos=1.0, spread_cb=1.0)
    states = np.zeros((count, 16))
    states[:, 0] = np.arange(count)
    total = np.zeros(count)
    for seed in range(reps):
        pf.set_particle_states(states)
        pf.set_log_weights(log_weights)
        device_native.pf_device_resample_systematic(pf._state, seed)
        ancestors = pf.get_particle_states()[:, 0].astype(int)
        np.testing.assert_array_equal(pf.get_resample_ancestors(), ancestors)
        total += np.bincount(ancestors, minlength=count)
    return total / reps


def _host_offspring_mean(log_weights, reps):
    count = len(log_weights)
    total = np.zeros(count)
    for seed in range(reps):
        px = np.arange(count, dtype=np.float64)
        zeros = np.zeros(count)
        host_native.pf_resample_systematic(
            px, zeros.copy(), zeros.copy(), zeros.copy(),
            np.ascontiguousarray(log_weights, dtype=np.float64), count, seed)
        total += np.bincount(px.astype(int), minlength=count)
    return total / reps


@needs_device
def test_device_systematic_small_population_is_unbiased():
    mean = _device_offspring_mean(np.log(SMALL_WEIGHTS), SMALL_REPS)
    np.testing.assert_allclose(mean, len(SMALL_WEIGHTS) * SMALL_WEIGHTS, atol=0.05)


@needs_device
def test_device_systematic_large_population_matches_weights():
    lw = _large_log_weights()
    share = _device_offspring_mean(lw, LARGE_REPS) / LARGE_N
    assert 0.5 * np.abs(share - _normalized(lw)).sum() < 0.04


@needs_host
def test_host_systematic_small_population_is_unbiased():
    mean = _host_offspring_mean(np.log(SMALL_WEIGHTS), SMALL_REPS)
    np.testing.assert_allclose(mean, len(SMALL_WEIGHTS) * SMALL_WEIGHTS, atol=0.05)


@needs_host
def test_host_systematic_large_population_matches_weights():
    lw = _large_log_weights()
    share = _host_offspring_mean(lw, LARGE_REPS) / LARGE_N
    assert 0.5 * np.abs(share - _normalized(lw)).sum() < 0.04


@needs_device
def test_device_wrapper_advances_systematic_seed_per_call():
    pf = ParticleFilterDevice(n_particles=64, seed=5, resampling="systematic")
    pf.initialize([0.0, 0.0, 0.0], clock_bias=0.0, spread_pos=1.0, spread_cb=1.0)
    calls = []
    pf._pf_device_resample_systematic = lambda state, seed: calls.append(seed)
    pf._resample()
    pf._resample()
    pf.resample()
    assert calls == [5, 6, 7]
    pf.initialize([0.0, 0.0, 0.0], clock_bias=0.0, spread_pos=1.0, spread_cb=1.0)
    pf._resample()
    assert calls[-1] == 5


@needs_device
def test_device_wrapper_keeps_megopolis_seed_fixed():
    pf = ParticleFilterDevice(n_particles=64, seed=5)
    calls = []
    pf._pf_device_resample_megopolis_coalesced = lambda state, it, seed: calls.append(seed)
    pf._resample()
    pf._resample()
    assert calls == [5, 5]


@needs_host
def test_host_wrapper_advances_systematic_seed_per_call():
    pf = ParticleFilter(n_particles=64, seed=11, resampling="systematic")
    pf.initialize([0.0, 0.0, 0.0], clock_bias=0.0, spread_pos=1.0, spread_cb=1.0)
    calls = []
    pf._pf_resample_systematic = lambda *args: calls.append(args[-1])
    pf._resample()
    pf._resample()
    assert calls == [11, 12]
    pf.initialize([0.0, 0.0, 0.0], clock_bias=0.0, spread_pos=1.0, spread_cb=1.0)
    pf._resample()
    assert calls[-1] == 11
