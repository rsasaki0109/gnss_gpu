"""Statistical and contract tests for ParticleFilterDevice resampling modes.

Requires the compiled ``_gnss_gpu_pf_device`` extension (CUDA).
"""

import numpy as np
import pytest

try:
    from gnss_gpu import _gnss_gpu_pf_device as native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    from gnss_gpu.pf_device_config import clone_pf_device_init_kwargs
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")

N = 2048
REPS = 200


def _filter():
    pf = ParticleFilterDevice(n_particles=N, seed=1)
    pf.initialize([0.0, 0.0, 0.0], clock_bias=0.0, spread_pos=1.0, spread_cb=1.0)
    return pf


def _log_weights(sigma):
    return np.random.default_rng(0).normal(0.0, sigma, N)


def _mean_offspring_share(pf, log_weights, resample):
    """Average offspring count / N per original particle over REPS seeds."""
    states = np.zeros((N, 16))
    states[:, 0] = np.arange(N)  # tag each particle with its index
    seeds = np.random.default_rng(12345).integers(0, 2**31, REPS)
    counts = np.zeros(N)
    for seed in seeds:
        pf.set_particle_states(states)
        pf.set_log_weights(log_weights)
        resample(pf, int(seed))
        ancestors = pf.get_particle_states()[:, 0]
        assert np.array_equal(ancestors, np.rint(ancestors))
        counts += np.bincount(ancestors.astype(int), minlength=N)
    return counts / REPS / N


def _normalized(log_weights):
    w = np.exp(log_weights - log_weights.max())
    return w / w.sum()


def test_coalesced_megopolis_converges_to_weights():
    pf = _filter()
    lw = _log_weights(1.5)
    w = _normalized(lw)
    share = _mean_offspring_share(
        pf, lw, lambda f, s: native.pf_device_resample_megopolis_coalesced(f._state, 240, s)
    )
    top = np.argsort(w)[-20:]
    assert 0.5 * np.abs(share - w).sum() < 0.04
    assert abs(share[top].sum() / w[top].sum() - 1.0) < 0.05


def test_historical_megopolis_does_not_converge():
    """Documents why the coalesced mode exists: more iterations do not help."""
    pf = _filter()
    lw = _log_weights(1.5)
    w = _normalized(lw)
    share = _mean_offspring_share(
        pf, lw, lambda f, s: native.pf_device_resample_megopolis(f._state, 240, s)
    )
    assert 0.5 * np.abs(share - w).sum() > 0.1


def test_coalesced_megopolis_resets_weights_and_keeps_particles():
    pf = _filter()
    before = pf.get_particle_states()
    pf.set_log_weights(_log_weights(1.0))
    native.pf_device_resample_megopolis_coalesced(pf._state, 15, 7)
    after = pf.get_particle_states()
    np.testing.assert_array_equal(pf.get_log_weights(), np.zeros(N))
    rows_before = {tuple(row) for row in before}
    assert all(tuple(row) in rows_before for row in after)


def test_filter_dispatches_coalesced_mode_with_configured_iterations():
    pf = ParticleFilterDevice(n_particles=N, seed=3, resampling="megopolis_coalesced",
                              megopolis_iterations=60)
    pf.initialize([0.0, 0.0, 0.0], clock_bias=0.0, spread_pos=5.0, spread_cb=1.0)
    pf.set_log_weights(_log_weights(1.0))
    pf.resample()
    np.testing.assert_array_equal(pf.get_log_weights(), np.zeros(N))
    assert clone_pf_device_init_kwargs(pf)["megopolis_iterations"] == 60
    assert clone_pf_device_init_kwargs(pf)["resampling"] == "megopolis_coalesced"


def test_rejects_unknown_resampling_and_bad_iterations():
    with pytest.raises(ValueError, match="resampling must be one of"):
        ParticleFilterDevice(n_particles=16, resampling="invalid")
    with pytest.raises(ValueError, match="megopolis_iterations"):
        ParticleFilterDevice(n_particles=16, megopolis_iterations=0)
