"""Epoch batching preserves ordered IQ, seed/local time, ownership and laziness."""

import pickle
import types

import numpy as np
import pytest

from gnss_gpu.signal_sim import SignalSimulator
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator
from test_signal_sim_workspace import _bits, _channels, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.mark.parametrize('samples', [1, 31, 256, 257])
@pytest.mark.parametrize('counts', [(0, 1, 12), (12, 32, 33), (0, 65, 256), (1, 256, 257)])
@pytest.mark.parametrize('seeds', [None, [1, 37, 2**64 - 1]])
def test_epoch_cache_boundaries_mixed_systems_nav_and_noise_match_single_kernel(samples, counts, seeds):
    rows = [_channels(n) for n in counts]
    rows[0].append(dict(prn=1, system=99, code_phase=0., carrier_phase=0.,
                        doppler_hz=0., amplitude=1., nav_bit=1))
    workspace = native.DeviceSignalWorkspace()
    try:
        expected = np.stack([workspace.generate_signal(2.6e6, -1.1e6, row, samples, -20.,
            42 if seeds is None else seeds[e]) for e, row in enumerate(rows)])
        actual = workspace.generate_epochs(2.6e6, -1.1e6, rows, samples, -20., 42, seeds)
        _bits(actual, expected)
        assert actual.shape == (len(rows), 2 * samples) and actual.flags.c_contiguous
        snapshot = actual.copy()
        workspace.generate_epochs(4.092e6, 1.25e6, [_channels(257)] * 3, 513, -20., 37)
        for row in rows:
            for channel in row:
                if 'nav_bits' in channel:
                    channel['nav_bits'].fill(0)
        _bits(actual, snapshot)
    finally:
        workspace.close()
    assert workspace.allocated_bytes == 0
    _bits(actual, snapshot)


def test_empty_closed_invalid_batch_and_seed_contracts():
    workspace = native.DeviceSignalWorkspace()
    try:
        result = workspace.generate_epochs(2.6e6, 0., [], 257, -20., 42)
        assert result.shape == (0, 514) and result.dtype == np.float32
        assert workspace.allocated_bytes == 0
        for rows, samples, seeds in (([[]], 0, None), ([[]], 257, [1, 2]), ([[]] * 65536, 1, None)):
            with pytest.raises(RuntimeError):
                workspace.generate_epochs(2.6e6, 0., rows, samples, -20., 42, seeds)
        rows = [_channels(12), []]
        _bits(workspace.generate_epochs(2.6e6, 0., rows, 257, -20., 42),
              native.generate_epochs(2.6e6, 0., rows, 257, -20., 42))
    finally:
        workspace.close()
    with pytest.raises(RuntimeError, match='closed'):
        workspace.generate_epochs(2.6e6, 0., [], 1, -20., 42)


def test_zero_seed_uses_fresh_independent_noise_and_no_noise_matches_exactly():
    workspace = native.DeviceSignalWorkspace()
    try:
        rows = [[], [], []]
        first = workspace.generate_epochs(2.6e6, 0., rows, 257, -20., 0)
        second = workspace.generate_epochs(2.6e6, 0., rows, 257, -20., 0)
        assert not np.array_equal(first, second)
        assert not np.array_equal(first[0], first[1])
        rows = [_channels(3, nav=False)] * 3
        expected = np.stack([workspace.generate_signal(2.6e6, 0., row, 257, -10000., 0) for row in rows])
        _bits(workspace.generate_epochs(2.6e6, 0., rows, 257, -10000., 0), expected)
    finally:
        workspace.close()


def test_public_batch_seeded_rows_release_pickle_and_old_extension_fallback(monkeypatch):
    sim = SignalSimulator(noise_seed=42)
    rows = [_channels(n) for n in (0, 12, 33)]
    seeds = [1, 37, 2**64 - 1]
    expected = np.stack([native.generate_signal(sim.sampling_freq, sim.intermediate_freq,
        row, 257, sim.noise_floor_db, seeds[e]) for e, row in enumerate(rows)])
    try:
        _bits(sim.generate_epochs(rows, 257, seeds), expected)
        cloned = pickle.loads(pickle.dumps(sim))
        try:
            _bits(cloned.generate_epochs(rows, 257, seeds), expected)
        finally:
            cloned.release_gpu()
        sim.release_gpu()
        import gnss_gpu
        monkeypatch.setattr(gnss_gpu, '_gnss_gpu_signal_sim', types.SimpleNamespace(generate_signal=native.generate_signal))
        _bits(sim.generate_epochs(rows, 257, seeds), expected)
        assert sim.generate_epochs([], 257).shape == (0, 514)
        with pytest.raises(ValueError, match='noise_seeds'):
            sim.generate_epochs(rows, 257, [1])
    finally:
        sim.release_gpu()


@pytest.mark.parametrize('batch_size', [1, 3, 32])
def test_trajectory_preserves_metadata_iq_and_default_callback_laziness(batch_size):
    sim = UrbanSignalSimulator()
    sim.sim.noise_seed = 42
    positions = np.repeat([[6378137., 0., 0.]], 7, axis=0)
    positions[:, 1] = np.arange(7) * .1
    sats = np.array([[2.7e7, 1e7, 0.], [2.6e7, -1e7, 1e7]])
    calls = []
    def satellites(i):
        calls.append(i)
        return sats
    try:
        expected = list(sim.simulate_trajectory(positions, lambda i: sats, n_samples=257))
        stream = sim.simulate_trajectory(positions, satellites, n_samples=257, batch_size=batch_size)
        assert calls == []
        first = next(stream)
        assert calls == list(range(min(batch_size, len(positions))))
        actual = [first, *stream]
        for (i, old), (j, new) in zip(expected, actual):
            assert i == j and old.keys() == new.keys()
            _bits(new['iq'], old['iq'])
            for key in old:
                if key == 'iq':
                    continue
                if isinstance(old[key], np.ndarray):
                    np.testing.assert_array_equal(new[key], old[key])
                else:
                    assert new[key] == old[key]
        retained = first[1]['iq'].copy()
        list(sim.simulate_trajectory(positions, lambda i: sats, n_samples=257, batch_size=3))
        _bits(first[1]['iq'], retained)
    finally:
        sim.sim.release_gpu()


def test_trajectory_caps_chunk_output_without_evaluating_the_full_input(monkeypatch):
    sim = UrbanSignalSimulator()
    calls, batches = [], []
    samples = 2**20
    positions = np.repeat([[6378137., 0., 0.]], 9, axis=0)
    def prepare(**kwargs):
        calls.append(len(calls))
        return {'channels': [], 'iq': None}
    def generate(rows, n_samples):
        batches.append((len(rows), n_samples))
        return np.zeros((len(rows), 2), dtype=np.float32)
    monkeypatch.setattr(sim, '_prepare_epoch', prepare)
    monkeypatch.setattr(sim.sim, 'generate_epochs', generate)
    stream = sim.simulate_trajectory(positions, lambda i: np.empty((0, 3)),
                                     n_samples=samples, batch_size=1000)
    next(stream)
    assert calls == [0, 1, 2, 3] and batches == [(4, samples)]
    list(stream)
    assert batches == [(4, samples), (4, samples), (1, samples)]
