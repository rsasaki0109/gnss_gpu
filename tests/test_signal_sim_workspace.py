"""Exercise reusable IQ work buffers, input refresh, and object lifetime."""

import copy
import gc
import pickle
import sys
import types

import numpy as np
import pytest

from gnss_gpu.signal_sim import SignalSimulator

native = pytest.importorskip("gnss_gpu._gnss_gpu_signal_sim")
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _channels(count, nav=True):
    limits = (32, 24, 36, 37, 32)
    channels = []
    for index in range(count):
        system = index % 5
        channel = dict(prn=index % limits[system] + 1, system=system,
                       code_phase=-.25 + index, carrier_phase=.37 * index,
                       doppler_hz=-550. + 100. * index, amplitude=1. / (index + 1),
                       nav_bit=(-1) ** index)
        if nav:
            channel.update(nav_bits=np.array([1, -1, 0, 1], dtype=np.int32),
                           nav_bit_rate=26000. * (index + 1))
        channels.append(channel)
    return channels


def _bits(actual, expected):
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("count", [0, 1, 32, 33, 65])
@pytest.mark.parametrize("samples", [1, 257, 2600])
def test_reuse_refreshes_channels_nav_and_config(count, samples):
    workspace = native.DeviceSignalWorkspace()
    retained = []
    channels = _channels(count)
    try:
        for iteration, noise in enumerate((-20., -10000., 0.)):
            args = (2.6e6 + iteration * 1e6, -1e6 + iteration * 5e5,
                    channels, samples, noise, 2**64 - 1)
            expected = native.generate_signal(*args)
            actual = workspace.generate_signal(*args)
            _bits(actual, expected)
            retained.append((actual, actual.copy()))
            for channel in channels:
                channel["nav_bits"] *= -1
                channel["code_phase"] += .123
                channel["doppler_hz"] += 345.67
            for output, snapshot in retained:
                _bits(output, snapshot)
    finally:
        workspace.close()
    assert workspace.allocated_bytes == 0


def test_growth_shrink_empty_invalid_and_returned_storage():
    workspace = native.DeviceSignalWorkspace()
    retained = []
    try:
        for samples, count, nav in ((1, 1, False), (8193, 65, True), (2, 32, False),
                                    (26000, 33, True), (257, 0, False), (1, 65, True)):
            channels = _channels(count, nav)
            channels += [dict(prn=99, code_phase=0., carrier_phase=0.,
                              doppler_hz=0., amplitude=1., nav_bit=1)]
            args = (2.6e6, 0., channels, samples, -20., 37)
            actual = workspace.generate_signal(*args)
            _bits(actual, native.generate_signal(*args))
            retained.append((actual, actual.copy()))
            for channel in channels:
                if "nav_bits" in channel:
                    channel["nav_bits"].fill(0)
            for output, snapshot in retained:
                _bits(output, snapshot)
        allocated = workspace.allocated_bytes
        workspace.generate_signal(2.6e6, 0., [], 1, -10000., 1)
        assert workspace.allocated_bytes == allocated
    finally:
        workspace.close()
    for output, snapshot in retained:
        _bits(output, snapshot)
    workspace.close()
    assert workspace.allocated_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        workspace.generate_signal(2.6e6, 0., [], 1, -20., 1)


def test_independent_owners_and_destructor_preserve_outputs():
    first, second = native.DeviceSignalWorkspace(), native.DeviceSignalWorkspace()
    args = (2.6e6, 0., _channels(33), 257, -20., 42)
    output = first.generate_signal(*args)
    snapshot = output.copy()
    second.generate_signal(2.6e6, 0., [], 8193, -20., 1)
    first.close()
    _bits(second.generate_signal(*args), snapshot)
    del first, second
    gc.collect()
    _bits(output, snapshot)


@pytest.mark.parametrize("field,value", [("sampling_freq", 0.), ("intermediate_freq", np.nan),
                                       ("n_samples", 0), ("noise_floor_db", np.inf)])
def test_invalid_call_leaves_workspace_usable(field, value):
    workspace = native.DeviceSignalWorkspace()
    arguments = dict(sampling_freq=2.6e6, intermediate_freq=0., channels=_channels(1),
                     n_samples=257, noise_floor_db=-20., noise_seed=42)
    try:
        expected = workspace.generate_signal(**arguments)
        invalid = dict(arguments, **{field: value})
        with pytest.raises(RuntimeError):
            workspace.generate_signal(**invalid)
        _bits(workspace.generate_signal(**arguments), expected)
    finally:
        workspace.close()


def test_fresh_seed_and_noise_disabled():
    workspace = native.DeviceSignalWorkspace()
    try:
        args = (2.6e6, 0., _channels(1), 257, -20., 0)
        first = workspace.generate_signal(*args)
        assert not np.array_equal(first, workspace.generate_signal(*args))
        _bits(workspace.generate_signal(2.6e6, 0., [], 257, -10000., 0),
              np.zeros(514, dtype=np.float32))
    finally:
        workspace.close()


def test_public_release_recreate_and_settings_changes():
    simulator = SignalSimulator(noise_seed=37)
    channels = _channels(33)
    assert simulator._workspace is None
    try:
        expected = simulator.generate_epoch(channels, 257)
        owner = simulator._workspace
        simulator.release_gpu()
        assert owner.allocated_bytes == 0 and simulator._workspace is None
        _bits(simulator.generate_epoch(channels, 257), expected)
        assert simulator._workspace is not owner
        simulator.sampling_freq = 4.092e6
        simulator.intermediate_freq = -1e6
        simulator.noise_floor_db = -40.
        simulator.noise_seed = 2**64 - 1
        _bits(simulator.generate_epoch(channels, 257), native.generate_signal(
            4.092e6, -1e6, channels, 257, -40., 2**64 - 1))
        simulator.generate_test_signal(7, code_phase=-.25, doppler=-999.)
        _bits(simulator.generate_epoch(channels, 257), native.generate_signal(
            4.092e6, -1e6, channels, 257, -40., 2**64 - 1))
    finally:
        simulator.release_gpu()


@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy,
                                   lambda value: pickle.loads(pickle.dumps(value))])
def test_copy_pickle_create_independent_lazy_workspaces(clone):
    simulator = SignalSimulator(noise_seed=42)
    restored = None
    try:
        channels = _channels(11)
        expected = simulator.generate_epoch(channels, 257)
        restored = clone(simulator)
        assert restored._workspace is None
        _bits(restored.generate_epoch(channels, 257), expected)
        assert restored._workspace is not simulator._workspace
        simulator.release_gpu()
        _bits(restored.generate_epoch(channels, 257), expected)
    finally:
        simulator.release_gpu()
        if restored is not None:
            restored.release_gpu()


def test_wrapper_supports_function_only_extension(monkeypatch):
    import gnss_gpu

    legacy = types.ModuleType("gnss_gpu._gnss_gpu_signal_sim")
    legacy.generate_signal = native.generate_signal
    monkeypatch.setitem(sys.modules, legacy.__name__, legacy)
    monkeypatch.setattr(gnss_gpu, "_gnss_gpu_signal_sim", legacy)
    simulator = SignalSimulator(noise_seed=42)
    channels = _channels(1)
    _bits(simulator.generate_epoch(channels, 257),
          native.generate_signal(2.6e6, 0., channels, 257, -20., 42))
    assert simulator._workspace is None
