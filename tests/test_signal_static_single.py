"""Single static-NAV output matches the retained batch producer exactly."""

import numpy as np
import pytest

from test_signal_sim_workspace import _bits, _channels, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.mark.parametrize('count', [1, 31, 32, 33, 65, 255, 256, 257])
@pytest.mark.parametrize('samples', [1, 31, 255, 256, 257, 4097])
@pytest.mark.parametrize('noise', [-20., -10000.])
@pytest.mark.parametrize('seed', [37, 2**64-1])
def test_static_single_matches_retained_batch(count, samples, noise, seed):
    workspace = native.DeviceSignalWorkspace()
    channels = _channels(count, False)
    for index, channel in enumerate(channels):
        channel['nav_bit'] = (-3, 0, 2, 1, -1)[index % 5]
    try:
        actual = workspace.generate_signal(4.092e6, -1.1e6, channels, samples, noise, seed)
        assert actual.flags.owndata and actual.flags.c_contiguous
        expected = workspace.generate_epochs(4.092e6, -1.1e6, [channels], samples, noise, seed)[0]
        _bits(actual, expected)
    finally:
        workspace.close()
    assert workspace.allocated_bytes == 0
    _bits(actual, expected)


@pytest.mark.parametrize('count', [1, 32, 33, 65])
@pytest.mark.parametrize('seed', [37, 2**64-1])
def test_single_nav_switch_invalid_nav_and_returned_owner(count, seed):
    workspace = native.DeviceSignalWorkspace()
    retained = []
    try:
        for mode in ('plain', 'dynamic', 'empty', 'fallback', 'plain'):
            channels = _channels(257 if mode == 'fallback' else count,
                                 mode in ('dynamic', 'fallback'))
            if mode == 'empty':
                for channel in channels:
                    channel['nav_bits'] = np.empty(0, dtype=np.int32)
            invalid = _channels(1, False)[0]
            invalid.update(prn=99, nav_bits=np.array([-3, 0, 2], np.int32))
            channels.append(invalid)
            actual = workspace.generate_signal(4.092e6, 1e6, channels, 257, -20., seed)
            expected = workspace.generate_epochs(4.092e6, 1e6, [channels], 257, -20., seed)[0]
            _bits(actual, expected)
            retained.append((actual, actual.copy()))
            for output, snapshot in retained:
                _bits(output, snapshot)
    finally:
        workspace.close()
    for output, snapshot in retained:
        _bits(output, snapshot)


@pytest.mark.parametrize('intermediate', [-1e16, 1e16, -1e308, 1e308, 1e-200])
def test_single_extreme_phase_matches_retained_batch(intermediate):
    channels = _channels(65, False)
    for index, channel in enumerate(channels):
        channel.update(doppler_hz=-intermediate+index*.125,
            code_phase=-2147483648. if index % 2 else 2147483647., nav_bits=[])
    workspace = native.DeviceSignalWorkspace()
    try:
        actual = workspace.generate_signal(4.092e6, intermediate, channels, 4097, -10000., 37)
        expected = workspace.generate_epochs(4.092e6, intermediate, [channels], 4097, -10000., 37)[0]
        _bits(actual, expected)
    finally:
        workspace.close()
