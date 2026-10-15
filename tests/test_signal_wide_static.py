"""Wide static NAV agrees bitwise with the retained dynamic GPU producer."""

import numpy as np
import pytest

from test_signal_sim_workspace import _bits, _channels, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _dynamic_constant(row):
    return [dict(ch, nav_bits=np.array([ch['nav_bit']], np.int32),
                 nav_bit_rate=1.) for ch in row]


@pytest.mark.parametrize('count', [257, 258, 512, 1024])
@pytest.mark.parametrize('samples', [1, 255, 256, 257, 4097])
@pytest.mark.parametrize('noise', [-20., -10000.])
@pytest.mark.parametrize('seed', [37, 2**64-1])
def test_wide_static_matches_retained_dynamic(count, samples, noise, seed):
    row = _channels(count, False)
    for i, ch in enumerate(row):
        ch['nav_bit'] = (-3, 0, 2, 1, -1)[i % 5]
    oracle = _dynamic_constant(row)
    workspace = native.DeviceSignalWorkspace()
    retained = []
    try:
        for batch in (False, True):
            call = workspace.generate_epochs if batch else workspace.generate_signal
            actual = call(4.092e6, -1.1e6, [row]*3 if batch else row, samples, noise, seed)
            expected = call(4.092e6, -1.1e6, [oracle]*3 if batch else oracle,
                            samples, noise, seed)
            assert actual.flags.owndata and actual.flags.c_contiguous
            _bits(actual, expected)
            retained.append((actual, actual.copy()))
    finally:
        workspace.close()
    assert workspace.allocated_bytes == 0
    for actual, snapshot in retained:
        _bits(actual, snapshot)


@pytest.mark.parametrize('intermediate', [-1e16, 1e16, -1e308, 1e308, 1e-200])
@pytest.mark.parametrize('batch', [False, True])
def test_wide_static_extreme_phase(intermediate, batch):
    row = _channels(257, False)
    for i, ch in enumerate(row):
        ch.update(doppler_hz=-intermediate+i*.125,
                  code_phase=-2147483648. if i % 2 else 2147483647.)
    oracle = _dynamic_constant(row)
    workspace = native.DeviceSignalWorkspace()
    try:
        call = workspace.generate_epochs if batch else workspace.generate_signal
        actual = call(4.092e6, intermediate, [row]*3 if batch else row, 4097, -10000., 37)
        expected = call(4.092e6, intermediate, [oracle]*3 if batch else oracle,
                        4097, -10000., 37)
        _bits(actual, expected)
    finally:
        workspace.close()


@pytest.mark.parametrize('batch', [False, True])
def test_wide_static_switches_empty_dynamic_invalid_and_owning_outputs(batch):
    workspace = native.DeviceSignalWorkspace()
    retained = []
    try:
        for count, dynamic in ((257, False), (512, True), (0, False),
                               (256, False), (1024, False), (33, True), (258, False)):
            row = _channels(count, dynamic)
            invalid = _channels(1, False)[0]
            invalid.update(prn=99, nav_bits=np.array([-3, 0, 2], np.int32))
            row.append(invalid)
            oracle = row if dynamic else _dynamic_constant(row)
            call = workspace.generate_epochs if batch else workspace.generate_signal
            # A mixed batch includes empty and <=256 epochs with a wide maximum.
            rows = [[], row, _channels(33, False)]
            oracles = [[], oracle, _dynamic_constant(rows[2])]
            actual = call(4.092e6, 1e6, rows if batch else row, 257, -20., 2**64-1)
            expected = call(4.092e6, 1e6, oracles if batch else oracle,
                            257, -20., 2**64-1)
            _bits(actual, expected)
            retained.append((actual, actual.copy()))
            for output, snapshot in retained:
                _bits(output, snapshot)
    finally:
        workspace.close()
    for output, snapshot in retained:
        _bits(output, snapshot)
