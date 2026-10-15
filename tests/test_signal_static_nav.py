"""Static NAV batches match the unchanged single-epoch producer bit for bit."""

import numpy as np
import pytest

from test_signal_sim_workspace import _bits, _channels, native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.mark.parametrize('count', [1, 31, 32, 33, 65, 255, 256, 257])
@pytest.mark.parametrize('samples', [1, 31, 255, 256, 257, 4097])
@pytest.mark.parametrize('noise', [-20., -10000.])
def test_static_nav_batch_matches_original_single_epoch(count, samples, noise):
    workspace = native.DeviceSignalWorkspace()
    rows = [[], _channels(count, False), _channels(max(1, count-1), False)]
    for row in rows:
        for index, channel in enumerate(row):
            channel['nav_bit'] = (-3, 0, 2, 1, -1)[index % 5]
    seeds = [1, 37, 2**64-1]
    try:
        actual = workspace.generate_epochs(2.6e6, -1.1e6, rows, samples, noise, 42, seeds)
        assert actual.flags.owndata and actual.flags.c_contiguous
        expected = np.stack([native.generate_signal(2.6e6, -1.1e6, row, samples, noise, seed)
            for row, seed in zip(rows, seeds, strict=True)])
        _bits(actual, expected)
    finally:
        workspace.close()
    assert workspace.allocated_bytes == 0
    _bits(actual, expected)


@pytest.mark.parametrize('count', [1, 32, 33, 65])
@pytest.mark.parametrize('seed', [37, 2**64-1])
def test_dynamic_empty_nav_fallback_and_closed_owner_keep_outputs(count, seed):
    workspace = native.DeviceSignalWorkspace()
    retained = []
    try:
        for mode in ('plain', 'dynamic', 'empty', 'fallback', 'plain'):
            channels = 257 if mode == 'fallback' else count
            rows = [_channels(channels, mode in ('dynamic', 'fallback')),
                    _channels(channels, False)]
            if mode == 'empty':
                for row in rows:
                    for channel in row:
                        channel['nav_bits'] = np.empty(0, dtype=np.int32)
            actual = workspace.generate_epochs(4.092e6, 1e6, rows, 257, -20., seed)
            expected = np.stack([native.generate_signal(4.092e6, 1e6, row, 257, -20., seed)
                for row in rows])
            _bits(actual, expected)
            retained.append((actual, actual.copy()))
            for output, snapshot in retained:
                _bits(output, snapshot)
    finally:
        workspace.close()
    for output, snapshot in retained:
        _bits(output, snapshot)
