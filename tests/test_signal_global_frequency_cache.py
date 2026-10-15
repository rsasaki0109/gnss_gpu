"""Global coefficient cache boundaries, original math and partial-block sync."""

import pytest

from test_signal_frequency_cache import _assert_bits, _channels, _single_reference

native = pytest.importorskip("gnss_gpu._gnss_gpu_signal_sim")
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.mark.parametrize("count", [33, 64, 65, 127, 128, 255, 256])
@pytest.mark.parametrize("samples", [1, 127, 256, 257])
def test_global_cache_matches_unchanged_uncached_kernel_with_seeded_nav(count, samples):
    channels = _channels(count, nav=True)
    padding = dict(prn=1, system=0, code_phase=0., carrier_phase=0.,
                   doppler_hz=0., amplitude=0., nav_bit=1)
    fallback = channels + [padding] * (257 - count)
    workspace = native.DeviceSignalWorkspace()
    try:
        expected = workspace.generate_signal(2.6e6, -1.1e6, fallback, samples, -20., 42)
        actual = workspace.generate_signal(2.6e6, -1.1e6, channels, samples, -20., 42)
        _assert_bits(actual, expected)
        snapshot = actual.copy()
        workspace.generate_signal(4.092e6, 1.25e6, fallback, 8193, -20., 37)
        _assert_bits(actual, snapshot)
    finally:
        workspace.close()
    assert workspace.allocated_bytes == 0


@pytest.mark.parametrize("count", [33, 256])
@pytest.mark.parametrize("samples", [1, 257])
@pytest.mark.parametrize("intermediate,doppler", [
    (0., -550.), (1.25e6, -1.25e6), (1e16, -1e16 + 2.),
    (-1e16, 1e16 - 2.), (1e-200, -1e-200), (1e308, -1e308),
])
def test_global_coefficient_matches_independent_single_kernel(count, samples, intermediate, doppler):
    channels = _channels(count, doppler)
    expected = _single_reference(channels, samples, intermediate)
    workspace = native.DeviceSignalWorkspace()
    try:
        actual = workspace.generate_signal(2.6e6, intermediate, channels, samples, -10000., 42)
        _assert_bits(actual, expected)
    finally:
        workspace.close()


@pytest.mark.parametrize("count", [256, 257])
def test_invalid_channels_do_not_change_the_cache_dispatch_boundary(count):
    channels = _channels(count, nav=True)
    invalid = dict(prn=1, system=99, code_phase=0., carrier_phase=0.,
                   doppler_hz=0., amplitude=1., nav_bit=1)
    workspace = native.DeviceSignalWorkspace()
    try:
        expected = workspace.generate_signal(2.6e6, 1.25e6, channels, 257, -20., 37)
        for position in (0, 32, 128, count):
            mixed = channels[:position] + [invalid] * 17 + channels[position:]
            _assert_bits(workspace.generate_signal(2.6e6, 1.25e6, mixed, 257, -20., 37), expected)
    finally:
        workspace.close()


def test_owner_rebuilds_coefficients_when_values_and_dispatch_change():
    workspace = native.DeviceSignalWorkspace()
    retained = []
    try:
        channels = _channels(256, nav=True)
        for index, (count, samples) in enumerate(((256, 1), (33, 257), (32, 127), (257, 8193),
                                                 (65, 2), (513, 257), (0, 513), (255, 256), (256, 1))):
            if count <= 256:
                selected = channels[:count]
            else:
                selected = _channels(count, nav=True)
            for channel in selected:
                channel["doppler_hz"] += index * 123.5
                channel["carrier_phase"] -= index * .1
                if "nav_bits" in channel:
                    channel["nav_bits"] *= -1
            arguments = (4.092e6, -1.25e6 + index * 100., selected, samples, -20., 37)
            expected = native.generate_signal(*arguments)
            actual = workspace.generate_signal(*arguments)
            _assert_bits(actual, expected)
            retained.append((actual, actual.copy()))
            for array, snapshot in retained:
                _assert_bits(array, snapshot)
    finally:
        workspace.close()
    assert workspace.allocated_bytes == 0
