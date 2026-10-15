"""GPU coefficient rounding, partial blocks, fallback and owner reuse."""

import numpy as np
import pytest

native = pytest.importorskip("gnss_gpu._gnss_gpu_signal_sim")
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _channels(count, doppler=-550., nav=False):
    result = []
    for index in range(count):
        system = index % 5
        channel = dict(prn=index % (32, 24, 36, 37, 32)[system] + 1,
                       system=system, code_phase=index - .25,
                       carrier_phase=index * .37 - .13, doppler_hz=doppler + index * .125,
                       amplitude=1. / (index + 1), nav_bit=(-1) ** index)
        if nav and index % 3:
            channel.update(nav_bits=np.array([1, -1, 0, 1, -1], dtype=np.int32),
                           nav_bit_rate=2.6e6 / (index + 1))
        result.append(channel)
    return result


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def _single_reference(channels, samples, intermediate):
    result = np.zeros(2 * samples, dtype=np.float32)
    for channel in channels:
        single = native.generate_single(
            channel["prn"], channel["code_phase"], channel["carrier_phase"],
            channel["doppler_hz"], channel["amplitude"], channel["nav_bit"],
            2.6e6, intermediate, samples, channel["system"])
        np.add(result, single, out=result)
    return result


@pytest.mark.parametrize("intermediate,doppler", [
    (0., -550.), (-1.1e6, 999.5), (1.25e6, -1.25e6),
    (1e16, -1e16 + 2.), (-1e16, 1e16 - 2.),
    (1.25e6, np.nextafter(-1.25e6, np.inf)),
    (1e-200, -1e-200), (1e308, -1e308),
])
@pytest.mark.parametrize("samples", [1, 257])
@pytest.mark.parametrize("count", [1, 32])
def test_coefficient_rounding_matches_unchanged_single_kernel(intermediate, doppler, samples, count):
    channels = _channels(count, doppler)
    expected = _single_reference(channels, samples, intermediate)
    workspace = native.DeviceSignalWorkspace()
    try:
        actual = workspace.generate_signal(2.6e6, intermediate, channels, samples, -10000., 42)
        _assert_bits(actual, expected)
    finally:
        workspace.close()
    assert workspace.allocated_bytes == 0


@pytest.mark.parametrize("count", [1, 12, 31, 32])
@pytest.mark.parametrize("samples", [1, 127, 256, 257])
def test_inline_matches_original_global_descriptor_math_and_seeded_noise(count, samples):
    channels = _channels(count, nav=True)
    # 257 valid descriptors select the original uncached kernel. Added channels
    # contribute only zero; the fixed Philox seed/sequence and nav bits match.
    padding = dict(prn=1, system=0, code_phase=0., carrier_phase=0.,
                   doppler_hz=0., amplitude=0., nav_bit=1)
    fallback = channels + [padding] * (257 - count)
    workspace = native.DeviceSignalWorkspace()
    try:
        expected = workspace.generate_signal(2.6e6, -1.1e6, fallback, samples, -20., 42)
        actual = workspace.generate_signal(2.6e6, -1.1e6, channels, samples, -20., 42)
        _assert_bits(actual, expected)
        # Returned storage must stay independent when the alternate kernel runs.
        snapshot = actual.copy()
        workspace.generate_signal(2.6e6, 1.25e6, fallback, 8193, -20., 1)
        _assert_bits(actual, snapshot)
    finally:
        workspace.close()


def test_reused_owner_alternates_inline_global_and_noise_only_partial_blocks():
    workspace = native.DeviceSignalWorkspace()
    retained = []
    try:
        for samples, count in ((1, 32), (8193, 33), (257, 31), (127, 65),
                               (2, 1), (513, 0), (256, 32), (1, 33)):
            channels = _channels(count, nav=True)
            expected = native.generate_signal(2.6e6, -1.1e6, channels, samples, -20., 37)
            actual = workspace.generate_signal(2.6e6, -1.1e6, channels, samples, -20., 37)
            _assert_bits(actual, expected)
            retained.append((actual, actual.copy()))
            for array, snapshot in retained:
                _assert_bits(array, snapshot)
    finally:
        workspace.close()
    assert workspace.allocated_bytes == 0
    for array, snapshot in retained:
        _assert_bits(array, snapshot)
