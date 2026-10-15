"""Check fused generation against the independent single-channel CUDA kernel."""

import numpy as np
import pytest

native = pytest.importorskip("gnss_gpu._gnss_gpu_signal_sim")
pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


def _channel(prn=1, **overrides):
    result = dict(prn=prn, system=0, code_phase=-.25, carrier_phase=.37,
                  doppler_hz=-550., amplitude=.731, nav_bit=1)
    result.update(overrides)
    return result


def _single(channel, samples, frequency=2.6e6, intermediate=0., nav_bit=None):
    return native.generate_single(
        channel["prn"], channel["code_phase"], channel["carrier_phase"],
        channel["doppler_hz"], channel["amplitude"],
        channel["nav_bit"] if nav_bit is None else nav_bit,
        frequency, intermediate, samples, channel["system"])


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("samples", [1, 255, 256, 257, 2600, 8193])
@pytest.mark.parametrize("count", [1, 11, 31, 32, 33, 65])
def test_channel_sum_matches_single_kernel(samples, count):
    channels = [_channel(index % 32 + 1, code_phase=index - .25,
                         carrier_phase=index * .37, doppler_hz=index * 100. - 550.,
                         amplitude=1. / (index + 1), nav_bit=(-1) ** index)
                for index in range(count)]
    expected = np.zeros(2 * samples, dtype=np.float32)
    for channel in channels:
        np.add(expected, _single(channel, samples), out=expected)
    actual = native.generate_signal(2.6e6, 0., channels, samples, -10000., 42)
    _assert_bits(actual, expected)


@pytest.mark.parametrize("system,prn", [(0, 32), (1, 24), (2, 36), (3, 37), (4, 32)])
def test_constellations_match_single_kernel(system, prn):
    channel = _channel(prn, system=system, code_phase=-4092.75,
                       carrier_phase=-1e5, doppler_hz=-12345.67,
                       amplitude=-.731, nav_bit=-1)
    expected = np.zeros(2 * 4097, dtype=np.float32)
    np.add(expected, _single(channel, 4097, 4.092e6, 1.25e6), out=expected)
    actual = native.generate_signal(4.092e6, 1.25e6, [channel], 4097, -10000., 1)
    _assert_bits(actual, expected)


@pytest.mark.parametrize("length", [0, 1, 2, 7])
@pytest.mark.parametrize("count", [11, 33])
def test_nav_bits_batch_boundaries_and_clamping(length, count):
    samples = 257
    frequency = 26000.
    channels = []
    originals = []
    expected = np.zeros((samples, 2), dtype=np.float32)
    for index in range(count):
        bits = np.resize(np.array([-3, 0, 2, 1, -1], dtype=np.int32), length)
        bits.setflags(write=False)
        channel = _channel(index % 32 + 1, nav_bits=bits,
                           nav_bit_rate=frequency / (index + 1), nav_bit=-2)
        channels.append(channel)
        originals.append(bits.copy())
        if length:
            bit_index = ((np.arange(samples) / frequency)
                         * channel["nav_bit_rate"]).astype(int)
            selected = bits[np.minimum(bit_index, length - 1)]
            contribution = np.empty((samples, 2), dtype=np.float32)
            for value in np.unique(selected):
                single = _single(channel, samples, frequency, nav_bit=int(value)).reshape(-1, 2)
                contribution[selected == value] = single[selected == value]
        else:
            contribution = _single(channel, samples, frequency).reshape(-1, 2)
        np.add(expected, contribution, out=expected)
    actual = native.generate_signal(frequency, 0., channels, samples, -10000., 42)
    _assert_bits(actual, expected.ravel())
    for channel, original in zip(channels, originals):
        np.testing.assert_array_equal(channel["nav_bits"], original)


def test_invalid_channels_are_skipped_without_changing_order():
    valid = [_channel(amplitude=value) for value in (1e8, 1., -1e8)]
    invalid = [_channel(33), _channel(25, system=1), _channel(37, system=2),
               _channel(38, system=3), _channel(33, system=4), _channel(system=99)]
    mixed = [valid[0], *invalid, valid[1], valid[2]]
    reference = native.generate_signal(2.6e6, 0., valid, 257, -20., 42)
    _assert_bits(native.generate_signal(2.6e6, 0., mixed, 257, -20., 42), reference)
    _assert_bits(native.generate_signal(2.6e6, 0., invalid, 257, -20., 42),
                 native.generate_signal(2.6e6, 0., [], 257, -20., 42))


def test_input_order_retains_float_rounding():
    def generate(amplitudes):
        channels = [_channel(code_phase=0., carrier_phase=0., doppler_hz=0., amplitude=value)
                    for value in amplitudes]
        return native.generate_signal(2.6e6, 0., channels, 257, -10000., 42)

    cancelled = generate((1e8, 1., -1e8))
    retained = generate((1e8, -1e8, 1.))
    assert np.all(cancelled[::2] == 0.)
    assert np.all(np.abs(retained[::2]) == 1.)


@pytest.mark.parametrize("count", [0, 1, 32, 33])
@pytest.mark.parametrize("seed", [1, 42, 2**64 - 1])
def test_seed_is_reproducible_and_distinguishes_noise(count, seed):
    channels = [_channel(index % 32 + 1) for index in range(count)]
    args = (2.6e6, 0., channels, 1025, -20.)
    first = native.generate_signal(*args, seed)
    _assert_bits(native.generate_signal(*args, seed), first)
    other = native.generate_signal(*args, 2 if seed == 1 else 1)
    assert not np.array_equal(other, first)


def test_zero_seed_selects_fresh_noise_and_disabled_noise_is_zero():
    args = (2.6e6, 0., [_channel()], 257)
    first = native.generate_signal(*args, -20., 0)
    assert not np.array_equal(native.generate_signal(*args, -20., 0), first)
    _assert_bits(native.generate_signal(2.6e6, 0., [], 257, -10000., 0),
                 np.zeros(514, dtype=np.float32))
