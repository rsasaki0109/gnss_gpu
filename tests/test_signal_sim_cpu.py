"""Physical waveform and parallel execution checks without native bindings."""

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from gnss_gpu import CpuSignalSimulator, UrbanSignalSimulator
from gnss_gpu.signal_sim_cpu import CHIP_RATE, L1_HZ
from gnss_gpu.validation.tracking_residuals import generate_ca_code


def channel(**overrides):
    return dict(dict(prn=1, code_phase=0., carrier_phase=0., doppler_hz=0.,
                     amplitude=1., nav_bit=1), **overrides)


def test_gps_prn1_published_initial_chips_and_full_code_period():
    sim = CpuSignalSimulator(sampling_freq=CHIP_RATE, noise_floor_db=-10000)
    iq = sim.generate_epoch([channel()], 2046).reshape(-1, 2)
    # IS-GPS-200 PRN 1 initial ten chips (octal 1440), using 2*bit-1.
    np.testing.assert_array_equal(iq[:10, 0], [1, 1, -1, -1, 1, -1, -1, -1, -1, -1])
    np.testing.assert_array_equal(iq[:1023], iq[1023:])
    np.testing.assert_array_equal(iq[:, 1], 0)


def test_carrier_code_doppler_and_navigation_bits_match_numpy_reference():
    fs, count = 2.6e6, 120001
    ch = channel(prn=7, code_phase=-2.25, carrier_phase=.7, doppler_hz=5200.,
                 nav_bits=[1, -1, 1], amplitude=.4)
    sim = CpuSignalSimulator(fs, intermediate_freq=120000., noise_floor_db=-10000)
    output = sim.generate_epoch([ch], count).reshape(-1, 2)
    t = np.arange(count) / fs
    index = np.floor(ch["code_phase"] + CHIP_RATE*(1+ch["doppler_hz"]/L1_HZ)*t).astype(int) % 1023
    nav = np.asarray(ch["nav_bits"])[np.floor(t*50).astype(int)]
    expected = .4 * generate_ca_code(7)[index] * nav * np.exp(1j*(.7+2*np.pi*125200*t))
    np.testing.assert_allclose(output[:, 0], expected.real, atol=3e-8, rtol=1e-7)
    np.testing.assert_allclose(output[:, 1], expected.imag, atol=3e-8, rtol=1e-7)
    # The carrier Doppler must move some code boundaries during this interval.
    assert np.any(index != np.floor(ch["code_phase"]+CHIP_RATE*t).astype(int) % 1023)


def test_streaming_blocks_keep_carrier_code_and_nav_boundaries():
    sim = CpuSignalSimulator(noise_floor_db=-10000)
    channels = [channel(code_phase=-.75, carrier_phase=1.1, doppler_hz=-4200.,
                        nav_bits=[1, -1, 1]),
                channel(prn=2, amplitude=.3, doppler_hz=500.)]
    whole = sim.generate_epoch(channels, 110000)
    chunks = [sim.generate_epoch(channels, stop-start, start_sample=start)
              for start, stop in [(0, 51999), (51999, 52001), (52001, 110000)]]
    np.testing.assert_array_equal(np.concatenate(chunks), whole)


def test_seeded_noise_and_multichannel_sums_are_thread_count_independent():
    threads = numba.get_num_threads()
    sim = CpuSignalSimulator(noise_seed=31)
    channels = [channel(prn=prn, doppler_hz=prn*100., amplitude=1/prn)
                for prn in range(1, 13)]
    try:
        numba.set_num_threads(1)
        serial = sim.generate_epoch(channels, 8193)
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        parallel = sim.generate_epoch(channels, 8193)
    finally:
        numba.set_num_threads(threads)
    assert serial.dtype == np.float32
    np.testing.assert_array_equal(serial, parallel)
    assert not np.array_equal(sim.generate_epoch([], 8193), np.zeros_like(serial))


def test_test_signal_cn0_uses_sample_rate_and_complex_noise_bandwidth():
    fs, cn0 = 100000., 40.
    sim = CpuSignalSimulator(fs, noise_seed=42)
    noisy = sim.generate_test_signal(1, cn0_dbhz=cn0, duration_s=1.)
    clean = CpuSignalSimulator(fs, noise_floor_db=-10000).generate_epoch([channel()], int(fs))
    noise = (noisy-clean).reshape(-1, 2)
    expected_variance = fs / (2*10**(cn0/10))
    np.testing.assert_allclose(noise.var(axis=0), expected_variance, rtol=.02)
    assert abs(np.corrcoef(noise.T)[0, 1]) < .02


def test_batch_seed_contract_and_empty_batch():
    sim = CpuSignalSimulator(noise_seed=5)
    rows = sim.generate_epochs([[channel()], [channel()]], 32, noise_seeds=[7, 9])
    for row, seed in zip(rows, [7, 9]):
        expected = CpuSignalSimulator(noise_seed=seed).generate_epoch([channel()], 32)
        np.testing.assert_array_equal(row, expected)
    assert sim.generate_epochs([], 32).shape == (0, 64)
    np.testing.assert_array_equal(
        sim.generate_epochs([[]], 32, noise_seeds=[2**64-1])[0],
        CpuSignalSimulator(noise_seed=2**64-1).generate_epoch([], 32),
    )


@pytest.mark.parametrize("noise_db", [-10000., -17.])
@pytest.mark.parametrize("threads", [1, min(4, numba.config.NUMBA_NUM_THREADS)])
@pytest.mark.parametrize("sampling_freq,samples", [(260000., 5201), (2600000., 52001)])
def test_batch_matches_independent_epochs_with_ragged_channels_and_nav_transitions(noise_db, threads, sampling_freq, samples):
    original_threads = numba.get_num_threads()
    rows = [[], [channel(prn=7, nav_bits=[1, -1], doppler_hz=-2700.),
                 channel(prn=7, nav_bits=[-1, 1], amplitude=-.2, carrier_phase=.7)],
            [channel(prn=32, nav_bit=-1, code_phase=-.3, nav_bits=[], doppler_hz=1330.25)],
            [], [channel(prn=12, nav_bits=[-1, 1], nav_bit_rate=25., amplitude=.4)]]
    seeds = [31, 42, 2**64-1, 31, 9]
    simulator = CpuSignalSimulator(sampling_freq, intermediate_freq=123456., noise_floor_db=noise_db)
    try:
        numba.set_num_threads(threads)
        # Cross the 50 bit/s navigation boundary in both the short-batch and
        # long-epoch fallback paths. Lower fs is an intentional sampling test.
        expected = np.stack([CpuSignalSimulator(sampling_freq, intermediate_freq=123456., noise_floor_db=noise_db,
                                                noise_seed=seed).generate_epoch(row, samples)
                             for row, seed in zip(rows, seeds)])
        actual = simulator.generate_epochs(iter(rows), samples, noise_seeds=iter(seeds))
    finally:
        numba.set_num_threads(original_threads)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    assert rows[1][0]["nav_bits"] == [1, -1]


@pytest.mark.parametrize("noise_db", [-10000., -17.])
def test_batch_all_empty_epochs_and_repeated_seeds(noise_db):
    simulator = CpuSignalSimulator(noise_floor_db=noise_db, noise_seed=31)
    actual = simulator.generate_epochs([[], [], []], 33)
    expected = simulator.generate_epoch([], 33)
    np.testing.assert_array_equal(actual, np.stack([expected]*3))
    actual[0, 0] = 123.
    np.testing.assert_array_equal(actual[1], expected)


@pytest.mark.parametrize("invalid", [channel(prn=33), channel(nav_bits=[1], nav_bit_rate=1e9),
                                      channel(code_phase=1e100)])
def test_batch_rejects_invalid_later_epoch_before_kernel_indexing(invalid):
    with pytest.raises(ValueError):
        CpuSignalSimulator().generate_epochs([[channel()], [], [invalid]], 64)


@pytest.mark.parametrize("samples", [8192, 8193])
def test_batch_dispatch_boundary_preserves_samples(samples):
    original_threads = numba.get_num_threads()
    rows = [[channel(prn=32, doppler_hz=-1800.)], [channel(prn=7, carrier_phase=.3)]]
    simulator = CpuSignalSimulator(noise_floor_db=-10000)
    try:
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        expected = np.stack([simulator.generate_epoch(row, samples) for row in rows])
        actual = simulator.generate_epochs(rows, samples)
    finally:
        numba.set_num_threads(original_threads)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("threads", [1, min(4, numba.config.NUMBA_NUM_THREADS)])
def test_offset_batch_concatenates_to_continuous_noiseless_signal(threads):
    simulator = CpuSignalSimulator(noise_floor_db=-10000)
    channels = [channel(code_phase=-.75, carrier_phase=1.1, doppler_hz=-4200.,
                        nav_bits=[1, -1, 1]), channel(prn=32, amplitude=.3, doppler_hz=1330.25)]
    original_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(threads)
        starts = list(range(0, 78000, 2600))
        actual = simulator.generate_epochs([channels]*len(starts), 2600, start_samples=iter(starts))
        whole = simulator.generate_epoch(channels, 78000)
    finally:
        numba.set_num_threads(original_threads)
    np.testing.assert_array_equal(actual.ravel(), whole)


@pytest.mark.parametrize("noise_db", [-10000., -17.])
@pytest.mark.parametrize("threads", [1, min(4, numba.config.NUMBA_NUM_THREADS)])
def test_offset_batch_preserves_arbitrary_order_ragged_rows_and_noise(noise_db, threads):
    simulator = CpuSignalSimulator(noise_floor_db=noise_db)
    channels = [channel(prn=7, nav_bits=[1, -1]*8, doppler_hz=-1700.), channel(prn=32, carrier_phase=.2)]
    rows, starts, seeds = [channels, [], channels[:1]], [52000, 0, 203000], [31, 2**64-1, 19]
    original_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(threads)
        actual = simulator.generate_epochs(rows, 2600, noise_seeds=seeds, start_samples=starts)
        expected = np.stack([CpuSignalSimulator(noise_floor_db=noise_db, noise_seed=seed).generate_epoch(
            row, 2600, start_sample=start) for row, start, seed in zip(rows, starts, seeds)])
    finally:
        numba.set_num_threads(original_threads)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("starts", [[0], [0, -1], [0, .5], [0, True], [0, 2**53-1]])
def test_invalid_offset_batches_are_rejected_before_indexing(starts):
    with pytest.raises(ValueError):
        CpuSignalSimulator().generate_epochs([[channel()], [channel()]], 64, start_samples=starts)


def test_offset_batch_checks_global_navigation_coverage_and_empty_batch_lengths():
    with pytest.raises(ValueError, match="cover"):
        CpuSignalSimulator().generate_epochs([[channel(nav_bits=[1])]]*2, 2600, start_samples=[0, 52000])
    with pytest.raises(ValueError, match="one value"):
        CpuSignalSimulator().generate_epochs([], 32, start_samples=[0])
    assert CpuSignalSimulator().generate_epochs([], 32, start_samples=[]).shape == (0, 64)


def test_generated_iq_recovers_known_code_delay_with_independent_fft_acquisition():
    fs = 2 * CHIP_RATE
    sim = CpuSignalSimulator(fs, noise_floor_db=-10000)
    # Positive propagation delay retards both code and carrier phase.
    delay_samples = 37
    delay_s = delay_samples / fs
    path = channel(code_phase=-CHIP_RATE*delay_s,
                   carrier_phase=(-2*np.pi*L1_HZ*delay_s) % (2*np.pi))
    iq = sim.generate_epoch([path], 2046).reshape(-1, 2)
    signal = iq[:, 0] + 1j*iq[:, 1]
    local = np.repeat(generate_ca_code(1), 2)
    correlation = np.abs(np.fft.ifft(np.fft.fft(signal)*np.conj(np.fft.fft(local))))
    assert np.argmax(correlation) == delay_samples
    assert correlation[delay_samples] > 2000


def test_reflected_path_is_coherent_sum_of_delayed_replicas():
    sim = CpuSignalSimulator(noise_floor_db=-10000)
    direct = channel(carrier_phase=.3, code_phase=5.2, doppler_hz=900.)
    delay = 80. / 299792458.
    reflected = dict(direct, amplitude=.4,
                     code_phase=direct["code_phase"]-CHIP_RATE*delay,
                     carrier_phase=direct["carrier_phase"]-2*np.pi*L1_HZ*delay)
    a = sim.generate_epoch([direct], 8192)
    b = sim.generate_epoch([reflected], 8192)
    combined = sim.generate_epoch([direct, reflected], 8192)
    np.testing.assert_allclose(combined, a+b, atol=1.3e-7, rtol=1e-6)


@pytest.mark.parametrize("overrides,match", [
    ({"system": 2}, "GPS"), ({"prn": 33}, "prn"), ({"nav_bit": 0}, "nav_bit"),
    ({"nav_bits": [1, 0]}, "nav_bits"), ({"nav_bits": [[1]]}, "nav_bits"),
    ({"nav_bits": [np.nan]}, "nav_bits"), ({"nav_bits": [.5]}, "nav_bits"),
    ({"nav_bits": np.empty((0, 1))}, "nav_bits"),
    ({"nav_bits": [1], "nav_bit_rate": 1e9}, "cover"),
])
def test_invalid_channel_is_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        CpuSignalSimulator().generate_epoch([channel(**overrides)], 100)


def test_explicit_empty_navigation_sequence_holds_constant_bit():
    sim = CpuSignalSimulator(noise_floor_db=-10000)
    for empty in ([], np.array([], dtype=np.int8)):
        np.testing.assert_array_equal(
            sim.generate_epoch([channel(nav_bit=-1, nav_bits=empty)], 2046),
            sim.generate_epoch([channel(nav_bit=-1)], 2046))


def test_invalid_intervals_and_seeds():
    sim = CpuSignalSimulator()
    for start in (-1, .5, float("inf"), 2**53):
        with pytest.raises(ValueError):
            sim.generate_epoch([channel()], 32, start_sample=start)
    with pytest.raises(ValueError):
        CpuSignalSimulator(noise_seed=-1)
    with pytest.raises(ValueError):
        sim.generate_epochs([[channel()]], 32, noise_seeds=[])


@pytest.mark.parametrize("sim,channels", [
    (CpuSignalSimulator(noise_floor_db=800), [channel()]),
    (CpuSignalSimulator(), [channel(code_phase=1e100)]),
    (CpuSignalSimulator(sampling_freq=1e-320), [channel()]),
])
def test_numeric_overflow_is_rejected_before_jit_indexing(sim, channels):
    with pytest.raises(ValueError, match="range"):
        sim.generate_epoch(channels, 1)


def test_urban_simulator_produces_real_iq_on_cpu():
    sim = UrbanSignalSimulator(signal_backend="cpu", noise_floor_db=-10000)
    rx = np.array([6378137., 0., 0.])
    result = sim.compute_epoch(rx, np.array([[26378137., 0., 0.]]),
                               prn_list=[1], n_samples=64)
    assert result["is_los"][0]
    expected = sim.sim.generate_epoch(result["channels"], 64)
    np.testing.assert_array_equal(result["iq"], expected)
    np.testing.assert_allclose(np.linalg.norm(result["iq"].reshape(-1, 2), axis=1), 1., atol=1e-7)
    with pytest.raises(ValueError, match="signal_backend"):
        UrbanSignalSimulator(signal_backend="auto")
