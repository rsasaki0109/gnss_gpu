"""Independent fractional-delay calibration and receiver-aperture checks."""

import numpy as np
import pytest

pytest.importorskip("numba")

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.code_delay_cpu import estimate_ca_delay
from gnss_gpu.signal_sim_cpu import CHIP_RATE, L1_HZ
from gnss_gpu.validation.tracking_residuals import generate_ca_code
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal


def signal(delay_samples, samples_per_chip=20, count=None, **kwargs):
    fs = samples_per_chip*CHIP_RATE
    sim = CpuSignalSimulator(fs, noise_floor_db=-10000, intermediate_freq=kwargs.pop("intermediate_freq", 0.))
    rx, sat = np.array([-10., 0., 2.]), np.array([-100., 0., 2.])
    reference = 90-delay_samples*C_LIGHT/fs
    return simulate_wall_signal(sim, VerticalWall(0., -100., 100., 0., 100.),
                                rx, [0., 0., 0.], sat, count or 1023*samples_per_chip,
                                reference_range_m=reference, include_reflection=False,
                                integrate_samples=True, **kwargs)


@pytest.mark.parametrize("samples_per_chip", [2, 20])
@pytest.mark.parametrize("delay", [-12.49, -.5, -.23, 0., .17, .5, 12.37, 1022.9])
def test_fractional_delay_fit_recovers_known_physical_delay(delay, samples_per_chip):
    block = signal(delay, samples_per_chip)
    estimate = estimate_ca_delay(block.iq, samples_per_chip*CHIP_RATE)
    assert estimate is not None
    assert estimate.delay_samples == pytest.approx(delay, abs=2e-6)
    assert estimate.fit_residual_fraction < 1e-10


def test_sampling_aperture_matches_independent_dense_rf_integration():
    fs, delay, count, frequency = 2*CHIP_RATE, .33, 96, 200000.
    block = signal(delay, 2, count=count, intermediate_freq=frequency)
    subdivisions = 4096
    t = (np.arange(count)[:, None]+(np.arange(subdivisions)+.5)/subdivisions-.5)/fs
    tau = delay/fs
    chips = np.floor((t-tau)*CHIP_RATE).astype(int) % 1023
    fine = generate_ca_code(1)[chips]*np.exp(2j*np.pi*(frequency*t-L1_HZ*tau))
    expected = fine.mean(axis=1)
    observed = block.iq.reshape(-1, 2)
    np.testing.assert_allclose(observed[:, 0]+1j*observed[:, 1], expected, atol=5e-4)


def test_aperture_resolves_delays_that_instantaneous_samples_cannot_distinguish():
    a, b = signal(.17), signal(.37)
    # A constant unknown carrier phase must not supply the true delay.
    for block in (a, b):
        z = block.iq.reshape(-1, 2)
        complex_iq = (z[:, 0]+1j*z[:, 1])*np.exp(1j*.91)*.3
        z[:, 0], z[:, 1] = complex_iq.real, complex_iq.imag
    x = estimate_ca_delay(a.iq, 20*CHIP_RATE)
    y = estimate_ca_delay(b.iq, 20*CHIP_RATE)
    assert x is not None and y is not None
    assert y.delay_samples-x.delay_samples == pytest.approx(.2, abs=2e-6)


def test_seeded_noise_keeps_delay_estimate_close_and_increases_fit_residual():
    block = signal(.31)
    rng = np.random.default_rng(19)
    noisy = block.iq+rng.normal(0., .03, block.iq.shape)
    estimate = estimate_ca_delay(noisy, 20*CHIP_RATE)
    assert estimate is not None
    assert abs(estimate.delay_m-.31*C_LIGHT/(20*CHIP_RATE)) < .15
    assert 0 < estimate.fit_residual_fraction < .01


def test_zero_signal_has_no_estimate():
    assert estimate_ca_delay(np.zeros(4092), 2*CHIP_RATE) is None


def test_delay_is_wrapped_modulo_one_code_period():
    block = signal(2045.7, 2)
    estimate = estimate_ca_delay(block.iq, 2*CHIP_RATE)
    assert estimate is not None
    assert estimate.delay_samples == pytest.approx(-.3, abs=2e-6)


def test_supplied_low_carrier_is_wiped_before_code_delay_fit():
    block = signal(.17, intermediate_freq=500.)
    estimate = estimate_ca_delay(block.iq, 20*CHIP_RATE, carrier_hz=500.)
    assert estimate is not None
    assert estimate.delay_samples == pytest.approx(.17, abs=1e-4)


@pytest.mark.parametrize("prn", [None, np.nan, 33, True])
def test_invalid_prn_is_rejected(prn):
    with pytest.raises(ValueError):
        estimate_ca_delay(np.zeros(4092), 2*CHIP_RATE, prn=prn)


@pytest.mark.parametrize("fs,iq", [(CHIP_RATE, np.zeros(2046)),
                                      (2.6e6, np.zeros(5200)),
                                      (2*CHIP_RATE, np.zeros(4091)),
                                      (2*CHIP_RATE, np.full(4092, np.nan))])
def test_bad_sampling_or_iq_is_rejected(fs, iq):
    with pytest.raises(ValueError):
        estimate_ca_delay(iq, fs)


def test_integrated_moving_blocks_preserve_continuity():
    fs = 2*CHIP_RATE
    kwargs = dict(simulator=CpuSignalSimulator(fs, noise_floor_db=-10000),
                  wall=VerticalWall(0., -100., 100., 0., 100.),
                  rx_position_m=[-10., -3., 2.], rx_velocity_mps=[3., 7., 0.],
                  transmitter_m=[-20., 9., 12.], integrate_samples=True,
                  reference_range_m=20.)
    whole = simulate_wall_signal(n_samples=4096, **kwargs)
    first = simulate_wall_signal(n_samples=1025, **kwargs)
    second = simulate_wall_signal(n_samples=3071, start_sample=1025, **kwargs)
    np.testing.assert_array_equal(whole.iq, np.concatenate([first.iq, second.iq]))
