"""Geometry-driven range, range-rate and IQ Doppler with two moving bodies."""

import numpy as np
import pytest

pytest.importorskip("numba")

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.signal_sim_cpu import CHIP_RATE, L1_HZ
from gnss_gpu.validation.tracking_residuals import generate_ca_code
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal


def scenario(**kwargs):
    args = dict(simulator=CpuSignalSimulator(2*CHIP_RATE, noise_floor_db=-10000),
                wall=VerticalWall(0., -100., 100., 0., 100., antenna_lhcp_response=.1),
                rx_position_m=[-20., 0., 2.], rx_velocity_mps=[5., 0., 0.],
                transmitter_m=[-20000020., 0., 2.], transmitter_velocity_mps=[1000., 0., 0.],
                rx_acceleration_mps2=[200., 0., 0.], transmitter_acceleration_mps2=[2., 0., 0.],
                reference_range_m=20000000., n_samples=4096)
    args.update(kwargs)
    return simulate_wall_signal(**args)


def test_radial_geometry_matches_paper_relative_velocity_doppler():
    start = 20000
    result = scenario(start_sample=start)
    t = (start+np.arange(4096))/(2*CHIP_RATE)
    # In this radial scene: R0=20 Mm, v_rx-v_sat=-995, a_rx-a_sat=198.
    expected_range = 20000000.-995*t+99*t*t
    expected_rate = -995+198*t
    np.testing.assert_allclose(result.direct_range_m, expected_range, atol=1e-8, rtol=0)
    np.testing.assert_allclose(result.direct_range_rate_mps, expected_rate, atol=1e-12)
    np.testing.assert_allclose(result.direct_doppler_hz, -L1_HZ*expected_rate/C_LIGHT, atol=1e-10)
    # Image transmitter velocity reverses its normal component.
    np.testing.assert_allclose(result.reflected_range_rate_mps, -1005-202*t, atol=1e-12)


def test_iq_doppler_is_recovered_without_true_code_or_carrier_phase():
    fs = 2*CHIP_RATE
    result = scenario(include_reflection=False, start_sample=200000)
    pairs = result.iq.reshape(-1, 2).astype(np.float64)
    z = pairs[:, 0]+1j*pairs[:, 1]
    # Squaring the adjacent complex product removes unknown BPSK polarity.
    measured = np.angle((z[1:]*np.conj(z[:-1]))**2)*fs/(4*np.pi)
    expected = .5*(result.direct_doppler_hz[1:]+result.direct_doppler_hz[:-1])
    assert np.median(measured-expected) == pytest.approx(0., abs=.01)


def test_moving_satellite_code_and_carrier_use_the_same_geometric_delay():
    fs, count = 2*CHIP_RATE, 4096
    result = scenario(include_reflection=False)
    t = np.arange(count)/fs
    true_delay = (-995*t+99*t*t)/C_LIGHT
    expected_chip = np.floor((t-true_delay)*CHIP_RATE).astype(int) % 1023
    data = result.iq.reshape(-1, 2)
    baseband = data[:, 0]+1j*data[:, 1]
    stripped = baseband*np.exp(2j*np.pi*L1_HZ*true_delay)
    np.testing.assert_array_equal(np.where(stripped.real >= 0, 1., -1.), generate_ca_code(1)[expected_chip])


@pytest.mark.parametrize("integrate", [False, True])
def test_accelerating_bodies_preserve_exact_block_continuity(integrate):
    whole = scenario(n_samples=8192, integrate_samples=integrate)
    a = scenario(n_samples=3001, integrate_samples=integrate)
    b = scenario(n_samples=5191, start_sample=3001, integrate_samples=integrate)
    np.testing.assert_array_equal(np.concatenate([a.iq, b.iq]), whole.iq)
    np.testing.assert_array_equal(np.concatenate([a.direct_doppler_hz, b.direct_doppler_hz]), whole.direct_doppler_hz)


def test_oblique_path_range_rates_match_independent_finite_differences():
    rx, vr, ar = np.array([-20., -20., 1.5]), np.array([1., 5., 0.]), np.array([2., 3., 1.])
    sat, vs, ass = np.array([-1e7, 1e7, 1e7]), np.array([1200., 2500., -500.]), np.array([-.1, .2, -.3])
    fs, start, step = 2*CHIP_RATE, 100000, .01
    t = start/fs

    def ranges(time):
        receiver = rx+vr*time+.5*ar*time*time
        transmitter = sat+vs*time+.5*ass*time*time
        image = transmitter.copy()
        image[0] *= -1
        return np.array([np.linalg.norm(transmitter-receiver), np.linalg.norm(image-receiver)])

    expected = (ranges(t+step)-ranges(t-step))/(2*step)
    result = scenario(rx_position_m=rx, rx_velocity_mps=vr, rx_acceleration_mps2=ar,
                      transmitter_m=sat, transmitter_velocity_mps=vs,
                      transmitter_acceleration_mps2=ass, start_sample=start, n_samples=1)
    np.testing.assert_allclose([result.direct_range_rate_mps[0], result.reflected_range_rate_mps[0]], expected, atol=1e-5)


@pytest.mark.parametrize("kwargs", [dict(transmitter_velocity_mps=[np.nan, 0., 0.]),
                                       dict(rx_acceleration_mps2=[1., 2.]),
                                       dict(transmitter_velocity_mps=[C_LIGHT, 0., 0.]),
                                       dict(rx_acceleration_mps2=[C_LIGHT, 0., 0.], start_sample=2*2046000)])
def test_invalid_motion_is_rejected(kwargs):
    with pytest.raises(ValueError):
        scenario(**kwargs)
