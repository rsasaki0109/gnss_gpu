"""Analytic image geometry, moving carrier phase and acquired code delays."""

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.fresnel import complex_permittivity
from gnss_gpu.signal_sim_cpu import CHIP_RATE, L1_HZ
from gnss_gpu.validation.tracking_residuals import generate_ca_code
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal


def wall(**kwargs):
    return VerticalWall(0., -100., 100., 0., 100., **kwargs)


def simulate(rx=(-10., -3., 2.), sat=(-20., 9., 12.), velocity=(0., 0., 0.),
             count=4096, surface=None, simulator=None, **kwargs):
    return simulate_wall_signal(simulator or CpuSignalSimulator(noise_floor_db=-10000),
                                surface or wall(), rx, velocity, sat, count, **kwargs)


def test_image_geometry_and_complex_fresnel_coefficient_match_analytic_solution():
    block = simulate()
    expected_point = np.array([0., 1., 16./3.])
    length = np.sqrt(30.**2+12.**2+10.**2)
    np.testing.assert_allclose(block.reflection_points_m, np.broadcast_to(expected_point, (4096, 3)), atol=1e-12)
    np.testing.assert_allclose(block.reflected_range_m, length, atol=1e-12)
    assert block.direct_visible.all() and block.reflection_visible.all()
    cosine = 30/length
    eps = complex_permittivity("concrete")
    root = np.sqrt(eps-(1-cosine*cosine))
    r_parallel = (eps*cosine-root)/(eps*cosine+root)
    r_perpendicular = (cosine-root)/(cosine+root)
    expected_coefficient = (r_parallel+r_perpendicular)/2
    np.testing.assert_allclose(block.reflection_coefficient, expected_coefficient, atol=1e-15)
    assert expected_coefficient.imag != 0


def test_static_geometry_waveform_matches_independent_complex_expression():
    fs = 2.6e6
    reference = 20.
    sim = CpuSignalSimulator(fs, noise_floor_db=-10000)
    block = simulate(simulator=sim, reference_range_m=reference)
    t = np.arange(4096)/fs
    expected = np.zeros(4096, dtype=complex)
    for path, gain in [(block.direct_range_m, np.ones(4096)),
                       (block.reflected_range_m, block.reflection_coefficient)]:
        delay = (path-reference)/C_LIGHT
        chips = np.floor((t-delay)*CHIP_RATE).astype(int) % 1023
        expected += (np.sqrt(10.**2+12.**2+10.**2)/path * gain
                     * generate_ca_code(1)[chips] * np.exp(-2j*np.pi*L1_HZ*delay))
    measured = block.iq.reshape(-1, 2)
    np.testing.assert_allclose(measured[:, 0], expected.real, atol=1e-7)
    np.testing.assert_allclose(measured[:, 1], expected.imag, atol=1e-7)


def test_opaque_wall_blocks_direct_path_but_does_not_invent_reflection():
    # The transmitter image coincides with the receiver; this is a valid shadow.
    block = simulate(rx=(-10., 0., 2.), sat=(10., 0., 2.))
    assert not block.direct_visible.any()
    assert not block.reflection_visible.any()
    np.testing.assert_array_equal(block.iq, 0)
    assert np.isnan(block.reflection_points_m).all()
    # Same side-crossing outside the finite rectangle stays unobstructed.
    clear = simulate(rx=(-10., 110., 2.), sat=(10., 110., 2.))
    assert clear.direct_visible.all()
    assert np.any(clear.iq)


def test_reflection_disappears_at_finite_wall_edge_during_motion():
    surface = VerticalWall(0., -10., .02, 0., 100.)
    block = simulate(rx=(-10., -10., 2.), sat=(-20., 20., 12.),
                     velocity=(0., 100., 0.), count=2600, surface=surface)
    # Mirror intersection y=(2/3)*rx_y + (1/3)*sat_y crosses y=.02.
    expected = (2/3)*(-10+100*np.arange(2600)/2.6e6)+(1/3)*20 <= .02
    np.testing.assert_array_equal(block.reflection_visible, expected)
    assert expected.any() and not expected.all()
    assert block.direct_visible.all()


def test_moving_receiver_blocks_match_whole_waveform_exactly():
    kwargs = dict(velocity=(3., 7., 0.), reference_range_m=20.)
    whole = simulate(count=16384, **kwargs)
    a = simulate(count=7001, **kwargs)
    b = simulate(count=9383, start_sample=7001, **kwargs)
    np.testing.assert_array_equal(np.concatenate([a.iq, b.iq]), whole.iq)
    np.testing.assert_array_equal(np.concatenate([a.reflected_range_m, b.reflected_range_m]),
                                  whole.reflected_range_m)


def test_moving_reflection_carrier_tracks_image_path_range_rate():
    fs, count = 2.6e6, 10000
    block = simulate(velocity=(3., 7., 0.), count=count, reference_range_m=20., include_direct=False)
    t = np.arange(count)/fs
    # Independent image range, for image transmitter (20,9,12).
    path = np.sqrt((30-3*t)**2+(12-7*t)**2+10.**2)
    np.testing.assert_allclose(block.reflected_range_m, path, atol=1e-13)
    chip = np.floor((t-(path-20)/C_LIGHT)*CHIP_RATE).astype(int) % 1023
    data = block.iq.reshape(-1, 2)
    # Strip C/A signs and material field phase before unwrapping the carrier.
    carrier = ((data[:, 0]+1j*data[:, 1]) * generate_ca_code(1)[chip]
               / block.reflection_coefficient)
    phase = np.unwrap(np.angle(carrier))
    observed_change = phase-phase[0]
    expected_change = -2*np.pi*(path-path[0])/(C_LIGHT/L1_HZ)
    np.testing.assert_allclose(observed_change, expected_change, atol=2e-7)


def test_reflection_only_fft_acquisition_measures_geometric_excess_delay():
    fs = 20*CHIP_RATE
    sim = CpuSignalSimulator(fs, noise_floor_db=-10000)
    rx, sat = np.array([-20., 0., 2.]), np.array([-2e7, 0., 2.])
    direct = np.linalg.norm(sat-rx)
    block = simulate(rx=rx, sat=sat, count=20460, simulator=sim,
                     reference_range_m=direct, include_direct=False,
                     surface=wall(polarization="perpendicular"))
    iq = block.iq.reshape(-1, 2)
    signal = iq[:, 0]+1j*iq[:, 1]
    local = np.repeat(generate_ca_code(1), 20)
    corr = np.abs(np.fft.ifft(np.fft.fft(signal)*np.conj(np.fft.fft(local))))
    delay_m = int(np.argmax(corr))*C_LIGHT/fs
    expected_m = block.reflected_range_m[0]-direct
    assert expected_m == pytest.approx(40.)
    assert abs(delay_m-expected_m) <= C_LIGHT/fs


def test_sample_generation_is_exact_across_numba_thread_counts():
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        a = simulate(velocity=(3., 7., 0.)).iq
        numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
        b = simulate(velocity=(3., 7., 0.)).iq
    finally:
        numba.set_num_threads(previous)
    np.testing.assert_array_equal(a, b)


def test_material_changes_reflection_phase_and_magnitude_without_changing_geometry():
    a = simulate(surface=wall(material="concrete"), include_direct=False)
    b = simulate(surface=wall(material="glass"), include_direct=False)
    np.testing.assert_array_equal(a.reflected_range_m, b.reflected_range_m)
    assert not np.array_equal(a.iq, b.iq)


@pytest.mark.parametrize("kwargs", [dict(x_m=np.nan), dict(y_min_m=100.)])
def test_invalid_wall_is_rejected(kwargs):
    values = dict(x_m=0., y_min_m=-100., y_max_m=100., z_min_m=0., z_max_m=100.)
    with pytest.raises(ValueError):
        VerticalWall(**dict(values, **kwargs))


def test_invalid_geometry_and_interval_are_rejected():
    with pytest.raises(ValueError, match="wall plane"):
        simulate(rx=(0., 0., 2.))
    with pytest.raises(ValueError):
        simulate(velocity=(0., np.nan, 0.))
    with pytest.raises(ValueError):
        simulate(start_sample=-1)
