"""Reference attenuation anchors, wet-film boundary limits and IQ conventions."""

import numpy as np
import pytest

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.fresnel import fresnel_coefficients, complex_permittivity
from gnss_gpu.rain_cpu import (
    RainLayer, circular_rain_coefficients, rain_specific_attenuation,
    segment_layer_length, water_film_fresnel,
)
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.wall_signal_sim import VerticalWall, simulate_wall_signal


@pytest.mark.parametrize("frequency,k,alpha", [
    (1e9, 2.8345032965178754e-5, .9093953660534237),
    (1.57542e9, 5.6009072377142174e-5, .9567177715408957),
    (10e9, .011729429146503366, 1.2371441004955788),
    (20e9, .09387693776663214, 1.0198776311671574)])
def test_p838_3_circular_coefficients_match_version_pinned_external_reference(frequency, k, alpha):
    assert circular_rain_coefficients(frequency) == pytest.approx((k, alpha), rel=1e-12)


def test_l1_loss_units_and_zero_rain():
    assert rain_specific_attenuation(100.) == pytest.approx(.004588744986528188)
    assert RainLayer(100., 3000.).loss_db([0, 0, 0], [0, 0, 20000000]) == pytest.approx(3*.004588744986528188)
    assert rain_specific_attenuation(0.) == 0
    assert RainLayer(0.).loss_db([0, 0, 0], [0, 0, 20000000]) == 0


def test_segment_slab_clipping_is_symmetric_and_handles_horizontal_and_zero_depth():
    a = [[0, 0, -10], [0, 0, 5], [0, 0, 11]]
    b = [[0, 0, 20], [3, 4, 5], [3, 4, 11]]
    expected = [10., 5., 0.]
    np.testing.assert_allclose(segment_layer_length(a, b, 10.), expected)
    np.testing.assert_allclose(segment_layer_length(b, a, 10.), expected)
    np.testing.assert_array_equal(segment_layer_length(a, b, 0.), [0, 0, 0])


def test_film_zero_depth_and_equal_substrate_match_half_space_fresnel():
    angles = np.linspace(0., np.pi/2, 30)
    expected = fresnel_coefficients(angles, complex_permittivity("concrete"))
    for got, reference in zip(water_film_fresnel(angles), expected):
        np.testing.assert_allclose(got, reference, atol=2e-14)
    for got, reference in zip(water_film_fresnel(angles, water_permittivity=complex_permittivity("concrete"),
                                               thickness_m=.001), expected):
        np.testing.assert_allclose(got, reference, atol=2e-14)


def test_thick_lossy_film_approaches_water_half_space_and_is_passive():
    angles = np.linspace(0., np.pi/2, 30)
    for got, expected in zip(water_film_fresnel(angles, thickness_m=10.), fresnel_coefficients(angles, 80-8j)):
        np.testing.assert_allclose(got, expected, atol=1e-14)
    for coefficient in water_film_fresnel(angles, thickness_m=.001):
        assert np.all(np.abs(coefficient) <= 1+1e-14)


def block(*, rain=None, thickness=0., reflection=False, noise=-10000):
    return simulate_wall_signal(
        CpuSignalSimulator(2*CHIP_RATE, noise_floor_db=noise, noise_seed=71),
        VerticalWall(0., -100., 100., 0., 100., water_film_thickness_m=thickness),
        [-10., 0., 2.], [0., 0., 0.], [-100., 0., 2.], 2046,
        reference_range_m=90., include_reflection=reflection, rain_layer=rain,
        integrate_samples=True)


def test_rain_scales_voltage_by_half_power_db_and_only_for_path_inside_layer():
    dry = block()
    rain = block(rain=RainLayer(100., 10.))
    attenuation = .09*rain_specific_attenuation(100.)
    np.testing.assert_allclose(rain.direct_rain_loss_db, attenuation)
    np.testing.assert_allclose(rain.reflected_rain_loss_db, .11*rain_specific_attenuation(100.))
    np.testing.assert_allclose(rain.iq, dry.iq*10**(-attenuation/20), atol=1e-7)
    np.testing.assert_array_equal(block(rain=RainLayer(100., 1.)).iq, dry.iq)
    np.testing.assert_array_equal(block(rain=RainLayer(0.)).iq, dry.iq)


def test_wet_wall_changes_reflected_wave_only_and_receiver_noise_is_not_attenuated():
    np.testing.assert_array_equal(block(thickness=.001).iq, block().iq)
    assert not np.array_equal(block(thickness=.001, reflection=True).iq, block(reflection=True).iq)
    dry_noise = block(noise=-20).iq-block().iq
    rain_noise = block(rain=RainLayer(100., 10.), noise=-20).iq-block(rain=RainLayer(100., 10.)).iq
    # Subtracting two independently rounded float32 sums can differ by two ulps.
    np.testing.assert_allclose(rain_noise, dry_noise, atol=2*np.finfo(np.float32).eps)


def test_air_film_replacing_air_does_not_change_wall_iq_phase():
    simulator = CpuSignalSimulator(2*CHIP_RATE, noise_floor_db=-10000)
    kwargs = dict(rx_position_m=[-10., 0., 2.], rx_velocity_mps=[0., 0., 0.],
                  transmitter_m=[-100., 20., 20.], n_samples=2046,
                  reference_range_m=90., integrate_samples=True)
    base = simulate_wall_signal(simulator, VerticalWall(0., -100., 100., 0., 100.), **kwargs)
    air = simulate_wall_signal(simulator, VerticalWall(0., -100., 100., 0., 100.,
                               water_film_thickness_m=.001, water_permittivity=1+0j), **kwargs)
    np.testing.assert_allclose(air.iq, base.iq, atol=1e-7)


@pytest.mark.parametrize("function", [lambda: rain_specific_attenuation(-1),
    lambda: circular_rain_coefficients(1e8), lambda: RainLayer(1, 0, 1),
    lambda: RainLayer(np.nan), lambda: water_film_fresnel(0, thickness_m=-1),
    lambda: water_film_fresnel(0, water_permittivity=80+8j),
    lambda: water_film_fresnel(np.nan), lambda: VerticalWall(0, -1, 1, 0, 1, water_film_thickness_m=-1)])
def test_invalid_weather_parameters_are_rejected(function):
    with pytest.raises(ValueError):
        function()
