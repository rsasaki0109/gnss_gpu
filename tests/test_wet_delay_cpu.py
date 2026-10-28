"""Flat wet-delay mapping, path derivatives and consistent code/carrier IQ."""

import numpy as np
import pytest

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.code_delay_cpu import estimate_ca_delay
from gnss_gpu.signal_sim_cpu import CHIP_RATE, L1_HZ
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal
from gnss_gpu.wet_delay_cpu import WetDelayResidualLayer


@pytest.mark.parametrize("elevation", [15., 30., 60., 90.])
def test_full_flat_slab_gives_cosecant_mapping_and_signed_residual(elevation):
    angle = np.deg2rad(elevation)
    end = 22e6*np.array([np.cos(angle), 0., np.sin(angle)])
    for residual in (-.3, 0., .3):
        value, rate = WetDelayResidualLayer(residual).excess_range_and_rate([0, 0, 0], end)
        assert value == pytest.approx(residual/np.sin(angle), abs=1e-12)
        assert rate == 0


def test_finite_layer_only_and_reciprocity():
    layer = WetDelayResidualLayer(.3, 3000.)
    a, b = [0, 0, 10], [30, 40, 10]
    assert layer.excess_range_and_rate(a, b)[0] == pytest.approx(.005)
    assert layer.excess_range_and_rate(b, a)[0] == pytest.approx(.005)
    assert layer.excess_range_and_rate([0, 0, 4000], [50, 0, 4000])[0] == 0


def test_oblique_delay_rate_matches_independent_central_finite_difference():
    layer = WetDelayResidualLayer(.3)
    a, b = np.array([-20., 3., 1.5]), np.array([-1e7, 1e7, 2e7])
    va, vb = np.array([3., 7., 1.]), np.array([1300., -2300., 800.])
    delay, rate = layer.excess_range_and_rate(a, b, va, vb)
    h = .01
    plus = layer.excess_range_and_rate(a+h*va, b+h*vb)[0]
    minus = layer.excess_range_and_rate(a-h*va, b-h*vb)[0]
    assert rate == pytest.approx((plus-minus)/(2*h), abs=1e-11)
    assert delay > 0


def scene(*, layer=None, start=0, count=20460, integrate=True, velocity=(0., 0., 0.), fs=20*CHIP_RATE):
    return simulate_wall_signal(
        CpuSignalSimulator(fs, noise_floor_db=-10000), VerticalWall(0., -100., 100., 0., 100.),
        [-10., 0., 2.], velocity, [-100., 0., 2.], count,
        reference_range_m=90.-.17*C_LIGHT/fs, include_reflection=False,
        start_sample=start, integrate_samples=integrate, wet_delay_layer=layer)


def test_constant_wet_delay_retards_code_and_carrier_with_same_positive_sign():
    dry, wet = scene(), scene(layer=WetDelayResidualLayer(.1, 100.))
    a, b = estimate_ca_delay(dry.iq, 20*CHIP_RATE), estimate_ca_delay(wet.iq, 20*CHIP_RATE)
    assert b.delay_m-a.delay_m == pytest.approx(.09, abs=2e-5)
    z0, z1 = dry.iq.reshape(-1, 2)[10], wet.iq.reshape(-1, 2)[10]
    ratio = complex(*z1)/complex(*z0)
    assert ratio == pytest.approx(np.exp(-2j*np.pi*L1_HZ*.09/C_LIGHT), abs=1e-6)
    np.testing.assert_array_equal(dry.direct_range_m, wet.direct_range_m)
    np.testing.assert_allclose(wet.direct_wet_delay_m, .09)
    np.testing.assert_allclose(wet.direct_propagation_range_m-dry.direct_range_m, .09, atol=1e-14)


def test_moving_code_carrier_and_doppler_include_wet_delay_derivative():
    fs = 2*CHIP_RATE
    wet = scene(layer=WetDelayResidualLayer(.3, 100.), velocity=(5., 0., 0.),
                fs=fs, count=4096, integrate=False)
    pairs = wet.iq.reshape(-1, 2).astype(float)
    z = pairs[:, 0]+1j*pairs[:, 1]
    measured = np.angle((z[1:]*np.conj(z[:-1]))**2)*fs/(4*np.pi)
    expected = .5*(wet.direct_doppler_hz[1:]+wet.direct_doppler_hz[:-1])
    assert np.median(measured-expected) == pytest.approx(0., abs=.001)
    assert abs(float(np.median(L1_HZ*wet.direct_wet_delay_rate_mps/C_LIGHT))) > .05


@pytest.mark.parametrize("integrate", [False, True])
def test_zero_layer_is_exact_noop_and_wet_blocks_are_contiguous(integrate):
    args = dict(fs=2*CHIP_RATE, count=4096, velocity=(3., 0., 0.), integrate=integrate)
    np.testing.assert_array_equal(scene(**args).iq, scene(layer=WetDelayResidualLayer(0.), **args).iq)
    layer = WetDelayResidualLayer(.3, 100.)
    full = scene(layer=layer, **args)
    first = scene(layer=layer, **dict(args, count=1555))
    second = scene(layer=layer, start=1555, **dict(args, count=2541))
    np.testing.assert_array_equal(full.iq, np.concatenate([first.iq, second.iq]))
    np.testing.assert_array_equal(full.direct_wet_delay_m, np.concatenate([first.direct_wet_delay_m, second.direct_wet_delay_m]))


def test_reflected_two_leg_rate_matches_finite_difference_of_moving_intersection():
    fs, t, h = 2*CHIP_RATE, .1, .01
    args = dict(simulator=CpuSignalSimulator(fs, noise_floor_db=-10000),
                wall=VerticalWall(0., -100., 100., 0., 100.), rx_position_m=[-20., -20., 1.5],
                rx_velocity_mps=[1., 5., .1], transmitter_m=[-1e7, 1e7, 1e7],
                transmitter_velocity_mps=[1000., -2000., 500.], n_samples=1,
                wet_delay_layer=WetDelayResidualLayer(.3), reference_range_m=17e6)
    middle = simulate_wall_signal(start_sample=round(t*fs), **args)
    plus = simulate_wall_signal(start_sample=round((t+h)*fs), **args)
    minus = simulate_wall_signal(start_sample=round((t-h)*fs), **args)
    assert middle.reflection_visible[0]
    assert middle.reflected_wet_delay_rate_mps[0] == pytest.approx(
        (plus.reflected_wet_delay_m[0]-minus.reflected_wet_delay_m[0])/(2*h), abs=1e-10)


@pytest.mark.parametrize("kwargs", [dict(zenith_residual_m=np.nan), dict(top_m=0),
                                      dict(bottom_m=4000), dict(top_m=np.inf)])
def test_invalid_layers_are_rejected(kwargs):
    with pytest.raises(ValueError):
        WetDelayResidualLayer(**kwargs)
