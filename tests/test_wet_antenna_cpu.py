"""Independent antenna controls, response derivatives and continuous IQ."""

import numpy as np
import pytest

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.code_delay_cpu import estimate_ca_delay
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal
from gnss_gpu.wet_antenna_cpu import WetAntennaResponse


def test_wetting_plateau_drying_endpoints_and_smooth_rates():
    w, rate = WetAntennaResponse().wetness([0., .05, .075, .1, .2, .3, .4, .5])
    np.testing.assert_allclose(w, [0, 0, .5, 1, 1, .5, 0, 0], atol=1e-14)
    np.testing.assert_allclose(rate, [0, 0, 30, 0, 0, -7.5, 0, 0], atol=1e-13)


def test_time_and_direction_derivatives_match_finite_difference():
    model = WetAntennaResponse(code_bias_m=.3, elevation_code_bias_m=.5,
                               phase_bias_rad=.7, elevation_phase_bias_rad=.2)
    ray, velocity = np.array([2., 3., 4.]), np.array([1., 2., -1.])
    t, h = .075, 1e-6
    center = model.sample(t, ray, velocity)
    a, b = model.sample(t-h, ray-h*velocity), model.sample(t+h, ray+h*velocity)
    assert center.code_bias_rate_mps == pytest.approx((b.code_bias_m-a.code_bias_m)/(2*h), abs=1e-7)
    assert center.phase_rate_rad_s == pytest.approx((b.phase_bias_rad-a.phase_bias_rad)/(2*h), abs=1e-7)


def signal(response=None, *, start_s=.12, count=20460, integrate=True, noise=-10000):
    fs = 20*CHIP_RATE
    return simulate_wall_signal(
        CpuSignalSimulator(fs, noise_floor_db=noise, noise_seed=71),
        VerticalWall(0., -100., 100., 0., 100.), [-10., 0., 2.], [0., 0., 0.], [-100., 0., 2.],
        count, start_sample=round(start_s*fs), reference_range_m=90-.17*C_LIGHT/fs,
        include_reflection=False, integrate_samples=integrate, antenna_response=response)


def test_loss_is_voltage_db_and_receiver_noise_is_added_after_antenna():
    response = WetAntennaResponse(loss_db=6.)
    dry, wet = signal(), signal(response)
    np.testing.assert_allclose(wet.iq, dry.iq*10**(-6/20), atol=1e-7)
    noise0, noise1 = signal(noise=-20).iq-dry.iq, signal(response, noise=-20).iq-wet.iq
    np.testing.assert_allclose(noise0, noise1, atol=2*np.finfo(np.float32).eps)


def test_code_delay_does_not_impose_rf_phase_and_phase_does_not_impose_code_delay():
    dry = signal()
    code = signal(WetAntennaResponse(code_bias_m=.3))
    phase = signal(WetAntennaResponse(phase_bias_rad=.7))
    fs = 20*CHIP_RATE
    a, b, c = [estimate_ca_delay(block.iq, fs) for block in (dry, code, phase)]
    assert b.delay_m-a.delay_m == pytest.approx(.3, abs=2e-5)
    assert c.delay_m-a.delay_m == pytest.approx(0., abs=2e-5)
    def at_center(block):
        return complex(*block.iq.reshape(-1, 2)[10])
    assert at_center(code)/at_center(dry) == pytest.approx(1., abs=1e-6)
    assert at_center(phase)/at_center(dry) == pytest.approx(np.exp(.7j), abs=1e-6)


def test_time_varying_phase_changes_doppler_while_code_bias_rate_is_independent():
    fs = 20*CHIP_RATE
    response = WetAntennaResponse(phase_bias_rad=1., code_bias_m=.3)
    block = signal(response, start_s=.075, count=4096, integrate=False)
    pairs = block.iq.reshape(-1, 2).astype(float)
    z = pairs[:, 0]+1j*pairs[:, 1]
    # A longer lag resolves a few Hz at 20 MHz without adjacent-float32 phase
    # quantization dominating the diagnostic; squaring still removes BPSK.
    lag = 64
    measured = np.angle((z[lag:]*np.conj(z[:-lag]))**2)*fs/(4*np.pi*lag)
    expected = .5*(block.direct_doppler_hz[lag:]+block.direct_doppler_hz[:-lag])
    assert np.median(measured-expected) == pytest.approx(0., abs=.002)
    assert np.median(block.direct_antenna_code_rate_mps) > 8.9
    code_only = signal(WetAntennaResponse(code_bias_m=.3), start_s=.075, count=100)
    np.testing.assert_array_equal(code_only.direct_doppler_hz, 0.)


@pytest.mark.parametrize("integrate", [False, True])
def test_zero_response_is_exact_noop_and_blocks_stay_contiguous(integrate):
    np.testing.assert_array_equal(signal(count=4096).iq, signal(WetAntennaResponse(), count=4096).iq)
    response = WetAntennaResponse(loss_db=6., code_bias_m=.3, phase_bias_rad=.7)
    fs, start = 20*CHIP_RATE, .075
    full = signal(response, start_s=start, count=4096, integrate=integrate)
    first = signal(response, start_s=start, count=1555, integrate=integrate)
    second = signal(response, start_s=start+1555/fs, count=2541, integrate=integrate)
    np.testing.assert_array_equal(full.iq, np.concatenate([first.iq, second.iq]))
    np.testing.assert_array_equal(full.direct_antenna_phase_rad,
                                  np.concatenate([first.direct_antenna_phase_rad, second.direct_antenna_phase_rad]))


def test_reflected_ray_uses_its_own_arrival_direction():
    model = WetAntennaResponse(elevation_code_bias_m=.3, elevation_phase_bias_rad=.7)
    fs = 2*CHIP_RATE
    block = simulate_wall_signal(CpuSignalSimulator(fs, noise_floor_db=-10000),
                                VerticalWall(0., -100., 100., 0., 100.),
                                [-20., 0., 1.5], [0., 0., 0.], [-100., 50., 50.], 10,
                                start_sample=round(.12*fs), antenna_response=model)
    assert np.all(block.reflection_visible)
    assert not np.allclose(block.direct_antenna_code_bias_m, block.reflected_antenna_code_bias_m)
    assert not np.allclose(block.direct_antenna_phase_rad, block.reflected_antenna_phase_rad)


@pytest.mark.parametrize("kwargs", [dict(loss_db=-1), dict(wetting_s=0), dict(drying_s=0),
                                      dict(wet_start_s=-1), dict(drying_start_s=.06),
                                      dict(phase_bias_rad=np.nan), dict(code_bias_m=np.inf)])
def test_invalid_profiles_are_rejected(kwargs):
    with pytest.raises(ValueError):
        WetAntennaResponse(**kwargs)


def test_invalid_samples_are_rejected():
    model = WetAntennaResponse()
    with pytest.raises(ValueError):
        model.sample(0., [0., 0., 0.])
    with pytest.raises(ValueError):
        model.sample(np.nan, [1., 0., 0.])
