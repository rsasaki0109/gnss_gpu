import numpy as np
import pytest
from gnss_gpu.accelerating_echo_cpu import accelerating_echo_blocks, minimum_excess_range
from gnss_gpu.moving_echo_cpu import moving_echo_blocks, range_rate_from_doppler, L1_FREQUENCY_HZ
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED
from experiments.run_cpu_accelerating_echo_pilot import synthesize
from experiments.run_cpu_moving_echo_pilot import synthesize as baseline


def test_independent_delay_integral_and_zero_acceleration():
    f = np.array([-7e6, 0., 7e6]); v = range_rate_from_doppler(.7)
    np.testing.assert_array_equal(accelerating_echo_blocks(f, .2, -.7, v, 0),
                                  moving_echo_blocks(f, .2, -.7, v))
    for a in (-.4, .4):
        actual = accelerating_echo_blocks(f, .2, -.7, v, a)
        for k in (0, 19, 39):
            t = k*.02+(np.arange(50000)+.5)*.02/50000
            delay = (v*t+.5*a*t*t)/LIGHT_SPEED
            phase = 2j*np.pi*(.2*t[:, None]-.35*t[:, None]**2)
            phase = phase-2j*np.pi*(L1_FREQUENCY_HZ+f)*delay[:, None]
            np.testing.assert_allclose(actual[k], np.mean(np.exp(phase), axis=0), rtol=1e-10, atol=1e-11)


def test_minimum_range_and_validation():
    # Positive endpoints can hide a nonphysical negative interior minimum.
    assert minimum_excess_range(.1, -1., 2., 1.) == pytest.approx(-.15)
    assert minimum_excess_range(.1, 1., -2., 1.) == pytest.approx(.1)
    for count in (40., True, 0, 41):
        with pytest.raises(ValueError):
            accelerating_echo_blocks([0.], 0, 0, 0, 0, count)
    with pytest.raises(ValueError):
        accelerating_echo_blocks([0.], 0, 0, 0, np.nan)
    with pytest.raises(ValueError):
        accelerating_echo_blocks([], 0, 0, 0, 0)
    with pytest.raises(ValueError):
        minimum_excess_range(1, 0, 0, -1)


def test_generator_zero_exact_and_shared_noise():
    b, s, t = synthesize((0., .5, -.3, 1350000))
    old = baseline(('moving_code', .7, .5, -.3, 1350000))
    for x, y in zip((b, s, t), old):
        np.testing.assert_array_equal(x, y)
    for rate in (-2., 2.):
        other, changed, truth = synthesize((rate, .5, -.3, 1350000))
        assert t == truth
        np.testing.assert_allclose(other-changed, b-s, rtol=1e-12, atol=1e-15)
        assert np.linalg.norm(changed-s) > 0
