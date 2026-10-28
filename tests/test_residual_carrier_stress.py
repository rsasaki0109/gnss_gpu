import numpy as np
from experiments.stress_cpu_residual_carrier import synthesize
from experiments.stress_cpu_dynamic_noise import synthesize as baseline
from gnss_gpu.multipath_information_cpu import CHIP_RATE, ca_passband_spectrum


def test_zero_offset_is_bit_exact_and_noise_is_paired():
    for ratio in (0., .8):
        b, s, d = synthesize((ratio, 0., 760000))
        expected, signal, delay = baseline(('white', ratio, .7, 0., 760000))
        np.testing.assert_array_equal(b, expected)
        np.testing.assert_array_equal(s, signal)
        assert d == delay
        for offset in (-1., .1, .5, 1.):
            other, changed, truth = synthesize((ratio, offset, 760000))
            assert truth == d
            np.testing.assert_allclose(other-changed, b-s, atol=1e-15, rtol=1e-12)


def test_signal_matches_independent_time_quadrature():
    ratio, offset, replica = .8, -.5, 760001
    _, signal, delay = synthesize((ratio, offset, replica))
    rng = np.random.default_rng(np.random.SeedSequence([20261026, 0x44594741, replica]))
    rng.uniform(-.3, .3)
    common = rng.uniform(-np.pi, np.pi)
    f, s = ca_passband_spectrum(3.6*CHIP_RATE)
    indices = [0, len(f)//2, len(f)-1]
    f, s = f[indices], s[indices]
    for block in (0, 4, 9):
        t = block*.02 + (np.arange(20000)+.5)*.02/20000
        direct = np.mean(np.exp(2j*np.pi*offset*t))
        echo = np.mean(np.exp(2j*np.pi*(offset+.7)*t))
        expected = s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*common)*(direct+ratio*echo*np.exp(-2j*np.pi*f*.1/CHIP_RATE))
        np.testing.assert_allclose(signal[block, indices], expected, atol=1e-13, rtol=1e-11)
