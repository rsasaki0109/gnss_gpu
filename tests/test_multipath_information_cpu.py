import numpy as np
import pytest

from gnss_gpu.multipath_information_cpu import (
    CHIP_RATE, LIGHT_SPEED, ca_passband_spectrum,
    two_path_spectrum, two_path_information,
)
from gnss_gpu.validation.tracking_residuals import generate_ca_code


def bound(**kwargs):
    settings = dict(sampling_freq=8*CHIP_RATE, bandwidth_hz=.9*CHIP_RATE,
                    excess_chips=.1)
    settings.update(kwargs)
    return two_path_information(**settings)


def test_analytic_coefficients_against_dense_rectangular_waveform():
    # Independent midpoint quadrature of the original piecewise-constant code.
    subdivisions = 128
    code = np.repeat(generate_ca_code(1), subdivisions)
    frequencies, coefficients = ca_passband_spectrum(.9*CHIP_RATE)
    k = np.rint(frequencies/1000).astype(int)
    measured = np.fft.fft(code)[k % len(code)]/len(code)*np.exp(-1j*np.pi*k/len(code))
    np.testing.assert_allclose(measured, coefficients, atol=2e-6, rtol=1e-4)


def test_delay_derivative_against_finite_difference():
    design, derivative = two_path_spectrum(.9*CHIP_RATE, .173, .1)
    h = 1e-6
    plus, _ = two_path_spectrum(.9*CHIP_RATE, .173+h, .1)
    minus, _ = two_path_spectrum(.9*CHIP_RATE, .173-h, .1)
    np.testing.assert_allclose((plus-minus)/(2*h), derivative, atol=1e-10)


@pytest.mark.parametrize('spc', [2, 4, 8, 16])
def test_bound_against_full_real_time_domain_fisher_schur_complement(spc):
    # Independent 6-parameter real Fisher matrix: 4 amplitude + 2 delay.
    # Parseval with 1/N preserves fixed noise PSD as sampling increases.
    bandwidth = .9*CHIP_RATE
    frequencies, _ = ca_passband_spectrum(bandwidth)
    design, derivative = two_path_spectrum(bandwidth, .173, .1)
    derivative *= np.array([1., .8*np.exp(.7j)])
    n = 1023*spc
    harmonics = np.rint(frequencies/1000).astype(int)
    bins = np.zeros((n, 4), dtype=complex)
    bins[harmonics % n] = np.column_stack([design, derivative])
    samples = np.fft.ifft(bins, axis=0)*n
    a, d = samples[:, :2], samples[:, 2:]
    jacobian = np.column_stack([a, 1j*a, d])
    fisher = 2*10**3.7*.020*np.real(jacobian.conj().T@jacobian)/n
    covariance = np.linalg.inv(fisher)
    direct = np.sqrt(covariance[4, 4])*LIGHT_SPEED/CHIP_RATE
    result = bound(sampling_freq=spc*CHIP_RATE, phase_rad=.7)
    assert result.direct_std_bound_m == pytest.approx(direct, rel=1e-8)


def test_power_duration_bandwidth_and_sampling_controls():
    original = bound()
    assert bound(cn0_dbhz=47).direct_std_bound_m == pytest.approx(original.direct_std_bound_m/np.sqrt(10))
    assert bound(integration_s=.080).direct_std_bound_m == pytest.approx(original.direct_std_bound_m/2)
    assert bound(sampling_freq=16*CHIP_RATE) == original
    wide = bound(bandwidth_hz=1.8*CHIP_RATE)
    assert wide.direct_std_bound_m < original.direct_std_bound_m
    assert 0 < original.retained_direct_power < wide.retained_direct_power < 1


def test_degenerate_paths_are_not_given_finite_bounds():
    assert bound(excess_chips=0).status == 'singular_amplitudes'
    assert bound(echo_voltage=0).status == 'singular_delays'
    assert bound(excess_chips=0).direct_std_bound_m is None


@pytest.mark.parametrize('kwargs', [dict(sampling_freq=CHIP_RATE),
    dict(bandwidth_hz=-1), dict(excess_chips=-.1), dict(phase_rad=np.nan),
    dict(echo_voltage=-1), dict(integration_s=.0015), dict(cn0_dbhz=np.inf)])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        bound(**kwargs)
