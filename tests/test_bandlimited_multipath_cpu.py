import numpy as np
import pytest
from gnss_gpu.bandlimited_multipath_cpu import fit_bandlimited_paths
from gnss_gpu.multipath_information_cpu import CHIP_RATE, two_path_spectrum


@pytest.mark.parametrize('ratio,phase', [(.8, 0.), (.8, np.pi/2), (1.5, np.pi)])
def test_noiseless_two_path_recovery_without_los_strength_prior(ratio, phase):
    bandwidth = 1.8*CHIP_RATE
    design, _ = two_path_spectrum(bandwidth, .173, .5)
    y = design@np.array([1., ratio*np.exp(1j*phase)])
    saved = y.copy()
    fit = fit_bandlimited_paths(y, bandwidth)
    assert fit.selected_model == 'two_path'
    assert fit.first_delay_chips == pytest.approx(.173, abs=1e-5)
    assert fit.excess_chips == pytest.approx(.5, abs=1e-5)
    np.testing.assert_array_equal(y, saved)


def test_single_path_and_scale_phase_invariance():
    bandwidth = .9*CHIP_RATE
    a, _ = two_path_spectrum(bandwidth, -.213, .5)
    rng = np.random.default_rng(718)
    y = a[:, 0]+.0001*(rng.normal(size=len(a))+1j*rng.normal(size=len(a)))
    fit = fit_bandlimited_paths(y, bandwidth)
    scaled = fit_bandlimited_paths(y*4*np.exp(.9j), bandwidth)
    assert fit.selected_model == scaled.selected_model == 'single_path'
    assert fit.single_delay_chips == pytest.approx(-.213, abs=.002)
    assert fit.single_delay_chips == pytest.approx(scaled.single_delay_chips, abs=1e-7)


def test_boundary_fit_is_not_published_as_two_path():
    bandwidth = 1.8*CHIP_RATE
    a, _ = two_path_spectrum(bandwidth, 1., .5)
    fit = fit_bandlimited_paths(a@np.array([1., .8j]), bandwidth)
    assert fit.selected_model == 'ambiguous'
    assert fit.selected_delay_chips is None
    assert fit.reason == 'two-path search boundary'


@pytest.mark.parametrize('kind', ['zero', 'nan', 'shape'])
def test_invalid_observation(kind):
    a, _ = two_path_spectrum(.9*CHIP_RATE, 0., .5)
    y = a[:, 0].copy()
    if kind == 'zero': y[:] = 0
    if kind == 'nan': y[0] = np.nan
    if kind == 'shape': y = y[:-1]
    with pytest.raises(ValueError):
        fit_bandlimited_paths(y, .9*CHIP_RATE)
