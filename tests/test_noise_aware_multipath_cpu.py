import numpy as np
import pytest
from gnss_gpu.noise_aware_multipath_cpu import estimate_noise_weights,_fit_weighted_spectrum,fit_noise_aware_paths
from gnss_gpu.bandlimited_multipath_v2_cpu import fit_bandlimited_paths
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE,two_path_spectrum
from gnss_gpu.spectral_noise_cpu import spectral_noise_shape


@pytest.mark.parametrize('profile',['white','center','edge'])
def test_estimated_shape_from_data_without_signal_or_psd_truth(profile):
    f,_=ca_passband_spectrum(3.6*CHIP_RATE)
    shape=spectral_noise_shape(f,profile)
    rng=np.random.default_rng(880)
    noise=(rng.normal(size=(10,len(f)))+1j*rng.normal(size=(10,len(f))))*np.sqrt(shape/2)
    signal=3*np.exp(1j*f/1e5)
    weights=estimate_noise_weights(noise+signal)
    np.testing.assert_allclose(weights,estimate_noise_weights(noise),atol=1e-14)
    np.testing.assert_allclose(weights,estimate_noise_weights((noise+signal)*7*np.exp(.7j)),atol=1e-14)
    assert np.sqrt(np.mean((weights/shape-1)**2)) < .08
    assert weights.mean()==pytest.approx(1.)


def test_unit_weights_reproduce_frozen_estimator():
    a,_=two_path_spectrum(.9*CHIP_RATE,.17,.5)
    y=a@np.array([1.,.8j])
    old=fit_bandlimited_paths(y,.9*CHIP_RATE)
    new=_fit_weighted_spectrum(y,.9*CHIP_RATE,np.ones(len(y)))
    assert old.__dict__==new.__dict__


@pytest.mark.parametrize('blocks',[np.zeros((10,200)),np.ones((2,200)),np.full((10,200),np.nan)])
def test_invalid_noise_estimation(blocks):
    with pytest.raises(ValueError): estimate_noise_weights(blocks)


def test_wrong_observation_length():
    with pytest.raises(ValueError): fit_noise_aware_paths(np.ones((9,7365)))
