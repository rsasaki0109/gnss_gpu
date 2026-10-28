from dataclasses import asdict
import numpy as np
import pytest
from gnss_gpu.dynamic_multipath_cpu import fit_dynamic_paths
from gnss_gpu.calibrated_dynamic_paths_cpu import fit_single_path
from gnss_gpu.residual_dynamic_paths_cpu import fit_weighted_dynamic_paths,fit_weighted_single_path,estimate_residual_weights,_problem
from gnss_gpu.spectral_noise_cpu import spectral_noise_shape
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE
from experiments.stress_cpu_dynamic_noise import synthesize


def test_unit_weights_preserve_frozen_fit():
    blocks,_,_=synthesize(('white',.8,.7,0.,620000))
    ones=np.ones(blocks.shape[1])
    assert asdict(fit_weighted_dynamic_paths(blocks,ones))==asdict(fit_dynamic_paths(blocks))
    assert fit_weighted_single_path(blocks,ones)==fit_single_path(blocks)


@pytest.mark.parametrize('profile',['white','center','edge'])
def test_estimated_noise_shape_from_fitted_dynamic_residual(profile):
    blocks,_,_=synthesize((profile,.8,.7,0.,620001))
    fit=fit_dynamic_paths(blocks)
    weights=estimate_residual_weights(blocks,fit)
    f,_=ca_passband_spectrum(3.6*CHIP_RATE);truth=spectral_noise_shape(f,profile)
    assert np.sqrt(np.mean(((weights-truth)/truth)**2))<.08
    np.testing.assert_allclose(estimate_residual_weights(blocks*(2+3j),fit),weights,rtol=1e-12,atol=1e-12)


def test_weighted_gradient_and_dense_residual_agree():
    blocks,_,_=synthesize(('center',.8,.7,0.,620002))
    f,_=ca_passband_spectrum(3.6*CHIP_RATE)
    solve=_problem(blocks,spectral_noise_shape(f,'center'));p=np.array([.12,.1,.5])
    loss,gradient=solve(p);dense,_,_=solve(p,True)
    np.testing.assert_allclose(loss,dense,rtol=1e-12)
    h=1e-5
    fd=[(solve(p+h*np.eye(3)[i])[0]-solve(p-h*np.eye(3)[i])[0])/(2*h) for i in range(3)]
    np.testing.assert_allclose(gradient,fd,rtol=1e-4,atol=1e-6)
    with pytest.raises(ValueError):_problem(blocks,np.zeros(len(f)))
