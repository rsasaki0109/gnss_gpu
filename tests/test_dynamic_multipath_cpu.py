import numpy as np
import pytest
from gnss_gpu.dynamic_multipath_cpu import _problem,temporal_echo,fit_dynamic_paths
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE


def signal(delay=.12,gap=.1,fd=2.3):
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    return s*np.exp(-2j*np.pi*f*delay/CHIP_RATE)*(1+.8j*temporal_echo(fd)[0][:,None]*np.exp(-2j*np.pi*f*gap/CHIP_RATE))


def test_reduced_loss_matches_dense_and_gradient():
    y=signal();solve=_problem(y);p=np.array([.11,.09,2.1])
    loss,grad=solve(p);dense,_,_=solve(p,True)
    np.testing.assert_allclose(loss,dense,atol=1e-12)
    h=1e-5
    expected=[(solve(p+h*np.eye(3)[i])[0]-solve(p-h*np.eye(3)[i])[0])/(2*h) for i in range(3)]
    np.testing.assert_allclose(grad,expected,rtol=2e-5,atol=1e-6)


@pytest.mark.parametrize('fd',[-2.3,2.3])
def test_noiseless_unknown_doppler_recovery(fd):
    fit=fit_dynamic_paths(signal(fd=fd))
    np.testing.assert_allclose([fit.first_delay_chips,fit.excess_chips,fit.echo_doppler_hz],[.12,.1,fd],atol=2e-5)
    assert fit.locally_usable


@pytest.mark.parametrize('fd',[0.,1e-6,2.3,-6.7])
def test_temporal_derivative(fd):
    h=1e-5
    np.testing.assert_allclose(temporal_echo(fd)[1],(temporal_echo(fd+h)[0]-temporal_echo(fd-h)[0])/(2*h),atol=1e-10)


def test_reject_invalid_blocks():
    for y in [np.zeros((10,7365)),np.ones((9,7365)),np.full((10,7365),np.nan)]:
        with pytest.raises(ValueError):fit_dynamic_paths(y)


def test_reduced_noisy_loss_preserves_full_block_residual():
    rng=np.random.default_rng(42)
    y=signal()+.01*(rng.normal(size=(10,7365))+1j*rng.normal(size=(10,7365)))
    solve=_problem(y)
    for p in ((.1,.01,0.),(.12,.1,2.3),(-.2,.5,-5.)):
        loss,_=solve(p)
        dense,_,_=solve(p,True)
        np.testing.assert_allclose(loss,dense,rtol=1e-12,atol=1e-10)
