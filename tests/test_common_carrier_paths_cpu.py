import numpy as np
import pytest
from gnss_gpu.common_carrier_paths_cpu import _problem, fit_carrier_paths, estimate_weights
from experiments.stress_cpu_residual_carrier import synthesize


@pytest.mark.parametrize('two',[False,True])
def test_gradient_and_explicit_residual(two):
    blocks,_,_ = synthesize((.8,.5,780000))
    weights = np.linspace(.4,2.,blocks.shape[1])
    solve = _problem(blocks,weights,two)
    p = np.array([.12,-.3,.2,.9] if two else [.12,-.3])
    loss,grad = solve(p)
    dense,a,_ = solve(p,True)
    np.testing.assert_allclose(loss,dense,rtol=1e-12)
    y = blocks/np.sqrt(weights/weights.mean())
    y = (y/np.max(np.abs(y))).ravel()
    residual = y-a@np.linalg.lstsq(a,y,rcond=None)[0]
    np.testing.assert_allclose(dense,np.vdot(residual,residual).real,rtol=1e-12)
    h = 1e-5
    numeric = [(solve(p+h*e)[0]-solve(p-h*e)[0])/(2*h) for e in np.eye(len(p))]
    np.testing.assert_allclose(grad,numeric,rtol=1e-4,atol=1e-6)


@pytest.mark.parametrize('offset',[-.8,.5])
@pytest.mark.parametrize('two',[False,True])
def test_noiseless_unknown_carrier_recovery(offset,two):
    _,signal,delay = synthesize((.8 if two else 0.,offset,780001))
    fit = fit_carrier_paths(signal,np.ones(signal.shape[1]),two=two)
    assert fit.locally_usable
    assert abs(fit.first_delay_chips-delay)<1e-5
    assert abs(fit.common_carrier_hz-offset)<1e-4
    if two:
        assert abs(fit.excess_chips-.1)<1e-5
        assert abs(fit.relative_doppler_hz-.7)<1e-4


def test_validation_and_noise_weights_scaling():
    blocks,_,_ = synthesize((.8,.5,780002))
    ones = np.ones(blocks.shape[1])
    fit = fit_carrier_paths(blocks,ones,two=True)
    weights = estimate_weights(blocks,fit)
    np.testing.assert_allclose(estimate_weights(blocks*(2+3j),fit),weights,rtol=1e-11)
    assert np.sqrt(np.mean((weights-1)**2))<.08
    for bad in (np.zeros_like(blocks), blocks[:2], blocks*np.nan):
        with pytest.raises(ValueError): _problem(bad,ones,True)
    with pytest.raises(ValueError): _problem(blocks,ones*0,True)
