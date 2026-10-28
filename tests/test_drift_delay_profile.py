import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from gnss_gpu.drift_delay_profile_cpu import fixed_delay_objective, delay_profile
from gnss_gpu.cached_drift_carrier_paths_cpu import _problem, DriftFit
from experiments.run_cpu_drift_null_selection import synthesize


@pytest.mark.parametrize('two',[False,True])
def test_fixed_delay_gradient_and_dense_residual(two):
    b,_,_=synthesize((10,.5,-.3,990000))
    solve=_problem(b,np.ones(b.shape[1]),two)
    nuisance=np.array([-.2,.4,.1,.2] if two else [-.2,.4])
    objective=fixed_delay_objective(solve,.1)
    loss,grad=objective(nuisance)
    numerical=approx_derivative(lambda x:objective(x)[0],nuisance,method='3-point').ravel()
    np.testing.assert_allclose(grad,numerical,rtol=2e-5,atol=1e-5)
    assert abs(loss-solve(np.r_[.1,nuisance],True)[0])<1e-9


def test_profile_preserves_grid_and_reports_loss():
    b,_,d=synthesize((10,.5,-.3,990000))
    weights=np.ones(b.shape[1])
    single=DriftFit(d,-.3,.5,None,None,None,True,False,'test',1e10)
    two=DriftFit(d,-.3,.5,.1,0.,None,True,False,'test',1e10)
    result=delay_profile(b,weights,[d],single,two)
    for name in ('single','two'):
        p=result['profiles'][name][0]
        assert p['delay_chips']==d and p['relative_loss']>=0
        assert p['compact_dense_abs_difference']<1e-9
        assert p['loss']<=_problem(b,weights,name=='two')(np.r_[d,[-.3,.5,.1,0.] if name=='two' else [-.3,.5]])[0]+1e-8
    with pytest.raises(ValueError):delay_profile(b,weights,[float('nan')],single,two)
