import numpy as np
import pytest
from gnss_gpu.drift_carrier_paths_cpu import _problem,temporal_chirp,fit_drift_paths,estimate_weights
from gnss_gpu.duration_carrier_paths_cpu import _problem as constant_problem
from experiments.run_cpu_drift_null_selection import synthesize as null
from experiments.stress_cpu_carrier_drift import synthesize as echo


@pytest.mark.parametrize('two',[False,True])
def test_analytic_gradient_full_sse_and_zero_drift_nesting(two):
    b,_,_=echo((40,.5,-.3,960000));weights=np.linspace(.3,2.,b.shape[1])
    solve=_problem(b,weights,two);p=np.array([.12,-.2,.4,.15,.3] if two else [.12,-.2,.4]);h=1e-5
    numeric=[(solve(p+h*e)[0]-solve(p-h*e)[0])/(2*h) for e in np.eye(len(p))]
    np.testing.assert_allclose(solve(p)[1],numeric,rtol=1e-4,atol=1e-5)
    np.testing.assert_allclose(solve(p)[0],solve(p,True)[0],rtol=1e-12)
    p[2]=0
    np.testing.assert_allclose(solve(p)[0],constant_problem(b,weights,two)(np.delete(p,2))[0],rtol=1e-12)


@pytest.mark.parametrize('drift',[-1.,2.])
def test_unknown_noiseless_single_drift_recovery(drift):
    _,s,d=null((40,drift,-.3,960001))
    fit=fit_drift_paths(s,np.ones(s.shape[1]),two=False)
    assert fit.locally_usable and abs(fit.first_delay_chips-d)<1e-6
    assert abs(fit.common_carrier_hz+.3)<1e-5 and abs(fit.common_drift_hz_s-drift)<1e-5


def test_noiseless_two_path_recovery():
    _,s,d=echo((40,.5,-.3,960002))
    fit=fit_drift_paths(s,np.ones(s.shape[1]),two=True)
    assert fit.locally_usable and abs(fit.first_delay_chips-d)<1e-5
    assert abs(fit.excess_chips-.1)<1e-5 and abs(fit.relative_doppler_hz-.1)<1e-4
    assert abs(fit.common_carrier_hz+.3)<1e-4 and abs(fit.common_drift_hz_s-.5)<1e-4


def test_chirp_derivatives_and_invalid_data():
    for f,a in [(0.,0.),(-.3,-2.),(.8,2.)]:
        c,df,da=temporal_chirp(f,a,40);h=1e-5
        np.testing.assert_allclose(df,(temporal_chirp(f+h,a,40)[0]-temporal_chirp(f-h,a,40)[0])/(2*h),rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(da,(temporal_chirp(f,a+h,40)[0]-temporal_chirp(f,a-h,40)[0])/(2*h),rtol=1e-8,atol=1e-8)
    b,_,_=null((10,0.,0.,960003));ones=np.ones(b.shape[1])
    for bad in (b[:3],b*0,b*np.nan):
        with pytest.raises(ValueError):_problem(bad,ones,True)
    with pytest.raises(ValueError):_problem(b,ones*0,True)
