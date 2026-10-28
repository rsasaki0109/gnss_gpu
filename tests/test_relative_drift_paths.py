import numpy as np
from scipy.optimize._numdiff import approx_derivative
from gnss_gpu import relative_drift_paths_cpu as new
from gnss_gpu import bandwidth_drift_paths_cpu as old
from gnss_gpu.multipath_information_cpu import CHIP_RATE
from experiments.run_cpu_echo_sign_reversal import synthesize


def test_nested_model_and_independent_gradient():
    b, _, _ = synthesize((.7,-2.,.5,-.3,1390000))
    b=b[:10]; w=np.ones(b.shape[1])
    solve=new._problem(b,w,True,7.2*CHIP_RATE)
    p=np.array([.1,.2,-.5,.1,.3,0.])
    loss, grad=solve(p)
    oldloss, oldgrad=old._problem(b,w,True,7.2*CHIP_RATE)(p[:5])
    assert loss==oldloss
    np.testing.assert_array_equal(grad[:5],oldgrad)
    for q in (-2.,2.):
        p[-1]=q
        loss, grad=solve(p)
        numeric=approx_derivative(lambda x:solve(x)[0],p,method='3-point').ravel()
        np.testing.assert_allclose(grad,numeric,rtol=3e-5,atol=1e-5)
        np.testing.assert_allclose(loss,solve(p,True)[0],rtol=0,atol=1e-9)


def test_independent_complex_least_squares():
    b, _, _=synthesize((-.7,2.,.5,-.3,1390001));b=b[:10]
    w=np.linspace(.5,1.5,b.shape[1]);p=np.array([.1,.2,-.5,.1,.3,1.8])
    solve=new._problem(b,w,True,7.2*CHIP_RATE)
    loss,a,j=solve(p,True)
    from gnss_gpu.residual_dynamic_paths_cpu import _inverse_std
    y=b*_inverse_std(w,b.shape[1]);y=(y/np.max(abs(y))).ravel()
    residual=y-a@np.linalg.lstsq(a,y,rcond=None)[0]
    np.testing.assert_allclose(loss,np.vdot(residual,residual).real,rtol=1e-12)
    assert j.shape==(b.size,6)
