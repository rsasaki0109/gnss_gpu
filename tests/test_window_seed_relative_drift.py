import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from gnss_gpu.window_seed_relative_drift_cpu import join_windows,constant_carrier_objective
from gnss_gpu.relative_drift_paths_cpu import _problem
from experiments.run_cpu_echo_sign_reversal import synthesize
from gnss_gpu.multipath_information_cpu import CHIP_RATE


def test_midpoint_frequency_extrapolation_and_clock_origin():
    left=[.1,.3+.2*.8,.05,-.6+.2*(-1.7)]
    right=[.12,.3+.6*.8,.07,-.6+.6*(-1.7)]
    np.testing.assert_allclose(join_windows(left,right,(.2,.6)),[.11,.3,.8,.06,-.6,-1.7],atol=1e-14)
    with pytest.raises(ValueError):join_windows(left,right,(.6,.2))


def test_restricted_objective_gradient_on_late_window():
    b,_,_=synthesize((.7,-2.,.5,-.3,1390000));b=b[20:]
    solve=_problem(b,np.ones(b.shape[1]),True,7.2*CHIP_RATE)
    objective=constant_carrier_objective(solve);p=np.array([.1,.2,.1,-.4])
    loss,grad=objective(p)
    assert loss==solve((.1,.2,0.,.1,-.4,0.))[0]
    numeric=approx_derivative(lambda x:objective(x)[0],p,method='3-point').ravel()
    np.testing.assert_allclose(grad,numeric,rtol=3e-5,atol=1e-5)
