import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
from gnss_gpu.dominant_path_amplitudes_cpu import dominant_amplitudes
from gnss_gpu.dominant_drift_paths_cpu import dominant_problem
from experiments.run_cpu_drift_null_selection import synthesize


def test_amplitude_solution_against_independent_constrained_optimization():
    rng=np.random.default_rng(731)
    for _ in range(12):
        a=rng.normal(size=(20,2))+1j*rng.normal(size=(20,2))
        truth=np.array([.2+.3j,1.2-.4j]);y=a@truth+.2*(rng.normal(size=20)+1j*rng.normal(size=20))
        gram=a.conj().T@a;rhs=a.conj().T@y
        amp=dominant_amplitudes(gram,rhs)
        assert abs(amp[1])<=abs(amp[0])+1e-12
        loss=np.linalg.norm(y-a@amp)**2
        unpack=lambda x:x[:2]+1j*x[2:]
        objective=lambda x:np.linalg.norm(y-a@unpack(x))**2
        constraint=lambda x:abs(unpack(x)[0])**2-abs(unpack(x)[1])**2
        fits=[minimize(objective,start,method='SLSQP',constraints={'type':'ineq','fun':constraint},
              options=dict(ftol=1e-10,maxiter=500)) for start in ([1,0,0,0],[-1,0,0,0],[0,0,1,0])]
        good=[r for r in fits if r.success and constraint(r.x)>-1e-7]
        assert good
        np.testing.assert_allclose(loss,min(r.fun for r in good),rtol=1e-8,atol=1e-7)


def test_interior_and_zero_solutions_are_unchanged():
    g=np.array([[2,.2j],[-.2j,1.]])
    truth=np.array([1+.1j,.2j]);b=g@truth
    np.testing.assert_allclose(dominant_amplitudes(g,b),truth,atol=1e-14)
    np.testing.assert_array_equal(dominant_amplitudes(g,np.zeros(2)),np.zeros(2))
    with pytest.raises(ValueError):dominant_amplitudes(np.zeros((2,2)),np.ones(2))


@pytest.mark.parametrize('delay',[-.15,.1])
def test_constrained_gradient_and_dense_loss(delay):
    b,_,_=synthesize((10,.5,-.3,990000));w=np.ones(b.shape[1])
    solve=dominant_problem(b,w,True);p=np.array([delay,-.2,.4,.15,.2])
    loss,grad=solve(p)
    numerical=approx_derivative(lambda x:solve(x)[0],p,method='3-point').ravel()
    np.testing.assert_allclose(grad,numerical,rtol=3e-5,atol=1e-5)
    np.testing.assert_allclose(loss,solve(p,True)[0],rtol=0,atol=1e-9)


def test_geometric_interior_returns_json_serializable_boolean():
    import json
    from gnss_gpu.dominant_drift_paths_cpu import geometric_interior
    assert json.dumps(geometric_interior(np.array([0.,0.,0.,.1,0.])))=='true'
    assert json.dumps(geometric_interior(np.array([0.,0.,0.,.01,0.])))=='false'
