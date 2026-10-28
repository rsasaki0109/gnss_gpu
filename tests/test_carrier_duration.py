from dataclasses import asdict
import numpy as np
import pytest
from gnss_gpu.duration_carrier_paths_cpu import _problem,temporal_echo,fit_carrier_paths
from gnss_gpu.common_carrier_paths_cpu import _problem as old_problem,fit_carrier_paths as old_fit
from experiments.run_cpu_carrier_duration import synthesize


def test_prefixes_and_ten_block_fit_preserved():
    b,s,d=synthesize((40,.1,-.3,900000))
    for count in (10,20):
        other,signal,delay=synthesize((count,.1,-.3,900000))
        np.testing.assert_array_equal(other,b[:count]);np.testing.assert_array_equal(signal,s[:count]);assert d==delay
    ones=np.ones(b.shape[1]);p=(.12,-.3,.1,.1)
    np.testing.assert_array_equal(_problem(b[:10],ones,True)(p)[1],old_problem(b[:10],ones,True)(p)[1])
    assert asdict(fit_carrier_paths(b[:10],ones,two=True))==asdict(old_fit(b[:10],ones,two=True))


@pytest.mark.parametrize('count',[20,40])
def test_long_duration_gradient_and_temporal_derivative(count):
    b,_,_=synthesize((count,.1,-.3,900001));solve=_problem(b,np.ones(b.shape[1]),True)
    p=np.array([.12,-.3,.1,.1]);h=1e-5
    numeric=[(solve(p+h*e)[0]-solve(p-h*e)[0])/(2*h) for e in np.eye(4)]
    np.testing.assert_allclose(solve(p)[1],numeric,rtol=1e-4,atol=1e-5)
    np.testing.assert_allclose(solve(p)[0],solve(p,True)[0],rtol=1e-12)
    for fd in (0.,-.4,.7):
        c,dc=temporal_echo(fd,count)
        np.testing.assert_allclose(dc,(temporal_echo(fd+h,count)[0]-temporal_echo(fd-h,count)[0])/(2*h),rtol=1e-8,atol=1e-8)
    with pytest.raises(ValueError):_problem(b[:3],np.ones(b.shape[1]),True)
