from dataclasses import asdict
import numpy as np
import pytest
from gnss_gpu import drift_carrier_paths_cpu as old
from gnss_gpu import cached_drift_carrier_paths_cpu as new
from experiments.run_cpu_drift_null_selection import synthesize


@pytest.mark.parametrize('two',[False,True])
def test_cache_eviction_and_problem_isolation_exact(two):
    b,_,_=synthesize((40,.5,-.3,990000));weights=np.linspace(.3,2.,b.shape[1])
    a=old._problem(b,weights,two);c=new._problem(b,weights,two)
    points=[(.1+i*.002,-.3,.5,.1,.2) if two else (.1+i*.002,-.3,.5) for i in range(75)]
    for p in points+[points[0],points[-1]]:
        x,y=a(p),c(p);assert x[0]==y[0];np.testing.assert_array_equal(x[1],y[1])
    for x,y in zip(a(points[0],True),c(points[0],True)):np.testing.assert_array_equal(x,y)
    other=new._problem(b*(2+3j),weights[::-1],two)
    other(points[0])
    assert c(points[0])[0]==a(points[0])[0]


def test_full_pipeline_bit_exact():
    b,_,_=synthesize((10,-.5,.2,990001))
    a,wa=old.score_drift_paths(b);c,wc=new.score_drift_paths(b)
    assert a['score']==c['score']
    for key in ('single','two','preliminary'):assert asdict(a[key])==asdict(c[key])
    np.testing.assert_array_equal(wa,wc)
