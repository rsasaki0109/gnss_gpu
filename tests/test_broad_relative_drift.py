from types import SimpleNamespace
import numpy as np
from gnss_gpu.broad_relative_drift_cpu import search_seeds,BOUNDS


def test_diverse_search_and_observation_relative_delays():
    seed=SimpleNamespace(first_delay_chips=.98,excess_chips=.04)
    visited=[]
    def objective(p):
        visited.append(p)
        return sum(x*x for x in p), None
    starts=search_seeds(objective,seed)
    assert len(visited)==729 and len(starts)==27
    assert len({(p[1],p[2],p[5]) for p in starts})==27
    np.testing.assert_allclose(sorted({p[0] for p in visited}),[.93,.98,1.],rtol=0,atol=1e-15)
    for p in visited:
        assert all(lo<=x<=hi for x,(lo,hi) in zip(p,BOUNDS))
    for p in starts:
        group=[x for x in visited if (x[1],x[2],x[5])==(p[1],p[2],p[5])]
        assert sum(x*x for x in p)==min(sum(x*x for x in candidate) for candidate in group)
