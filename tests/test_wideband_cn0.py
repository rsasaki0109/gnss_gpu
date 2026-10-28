import numpy as np
import pytest
from experiments.run_cpu_wideband_cn0 import synthesize
from experiments.replicate_cpu_drift_bandwidth import synthesize as baseline


@pytest.mark.parametrize('kind,replica',[('null',1280000),('echo',1290000)])
def test_exact_baseline_and_paired_noise_scaling(kind,replica):
    original,signal,truth=baseline((kind,7.2,.5,-.3,replica))
    b,s,d=synthesize((kind,37.,.5,-.3,replica))
    np.testing.assert_array_equal(b,original);np.testing.assert_array_equal(s,signal);assert d==truth
    noise=b-s
    for cn0 in (43.,49.):
        changed,cs,cd=synthesize((kind,cn0,.5,-.3,replica))
        np.testing.assert_array_equal(cs,s);assert cd==d
        scale=10**((37-cn0)/20)
        np.testing.assert_allclose(changed-cs,noise*scale,rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(np.vdot(changed-cs,changed-cs).real/np.vdot(noise,noise).real,10**((37-cn0)/10),rtol=1e-12)
    with pytest.raises(ValueError):synthesize((kind,50.,.5,-.3,replica))
