import numpy as np
import pytest
from experiments.replicate_cpu_drift_bandwidth import synthesize,paired_outcomes
from experiments.run_cpu_drift_bandwidth import synthesize as baseline
from gnss_gpu.multipath_information_cpu import CHIP_RATE,ca_passband_spectrum


def test_null_is_direct_only_and_nested_with_paired_noise():
    b,s,d=synthesize(('null',7.2,.5,-.3,1250000))
    old,echo,truth=baseline((7.2,.5,-.3,1250000));assert d==truth
    np.testing.assert_allclose(b-s,old-echo,rtol=1e-12,atol=1e-15)
    f,_=ca_passband_spectrum(7.2*CHIP_RATE)
    np.testing.assert_allclose(echo,s*(1-.8*np.exp(-2j*np.pi*f*.03/CHIP_RATE)),rtol=1e-12,atol=1e-15)
    narrow,ns,nd=synthesize(('null',3.6,.5,-.3,1250000));mask=abs(f)<=3.6*CHIP_RATE
    assert nd==d
    np.testing.assert_array_equal(narrow,b[:,mask]);np.testing.assert_array_equal(ns,s[:,mask])


def test_echo_unchanged_and_exact_paired_test_rejects_bad_pairing():
    a=synthesize(('echo',3.6,-.5,.3,1260000));b=baseline((3.6,-.5,.3,1260000))
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    narrow=[dict(replica=i,success=False) for i in range(10)]
    wide=[dict(replica=i,success=True) for i in range(10)]
    r=paired_outcomes(narrow,wide)
    assert (r['gains'],r['losses'],r['success_rate_difference'])==(10,0,1)
    assert r['exact_two_sided_p']==2/2**10
    assert paired_outcomes(wide,wide)['exact_two_sided_p']==1
    with pytest.raises(ValueError):paired_outcomes(narrow,wide[:-1])
    with pytest.raises(ValueError):paired_outcomes(narrow+[narrow[0]],wide)
