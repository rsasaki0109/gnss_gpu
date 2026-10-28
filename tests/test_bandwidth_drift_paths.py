from dataclasses import asdict
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from gnss_gpu import bandwidth_drift_paths_cpu as new
from gnss_gpu import cached_drift_carrier_paths_cpu as old
from gnss_gpu.multipath_information_cpu import CHIP_RATE,ca_passband_spectrum
from experiments.run_cpu_drift_bandwidth import synthesize


def test_default_pipeline_matches_frozen_receiver_exactly():
    b,_,_=synthesize((3.6,.5,-.3,1240000));b=b[:10]
    a,wa=old.score_drift_paths(b);c,wc=new.score_drift_paths(b)
    assert a['score']==c['score']
    for k in ('single','two','preliminary'):assert asdict(a[k])==asdict(c[k])
    np.testing.assert_array_equal(wa,wc)


def test_nested_iq_and_signal_without_power_renormalization():
    wide,signal,truth=synthesize((7.2,.5,-.3,1240001))
    f,s=ca_passband_spectrum(7.2*CHIP_RATE)
    for band in (1.8,3.6):
        b,small,d=synthesize((band,.5,-.3,1240001));keep=abs(f)<=band*CHIP_RATE
        assert truth==d
        np.testing.assert_array_equal(b,wide[:,keep]);np.testing.assert_array_equal(small,signal[:,keep])
        _,ss=ca_passband_spectrum(band*CHIP_RATE);np.testing.assert_array_equal(ss,s[keep])


def test_wideband_gradient_matches_dense_residual():
    b,_,_=synthesize((7.2,-.5,.2,1240002));b=b[:10]
    solve=new._problem(b,np.ones(b.shape[1]),True,7.2*CHIP_RATE)
    p=np.array([.1,.2,-.5,.1,.3]);loss,g=solve(p)
    numeric=approx_derivative(lambda x:solve(x)[0],p,method='3-point').ravel()
    np.testing.assert_allclose(g,numeric,rtol=3e-5,atol=1e-5)
    np.testing.assert_allclose(loss,solve(p,True)[0],rtol=0,atol=1e-9)
