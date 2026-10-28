import numpy as np
from experiments.stress_cpu_carrier_drift import synthesize,chirp_blocks
from experiments.run_cpu_carrier_duration import synthesize as baseline


def test_zero_drift_prefixes_and_paired_noise():
    b,s,d=synthesize((40,0.,-.4,920000))
    old,signal,delay=baseline((40,.1,-.4,920000))
    np.testing.assert_array_equal(b,old);np.testing.assert_array_equal(s,signal);assert d==delay
    for rate in (-2.,.5,2.):
        other,changed,truth=synthesize((40,rate,-.4,920000))
        short,short_signal,short_truth=synthesize((10,rate,-.4,920000))
        np.testing.assert_array_equal(short,other[:10]);np.testing.assert_array_equal(short_signal,changed[:10]);assert short_truth==truth==d
        np.testing.assert_allclose(other-changed,b-s,rtol=1e-12,atol=1e-15)


def test_chirp_matches_independent_midpoint_quadrature():
    for frequency,rate in [(-.7,-2.),(.4,.5),(.8,2.)]:
        c=chirp_blocks(frequency,rate,40)
        for k in (0,19,39):
            t=k*.02+(np.arange(40000)+.5)*.02/40000
            expected=np.mean(np.exp(2j*np.pi*(frequency*t+.5*rate*t*t)))
            np.testing.assert_allclose(c[k],expected,rtol=1e-10,atol=1e-11)
        assert np.all(np.abs(c)<=1+1e-14)
