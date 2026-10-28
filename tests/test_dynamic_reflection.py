import numpy as np
from experiments.run_cpu_dynamic_reflection import echo_coefficients, synthesize


def test_block_average_matches_numerical_phase_integration():
    for fd in (0.,1.,5.,10.,-5.):
        t=(np.arange(10)[:,None]+(np.arange(2000)+.5)/2000)*.020
        expected=np.exp(2j*np.pi*fd*t).mean(axis=1)
        np.testing.assert_allclose(echo_coefficients(fd),expected,atol=2e-8)
        np.testing.assert_allclose(echo_coefficients(fd).mean(),np.sinc(fd*.2)*np.exp(1j*np.pi*fd*.2),atol=1e-15)


def test_paired_noise_and_static_signal_cancellation():
    a,sa,da,va=synthesize(.8,0.,510000)
    b,sb,db,vb=synthesize(.8,5.,510000)
    assert da==db and va==vb
    np.testing.assert_allclose(a-sa,b-sb,atol=1e-16)
    np.testing.assert_allclose(sa, np.broadcast_to(sa[0],sa.shape))
    assert np.mean(np.var(sb,axis=0))>0
    x,_,_,_=synthesize(0.,0.,510001)
    y,_,_,_=synthesize(0.,10.,510001)
    np.testing.assert_array_equal(x,y)
