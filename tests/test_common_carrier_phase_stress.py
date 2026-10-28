import numpy as np
from experiments.stress_cpu_common_carrier_phase import synthesize
from experiments.stress_cpu_residual_carrier import synthesize as baseline
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE


def test_baseline_and_shared_noise():
    b,s,d=synthesize((.8,-.4,.7,0.,880000))
    old,signal,delay=baseline((.8,-.4,880000))
    np.testing.assert_array_equal(b,old)
    np.testing.assert_array_equal(s,signal)
    assert d==delay
    for fd in (0.,.1,.7):
        for phase in (0.,np.pi/2,np.pi):
            other,changed,truth=synthesize((.8,-.4,fd,phase,880000))
            assert truth==d
            np.testing.assert_allclose(other-changed,b-s,atol=1e-15,rtol=1e-12)


def test_signal_against_time_quadrature():
    ratio,offset,fd,phase,replica=.8,-.4,.1,np.pi,880001
    _,signal,delay=synthesize((ratio,offset,fd,phase,replica))
    rng=np.random.default_rng(np.random.SeedSequence([20261026,0x44594741,replica]))
    rng.uniform(-.3,.3);common=rng.uniform(-np.pi,np.pi)
    f,s=ca_passband_spectrum(3.6*CHIP_RATE);indices=[0,len(f)//2,len(f)-1];f=f[indices];s=s[indices]
    for k in (0,4,9):
        t=k*.02+(np.arange(20000)+.5)*.02/20000
        direct=np.mean(np.exp(2j*np.pi*offset*t))
        echo=np.mean(np.exp(2j*np.pi*(offset+fd)*t))
        expected=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*common)*(direct+ratio*np.exp(1j*phase)*echo*np.exp(-2j*np.pi*f*.1/CHIP_RATE))
        np.testing.assert_allclose(signal[k,indices],expected,atol=1e-13,rtol=1e-11)
