import numpy as np
from experiments.stress_cpu_drift_phase import synthesize,CONDITIONS
from experiments.stress_cpu_carrier_drift import synthesize as baseline
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE


def test_old_signal_and_noise_pairing():
    b,s,d=synthesize(('echo',.1,np.pi,-.7,.3,1030000))
    old,signal,delay=baseline((40,-.7,.3,1030000))
    np.testing.assert_array_equal(b,old);np.testing.assert_array_equal(s,signal);assert d==delay
    for fd,phase in CONDITIONS:
        other,changed,truth=synthesize(('echo',fd,phase,-.7,.3,1030000))
        assert truth==d
        np.testing.assert_allclose(other-changed,b-s,rtol=1e-12,atol=1e-15)


def test_phase_doppler_signal_against_midpoint_integral():
    fd,phase,drift,offset,replica=.7,np.pi/2,-1.,.4,1030001
    _,signal,delay=synthesize(('echo',fd,phase,drift,offset,replica))
    rng=np.random.default_rng(np.random.SeedSequence([20261027,0x44555241,replica]));rng.uniform(-.3,.3);common=rng.uniform(-np.pi,np.pi)
    f,s=ca_passband_spectrum(3.6*CHIP_RATE);indices=[0,len(f)//2,len(f)-1];f=f[indices];s=s[indices]
    for k in (0,19,39):
        t=k*.02+(np.arange(40000)+.5)*.02/40000
        c=np.mean(np.exp(2j*np.pi*(offset*t+.5*drift*t*t)))
        e=np.mean(np.exp(2j*np.pi*((offset+fd)*t+.5*drift*t*t)))
        expected=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*common)*(c+.8*np.exp(1j*phase)*e*np.exp(-2j*np.pi*f*.1/CHIP_RATE))
        np.testing.assert_allclose(signal[k,indices],expected,rtol=1e-10,atol=1e-12)
