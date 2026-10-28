import numpy as np
from experiments.stress_cpu_drift_gap_amplitude import synthesize,CONDITIONS
from experiments.stress_cpu_drift_phase import synthesize as baseline
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE


def test_baseline_signal_and_paired_noise():
    b,s,d=synthesize(('echo',.8,.1,0.,-.5,.3,1060000))
    old,signal,delay=baseline(('echo',0.,0.,-.5,.3,1060000))
    np.testing.assert_array_equal(b,old);np.testing.assert_array_equal(s,signal);assert d==delay
    for ratio,gap,phase in CONDITIONS:
        other,changed,truth=synthesize(('echo',ratio,gap,phase,-.5,.3,1060000))
        assert truth==d
        np.testing.assert_allclose(other-changed,b-s,rtol=1e-12,atol=1e-15)


def test_weak_close_echo_matches_independent_integral():
    ratio,gap,phase,drift,offset,replica=.2,.03,np.pi,-.5,.3,1060001
    _,signal,delay=synthesize(('echo',ratio,gap,phase,drift,offset,replica))
    rng=np.random.default_rng(np.random.SeedSequence([20261027,0x44555241,replica]));rng.uniform(-.3,.3);common=rng.uniform(-np.pi,np.pi)
    f,s=ca_passband_spectrum(3.6*CHIP_RATE);indices=[0,len(f)//2,len(f)-1];f=f[indices];s=s[indices]
    for k in (0,19,39):
        t=k*.02+(np.arange(40000)+.5)*.02/40000
        c=np.mean(np.exp(2j*np.pi*(offset*t+.5*drift*t*t)))
        direct=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*common)*c
        echo=ratio*s*np.exp(-2j*np.pi*f*(delay+gap)/CHIP_RATE+1j*(common+phase))*c
        np.testing.assert_allclose(signal[k,indices],direct+echo,rtol=1e-10,atol=1e-12)
