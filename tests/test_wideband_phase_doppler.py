import numpy as np
from experiments.stress_cpu_wideband_phase_doppler import synthesize,PHASES,FREQUENCIES
from experiments.run_cpu_wideband_cn0 import synthesize as baseline
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE


def test_baseline_exact_and_all_cell_noise_shared():
    for cn0 in (37.,49.):
        b,s,d=synthesize((cn0,0.,np.pi,.5,-.3,1310000))
        old,signal,truth=baseline(('echo',cn0,.5,-.3,1310000))
        np.testing.assert_array_equal(b,old);np.testing.assert_array_equal(s,signal);assert d==truth
        for fd in FREQUENCIES:
            for phase in PHASES:
                other,changed,dt=synthesize((cn0,fd,phase,.5,-.3,1310000))
                assert dt==d
                np.testing.assert_allclose(other-changed,b-s,rtol=1e-12,atol=1e-15)


def test_changed_carrier_phase_matches_independent_block_integration():
    drift,offset,replica=-.5,.3,1310001
    _,signal,delay=synthesize((49.,.7,np.pi/2,drift,offset,replica))
    rng=np.random.default_rng(np.random.SeedSequence([20261027,0x42414E44,replica]));rng.uniform(-.3,.3);common=rng.uniform(-np.pi,np.pi)
    f,s=ca_passband_spectrum(7.2*CHIP_RATE);indices=[0,len(f)//2,len(f)-1];f=f[indices];s=s[indices]
    for block in (0,19,39):
        t=block*.02+(np.arange(40000)+.5)*.02/40000
        c=np.mean(np.exp(2j*np.pi*(offset*t+.5*drift*t*t)))
        e=np.mean(np.exp(2j*np.pi*((offset+.7)*t+.5*drift*t*t)))
        expected=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*common)*(c+.8j*e*np.exp(-2j*np.pi*f*.03/CHIP_RATE))
        np.testing.assert_allclose(signal[block,indices],expected,rtol=1e-10,atol=1e-12)
