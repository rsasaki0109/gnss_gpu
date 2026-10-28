import numpy as np
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE
from gnss_gpu.spectral_noise_cpu import spectral_noise_shape
from experiments.stress_cpu_dynamic_noise import synthesize


def test_noise_shape_intervention_preserves_signal_and_pairs_noise():
    white,signal,delay=synthesize(('white',.8,.1,0.,600000))
    f,_=ca_passband_spectrum(3.6*CHIP_RATE)
    for profile in ('center','edge'):
        colored,other,d=synthesize((profile,.8,.1,0.,600000))
        np.testing.assert_array_equal(signal,other)
        assert delay==d
        np.testing.assert_allclose((colored-signal)/np.sqrt(spectral_noise_shape(f,profile)),white-signal,rtol=1e-12,atol=1e-15)
