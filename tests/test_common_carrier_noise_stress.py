import numpy as np
import pytest
from experiments.stress_cpu_common_carrier_noise import synthesize
from experiments.stress_cpu_residual_carrier import synthesize as white
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE
from gnss_gpu.spectral_noise_cpu import spectral_noise_shape


@pytest.mark.parametrize('ratio,offset',[(0.,-.7),(.8,.5)])
def test_only_noise_changes_with_profile(ratio,offset):
    task=(ratio,offset,840000)
    b,s,d=synthesize(('white',*task))
    expected,signal,delay=white(task)
    np.testing.assert_array_equal(b,expected)
    np.testing.assert_array_equal(s,signal)
    assert d==delay
    f,_=ca_passband_spectrum(3.6*CHIP_RATE)
    for profile in ('center','edge'):
        other,other_signal,other_delay=synthesize((profile,*task))
        np.testing.assert_array_equal(s,other_signal)
        assert other_delay==d
        shape=spectral_noise_shape(f,profile)
        np.testing.assert_allclose(shape.mean(),1.,rtol=1e-15)
        np.testing.assert_allclose((other-other_signal)/np.sqrt(shape),b-s,rtol=1e-12,atol=1e-15)
