import numpy as np
from experiments.run_cpu_drift_null_selection import synthesize
from gnss_gpu.duration_carrier_paths_cpu import fit_carrier_paths,temporal_echo


def test_null_noiseless_fit_and_prefix_noise_pairing():
    blocks,signal,delay=synthesize((40,0.,-.3,940000))
    fit=fit_carrier_paths(signal,np.ones(signal.shape[1]),two=False)
    assert fit.locally_usable and abs(fit.first_delay_chips-delay)<1e-6 and abs(fit.common_carrier_hz+.3)<1e-6
    np.testing.assert_allclose(signal/temporal_echo(-.3,40)[0][:,None],np.broadcast_to(signal[0]/temporal_echo(-.3,40)[0][0],signal.shape),atol=1e-15)
    for rate in (0.,.5,2.):
        b,s,d=synthesize((40,rate,-.3,940000));short,short_signal,truth=synthesize((10,rate,-.3,940000))
        assert d==truth==delay
        np.testing.assert_array_equal(short,b[:10]);np.testing.assert_array_equal(short_signal,s[:10])
        np.testing.assert_allclose(b-s,blocks-signal,rtol=1e-12,atol=1e-15)
