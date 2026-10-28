import numpy as np
from experiments.run_cpu_matched_dynamic_reflection import synthesize, CONDITIONS
from gnss_gpu.noise_aware_multipath_cpu import estimate_noise_weights


def test_matched_mean_and_static_zero_doppler():
    for fd,phase in CONDITIONS:
        dynamic,control,_,coefficient=synthesize(fd,phase,520000)
        np.testing.assert_allclose(dynamic.mean(axis=0),control.mean(axis=0),rtol=0,atol=1e-15)
        assert abs(coefficient)>.1  # no exact integration nulls
        # Static-control residuals must be identical for every signal condition.
        residual=control-control.mean(axis=0)
        if fd==CONDITIONS[0][0] and phase==0:reference=residual
        np.testing.assert_allclose(residual,reference,rtol=0,atol=1e-15)
    dynamic,control,_,_=synthesize(0.,.4,520001)
    np.testing.assert_allclose(dynamic,control,rtol=0,atol=1e-15)
    np.testing.assert_allclose(estimate_noise_weights(dynamic),estimate_noise_weights(control),rtol=0,atol=1e-14)


def test_phase_rotation_preserves_coherent_magnitude_and_changes_signal():
    a,_,delay_a,ca=synthesize(2.3,0.,520002)
    b,_,delay_b,cb=synthesize(2.3,np.pi/2,520002)
    assert delay_a==delay_b
    np.testing.assert_allclose(cb,1j*ca,atol=1e-15)
    assert np.max(np.abs(a-b))>.01
