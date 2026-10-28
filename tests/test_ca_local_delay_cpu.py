import numpy as np
import pytest

from gnss_gpu.ca_cancellation_cpu import CaCancellationConfig,cancel_other_ca
from gnss_gpu.ca_local_delay_cpu import cancel_other_ca_local_delay
from tests.test_ca_cancellation_cpu import observation,FS,N


def test_one_sample_nuisance_epoch_slip_is_removed_without_true_delay_inputs():
    iq=observation(2,300.,37)
    iq[10:]=observation(2,300.,36)[10:]
    cfg=CaCancellationConfig(FS,nuisance_prns=(2,))
    old=cancel_other_ca(iq,cfg);new=cancel_other_ca_local_delay(iq,cfg)
    assert np.sum(old.iq**2)>.1*np.sum(iq**2)
    # Carrier is estimated too; the epoch slip slightly perturbs its phase fit.
    assert np.sum(new.iq**2)<1e-7*np.sum(iq**2)
    assert new.gram_condition<100.


def test_local_projection_retains_controlled_weak_target_and_preserves_input():
    target=.1*observation(1,175.,201)
    iq=2*observation()+target;saved=iq.copy()
    result=cancel_other_ca_local_delay(iq,CaCancellationConfig(FS,nuisance_prns=(2,)))
    assert np.sum((result.iq-target)**2)/np.sum(target**2)<.05
    np.testing.assert_array_equal(iq,saved)


def test_radius_zero_matches_original_and_zero_invalid_cases():
    iq=observation();cfg=CaCancellationConfig(FS,nuisance_prns=(2,))
    original=cancel_other_ca(iq,cfg)
    zero=cancel_other_ca_local_delay(iq,cfg,delay_radius_samples=0)
    np.testing.assert_array_equal(original.iq,zero.iq)
    assert not np.any(cancel_other_ca_local_delay(np.zeros((20,2*N)),cfg).iq)
    with pytest.raises(ValueError):cancel_other_ca_local_delay(iq,cfg,delay_radius_samples=3)
