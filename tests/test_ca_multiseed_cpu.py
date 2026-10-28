from dataclasses import replace

import numpy as np
import pytest

from gnss_gpu.ca_cancellation_cpu import CaCancellationConfig, cancel_other_ca
from gnss_gpu.ca_multiseed_cpu import cancel_other_ca_multiseed
from gnss_gpu.ca_local_delay_cpu import _refine_local_delays
from tests.test_ca_cancellation_cpu import observation, FS, N


def test_neighboring_seed_recovers_component_rejected_by_single_seed():
    # An initial transient moves the discovery peak into an adjacent cell.
    # The full record still supports 600 Hz; no frequency is passed to inference.
    iq = observation(frequency=600.)
    iq[:4] = .1*observation(frequency=750.)[:4]
    saved = iq.copy()
    config = CaCancellationConfig(FS, nuisance_prns=(2,), carrier_grid_hz=(250., 500., 750., 1000.))
    old = cancel_other_ca(iq, config)
    new = cancel_other_ca_multiseed(iq, config)
    assert not old.components
    assert len(new.components) == 1
    fit = new.components[0]
    assert fit.coarse_carrier_hz == 500.
    assert fit.carrier_hz == pytest.approx(600., abs=.02)
    assert fit.delay_samples == 37
    assert new.removed_energy_fraction > .995
    np.testing.assert_array_equal(iq, saved)


def test_one_candidate_reproduces_historical_processor_exactly():
    iq = 2*observation()+.7*observation(3, -420., -100)
    config = CaCancellationConfig(FS, nuisance_prns=(2, 3))
    old = cancel_other_ca(iq, config)
    new = cancel_other_ca_multiseed(iq, config, coarse_candidates=1)
    assert new.components == old.components
    assert new.removed_energy_fraction == old.removed_energy_fraction
    np.testing.assert_array_equal(new.iq, old.iq)


def test_local_delay_fit_retains_controlled_target_with_multiple_seeds():
    target = .1*observation(1, 175., 201)
    iq = 2*observation()+target
    config = CaCancellationConfig(FS, nuisance_prns=(2,))
    result = _refine_local_delays(iq, config, cancel_other_ca_multiseed(iq, config))
    assert len(result.components) == 1
    assert np.sum((result.iq-target)**2)/np.sum(target**2) < .05


@pytest.mark.parametrize('scale', [1e-150, 1e150])
def test_scale_invariance(scale):
    iq = observation()+np.random.default_rng(54).normal(0., .01, size=(20, 2*N))
    config = CaCancellationConfig(FS, nuisance_prns=(2,))
    reference = cancel_other_ca_multiseed(iq, config)
    scaled = cancel_other_ca_multiseed(iq*scale, config)
    assert scaled.components[0].carrier_hz == pytest.approx(reference.components[0].carrier_hz, abs=1e-9)
    np.testing.assert_allclose(scaled.iq/scale, reference.iq, atol=1e-12)


def test_zero_disabled_and_invalid_inputs():
    config = CaCancellationConfig(FS)
    assert cancel_other_ca_multiseed(np.zeros((20, 2*N)), config).status == 'zero'
    iq = observation()
    result = cancel_other_ca_multiseed(iq, replace(config, nuisance_prns=()))
    np.testing.assert_array_equal(result.iq, iq)
    for count in (0, -1, 1.5):
        with pytest.raises(ValueError):
            cancel_other_ca_multiseed(iq, config, coarse_candidates=count)
    for invalid in (np.zeros((2, 2*N)), np.full((20, 2*N), np.nan), iq.astype(complex)):
        with pytest.raises(ValueError):
            cancel_other_ca_multiseed(invalid, config)
