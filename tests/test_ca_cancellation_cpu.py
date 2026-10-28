from dataclasses import replace

import numpy as np
import pytest

from gnss_gpu.ca_cancellation_cpu import CaCancellationConfig, cancel_other_ca
from gnss_gpu.validation.tracking_residuals import generate_ca_code

FS, N = 2046000., 2046


def observation(prn=2, frequency=300., lag=37, periods=20):
    code = np.roll(np.repeat(generate_ca_code(prn), 2), lag)
    time = np.arange(periods*N).reshape(periods, N)/FS
    z = code*np.exp(2j*np.pi*frequency*time+.3j)
    return np.stack([np.column_stack((row.real, row.imag)).ravel() for row in z])


def test_blind_frequency_delay_and_amplitude_fit_removes_noiseless_nuisance():
    iq = observation()
    saved = iq.copy()
    result = cancel_other_ca(iq, CaCancellationConfig(FS, nuisance_prns=(2,)))
    assert result.status == "projected"
    assert len(result.components) == 1
    fit = result.components[0]
    assert fit.carrier_hz == pytest.approx(300., abs=1e-8)
    assert fit.delay_samples == 37
    assert fit.fitted_rms_amplitude == pytest.approx(1., abs=1e-10)
    assert np.sum(result.iq**2) < 1e-16*np.sum(iq**2)
    np.testing.assert_array_equal(iq, saved)


def test_whole_period_sign_flips_do_not_corrupt_frequency_refinement():
    iq = observation(frequency=-170., lag=-12)
    iq[1::2] *= -1
    result = cancel_other_ca(iq, CaCancellationConfig(FS, nuisance_prns=(2,)))
    assert result.components[0].carrier_hz == pytest.approx(-170., abs=1e-8)
    assert np.sum(result.iq**2) < 1e-16*np.sum(iq**2)


def test_joint_fit_of_two_distinct_nuisances_reduces_residual_energy():
    iq = 2*observation()+.7*observation(3, -420., -100)
    result = cancel_other_ca(iq, CaCancellationConfig(FS, nuisance_prns=(2, 3)))
    assert len(result.components) == 2
    assert result.removed_energy_fraction > .99
    assert result.gram_condition < 2.


def test_target_is_excluded_and_discovery_can_be_disabled_without_mutation():
    with pytest.raises(ValueError, match="excluding"):
        CaCancellationConfig(FS, nuisance_prns=(1, 2))
    iq = observation()
    result = cancel_other_ca(iq, CaCancellationConfig(FS, nuisance_prns=()))
    assert result.status == "no_components"
    np.testing.assert_array_equal(result.iq, iq)
    assert result.iq is not iq


def test_zero_and_invalid_inputs_are_explicit():
    cfg = CaCancellationConfig(FS)
    assert cancel_other_ca(np.zeros((20, 2*N)), cfg).status == "zero"
    for iq in (np.zeros((2, 2*N)), np.zeros((20, 2*N-1)), np.full((20, 2*N), np.nan)):
        with pytest.raises(ValueError):
            cancel_other_ca(iq, cfg)
    for changed in (dict(nuisance_prns=(2, 2)), dict(max_frequency_correction_hz=250.),
                    dict(min_phase_coherence=1.1), dict(discovery_periods=0)):
        with pytest.raises(ValueError):
            replace(cfg, **changed)


def test_controlled_weak_target_survives_joint_nuisance_estimation():
    target = .1*observation(1, 170., 201)
    mixture = 2*observation()+target
    result = cancel_other_ca(mixture, CaCancellationConfig(FS, nuisance_prns=(2,)))
    assert np.sum((result.iq-target)**2)/np.sum(target**2) < .001


@pytest.mark.parametrize("scale", [1e-150, 1e150])
def test_extreme_scale_preserves_estimated_component(scale):
    iq = observation()+np.random.default_rng(4).normal(0., .01, size=(20, 2*N))
    cfg = CaCancellationConfig(FS, nuisance_prns=(2,))
    reference, scaled = cancel_other_ca(iq, cfg), cancel_other_ca(iq*scale, cfg)
    assert scaled.components[0].carrier_hz == pytest.approx(reference.components[0].carrier_hz, abs=1e-9)
    np.testing.assert_allclose(scaled.iq/scale, reference.iq, atol=1e-12)
