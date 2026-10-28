"""Two-path inference against physical aperture IQ and rank-degenerate controls."""

import numpy as np
import pytest

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.multipath_fit_cpu import fit_ca_two_paths
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal


def observation(spc=20, delay=.17, include_reflection=True):
    fs = spc*CHIP_RATE
    receiver = np.array([-20., -20., 1.5])
    transmitter = np.array([-1e7, 1e7, 1e7])
    reference = np.linalg.norm(transmitter-receiver)-delay*C_LIGHT/fs
    return simulate_wall_signal(
        CpuSignalSimulator(fs, noise_floor_db=-10000), VerticalWall(0., -30., 5., 0., 100.),
        receiver, [0., 0., 0.], transmitter, 1023*spc,
        reference_range_m=reference, include_reflection=include_reflection,
        integrate_samples=True)


@pytest.mark.parametrize("delay", [-.27, .17, .31, 10229.8])
def test_resolved_noiseless_paths_recover_earliest_geometric_delay(delay):
    block = observation(delay=delay)
    result = fit_ca_two_paths(block.iq, 20*CHIP_RATE)
    expected_gap = (block.reflected_range_m[0]-block.direct_range_m[0])*20*CHIP_RATE/C_LIGHT
    assert result.selected_model == "two_path"
    assert result.delay_samples == pytest.approx(delay, abs=2e-6)
    gap = (result.second_delay_samples-result.first_delay_samples+10230) % 20460-10230
    assert gap == pytest.approx(expected_gap, abs=2e-6)
    assert result.fit_residual_fraction < 1e-12
    assert result.bic_improvement > 0
    assert result.delay_condition_number is not None


def test_subsample_affine_template_ambiguity_is_not_claimed_as_mitigation():
    result = fit_ca_two_paths(observation(spc=2, delay=0.).iq, 2*CHIP_RATE)
    assert result.fit_residual_fraction < 1e-12  # a perfect fit alone is insufficient
    assert result.selected_model == "ambiguous"
    assert result.delay_samples is None
    assert result.delay_condition_number is None
    assert result.first_delay_std_samples is None
    assert "rank" in result.reason


def test_single_path_is_retained_instead_of_overfitting_numerical_roundoff():
    block = observation(include_reflection=False)
    result = fit_ca_two_paths(block.iq, 20*CHIP_RATE)
    assert result.selected_model == "single_path"
    assert result.delay_samples == result.single.delay_samples
    assert result.bic_improvement < 0


def test_unknown_nav_polarity_complex_phase_and_scale_do_not_supply_true_delay():
    block = observation()
    iq = block.iq.reshape(-1, 2)
    z = (iq[:, 0]+1j*iq[:, 1])*(-.3)*np.exp(.93j)
    data = np.column_stack([z.real, z.imag]).ravel()
    result = fit_ca_two_paths(data, 20*CHIP_RATE)
    assert result.selected_model == "two_path"
    assert result.delay_samples == pytest.approx(.17, abs=2e-6)


def test_noisy_single_path_can_retain_single_model_without_true_path_count():
    iq = observation(include_reflection=False).iq
    result = fit_ca_two_paths(iq+np.random.default_rng(71).normal(0., .1, iq.size), 20*CHIP_RATE)
    assert result.selected_model == "single_path"


@pytest.mark.parametrize("kwargs", [dict(search_radius_chips=0), dict(max_excess_chips=0),
                                      dict(max_excess_chips=.001), dict(max_delay_std_samples=0),
                                      dict(starts=0), dict(max_iterations=0), dict(search_radius_chips=300)])
def test_invalid_fit_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_ca_two_paths(observation().iq, 20*CHIP_RATE, **kwargs)


def test_zero_and_invalid_iq_use_existing_input_validation():
    assert fit_ca_two_paths(np.zeros(4092), 2*CHIP_RATE) is None
    with pytest.raises(ValueError):
        fit_ca_two_paths(np.zeros(100), 2*CHIP_RATE)
    with pytest.raises(ValueError):
        fit_ca_two_paths(np.full(4092, np.nan), 2*CHIP_RATE)
