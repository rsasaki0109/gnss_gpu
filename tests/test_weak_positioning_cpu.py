from types import SimpleNamespace

import numpy as np
import pytest

from gnss_gpu.weak_positioning_cpu import C_LIGHT, resolve_ca_range, position_from_weak_tracks
from tests.test_positioning_cpu import constellation

FS = 2046000.


def test_positive_negative_and_wrapped_lags_resolve_the_same_assisted_range():
    expected = 22e6+1700.
    lag = expected*FS/C_LIGHT
    for value in (lag, lag % 2046, (lag+1023) % 2046-1023, lag-2046):
        result = resolve_ca_range(value, FS, prior_range_m=22e6, prior_bound_m=6000.)
        assert result.status == 'resolved'
        assert result.pseudorange_m == pytest.approx(expected, abs=1e-7)


def test_missing_or_ambiguous_period_is_not_silently_replaced_by_nearest():
    assert resolve_ca_range(100., FS, prior_range_m=22e6, prior_bound_m=1.).status == 'outside_prior'
    result = resolve_ca_range(100., FS, prior_range_m=22e6, prior_bound_m=400000.)
    assert result.status == 'ambiguous' and result.candidate_count >= 2
    assert result.pseudorange_m is None


def observations():
    satellites = {i+1: satellite for i, satellite in enumerate(constellation())}
    truth = np.array([120., -80., 20.])
    ranges = {i: np.linalg.norm(satellite-truth)+1500. for i, satellite in satellites.items()}
    states = {i: SimpleNamespace(measurement_used=True, status='tracking',
        delay_samples=(value*FS/C_LIGHT-137+1023) % 2046-1023, start_sample=137)
        for i, value in ranges.items()}
    priors = {i: np.linalg.norm(satellite) for i, satellite in satellites.items()}
    return truth, satellites, states, priors, {i: 6000. for i in satellites}


def test_fresh_synchronized_ranges_recover_position_clock_and_receiver_epoch():
    truth, satellites, states, priors, bounds = observations()
    result = position_from_weak_tracks(states, satellites, priors, bounds, FS)
    assert result.status == 'solved' and result.start_sample == 137
    np.testing.assert_allclose(result.solution.position_m, truth, atol=1e-7)
    assert result.solution.clock_bias_m == pytest.approx(1500., abs=1e-7)


def test_coasting_and_ambiguous_ranges_are_excluded_even_with_values_present():
    _, satellites, states, priors, bounds = observations()
    for i in range(1, 10):
        states[i].measurement_used = False
        states[i].status = 'coasting'
    result = position_from_weak_tracks(states, satellites, priors, bounds, FS)
    assert result.status == 'insufficient_ranges' and result.solution is None
    assert len(result.used_prns) == 3
    states[10].measurement_used = False
    bounds[11] = 400000.
    result = position_from_weak_tracks(states, satellites, priors, bounds, FS)
    assert (11, 'ambiguous') in result.rejected


def test_mixed_measurement_epochs_are_rejected():
    _, satellites, states, priors, bounds = observations()
    states[1].start_sample += 2046
    with pytest.raises(ValueError, match='timestamp'):
        position_from_weak_tracks(states, satellites, priors, bounds, FS)


@pytest.mark.parametrize('changed', [dict(delay_samples=np.nan), dict(sampling_freq=1.), dict(prior_bound_m=-1.)])
def test_invalid_range_inputs(changed):
    args = dict(delay_samples=1., sampling_freq=FS, prior_range_m=22e6, prior_bound_m=6000.)
    args.update(changed)
    with pytest.raises(ValueError):
        resolve_ca_range(**args)
