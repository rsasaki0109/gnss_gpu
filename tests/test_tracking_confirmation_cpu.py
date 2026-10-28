"""Consecutive acquisition evidence, delay wrap and pending-state lifecycle."""

import numpy as np
import pytest

from gnss_gpu.acquisition_cpu import CaSearch, calibrate_ca_search
from gnss_gpu.tracking_cpu import CpuCaTracker
from test_tracking_cpu import COUNT, FS, gate, observation


def test_default_requires_two_consistent_candidates_and_exposes_pending_without_state():
    tracker = CpuCaTracker(gate())
    first = tracker.update(observation(), start_sample=0)
    assert first.status == "confirming" and first.confirmation_count == 1
    assert first.detection_accepted and not first.measurement_used
    assert first.carrier_hz is None and first.delay_samples is None
    assert first.refined_candidate is not None
    second = tracker.update(-observation(start=COUNT), start_sample=COUNT)
    assert second.status == "acquired" and second.measurement_used
    assert second.confirmation_count == 2 and second.missed_blocks == 0
    third = tracker.update(observation(start=2*COUNT), start_sample=2*COUNT)
    assert third.status == "tracking" and third.confirmation_count == 0


def test_rejection_breaks_chain_instead_of_accumulating_separated_hits():
    tracker = CpuCaTracker(gate(), confirmation_blocks=3)
    for index in (0, 1):
        result = tracker.update(observation(start=index*COUNT), start_sample=index*COUNT)
        assert result.confirmation_count == index+1
    assert tracker.update(np.zeros(4092), start_sample=2*COUNT).status == "searching"
    result = tracker.update(observation(start=3*COUNT), start_sample=3*COUNT)
    assert result.confirmation_count == 1 and result.carrier_hz is None
    tracker.update(observation(start=4*COUNT), start_sample=4*COUNT)
    assert tracker.update(observation(start=5*COUNT), start_sample=5*COUNT).status == "acquired"


@pytest.mark.parametrize("changed", [dict(carrier=2400.), dict(delay=10.)])
def test_inconsistent_hit_replaces_pending_chain(changed):
    tracker = CpuCaTracker(gate((1500., 2000., 2500.)), max_frequency_step_hz=100.)
    tracker.update(observation(), start_sample=0)
    result = tracker.update(observation(start=COUNT, **changed), start_sample=COUNT)
    assert result.status == "confirming" and result.confirmation_count == 1
    assert result.detection_accepted and result.carrier_hz is None
    result = tracker.update(observation(start=2*COUNT, **changed), start_sample=2*COUNT)
    assert result.status == "acquired" and result.confirmation_count == 2


def test_confirmation_uses_wrapped_code_prediction_with_known_if():
    # A known IF would falsely move the predicted code by 0.065 samples
    # per block if treated as L1 Doppler. Wrapped delays straddle +/-N/2.
    search = CaSearch(FS, (49500., 50000., 50500.))
    threshold = calibrate_ca_search(np.full(99, 20.), search)
    tracker = CpuCaTracker(threshold, intermediate_freq=50000., max_delay_step_samples=.03)
    tracker.update(observation(50000., delay=1022.99), start_sample=0)
    second = tracker.update(observation(50000., delay=-1022.99, start=COUNT), start_sample=COUNT)
    assert second.status == "acquired"


def test_loss_requires_fresh_confirmation_and_isolated_hit_stays_unpublished():
    tracker = CpuCaTracker(gate(), coast_blocks=1)
    tracker.update(observation(), start_sample=0)
    assert tracker.update(observation(start=COUNT), start_sample=COUNT).status == "acquired"
    assert tracker.update(np.zeros(4092), start_sample=2*COUNT).status == "lost"
    isolated = tracker.update(observation(start=3*COUNT), start_sample=3*COUNT)
    assert isolated.status == "confirming" and isolated.carrier_hz is None
    assert tracker.update(np.zeros(4092), start_sample=4*COUNT).status == "searching"
    assert tracker.update(observation(start=5*COUNT), start_sample=5*COUNT).confirmation_count == 1


def test_invalid_update_preserves_pending_and_reset_discards_it():
    tracker = CpuCaTracker(gate())
    tracker.update(observation(), start_sample=0)
    with pytest.raises(ValueError):
        tracker.update(np.zeros(10), start_sample=COUNT)
    with pytest.raises(ValueError, match="contiguous"):
        tracker.update(observation(), start_sample=3*COUNT)
    assert tracker.update(observation(start=COUNT), start_sample=COUNT).status == "acquired"
    tracker.reset()
    first = tracker.update(observation(start=8*COUNT), start_sample=8*COUNT)
    assert first.status == "confirming" and first.confirmation_count == 1


@pytest.mark.parametrize("value", [0, -1, 1.5, np.nan, np.inf, True])
def test_invalid_confirmation_count_is_rejected(value):
    with pytest.raises(ValueError):
        CpuCaTracker(gate(), confirmation_blocks=value)
