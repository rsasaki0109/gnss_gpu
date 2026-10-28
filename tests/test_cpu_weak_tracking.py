from experiments.run_cpu_weak_tracking import continuous_scene, summarize, SIGNAL_DOMAIN
from experiments.run_cpu_batch_receiver import COUNT


def test_continuous_scene_preserves_absolute_epochs_and_only_target_is_muted():
    rows, starts, seeds = continuous_scene('near_far_20db', SIGNAL_DOMAIN, 0, target=True, outage=True)
    assert len(rows) == len(starts) == len(seeds) == 240
    assert all(b-a == COUNT for a, b in zip(starts, starts[1:]))
    assert len(set(seeds)) == 240
    for block, channels in enumerate(rows):
        assert len(channels) == 12
        assert (channels[0]['amplitude'] == 0.) == (80 <= block < 120)
        assert len(channels[0]['nav_bits']) >= 13
        assert channels[1:] == rows[0][1:]


def test_late_initial_acquisition_is_not_misreported_as_reacquisition():
    trace = [dict(relative_start_ms=i*20, available_after_ms=(i+1)*20, status='acquired' if i == 7 else 'searching',
        measurement_used=i == 7, target_present=i not in (4, 5), detection_accepted=i == 7,
        carrier_hz=175. if i == 7 else None, carrier_error_hz=5. if i == 7 else None,
        lag_error_samples=0. if i == 7 else None) for i in range(12)]
    result = summarize(trace, target=True, outage=True)
    assert result['first_acquisition_available_ms'] == 160
    assert not result['acquired_before_outage']
    assert result['reacquisition_available_after_return_ms'] is None
    assert not result['lost_during_outage']
    assert result['present_windows'] == 10
