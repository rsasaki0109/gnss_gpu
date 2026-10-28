"""Confirmation experiment denominators and acquisition latency conventions."""

from experiments.run_cpu_tracking_confirmation import summarize_confirmation


def rows():
    return [dict(block=index, signal_outage=80 <= index < 100, status="searching",
                 measurement_used=False, detection_accepted=False,
                 tracked_delay_error_m=None, tracked_error_hz=None) for index in range(200)]


def test_confirmation_waiting_rejections_and_coasting_remain_in_all_block_counts():
    observations = rows()
    for index in (0, 85, 100):
        observations[index].update(status="confirming", detection_accepted=True)
    for index in (1, 101):
        observations[index].update(status="acquired", measurement_used=True,
                                   detection_accepted=True, tracked_delay_error_m=31., tracked_error_hz=501.)
    observations[80].update(status="coasting", tracked_delay_error_m=100., tracked_error_hz=1000.)
    summary = summarize_confirmation(observations)
    assert summary["signal_present_blocks"] == 180
    assert summary["unused_signal_present_blocks"] == 178
    assert summary["used_signal_present_blocks"] == 2
    assert summary["initial_acquisition_latency_ms"] == 2
    assert summary["reacquisition_latency_ms"] == 2
    assert summary["false_acquisition_events"] == 0
    assert summary["pending_signal_absent_blocks"] == 1
    assert summary["coarse_gate_passed_signal_absent_blocks"] == 1
    assert summary["used_signal_absent_blocks"] == 0
    assert summary["used_signal_present_delay_error_over_30_m"] == 2
    assert summary["used_signal_present_carrier_error_over_500_hz"] == 2


def test_false_state_is_counted_and_missing_acquisition_is_not_zero_latency():
    observations = rows()
    observations[85].update(status="acquired", measurement_used=True,
                            detection_accepted=True, tracked_delay_error_m=100000., tracked_error_hz=9000.)
    summary = summarize_confirmation(observations)
    assert summary["false_acquisition_events"] == 1
    assert summary["used_signal_absent_blocks"] == 1
    assert summary["initial_acquisition_latency_ms"] is None
    assert summary["reacquisition_latency_ms"] is None
