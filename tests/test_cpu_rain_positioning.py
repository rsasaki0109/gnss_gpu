"""Availability denominators and paired position-shift reporting."""

import pytest

from experiments.run_cpu_rain_positioning import summarize_positions


def test_unavailable_positions_stay_in_denominator_and_are_not_zero_errors():
    rows = [dict(available=True, horizontal_error_m=3., position_error_m=5., paired_position_shift_m=2.),
            dict(available=True, horizontal_error_m=4., position_error_m=200., paired_position_shift_m=None),
            dict(available=False, horizontal_error_m=None, position_error_m=None, paired_position_shift_m=None)]
    summary = summarize_positions(rows)
    assert summary["trials"] == 3
    assert summary["available"] == 2 and summary["unavailable"] == 1
    assert summary["horizontal_p50_m_conditional"] == pytest.approx(3.5)
    assert summary["available_error_over_100_m"] == 1
    assert summary["position_max_m_conditional"] == 200
    assert summary["paired_trials"] == 1
    assert summary["paired_shift_p50_m"] == 2.


def test_no_solution_or_pair_is_missing_not_perfect_accuracy():
    summary = summarize_positions([dict(available=False, paired_position_shift_m=None)])
    assert summary["unavailable"] == 1
    assert summary["horizontal_p50_m_conditional"] is None
    assert summary["position_p95_m_conditional"] is None
    assert summary["paired_shift_p95_m"] is None
