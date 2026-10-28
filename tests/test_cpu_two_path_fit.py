"""Missing/ambiguous fits and matched-subset comparison in two-path reporting."""

import pytest

from experiments.run_cpu_two_path_fit import summarize_rows


def test_missing_and_worse_selected_estimates_remain_visible_with_equal_subset_baseline():
    rows = [dict(detection_accepted=True, single_error_m=2., selected_error_m=1., selected_model="two_path"),
            dict(detection_accepted=True, single_error_m=3., selected_error_m=None, selected_model="ambiguous"),
            dict(detection_accepted=True, single_error_m=1., selected_error_m=40., selected_model="two_path"),
            dict(detection_accepted=False, single_error_m=None, selected_error_m=None, selected_model="no_fit")]
    summary = summarize_rows(rows)
    assert summary["blocks"] == 4 and summary["rejected_blocks"] == 1
    assert summary["missing_delay_blocks"] == 2
    assert summary["ambiguous_blocks"] == 1
    assert summary["two_path_blocks"] == 2
    assert summary["single_rmse_m_same_selected_blocks"] == pytest.approx((2.5)**.5)
    assert summary["selected_rmse_m_conditional"] == pytest.approx((1601/2)**.5)
    assert summary["selected_delay_error_over_30_m"] == 1


def test_no_baseline_fit_and_all_missing_have_no_zero_error_claim():
    summary = summarize_rows([dict(detection_accepted=True, single_error_m=None,
                                   selected_error_m=None, selected_model="no_fit")])
    assert summary["detected_blocks"] == 1
    assert summary["baseline_delay_blocks"] == 0
    assert summary["missing_delay_blocks"] == 1
    assert summary["single_rmse_m_conditional_baseline"] is None
    assert summary["selected_rmse_m_conditional"] is None
