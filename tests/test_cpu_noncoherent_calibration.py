from types import SimpleNamespace
import copy

import pytest

from experiments.validate_cpu_noncoherent_calibration import summarize_null, rethreshold, CALIBRATION_DOMAIN, EVALUATION_DOMAIN
from experiments.run_cpu_noncoherent_sensitivity import CALIBRATION_DOMAIN as OLD_CAL, EVALUATION_DOMAIN as OLD_EVAL
from experiments.audit_cpu_noncoherent_thresholds import compare_same_stream


def test_paired_null_summary_keeps_discordant_and_shared_passes():
    rows = [{"1": False, "4": False}, {"1": True, "4": False},
            {"1": False, "4": True}, {"1": True, "4": True}]
    found = summarize_null(rows)
    assert found["paired_counts"] == dict(neither=1, both=1, only_1ms=1, only_4ms=1)
    assert found["per_detector"]["1"]["accepted"] == 2
    assert found["observed_rate_difference_4ms_minus_1ms"] == 0.


def test_zero_observed_false_alarms_still_has_nonzero_upper_interval():
    result = summarize_null([{"1": False, "4": False}]*1999)
    assert result["per_detector"]["4"]["pointwise_wilson_95"][1] > 0.
    with pytest.raises(ValueError):
        summarize_null([])


def test_rethreshold_recomputes_acceptance_and_criterion_without_changing_candidates():
    outcome = dict(candidate=dict(statistic=25., carrier_hz=500., delay_samples=37, window_index=0),
                   statistic=25., accepted=False, passes_declared_criterion=False,
                   carrier_error_hz=200., initial_phase_lag_error_samples=0.)
    rows = [dict(detectors={"1": copy.deepcopy(outcome), "4": copy.deepcopy(outcome)})]
    saved = copy.deepcopy(rows)
    thresholds = {1: SimpleNamespace(calibration=SimpleNamespace(score=24.)),
                  4: SimpleNamespace(calibration=SimpleNamespace(score=25.))}
    result = rethreshold(rows, thresholds, dict(max_carrier_error_hz=250., max_initial_phase_lag_error_samples=2.))
    assert result[0]["detectors"]["1"]["passes_declared_criterion"]
    assert not result[0]["detectors"]["4"]["accepted"]  # strict tie
    assert result[0]["detectors"]["1"]["candidate"] == outcome["candidate"]
    assert rows == saved


def test_rethreshold_rejects_inconsistent_cached_score():
    rows = [dict(detectors={"1": dict(candidate=dict(statistic=1.), statistic=2.)})]
    with pytest.raises(ValueError, match="statistics"):
        rethreshold(rows, {1: SimpleNamespace(calibration=SimpleNamespace(score=0.))}, {})


def test_new_calibration_and_evaluation_seed_domains_do_not_reuse_prior_domains():
    assert len({CALIBRATION_DOMAIN, EVALUATION_DOMAIN, OLD_CAL, OLD_EVAL}) == 4


def test_same_stream_control_exposes_passes_added_by_lower_threshold():
    records = [dict(candidates={"1": dict(statistic=value), "4": dict(statistic=value)},
                    accepted={"1": value > 20., "4": value > 20.}) for value in (19., 21., 25.)]
    result = compare_same_stream(records, {"1": 24., "4": 24.}, {"1": 20., "4": 20.})
    assert result["4"]["old_accepted"] == 1
    assert result["4"]["new_accepted"] == 2
    assert result["4"]["only_new"] == 1
    assert result["4"]["only_old"] == 0


def test_same_stream_control_rejects_inconsistent_acceptance_and_empty_data():
    records = [dict(candidates={"1": dict(statistic=30.), "4": dict(statistic=30.)},
                    accepted={"1": False, "4": False})]
    with pytest.raises(ValueError, match="acceptance"):
        compare_same_stream(records, {"1": 20., "4": 20.}, {"1": 20., "4": 20.})
    with pytest.raises(ValueError):
        compare_same_stream([], {}, {})
