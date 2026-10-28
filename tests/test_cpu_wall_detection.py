"""Held-out detector evaluation and seed-domain separation."""

import pytest

pytest.importorskip("numba")

from experiments import run_cpu_wall_detection as experiment


def test_noise_splits_cannot_overlap_signal_pose_keys():
    assert experiment.CALIBRATION_DOMAIN != experiment.EVALUATION_DOMAIN
    assert experiment.CALIBRATION_DOMAIN not in (0, 1, 2)
    assert experiment.EVALUATION_DOMAIN not in (0, 1, 2)
    a = experiment._null_score((12, experiment.CALIBRATION_DOMAIN, 0, 2.046e6))
    b = experiment._null_score((12, experiment.EVALUATION_DOMAIN, 0, 2.046e6))
    assert a["statistic"] != b["statistic"]


def test_null_scores_are_exact_in_serial_and_spawn_workers():
    serial = experiment._null_trials(12, experiment.CALIBRATION_DOMAIN, 4, 2.046e6, 1)
    spawned = experiment._null_trials(12, experiment.CALIBRATION_DOMAIN, 4, 2.046e6, 2)
    assert serial == spawned


def test_evaluation_size_cannot_retune_calibrated_threshold():
    kwargs = dict(calibration_trials=39, false_alarm_probability=.1, signal_trials=1,
                  samples_per_chip=2, root_seed=52)
    full = experiment.run_detection_experiment(evaluation_trials=60, **kwargs)
    short = experiment.run_detection_experiment(evaluation_trials=5, **kwargs)
    assert full["calibration"] == short["calibration"]
    assert full["metadata"]["threshold"] == short["metadata"]["threshold"]
    assert full["trials"] == short["trials"]
    assert full["heldout_noise"][:5] == short["heldout_noise"]
    assert len(full["calibration"]) == 39 and len(full["heldout_noise"]) == 60
    assert len(full["trials"]) == 36
    for row in full["summary"]:
        assert row["trials"] == 3
        assert row["accepted"]+row["rejected"] == 3
        if row["condition"] != "noise_only":
            assert row["accepted_good_fraction_all"]+row["accepted_wrong_fraction_all"] == pytest.approx(row["acceptance_fraction"])
        else:
            assert row["accepted_good_fraction_all"] is None
            assert row["conditional_error_count"] == 0
    expected = sum(row["accepted"] for row in full["heldout_noise"])
    assert full["metadata"]["heldout_false_alarms"] == expected
    assert full["metadata"]["measured_false_alarm_fraction"] == expected/60


def test_wilson_interval_includes_nonzero_upper_limit_for_zero_alarms():
    lower, upper = experiment.wilson_interval(0, 100)
    assert lower == pytest.approx(0., abs=1e-15)
    assert upper == pytest.approx(.0369934982, abs=1e-9)
    lower_all, upper_all = experiment.wilson_interval(100, 100)
    assert lower_all == pytest.approx(1-upper)
    assert upper_all == 1.


@pytest.mark.parametrize("kwargs", [dict(calibration_trials=0), dict(evaluation_trials=0),
                                       dict(calibration_trials=10), dict(false_alarm_probability=0),
                                       dict(false_alarm_probability=1), dict(root_seed=None),
                                       dict(samples_per_chip=1)])
def test_invalid_detection_experiment_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        experiment.run_detection_experiment(**kwargs)
