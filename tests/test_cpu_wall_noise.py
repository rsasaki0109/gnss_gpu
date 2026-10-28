"""Monte Carlo pairing, physical noise calibration and unbiased accounting."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("numba")

from experiments import run_cpu_wall_noise as experiment


def test_noise_power_recovers_specified_cn0_for_complex_samples():
    fs, cn0 = 20.46e6, 45.
    rng = np.random.default_rng(12)
    noise = rng.normal(0., experiment.noise_sigma(fs, cn0), (200000, 2))
    measured_cn0 = 10*np.log10(fs/noise.var(axis=0).sum())
    assert measured_cn0 == pytest.approx(cn0, abs=.05)
    assert abs(np.corrcoef(noise.T)[0, 1]) < .01


def test_conditions_share_exact_noise_and_cn0_levels_share_standard_draws(monkeypatch):
    received = []

    def estimate(iq, fs):
        received.append(iq.copy())
        return SimpleNamespace(delay_m=4.5, fit_residual_fraction=.5)

    monkeypatch.setattr(experiment, "estimate_ca_delay", estimate)
    direct, full = np.arange(16)/10, np.arange(16)/10+.25
    job = (0, 0, 12, (35., 45.), 20.46e6, (direct, full), 4.5, 30.)
    rows = experiment._estimate_trial(job)
    np.testing.assert_allclose(received[0]-received[2], direct, atol=1e-14)
    np.testing.assert_allclose(received[1]-received[2], full, atol=1e-14)
    np.testing.assert_allclose(received[2], received[5]*np.sqrt(10), atol=1e-13)
    assert all(row["expected_delay_m"] is None and row["bad_or_missing"] is None
               for row in rows if row["condition"] == "noise_only")
    assert rows == experiment._estimate_trial(job)


def test_wrong_peaks_and_missing_estimates_stay_in_statistics():
    rows = [dict(cn0_dbhz=35., condition="direct", error_m=error,
                 status="no_estimate" if error is None else "peak_returned")
            for error in (0., 40., 100000., None)]
    rows.append(dict(cn0_dbhz=35., condition="noise_only", error_m=None, status="peak_returned"))
    summaries = experiment.summarize_trials(rows, 30.)
    direct = next(row for row in summaries if row["condition"] == "direct")
    assert direct["trials"] == 4 and direct["error_count"] == 3
    assert direct["no_estimate"] == 1
    assert direct["bad_or_missing_fraction"] == .75
    assert direct["median_abs_error_m"] == 40.
    assert direct["p95_abs_error_m"] == 100000.
    assert direct["rmse_m"] == pytest.approx(np.sqrt((40**2+100000**2)/3))
    control = next(row for row in summaries if row["condition"] == "noise_only")
    assert control["peak_return_fraction"] == 1.
    assert control["rmse_m"] is None and control["bad_or_missing_fraction"] is None


def test_serial_and_spawned_trials_are_exactly_reproducible():
    kwargs = dict(cn0_values=(55.,), trials=1, root_seed=123, samples_per_chip=2)
    serial = experiment.run_experiment(workers=1, **kwargs)
    parallel = experiment.run_experiment(workers=2, **kwargs)
    assert serial["trials"] == parallel["trials"]
    assert serial["summary"] == parallel["summary"]
    assert serial["per_pose_summary"] == parallel["per_pose_summary"]
    assert len(serial["trials"]) == 9  # three poses, three conditions
    assert len(serial["per_pose_summary"]) == 9
    assert sum(row["trials"] for row in serial["summary"]) == 9
    assert serial["metadata"]["workers"] == 1
    assert parallel["metadata"]["workers"] == 2


def test_code_period_errors_wrap_without_losing_wrong_peaks():
    half = experiment.C_LIGHT*.0005
    assert experiment.wrapped_error_m(-half+.5, half-.5) == pytest.approx(1.)
    assert experiment.wrapped_error_m(100000., 4.5) == pytest.approx(99995.5)


@pytest.mark.parametrize("kwargs", [dict(trials=0), dict(workers=0),
                                       dict(cn0_values=()), dict(cn0_values=(45., 45.)),
                                       dict(cn0_values=(np.inf,)), dict(samples_per_chip=1),
                                       dict(threshold_m=0), dict(root_seed=None)])
def test_invalid_experiment_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        experiment.run_experiment(**kwargs)
