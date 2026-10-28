from types import SimpleNamespace
import json

import numpy as np
import pytest

from experiments.run_cpu_receiver_sensitivity import wilson_interval, score_trace, aggregate, load_frozen_wide_threshold
from gnss_gpu.acquisition_cpu import CaSearch, calibrate_ca_search
from dataclasses import asdict


def row(status, carrier=None):
    return SimpleNamespace(status=status, carrier_hz=carrier, measurement_used=status in ("acquired", "tracking"),
                           detection_accepted=status in ("acquired", "tracking"))


def test_wilson_extremes_retain_uncertainty_for_small_replica_count():
    low, high = wilson_interval(0, 12)
    assert low == pytest.approx(0.) and high == pytest.approx(.242494, abs=1e-6)
    low2, high2 = wilson_interval(12, 12)
    assert low2 == pytest.approx(1-high) and high2 == pytest.approx(1.)
    assert wilson_interval(6, 12)[0] < .5 < wilson_interval(6, 12)[1]


@pytest.mark.parametrize("successes,trials", [(13, 12), (-1, 12), (1.5, 12), (0, 0)])
def test_invalid_binomial_counts(successes, trials):
    with pytest.raises(ValueError):
        wilson_interval(successes, trials)


def test_carrier_criterion_does_not_count_wrong_or_post_deadline_acquisitions():
    incorrect = score_trace([row("acquired", 500.), row("lost"), row("acquired", 700.)], (1, 2), 700.)
    missing = score_trace([row("searching"), row("searching"), row("acquired", 700.)], (1, 2), 700.)
    assert incorrect["acquired_before_outage"]
    assert not incorrect["acquired_before_outage_with_carrier_error_le_100hz"]
    assert missing["first_acquisition_carrier_error_hz"] is None
    assert not missing["acquired_before_outage_with_carrier_error_le_100hz"]


def test_aggregate_keeps_failed_replicas_in_event_denominators():
    records = [score_trace([row("acquired", 700.), row("lost"), row("tracking", 700.)], (1, 2), 700.),
               score_trace([row("searching"), row("searching"), row("searching")], (1, 2), 700.)]
    rows = [dict(nominal_primary_cn0_db_hz=35., primary_doppler_hz=700., **record) for record in records]
    result = aggregate(rows)[0]
    assert result["replicas"] == 2
    assert result["acquisition_fraction"] == .5
    assert result["carrier_criterion_fraction"] == .5
    assert result["conditional_median_initial_acquisition_ms"] == 1.


def test_frozen_gate_load_preserves_metadata_and_rejects_different_search(tmp_path):
    expected = calibrate_ca_search(np.full(199, 20.), CaSearch(2046000., (-1000., -500., 0., 500., 1000.)))
    path = tmp_path/"report.json"
    data = dict(thresholds=dict(wide=asdict(expected)))
    path.write_text(json.dumps(data))
    assert load_frozen_wide_threshold(path) == expected
    data["thresholds"]["wide"]["search"]["carrier_grid_hz"] = [-500., 0., 500.]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="five-bin"):
        load_frozen_wide_threshold(path)
