from types import SimpleNamespace

import numpy as np

from experiments.run_cpu_receiver_stress import make_scene, summarize, seed_for, CALIBRATION, EVALUATION, SIGNAL


def row(status, used=False):
    return SimpleNamespace(status=status, measurement_used=used, detection_accepted=used,
                           carrier_hz=170. if used else None)


def test_short_coast_recovers_without_claiming_reacquisition():
    trace = [row("acquired", True), row("coasting"), row("coasting"), row("tracking", True)]
    result = summarize(trace, (1, 3), 170.)
    assert result["first_fresh_after_return_ms"] == 1
    assert not result["lost_during_outage"]
    assert result["reacquisition_after_loss_ms"] is None
    assert result["used_absent_blocks"] == 0


def test_loss_and_reacquisition_count_decision_block_end():
    trace = [row("acquired", True), row("coasting"), row("lost"), row("confirming"), row("acquired", True)]
    result = summarize(trace, (1, 3), 170.)
    assert result["lost_during_outage"]
    assert result["reacquisition_after_loss_ms"] == 2
    assert result["first_fresh_after_return_ms"] == 2


def test_first_acquisition_after_outage_is_not_counted_as_recovery():
    trace = [row("searching"), row("searching"), row("confirming"), row("acquired", True)]
    result = summarize(trace, (1, 2), 170.)
    assert not result["acquired_before_outage"]
    assert result["initial_acquisition_before_outage_ms"] is None
    assert result["first_fresh_after_return_ms"] is None
    assert result["reacquisition_after_loss_ms"] is None


def test_scene_preserves_time_and_pairs_noise_but_changes_primary_power():
    low, starts, seeds, outage = make_scene(35., 170., 8, 0)
    high, starts2, seeds2, _ = make_scene(55., 700., 8, 0)
    assert starts == starts2 == [2046*i for i in range(64)]
    assert seeds == seeds2
    assert low[0][0]["amplitude"] == .1
    assert high[0][0]["amplitude"] == 1.
    assert all(ch["amplitude"] == 0 for row_ in low[outage[0]:outage[1]] for ch in row_)
    for rows in (low, high):
        assert rows[0][0]["nav_bits"] == rows[-1][0]["nav_bits"]
        assert rows[0][0]["carrier_phase"] == rows[-1][0]["carrier_phase"]
    assert low[0][1]["amplitude"] == high[0][1]["amplitude"]
    assert not np.array_equal(seeds, make_scene(35., 170., 8, 1)[2])


def test_calibration_null_and_signal_seed_sets_are_disjoint():
    sets = [{seed_for(domain, 0, i) for i in range(199)} for domain in (CALIBRATION, EVALUATION, SIGNAL)]
    assert all(len(values) == 199 for values in sets)
    assert len(set.union(*sets)) == 597
