import numpy as np
import pytest

from experiments.run_cpu_other_prn_nulls import scene, gate_candidate, CALIBRATION_DOMAIN, EVALUATION_DOMAIN, SIGNAL_DOMAIN
from experiments.run_cpu_batch_receiver import render
from gnss_gpu.noncoherent_acquisition_cpu import CaNoncoherentCandidate


def test_target_absent_profiles_and_near_far_amplitude_scaling():
    base, starts, seeds, _ = scene("other_prns", CALIBRATION_DOMAIN, 0)
    high, starts2, seeds2, _ = scene("near_far_20db", CALIBRATION_DOMAIN, 0)
    assert starts == starts2 and seeds == seeds2
    assert {channel["prn"] for row in base for channel in row} == set(range(2, 13))
    for a, b in zip(base[0], high[0]):
        assert b["amplitude"] == pytest.approx(10*a["amplitude"])
        assert (a["code_phase"], a["carrier_phase"], a["doppler_hz"]) == (b["code_phase"], b["carrier_phase"], b["doppler_hz"])
    assert all(not row for row in scene("white", CALIBRATION_DOMAIN, 0)[0])


def test_present_target_does_not_change_other_satellite_parameters():
    null, starts, seeds, _ = scene("other_prns", SIGNAL_DOMAIN, 0)
    present, starts2, seeds2, nominal = scene("other_prns", SIGNAL_DOMAIN, 0, target_cn0=37., target_doppler=700.)
    assert starts == starts2 and seeds == seeds2
    assert present[0][1:] == null[0]
    assert present[0][0]["prn"] == 1 and present[0][0]["nav_bits"] == [1, -1, 1]
    np.testing.assert_array_equal(render("batch", present, starts, seeds, -10000.),
                                  render("individual", present, starts, seeds, -10000.))


def test_domains_independent_and_gate_has_no_profile_or_truth_argument():
    assert len({CALIBRATION_DOMAIN, EVALUATION_DOMAIN, SIGNAL_DOMAIN}) == 3
    candidate = CaNoncoherentCandidate(0., 0, 0, 20.)
    assert gate_candidate(candidate, 19.).accepted
    assert not gate_candidate(candidate, 20.).accepted
    assert not gate_candidate(None, 20.).accepted
    with pytest.raises(ValueError):
        scene("unknown", SIGNAL_DOMAIN, 0)
