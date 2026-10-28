"""Redundancy, observable geometry/residual checks and publication state."""

from dataclasses import replace

import numpy as np
import pytest

from experiments.run_cpu_rain_positioning import constellation
from gnss_gpu.positioning_cpu import PositionSolution, solve_pseudorange_position
from gnss_gpu.position_quality_cpu import PositionGateConfig, PositionPublicationGate, position_quality


def ranges(sat, receiver=(0., 0., 0.), clock=0.):
    return np.linalg.norm(sat-np.asarray(receiver), axis=1)+clock


def test_tetrahedron_dop_has_independent_closed_form_and_zero_residual_dof():
    sat = 22e6*np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])/np.sqrt(3)
    solution = solve_pseudorange_position(sat, ranges(sat))
    quality = position_quality(sat, solution)
    assert quality.pdop == pytest.approx(1.5)
    assert quality.hdop == pytest.approx(np.sqrt(1.5))
    assert quality.vdop == pytest.approx(np.sqrt(.75))
    assert quality.degrees_of_freedom == 0 and quality.postfit_sigma_m is None


def test_four_satellite_exact_fit_is_not_publishable_despite_tiny_residuals():
    sat = constellation()[:4]
    result = PositionPublicationGate().update(sat, ranges(sat)+[300., -200., 100., -150.], time_s=0.)
    assert result.candidate is not None and result.quality is not None
    assert result.quality.residual_rms_m < 1e-6
    assert np.linalg.norm(result.candidate.position_m) > 100
    assert not result.accepted and result.reason == "too_few_satellites"
    assert result.position_m is None


def test_bad_geometry_and_large_residuals_are_distinct_rejections():
    sat = constellation()
    poor = sat.copy()
    poor[:, 2] *= 1000
    poor *= 22e6/np.linalg.norm(poor, axis=1)[:, None]
    result = PositionPublicationGate().update(poor, ranges(poor), time_s=0.)
    assert result.reason == "pdop" and result.quality.pdop > 6
    biased = ranges(sat)
    biased[0] += 300.
    result = PositionPublicationGate().update(sat, biased, time_s=0.)
    assert result.reason == "residual_rms" and result.quality.residual_rms_m > 15


def test_confirm_then_reject_jump_without_moving_anchor_and_require_new_confirmation():
    sat = constellation()
    gate = PositionPublicationGate(PositionGateConfig(confirmation_blocks=2))
    a = gate.update(sat, ranges(sat), time_s=0.)
    b = gate.update(sat, ranges(sat), time_s=.001)
    jump = gate.update(sat, ranges(sat, [100., 0., 0.]), time_s=.002)
    c = gate.update(sat, ranges(sat), time_s=.003)
    d = gate.update(sat, ranges(sat), time_s=.004)
    assert [a.reason, b.reason, jump.reason, c.reason, d.reason] == ["confirming", "accepted", "anchor_jump", "confirming", "accepted"]
    assert jump.anchor_jump_m == pytest.approx(100., abs=1e-5)
    assert jump.position_m is None and jump.clock_bias_m is None
    np.testing.assert_allclose(d.position_m, 0., atol=1e-6)


def test_old_anchor_expires_and_bootstrap_can_reacquire_new_consistent_position():
    sat = constellation()
    gate = PositionPublicationGate(PositionGateConfig(confirmation_blocks=2))
    gate.update(sat, ranges(sat), time_s=0.)
    assert gate.update(sat, ranges(sat), time_s=.001).accepted
    a = gate.update(sat, ranges(sat, [100, 0, 0]), time_s=3.)
    b = gate.update(sat, ranges(sat, [100, 0, 0]), time_s=3.001)
    assert a.reason == "confirming" and a.anchor_age_s is None
    assert b.accepted
    np.testing.assert_allclose(b.position_m, [100, 0, 0], atol=1e-6)


def test_missing_epoch_breaks_confirmation_and_invalid_input_does_not_advance_time():
    sat = constellation()
    gate = PositionPublicationGate(PositionGateConfig(confirmation_blocks=2))
    gate.update(sat, ranges(sat), time_s=0.)
    missing = gate.update([], [], time_s=.001)
    assert missing.reason == "too_few_satellites" and missing.candidate is None
    assert gate.update(sat, ranges(sat), time_s=.002).confirmation_count == 1
    with pytest.raises(ValueError):
        gate.update(sat, [1.], time_s=.003)
    assert gate.update(sat, ranges(sat), time_s=.003).accepted
    with pytest.raises(ValueError, match="strictly"):
        gate.update(sat, ranges(sat), time_s=.003)
    gate.reset()
    assert gate.update(sat, ranges(sat), time_s=0.).reason == "confirming"


def test_inconsistent_bootstrap_restarts_and_coherent_wrong_position_can_still_pass():
    sat = constellation()
    gate = PositionPublicationGate(PositionGateConfig(confirmation_blocks=2))
    gate.update(sat, ranges(sat), time_s=0.)
    a = gate.update(sat, ranges(sat, [200, 0, 0]), time_s=.001)
    b = gate.update(sat, ranges(sat, [200, 0, 0]), time_s=.002)
    assert a.reason == "confirming" and a.confirmation_count == 1
    assert b.accepted
    # A self-consistent wrong range field can initialize the gate: no ground
    # truth or absolute integrity guarantee is present in this interface.
    np.testing.assert_allclose(b.position_m, [200, 0, 0], atol=1e-6)


def test_rank_deficient_diagnostics_are_unavailable():
    sat = np.tile([22e6, 0, 0], (6, 1))
    candidate = PositionSolution(np.zeros(3), 0., np.zeros(6), 1)
    assert position_quality(sat, candidate) is None


@pytest.mark.parametrize("kwargs", [dict(min_observations=4), dict(max_pdop=np.nan),
                                      dict(max_residual_rms_m=-1), dict(jump_allowance_m=-1),
                                      dict(max_anchor_age_s=0), dict(confirmation_blocks=0)])
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(PositionGateConfig(), **kwargs)
