"""Strong local recovery clusters replace stale references, not single jumps."""

from dataclasses import replace

import numpy as np
import pytest

from experiments.run_cpu_rain_positioning import constellation
from gnss_gpu.position_quality_cpu import PositionGateConfig, PositionPublicationGate, PositionRecoveryConfig


def ranges(sat, x=0.):
    return np.linalg.norm(sat-[x, 0., 0.], axis=1)


def initialized(**kwargs):
    sat = constellation()
    gate = PositionPublicationGate(PositionGateConfig(confirmation_blocks=2),
                                   recovery_config=PositionRecoveryConfig(**kwargs))
    gate.update(sat, ranges(sat), time_s=0.)
    assert gate.update(sat, ranges(sat), time_s=.001).accepted
    return sat, gate


def test_blocking_reference_is_replaced_only_after_strong_window_and_new_confirmation():
    sat, gate = initialized()
    for index in range(24):
        result = gate.update(sat, ranges(sat, 50.), time_s=.002+index*.001)
        assert not result.accepted and not result.reanchored
        assert result.recovery_count == index+1
    result = gate.update(sat, ranges(sat, 50.), time_s=.026)
    assert result.reanchored and result.reason == "confirming"
    assert result.position_m is None and result.recovery_count == 25
    assert result.anchor_source == "recovery_cluster"
    accepted = gate.update(sat, ranges(sat, 50.), time_s=.027)
    assert accepted.accepted and accepted.anchor_source == "published_median"
    np.testing.assert_allclose(accepted.position_m, [50, 0, 0], atol=1e-6)


def test_bad_epoch_or_time_gap_breaks_recovery_evidence():
    sat, gate = initialized()
    for index in range(24):
        gate.update(sat, ranges(sat, 50.), time_s=.002+index*.001)
    bad = ranges(sat, 50.)
    bad[0] += 300.
    assert not gate.update(sat, bad, time_s=.026).reanchored
    result = gate.update(sat, ranges(sat, 50.), time_s=.027)
    assert result.recovery_count == 1 and not result.reanchored
    gap = gate.update(sat, ranges(sat, 50.), time_s=.04)
    assert gap.recovery_count == 1 and not gap.reanchored


def test_ordinary_quality_is_not_sufficient_to_reanchor():
    sat, gate = initialized()
    for index in range(30):
        # Six good ranges satisfy the ordinary gate but not the seven-range
        # recovery requirement; a blocking old anchor cannot be bypassed.
        result = gate.update(sat[:6], ranges(sat[:6], 50.), time_s=.002+index*.001)
        assert not result.reanchored
        assert result.recovery_count == 0


def test_cluster_spread_and_elapsed_span_are_required():
    sat, gate = initialized()
    for index, phase in enumerate(np.linspace(0., 2*np.pi, 25)):
        result = gate.update(sat, ranges(sat, 50.+15*np.sin(phase)), time_s=.002+index*.001)
    assert result.recovery_count == 25 and result.recovery_radius_m > 10.36
    assert not result.reanchored
    sat, gate = initialized()
    for index in range(25):
        result = gate.update(sat, ranges(sat, 50.), time_s=.002+index*.00001)
    assert result.recovery_count == 25 and not result.reanchored


def test_published_reference_uses_median_while_output_remains_current_candidate():
    sat = constellation()
    gate = PositionPublicationGate(PositionGateConfig(confirmation_blocks=1),
                                   recovery_config=PositionRecoveryConfig(anchor_window_blocks=3))
    gate.update(sat, ranges(sat), time_s=0.)
    gate.update(sat, ranges(sat, 10.), time_s=.001)
    result = gate.update(sat, ranges(sat, 25.), time_s=.002)
    assert result.accepted and result.anchor_jump_m == pytest.approx(20., abs=1e-6)
    np.testing.assert_allclose(result.position_m, [25, 0, 0], atol=1e-6)


def test_invalid_input_preserves_window_and_reset_discards_it():
    sat, gate = initialized()
    for index in range(24):
        gate.update(sat, ranges(sat, 50.), time_s=.002+index*.001)
    with pytest.raises(ValueError):
        gate.update(sat, [1.], time_s=.026)
    assert gate.update(sat, ranges(sat, 50.), time_s=.026).reanchored
    gate.reset()
    result = gate.update(sat, ranges(sat, 100.), time_s=0.)
    assert result.reason == "confirming" and result.anchor_source is None and result.recovery_count == 0


def test_coherent_wrong_cluster_can_still_pass_not_an_integrity_bound():
    sat, gate = initialized()
    for index in range(26):
        result = gate.update(sat, ranges(sat, 200.), time_s=.002+index*.001)
    assert result.accepted
    np.testing.assert_allclose(result.position_m, [200, 0, 0], atol=1e-6)


@pytest.mark.parametrize("kwargs", [dict(window_blocks=1), dict(min_span_s=0), dict(max_gap_s=0),
                                      dict(radius_m=-1), dict(anchor_window_blocks=0)])
def test_invalid_recovery_settings(kwargs):
    with pytest.raises(ValueError):
        replace(PositionRecoveryConfig(), **kwargs)


def test_recovery_cannot_loosen_ordinary_quality_checks():
    with pytest.raises(ValueError):
        PositionPublicationGate(recovery_config=PositionRecoveryConfig(min_observations=5))
