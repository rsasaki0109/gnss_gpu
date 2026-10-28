"""Finite-wall experiment controls, paired wrapping and honest denominators."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.run_cpu_multipath_tracking import (
    COUNT, FS, MODES, paired_delay_bias, summarize_rows, wall_observations,
)
from gnss_gpu.wall_signal_sim import C_LIGHT


def test_wall_reflection_present_then_absent_with_exact_path_isolation_controls():
    before = wall_observations(0)
    after = wall_observations(350*COUNT)
    assert np.all(before["ideal_rhcp"].reflection_visible)
    assert not np.any(after["ideal_rhcp"].reflection_visible)
    assert np.all(before["direct"].direct_visible)
    assert not np.array_equal(before["direct"].iq, before["ideal_rhcp"].iq)
    assert not np.array_equal(before["ideal_rhcp"].iq, before["lhcp_leakage_90deg"].iq)
    for mode in MODES:
        np.testing.assert_array_equal(after[mode].iq, after["direct"].iq)
        np.testing.assert_array_equal(before[mode].direct_range_m, before["direct"].direct_range_m)


def test_visibility_transition_is_between_evaluated_endpoints():
    # Independent intersection geometry: x=0 image-path parameter, then y/z.
    rx0 = np.array([-20., -20., 1.5])
    velocity = np.array([1., 20., 0.])
    image = np.array([1e7, 1e7, 1e7])
    for block in (250, 280):
        t = (block*COUNT+np.arange(COUNT))/FS
        receiver = rx0+velocity*t[:, None]
        u = -receiver[:, 0]/(image[0]-receiver[:, 0])
        points = receiver+u[:, None]*(image-receiver)
        expected = ((points[:, 1] >= -30.) & (points[:, 1] <= 5.)
                    & (points[:, 2] >= 0.) & (points[:, 2] <= 100.))
        np.testing.assert_array_equal(wall_observations(block*COUNT)["ideal_rhcp"].reflection_visible, expected)


def test_paired_delay_difference_wraps_period_and_requires_two_used_measurements():
    direct = SimpleNamespace(measurement_used=True, delay_samples=COUNT/2-.1)
    composite = SimpleNamespace(measurement_used=True, delay_samples=-COUNT/2+.1)
    assert paired_delay_bias(direct, composite) == pytest.approx(.2*C_LIGHT/FS)
    composite.measurement_used = False
    assert paired_delay_bias(direct, composite) is None
    direct.measurement_used, composite.measurement_used = False, True
    assert paired_delay_bias(direct, composite) is None
    direct.measurement_used, direct.delay_samples = True, None
    assert paired_delay_bias(direct, composite) is None


def test_summaries_include_unused_states_but_label_measured_and_paired_errors_conditionally():
    rows = [dict(status="confirming", measurement_used=False, delay_error_m=None, paired_incremental_bias_m=None),
            dict(status="acquired", measurement_used=True, delay_error_m=4., paired_incremental_bias_m=1.),
            dict(status="coasting", measurement_used=False, delay_error_m=1000., paired_incremental_bias_m=None),
            dict(status="lost", measurement_used=False, delay_error_m=None, paired_incremental_bias_m=None),
            dict(status="tracking", measurement_used=True, delay_error_m=-6., paired_incremental_bias_m=-2.)]
    result = summarize_rows(rows)
    assert result["blocks"] == 5
    assert result["measurement_used_blocks"] == 2
    assert result["unused_blocks"] == 3
    assert result["delay_rmse_m_conditional_used"] == pytest.approx(np.sqrt(26.))
    assert result["paired_measurement_blocks"] == 2
    assert result["paired_bias_rmse_m_conditional"] == pytest.approx(np.sqrt(2.5))
    assert result["paired_bias_max_abs_m_conditional"] == 2.
    assert result["states"]["coasting"] == 1


def test_empty_used_set_is_missing_error_not_perfect_accuracy():
    result = summarize_rows([dict(status="searching", measurement_used=False,
                                  delay_error_m=None, paired_incremental_bias_m=None)])
    assert result["measurement_used_blocks"] == 0
    assert result["delay_rmse_m_conditional_used"] is None
    assert result["paired_bias_rmse_m_conditional"] is None
    assert result["paired_bias_max_abs_m_conditional"] is None
