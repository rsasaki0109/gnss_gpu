"""Noiseless geometry, carrier-frame noise coupling and signed reporting."""

import numpy as np
import pytest

from experiments.run_cpu_rain_positioning import constellation
from experiments.run_cpu_wet_delay_positioning import geometric_control, paired_noise, summarize_positions
from gnss_gpu.signal_sim_cpu import L1_HZ
from gnss_gpu.wall_signal_sim import C_LIGHT


def test_geometric_control_separates_height_and_clock_and_preserves_sign():
    receiver = np.array([-20., -1., 1.5])
    a = geometric_control(receiver, constellation(), .3)
    b = geometric_control(receiver, constellation(), -.3)
    zero = geometric_control(receiver, constellation(), 0.)
    np.testing.assert_allclose(zero.position_m, receiver, atol=1e-7)
    assert a.position_m[2]-receiver[2] > .3
    assert a.clock_bias_m > .3
    np.testing.assert_allclose(a.position_m-receiver, -(b.position_m-receiver), atol=1e-6)


def test_phase_paired_noise_preserves_marginal_power_and_independent_frames():
    base = paired_noise(0, 0, 1, 0, 0.).reshape(-1, 2)
    rotated = paired_noise(0, 0, 1, 0, .13).reshape(-1, 2)
    a, b = base[:, 0]+1j*base[:, 1], rotated[:, 0]+1j*rotated[:, 1]
    np.testing.assert_allclose(b*np.exp(2j*np.pi*L1_HZ*.13/C_LIGHT), a, atol=2e-15)
    np.testing.assert_allclose(np.abs(a)**2, np.abs(b)**2, atol=1e-13)
    assert not np.array_equal(base.ravel(), paired_noise(0, 0, 1, 1, 0.))


def test_signed_shifts_and_missing_positions_are_not_replaced_by_zero():
    rows = [dict(available=True, error_up_m=-3., horizontal_error_m=4., position_error_m=5.,
                 paired_position_shift_m=1., paired_up_shift_m=-1., paired_clock_shift_m=-2.),
            dict(available=True, error_up_m=6., horizontal_error_m=8., position_error_m=10.,
                 paired_position_shift_m=3., paired_up_shift_m=3., paired_clock_shift_m=6.),
            dict(available=False, paired_position_shift_m=None, paired_up_shift_m=None, paired_clock_shift_m=None)]
    summary = summarize_positions(rows)
    assert summary["trials"] == 3 and summary["unavailable"] == 1
    assert summary["vertical_abs_p95_m_conditional"] == pytest.approx(5.85)
    assert summary["paired_up_shift_p50_m"] == 1.
    assert summary["paired_up_shift_p05_m"] == pytest.approx(-.8)
    assert summary["paired_up_shift_p95_m"] == pytest.approx(2.8)
    assert summary["paired_clock_shift_p50_m"] == 2.
    missing = summarize_positions(rows[2:])
    assert missing["paired_up_shift_p50_m"] is None
    assert missing["vertical_abs_p95_m_conditional"] is None
