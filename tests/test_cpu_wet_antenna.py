"""Recovery latency semantics, unavailable metrics and clock-only controls."""

import numpy as np
import pytest

from experiments.run_cpu_wet_antenna import noise_iq, recovery_metrics, stage_at, stage_summary
from experiments.run_cpu_rain_positioning import constellation
from gnss_gpu import CpuSignalSimulator
from gnss_gpu.code_delay_cpu import estimate_ca_delay
from gnss_gpu.positioning_cpu import solve_pseudorange_position
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.wall_signal_sim import VerticalWall, simulate_wall_signal
from gnss_gpu.wet_antenna_cpu import WetAntennaResponse


def rows(available):
    return [dict(block=block, stage=stage_at((block+.5)*.001), available=value) for block, value in available]


def test_recovery_is_decided_at_end_of_five_blocks_and_no_loss_is_not_zero_latency():
    observations = rows([(0, False), (1, True), (100, False), (200, False),
                         (201, True), (202, True), (203, True), (204, True), (205, True)])
    result = recovery_metrics(observations)
    assert result["first_position_latency_ms"] == 2
    assert result["recovery_after_drying_start_ms"] == 6
    assert result["recovery_decision_time_ms"] == 206
    assert result["recovery_reason"] == "five_consecutive_positions"
    assert result["position_unavailable"] == 3
    no_loss = recovery_metrics(rows([(100, True), (200, True)]))
    assert no_loss["recovery_after_drying_start_ms"] is None
    assert no_loss["recovery_reason"] == "no_plateau_loss"


def test_gaps_rejections_and_never_recovered_are_retained():
    result = recovery_metrics(rows([(100, False), (200, True), (202, True), (203, True), (204, False), (205, True)]))
    assert result["recovery_decision_time_ms"] is None
    assert result["recovery_reason"] == "not_recovered"
    missing = stage_summary([dict(available=False, paired_up_shift_m=None)])
    assert missing["unavailable"] == 1
    assert missing["horizontal_p95_m_conditional"] is None
    assert missing["paired_up_p50_m"] is None


def test_phase_coupling_preserves_noise_power_and_independent_blocks():
    a = noise_iq(24683, 1, 0, 0.).reshape(-1, 2)
    b = noise_iq(24683, 1, 0, .7).reshape(-1, 2)
    x, y = a[:, 0]+1j*a[:, 1], b[:, 0]+1j*b[:, 1]
    np.testing.assert_allclose(y*np.exp(-.7j), x, atol=2e-15)
    np.testing.assert_allclose(np.abs(x)**2, np.abs(y)**2, atol=1e-13)
    assert not np.array_equal(a.ravel(), noise_iq(24683, 1, 1, 0.))


@pytest.mark.parametrize("response,clock,up", [
    (WetAntennaResponse(code_bias_m=.3), .3, 0.),
    (WetAntennaResponse(elevation_code_bias_m=.3), .3, .3)])
def test_noiseless_iq_distinguishes_common_clock_bias_from_elevation_dependent_position_bias(response, clock, up):
    fs = 20*CHIP_RATE
    receiver = np.array([-20., 0., 1.5])
    satellites = constellation()
    ranges = []
    for prn, satellite in enumerate(satellites, 1):
        block = simulate_wall_signal(
            CpuSignalSimulator(fs, noise_floor_db=-10000), VerticalWall(0., 1e9, 1e9+100, 0., 100.),
            receiver, [0., 0., 0.], satellite, 20460, start_sample=round(.12*fs),
            prn=prn, reference_range_m=22e6, include_reflection=False, integrate_samples=True,
            antenna_response=response)
        fitted = estimate_ca_delay(block.iq, fs, prn=prn)
        assert fitted is not None
        ranges.append(22e6+fitted.delay_m)
    solution = solve_pseudorange_position(satellites, ranges)
    assert solution is not None
    np.testing.assert_allclose(solution.position_m, receiver+[0., 0., up], atol=1e-4)
    assert solution.clock_bias_m == pytest.approx(clock, abs=1e-4)
