"""Off-grid carrier fitting and state transitions on physically sampled IQ."""

from dataclasses import replace

import numpy as np
import pytest

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import CaSearch, acquire_ca_signal, calibrate_ca_search
from gnss_gpu.signal_sim_cpu import CHIP_RATE, L1_HZ
from gnss_gpu.tracking_cpu import CpuCaTracker, refine_ca_carrier
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal

FS, COUNT = 2*CHIP_RATE, 2046


def observation(carrier=1820., delay=.31, start=0):
    simulator = CpuSignalSimulator(FS, noise_floor_db=-10000, intermediate_freq=carrier)
    return simulate_wall_signal(
        simulator, VerticalWall(0., -100., 100., 0., 100.),
        [-10., 0., 2.], [0., 0., 0.], [-100., 0., 2.], COUNT,
        start_sample=start, reference_range_m=90-delay*C_LIGHT/FS,
        include_reflection=False, integrate_samples=True).iq


def gate(grid=(1500., 1750., 2000.)):
    # Fixed synthetic threshold for strong-signal state tests, not an RF
    # false-alarm calibration claim. The experiment calibrates real null scores.
    return calibrate_ca_search(np.full(99, 20.), CaSearch(FS, grid))


@pytest.mark.parametrize("carrier", [-1820., -170., 1820.])
def test_refinement_recovers_off_grid_carrier_without_true_frequency(carrier):
    search = CaSearch(FS, tuple(np.arange(-2500., 2501., 500.)))
    iq = observation(carrier)
    coarse = acquire_ca_signal(iq, search)
    result = refine_ca_carrier(iq, search, coarse)
    assert abs(result.carrier_hz-carrier) < .2
    assert result.delay.delay_samples == pytest.approx(.31, abs=2e-4)
    assert result.statistic >= coarse.statistic
    assert abs(result.carrier_hz-carrier) < abs(coarse.carrier_hz-carrier)


def test_refinement_is_invariant_to_unknown_phase_scale_and_nav_polarity():
    search = gate().search
    iq = observation()
    z = iq.reshape(-1, 2)
    rotated = (z[:, 0]+1j*z[:, 1])*np.exp(.91j)*(-.3)
    other = np.column_stack([rotated.real, rotated.imag]).ravel()
    a = refine_ca_carrier(iq, search, acquire_ca_signal(iq, search))
    b = refine_ca_carrier(other, search, acquire_ca_signal(other, search))
    assert a.carrier_hz == pytest.approx(b.carrier_hz, abs=.1)
    assert a.delay.delay_samples == pytest.approx(b.delay.delay_samples, abs=1e-6)


def test_refinement_respects_grid_boundary_and_single_bin():
    iq = observation(1820.)
    for grid in ((1500.,), (1000., 1500.)):
        search = CaSearch(FS, grid)
        coarse = acquire_ca_signal(iq, search)
        result = refine_ca_carrier(iq, search, coarse)
        assert result.carrier_hz == 1500.


def test_detection_gate_uses_coarse_score_even_if_refinement_crosses_threshold():
    iq = observation()
    search = gate().search
    coarse = acquire_ca_signal(iq, search)
    refined = refine_ca_carrier(iq, search, coarse)
    assert refined.statistic > coarse.statistic
    threshold = calibrate_ca_search(np.full(99, (coarse.statistic+refined.statistic)/2), search)
    result = CpuCaTracker(threshold).update(iq, start_sample=0)
    assert not result.detection_accepted and not result.measurement_used
    assert result.refined_candidate is None
    assert result.status == "searching"
    assert result.detection_statistic == coarse.statistic


def test_tracker_acquires_coasts_loses_and_reacquires_with_all_statuses_visible():
    tracker = CpuCaTracker(gate(), confirmation_blocks=1, coast_blocks=2)
    a = tracker.update(observation(), start_sample=0)
    b = tracker.update(-observation(start=COUNT), start_sample=COUNT)
    c = tracker.update(np.zeros(4092), start_sample=2*COUNT)
    d = tracker.update(np.zeros(4092), start_sample=3*COUNT)
    e = tracker.update(observation(start=4*COUNT), start_sample=4*COUNT)
    assert [x.status for x in (a, b, c, d, e)] == ["acquired", "tracking", "coasting", "lost", "acquired"]
    assert c.carrier_hz is not None and not c.measurement_used
    assert d.carrier_hz is None and d.delay_samples is None
    assert e.missed_blocks == 0
    assert c.delay_samples == pytest.approx(b.delay_samples-COUNT*b.carrier_hz/L1_HZ)


def test_searching_reports_consecutive_unused_blocks():
    tracker = CpuCaTracker(gate(), confirmation_blocks=1)
    for index in range(3):
        result = tracker.update(np.zeros(4092), start_sample=index*COUNT)
        assert result.status == "searching"
        assert result.missed_blocks == index+1


def test_if_is_removed_from_doppler_code_aid_and_wrap_is_continuous():
    tracker = CpuCaTracker(gate((-5500., -5000., -4500.)), confirmation_blocks=1, intermediate_freq=-5000.)
    a = tracker.update(observation(-5000., delay=1022.999), start_sample=0)
    b = tracker.update(np.zeros(4092), start_sample=COUNT)
    assert b.delay_samples == pytest.approx(a.delay_samples, abs=1e-5)
    tracker = CpuCaTracker(gate((-5500., -5000., -4500.)), confirmation_blocks=1)
    a = tracker.update(observation(-5000., delay=1022.999), start_sample=0)
    b = tracker.update(np.zeros(4092), start_sample=COUNT)
    expected = (a.delay_samples-COUNT*a.carrier_hz/L1_HZ+COUNT/2) % COUNT-COUNT/2
    assert b.delay_samples == pytest.approx(expected)
    assert b.delay_samples < 0


@pytest.mark.parametrize("changed", [dict(carrier=2400.), dict(delay=10.)])
def test_accepted_but_discontinuous_measurement_does_not_steer_state(changed):
    tracker = CpuCaTracker(gate((1500., 2000., 2500.)), confirmation_blocks=1, max_frequency_step_hz=100.)
    first = tracker.update(observation(), start_sample=0)
    second = tracker.update(observation(start=COUNT, **changed), start_sample=COUNT)
    assert second.detection_accepted
    assert not second.measurement_used and second.status == "coasting"
    assert second.refined_candidate is not None
    assert second.carrier_hz == first.carrier_hz


def test_timestamps_and_invalid_iq_do_not_corrupt_state_and_reset_allows_gap():
    tracker = CpuCaTracker(gate(), confirmation_blocks=1)
    tracker.update(observation(), start_sample=0)
    with pytest.raises(ValueError, match="contiguous"):
        tracker.update(observation(), start_sample=3*COUNT)
    with pytest.raises(ValueError, match="1 ms"):
        tracker.update(np.zeros(10), start_sample=COUNT)
    assert tracker.update(observation(start=COUNT), start_sample=COUNT).status == "tracking"
    tracker.reset()
    assert tracker.update(observation(start=5*COUNT), start_sample=5*COUNT).status == "acquired"


def test_accelerating_geometry_is_followed_without_truth_aiding():
    tracker = CpuCaTracker(gate((4500., 5000., 5500.)), confirmation_blocks=1)
    simulator = CpuSignalSimulator(FS, noise_floor_db=-10000)
    wall = VerticalWall(0., -100., 100., 0., 100.)
    smoothed_errors = []
    for index in range(25):
        block = simulate_wall_signal(
            simulator, wall, [-20., 0., 2.], [5., 0., 0.], [-20000020., 0., 2.], COUNT,
            start_sample=index*COUNT, reference_range_m=20000000.,
            transmitter_velocity_mps=[1000., 0., 0.], rx_acceleration_mps2=[200., 0., 0.],
            include_reflection=False, integrate_samples=True)
        result = tracker.update(block.iq*(-1 if index % 2 else 1), start_sample=index*COUNT)
        truth = float(np.median(block.direct_doppler_hz))
        assert result.measurement_used
        assert abs(result.refined_candidate.carrier_hz-truth) < .3
        smoothed_errors.append(result.carrier_hz-truth)
    # Constant-frequency prediction plus exponential smoothing has acceleration
    # lag; it is bounded here, not claimed to estimate acceleration exactly.
    assert max(abs(error) for error in smoothed_errors) < 4.


@pytest.mark.parametrize("kwargs", [dict(gain=0), dict(gain=1.1), dict(coast_blocks=0),
                                      dict(max_frequency_step_hz=np.nan), dict(workers=0),
                                      dict(max_delay_step_samples=-1), dict(intermediate_freq=np.inf)])
def test_invalid_tracker_configuration(kwargs):
    with pytest.raises(ValueError):
        CpuCaTracker(gate(), confirmation_blocks=1, **kwargs)


def test_refinement_validates_configuration_and_input():
    search = gate().search
    iq = observation()
    candidate = acquire_ca_signal(iq, search)
    for kwargs in (dict(tolerance_hz=0), dict(max_iterations=0)):
        with pytest.raises(ValueError):
            refine_ca_carrier(iq, search, candidate, **kwargs)
    with pytest.raises(ValueError, match="match"):
        refine_ca_carrier(iq, search, replace(candidate, carrier_hz=123.))
    assert refine_ca_carrier(np.zeros(4092), search, candidate) is None
