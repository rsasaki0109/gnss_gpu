"""Unknown carrier/delay recovery, search calibration and parallel equivalence."""

from dataclasses import replace

import numpy as np
import pytest

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.acquisition_cpu import (
    CaSearch, acquire_ca_signal, calibrate_ca_search, detect_ca_search,
)
from gnss_gpu.code_delay_cpu import estimate_ca_delay
from gnss_gpu.code_detection_cpu import ca_detection_statistic
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal

FS = 2*CHIP_RATE


def observation(carrier, delay=.31):
    # Physical propagation and finite sampling aperture, independent of fitter.
    simulator = CpuSignalSimulator(FS, noise_floor_db=-10000, intermediate_freq=carrier)
    block = simulate_wall_signal(
        simulator, VerticalWall(0., -100., 100., 0., 100.),
        [-10., 0., 2.], [0., 0., 0.], [-100., 0., 2.], 2046,
        reference_range_m=90-delay*C_LIGHT/FS, include_reflection=False,
        integrate_samples=True)
    return block.iq


@pytest.mark.parametrize("carrier", [-2500., 0., 1750.])
def test_recovers_unknown_carrier_and_fractional_delay(carrier):
    result = acquire_ca_signal(observation(carrier), CaSearch(FS, tuple(np.arange(-3000, 3001, 250))))
    assert result is not None
    assert result.carrier_hz == carrier
    assert result.delay.delay_samples == pytest.approx(.31, abs=2e-4)


def test_off_grid_carrier_is_reported_as_grid_estimate():
    result = acquire_ca_signal(observation(1820.), CaSearch(FS, (1500., 1750., 2000.)))
    assert result.carrier_hz == 1750.
    assert result.delay.fit_residual_fraction > .001


def test_single_bin_reproduces_existing_fitter():
    iq = observation(-500.)
    result = acquire_ca_signal(iq, CaSearch(FS, (-500.,)))
    expected = estimate_ca_delay(iq, FS, carrier_hz=-500.)
    assert result.delay == expected
    assert result.statistic == ca_detection_statistic(expected)


def test_parallel_and_serial_return_identical_candidates_and_do_not_mutate_iq():
    iq = observation(750.)+np.random.default_rng(5).normal(0, .7, 4092)
    saved = iq.copy()
    search = CaSearch(FS, tuple(np.arange(-3000, 3001, 250)))
    assert acquire_ca_signal(iq, search) == acquire_ca_signal(iq, search, workers=4)
    np.testing.assert_array_equal(iq, saved)


def test_full_search_maximum_exceeds_single_bin_null_and_is_calibrated():
    rng = np.random.default_rng(84)
    search = CaSearch(FS, (-2000., -1000., 0., 1000., 2000.))
    scores, single = [], []
    for _ in range(99):
        iq = rng.normal(size=4092)
        scores.append(acquire_ca_signal(iq, search).statistic)
        single.append(ca_detection_statistic(estimate_ca_delay(iq, FS)))
    assert np.all(np.asarray(scores) >= np.asarray(single)-1e-10)
    assert np.median(scores) > np.median(single)
    gate = calibrate_ca_search(scores, search, false_alarm_probability=.05)
    assert gate.calibration.score == sorted(scores)[94]
    assert gate.calibration.marginal_false_alarm_bound == .05
    assert detect_ca_search(observation(2000.), gate).accepted
    assert not detect_ca_search(np.zeros(4092), gate).accepted


def test_strict_gate_retains_rejected_candidate_and_search_metadata():
    search = CaSearch(FS, (-1000., 0., 1000.))
    iq = observation(1000.)
    score = acquire_ca_signal(iq, search).statistic
    gate = calibrate_ca_search(np.full(99, score), search)
    result = detect_ca_search(iq, gate)
    assert not result.accepted
    assert result.candidate.carrier_hz == 1000.
    assert gate.search.carrier_grid_hz == (-1000., 0., 1000.)
    with pytest.raises(ValueError, match="metadata"):
        replace(gate, search=CaSearch(4*CHIP_RATE, (0.,)))


@pytest.mark.parametrize("grid", [(), (0., 0.), (1., 0.), (np.nan,), (np.inf,),
                                    ((0.,),), (FS/2,), (-FS/2,)])
def test_invalid_grids_are_rejected(grid):
    with pytest.raises(ValueError):
        CaSearch(FS, grid)


def test_grid_is_immutable_copy_and_inputs_are_validated():
    grid = np.array([0., 1000.])
    search = CaSearch(FS, grid)
    grid[0] = 42.
    assert search.carrier_grid_hz == (0., 1000.)
    with pytest.raises(ValueError):
        acquire_ca_signal(np.zeros(4092), search, workers=0)
    with pytest.raises(ValueError):
        acquire_ca_signal(np.zeros(4091), search)
    with pytest.raises(ValueError):
        CaSearch(FS, (0.,), prn=33)
    with pytest.raises(ValueError):
        CaSearch(2.6e6, (0.,))
