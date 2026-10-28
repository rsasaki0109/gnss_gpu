"""Position/clock separation and unavailable solutions, without native CUDA."""

import numpy as np
import pytest

from gnss_gpu.positioning_cpu import solve_pseudorange_position


def constellation():
    azimuth = np.deg2rad(np.arange(15., 360., 30.))
    elevation = np.deg2rad([20, 40, 65, 30, 50, 75, 25, 45, 70, 35, 55, 80])
    return 22e6*np.column_stack([np.cos(elevation)*np.sin(azimuth),
                                np.cos(elevation)*np.cos(azimuth), np.sin(elevation)])


def test_full_position_and_common_clock_are_recovered_from_ranges():
    satellites = constellation()
    truth, clock = np.array([-20., 5., 1.5]), 137.
    measured = np.linalg.norm(satellites-truth, axis=1)+clock
    solution = solve_pseudorange_position(satellites, measured, initial_position_m=[100., -100., 50.])
    assert solution is not None
    np.testing.assert_allclose(solution.position_m, truth, atol=1e-7)
    assert solution.clock_bias_m == pytest.approx(clock, abs=1e-7)
    np.testing.assert_allclose(solution.residuals_m, 0., atol=1e-7)


def test_common_range_change_goes_into_clock_and_not_position():
    satellites = constellation()
    ranges = np.linalg.norm(satellites, axis=1)
    a = solve_pseudorange_position(satellites, ranges)
    b = solve_pseudorange_position(satellites, ranges+7.)
    np.testing.assert_allclose(a.position_m, b.position_m, atol=1e-7)
    assert b.clock_bias_m-a.clock_bias_m == pytest.approx(7., abs=1e-7)


def test_insufficient_rank_and_nonconvergence_are_unavailable_not_fake_position():
    satellites = constellation()
    assert solve_pseudorange_position(satellites[:3], np.full(3, 22e6)) is None
    assert solve_pseudorange_position(np.tile(satellites[0], (5, 1)), np.full(5, 22e6)) is None
    assert solve_pseudorange_position(satellites, np.full(12, 22e6), initial_position_m=[100, 100, 100], max_iterations=1) is None


def test_invalid_ranges_are_rejected():
    with pytest.raises(ValueError):
        solve_pseudorange_position(constellation(), np.full(12, np.nan))
