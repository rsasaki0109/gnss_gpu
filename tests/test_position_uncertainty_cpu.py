"""Weighted estimation, covariance scaling and independent-noise calibration."""

import numpy as np
import pytest

from experiments.run_cpu_rain_positioning import constellation
from gnss_gpu.position_uncertainty_cpu import solve_weighted_pseudorange_position as solve
from gnss_gpu.positioning_cpu import solve_pseudorange_position


def test_equal_sigmas_reproduce_equal_weight_position_and_clock():
    sat = constellation()
    ranges = np.linalg.norm(sat-[12., -3., 5.], axis=1)+37.+np.linspace(-2, 2, len(sat))
    old = solve_pseudorange_position(sat, ranges)
    new = solve(sat, ranges, np.full(len(sat), 4.))
    assert old is not None and new is not None
    np.testing.assert_allclose(new.solution.position_m, old.position_m, atol=1e-6)
    assert new.solution.clock_bias_m == pytest.approx(old.clock_bias_m, abs=1e-6)
    assert new.degrees_of_freedom == len(sat)-4
    np.testing.assert_allclose(new.covariance_m2, new.covariance_m2.T, atol=1e-12)
    assert np.linalg.eigvalsh(new.covariance_m2).min() > 0


def test_scaling_sigmas_changes_covariance_not_estimated_position():
    sat = constellation()
    ranges = np.linalg.norm(sat, axis=1)+np.arange(len(sat))
    a, b = solve(sat, ranges, np.ones(len(sat))), solve(sat, ranges, np.full(len(sat), 3.))
    assert a is not None and b is not None
    np.testing.assert_allclose(a.solution.position_m, b.solution.position_m, atol=1e-6)
    np.testing.assert_allclose(b.covariance_m2, 9*a.covariance_m2, rtol=1e-7)
    assert b.normalized_residual_energy == pytest.approx(a.normalized_residual_energy/9, rel=1e-6)


def test_heterogeneous_independent_noise_improves_fit_and_matches_covariance():
    rng = np.random.default_rng(98743)
    sat, truth = constellation(), np.array([12., -3., 5.])
    sigma = np.array([2.]*8+[40.]*4)
    weighted, equal, energies, covered = [], [], [], []
    for _ in range(300):
        ranges = np.linalg.norm(sat-truth, axis=1)+17.+rng.normal(size=len(sat))*sigma
        a, b = solve(sat, ranges, sigma), solve_pseudorange_position(sat, ranges)
        assert a is not None and b is not None
        error = a.solution.position_m-truth
        weighted.append(error@error)
        equal.append(np.sum((b.position_m-truth)**2))
        energy = error@np.linalg.solve(a.covariance_m2[:3, :3], error)
        energies.append(energy)
        covered.append(energy <= 7.814727903251179)
    assert np.mean(weighted) < .2*np.mean(equal)
    assert 2.5 < np.mean(energies) < 3.5
    assert .91 < np.mean(covered) < .99


def test_four_ranges_have_covariance_but_no_redundant_residual_degrees():
    sat = constellation()[:4]
    result = solve(sat, np.linalg.norm(sat-[4, 2, 1], axis=1)+8, np.ones(4))
    assert result is not None and result.degrees_of_freedom == 0
    assert result.normalized_residual_energy < 1e-10
    assert np.trace(result.covariance_m2) > 0


def test_degenerate_and_insufficient_geometry_have_no_solution():
    assert solve(np.zeros((4, 3)), np.ones(4), np.ones(4)) is None
    sat = constellation()[:3]
    assert solve(sat, np.linalg.norm(sat, axis=1), np.ones(3)) is None


@pytest.mark.parametrize("sigma", [np.zeros(12), np.full(12, -1), np.full(12, np.nan), np.ones(11)])
def test_invalid_uncertainties_are_rejected(sigma):
    sat = constellation()
    with pytest.raises(ValueError):
        solve(sat, np.linalg.norm(sat, axis=1), sigma)
