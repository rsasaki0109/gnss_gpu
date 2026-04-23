from __future__ import annotations

import numpy as np
import pytest

from gnss_gpu.dd_likelihood import (
    dd_log_likelihood_gradient,
    dd_log_likelihood_gradients,
    dd_pseudorange_residual_and_design,
)
from gnss_gpu.dd_pseudorange import DDPseudorangeResult
from gnss_gpu.reservoir_stein import ReservoirSteinConfig, reservoir_stein_update


def _synthetic_dd_result(true_pos: np.ndarray) -> DDPseudorangeResult:
    sat_k = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    sat_ref = np.repeat(np.array([[0.0, -10.0, 0.0]], dtype=np.float64), 3, axis=0)
    base_range_k = np.linalg.norm(sat_k, axis=1)
    base_range_ref = np.linalg.norm(sat_ref, axis=1)
    expected = (
        np.linalg.norm(sat_k - true_pos, axis=1)
        - np.linalg.norm(sat_ref - true_pos, axis=1)
        - base_range_k
        + base_range_ref
    )
    return DDPseudorangeResult(
        dd_pseudorange_m=expected,
        sat_ecef_k=sat_k,
        sat_ecef_ref=sat_ref,
        base_range_k=base_range_k,
        base_range_ref=base_range_ref,
        dd_weights=np.ones(3, dtype=np.float64),
        ref_sat_ids=("G01", "G01", "G01"),
        n_dd=3,
    )


def test_dd_pseudorange_residual_and_design_zero_at_truth():
    true_pos = np.array([1.0, 2.0, 1.5], dtype=np.float64)
    dd = _synthetic_dd_result(true_pos)

    residuals, design = dd_pseudorange_residual_and_design(dd, true_pos)

    np.testing.assert_allclose(residuals, np.zeros(3), atol=1.0e-12)
    assert design.shape == (3, 3)


def test_dd_log_likelihood_gradient_points_toward_better_position():
    true_pos = np.array([1.0, 2.0, 1.5], dtype=np.float64)
    pos = true_pos + np.array([0.5, -0.4, 0.2], dtype=np.float64)
    dd = _synthetic_dd_result(true_pos)

    grad = dd_log_likelihood_gradient(dd, pos, sigma_m=1.0).gradient

    assert float(np.dot(grad, true_pos - pos)) > 0.0
    old_cost = np.sum(dd_pseudorange_residual_and_design(dd, pos)[0] ** 2)
    new_cost = np.sum(dd_pseudorange_residual_and_design(dd, pos + 0.1 * grad)[0] ** 2)
    assert new_cost < old_cost


def test_dd_log_likelihood_gradient_huber_downweights_outlier():
    true_pos = np.array([1.0, 2.0, 1.5], dtype=np.float64)
    dd = _synthetic_dd_result(true_pos)
    dd.dd_pseudorange_m = dd.dd_pseudorange_m.copy()
    dd.dd_pseudorange_m[0] += 100.0

    result = dd_log_likelihood_gradient(dd, true_pos, sigma_m=1.0, huber_k_m=1.0)

    assert result.robust_weights[0] < result.robust_weights[1]
    assert result.n_dd == 3


def test_dd_log_likelihood_gradients_returns_particle_matrix():
    true_pos = np.array([1.0, 2.0, 1.5], dtype=np.float64)
    dd = _synthetic_dd_result(true_pos)
    particles = np.vstack([true_pos, true_pos + np.array([0.2, 0.0, 0.0])])

    gradients = dd_log_likelihood_gradients(dd, particles, sigma_m=1.0)

    assert gradients.shape == (2, 3)
    np.testing.assert_allclose(gradients[0], np.zeros(3), atol=1.0e-12)
    assert np.linalg.norm(gradients[1]) > 0.0


def test_dd_log_likelihood_gradient_rejects_bad_sigma():
    dd = _synthetic_dd_result(np.zeros(3, dtype=np.float64))

    with pytest.raises(ValueError):
        dd_log_likelihood_gradient(dd, np.zeros(3, dtype=np.float64), sigma_m=0.0)


def test_dd_gradients_can_drive_reservoir_stein_particles_toward_truth():
    true_pos = np.array([1.0, 2.0, 1.5], dtype=np.float64)
    dd = _synthetic_dd_result(true_pos)
    particles = np.array(
        [
            true_pos + [0.8, -0.4, 0.2],
            true_pos + [0.6, -0.3, 0.1],
            true_pos + [-0.4, 0.2, -0.2],
            true_pos + [0.2, 0.2, 0.4],
        ],
        dtype=np.float64,
    )
    gradients = dd_log_likelihood_gradients(dd, particles, sigma_m=1.0)
    log_weights = -np.sum((particles - true_pos) ** 2, axis=1)

    result = reservoir_stein_update(
        particles,
        log_weights,
        gradients,
        ReservoirSteinConfig(
            reservoir_size=4,
            elite_fraction=0.25,
            stein_steps=1,
            stein_step_size=0.1,
            repulsion_scale=0.0,
            seed=5,
        ),
    )

    before = np.mean(np.linalg.norm(particles - true_pos, axis=1))
    after = np.mean(np.linalg.norm(result.particles - true_pos, axis=1))
    assert after < before
