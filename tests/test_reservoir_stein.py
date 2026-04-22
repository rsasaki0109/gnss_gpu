from __future__ import annotations

import numpy as np
import pytest

from gnss_gpu.reservoir_stein import (
    ReservoirSteinConfig,
    effective_sample_size,
    normalize_log_weights,
    reservoir_stein_update,
    rbf_median_bandwidth,
    stein_rejuvenate_particles,
    weighted_reservoir_indices,
)


def test_normalize_log_weights_is_stable_for_large_offsets():
    weights = normalize_log_weights(np.array([1000.0, 999.0, -np.inf]))

    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > weights[1] > weights[2]
    assert weights[2] == pytest.approx(0.0)


def test_effective_sample_size_matches_uniform_case():
    assert effective_sample_size(np.ones(5)) == pytest.approx(5.0)


def test_weighted_reservoir_indices_keeps_elites_and_unique_indices():
    log_weights = np.log(np.array([0.01, 0.02, 0.03, 0.44, 0.50]))

    indices = weighted_reservoir_indices(
        log_weights,
        reservoir_size=3,
        elite_fraction=1.0 / 3.0,
        seed=3,
    )

    assert indices[0] == 4
    assert len(indices) == 3
    assert len(set(indices.tolist())) == 3


def test_rbf_median_bandwidth_is_positive_for_repeated_particles():
    particles = np.zeros((4, 2), dtype=np.float64)

    bandwidth = rbf_median_bandwidth(particles)

    assert bandwidth > 0.0


def test_stein_rejuvenate_particles_repels_without_score_gradient():
    particles = np.array([[-1.0], [1.0]], dtype=np.float64)
    gradients = np.zeros_like(particles)

    updated, bandwidth = stein_rejuvenate_particles(
        particles,
        gradients,
        step_size=0.5,
        bandwidth=2.0,
    )

    assert bandwidth == pytest.approx(2.0)
    assert updated[0, 0] < particles[0, 0]
    assert updated[1, 0] > particles[1, 0]


def test_reservoir_stein_update_selects_and_transports_subset():
    particles = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )
    log_weights = np.log(np.array([0.05, 0.10, 0.50, 0.25, 0.10]))
    gradients = -particles

    result = reservoir_stein_update(
        particles,
        log_weights,
        gradients,
        ReservoirSteinConfig(
            reservoir_size=3,
            elite_fraction=1.0 / 3.0,
            stein_steps=2,
            stein_step_size=0.1,
            seed=7,
        ),
    )

    assert result.particles.shape == (3, 2)
    assert result.source_indices[0] == 2
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.ess_before == pytest.approx(1.0 / np.sum(normalize_log_weights(log_weights) ** 2))
    assert len(result.bandwidths) == 2
    assert np.all(np.isfinite(result.particles))
