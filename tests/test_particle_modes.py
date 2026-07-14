import numpy as np
import pytest

from gnss_gpu.particle_modes import extract_particle_modes, select_particle_mode


def _cloud(seed=4):
    rng = np.random.default_rng(seed)
    left = rng.normal([0.0, 0.0, 0.0], 0.22, size=(700, 3))
    right = rng.normal([8.0, 0.0, 0.0], 0.25, size=(300, 3))
    particles = np.vstack((left, right))
    log_weights = np.zeros(len(particles), dtype=np.float64)
    return particles, log_weights


def test_extract_particle_modes_keeps_two_peaks_separate():
    particles, log_weights = _cloud()
    result = extract_particle_modes(
        particles,
        log_weights,
        voxel_size_m=0.5,
        min_core_cell_mass=0.003,
        assignment_radius_m=1.5,
    )

    assert len(result.modes) == 2
    assert result.assigned_mass == pytest.approx(1.0)
    assert result.modes[0].mass == pytest.approx(0.7, abs=0.02)
    assert result.modes[1].mass == pytest.approx(0.3, abs=0.02)
    assert np.linalg.norm(result.modes[0].position) < 0.1
    assert np.linalg.norm(result.modes[1].position - [8.0, 0.0, 0.0]) < 0.1
    # The global mean lies in the low-density valley, not in either peak.
    assert 2.2 < result.weighted_mean[0] < 2.6


def test_temporal_prior_can_select_reachable_secondary_mode():
    particles, log_weights = _cloud()
    result = extract_particle_modes(
        particles,
        log_weights,
        voxel_size_m=0.5,
        min_core_cell_mass=0.003,
        assignment_radius_m=1.5,
    )
    selection = select_particle_mode(
        result,
        predicted_position=np.array([8.1, 0.0, 0.0]),
        prediction_sigma_m=1.0,
        min_selected_mass=0.2,
        min_score_ratio=2.0,
    )

    assert selection.accepted
    assert selection.mode_index == 1
    assert np.linalg.norm(selection.position - [8.0, 0.0, 0.0]) < 0.1


def test_selector_abstains_on_ambiguous_equal_modes():
    rng = np.random.default_rng(9)
    particles = np.vstack(
        (
            rng.normal([-2.0, 0.0, 0.0], 0.2, size=(500, 3)),
            rng.normal([2.0, 0.0, 0.0], 0.2, size=(500, 3)),
        )
    )
    result = extract_particle_modes(
        particles,
        np.zeros(1000),
        voxel_size_m=0.5,
        min_core_cell_mass=0.003,
        assignment_radius_m=1.5,
    )
    selection = select_particle_mode(result, min_score_ratio=1.5)

    assert not selection.accepted
    assert selection.reason == "score_ratio"
    assert selection.position is None


def test_selector_can_require_multimodality_and_material_mean_shift():
    rng = np.random.default_rng(12)
    particles = rng.normal([0.0, 0.0, 0.0], 0.1, size=(1000, 3))
    result = extract_particle_modes(
        particles,
        np.zeros(1000),
        voxel_size_m=0.5,
        min_core_cell_mass=0.003,
    )
    selection = select_particle_mode(result, require_multiple_modes=True)
    assert not selection.accepted
    assert selection.reason == "single_mode"

    particles, log_weights = _cloud()
    result = extract_particle_modes(
        particles,
        log_weights,
        voxel_size_m=0.5,
        min_core_cell_mass=0.003,
        assignment_radius_m=1.5,
    )
    selection = select_particle_mode(result, min_weighted_mean_distance_m=3.0)
    assert not selection.accepted
    assert selection.reason == "weighted_mean_proximity"


def test_sparse_bridge_does_not_join_dense_modes():
    particles, log_weights = _cloud()
    bridge = np.column_stack((np.linspace(1.0, 7.0, 13), np.zeros((13, 2))))
    particles = np.vstack((particles, bridge))
    log_weights = np.concatenate((log_weights, np.full(len(bridge), -3.0)))
    result = extract_particle_modes(
        particles,
        log_weights,
        voxel_size_m=0.5,
        min_core_cell_mass=0.003,
        assignment_radius_m=1.0,
    )

    assert len(result.modes) == 2
    assert np.linalg.norm(result.modes[0].position - result.modes[1].position) > 7.0


def test_invalid_shapes_and_empty_finite_cloud():
    with pytest.raises(ValueError):
        extract_particle_modes(np.zeros((3, 2)), np.zeros(3))
    with pytest.raises(ValueError):
        extract_particle_modes(np.zeros((3, 3)), np.zeros(2))

    result = extract_particle_modes(np.full((4, 3), np.nan), np.zeros(4))
    assert result.modes == ()
    assert np.isnan(result.weighted_mean).all()


def test_deterministic_systematic_reduction_preserves_mode_mass():
    rng = np.random.default_rng(17)
    particles = np.vstack(
        (
            rng.normal([0.0, 0.0, 0.0], 0.3, size=(7000, 3)),
            rng.normal([10.0, 0.0, 0.0], 0.3, size=(3000, 3)),
        )
    )
    log_weights = np.zeros(10000)
    kwargs = dict(
        voxel_size_m=0.5,
        min_core_cell_mass=0.001,
        assignment_radius_m=1.5,
        max_particles=2000,
    )
    first = extract_particle_modes(particles, log_weights, **kwargs)
    second = extract_particle_modes(particles, log_weights, **kwargs)

    assert first.input_particle_count == 10000
    assert first.analyzed_particle_count == 2000
    assert [mode.mass for mode in first.modes] == pytest.approx([0.7, 0.3], abs=0.02)
    assert [mode.mass for mode in first.modes] == pytest.approx(
        [mode.mass for mode in second.modes]
    )
