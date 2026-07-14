import numpy as np

from gnss_gpu.particle_fixed_lag import (
    FixedLagGenealogySmoother,
    nearest_mode_mask,
)


def test_nearest_mode_mask_separates_terminal_modes():
    particles = np.array(
        [[0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0], [9.8, 0.0, 0.0, 0.0]]
    )
    mask = nearest_mode_mask(particles, np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), 1)
    np.testing.assert_array_equal(mask, [False, False, True])


def test_fixed_lag_terminal_mode_selects_matching_genealogy():
    smoother = FixedLagGenealogySmoother(
        lag_epochs=2, n_paths=32, seed=5, smoother_mode="genealogy"
    )
    n = 20
    ancestors = np.arange(n, dtype=np.int64)
    output = None
    for epoch in range(3):
        left = np.column_stack(
            (
                np.full(n // 2, float(epoch)),
                np.zeros((n // 2, 2)),
                np.zeros(n // 2),
            )
        )
        right = np.column_stack(
            (
                np.full(n // 2, 100.0 + epoch),
                np.zeros((n // 2, 2)),
                np.zeros(n // 2),
            )
        )
        particles = np.vstack((left, right))
        terminal_mask = np.arange(n) >= n // 2
        output = smoother.append(
            epoch,
            particles,
            np.zeros(n),
            ancestors,
            terminal_mask=terminal_mask,
        )

    assert output is not None
    assert output.epoch_index == 0
    assert output.terminal_conditioned
    assert output.terminal_particle_count == 10
    np.testing.assert_allclose(output.position, [100.0, 0.0, 0.0])


def test_marginal_fixed_lag_retains_multiple_oldest_paths():
    smoother = FixedLagGenealogySmoother(
        lag_epochs=2, n_paths=64, seed=15, smoother_mode="marginal", sigma_cb=10.0
    )
    n = 30
    output = None
    for epoch in range(3):
        particles = np.zeros((n, 4), dtype=np.float64)
        particles[:, 0] = np.linspace(-0.2, 0.2, n) + epoch
        output = smoother.append(
            epoch,
            particles,
            np.zeros(n),
            np.arange(n),
            velocities=np.ones((n, 3)) * np.array([1.0, 0.0, 0.0]),
            dt=1.0,
            sigma_pos=0.2,
        )
    assert output is not None
    assert output.smoother_mode == "marginal"
    assert output.unique_oldest_particles > 1
    assert output.covariance[0, 0] > 0.0


def test_fixed_lag_waits_then_flushes_every_epoch():
    smoother = FixedLagGenealogySmoother(lag_epochs=3, n_paths=4, seed=1)
    n = 5
    for epoch in range(2):
        particles = np.zeros((n, 4), dtype=np.float64)
        particles[:, 0] = epoch
        assert smoother.append(epoch, particles, np.zeros(n), np.arange(n)) is None
    outputs = smoother.flush()
    assert [output.epoch_index for output in outputs] == [0, 1]
    assert smoother.buffered_epochs == 0


def test_empty_terminal_mask_disables_rewrite():
    smoother = FixedLagGenealogySmoother(lag_epochs=1, n_paths=2, seed=1)
    particles = np.zeros((4, 4), dtype=np.float64)
    assert smoother.append(0, particles, np.zeros(4), np.arange(4)) is None
    output = smoother.append(
        1,
        particles,
        np.zeros(4),
        np.arange(4),
        terminal_mask=np.zeros(4, dtype=bool),
    )
    assert output is not None
    assert not output.rewrite_allowed
