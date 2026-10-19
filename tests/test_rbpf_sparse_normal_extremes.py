"""Exercise the indexed sum itself on bounded binary64 edge inputs."""

import numpy as np
import pytest

from test_rbpf_sparse_normal import normal_probe, scorers  # noqa: F401


@pytest.mark.parametrize("dimension", [4, 7, 11])
@pytest.mark.parametrize("rows", [17, 128])
def test_bounded_signed_zero_subnormal_and_ordered_cancellation(scorers, dimension, rows):
    rng = np.random.default_rng(2026101700 + dimension + rows)
    pool = np.array([0., -0., np.nextafter(0., 1.), -np.nextafter(0., 1.),
        np.finfo(float).tiny, -np.finfo(float).tiny, 1., -1., 2.**500, -2.**500, 2.**-500, -2.**-500])
    data = np.zeros((128, 1657))
    groups = np.full((128, 128), -1, dtype=np.int32)
    for h in range(128):
        aw = data[h, :1408].reshape(128, 11)
        bw = data[h, 1408:1536]
        prior = data[h, 1536:].reshape(11, 11)
        groups[h, :rows] = rng.integers(-1, dimension - 3, rows)
        aw[:rows, :3] = rng.choice(pool, (rows, 3))
        bw[:rows] = rng.choice(pool, rows)
        prior[:] = rng.choice(pool, (11, 11))
        for r in range(rows):
            if groups[h, r] >= 0:
                aw[r, 3 + groups[h, r]] = rng.choice(pool)
        if h % 4 == 0:
            aw[:rows, 0] = np.resize([2.**500, -2.**500, np.nextafter(0., 1.), -np.nextafter(0., 1.)], rows)
            bw[:rows] = 1.
        elif h % 4 == 1:
            aw[:rows, :3] = np.resize([0., -0.], (rows, 3))
            bw[:rows] = -0.
        elif h % 4 == 2:
            prior[0, 0] = np.finfo(float).max
    # All operands satisfy the fast-path guard;the comparison must exercise it.
    assert np.all(normal_probe(scorers[-1], data, groups, rows, dimension) == 1)
