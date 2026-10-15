"""Shared exact-copy and prediction cases for covariance broadcast validation."""

import numpy as np


COUNTS = (1, 31, 32, 33, 255, 256, 257, 8193)
PREDICTIONS = ((.3, .17, 1.3, 2.7, .13, 0.),
               (2., 0., 1e-300, 2.7, 0., .2),
               (-1., 1e-300, 1e-300, 1e-300, -1., .2),
               (.3, 1e154, 1.3, 2.7, .13, 0.))


def covariance(case):
    matrices = (
        [[2., .3, 0.], [.3, 1.5, .1], [0., .1, .8]],
        [[4., 1., .5], [1., 3., .25], [.5, .25, 2.]],
        [[4., 1.3, -.5], [-7., 3., .2], [9., .8, 2.]],
        [[-2., 1., -.5], [3., -4., .25], [.8, -2., -1.]],
        [[-0., 0., -0.], [0., -0., 0.], [-0., 0., -0.]],
        [[1e-300, 5e-324, -5e-324], [5e-324, 1e-300, 0.], [-5e-324, 0., 1e-300]],
        [[1e150, -1e100, 1e100], [-1e100, 1e150, .25], [1e100, .25, 1e150]],
        [[1e308, -1e308, 1e308], [-1e308, 1e308, -1e308], [1e308, -1e308, 1e308]],
    )
    return np.array(matrices[case], dtype=np.float64)


def representation(matrix, form):
    if form == 0:
        return matrix.copy(order='C')
    if form == 1:
        return matrix.copy(order='F')
    storage = np.zeros((6, 6), dtype=np.float64)
    storage[::2, ::2] = matrix
    return storage[::2, ::2]


def initial_arrays(states):
    states = states.copy()
    count = len(states)
    rng = np.random.default_rng(1493+count)
    states[:, 4:7] = rng.normal(size=(count, 3))
    states[:, 7:] = rng.normal(size=(count, 9))
    states[::17, 4] = -0.
    # These fields are copied without arithmetic, including NaN payload bits.
    states.view(np.uint64)[::29, 3] = 0x7ff8000000000042
    weights = rng.normal(size=count)
    weights.view(np.uint64)[::31] = 0x7ff8000000000093
    weights[::37] = -np.inf
    weights[::41] = -0.
    return states, weights


def invalid_covariances():
    return (np.zeros(8), np.zeros(10), np.full((3, 3), np.nan),
            np.full((3, 3), np.inf), np.full((3, 3), -np.inf), [['bad']*3]*3)
