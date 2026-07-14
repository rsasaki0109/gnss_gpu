import numpy as np

from gnss_gpu.switchable_factor import optimal_switch, reduce_switchable_factor


def test_switch_decreases_monotonically_with_residual():
    switch = optimal_switch(np.array([0.0, 1.0, 10.0]), prior_strength=2.0)
    assert switch[0] == 1.0
    assert np.all(np.diff(switch) < 0.0)


def test_reduced_cost_equals_explicit_optimized_switch_cost():
    r = np.array([-4.0, 0.5, 3.0])
    j = np.eye(3)
    strength = 2.5
    out = reduce_switchable_factor(r, j, strength)
    explicit = (out.switches * r) ** 2 + strength * (1.0 - out.switches) ** 2
    assert np.allclose(out.residual**2, explicit)


def test_reduced_switch_jacobian_matches_finite_difference():
    r = np.array([2.0])
    j = np.array([[3.0]])
    out = reduce_switchable_factor(r, j, prior_strength=1.5)
    eps = 1e-7
    moved = reduce_switchable_factor(r + j[:, 0] * eps, j, prior_strength=1.5)
    derivative = (moved.residual - out.residual) / eps
    assert np.allclose(derivative, out.jacobian[:, 0], rtol=1e-6, atol=1e-8)
