import numpy as np
import pytest

from gnss_gpu.wcp_factor import left_nullspace_project, single_arc_wcp


def test_wcp_eliminates_shared_ambiguity_exactly():
    residual = np.array([12.0, 13.0, 15.0])
    jacobian = np.eye(3)
    out = single_arc_wcp(residual, jacobian, 0.2)
    assert out.residual.shape == (2,)
    assert out.jacobian.shape == (2, 3)
    assert np.allclose(out.left_nullspace.T @ np.ones(3), 0.0, atol=1e-12)
    shifted = single_arc_wcp(residual + 100.0, jacobian, 0.2)
    assert np.allclose(out.residual, shifted.residual, atol=1e-12)


def test_wcp_whitening_and_jacobian_match_finite_difference():
    residual = np.array([2.0, -1.0, 3.0])
    jacobian = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
    sigma = np.array([0.1, 0.2, 0.4])
    out = single_arc_wcp(residual, jacobian, sigma)
    step = np.array([1e-6, -2e-6])
    moved = single_arc_wcp(residual + jacobian @ step, jacobian, sigma)
    assert np.allclose(moved.residual - out.residual, out.jacobian @ step, atol=1e-12)


def test_wcp_supports_multiple_ambiguity_columns():
    a = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    out = left_nullspace_project(np.arange(4.0), np.eye(4), a, np.eye(4))
    assert out.rank_ambiguity == 2
    assert out.residual.shape == (2,)
    assert np.allclose(out.left_nullspace.T @ a, 0.0, atol=1e-12)


def test_wcp_rejects_single_epoch_arc():
    with pytest.raises(ValueError):
        single_arc_wcp(np.array([1.0]), np.ones((1, 2)), 0.1)
