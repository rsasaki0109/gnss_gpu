from __future__ import annotations

import numpy as np
import pytest

from gnss_gpu.realtime_batch import (
    BatchWorkspaceCapacity,
    CudaBatchWorkspace,
    cpu_affine_refit,
    cpu_arc_outlier_fraction,
    cpu_candidate_rms,
    cuda,
)


def test_candidate_batch_scores_all_candidates() -> None:
    residuals = np.array([[1.0, 1.0], [0.0, 2.0], [3.0, 4.0]])
    weights = np.ones_like(residuals)
    result = cpu_candidate_rms(residuals, weights)
    np.testing.assert_allclose(result, [1.0, np.sqrt(2.0), np.sqrt(12.5)])


def test_affine_batch_recovers_multiple_profiles() -> None:
    epochs = np.arange(6, dtype=np.float64)
    offsets = np.stack(
        [
            np.column_stack((epochs, 2 * epochs, -epochs)),
            np.column_stack((3 + 0.5 * epochs, epochs * 0, 2 - epochs)),
        ]
    )
    endpoints, rms = cpu_affine_refit(epochs, offsets, np.ones((2, 6)))
    np.testing.assert_allclose(endpoints[:, 0], offsets[:, 0], atol=1e-12)
    np.testing.assert_allclose(endpoints[:, 1], offsets[:, -1], atol=1e-12)
    np.testing.assert_allclose(rms, 0.0, atol=1e-12)


def test_arc_batch_detects_persistent_satellite_bias() -> None:
    residuals = np.array(
        [
            [0.0, 0.1, -0.1, 20.0],
            [0.1, 0.0, -0.2, 21.0],
            [0.0, 0.2, -0.1, 19.0],
        ]
    )
    result = cpu_arc_outlier_fraction(
        residuals,
        np.ones_like(residuals, dtype=bool),
        edge_m=5.0,
    )
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 1.0])


@pytest.mark.cuda
def test_persistent_cuda_workspace_matches_cpu_batches() -> None:
    if cuda is None or not cuda.is_available():
        pytest.skip("Numba CUDA unavailable")
    rng = np.random.default_rng(7)
    residuals = rng.normal(size=(8, 10))
    weights = rng.uniform(0.2, 1.0, size=(8, 10))
    epochs = np.arange(10, dtype=np.float64)
    offsets = rng.normal(size=(8, 10, 3))
    screen = rng.normal(size=(5, 6))
    screen[:, -1] += 20.0
    valid = np.ones_like(screen, dtype=bool)
    workspace = CudaBatchWorkspace(BatchWorkspaceCapacity(8, 10, 5, 6))

    np.testing.assert_allclose(
        workspace.candidate_rms(residuals, weights),
        cpu_candidate_rms(residuals, weights),
        rtol=1e-12,
    )
    gpu_endpoints, gpu_rms = workspace.affine_refit(epochs, offsets, weights)
    cpu_endpoints, cpu_rms = cpu_affine_refit(epochs, offsets, weights)
    np.testing.assert_allclose(gpu_endpoints, cpu_endpoints, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(gpu_rms, cpu_rms, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        workspace.arc_outlier_fraction(screen, valid, edge_m=5.0),
        cpu_arc_outlier_fraction(screen, valid, edge_m=5.0),
        rtol=1e-12,
    )
