"""Regression for the real degenerate-chain overflow, with no GPU calls."""

import importlib.util
import os
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_bvh_multipath_compare import _load_extension  # noqa: E402


@pytest.fixture(scope="module")
def builders():
    target = os.environ.get("GNSS_BVH_CPU_EXTENSION") or os.environ.get(
        "GNSS_BVH_TILED_EXTENSION"
    )
    if target:
        candidate = _load_extension(Path(target), "duplicate_candidate._bvh")
    else:
        candidate = pytest.importorskip("gnss_gpu._bvh")
    if not getattr(candidate, "_bvh_iterative_build", False):
        pytest.skip(
            "Rebuild the stack-safe CPU BVH before running duplicate regression"
        )
    reference = None
    if original := os.environ.get("GNSS_BVH_BASELINE_EXTENSION"):
        reference = _load_extension(Path(original), "duplicate_reference._bvh")
    return candidate, reference


def reference_with_stack(module, mesh):
    result, error = [], []

    def call():
        try:
            result.append(module.bvh_build(mesh))
        except BaseException as exc:
            error.append(exc)

    previous = threading.stack_size(64 * 1024 * 1024)
    try:
        thread = threading.Thread(target=call)
        thread.start()
    finally:
        threading.stack_size(previous)
    thread.join()
    if error:
        raise error[0]
    assert len(result) == 1
    return result[0]


@pytest.mark.parametrize("count", [65, 4096, 8192])
@pytest.mark.parametrize("kind", ["zero", "plane", "line"])
def test_duplicate_chains_retain_every_triangle_split_and_node(builders, count, kind):
    candidate, reference = builders
    triangle = {
        "zero": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "plane": [[1.0, -20.0, -10.0], [1.0, 20.0, -10.0], [1.0, 0.0, 20.0]],
        "line": [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    }[kind]
    mesh = np.repeat(np.array([triangle]), count, axis=0)
    nodes, indices = candidate.bvh_build(mesh)
    assert nodes.shape == (2 * count - 7, 10)
    bound = np.concatenate((mesh[0].min(axis=0), mesh[0].max(axis=0)))
    np.testing.assert_array_equal(
        nodes[:, :6], np.broadcast_to(bound, nodes[:, :6].shape)
    )
    parents = np.arange(0, len(nodes) - 1, 2)
    np.testing.assert_array_equal(nodes[parents, 6], parents + 1)
    np.testing.assert_array_equal(nodes[parents, 7], parents + 2)
    np.testing.assert_array_equal(nodes[parents, 8:10], 0.0)
    leaves = nodes[1::2]
    np.testing.assert_array_equal(leaves[:, 6:8], -1.0)
    np.testing.assert_array_equal(leaves[:, 8], np.arange(count - 4))
    np.testing.assert_array_equal(leaves[:, 9], 1.0)
    np.testing.assert_array_equal(nodes[-1, 6:10], [-1.0, -1.0, count - 4, 4.0])
    np.testing.assert_array_equal(np.sort(indices), np.arange(count))
    if reference is not None:
        for value, old in zip(
            (nodes, indices), reference_with_stack(reference, mesh), strict=True
        ):
            assert value.dtype == old.dtype and value.shape == old.shape
            np.testing.assert_array_equal(value.view(np.uint8), old.view(np.uint8))
