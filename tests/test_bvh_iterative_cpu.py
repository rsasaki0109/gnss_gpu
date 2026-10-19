"""Exact CPU node/permutation comparison on the CPU-only production candidate."""

import importlib.util
import os
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_bvh_multipath_compare import _load_extension, _random_mesh  # noqa: E402


@pytest.fixture(scope="module")
def modules():
    if path := os.environ.get("GNSS_BVH_CPU_EXTENSION"):
        new = _load_extension(Path(path), "iterative_cpu_candidate._bvh")
    else:
        new = pytest.importorskip("gnss_gpu._bvh")
    if not getattr(new, "_bvh_iterative_build", False):
        pytest.skip("Rebuild the iterative CPU BVH")
    old = None
    if path := os.environ.get("GNSS_BVH_BASELINE_EXTENSION"):
        old = _load_extension(Path(path), "iterative_cpu_original._bvh")
    return old, new


def sufficient_stack_original(native, mesh):
    outputs, errors = [], []

    def call():
        try:
            outputs.append(native.bvh_build(mesh))
        except BaseException as exc:
            errors.append(exc)

    previous = threading.stack_size(64 * 1024 * 1024)
    try:
        thread = threading.Thread(target=call)
        thread.start()
    finally:
        threading.stack_size(previous)
    thread.join()
    if errors:
        raise errors[0]
    assert len(outputs) == 1
    return outputs[0]


@pytest.mark.parametrize("count", [1, 64, 4096, 4097, 8192])
def test_normal_stack_cpu_build_has_exact_original_nodes_and_permutation(
    modules, count
):
    old, new = modules
    assert not hasattr(new.DeviceBVH, "set_tiled_reflections")
    for mesh in (_random_mesh(count), np.zeros((count, 3, 3))):
        before = mesh.tobytes()
        actual = new.bvh_build(mesh)
        assert mesh.tobytes() == before
        assert len(actual[1]) == count
        if old is not None:
            for a, b in zip(actual, sufficient_stack_original(old, mesh), strict=True):
                assert (
                    a.dtype == b.dtype
                    and a.shape == b.shape
                    and a.tobytes() == b.tobytes()
                )


def test_product_source_is_only_the_verified_cpu_projection():
    path = ROOT / "benchmarks/probes/bvh_iterative_build/generate.py"
    spec = importlib.util.spec_from_file_location(
        "iterative_cpu_product_projection", path
    )
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    expected, proof = generator.project()
    assert proof["all_GPU_source_unchanged"]
    assert (ROOT / "src/raytrace/bvh.cu").read_text(encoding="utf8") == expected
    original = (generator.BASE / "python/gnss_gpu/_bvh_bindings.cpp").read_text(
        encoding="utf8"
    )
    assert (ROOT / "python/gnss_gpu/_bvh_bindings.cpp").read_text(
        encoding="utf8"
    ) == generator.binding_marker(original)
