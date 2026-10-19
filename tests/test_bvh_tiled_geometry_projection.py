"""Portable checks of complete source restoration and untouched GPU math."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks/probes/bvh_tiled_geometry/generate.py"
SPEC = importlib.util.spec_from_file_location("tiled_geometry_projection", PATH)
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


def function(source, name):
    begin = source.index(name + "(")
    opening = source.index("{", begin)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


@pytest.mark.parametrize(
    "name",
    [
        "mt_intersect",
        "ray_aabb_intersect",
        "los_check_bvh_kernel",
        "los_check_bvh_batch_kernel",
        "los_frontier_kernel",
        "multipath_bvh_batch_kernel",
        "reflection_normals_kernel",
        "multipath_cached_serial_kernel",
        "reflection_mirrors_kernel",
        "better_multipath",
        "multipath_candidate",
        "multipath_candidates_kernel",
        "multipath_finalize_kernel",
    ],
)
def test_every_original_gpu_function_remains_literal(name):
    projected, proof = GEN.project()
    path = "src/raytrace/bvh.cu"
    original = (ROOT / path).read_text(encoding="utf8")
    assert function(original, name) == function(projected[path], name)
    assert proof[path]["full_source_undo_exact"]


@pytest.mark.parametrize("path", list(GEN.BASE_SHA256))
def test_all_three_sources_have_exact_reversible_projection(path):
    projected, proof = GEN.project()
    assert projected[path]
    assert proof[path]["original_sha256"] == GEN.BASE_SHA256[path]
    assert proof[path]["full_source_undo_exact"]
