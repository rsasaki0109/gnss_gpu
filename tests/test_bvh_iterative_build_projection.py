"""The stack-safe scheduler copies original split math and every GPU byte."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks/probes/bvh_iterative_build/generate.py"
SPEC = importlib.util.spec_from_file_location("iterative_bvh_projection", PATH)
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


def clean(text):
    return "\n".join(v.strip() for v in text.splitlines()).strip()


def test_original_sah_arithmetic_sort_and_ties_are_copied():
    old = (GEN.BASE / "src/raytrace/bvh.cu").read_text(encoding="utf8")
    new, proof = GEN.project()
    a, b = "  // SAH split:", "  // Allocate internal node"
    assert clean(old[old.index(a) : old.index(b)]) == clean(
        new[new.index(a) : new.index(b)]
    )
    assert (
        proof["full_source_undo_exact"]
        and proof["split_tie_sort_and_node_order_unchanged"]
    )


def test_all_gpu_functions_and_dispatch_remain_literal():
    old = (GEN.BASE / "src/raytrace/bvh.cu").read_text(encoding="utf8")
    new, proof = GEN.project()
    mark = "// GPU-side BVH traversal"
    assert old[old.index(mark) :] == new[new.index(mark) :]
    assert proof["all_GPU_source_unchanged"]


def test_cpu_projection_composes_without_editing_tiled_gpu_source():
    path = ROOT / "benchmarks/probes/bvh_tiled_geometry/generate.py"
    spec = importlib.util.spec_from_file_location("tiled_bvh_projection_composed", path)
    tiled = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tiled)
    tiled.ROOT = GEN.BASE
    files, _ = tiled.project()
    old = files["src/raytrace/bvh.cu"]
    new, proof = GEN.project(old)
    mark = "// GPU-side BVH traversal"
    assert old[old.index(mark) :] == new[new.index(mark) :]
    assert proof["full_source_undo_exact"]
