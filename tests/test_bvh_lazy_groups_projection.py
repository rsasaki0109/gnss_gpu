"""Literal original GPU functions and exact grouping/normalization inputs."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks/probes/bvh_lazy_groups/generate.py"
SPEC = importlib.util.spec_from_file_location("grouped_direction_projection", PATH)
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


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
def test_all_original_gpu_functions_are_literal(name):
    projected, _ = GEN.project()
    original = (ROOT / "src/raytrace/bvh.cu").read_text(encoding="utf8")
    assert GEN.function(projected["src/raytrace/bvh.cu"], name) == GEN.function(
        original, name
    )


@pytest.mark.parametrize("name", list(GEN.BASE_SHA256))
def test_whole_source_undo_exact(name):
    _, proof = GEN.project()
    assert proof[name]["full_composed_source_undo_exact"]
    assert proof[name]["original_D295_sha256"] == GEN.BASE_SHA256[name]


def test_original_cpu_builder_and_normalization_tail_are_literal():
    projected, _ = GEN.project()
    source = (ROOT / "src/raytrace/bvh.cu").read_text(encoding="utf8")
    candidate = GEN.function(source, "multipath_candidate")
    tail = candidate[candidate.index("  double t;") : candidate.rfind("}")]
    assert tail in projected["src/raytrace/bvh.cu"]
    mark = "// GPU-side BVH traversal"
    assert (
        source[source.index("namespace gnss_gpu") : source.index(mark)]
        in projected["src/raytrace/bvh.cu"]
    )
    for statement in (
        "double dir_len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);",
        "dir[0] /= dir_len; dir[1] /= dir_len; dir[2] /= dir_len;",
    ):
        assert statement in GEN.function(
            projected["src/raytrace/bvh.cu"], "multipath_register_groups_kernel"
        )


def test_grouping_uses_original_normal_bits_and_full_collision_comparison():
    projected, _ = GEN.project()
    source = projected["src/raytrace/bvh.cu"]
    assert "std::unordered_map<Key, int, Hash> groups;" in source
    assert (
        "std::memcpy(key.data(), direction_v0[ordinal].data(), 3 * sizeof(double));"
        in source
    )
    assert (
        "std::memcpy(key.data() + 3, saved_normals[ordinal].value, 3 * sizeof(double));"
        in source
    )
    assert "representatives.push_back(ordinal)" in source
    assert "cudaMemcpyDeviceToHost" in GEN.function(source, "prepare_direction_groups")
    assert "group_sizes[found->second] == 8" in source
    assert "std::vector<int> offsets(representatives.size() + 1, 0);" in source
    assert "DeviceBuffer<GroupedDirection>" not in source


def test_constructor_does_no_group_vertex_work():
    projected, _ = GEN.project()
    source = projected["src/raytrace/bvh.cu"]
    ctor = source[source.index("DeviceBVH::DeviceBVH(") : source.index("DeviceBVH::~DeviceBVH()")]
    assert "direction_v0" not in ctor and "cudaMemcpyDeviceToHost" not in ctor
    assert "state->direction_requested = grouped_candidates && state->direction_capable;" in ctor
    assert "direction_preparation_scratch = (size_t)n_order * 3 * sizeof(double);" in source
    prepare = GEN.function(source, "prepare_direction_groups")
    assert prepare.index("packed_vertices.reset();") < prepare.index("using Key =")


def test_parent_group_kernel_and_host_grouping_tail_are_literal():
    projected, _ = GEN.project()
    parent, _ = GEN.OLD.project()
    before, after = parent["src/raytrace/bvh.cu"], projected["src/raytrace/bvh.cu"]
    for name in ("multipath_register_groups_kernel", "multipath_register_candidate"):
        assert GEN.function(before, name) == GEN.function(after, name)
    tail = GEN.function(before, "prepare_direction_groups")
    assert tail[tail.index("    using Key ="):] in GEN.function(after, "prepare_direction_groups")
    kernel = GEN.function(after, "reflection_group_vertices_kernel")
    assert "sqrt" not in kernel and "__double2" not in kernel
