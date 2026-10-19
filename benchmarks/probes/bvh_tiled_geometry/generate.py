"""Project the frozen BVH into an explicit, default-off memory-layout probe."""

import argparse
from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE_SHA256 = {
    "src/raytrace/bvh.cu": "7d9d06f128710f12eea70b09b8f0d475d16223957ddfdf41a7cf5ec77ac09d98",
    "include/gnss_gpu/bvh.h": "50b44813b584f5b6778fb983e07b587b6dc8ee268326672de9efbbff7f496e35",
    "python/gnss_gpu/_bvh_bindings.cpp": "a043d669a98b24772c9ce4a3639d21bb869d65a35c30843979190025e4b44845",
}


def project():
    originals, changes, projected = {}, {}, {}
    for name, expected in BASE_SHA256.items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Frozen BVH source changed: " + name)
        originals[name] = data.decode().replace("\r\n", "\n")
        projected[name] = originals[name]
        changes[name] = []

    def replace(name, old, new):
        assert projected[name].count(old) == 1, (name, old[:120])
        projected[name] = projected[name].replace(old, new)
        changes[name].append((old, new))

    cu = "src/raytrace/bvh.cu"
    source = originals[cu]
    # Existing numerical functions and kernels are not edited. The new kernel
    # calls the same multipath_candidate specialization with copied inputs.
    start = source.index("__global__ void reflection_normals_kernel(")
    end = source.index("// Large ordered batches", start)
    prepare = source[start:end].rstrip()
    prepare = prepare.replace("reflection_normals_kernel", "reflection_tiles_kernel")
    prepare = prepare.replace("ReflectionNormal* normals", "ReflectionTile* geometry")
    prepare = prepare.replace(
        "  normals[ordinal] = result;",
        """  ReflectionTile& tile = geometry[ordinal / 32];
  int lane = ordinal % 32;
  #pragma unroll
  for (int d = 0; d < 3; ++d) {
    tile.vertex[d][lane] = tri.v0[d];
    tile.vertex[d + 3][lane] = tri.v1[d];
    tile.vertex[d + 6][lane] = tri.v2[d];
    tile.normal[d][lane] = result.value[d];
  }
  tile.valid[lane] = result.valid;""",
    )
    start = source.index("__global__ void reflection_mirrors_kernel(")
    end = source.index("struct MultipathCandidate", start)
    mirrors = source[start:end].rstrip()
    mirrors = mirrors.replace(
        "reflection_mirrors_kernel", "reflection_tiled_mirrors_kernel"
    )
    mirrors = mirrors.replace(
        "const Triangle* tris, const int* order,\n                                           const ReflectionNormal* normals,",
        "const ReflectionTile* geometry,",
    )
    mirrors = mirrors.replace(
        "  const ReflectionNormal& normal = normals[ordinal];",
        "  Triangle tri;\n  ReflectionNormal normal;\n  load_reflection_tile(geometry, ordinal, tri, normal);",
    )
    mirrors = mirrors.replace("    const Triangle& tri = tris[order[ordinal]];\n", "")
    start = source.index(
        "template <bool CACHED_NORMALS, bool SHARED_MIRRORS>\n__global__ void multipath_candidates_kernel"
    )
    end = source.index("__global__ void multipath_finalize_kernel", start)
    candidate = source[start:end].rstrip()
    candidate = candidate.replace(
        "template <bool CACHED_NORMALS, bool SHARED_MIRRORS>",
        "template <bool SHARED_MIRRORS>",
    )
    candidate = candidate.replace(
        "multipath_candidates_kernel", "multipath_tiled_candidates_kernel"
    )
    candidate = candidate.replace(
        "                                            const Triangle* tris,\n                                            const int* triangle_order,\n",
        "",
    )
    candidate = candidate.replace(
        "const ReflectionNormal* normals,", "const ReflectionTile* geometry,"
    )
    candidate = candidate.replace(
        "    const ReflectionNormal* normal = nullptr;\n    if constexpr (CACHED_NORMALS) normal = normals + ordinal;",
        "    Triangle tri;\n    ReflectionNormal normal;\n    load_reflection_tile(geometry, ordinal, tri, normal);",
    )
    candidate = candidate.replace(
        "multipath_candidate<CACHED_NORMALS, SHARED_MIRRORS>",
        "multipath_candidate<true, SHARED_MIRRORS>",
    )
    candidate = candidate.replace(
        "tris[triangle_order[ordinal]], candidate, normal, mirror",
        "tri, candidate, &normal, mirror",
    )
    assert "CACHED_NORMALS" not in candidate and "tris[" not in candidate
    assert "normals[" not in mirrors and "tris[" not in mirrors
    additions = (
        """
// BEGIN experimental tiled reflection inputs. Raw vertex bits are copied;
// edge, intersection, distance and winner arithmetic remain in the original.
struct ReflectionTile {
  double vertex[9][32];
  double normal[3][32];
  int valid[32];
};
static_assert(sizeof(ReflectionTile) == 3200, "Unexpected tile layout");

__device__ static void load_reflection_tile(const ReflectionTile* geometry,
    int ordinal, Triangle& tri, ReflectionNormal& normal) {
  const ReflectionTile& tile = geometry[ordinal / 32];
  int lane = ordinal % 32;
  #pragma unroll
  for (int d = 0; d < 3; ++d) {
    tri.v0[d] = tile.vertex[d][lane];
    tri.v1[d] = tile.vertex[d + 3][lane];
    tri.v2[d] = tile.vertex[d + 6][lane];
    normal.value[d] = tile.normal[d][lane];
  }
  normal.valid = tile.valid[lane];
}
"""
        + "\n\n".join((prepare, mirrors, candidate))
        + "\n// END experimental tiled reflection inputs.\n\n"
    )
    mark = "struct DeviceBVH::Impl {"
    replace(cu, mark, additions + mark)
    replace(
        cu,
        "  bool normals_ready = false;",
        """  bool normals_ready = false;
  DeviceBuffer<ReflectionTile> tiled_geometry;
  bool tiled_requested = false;
  bool tiled_ready = false;
  bool tiled_active = false;""",
    )
    replace(cu, "    mirrors.reset();", "    mirrors.reset(); tiled_geometry.reset();")
    replace(
        cu,
        "    mirrors_active = false;\n    mirror_epochs = 0;",
        "    tiled_active = false;\n    mirrors_active = false;\n    mirror_epochs = 0;",
    )
    old = """    // Prepare only for the parallel reflection path. LOS-only, small meshes and
    // serial fallback batches do not allocate or launch normal preparation.
    if (!normals_ready) {
      normals.reserve((size_t)n_order);
      reflection_normals_kernel<<<(n_order + 255) / 256, 256>>>(
          triangles.data, order.data, normals.data, n_order);
      CUDA_CHECK_LAST();
      normals_ready = true;
    }"""
    new = (
        """    // Explicit bulk-only probe; at most128 MiB, no allocation for controls.
    size_t tile_count = ((size_t)n_order + 31) / 32;
    tiled_active = tiled_requested && n_order >= 4096 && total >= 512 &&
        tile_count <= (128u * 1024u * 1024u) / sizeof(ReflectionTile);
    if (tiled_active) {
      if (!tiled_ready) {
        tiled_geometry.reserve(tile_count);
        reflection_tiles_kernel<<<(n_order - 1) / 256 + 1, 256>>>(
            triangles.data, order.data, tiled_geometry.data, n_order);
        CUDA_CHECK_LAST();
        tiled_ready = true;
      }
    } else {
"""
        + old
        + "\n    }"
    )
    replace(cu, old, new)
    # Add a separate bulk launch path. Original launch code remains literal.
    start = source.index("  void launch_multipath(")
    end = source.index("  void copy_multipath(", start)
    launch = source[start:end].rstrip()
    launch = launch.replace("void launch_multipath(", "void launch_tiled_multipath(")
    launch = (
        launch[: launch.index("    } else if (cached_serial)")]
        + "    }\n    CUDA_CHECK_LAST();\n  }"
    )
    launch = launch.replace(
        "reflection_mirrors_kernel", "reflection_tiled_mirrors_kernel"
    )
    launch = launch.replace(
        "triangles.data, order.data,\n              normals.data, mirrors.data, n_order",
        "tiled_geometry.data, mirrors.data, n_order",
    )
    launch = launch.replace(
        "multipath_candidates_kernel<true, true>",
        "multipath_tiled_candidates_kernel<true>",
    )
    launch = launch.replace(
        "multipath_candidates_kernel<true, false>",
        "multipath_tiled_candidates_kernel<false>",
    )
    launch = launch.replace("triangles.data, order.data, partial.data", "partial.data")
    launch = launch.replace(
        "normals.data, mirrors.data", "tiled_geometry.data, mirrors.data"
    )
    launch = launch.replace("normals.data, nullptr", "tiled_geometry.data, nullptr")
    assert (
        "normals.data" not in launch
        and "triangles.data" not in launch
        and "order.data" not in launch
    )
    mark = "  void launch_multipath(int n_epoch, int n_sat, int tiles) {\n"
    replace(
        cu,
        mark,
        launch
        + "\n\n"
        + mark
        + "    if (tiled_active) { launch_tiled_multipath(n_epoch, n_sat, tiles); return; }\n",
    )
    replace(
        cu,
        "      s.normals.capacity * sizeof(ReflectionNormal) +",
        "      s.normals.capacity * sizeof(ReflectionNormal) +\n      s.tiled_geometry.capacity * sizeof(ReflectionTile) +",
    )
    mark = "DeviceBVH::~DeviceBVH() = default;"
    api = """void DeviceBVH::set_tiled_reflections(bool enabled) {
  if (!impl_) throw std::runtime_error("DeviceBVH is closed");
  impl_->tiled_requested = enabled;
  if (!enabled) impl_->tiled_active = false;
}
bool DeviceBVH::tiled_reflections_active() const noexcept {
  return impl_ && impl_->tiled_active;
}
std::size_t DeviceBVH::tiled_geometry_bytes() const noexcept {
  return impl_ ? impl_->tiled_geometry.capacity * sizeof(ReflectionTile) : 0;
}

"""
    replace(cu, mark, api + mark)
    header = "include/gnss_gpu/bvh.h"
    mark = "  void close() noexcept;"
    replace(
        header,
        mark,
        "  void set_tiled_reflections(bool enabled);\n  bool tiled_reflections_active() const noexcept;\n  std::size_t tiled_geometry_bytes() const noexcept;\n"
        + mark,
    )
    binding = "python/gnss_gpu/_bvh_bindings.cpp"
    mark = '      .def("close", &gnss_gpu::DeviceBVH::close)'
    replace(
        binding,
        mark,
        """      .def("set_tiled_reflections", &gnss_gpu::DeviceBVH::set_tiled_reflections)
      .def_property_readonly("tiled_reflections_active", &gnss_gpu::DeviceBVH::tiled_reflections_active)
      .def_property_readonly("tiled_geometry_bytes", &gnss_gpu::DeviceBVH::tiled_geometry_bytes)
"""
        + mark,
    )
    proof = {}
    for name, text in projected.items():
        restored = text
        for old, new in reversed(changes[name]):
            assert restored.count(new) == 1
            restored = restored.replace(new, old)
        assert restored == originals[name], name
        proof[name] = dict(
            original_sha256=BASE_SHA256[name],
            full_source_undo_exact=True,
            projected_sha256=hashlib.sha256(
                text.replace("\n", "\r\n").encode()
            ).hexdigest(),
        )
    return projected, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    files, proof = project()
    args.output.mkdir(parents=True)
    for name, text in files.items():
        p = args.output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.replace("\n", "\r\n").encode())
    for name in ("include/gnss_gpu/raytrace.h", "include/gnss_gpu/cuda_check.h"):
        p = args.output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
        proof[name] = dict(
            copied_exact=True,
            original_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        )
    (args.output / "projection.json").write_text(
        json.dumps(proof, indent=2) + "\n", encoding="utf8"
    )
    print(
        json.dumps(dict(generated=str(args.output), full_source_undo_exact=True)),
        flush=True,
    )


if __name__ == "__main__":
    main()
