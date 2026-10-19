"""Exact raw-key grouping of original GPU reflection normalization inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE_SHA256 = {
    "src/raytrace/bvh.cu": "ed868c83a59b81fd8fb34a1d1c3a8ca92a075a15012060d8054c366196b5ffd9",
    "include/gnss_gpu/bvh.h": "50b44813b584f5b6778fb983e07b587b6dc8ee268326672de9efbbff7f496e35",
    "python/gnss_gpu/_bvh_bindings.cpp": "73a6de08c35b18d8629cf4d727d54643ef5b042216e814197c9a95da35402c0a",
}


def function(source, name):
    begin = source.index(name + "(")
    opening = source.index("{", begin)
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def project():
    original, result, edits = {}, {}, {}
    for name, expected in BASE_SHA256.items():
        raw = (ROOT / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, name
        original[name] = result[name] = raw.decode().replace("\r\n", "\n")
        edits[name] = []

    def replace(name, old, new):
        assert result[name].count(old) == 1, (name, old[:100])
        result[name] = result[name].replace(old, new)
        edits[name].append((old, new))

    cu = "src/raytrace/bvh.cu"
    source = original[cu]
    # Copy the exact mirror expression and exact normalization expression.
    mirror = function(source, "reflection_mirrors_kernel")
    mirror_math = mirror[
        mirror.index("    double dv[3]") : mirror.index("\n  }\n  mirrors[")
    ]
    mirror_math = mirror_math.replace("result.point", "result.mirror")
    candidate = function(source, "multipath_candidate")
    direction = candidate[
        candidate.index("  double dir[3]") : candidate.index("\n  double t;")
    ]
    direction = direction.replace(
        "return false;", "{ prepared[index] = result; return; }"
    )
    tail = candidate[candidate.index("  double t;") : candidate.rfind("}")]
    helpers = """
// One entry is valid only for this call's current receiver and satellite.
struct GroupedDirection {
  double mirror[3];
  double direction[3];
  int valid;
};
static_assert(sizeof(GroupedDirection) == 56);

__global__ void grouped_directions_kernel(const double* rx_ecef,
                                         const double* sat_ecef,
                                         const Triangle* tris, const int* order,
                                         const ReflectionNormal* normals,
                                         const int* representatives,
                                         GroupedDirection* prepared,
                                         int n_groups, int n_sat, int n_rays) {
  size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= (size_t)n_rays * n_groups) return;
  int ray = (int)(index / n_groups);
  int group = (int)(index % n_groups);
  int epoch = ray / n_sat;
  int ordinal = representatives[group];
  const ReflectionNormal& normal = normals[ordinal];
  const Triangle& tri = tris[order[ordinal]];
  GroupedDirection result = {};
  if (normal.valid) {
    double rx[3] = {rx_ecef[epoch * 3], rx_ecef[epoch * 3 + 1], rx_ecef[epoch * 3 + 2]};
    double n[3] = {normal.value[0], normal.value[1], normal.value[2]};
MIRROR_MATH
    double mirror[3] = {result.mirror[0], result.mirror[1], result.mirror[2]};
    double sat[3] = {sat_ecef[ray * 3], sat_ecef[ray * 3 + 1], sat_ecef[ray * 3 + 2]};
DIRECTION_MATH
    result.direction[0] = dir[0]; result.direction[1] = dir[1]; result.direction[2] = dir[2];
    result.valid = 1;
  }
  prepared[index] = result;
}

__device__ static bool multipath_grouped_candidate(const double rx[3], const double sat[3],
                                                   double d_rx_sat, const Triangle& tri,
                                                   MultipathCandidate& result,
                                                   const GroupedDirection& prepared) {
  if (!prepared.valid) return false;
  double mirror[3] = {prepared.mirror[0], prepared.mirror[1], prepared.mirror[2]};
  double dir[3] = {prepared.direction[0], prepared.direction[1], prepared.direction[2]};
CANDIDATE_TAIL
}

"""
    helpers = (
        helpers.replace("MIRROR_MATH", mirror_math)
        .replace("DIRECTION_MATH", direction)
        .replace("CANDIDATE_TAIL", tail)
    )
    start = source.index(
        "template <bool CACHED_NORMALS, bool SHARED_MIRRORS>\n__global__ void multipath_candidates_kernel("
    )
    end = source.index("\n__global__ void multipath_finalize_kernel", start)
    kernel = source[start:end]
    kernel = kernel.replace(
        "template <bool CACHED_NORMALS, bool SHARED_MIRRORS>\n", ""
    ).replace("multipath_candidates_kernel", "multipath_grouped_candidates_kernel")
    kernel = kernel.replace(
        "const ReflectionNormal* normals,\n                                            const ReflectionMirror* mirrors)",
        "const int* groups, const GroupedDirection* prepared, int n_groups)",
    )
    a = kernel.index("    const ReflectionNormal* normal = nullptr;")
    b = kernel.index(" &&\n        better_multipath", a)
    kernel = (
        kernel[:a]
        + """    int group = groups[ordinal];
    if (group < 0) continue;
    const GroupedDirection& cached = prepared[(size_t)ray * n_groups + group];
    if (multipath_grouped_candidate(
            rx, sat, d_rx_sat, tris[triangle_order[ordinal]], candidate, cached)"""
        + kernel[b:]
    )
    replace(
        cu,
        "\n__global__ void multipath_finalize_kernel",
        helpers + kernel + "\n__global__ void multipath_finalize_kernel",
    )
    replace(
        cu,
        "#include <vector>",
        "#include <vector>\n#include <array>\n#include <cstdint>\n#include <cstring>\n#include <unordered_map>",
    )
    replace(
        cu,
        "  bool normals_ready = false;",
        """  bool normals_ready = false;
  bool direction_capable = false, direction_requested = false;
  bool direction_attempted = false, direction_active = false;
  int direction_groups = 0, direction_epochs = 0;
  std::vector<std::array<double, 3>> direction_v0;
  DeviceBuffer<int> direction_map, direction_representatives;
  DeviceBuffer<GroupedDirection> directions;""",
    )
    replace(
        cu,
        "    mirrors.reset();",
        "    mirrors.reset();\n    direction_map.reset(); direction_representatives.reset(); directions.reset();",
    )
    method = """
  void prepare_direction_groups() {
    direction_attempted = true;
    std::vector<ReflectionNormal> saved_normals(n_order);
    CUDA_CHECK(cudaMemcpy(saved_normals.data(), normals.data,
        (size_t)n_order * sizeof(ReflectionNormal), cudaMemcpyDeviceToHost));
    using Key = std::array<std::uint64_t, 6>;
    struct Hash {
      size_t operator()(const Key& key) const noexcept {
        size_t value = 1469598103934665603ull;
        for (auto word : key) { value ^= (size_t)word; value *= 1099511628211ull; }
        return value;
      }
    };
    std::unordered_map<Key, int, Hash> groups;
    groups.reserve((size_t)n_order);
    std::vector<int> mapping(n_order, -1), representatives;
    representatives.reserve((size_t)n_order);
    for (int ordinal = 0; ordinal < n_order; ++ordinal) {
      if (!saved_normals[ordinal].valid) continue;
      Key key;
      // Compare full original bits, including signed zero. Hash collisions are
      // resolved by all six uint64 components; no geometric tolerance is used.
      std::memcpy(key.data(), direction_v0[ordinal].data(), 3 * sizeof(double));
      std::memcpy(key.data() + 3, saved_normals[ordinal].value, 3 * sizeof(double));
      auto found = groups.find(key);
      if (found == groups.end()) {
        int group = (int)representatives.size();
        groups.emplace(key, group);
        representatives.push_back(ordinal);
        mapping[ordinal] = group;
      } else mapping[ordinal] = found->second;
    }
    std::vector<std::array<double, 3>>().swap(direction_v0);
    if (representatives.empty() || representatives.size() * 10 > (size_t)n_order * 7) return;
    direction_map.upload(mapping.data(), mapping.size());
    direction_representatives.upload(representatives.data(), representatives.size());
    direction_groups = (int)representatives.size();
  }

  bool prepare_grouped_directions(int total, int n_epoch, int n_sat) {
    if (!direction_requested || !direction_capable || total < 512 || n_sat < 8) return false;
    if (!direction_attempted) prepare_direction_groups();
    if (direction_groups <= 0) return false;
    constexpr size_t budget = 128u * 1024u * 1024u;
    size_t max_epochs = budget / sizeof(GroupedDirection) / direction_groups / n_sat;
    if (max_epochs == 0) return false;
    direction_epochs = (int)std::min((size_t)n_epoch, max_epochs);
    if (parallel_epochs) direction_epochs = std::min(direction_epochs, parallel_epochs);
    directions.reserve((size_t)direction_epochs * n_sat * direction_groups);
    direction_active = true;
    return true;
  }

"""
    replace(
        cu,
        "  int prepare_multipath(int total, int n_epoch, int n_sat) {",
        method
        + "  int prepare_multipath(int total, int n_epoch, int n_sat) {\n    direction_active = false;\n    direction_epochs = 0;",
    )
    replace(
        cu,
        "    // Share receiver-dependent work only when a substantial mesh",
        "    if (prepare_grouped_directions(total, n_epoch, n_sat)) return tiles;\n    // Share receiver-dependent work only when a substantial mesh",
    )
    launch = """    if (direction_active) {
      for (int first_epoch = 0; first_epoch < n_epoch; first_epoch += direction_epochs) {
        int epochs = std::min(direction_epochs, n_epoch - first_epoch);
        int rays = epochs * n_sat;
        size_t ray_offset = (size_t)first_epoch * n_sat;
        size_t count = (size_t)rays * direction_groups;
        grouped_directions_kernel<<<(count + 255) / 256, 256>>>(
            rx.data + (size_t)first_epoch * 3, sat.data + ray_offset * 3,
            triangles.data, order.data, normals.data, direction_representatives.data,
            directions.data, direction_groups, n_sat, rays);
        CUDA_CHECK_LAST();
        multipath_grouped_candidates_kernel<<<rays * tiles, MULTIPATH_BLOCK_SIZE>>>(
            rx.data + (size_t)first_epoch * 3, sat.data + ray_offset * 3,
            triangles.data, order.data, partial.data, n_sat, n_order, tiles,
            direction_map.data, directions.data, direction_groups);
        CUDA_CHECK_LAST();
        multipath_finalize_kernel<<<(rays + MULTIPATH_BLOCK_SIZE - 1) / MULTIPATH_BLOCK_SIZE,
                                    MULTIPATH_BLOCK_SIZE>>>(
            partial.data, points.data + ray_offset * 3, delays.data + ray_offset, rays, tiles);
        CUDA_CHECK_LAST();
      }
      return;
    }
"""
    replace(
        cu,
        "    if (tiles && parallel_epochs) {",
        launch + "    if (tiles && parallel_epochs) {",
    )
    old_ctor = "DeviceBVH::DeviceBVH(const BVHNode* h_nodes, int n_nodes, const Triangle* h_tri, int n_tri)"
    replace(cu, old_ctor, old_ctor[:-1] + ", bool grouped_directions)")
    snapshot = """  if (grouped_directions && state->ordered && state->n_order >= 4096 && state->n_order <= 262144) {
    state->direction_capable = state->direction_requested = true;
    state->direction_v0.resize((size_t)state->n_order);
    for (int ordinal = 0; ordinal < state->n_order; ++ordinal) {
      const Triangle& triangle = h_tri[state->host_order[ordinal]];
      std::memcpy(state->direction_v0[ordinal].data(), triangle.v0, 3 * sizeof(double));
    }
  }
"""
    replace(cu, "  impl_ = std::move(state);", snapshot + "  impl_ = std::move(state);")
    replace(
        cu,
        "      s.mirrors.capacity * sizeof(ReflectionMirror);",
        "      s.mirrors.capacity * sizeof(ReflectionMirror) +\n      (s.direction_map.capacity + s.direction_representatives.capacity) * sizeof(int) +\n      s.directions.capacity * sizeof(GroupedDirection);",
    )
    api = """
void DeviceBVH::set_direction_cache_enabled(bool enabled) {
  if (!impl_) throw std::runtime_error("DeviceBVH is closed");
  // Small/unsupported meshes remain a defined fallback even when requested.
  impl_->direction_requested = enabled && impl_->direction_capable;
}
int DeviceBVH::direction_group_count() const noexcept { return impl_ ? impl_->direction_groups : 0; }
bool DeviceBVH::direction_cache_active() const noexcept { return impl_ && impl_->direction_active; }
std::size_t DeviceBVH::direction_cache_bytes() const noexcept {
  return impl_ ? impl_->directions.capacity * sizeof(GroupedDirection) : 0;
}

"""
    replace(
        cu,
        "void DeviceBVH::check_los(const double* h_rx",
        api + "void DeviceBVH::check_los(const double* h_rx",
    )
    header = "include/gnss_gpu/bvh.h"
    replace(
        header,
        "DeviceBVH(const BVHNode* nodes, int n_nodes, const Triangle* triangles, int n_tri);",
        "DeviceBVH(const BVHNode* nodes, int n_nodes, const Triangle* triangles, int n_tri, bool grouped_directions = false);",
    )
    replace(
        header,
        "  std::size_t allocated_bytes() const noexcept;",
        "  std::size_t allocated_bytes() const noexcept;\n  void set_direction_cache_enabled(bool enabled);\n  int direction_group_count() const noexcept;\n  bool direction_cache_active() const noexcept;\n  std::size_t direction_cache_bytes() const noexcept;",
    )
    binding = "python/gnss_gpu/_bvh_bindings.cpp"
    replace(
        binding,
        "py::init([](DoubleArray nodes_flat, DoubleArray sorted_tris)",
        "py::init([](DoubleArray nodes_flat, DoubleArray sorted_tris, bool grouped_directions)",
    )
    replace(
        binding,
        "nodes.data(), n_nodes, reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr), n_tri);",
        "nodes.data(), n_nodes, reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr), n_tri, grouped_directions);",
    )
    replace(
        binding,
        '}), py::arg("nodes_flat"), py::arg("sorted_tris"))',
        '}), py::arg("nodes_flat"), py::arg("sorted_tris"), py::arg("grouped_directions") = false)',
    )
    replace(
        binding,
        '      .def("close", &gnss_gpu::DeviceBVH::close)',
        """      .def("set_direction_cache_enabled", &gnss_gpu::DeviceBVH::set_direction_cache_enabled)
      .def_property_readonly("direction_group_count", &gnss_gpu::DeviceBVH::direction_group_count)
      .def_property_readonly("direction_cache_active", &gnss_gpu::DeviceBVH::direction_cache_active)
      .def_property_readonly("direction_cache_bytes", &gnss_gpu::DeviceBVH::direction_cache_bytes)
      .def("close", &gnss_gpu::DeviceBVH::close)""",
    )
    proof = {}
    for name, changes in edits.items():
        restored = result[name]
        for old, new in reversed(changes):
            assert restored.count(new) == 1
            restored = restored.replace(new, old)
        assert restored == original[name]
        proof[name] = dict(
            full_source_undo_exact=True,
            original_sha256=BASE_SHA256[name],
            sha256=hashlib.sha256(
                result[name].replace("\n", "\r\n").encode()
            ).hexdigest(),
        )
    return result, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    projected, proof = project()
    args.output.mkdir(parents=True)
    for name, source in projected.items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.replace("\n", "\r\n").encode())
    for name in ("include/gnss_gpu/cuda_check.h", "include/gnss_gpu/raytrace.h"):
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
        proof[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (args.output / "projection.json").write_text(
        json.dumps(proof, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(dict(projected=True, proof=proof)), flush=True)


if __name__ == "__main__":
    main()
