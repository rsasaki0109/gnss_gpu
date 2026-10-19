"""Fuse exact direction groups into candidate threads, with no ray cache."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parents[1] / "bvh_baseline_iterative_20261019"
PARENT = ROOT / "benchmarks/probes/bvh_grouped_directions/generate.py"
SPEC = importlib.util.spec_from_file_location("d296_frozen_group_projector", PARENT)
OLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OLD)
OLD.ROOT = BASE
BASE_SHA256 = OLD.BASE_SHA256
function = OLD.function


def project():
    projected, original_proof = OLD.project()
    initial = dict(projected)
    changes = []

    def replace(name, before, after):
        assert projected[name].count(before) == 1, (name, before[:100])
        projected[name] = projected[name].replace(before, after)
        changes.append((name, before, after))

    cu = "src/raytrace/bvh.cu"
    original = (BASE / cu).read_text(encoding="utf8")
    source = projected[cu]
    candidate = OLD.function(original, "multipath_candidate")
    tail = candidate[candidate.index("  double t;") : candidate.rfind("}")]
    helpers = """
// All operands are the original bits. Only independent triangles within an
// exact group reuse the same thread's mirror and normalized direction registers.
__device__ static bool multipath_register_candidate(const double rx[3], const double sat[3],
                                                   double d_rx_sat, const Triangle& tri,
                                                   MultipathCandidate& result,
                                                   const double mirror[3], const double dir[3]) {
CANDIDATE_TAIL
}

""".replace("CANDIDATE_TAIL", tail)
    a = original.index(
        "template <bool CACHED_NORMALS, bool SHARED_MIRRORS>\n__global__ void multipath_candidates_kernel("
    )
    b = original.index("\n__global__ void multipath_finalize_kernel", a)
    kernel = original[a:b].replace(
        "template <bool CACHED_NORMALS, bool SHARED_MIRRORS>\n", ""
    )
    kernel = kernel.replace(
        "multipath_candidates_kernel", "multipath_register_groups_kernel"
    )
    kernel = kernel.replace(
        "const ReflectionMirror* mirrors)",
        "const int* offsets, const int* members, int n_groups)",
    )
    a = kernel.index("  for (int ordinal = tile * blockDim.x + threadIdx.x;")
    b = kernel.index("  __shared__ MultipathCandidate winners", a)
    mirror_math = candidate[
        candidate.index("    double dv[3]") : candidate.index(
            "\n  }\n\n  double dir[3]"
        )
    ]
    normalize = candidate[
        candidate.index("  double dir[3]") : candidate.index("\n  double t;")
    ]
    normalize = normalize.replace("return false;", "continue;")
    loop = """  for (int group = tile * blockDim.x + threadIdx.x;
       group < n_groups; group += tiles * blockDim.x) {
    int begin = offsets[group], end = offsets[group + 1];
    int representative = members[begin];
    const ReflectionNormal& normal = normals[representative];
    const Triangle& tri = tris[triangle_order[representative]];
    double n[3] = {normal.value[0], normal.value[1], normal.value[2]};
    double mirror[3];
MIRROR_MATH
NORMALIZE
    for (int entry = begin; entry < end; ++entry) {
      int ordinal = members[entry];
      MultipathCandidate candidate = {};
      candidate.order = ordinal;
      if (multipath_register_candidate(rx, sat, d_rx_sat, tris[triangle_order[ordinal]],
                                       candidate, mirror, dir) &&
          better_multipath(candidate, best)) best = candidate;
    }
  }
""".replace("MIRROR_MATH", mirror_math).replace("NORMALIZE", normalize)
    kernel = kernel[:a] + loop + kernel[b:]
    a = source.index("\n// One entry is valid only for this call's")
    b = source.index("\n__global__ void multipath_finalize_kernel", a)
    replace(cu, source[a:b], helpers + kernel)
    replace(
        cu,
        "  int direction_groups = 0, direction_epochs = 0;",
        "  int direction_groups = 0;",
    )
    replace(
        cu,
        "  DeviceBuffer<int> direction_map, direction_representatives;\n  DeviceBuffer<GroupedDirection> directions;",
        "  DeviceBuffer<int> direction_members, direction_offsets;",
    )
    replace(
        cu,
        "    direction_map.reset(); direction_representatives.reset(); directions.reset();",
        "    direction_members.reset(); direction_offsets.reset();",
    )
    prepare = OLD.function(projected[cu], "prepare_direction_groups")
    updated = prepare.replace(
        "    representatives.reserve((size_t)n_order);",
        "    representatives.reserve((size_t)n_order);\n    std::vector<int> group_sizes;",
    )
    updated = updated.replace(
        "      if (found == groups.end()) {",
        "      if (found == groups.end() || group_sizes[found->second] == 8) {",
    )
    updated = updated.replace(
        "        groups.emplace(key, group);",
        "        if (found == groups.end()) groups.emplace(key, group);\n        else found->second = group;",
    )
    updated = updated.replace(
        "        representatives.push_back(ordinal);",
        "        representatives.push_back(ordinal);\n        group_sizes.push_back(0);",
    )
    updated = updated.replace(
        "      } else mapping[ordinal] = found->second;",
        "      } else mapping[ordinal] = found->second;\n      ++group_sizes[mapping[ordinal]];",
    )
    old_upload = """    direction_map.upload(mapping.data(), mapping.size());
    direction_representatives.upload(representatives.data(), representatives.size());
    direction_groups = (int)representatives.size();"""
    new_upload = """    std::vector<int> offsets(representatives.size() + 1, 0);
    for (int group : mapping) if (group >= 0) ++offsets[(size_t)group + 1];
    for (size_t group = 1; group < offsets.size(); ++group) offsets[group] += offsets[group - 1];
    std::vector<int> members((size_t)offsets.back()), cursor = offsets;
    for (int ordinal = 0; ordinal < n_order; ++ordinal) {
      int group = mapping[ordinal];
      if (group >= 0) members[cursor[group]++] = ordinal;
    }
    direction_members.upload(members.data(), members.size());
    direction_offsets.upload(offsets.data(), offsets.size());
    direction_groups = (int)representatives.size();"""
    assert old_upload in updated
    updated = updated.replace(old_upload, new_upload)
    replace(cu, prepare, updated)
    prepare = OLD.function(projected[cu], "prepare_grouped_directions")
    updated = """prepare_grouped_directions(int total, int n_epoch, int n_sat) {
    if (!direction_requested || !direction_capable || total < 512 || n_sat < 8) return false;
    if (!direction_attempted) prepare_direction_groups();
    if (direction_groups <= 0) return false;
    direction_active = true;
    return true;
  }"""
    replace(cu, prepare, updated)
    replace(cu, "    direction_epochs = 0;\n", "    // Directions stay in registers; no epoch cache is needed.\n")
    source = projected[cu]
    a = source.index(
        "    if (direction_active) {", source.index("  void launch_multipath(")
    )
    b = source.index("    if (tiles && parallel_epochs) {", a)
    launch = """    if (direction_active) {
      int epoch_step = parallel_epochs ? parallel_epochs : n_epoch;
      for (int first_epoch = 0; first_epoch < n_epoch; first_epoch += epoch_step) {
        int epochs = std::min(epoch_step, n_epoch - first_epoch);
        int rays = epochs * n_sat;
        size_t ray_offset = (size_t)first_epoch * n_sat;
        multipath_register_groups_kernel<<<rays * tiles, MULTIPATH_BLOCK_SIZE>>>(
            rx.data + (size_t)first_epoch * 3, sat.data + ray_offset * 3,
            triangles.data, order.data, partial.data, n_sat, n_order, tiles,
            normals.data, direction_offsets.data, direction_members.data, direction_groups);
        CUDA_CHECK_LAST();
        multipath_finalize_kernel<<<(rays + MULTIPATH_BLOCK_SIZE - 1) / MULTIPATH_BLOCK_SIZE,
                                    MULTIPATH_BLOCK_SIZE>>>(
            partial.data, points.data + ray_offset * 3, delays.data + ray_offset, rays, tiles);
        CUDA_CHECK_LAST();
      }
      return;
    }
"""
    replace(cu, source[a:b], launch)
    replace(
        cu,
        "      (s.direction_map.capacity + s.direction_representatives.capacity) * sizeof(int) +\n      s.directions.capacity * sizeof(GroupedDirection);",
        "      (s.direction_members.capacity + s.direction_offsets.capacity) * sizeof(int);",
    )
    replace(
        cu,
        "  return impl_ ? impl_->directions.capacity * sizeof(GroupedDirection) : 0;",
        "  return impl_ ? (impl_->direction_members.capacity + impl_->direction_offsets.capacity) * sizeof(int) : 0;",
    )
    renames = {
        "grouped_directions": "grouped_candidates",
        "set_direction_cache_enabled": "set_grouped_candidates_enabled",
        "direction_cache_active": "grouped_candidates_active",
        "direction_cache_bytes": "group_metadata_bytes",
        "direction_group_count": "group_count",
    }
    for name in projected:
        for before, after in renames.items():
            if before in projected[name]:
                previous = projected[name]
                projected[name] = previous.replace(before, after)
                changes.append((name, previous, projected[name]))
    restored = dict(projected)
    for name, before, after in reversed(changes):
        assert restored[name].count(after) == 1
        restored[name] = restored[name].replace(after, before)
    assert restored == initial
    proof = {
        name: dict(
            sha256=hashlib.sha256(text.replace("\n", "\r\n").encode()).hexdigest(),
            full_composed_source_undo_exact=True,
            original_D295_sha256=OLD.BASE_SHA256[name],
            parent_D296_full_undo_exact=original_proof[name]["full_source_undo_exact"],
        )
        for name, text in projected.items()
    }
    return projected, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    projected, proof = project()
    args.output.mkdir(parents=True)
    for name, text in projected.items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.replace("\n", "\r\n").encode())
    for name in ("include/gnss_gpu/cuda_check.h", "include/gnss_gpu/raytrace.h"):
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BASE / name, path)
        proof[name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (args.output / "projection.json").write_text(
        json.dumps(proof, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(dict(projected=True, proof=proof)), flush=True)


if __name__ == "__main__":
    main()
