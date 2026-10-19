"""Prepare exact reflection groups only when an eligible call requests them."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / "benchmarks/probes/bvh_grouped_candidates/generate_checked.py"
SPEC = importlib.util.spec_from_file_location("d297_frozen_projector", PARENT)
OLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OLD)
BASE = OLD.BASE
BASE_SHA256 = OLD.BASE_SHA256
function = OLD.function
D297_SHA256 = {
    "src/raytrace/bvh.cu": "6bcda85f9daa56e7aa1ccffadc3f87bd908c9a227938b344b198eddb95fd7b64",
    "include/gnss_gpu/bvh.h": "cf66421b27ea0e5990463cb8430ac2f63b09c761215d55df2b2cea44f1bcbcea",
    "python/gnss_gpu/_bvh_bindings.cpp": "4d8384117a0dd84eb8349bb6069e9daac7ad15eec43b64bf9ccf91d33cf7e543",
}


def project():
    projected, parent_proof = OLD.project()
    initial = dict(projected)
    for name, digest in D297_SHA256.items():
        assert hashlib.sha256(projected[name].replace("\n", "\r\n").encode()).hexdigest() == digest
    changes = []

    def replace(name, before, after):
        assert projected[name].count(before) == 1, (name, before[:100])
        projected[name] = projected[name].replace(before, after)
        changes.append((name, before, after))

    cu = "src/raytrace/bvh.cu"
    marker = "// Mirror the serial kernel's right-first leaf traversal and stack limit."
    copy_kernel = """// Pack the original vertex bits from the scene's owned device snapshot.
// This kernel performs only copies; all reflection arithmetic stays unchanged.
__global__ void reflection_group_vertices_kernel(const Triangle* triangles,
                                                const int* order, double* vertices,
                                                int count) {
  int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= count) return;
  const Triangle& triangle = triangles[order[ordinal]];
  vertices[(size_t)ordinal * 3] = triangle.v0[0];
  vertices[(size_t)ordinal * 3 + 1] = triangle.v0[1];
  vertices[(size_t)ordinal * 3 + 2] = triangle.v0[2];
}

"""
    replace(cu, marker, copy_kernel + marker)
    replace(
        cu,
        "  std::vector<std::array<double, 3>> direction_v0;",
        "  int direction_preparations = 0;\n  size_t direction_preparation_scratch = 0;",
    )
    replace(
        cu,
        "    direction_attempted = true;\n    std::vector<ReflectionNormal> saved_normals(n_order);",
        """    direction_attempted = true;
    std::vector<std::array<double, 3>> direction_v0((size_t)n_order);
    DeviceBuffer<double> packed_vertices;
    packed_vertices.reserve((size_t)n_order * 3);
    direction_preparation_scratch = (size_t)n_order * 3 * sizeof(double);
    reflection_group_vertices_kernel<<<(n_order + 255) / 256, 256>>>(
        triangles.data, order.data, packed_vertices.data, n_order);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(direction_v0.data(), packed_vertices.data,
        direction_preparation_scratch, cudaMemcpyDeviceToHost));
    // The temporary is released before CPU grouping and CSR allocations.
    packed_vertices.reset();
    ++direction_preparations;
    std::vector<ReflectionNormal> saved_normals(n_order);""",
    )
    source = projected[cu]
    start = source.index("  if (grouped_candidates && state->ordered")
    stop = source.index("  impl_ = std::move(state);", start)
    replace(
        cu,
        source[start:stop],
        """  state->direction_capable = state->ordered && state->n_order >= 4096 && state->n_order <= 262144;
  state->direction_requested = grouped_candidates && state->direction_capable;
""",
    )
    marker = "void DeviceBVH::check_los(const double* h_rx"
    getters = """int DeviceBVH::group_preparation_count() const noexcept {
  return impl_ ? impl_->direction_preparations : 0;
}
std::size_t DeviceBVH::group_preparation_scratch_bytes() const noexcept {
  return impl_ ? impl_->direction_preparation_scratch : 0;
}

"""
    replace(cu, marker, getters + marker)
    replace(
        "include/gnss_gpu/bvh.h",
        "  std::size_t group_metadata_bytes() const noexcept;",
        """  std::size_t group_metadata_bytes() const noexcept;
  int group_preparation_count() const noexcept;
  // Last one-time vertex packing scratch request, released before API return.
  std::size_t group_preparation_scratch_bytes() const noexcept;""",
    )
    replace(
        "python/gnss_gpu/_bvh_bindings.cpp",
        '      .def_property_readonly("group_metadata_bytes", &gnss_gpu::DeviceBVH::group_metadata_bytes)',
        """      .def_property_readonly("group_metadata_bytes", &gnss_gpu::DeviceBVH::group_metadata_bytes)
      .def_property_readonly("group_preparation_count", &gnss_gpu::DeviceBVH::group_preparation_count)
      .def_property_readonly("group_preparation_scratch_bytes", &gnss_gpu::DeviceBVH::group_preparation_scratch_bytes,
                            "Last one-time packing scratch request; freed before grouping and API return")""",
    )
    restored = dict(projected)
    for name, before, after in reversed(changes):
        assert restored[name].count(after) == 1
        restored[name] = restored[name].replace(after, before)
    assert restored == initial
    proof = {
        name: dict(
            sha256=hashlib.sha256(text.replace("\n", "\r\n").encode()).hexdigest(),
            full_composed_source_undo_exact=True,
            original_D295_sha256=BASE_SHA256[name],
            parent_D297_sha256=D297_SHA256[name],
            parent_D297_full_undo_exact=parent_proof[name]["full_composed_source_undo_exact"],
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
    (args.output / "projection.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
