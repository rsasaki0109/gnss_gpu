#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/bvh.h"
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(_bvh, m) {
  m.doc() = "BVH-accelerated ray tracing for GNSS NLOS detection";

  // Build BVH from triangles, return (nodes_flat, sorted_tri_indices)
  // nodes_flat: [n_nodes, 10] (min[3], max[3], left, right, tri_start, tri_count)
  m.def("bvh_build", [](py::array_t<double> triangles) {
    auto btri = triangles.request();
    int n_tri = (int)btri.shape[0];

    const gnss_gpu::Triangle* tris =
        reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr);

    // Allocate max possible nodes
    int max_nodes = 2 * n_tri;
    std::vector<gnss_gpu::BVHNode> nodes(max_nodes);
    std::vector<int> sorted_indices(n_tri);
    int n_nodes = 0;

    gnss_gpu::bvh_build(tris, n_tri, nodes.data(), &n_nodes,
                         sorted_indices.data());

    // Pack nodes into flat array: [n_nodes, 10]
    // Layout: min_x, min_y, min_z, max_x, max_y, max_z, left, right, tri_start, tri_count
    auto nodes_flat = py::array_t<double>({n_nodes, 10});
    double* nptr = nodes_flat.mutable_data();
    for (int i = 0; i < n_nodes; i++) {
      nptr[i * 10 + 0] = nodes[i].bbox.min[0];
      nptr[i * 10 + 1] = nodes[i].bbox.min[1];
      nptr[i * 10 + 2] = nodes[i].bbox.min[2];
      nptr[i * 10 + 3] = nodes[i].bbox.max[0];
      nptr[i * 10 + 4] = nodes[i].bbox.max[1];
      nptr[i * 10 + 5] = nodes[i].bbox.max[2];
      nptr[i * 10 + 6] = (double)nodes[i].left;
      nptr[i * 10 + 7] = (double)nodes[i].right;
      nptr[i * 10 + 8] = (double)nodes[i].tri_start;
      nptr[i * 10 + 9] = (double)nodes[i].tri_count;
    }

    auto indices_arr = py::array_t<int>(std::vector<ssize_t>{n_tri});
    int* iptr = indices_arr.mutable_data();
    for (int i = 0; i < n_tri; i++) iptr[i] = sorted_indices[i];

    return py::make_tuple(nodes_flat, indices_arr);
  }, "Build BVH from triangle mesh", py::arg("triangles"));

  // LOS check with BVH
  m.def("raytrace_los_check_bvh", [](py::array_t<double> rx_ecef,
                                      py::array_t<double> sat_ecef,
                                      py::array_t<double> nodes_flat,
                                      py::array_t<double> sorted_tris) {
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto bnodes = nodes_flat.request();
    auto btri = sorted_tris.request();

    if (brx.size < 3)
      throw std::runtime_error("rx_ecef must have at least 3 elements");

    int n_sat = (int)bsat.shape[0];
    int n_nodes = (int)bnodes.shape[0];
    int n_tri = (int)btri.shape[0];

    // Unpack flat nodes back to BVHNode structs
    const double* nptr = static_cast<const double*>(bnodes.ptr);
    std::vector<gnss_gpu::BVHNode> nodes(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
      nodes[i].bbox.min[0] = nptr[i * 10 + 0];
      nodes[i].bbox.min[1] = nptr[i * 10 + 1];
      nodes[i].bbox.min[2] = nptr[i * 10 + 2];
      nodes[i].bbox.max[0] = nptr[i * 10 + 3];
      nodes[i].bbox.max[1] = nptr[i * 10 + 4];
      nodes[i].bbox.max[2] = nptr[i * 10 + 5];
      nodes[i].left = (int)nptr[i * 10 + 6];
      nodes[i].right = (int)nptr[i * 10 + 7];
      nodes[i].tri_start = (int)nptr[i * 10 + 8];
      nodes[i].tri_count = (int)nptr[i * 10 + 9];
    }

    auto is_los_int = py::array_t<int>(std::vector<ssize_t>{n_sat});
    int* int_ptr = is_los_int.mutable_data();

    gnss_gpu::raytrace_los_check_bvh(
        static_cast<double*>(brx.ptr),
        static_cast<double*>(bsat.ptr),
        nodes.data(),
        reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr),
        int_ptr,
        n_sat, n_nodes);

    auto is_los = py::array_t<bool>(std::vector<ssize_t>{n_sat});
    bool* bool_ptr = is_los.mutable_data();
    for (int i = 0; i < n_sat; i++) bool_ptr[i] = (int_ptr[i] != 0);
    return is_los;
  }, "BVH-accelerated LOS check",
     py::arg("rx_ecef"), py::arg("sat_ecef"),
     py::arg("nodes_flat"), py::arg("sorted_tris"));

  // Multipath reflection with BVH
  m.def("raytrace_multipath_bvh", [](py::array_t<double> rx_ecef,
                                      py::array_t<double> sat_ecef,
                                      py::array_t<double> nodes_flat,
                                      py::array_t<double> sorted_tris) {
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto bnodes = nodes_flat.request();
    auto btri = sorted_tris.request();

    if (brx.size < 3)
      throw std::runtime_error("rx_ecef must have at least 3 elements");

    int n_sat = (int)bsat.shape[0];
    int n_nodes = (int)bnodes.shape[0];

    const double* nptr = static_cast<const double*>(bnodes.ptr);
    std::vector<gnss_gpu::BVHNode> nodes(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
      nodes[i].bbox.min[0] = nptr[i * 10 + 0];
      nodes[i].bbox.min[1] = nptr[i * 10 + 1];
      nodes[i].bbox.min[2] = nptr[i * 10 + 2];
      nodes[i].bbox.max[0] = nptr[i * 10 + 3];
      nodes[i].bbox.max[1] = nptr[i * 10 + 4];
      nodes[i].bbox.max[2] = nptr[i * 10 + 5];
      nodes[i].left = (int)nptr[i * 10 + 6];
      nodes[i].right = (int)nptr[i * 10 + 7];
      nodes[i].tri_start = (int)nptr[i * 10 + 8];
      nodes[i].tri_count = (int)nptr[i * 10 + 9];
    }

    auto refl_arr = py::array_t<double>({n_sat, 3});
    auto delay_arr = py::array_t<double>({n_sat});

    gnss_gpu::raytrace_multipath_bvh(
        static_cast<double*>(brx.ptr),
        static_cast<double*>(bsat.ptr),
        nodes.data(),
        reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr),
        static_cast<double*>(refl_arr.mutable_data()),
        static_cast<double*>(delay_arr.mutable_data()),
        n_sat, n_nodes);

    return py::make_tuple(refl_arr, delay_arr);
  }, "BVH-accelerated multipath reflection",
     py::arg("rx_ecef"), py::arg("sat_ecef"),
     py::arg("nodes_flat"), py::arg("sorted_tris"));
}
