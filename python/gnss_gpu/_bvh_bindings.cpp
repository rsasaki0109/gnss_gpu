#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "gnss_gpu/bvh.h"
#include <vector>

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

void ensure_finite(const py::buffer_info& buf, const char* message) {
  const auto* values = static_cast<const double*>(buf.ptr);
  for (py::ssize_t i = 0; i < buf.size; ++i) {
    if (!std::isfinite(values[i])) {
      throw std::runtime_error(message);
    }
  }
}

void validate_triangles_mesh(const py::buffer_info& buf, int& n_tri, const char* function_name) {
  if (buf.ndim != 3 || buf.shape[1] != 3 || buf.shape[2] != 3) {
    throw std::runtime_error(std::string(function_name) +
                             ": triangles must have shape (n_tri, 3, 3)");
  }
  n_tri = static_cast<int>(buf.shape[0]);
  if (n_tri <= 0) {
    throw std::runtime_error(std::string(function_name) +
                             ": triangles must contain at least one triangle");
  }
  ensure_finite(buf, "triangles must be finite");
}

void validate_rx_ecef(const py::buffer_info& buf, const char* function_name) {
  if (buf.ndim != 1 || buf.size != 3) {
    throw std::runtime_error(std::string(function_name) + ": rx_ecef must have shape (3,)");
  }
  ensure_finite(buf, "rx_ecef must be finite");
}

void validate_sat_ecef(const py::buffer_info& buf, int& n_sat, const char* function_name) {
  if (buf.ndim == 1) {
    if (buf.size % 3 != 0 || buf.size == 0) {
      throw std::runtime_error(std::string(function_name) +
                               ": sat_ecef must have shape (n_sat, 3)");
    }
    n_sat = static_cast<int>(buf.size / 3);
  } else if (buf.ndim == 2 && buf.shape[1] == 3) {
    n_sat = static_cast<int>(buf.shape[0]);
    if (n_sat <= 0) {
      throw std::runtime_error(std::string(function_name) +
                               ": sat_ecef must contain at least one satellite");
    }
  } else {
    throw std::runtime_error(std::string(function_name) +
                             ": sat_ecef must have shape (n_sat, 3)");
  }
  ensure_finite(buf, "sat_ecef must be finite");
}

void validate_nodes_flat(const py::buffer_info& buf, int& n_nodes, const char* function_name) {
  if (buf.ndim != 2 || buf.shape[1] != 10) {
    throw std::runtime_error(std::string(function_name) +
                             ": nodes_flat must have shape (n_nodes, 10)");
  }
  n_nodes = static_cast<int>(buf.shape[0]);
  if (n_nodes <= 0) {
    throw std::runtime_error(std::string(function_name) +
                             ": nodes_flat must contain at least one node");
  }
  ensure_finite(buf, "nodes_flat must be finite");
}

std::vector<gnss_gpu::BVHNode> unpack_bvh_nodes(const py::buffer_info& bnodes, int n_nodes) {
  const double* nptr = static_cast<const double*>(bnodes.ptr);
  std::vector<gnss_gpu::BVHNode> nodes(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    nodes[i].bbox.min[0] = nptr[i * 10 + 0];
    nodes[i].bbox.min[1] = nptr[i * 10 + 1];
    nodes[i].bbox.min[2] = nptr[i * 10 + 2];
    nodes[i].bbox.max[0] = nptr[i * 10 + 3];
    nodes[i].bbox.max[1] = nptr[i * 10 + 4];
    nodes[i].bbox.max[2] = nptr[i * 10 + 5];
    nodes[i].left = static_cast<int>(nptr[i * 10 + 6]);
    nodes[i].right = static_cast<int>(nptr[i * 10 + 7]);
    nodes[i].tri_start = static_cast<int>(nptr[i * 10 + 8]);
    nodes[i].tri_count = static_cast<int>(nptr[i * 10 + 9]);
  }
  return nodes;
}

void validate_batch_rx_sat(const py::buffer_info& brx,
                           const py::buffer_info& bsat,
                           int& n_epoch,
                           int& n_sat,
                           const char* function_name) {
  if (brx.ndim != 2 || brx.shape[1] != 3) {
    throw std::runtime_error(std::string(function_name) + ": rx_ecef must have shape (N, 3)");
  }
  if (bsat.ndim != 3 || bsat.shape[2] != 3) {
    throw std::runtime_error(std::string(function_name) + ": sat_ecef must have shape (N, n_sat, 3)");
  }
  if (brx.shape[0] != bsat.shape[0]) {
    throw std::runtime_error(std::string(function_name) +
                             ": rx_ecef and sat_ecef must share the leading N");
  }
  n_epoch = static_cast<int>(brx.shape[0]);
  n_sat = static_cast<int>(bsat.shape[1]);
  if (n_epoch <= 0) {
    throw std::runtime_error(std::string(function_name) + ": n_epoch must be >= 1");
  }
  if (n_sat <= 0) {
    throw std::runtime_error(std::string(function_name) + ": n_sat must be >= 1");
  }
}

}  // namespace

PYBIND11_MODULE(_bvh, m) {
  m.doc() = "BVH-accelerated ray tracing for GNSS NLOS detection";

  m.def("bvh_build", [](DoubleArray triangles) {
    auto btri = triangles.request();
    int n_tri = 0;
    validate_triangles_mesh(btri, n_tri, "bvh_build");

    const gnss_gpu::Triangle* tris =
        reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr);

    int max_nodes = 2 * n_tri;
    std::vector<gnss_gpu::BVHNode> nodes(max_nodes);
    std::vector<int> sorted_indices(n_tri);
    int n_nodes = 0;

    gnss_gpu::bvh_build(tris, n_tri, nodes.data(), &n_nodes,
                         sorted_indices.data());

    auto nodes_flat = py::array_t<double>({n_nodes, 10});
    double* nptr = nodes_flat.mutable_data();
    for (int i = 0; i < n_nodes; i++) {
      nptr[i * 10 + 0] = nodes[i].bbox.min[0];
      nptr[i * 10 + 1] = nodes[i].bbox.min[1];
      nptr[i * 10 + 2] = nodes[i].bbox.min[2];
      nptr[i * 10 + 3] = nodes[i].bbox.max[0];
      nptr[i * 10 + 4] = nodes[i].bbox.max[1];
      nptr[i * 10 + 5] = nodes[i].bbox.max[2];
      nptr[i * 10 + 6] = static_cast<double>(nodes[i].left);
      nptr[i * 10 + 7] = static_cast<double>(nodes[i].right);
      nptr[i * 10 + 8] = static_cast<double>(nodes[i].tri_start);
      nptr[i * 10 + 9] = static_cast<double>(nodes[i].tri_count);
    }

    auto indices_arr = py::array_t<int>(std::vector<py::ssize_t>{n_tri});
    int* iptr = indices_arr.mutable_data();
    for (int i = 0; i < n_tri; i++) iptr[i] = sorted_indices[i];

    return py::make_tuple(nodes_flat, indices_arr);
  }, "Build BVH from triangle mesh", py::arg("triangles"));

  m.def("raytrace_los_check_bvh", [](DoubleArray rx_ecef,
                                      DoubleArray sat_ecef,
                                      DoubleArray nodes_flat,
                                      DoubleArray sorted_tris) {
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto bnodes = nodes_flat.request();
    auto btri = sorted_tris.request();

    validate_rx_ecef(brx, "raytrace_los_check_bvh");
    int n_sat = 0;
    validate_sat_ecef(bsat, n_sat, "raytrace_los_check_bvh");
    int n_nodes = 0;
    validate_nodes_flat(bnodes, n_nodes, "raytrace_los_check_bvh");
    int n_tri = 0;
    validate_triangles_mesh(btri, n_tri, "raytrace_los_check_bvh");

    auto nodes = unpack_bvh_nodes(bnodes, n_nodes);

    auto is_los_int = py::array_t<int>(std::vector<py::ssize_t>{n_sat});
    int* int_ptr = is_los_int.mutable_data();

    gnss_gpu::raytrace_los_check_bvh(
        static_cast<double*>(brx.ptr),
        static_cast<double*>(bsat.ptr),
        nodes.data(),
        reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr),
        int_ptr,
        n_sat, n_nodes);

    auto is_los = py::array_t<bool>(std::vector<py::ssize_t>{n_sat});
    bool* bool_ptr = is_los.mutable_data();
    for (int i = 0; i < n_sat; i++) bool_ptr[i] = (int_ptr[i] != 0);
    return is_los;
  }, "BVH-accelerated LOS check",
     py::arg("rx_ecef"), py::arg("sat_ecef"),
     py::arg("nodes_flat"), py::arg("sorted_tris"));

  m.def("raytrace_los_check_bvh_batch", [](DoubleArray rx_ecef,
                                            DoubleArray sat_ecef,
                                            DoubleArray nodes_flat,
                                            DoubleArray sorted_tris) {
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto bnodes = nodes_flat.request();
    auto btri = sorted_tris.request();

    int n_epoch = 0;
    int n_sat = 0;
    validate_batch_rx_sat(brx, bsat, n_epoch, n_sat, "raytrace_los_check_bvh_batch");
    int n_nodes = 0;
    validate_nodes_flat(bnodes, n_nodes, "raytrace_los_check_bvh_batch");
    int n_tri = 0;
    validate_triangles_mesh(btri, n_tri, "raytrace_los_check_bvh_batch");

    auto nodes = unpack_bvh_nodes(bnodes, n_nodes);

    auto is_los_int = py::array_t<int>({n_epoch, n_sat});
    int* int_ptr = is_los_int.mutable_data();

    gnss_gpu::raytrace_los_check_bvh_batch(
        static_cast<double*>(brx.ptr),
        static_cast<double*>(bsat.ptr),
        nodes.data(),
        reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr),
        int_ptr,
        n_epoch, n_sat, n_nodes);

    auto is_los = py::array_t<bool>({n_epoch, n_sat});
    bool* bool_ptr = is_los.mutable_data();
    int total = n_epoch * n_sat;
    for (int i = 0; i < total; i++) bool_ptr[i] = (int_ptr[i] != 0);
    return is_los;
  }, "Batched BVH-accelerated LOS check across N epochs",
     py::arg("rx_ecef"), py::arg("sat_ecef"),
     py::arg("nodes_flat"), py::arg("sorted_tris"));

  m.def("raytrace_multipath_bvh_batch", [](DoubleArray rx_ecef,
                                            DoubleArray sat_ecef,
                                            DoubleArray nodes_flat,
                                            DoubleArray sorted_tris) {
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto bnodes = nodes_flat.request();
    auto btri = sorted_tris.request();

    int n_epoch = 0;
    int n_sat = 0;
    validate_batch_rx_sat(brx, bsat, n_epoch, n_sat, "raytrace_multipath_bvh_batch");
    int n_nodes = 0;
    validate_nodes_flat(bnodes, n_nodes, "raytrace_multipath_bvh_batch");
    int n_tri = 0;
    validate_triangles_mesh(btri, n_tri, "raytrace_multipath_bvh_batch");

    auto nodes = unpack_bvh_nodes(bnodes, n_nodes);

    auto refl_arr = py::array_t<double>({n_epoch, n_sat, 3});
    auto delay_arr = py::array_t<double>({n_epoch, n_sat});

    gnss_gpu::raytrace_multipath_bvh_batch(
        static_cast<double*>(brx.ptr),
        static_cast<double*>(bsat.ptr),
        nodes.data(),
        reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr),
        static_cast<double*>(refl_arr.mutable_data()),
        static_cast<double*>(delay_arr.mutable_data()),
        n_epoch, n_sat, n_nodes);

    return py::make_tuple(refl_arr, delay_arr);
  }, "Batched BVH-accelerated multipath reflection across N epochs",
     py::arg("rx_ecef"), py::arg("sat_ecef"),
     py::arg("nodes_flat"), py::arg("sorted_tris"));
}
