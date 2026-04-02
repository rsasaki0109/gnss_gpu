#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/pf_3d_bvh.h"
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_pf3d_bvh, m) {
  m.doc() = "GPU-accelerated BVH-accelerated 3D-aware particle filter weight computation";

  // pf_weight_3d_bvh(px, py, pz, pcb, sat_ecef, pseudoranges, weights_sat,
  //                  nodes_flat, sorted_tris, log_weights,
  //                  n_particles, n_sat,
  //                  sigma_pr_los, sigma_pr_nlos, nlos_bias)
  //
  // nodes_flat: [n_nodes, 10] double array (same layout as _bvh_bindings.cpp)
  //   columns: min_x, min_y, min_z, max_x, max_y, max_z,
  //            left, right, tri_start, tri_count
  // sorted_tris: [n_tri, 3, 3] double array of reordered triangles
  m.def("pf_weight_3d_bvh",
    [](py::array_t<double> px, py::array_t<double> py_arr,
       py::array_t<double> pz, py::array_t<double> pcb,
       py::array_t<double> sat_ecef,
       py::array_t<double> pseudoranges,
       py::array_t<double> weights_sat,
       py::array_t<double> nodes_flat,
       py::array_t<double> sorted_tris,
       py::array_t<double> log_weights,
       int n_particles, int n_sat,
       double sigma_pr_los, double sigma_pr_nlos,
       double nlos_bias,
       double blocked_nlos_prob,
       double clear_nlos_prob) {

      auto bnodes = nodes_flat.request();
      auto btri   = sorted_tris.request();

      {
        auto bs = sat_ecef.request();
        // sat_ecef: accept (N,3) or (N*3,) flat
      }

      int n_nodes = (int)bnodes.shape[0];
      int n_tri   = (int)btri.shape[0];

      // Unpack flat node array into BVHNode structs
      const double* nptr = static_cast<const double*>(bnodes.ptr);
      std::vector<gnss_gpu::BVHNode> nodes(n_nodes);
      for (int i = 0; i < n_nodes; i++) {
        nodes[i].bbox.min[0] = nptr[i * 10 + 0];
        nodes[i].bbox.min[1] = nptr[i * 10 + 1];
        nodes[i].bbox.min[2] = nptr[i * 10 + 2];
        nodes[i].bbox.max[0] = nptr[i * 10 + 3];
        nodes[i].bbox.max[1] = nptr[i * 10 + 4];
        nodes[i].bbox.max[2] = nptr[i * 10 + 5];
        nodes[i].left      = (int)nptr[i * 10 + 6];
        nodes[i].right     = (int)nptr[i * 10 + 7];
        nodes[i].tri_start = (int)nptr[i * 10 + 8];
        nodes[i].tri_count = (int)nptr[i * 10 + 9];
      }

      gnss_gpu::pf_weight_3d_bvh(
          static_cast<double*>(px.request().ptr),
          static_cast<double*>(py_arr.request().ptr),
          static_cast<double*>(pz.request().ptr),
          static_cast<double*>(pcb.request().ptr),
          static_cast<double*>(sat_ecef.request().ptr),
          static_cast<double*>(pseudoranges.request().ptr),
          static_cast<double*>(weights_sat.request().ptr),
          nodes.data(), n_nodes,
          reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr), n_tri,
          static_cast<double*>(log_weights.request().ptr),
          n_particles, n_sat,
          sigma_pr_los, sigma_pr_nlos, nlos_bias,
          blocked_nlos_prob, clear_nlos_prob);
    },
    "Compute 3D-aware pseudorange likelihood weights using BVH-accelerated ray tracing",
    py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
    py::arg("sat_ecef"), py::arg("pseudoranges"),
    py::arg("weights_sat"),
    py::arg("nodes_flat"), py::arg("sorted_tris"),
    py::arg("log_weights"),
    py::arg("n_particles"), py::arg("n_sat"),
    py::arg("sigma_pr_los"), py::arg("sigma_pr_nlos"),
    py::arg("nlos_bias"),
    py::arg("blocked_nlos_prob") = 1.0,
    py::arg("clear_nlos_prob") = 0.0);
}
