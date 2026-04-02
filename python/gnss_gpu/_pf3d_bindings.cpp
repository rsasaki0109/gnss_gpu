#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/pf_3d.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_pf3d, m) {
  m.doc() = "GPU-accelerated 3D-aware particle filter weight computation";

  m.def("pf_weight_3d", [](py::array_t<double> px, py::array_t<double> py_arr,
                            py::array_t<double> pz, py::array_t<double> pcb,
                            py::array_t<double> sat_ecef,
                            py::array_t<double> pseudoranges,
                            py::array_t<double> weights_sat,
                            py::array_t<double> triangles,
                            py::array_t<double> log_weights,
                            int n_particles, int n_sat,
                            double sigma_pr_los, double sigma_pr_nlos,
                            double nlos_bias,
                            double blocked_nlos_prob,
                            double clear_nlos_prob) {
    {
      auto bs = sat_ecef.request();
      // sat_ecef: accept (N,3) or (N*3,) flat
    }
    auto btri = triangles.request();
    int n_tri = btri.shape[0];

    gnss_gpu::pf_weight_3d(
        static_cast<double*>(px.request().ptr),
        static_cast<double*>(py_arr.request().ptr),
        static_cast<double*>(pz.request().ptr),
        static_cast<double*>(pcb.request().ptr),
        static_cast<double*>(sat_ecef.request().ptr),
        static_cast<double*>(pseudoranges.request().ptr),
        static_cast<double*>(weights_sat.request().ptr),
        reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr),
        n_tri,
        static_cast<double*>(log_weights.request().ptr),
        n_particles, n_sat,
        sigma_pr_los, sigma_pr_nlos, nlos_bias,
        blocked_nlos_prob, clear_nlos_prob);
  }, "Compute 3D-aware pseudorange likelihood weights using ray tracing",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("sat_ecef"), py::arg("pseudoranges"),
     py::arg("weights_sat"), py::arg("triangles"),
     py::arg("log_weights"),
     py::arg("n_particles"), py::arg("n_sat"),
     py::arg("sigma_pr_los"), py::arg("sigma_pr_nlos"),
     py::arg("nlos_bias"),
     py::arg("blocked_nlos_prob") = 1.0,
     py::arg("clear_nlos_prob") = 0.0);
}
