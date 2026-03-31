#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/svgd.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_svgd, m) {
  m.doc() = "GPU-accelerated SVGD for particle filter (MegaParticles-style)";

  m.def("pf_estimate_bandwidth", [](py::array_t<double> px, py::array_t<double> py_arr,
                                     py::array_t<double> pz, py::array_t<double> pcb,
                                     int n_particles, int n_subsample,
                                     unsigned long long seed) {
    return gnss_gpu::pf_estimate_bandwidth(
        static_cast<double*>(px.request().ptr),
        static_cast<double*>(py_arr.request().ptr),
        static_cast<double*>(pz.request().ptr),
        static_cast<double*>(pcb.request().ptr),
        n_particles, n_subsample, seed);
  }, "Estimate RBF kernel bandwidth using median heuristic",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("n_particles"), py::arg("n_subsample"), py::arg("seed"));

  m.def("pf_svgd_step", [](py::array_t<double> px, py::array_t<double> py_arr,
                             py::array_t<double> pz, py::array_t<double> pcb,
                             py::array_t<double> sat_ecef, py::array_t<double> pseudoranges,
                             py::array_t<double> weights_sat,
                             int n_particles, int n_sat,
                             double sigma_pr, double step_size,
                             int n_neighbors, double bandwidth,
                             unsigned long long seed, int step) {
    {
      auto bs = sat_ecef.request();
      // sat_ecef: accept (N,3) or (N*3,) flat
    }
    gnss_gpu::pf_svgd_step(
        static_cast<double*>(px.request().ptr),
        static_cast<double*>(py_arr.request().ptr),
        static_cast<double*>(pz.request().ptr),
        static_cast<double*>(pcb.request().ptr),
        static_cast<double*>(sat_ecef.request().ptr),
        static_cast<double*>(pseudoranges.request().ptr),
        static_cast<double*>(weights_sat.request().ptr),
        n_particles, n_sat,
        sigma_pr, step_size,
        n_neighbors, bandwidth,
        seed, step);
  }, "Perform one SVGD step on particles",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights_sat"),
     py::arg("n_particles"), py::arg("n_sat"),
     py::arg("sigma_pr"), py::arg("step_size"),
     py::arg("n_neighbors"), py::arg("bandwidth"),
     py::arg("seed"), py::arg("step"));

  m.def("pf_svgd_estimate", [](py::array_t<double> px, py::array_t<double> py_arr,
                                 py::array_t<double> pz, py::array_t<double> pcb,
                                 py::array_t<double> result,
                                 int n_particles) {
    gnss_gpu::pf_svgd_estimate(
        static_cast<double*>(px.request().ptr),
        static_cast<double*>(py_arr.request().ptr),
        static_cast<double*>(pz.request().ptr),
        static_cast<double*>(pcb.request().ptr),
        static_cast<double*>(result.request().ptr),
        n_particles);
  }, "Compute simple mean estimate (equal weights after SVGD)",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("result"), py::arg("n_particles"));
}
