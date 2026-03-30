#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/particle_filter.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_pf, m) {
  m.doc() = "GPU-accelerated Mega Particle Filter for GNSS positioning";

  m.def("pf_initialize", [](py::array_t<double> px, py::array_t<double> py_arr,
                             py::array_t<double> pz, py::array_t<double> pcb,
                             double init_x, double init_y, double init_z, double init_cb,
                             double spread_pos, double spread_cb,
                             int n_particles, unsigned long long seed) {
    gnss_gpu::pf_initialize(
        static_cast<double*>(px.request().ptr),
        static_cast<double*>(py_arr.request().ptr),
        static_cast<double*>(pz.request().ptr),
        static_cast<double*>(pcb.request().ptr),
        init_x, init_y, init_z, init_cb,
        spread_pos, spread_cb,
        n_particles, seed);
  }, "Initialize particles around initial position",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("init_x"), py::arg("init_y"), py::arg("init_z"), py::arg("init_cb"),
     py::arg("spread_pos"), py::arg("spread_cb"),
     py::arg("n_particles"), py::arg("seed"));

  m.def("pf_predict", [](py::array_t<double> px, py::array_t<double> py_arr,
                          py::array_t<double> pz, py::array_t<double> pcb,
                          py::array_t<double> vx, py::array_t<double> vy,
                          py::array_t<double> vz,
                          double dt, double sigma_pos, double sigma_cb,
                          int n_particles, unsigned long long seed, int step) {
    gnss_gpu::pf_predict(
        static_cast<double*>(px.request().ptr),
        static_cast<double*>(py_arr.request().ptr),
        static_cast<double*>(pz.request().ptr),
        static_cast<double*>(pcb.request().ptr),
        static_cast<double*>(vx.request().ptr),
        static_cast<double*>(vy.request().ptr),
        static_cast<double*>(vz.request().ptr),
        dt, sigma_pos, sigma_cb,
        n_particles, seed, step);
  }, "Predict step with velocity and noise",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("vx"), py::arg("vy"), py::arg("vz"),
     py::arg("dt"), py::arg("sigma_pos"), py::arg("sigma_cb"),
     py::arg("n_particles"), py::arg("seed"), py::arg("step"));

  m.def("pf_weight", [](py::array_t<double> px, py::array_t<double> py_arr,
                         py::array_t<double> pz, py::array_t<double> pcb,
                         py::array_t<double> sat_ecef, py::array_t<double> pseudoranges,
                         py::array_t<double> weights_sat, py::array_t<double> log_weights,
                         int n_particles, int n_sat, double sigma_pr) {
    gnss_gpu::pf_weight(
        static_cast<double*>(px.request().ptr),
        static_cast<double*>(py_arr.request().ptr),
        static_cast<double*>(pz.request().ptr),
        static_cast<double*>(pcb.request().ptr),
        static_cast<double*>(sat_ecef.request().ptr),
        static_cast<double*>(pseudoranges.request().ptr),
        static_cast<double*>(weights_sat.request().ptr),
        static_cast<double*>(log_weights.request().ptr),
        n_particles, n_sat, sigma_pr);
  }, "Compute pseudorange likelihood weights",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("sat_ecef"), py::arg("pseudoranges"),
     py::arg("weights_sat"), py::arg("log_weights"),
     py::arg("n_particles"), py::arg("n_sat"), py::arg("sigma_pr"));

  m.def("pf_compute_ess", [](py::array_t<double> log_weights, int n_particles) {
    return gnss_gpu::pf_compute_ess(
        static_cast<double*>(log_weights.request().ptr),
        n_particles);
  }, "Compute Effective Sample Size",
     py::arg("log_weights"), py::arg("n_particles"));

  m.def("pf_resample_systematic", [](py::array_t<double> px, py::array_t<double> py_arr,
                                      py::array_t<double> pz, py::array_t<double> pcb,
                                      py::array_t<double> log_weights,
                                      int n_particles, unsigned long long seed) {
    gnss_gpu::pf_resample_systematic(
        static_cast<double*>(px.request().ptr),
        static_cast<double*>(py_arr.request().ptr),
        static_cast<double*>(pz.request().ptr),
        static_cast<double*>(pcb.request().ptr),
        static_cast<double*>(log_weights.request().ptr),
        n_particles, seed);
  }, "Systematic resampling with prefix-sum CDF",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("log_weights"), py::arg("n_particles"), py::arg("seed"));

  m.def("pf_resample_megopolis", [](py::array_t<double> px, py::array_t<double> py_arr,
                                     py::array_t<double> pz, py::array_t<double> pcb,
                                     py::array_t<double> log_weights,
                                     int n_particles, int n_iterations,
                                     unsigned long long seed) {
    gnss_gpu::pf_resample_megopolis(
        static_cast<double*>(px.request().ptr),
        static_cast<double*>(py_arr.request().ptr),
        static_cast<double*>(pz.request().ptr),
        static_cast<double*>(pcb.request().ptr),
        static_cast<double*>(log_weights.request().ptr),
        n_particles, n_iterations, seed);
  }, "Megopolis resampling (Chesser et al.)",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("log_weights"), py::arg("n_particles"),
     py::arg("n_iterations"), py::arg("seed"));

  m.def("pf_estimate", [](py::array_t<double> px, py::array_t<double> py_arr,
                           py::array_t<double> pz, py::array_t<double> pcb,
                           py::array_t<double> log_weights,
                           py::array_t<double> result,
                           int n_particles) {
    gnss_gpu::pf_estimate(
        static_cast<double*>(px.request().ptr),
        static_cast<double*>(py_arr.request().ptr),
        static_cast<double*>(pz.request().ptr),
        static_cast<double*>(pcb.request().ptr),
        static_cast<double*>(log_weights.request().ptr),
        static_cast<double*>(result.request().ptr),
        n_particles);
  }, "Compute weighted mean estimate",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("log_weights"), py::arg("result"), py::arg("n_particles"));

  m.def("pf_get_particles", [](py::array_t<double> px, py::array_t<double> py_arr,
                                py::array_t<double> pz, py::array_t<double> pcb,
                                py::array_t<double> output,
                                int n_particles) {
    gnss_gpu::pf_get_particles(
        static_cast<double*>(px.request().ptr),
        static_cast<double*>(py_arr.request().ptr),
        static_cast<double*>(pz.request().ptr),
        static_cast<double*>(pcb.request().ptr),
        static_cast<double*>(output.request().ptr),
        n_particles);
  }, "Get particle positions for visualization",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("output"), py::arg("n_particles"));
}
