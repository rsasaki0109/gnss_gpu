#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "gnss_gpu/particle_filter.h"
#include "gnss_gpu/pybind_validation.h"

namespace py = pybind11;

namespace {

using gnss_gpu::pybind_validation::DoubleArray;
namespace pv = gnss_gpu::pybind_validation;

void validate_particle_arrays(const py::buffer_info& px, const py::buffer_info& py_arr,
                              const py::buffer_info& pz, const py::buffer_info& pcb,
                              int n_particles) {
    if (px.ndim != 1 || px.size != n_particles) {
        throw std::runtime_error("px length must match n_particles");
    }
    if (py_arr.ndim != 1 || py_arr.size != n_particles) {
        throw std::runtime_error("py length must match n_particles");
    }
    if (pz.ndim != 1 || pz.size != n_particles) {
        throw std::runtime_error("pz length must match n_particles");
    }
    if (pcb.ndim != 1 || pcb.size != n_particles) {
        throw std::runtime_error("pcb length must match n_particles");
    }
}

void validate_log_weights(const py::buffer_info& buf, int n_particles) {
    if (buf.ndim != 1 || buf.size != n_particles) {
        throw std::runtime_error("log_weights length must match n_particles");
    }
    pv::ensure_finite_doubles(buf, "log_weights must be finite");
}

void validate_velocity_array(const py::buffer_info& buf, const char* name) {
    if (buf.ndim != 1 || buf.size < 1) {
        throw std::runtime_error(std::string(name) + " must contain at least 1 value");
    }
    pv::ensure_finite_doubles(buf, (std::string(name) + " must be finite").c_str());
}

void validate_sat_ecef(const py::buffer_info& buf, int n_sat) {
    if (buf.ndim == 2) {
        if (buf.shape[0] != n_sat || buf.shape[1] != 3) {
            throw std::runtime_error("sat_ecef shape must match n_sat satellites");
        }
    } else if (buf.ndim == 1) {
        if (buf.size != static_cast<py::ssize_t>(n_sat) * 3) {
            throw std::runtime_error("sat_ecef shape must match n_sat satellites");
        }
    } else {
        throw std::runtime_error("sat_ecef must be 1D or 2D");
    }
    pv::ensure_finite_doubles(buf, "sat_ecef must be finite");
}

void validate_pseudoranges(const py::buffer_info& buf, int n_sat) {
    if (buf.ndim != 1 || buf.size != n_sat) {
        throw std::runtime_error("pseudoranges length must match n_sat");
    }
    pv::ensure_finite_doubles(buf, "pseudoranges must be finite");
}

void validate_weights_sat(const py::buffer_info& buf, int n_sat) {
    if (buf.ndim != 1 || buf.size != n_sat) {
        throw std::runtime_error("weights_sat length must match n_sat");
    }
    const auto* ptr = static_cast<const double*>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        if (!std::isfinite(ptr[i])) {
            throw std::runtime_error("weights_sat must be finite");
        }
        if (ptr[i] < 0.0) {
            throw std::runtime_error("weights_sat must be non-negative");
        }
    }
}

void validate_estimate_result(const py::buffer_info& buf) {
    if (buf.ndim != 1 || buf.size != 4) {
        throw std::runtime_error("result must have shape (4,)");
    }
}

void validate_particles_output(const py::buffer_info& buf, int n_particles) {
    if (buf.ndim != 1 || buf.size != static_cast<py::ssize_t>(n_particles) * 4) {
        throw std::runtime_error("output length must match n_particles * 4");
    }
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_pf, m) {
  m.doc() = "GPU-accelerated Mega Particle Filter for GNSS positioning";

  m.def("pf_initialize", [](DoubleArray px, DoubleArray py_arr,
                             DoubleArray pz, DoubleArray pcb,
                             double init_x, double init_y, double init_z, double init_cb,
                             double spread_pos, double spread_cb,
                             int n_particles, unsigned long long seed) {
    pv::validate_n_particles(n_particles);
    auto bpx = px.request();
    auto bpy = py_arr.request();
    auto bpz = pz.request();
    auto bpcb = pcb.request();
    validate_particle_arrays(bpx, bpy, bpz, bpcb, n_particles);
    pv::validate_finite(init_x, "init_x");
    pv::validate_finite(init_y, "init_y");
    pv::validate_finite(init_z, "init_z");
    pv::validate_finite(init_cb, "init_cb");
    pv::validate_positive_finite(spread_pos, "spread_pos");
    pv::validate_positive_finite(spread_cb, "spread_cb");
    gnss_gpu::pf_initialize(
        static_cast<double*>(bpx.ptr),
        static_cast<double*>(bpy.ptr),
        static_cast<double*>(bpz.ptr),
        static_cast<double*>(bpcb.ptr),
        init_x, init_y, init_z, init_cb,
        spread_pos, spread_cb,
        n_particles, seed);
  }, "Initialize particles around initial position",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("init_x"), py::arg("init_y"), py::arg("init_z"), py::arg("init_cb"),
     py::arg("spread_pos"), py::arg("spread_cb"),
     py::arg("n_particles"), py::arg("seed"));

  m.def("pf_predict", [](DoubleArray px, DoubleArray py_arr,
                          DoubleArray pz, DoubleArray pcb,
                          DoubleArray vx, DoubleArray vy,
                          DoubleArray vz,
                          double dt, double sigma_pos, double sigma_cb,
                          int n_particles, unsigned long long seed, int step) {
    pv::validate_n_particles(n_particles);
    auto bpx = px.request();
    auto bpy = py_arr.request();
    auto bpz = pz.request();
    auto bpcb = pcb.request();
    validate_particle_arrays(bpx, bpy, bpz, bpcb, n_particles);
    validate_velocity_array(vx.request(), "vx");
    validate_velocity_array(vy.request(), "vy");
    validate_velocity_array(vz.request(), "vz");
    pv::validate_dt_nonnegative(dt);
    pv::validate_positive_finite(sigma_pos, "sigma_pos");
    pv::validate_positive_finite(sigma_cb, "sigma_cb");
    gnss_gpu::pf_predict(
        static_cast<double*>(bpx.ptr),
        static_cast<double*>(bpy.ptr),
        static_cast<double*>(bpz.ptr),
        static_cast<double*>(bpcb.ptr),
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

  m.def("pf_weight", [](DoubleArray px, DoubleArray py_arr,
                         DoubleArray pz, DoubleArray pcb,
                         DoubleArray sat_ecef, DoubleArray pseudoranges,
                         DoubleArray weights_sat, DoubleArray log_weights,
                         int n_particles, int n_sat, double sigma_pr) {
    pv::validate_n_particles(n_particles);
    pv::validate_n_sat(n_sat);
    auto bpx = px.request();
    auto bpy = py_arr.request();
    auto bpz = pz.request();
    auto bpcb = pcb.request();
    validate_particle_arrays(bpx, bpy, bpz, bpcb, n_particles);
    validate_sat_ecef(sat_ecef.request(), n_sat);
    validate_pseudoranges(pseudoranges.request(), n_sat);
    validate_weights_sat(weights_sat.request(), n_sat);
    validate_log_weights(log_weights.request(), n_particles);
    pv::validate_positive_finite(sigma_pr, "sigma_pr");
    gnss_gpu::pf_weight(
        static_cast<double*>(bpx.ptr),
        static_cast<double*>(bpy.ptr),
        static_cast<double*>(bpz.ptr),
        static_cast<double*>(bpcb.ptr),
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

  m.def("pf_compute_ess", [](DoubleArray log_weights, int n_particles) {
    pv::validate_n_particles(n_particles);
    validate_log_weights(log_weights.request(), n_particles);
    return gnss_gpu::pf_compute_ess(
        static_cast<double*>(log_weights.request().ptr),
        n_particles);
  }, "Compute Effective Sample Size",
     py::arg("log_weights"), py::arg("n_particles"));

  m.def("pf_resample_systematic", [](DoubleArray px, DoubleArray py_arr,
                                      DoubleArray pz, DoubleArray pcb,
                                      DoubleArray log_weights,
                                      int n_particles, unsigned long long seed) {
    pv::validate_n_particles(n_particles);
    auto bpx = px.request();
    auto bpy = py_arr.request();
    auto bpz = pz.request();
    auto bpcb = pcb.request();
    validate_particle_arrays(bpx, bpy, bpz, bpcb, n_particles);
    validate_log_weights(log_weights.request(), n_particles);
    gnss_gpu::pf_resample_systematic(
        static_cast<double*>(bpx.ptr),
        static_cast<double*>(bpy.ptr),
        static_cast<double*>(bpz.ptr),
        static_cast<double*>(bpcb.ptr),
        static_cast<double*>(log_weights.request().ptr),
        n_particles, seed);
  }, "Systematic resampling with prefix-sum CDF",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("log_weights"), py::arg("n_particles"), py::arg("seed"));

  m.def("pf_resample_megopolis", [](DoubleArray px, DoubleArray py_arr,
                                     DoubleArray pz, DoubleArray pcb,
                                     DoubleArray log_weights,
                                     int n_particles, int n_iterations,
                                     unsigned long long seed) {
    pv::validate_n_particles(n_particles);
    auto bpx = px.request();
    auto bpy = py_arr.request();
    auto bpz = pz.request();
    auto bpcb = pcb.request();
    validate_particle_arrays(bpx, bpy, bpz, bpcb, n_particles);
    validate_log_weights(log_weights.request(), n_particles);
    gnss_gpu::pf_resample_megopolis(
        static_cast<double*>(bpx.ptr),
        static_cast<double*>(bpy.ptr),
        static_cast<double*>(bpz.ptr),
        static_cast<double*>(bpcb.ptr),
        static_cast<double*>(log_weights.request().ptr),
        n_particles, n_iterations, seed);
  }, "Megopolis resampling (Chesser et al.)",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("log_weights"), py::arg("n_particles"),
     py::arg("n_iterations"), py::arg("seed"));

  m.def("pf_estimate", [](DoubleArray px, DoubleArray py_arr,
                           DoubleArray pz, DoubleArray pcb,
                           DoubleArray log_weights,
                           DoubleArray result,
                           int n_particles) {
    pv::validate_n_particles(n_particles);
    auto bpx = px.request();
    auto bpy = py_arr.request();
    auto bpz = pz.request();
    auto bpcb = pcb.request();
    validate_particle_arrays(bpx, bpy, bpz, bpcb, n_particles);
    validate_log_weights(log_weights.request(), n_particles);
    validate_estimate_result(result.request());
    gnss_gpu::pf_estimate(
        static_cast<double*>(bpx.ptr),
        static_cast<double*>(bpy.ptr),
        static_cast<double*>(bpz.ptr),
        static_cast<double*>(bpcb.ptr),
        static_cast<double*>(log_weights.request().ptr),
        static_cast<double*>(result.request().ptr),
        n_particles);
  }, "Compute weighted mean estimate",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("log_weights"), py::arg("result"), py::arg("n_particles"));

  m.def("pf_get_particles", [](DoubleArray px, DoubleArray py_arr,
                                DoubleArray pz, DoubleArray pcb,
                                DoubleArray output,
                                int n_particles) {
    pv::validate_n_particles(n_particles);
    auto bpx = px.request();
    auto bpy = py_arr.request();
    auto bpz = pz.request();
    auto bpcb = pcb.request();
    validate_particle_arrays(bpx, bpy, bpz, bpcb, n_particles);
    validate_particles_output(output.request(), n_particles);
    gnss_gpu::pf_get_particles(
        static_cast<double*>(bpx.ptr),
        static_cast<double*>(bpy.ptr),
        static_cast<double*>(bpz.ptr),
        static_cast<double*>(bpcb.ptr),
        static_cast<double*>(output.request().ptr),
        n_particles);
  }, "Get particle positions for visualization",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("output"), py::arg("n_particles"));
}
