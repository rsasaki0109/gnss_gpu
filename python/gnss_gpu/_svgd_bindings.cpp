#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "gnss_gpu/svgd.h"

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

void ensure_finite_doubles(const py::buffer_info& buf, const char* message) {
  const auto* ptr = static_cast<const double*>(buf.ptr);
  for (py::ssize_t i = 0; i < buf.size; ++i) {
    if (!std::isfinite(ptr[i])) {
      throw std::runtime_error(message);
    }
  }
}

void validate_positive_finite(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error(std::string(name) + " must be positive and finite");
  }
}

void validate_positive_count(int count, const char* name) {
  if (count <= 0) {
    throw std::runtime_error(std::string(name) + " must be positive");
  }
}

void validate_n_particles(int n_particles) {
  validate_positive_count(n_particles, "n_particles");
}

void validate_particle_array(const py::buffer_info& buf, int n_particles, const char* name) {
  if (buf.ndim != 1 || buf.size != n_particles) {
    throw std::runtime_error(std::string(name) + " length must match n_particles");
  }
  ensure_finite_doubles(buf, (std::string(name) + " must be finite").c_str());
}

void validate_particle_arrays(DoubleArray& px, DoubleArray& py_arr, DoubleArray& pz,
                              DoubleArray& pcb, int n_particles) {
  validate_n_particles(n_particles);
  validate_particle_array(px.request(), n_particles, "px");
  validate_particle_array(py_arr.request(), n_particles, "py");
  validate_particle_array(pz.request(), n_particles, "pz");
  validate_particle_array(pcb.request(), n_particles, "pcb");
}

void validate_n_sat(int n_sat) {
  if (n_sat < 1) {
    throw std::runtime_error("n_sat must be >= 1");
  }
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
  ensure_finite_doubles(buf, "sat_ecef must be finite");
}

void validate_pseudoranges(const py::buffer_info& buf, int n_sat) {
  if (buf.ndim != 1 || buf.size != n_sat) {
    throw std::runtime_error("pseudoranges length must match n_sat");
  }
  ensure_finite_doubles(buf, "pseudoranges must be finite");
}

void validate_weights(const py::buffer_info& buf, int n_sat) {
  if (buf.ndim != 1 || buf.size != n_sat) {
    throw std::runtime_error("weights length must match n_sat");
  }
  const auto* ptr = static_cast<const double*>(buf.ptr);
  for (py::ssize_t i = 0; i < buf.size; ++i) {
    if (!std::isfinite(ptr[i])) {
      throw std::runtime_error("weights must be finite");
    }
    if (ptr[i] < 0.0) {
      throw std::runtime_error("weights must be non-negative");
    }
  }
}

void validate_result_buffer(const py::buffer_info& buf) {
  if (buf.ndim != 1 || buf.size != 4) {
    throw std::runtime_error("result must have length 4");
  }
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_svgd, m) {
  m.doc() = "GPU-accelerated SVGD for particle filter (MegaParticles-style)";

  m.def("pf_estimate_bandwidth", [](DoubleArray px, DoubleArray py_arr,
                                     DoubleArray pz, DoubleArray pcb,
                                     int n_particles, int n_subsample,
                                     unsigned long long seed) {
    validate_particle_arrays(px, py_arr, pz, pcb, n_particles);
    validate_positive_count(n_subsample, "n_subsample");
    return gnss_gpu::pf_estimate_bandwidth(
        static_cast<double*>(px.request().ptr),
        static_cast<double*>(py_arr.request().ptr),
        static_cast<double*>(pz.request().ptr),
        static_cast<double*>(pcb.request().ptr),
        n_particles, n_subsample, seed);
  }, "Estimate RBF kernel bandwidth using median heuristic",
     py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
     py::arg("n_particles"), py::arg("n_subsample"), py::arg("seed"));

  m.def("pf_svgd_step", [](DoubleArray px, DoubleArray py_arr,
                             DoubleArray pz, DoubleArray pcb,
                             DoubleArray sat_ecef, DoubleArray pseudoranges,
                             DoubleArray weights_sat,
                             int n_particles, int n_sat,
                             double sigma_pr, double step_size,
                             int n_neighbors, double bandwidth,
                             unsigned long long seed, int step) {
    validate_particle_arrays(px, py_arr, pz, pcb, n_particles);
    validate_n_sat(n_sat);
    validate_sat_ecef(sat_ecef.request(), n_sat);
    validate_pseudoranges(pseudoranges.request(), n_sat);
    validate_weights(weights_sat.request(), n_sat);
    validate_positive_finite(sigma_pr, "sigma_pr");
    validate_positive_finite(step_size, "step_size");
    validate_positive_count(n_neighbors, "n_neighbors");
    validate_positive_finite(bandwidth, "bandwidth");
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

  m.def("pf_svgd_estimate", [](DoubleArray px, DoubleArray py_arr,
                                 DoubleArray pz, DoubleArray pcb,
                                 DoubleArray result,
                                 int n_particles) {
    validate_particle_arrays(px, py_arr, pz, pcb, n_particles);
    validate_result_buffer(result.request());
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
