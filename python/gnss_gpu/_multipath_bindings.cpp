#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "gnss_gpu/multipath.h"

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

void validate_positive_count(int count, const char* name) {
  if (count <= 0) {
    throw std::runtime_error(std::string(name) + " must be positive");
  }
}

void validate_positive_finite(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error(std::string(name) + " must be positive and finite");
  }
}

void validate_nonnegative_finite(double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::runtime_error(std::string(name) + " must be finite and non-negative");
  }
}

void validate_rx_positions(const py::buffer_info& buf, int n_rx, const char* function_name) {
  if (buf.ndim == 1) {
    if (buf.size != static_cast<py::ssize_t>(n_rx) * 3) {
      throw std::runtime_error(std::string(function_name) +
                               ": rx_ecef must have shape (n_rx, 3) or flat length n_rx*3");
    }
  } else if (buf.ndim == 2 && buf.shape[0] == n_rx && buf.shape[1] == 3) {
    // ok
  } else {
    throw std::runtime_error(std::string(function_name) +
                             ": rx_ecef must have shape (n_rx, 3) or flat length n_rx*3");
  }
  ensure_finite(buf, "rx_ecef must be finite");
}

void validate_sat_positions(const py::buffer_info& buf, int n_sat, const char* function_name) {
  if (buf.ndim == 1) {
    if (buf.size != static_cast<py::ssize_t>(n_sat) * 3) {
      throw std::runtime_error(std::string(function_name) +
                               ": sat_ecef must have shape (n_sat, 3) or flat length n_sat*3");
    }
  } else if (buf.ndim == 2 && buf.shape[0] == n_sat && buf.shape[1] == 3) {
    // ok
  } else {
    throw std::runtime_error(std::string(function_name) +
                             ": sat_ecef must have shape (n_sat, 3) or flat length n_sat*3");
  }
  ensure_finite(buf, "sat_ecef must be finite");
}

void validate_reflector_planes(const py::buffer_info& buf, int n_ref, const char* function_name) {
  if (buf.ndim == 1) {
    if (buf.size != static_cast<py::ssize_t>(n_ref) * 6) {
      throw std::runtime_error(std::string(function_name) +
                               ": reflector_planes must have shape (n_ref, 6) or flat length n_ref*6");
    }
  } else if (buf.ndim == 2 && buf.shape[0] == n_ref && buf.shape[1] == 6) {
    // ok
  } else {
    throw std::runtime_error(std::string(function_name) +
                             ": reflector_planes must have shape (n_ref, 6) or flat length n_ref*6");
  }
  ensure_finite(buf, "reflector_planes must be finite");
}

void validate_clean_pr(const py::buffer_info& buf, int n_epoch, int n_sat, const char* function_name) {
  const py::ssize_t expected = static_cast<py::ssize_t>(n_epoch) * n_sat;
  if (buf.ndim == 1) {
    if (buf.size != expected) {
      throw std::runtime_error(std::string(function_name) +
                               ": clean_pr must have shape (n_epoch, n_sat) or flat length n_epoch*n_sat");
    }
  } else if (buf.ndim == 2 && buf.shape[0] == n_epoch && buf.shape[1] == n_sat) {
    // ok
  } else {
    throw std::runtime_error(std::string(function_name) +
                             ": clean_pr must have shape (n_epoch, n_sat) or flat length n_epoch*n_sat");
  }
  ensure_finite(buf, "clean_pr must be finite");
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_multipath, m) {
  m.doc() = "GPU-accelerated GNSS multipath simulation";

  m.def("simulate_multipath", [](DoubleArray rx_ecef,
                                  DoubleArray sat_ecef,
                                  DoubleArray reflector_planes,
                                  int n_rx, int n_sat, int n_ref,
                                  double carrier_freq_hz, double chip_rate) {
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto bref = reflector_planes.request();

    validate_positive_count(n_rx, "n_rx");
    validate_positive_count(n_sat, "n_sat");
    validate_positive_count(n_ref, "n_ref");
    validate_positive_finite(carrier_freq_hz, "carrier_freq_hz");
    validate_positive_finite(chip_rate, "chip_rate");
    validate_rx_positions(brx, n_rx, "simulate_multipath");
    validate_sat_positions(bsat, n_sat, "simulate_multipath");
    validate_reflector_planes(bref, n_ref, "simulate_multipath");

    auto delays = py::array_t<double>(std::vector<py::ssize_t>{n_rx * n_sat});
    auto attenuations = py::array_t<double>(std::vector<py::ssize_t>{n_rx * n_sat});

    gnss_gpu::simulate_multipath(
        static_cast<double*>(brx.ptr),
        static_cast<double*>(bsat.ptr),
        static_cast<double*>(bref.ptr),
        static_cast<double*>(delays.request().ptr),
        static_cast<double*>(attenuations.request().ptr),
        n_rx, n_sat, n_ref,
        carrier_freq_hz, chip_rate);

    return py::make_tuple(delays, attenuations);
  }, "Simulate multipath excess delays and attenuations",
     py::arg("rx_ecef"), py::arg("sat_ecef"), py::arg("reflector_planes"),
     py::arg("n_rx"), py::arg("n_sat"), py::arg("n_ref"),
     py::arg("carrier_freq_hz"), py::arg("chip_rate"));

  m.def("apply_multipath_error", [](DoubleArray clean_pr,
                                     DoubleArray rx_ecef,
                                     DoubleArray sat_ecef,
                                     DoubleArray reflector_planes,
                                     int n_epoch, int n_sat, int n_ref,
                                     double carrier_freq_hz, double chip_rate,
                                     double correlator_spacing) {
    auto bpr = clean_pr.request();
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto bref = reflector_planes.request();

    validate_positive_count(n_epoch, "n_epoch");
    validate_positive_count(n_sat, "n_sat");
    validate_positive_count(n_ref, "n_ref");
    validate_positive_finite(carrier_freq_hz, "carrier_freq_hz");
    validate_positive_finite(chip_rate, "chip_rate");
    validate_positive_finite(correlator_spacing, "correlator_spacing");
    validate_clean_pr(bpr, n_epoch, n_sat, "apply_multipath_error");
    validate_rx_positions(brx, n_epoch, "apply_multipath_error");
    validate_sat_positions(bsat, n_epoch * n_sat, "apply_multipath_error");
    validate_reflector_planes(bref, n_ref, "apply_multipath_error");

    int total = n_epoch * n_sat;
    auto corrupted = py::array_t<double>(std::vector<py::ssize_t>{total});
    auto errors = py::array_t<double>(std::vector<py::ssize_t>{total});

    gnss_gpu::apply_multipath_error(
        static_cast<double*>(bpr.ptr),
        static_cast<double*>(brx.ptr),
        static_cast<double*>(bsat.ptr),
        static_cast<double*>(bref.ptr),
        static_cast<double*>(corrupted.request().ptr),
        static_cast<double*>(errors.request().ptr),
        n_epoch, n_sat, n_ref,
        carrier_freq_hz, chip_rate, correlator_spacing);

    return py::make_tuple(corrupted, errors);
  }, "Apply multipath DLL tracking error to clean pseudoranges",
     py::arg("clean_pr"), py::arg("rx_ecef"), py::arg("sat_ecef"),
     py::arg("reflector_planes"),
     py::arg("n_epoch"), py::arg("n_sat"), py::arg("n_ref"),
     py::arg("carrier_freq_hz"), py::arg("chip_rate"), py::arg("correlator_spacing"));
}
