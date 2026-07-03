#pragma once

#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <string>

namespace gnss_gpu {
namespace pybind_validation {

namespace py = pybind11;

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

inline void ensure_finite_doubles(const py::buffer_info& buf, const char* message) {
  const auto* ptr = static_cast<const double*>(buf.ptr);
  for (py::ssize_t i = 0; i < buf.size; ++i) {
    if (!std::isfinite(ptr[i])) {
      throw std::runtime_error(message);
    }
  }
}

inline void validate_finite(double value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::runtime_error(std::string(name) + " must be finite");
  }
}

inline void validate_positive_finite(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error(std::string(name) + " must be positive and finite");
  }
}

inline void validate_n_sat(int n_sat) {
  if (n_sat < 1) {
    throw std::runtime_error("n_sat must be >= 1");
  }
}

inline void validate_n_particles(int n_particles) {
  if (n_particles < 1) {
    throw std::runtime_error("n_particles must be >= 1");
  }
}

inline void validate_dt_nonnegative(double dt) {
  if (!std::isfinite(dt) || dt < 0.0) {
    throw std::runtime_error("dt must be non-negative and finite");
  }
}

inline void validate_dt_positive(double dt) {
  if (!std::isfinite(dt) || dt <= 0.0) {
    throw std::runtime_error("dt must be positive and finite");
  }
}

}  // namespace pybind_validation
}  // namespace gnss_gpu
