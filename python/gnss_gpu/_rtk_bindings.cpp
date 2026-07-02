#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "gnss_gpu/rtk.h"

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

void validate_positive_int(int value, const char* name) {
  if (value <= 0) {
    throw std::runtime_error(std::string(name) + " must be positive");
  }
}

void validate_base_ecef(const py::buffer_info& buf) {
  if (buf.ndim != 1 || buf.size != 3) {
    throw std::runtime_error("base_ecef must have shape (3,)");
  }
  ensure_finite_doubles(buf, "base_ecef must be finite");
}

void validate_n_sat(int n_sat) {
  if (n_sat < 2) {
    throw std::runtime_error("n_sat must be >= 2");
  }
}

void validate_obs_1d(const py::buffer_info& buf, int n_sat, const char* name) {
  if (buf.ndim != 1 || buf.size != n_sat) {
    throw std::runtime_error(std::string(name) + " length must match n_sat");
  }
  ensure_finite_doubles(buf, (std::string(name) + " must be finite").c_str());
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

void validate_rtk_params(double wavelength, int max_iter, double tol) {
  validate_positive_finite(wavelength, "wavelength");
  validate_positive_int(max_iter, "max_iter");
  validate_positive_finite(tol, "tol");
}

void validate_batch_obs_2d(const py::buffer_info& buf, int n_epoch, int n_sat,
                           const char* name) {
  if (buf.ndim != 2 || buf.shape[0] != n_epoch || buf.shape[1] != n_sat) {
    throw std::runtime_error(std::string(name) + " must have shape (n_epoch, n_sat)");
  }
  ensure_finite_doubles(buf, (std::string(name) + " must be finite").c_str());
}

void validate_batch_sat_ecef(const py::buffer_info& buf, int n_epoch, int n_sat) {
  const py::ssize_t expected = static_cast<py::ssize_t>(n_epoch) * n_sat * 3;
  if (buf.ndim == 3 && buf.shape[0] == n_epoch && buf.shape[1] == n_sat &&
      buf.shape[2] == 3) {
    // ok
  } else if (buf.ndim == 1 && buf.size == expected) {
    // ok
  } else if (buf.ndim == 2 && buf.shape[0] == n_epoch &&
             buf.shape[1] == n_sat * 3) {
    // ok
  } else {
    throw std::runtime_error(
        "sat_ecef must have shape (n_epoch, n_sat, 3) or compatible");
  }
  ensure_finite_doubles(buf, "sat_ecef must be finite");
}

int validate_float_amb(const py::buffer_info& buf) {
  if (buf.ndim != 1 || buf.size < 1) {
    throw std::runtime_error("float_amb must be a 1D array");
  }
  ensure_finite_doubles(buf, "float_amb must be finite");
  return static_cast<int>(buf.size);
}

void validate_Q_amb(const py::buffer_info& buf, int n) {
  const bool flat_ok = (buf.ndim == 1 && buf.size == static_cast<py::ssize_t>(n) * n);
  const bool matrix_ok =
      (buf.ndim == 2 && buf.shape[0] == n && buf.shape[1] == n);
  if (!flat_ok && !matrix_ok) {
    throw std::runtime_error("Q_amb must have shape (n, n) or n*n flat elements");
  }
  ensure_finite_doubles(buf, "Q_amb must be finite");
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_rtk, m) {
  m.doc() = "RTK carrier phase positioning";

  m.def("rtk_float", [](DoubleArray base_ecef,
                         DoubleArray rover_pr, DoubleArray base_pr,
                         DoubleArray rover_carrier, DoubleArray base_carrier,
                         DoubleArray sat_ecef,
                         double wavelength, int max_iter, double tol) {
    auto bb = base_ecef.request();
    auto brpr = rover_pr.request();
    auto bbpr = base_pr.request();
    auto brcp = rover_carrier.request();
    auto bbcp = base_carrier.request();
    auto bsat = sat_ecef.request();

    validate_base_ecef(bb);
    validate_rtk_params(wavelength, max_iter, tol);

    if (brpr.ndim != 1) {
      throw std::runtime_error("rover_pr must be a 1D array");
    }
    const int n_sat = static_cast<int>(brpr.size);
    validate_n_sat(n_sat);
    validate_obs_1d(brpr, n_sat, "rover_pr");
    validate_obs_1d(bbpr, n_sat, "base_pr");
    validate_obs_1d(brcp, n_sat, "rover_carrier");
    validate_obs_1d(bbcp, n_sat, "base_carrier");
    validate_sat_ecef(bsat, n_sat);

    int n_dd = n_sat - 1;

    auto result = py::array_t<double>({3}, {sizeof(double)});
    auto ambiguities = py::array_t<double>(std::vector<py::ssize_t>{n_dd});
    auto residuals = py::array_t<double>(std::vector<py::ssize_t>{2 * n_dd});

    int iters = gnss_gpu::rtk_float(
        static_cast<double*>(bb.ptr),
        static_cast<double*>(brpr.ptr), static_cast<double*>(bbpr.ptr),
        static_cast<double*>(brcp.ptr), static_cast<double*>(bbcp.ptr),
        static_cast<double*>(bsat.ptr),
        static_cast<double*>(result.request().ptr),
        static_cast<double*>(ambiguities.request().ptr),
        static_cast<double*>(residuals.request().ptr),
        n_sat, wavelength, max_iter, tol);

    return py::make_tuple(result, ambiguities, residuals, iters);
  }, "Single-epoch RTK float solution",
     py::arg("base_ecef"), py::arg("rover_pr"), py::arg("base_pr"),
     py::arg("rover_carrier"), py::arg("base_carrier"), py::arg("sat_ecef"),
     py::arg("wavelength") = 0.19029, py::arg("max_iter") = 20, py::arg("tol") = 1e-4);

  m.def("rtk_float_batch", [](DoubleArray base_ecef,
                                DoubleArray rover_pr, DoubleArray base_pr,
                                DoubleArray rover_carrier, DoubleArray base_carrier,
                                DoubleArray sat_ecef,
                                double wavelength, int max_iter, double tol) {
    auto bb = base_ecef.request();
    auto brpr = rover_pr.request();
    auto bbpr = base_pr.request();
    auto brcp = rover_carrier.request();
    auto bbcp = base_carrier.request();
    auto bsat = sat_ecef.request();

    validate_base_ecef(bb);
    validate_rtk_params(wavelength, max_iter, tol);

    if (brpr.ndim != 2) {
      throw std::runtime_error("rover_pr must be 2D array (n_epoch, n_sat)");
    }
    const int n_epoch = static_cast<int>(brpr.shape[0]);
    const int n_sat = static_cast<int>(brpr.shape[1]);
    if (n_epoch < 1) {
      throw std::runtime_error("n_epoch must be >= 1");
    }
    validate_n_sat(n_sat);
    validate_batch_obs_2d(brpr, n_epoch, n_sat, "rover_pr");
    validate_batch_obs_2d(bbpr, n_epoch, n_sat, "base_pr");
    validate_batch_obs_2d(brcp, n_epoch, n_sat, "rover_carrier");
    validate_batch_obs_2d(bbcp, n_epoch, n_sat, "base_carrier");
    validate_batch_sat_ecef(bsat, n_epoch, n_sat);

    int n_dd = n_sat - 1;

    auto results = py::array_t<double>({n_epoch, 3});
    auto ambiguities = py::array_t<double>({n_epoch, n_dd});
    auto iters = py::array_t<int>(std::vector<py::ssize_t>{n_epoch});

    gnss_gpu::rtk_float_batch(
        static_cast<double*>(bb.ptr),
        static_cast<double*>(brpr.ptr),
        static_cast<double*>(bbpr.ptr),
        static_cast<double*>(brcp.ptr),
        static_cast<double*>(bbcp.ptr),
        static_cast<double*>(bsat.ptr),
        static_cast<double*>(results.request().ptr),
        static_cast<double*>(ambiguities.request().ptr),
        static_cast<int*>(iters.request().ptr),
        n_epoch, n_sat, wavelength, max_iter, tol);

    return py::make_tuple(results, ambiguities, iters);
  }, "Batch RTK float solution (GPU parallel)",
     py::arg("base_ecef"), py::arg("rover_pr"), py::arg("base_pr"),
     py::arg("rover_carrier"), py::arg("base_carrier"), py::arg("sat_ecef"),
     py::arg("wavelength") = 0.19029, py::arg("max_iter") = 20, py::arg("tol") = 1e-4);

  m.def("lambda_integer", [](DoubleArray float_amb, DoubleArray Q_amb,
                               int n_candidates) {
    auto ba = float_amb.request();
    auto bq = Q_amb.request();

    validate_positive_int(n_candidates, "n_candidates");
    const int n = validate_float_amb(ba);
    validate_Q_amb(bq, n);

    auto fixed_amb = py::array_t<int>(std::vector<py::ssize_t>{n});

    double ratio = gnss_gpu::lambda_integer(
        static_cast<double*>(ba.ptr),
        static_cast<double*>(bq.ptr),
        static_cast<int*>(fixed_amb.request().ptr),
        n, n_candidates);

    return py::make_tuple(fixed_amb, ratio);
  }, "LAMBDA integer ambiguity resolution",
     py::arg("float_amb"), py::arg("Q_amb"), py::arg("n_candidates") = 100);
}
