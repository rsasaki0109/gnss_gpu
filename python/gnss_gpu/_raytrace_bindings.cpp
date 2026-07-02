#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "gnss_gpu/raytrace.h"

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

void validate_triangles(const py::buffer_info& buf, int& n_tri, const char* function_name) {
  if (buf.ndim != 3 || buf.shape[1] != 3 || buf.shape[2] != 3) {
    throw std::runtime_error(std::string(function_name) +
                             ": triangles must have shape (n_tri, 3, 3)");
  }
  n_tri = static_cast<int>(buf.shape[0]);
  if (n_tri < 0) {
    throw std::runtime_error(std::string(function_name) + ": n_tri must be >= 0");
  }
  ensure_finite(buf, "triangles must be finite");
}

}  // namespace

PYBIND11_MODULE(_raytrace, m) {
  m.doc() = "GPU-accelerated ray tracing for GNSS NLOS detection";

  m.def("raytrace_los_check", [](DoubleArray rx_ecef,
                                  DoubleArray sat_ecef,
                                  DoubleArray triangles) {
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto btri = triangles.request();

    validate_rx_ecef(brx, "raytrace_los_check");
    int n_sat = 0;
    validate_sat_ecef(bsat, n_sat, "raytrace_los_check");
    int n_tri = 0;
    validate_triangles(btri, n_tri, "raytrace_los_check");

    auto is_los_int = py::array_t<int>(std::vector<py::ssize_t>{n_sat});
    int* int_ptr = is_los_int.mutable_data();

    gnss_gpu::raytrace_los_check(
        static_cast<double*>(brx.ptr),
        static_cast<double*>(bsat.ptr),
        reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr),
        int_ptr,
        n_sat, n_tri);

    auto is_los = py::array_t<bool>(std::vector<py::ssize_t>{n_sat});
    bool* bool_ptr = is_los.mutable_data();
    for (int i = 0; i < n_sat; i++) bool_ptr[i] = (int_ptr[i] != 0);
    return is_los;
  }, "Batch LOS check using ray tracing against building triangles",
     py::arg("rx_ecef"), py::arg("sat_ecef"), py::arg("triangles"));

  m.def("raytrace_multipath", [](DoubleArray rx_ecef,
                                  DoubleArray sat_ecef,
                                  DoubleArray triangles) {
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto btri = triangles.request();

    validate_rx_ecef(brx, "raytrace_multipath");
    int n_sat = 0;
    validate_sat_ecef(bsat, n_sat, "raytrace_multipath");
    int n_tri = 0;
    validate_triangles(btri, n_tri, "raytrace_multipath");

    auto reflection_points = py::array_t<double>({n_sat, 3});
    auto excess_delays = py::array_t<double>(std::vector<py::ssize_t>{n_sat});

    gnss_gpu::raytrace_multipath(
        static_cast<double*>(brx.ptr),
        static_cast<double*>(bsat.ptr),
        reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr),
        static_cast<double*>(reflection_points.request().ptr),
        static_cast<double*>(excess_delays.request().ptr),
        n_sat, n_tri);

    return py::make_tuple(reflection_points, excess_delays);
  }, "Compute first-order multipath reflections",
     py::arg("rx_ecef"), py::arg("sat_ecef"), py::arg("triangles"));
}
