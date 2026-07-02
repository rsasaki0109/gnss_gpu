#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "gnss_gpu/skyplot.h"

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

void validate_finite_scalar(double value, const char* name) {
  if (!std::isfinite(value)) {
    throw std::runtime_error(std::string(name) + " must be finite");
  }
}

void validate_grid_ecef(const py::buffer_info& buf, int n_grid, const char* function_name) {
  validate_positive_count(n_grid, "n_grid");
  if (buf.ndim == 1) {
    if (buf.size != static_cast<py::ssize_t>(n_grid) * 3) {
      throw std::runtime_error(std::string(function_name) +
                               ": grid_ecef must have shape (n_grid, 3) or flat length n_grid*3");
    }
  } else if (buf.ndim == 2 && buf.shape[0] == n_grid && buf.shape[1] == 3) {
    // ok
  } else {
    throw std::runtime_error(std::string(function_name) +
                             ": grid_ecef must have shape (n_grid, 3) or flat length n_grid*3");
  }
  ensure_finite(buf, "grid_ecef must be finite");
}

void validate_sat_ecef(const py::buffer_info& buf, int n_sat, const char* function_name) {
  validate_positive_count(n_sat, "n_sat");
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

void validate_triangles(const py::buffer_info& buf, int n_tri, const char* function_name) {
  validate_positive_count(n_tri, "n_tri");
  if (buf.ndim != 3 || buf.shape[0] != n_tri || buf.shape[1] != 3 || buf.shape[2] != 3) {
    throw std::runtime_error(std::string(function_name) +
                             ": triangles must have shape (n_tri, 3, 3)");
  }
  ensure_finite(buf, "triangles must be finite");
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_skyplot, m) {
  m.doc() = "GPU-accelerated GNSS skyplot / vulnerability map";

  m.def("compute_grid_quality",
    [](DoubleArray grid_ecef, DoubleArray sat_ecef,
       int n_grid, int n_sat, double elevation_mask_rad) {
      auto bg = grid_ecef.request();
      auto bs = sat_ecef.request();

      validate_grid_ecef(bg, n_grid, "compute_grid_quality");
      validate_sat_ecef(bs, n_sat, "compute_grid_quality");
      validate_finite_scalar(elevation_mask_rad, "elevation_mask_rad");

      auto pdop = py::array_t<double>(std::vector<py::ssize_t>{n_grid});
      auto hdop = py::array_t<double>(std::vector<py::ssize_t>{n_grid});
      auto vdop = py::array_t<double>(std::vector<py::ssize_t>{n_grid});
      auto gdop = py::array_t<double>(std::vector<py::ssize_t>{n_grid});
      auto n_visible = py::array_t<int>(std::vector<py::ssize_t>{n_grid});

      gnss_gpu::compute_grid_quality(
          static_cast<double*>(bg.ptr),
          static_cast<double*>(bs.ptr),
          static_cast<double*>(pdop.request().ptr),
          static_cast<double*>(hdop.request().ptr),
          static_cast<double*>(vdop.request().ptr),
          static_cast<double*>(gdop.request().ptr),
          static_cast<int*>(n_visible.request().ptr),
          n_grid, n_sat, elevation_mask_rad);

      return py::make_tuple(pdop, hdop, vdop, gdop, n_visible);
    },
    "Compute DOP grid quality over multiple receiver positions",
    py::arg("grid_ecef"), py::arg("sat_ecef"),
    py::arg("n_grid"), py::arg("n_sat"),
    py::arg("elevation_mask_rad"));

  m.def("compute_sky_visibility",
    [](DoubleArray grid_ecef, DoubleArray triangles,
       int n_grid, int n_tri, int n_az, int n_el) {
      auto bg = grid_ecef.request();
      auto bt = triangles.request();

      validate_grid_ecef(bg, n_grid, "compute_sky_visibility");
      validate_triangles(bt, n_tri, "compute_sky_visibility");
      validate_positive_count(n_az, "n_az");
      validate_positive_count(n_el, "n_el");

      auto sky_mask = py::array_t<float>({n_grid * n_az * n_el});

      gnss_gpu::compute_sky_visibility(
          static_cast<double*>(bg.ptr),
          static_cast<double*>(bt.ptr),
          static_cast<float*>(sky_mask.request().ptr),
          n_grid, n_tri, n_az, n_el);

      return sky_mask.reshape({n_grid, n_az, n_el});
    },
    "Compute sky visibility mask using ray-triangle intersection",
    py::arg("grid_ecef"), py::arg("triangles"),
    py::arg("n_grid"), py::arg("n_tri"),
    py::arg("n_az"), py::arg("n_el"));
}
