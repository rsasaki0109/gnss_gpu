#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/skyplot.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_skyplot, m) {
  m.doc() = "GPU-accelerated GNSS skyplot / vulnerability map";

  m.def("compute_grid_quality",
    [](py::array_t<double> grid_ecef, py::array_t<double> sat_ecef,
       int n_grid, int n_sat, double elevation_mask_rad) {
      auto bg = grid_ecef.request();
      auto bs = sat_ecef.request();
      // sat_ecef: accept (N,3) or (N*3,) flat

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
    [](py::array_t<double> grid_ecef, py::array_t<double> triangles,
       int n_grid, int n_tri, int n_az, int n_el) {
      auto bg = grid_ecef.request();
      auto bt = triangles.request();

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
