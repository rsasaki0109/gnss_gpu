#include "gnss_gpu/diffraction.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

py::tuple diffraction_candidates(
    py::array_t<double> rx,
    py::array_t<double> sats,
    py::array_t<double> starts,
    py::array_t<double> ends,
    py::array_t<double> mids,
    int n_sat,
    int n_edge,
    double max_edge_range,
    double max_ray_edge_dist,
    double max_excess,
    double wavelength) {
  const py::ssize_t total =
      static_cast<py::ssize_t>(n_sat) * static_cast<py::ssize_t>(n_edge);

  auto valid = py::array_t<int>({total});
  auto excess = py::array_t<double>({total});
  auto amplitude = py::array_t<double>({total});
  auto fresnel_v = py::array_t<double>({total});
  auto atten_db = py::array_t<double>({total});
  auto point = py::array_t<double>({total * 3});

  py::buffer_info rx_info = rx.request();
  py::buffer_info sats_info = sats.request();
  py::buffer_info starts_info = starts.request();
  py::buffer_info ends_info = ends.request();
  py::buffer_info mids_info = mids.request();

  py::buffer_info valid_info = valid.request();
  py::buffer_info excess_info = excess.request();
  py::buffer_info amplitude_info = amplitude.request();
  py::buffer_info fresnel_v_info = fresnel_v.request();
  py::buffer_info atten_db_info = atten_db.request();
  py::buffer_info point_info = point.request();

  gnss_gpu::compute_diffraction_candidates(
      static_cast<const double*>(rx_info.ptr),
      static_cast<const double*>(sats_info.ptr),
      static_cast<const double*>(starts_info.ptr),
      static_cast<const double*>(ends_info.ptr),
      static_cast<const double*>(mids_info.ptr),
      n_sat,
      n_edge,
      max_edge_range,
      max_ray_edge_dist,
      max_excess,
      wavelength,
      static_cast<int*>(valid_info.ptr),
      static_cast<double*>(excess_info.ptr),
      static_cast<double*>(amplitude_info.ptr),
      static_cast<double*>(fresnel_v_info.ptr),
      static_cast<double*>(atten_db_info.ptr),
      static_cast<double*>(point_info.ptr));

  return py::make_tuple(valid, excess, amplitude, fresnel_v, atten_db, point);
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_diffraction, m) {
  m.def(
      "diffraction_candidates",
      &diffraction_candidates,
      py::arg("rx"),
      py::arg("sats"),
      py::arg("starts"),
      py::arg("ends"),
      py::arg("mids"),
      py::arg("n_sat"),
      py::arg("n_edge"),
      py::arg("max_edge_range"),
      py::arg("max_ray_edge_dist"),
      py::arg("max_excess"),
      py::arg("wavelength"));
}
