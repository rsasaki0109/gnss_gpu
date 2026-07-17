#include "gnss_gpu/diffraction.h"
#include "gnss_gpu/pybind_validation.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

using gnss_gpu::pybind_validation::DoubleArray;
namespace pv = gnss_gpu::pybind_validation;

void validate_rx_ecef(const py::buffer_info& buf) {
  if (buf.ndim != 1 || buf.size != 3) {
    throw std::runtime_error("rx must have shape (3,)");
  }
  pv::ensure_finite_doubles(buf, "rx must be finite");
}

void validate_flat_xyz(const py::buffer_info& buf, int count, const char* name) {
  const py::ssize_t expected = static_cast<py::ssize_t>(count) * 3;
  if (buf.ndim != 1 || buf.size != expected) {
    throw std::runtime_error(
        std::string(name) + " must have flat length n*3 matching n_sat/n_edge");
  }
  const std::string finite_msg = std::string(name) + " must be finite";
  pv::ensure_finite_doubles(buf, finite_msg.c_str());
}

void validate_diffraction_options(
    int n_sat,
    int n_edge,
    double max_edge_range,
    double max_ray_edge_dist,
    double max_excess,
    double wavelength) {
  if (n_sat < 0) {
    throw std::runtime_error("n_sat must be >= 0");
  }
  if (n_edge < 0) {
    throw std::runtime_error("n_edge must be >= 0");
  }
  pv::validate_positive_finite(max_edge_range, "max_edge_range");
  pv::validate_positive_finite(max_ray_edge_dist, "max_ray_edge_dist");
  pv::validate_positive_finite(max_excess, "max_excess");
  pv::validate_positive_finite(wavelength, "wavelength");
}

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
  validate_diffraction_options(
      n_sat, n_edge, max_edge_range, max_ray_edge_dist, max_excess, wavelength);

  py::buffer_info rx_info = rx.request();
  py::buffer_info sats_info = sats.request();
  py::buffer_info starts_info = starts.request();
  py::buffer_info ends_info = ends.request();
  py::buffer_info mids_info = mids.request();

  validate_rx_ecef(rx_info);
  validate_flat_xyz(sats_info, n_sat, "sats");
  validate_flat_xyz(starts_info, n_edge, "starts");
  validate_flat_xyz(ends_info, n_edge, "ends");
  validate_flat_xyz(mids_info, n_edge, "mids");

  const py::ssize_t total =
      static_cast<py::ssize_t>(n_sat) * static_cast<py::ssize_t>(n_edge);

  auto valid = py::array_t<int>({total});
  auto excess = py::array_t<double>({total});
  auto amplitude = py::array_t<double>({total});
  auto fresnel_v = py::array_t<double>({total});
  auto atten_db = py::array_t<double>({total});
  auto point = py::array_t<double>({total * 3});

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

void validate_utd_options(
    int n_sat,
    int n_edge,
    int n_wedge_n,
    double max_edge_range,
    double max_ray_edge_dist,
    double max_excess,
    double wavelength,
    int mode) {
  validate_diffraction_options(
      n_sat, n_edge, max_edge_range, max_ray_edge_dist, max_excess, wavelength);
  if (n_wedge_n < 0) {
    throw std::runtime_error("n_wedge_n must be >= 0");
  }
  // Any non-negative length is accepted here, matching the flexibility of
  // utd_diffraction._wedge_n_at on the host: 0 -> default 2.0 for every
  // edge, 1 -> broadcast scalar, and any other length is looked up
  // per-edge with out-of-range entries falling back to 2.0.
  if (mode < 0 || mode > 2) {
    throw std::runtime_error("mode must be 0 (absorbing), 1 (soft), or 2 (hard)");
  }
}

py::tuple utd_diffraction_candidates(
    py::array_t<double> rx,
    py::array_t<double> sats,
    py::array_t<double> starts,
    py::array_t<double> ends,
    py::array_t<double> mids,
    py::array_t<double> face_dir_a,
    py::array_t<double> face_dir_b,
    py::array_t<double> wedge_n,
    int n_sat,
    int n_edge,
    double max_edge_range,
    double max_ray_edge_dist,
    double max_excess,
    double wavelength,
    int mode) {
  py::buffer_info wedge_n_info = wedge_n.request();
  const int n_wedge_n = static_cast<int>(wedge_n_info.size);

  validate_utd_options(
      n_sat, n_edge, n_wedge_n, max_edge_range, max_ray_edge_dist, max_excess,
      wavelength, mode);

  py::buffer_info rx_info = rx.request();
  py::buffer_info sats_info = sats.request();
  py::buffer_info starts_info = starts.request();
  py::buffer_info ends_info = ends.request();
  py::buffer_info mids_info = mids.request();
  py::buffer_info face_a_info = face_dir_a.request();
  py::buffer_info face_b_info = face_dir_b.request();

  validate_rx_ecef(rx_info);
  validate_flat_xyz(sats_info, n_sat, "sats");
  validate_flat_xyz(starts_info, n_edge, "starts");
  validate_flat_xyz(ends_info, n_edge, "ends");
  validate_flat_xyz(mids_info, n_edge, "mids");
  validate_flat_xyz(face_a_info, n_edge, "face_dir_a");
  validate_flat_xyz(face_b_info, n_edge, "face_dir_b");
  if (n_wedge_n > 0) {
    pv::ensure_finite_doubles(wedge_n_info, "wedge_n must be finite");
  }

  const py::ssize_t total =
      static_cast<py::ssize_t>(n_sat) * static_cast<py::ssize_t>(n_edge);

  auto valid = py::array_t<int>({total});
  auto excess = py::array_t<double>({total});
  auto amplitude = py::array_t<double>({total});
  auto beta0 = py::array_t<double>({total});
  auto phi = py::array_t<double>({total});
  auto phi_p = py::array_t<double>({total});
  auto wedge_n_out = py::array_t<double>({total});
  auto atten_db = py::array_t<double>({total});
  auto point = py::array_t<double>({total * 3});

  py::buffer_info valid_info = valid.request();
  py::buffer_info excess_info = excess.request();
  py::buffer_info amplitude_info = amplitude.request();
  py::buffer_info beta0_info = beta0.request();
  py::buffer_info phi_info = phi.request();
  py::buffer_info phi_p_info = phi_p.request();
  py::buffer_info wedge_n_out_info = wedge_n_out.request();
  py::buffer_info atten_db_info = atten_db.request();
  py::buffer_info point_info = point.request();

  gnss_gpu::compute_utd_diffraction_candidates(
      static_cast<const double*>(rx_info.ptr),
      static_cast<const double*>(sats_info.ptr),
      static_cast<const double*>(starts_info.ptr),
      static_cast<const double*>(ends_info.ptr),
      static_cast<const double*>(mids_info.ptr),
      static_cast<const double*>(face_a_info.ptr),
      static_cast<const double*>(face_b_info.ptr),
      static_cast<const double*>(wedge_n_info.ptr),
      n_wedge_n,
      n_sat,
      n_edge,
      max_edge_range,
      max_ray_edge_dist,
      max_excess,
      wavelength,
      mode,
      static_cast<int*>(valid_info.ptr),
      static_cast<double*>(excess_info.ptr),
      static_cast<double*>(amplitude_info.ptr),
      static_cast<double*>(beta0_info.ptr),
      static_cast<double*>(phi_info.ptr),
      static_cast<double*>(phi_p_info.ptr),
      static_cast<double*>(wedge_n_out_info.ptr),
      static_cast<double*>(atten_db_info.ptr),
      static_cast<double*>(point_info.ptr));

  return py::make_tuple(
      valid, excess, amplitude, beta0, phi, phi_p, wedge_n_out, atten_db, point);
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

  m.def(
      "utd_diffraction_candidates",
      &utd_diffraction_candidates,
      py::arg("rx"),
      py::arg("sats"),
      py::arg("starts"),
      py::arg("ends"),
      py::arg("mids"),
      py::arg("face_dir_a"),
      py::arg("face_dir_b"),
      py::arg("wedge_n"),
      py::arg("n_sat"),
      py::arg("n_edge"),
      py::arg("max_edge_range"),
      py::arg("max_ray_edge_dist"),
      py::arg("max_excess"),
      py::arg("wavelength"),
      py::arg("mode"));
}
