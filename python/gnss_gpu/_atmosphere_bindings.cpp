#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "gnss_gpu/atmosphere.h"

namespace py = pybind11;

namespace {

using DoubleArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

void ensure_finite(const py::buffer_info& buffer, const std::string& message) {
  const auto* values = static_cast<const double*>(buffer.ptr);
  for (py::ssize_t i = 0; i < buffer.size; ++i) {
    if (!std::isfinite(values[i])) {
      throw std::runtime_error(message);
    }
  }
}

bool validate_rx_lla(const py::buffer_info& blla,
                     const char* name,
                     int& n_epoch) {
  if (blla.ndim == 1) {
    if (blla.shape[0] != 3) {
      throw std::runtime_error(std::string(name) +
                               ": rx_lla must have shape (3,) or (n_epoch, 3)");
    }
    n_epoch = 1;
    return true;
  }

  if (blla.ndim == 2 && blla.shape[1] == 3) {
    n_epoch = static_cast<int>(blla.shape[0]);
    if (n_epoch < 1) {
      throw std::runtime_error(std::string(name) + ": n_epoch must be >= 1");
    }
    return false;
  }

  throw std::runtime_error(std::string(name) +
                           ": rx_lla must have shape (3,) or (n_epoch, 3)");
}

int validate_sat_angles(const py::buffer_info& buffer,
                        const char* name,
                        const char* label,
                        bool single_epoch,
                        int n_epoch) {
  int n_sat = 0;
  if (single_epoch) {
    if (buffer.ndim == 0) {
      n_sat = 1;
    } else if (buffer.ndim == 1) {
      n_sat = static_cast<int>(buffer.shape[0]);
    } else {
      throw std::runtime_error(std::string(name) + ": " + label +
                               " must have shape (n_sat,)");
    }
  } else {
    if (buffer.ndim != 2 || buffer.shape[0] != n_epoch) {
      throw std::runtime_error(std::string(name) + ": " + label +
                               " must have shape (n_epoch, n_sat)");
    }
    n_sat = static_cast<int>(buffer.shape[1]);
  }

  if (n_sat < 1) {
    throw std::runtime_error(std::string(name) + ": n_sat must be >= 1");
  }
  ensure_finite(buffer, std::string(name) + ": " + label + " must be finite");
  return n_sat;
}

void validate_gps_times(const py::buffer_info& buffer,
                        const char* name,
                        int n_epoch) {
  if (buffer.ndim == 0) {
    if (n_epoch == 1) {
      ensure_finite(buffer, std::string(name) + ": gps_times must be finite");
      return;
    }
  } else if (buffer.ndim == 1 && buffer.shape[0] == n_epoch) {
    ensure_finite(buffer, std::string(name) + ": gps_times must be finite");
    return;
  }

  throw std::runtime_error(std::string(name) +
                           ": gps_times must have shape (n_epoch,)");
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_atmosphere, m) {
  m.doc() = "GPU-accelerated atmospheric delay correction models";

  m.def("tropo_saastamoinen", [](double lat, double alt, double el) {
    if (!std::isfinite(lat) || !std::isfinite(alt) || !std::isfinite(el)) {
      throw std::runtime_error("tropo_saastamoinen: inputs must be finite");
    }
    return gnss_gpu::tropo_saastamoinen(lat, alt, el);
  },
        "Saastamoinen tropospheric delay model\n"
        "Returns delay in meters\n"
        "lat: receiver latitude [rad]\n"
        "alt: receiver altitude [m]\n"
        "el: satellite elevation [rad]",
        py::arg("lat"), py::arg("alt"), py::arg("el"));

  m.def("iono_klobuchar", [](DoubleArray alpha, DoubleArray beta,
                              double lat, double lon, double az, double el,
                              double gps_time) {
    auto ba = alpha.request();
    auto bb = beta.request();
    if (ba.ndim != 1 || ba.size != 4 || bb.ndim != 1 || bb.size != 4) {
      throw std::runtime_error(
          "iono_klobuchar: alpha and beta must have shape (4,)");
    }
    ensure_finite(ba, "iono_klobuchar: alpha must be finite");
    ensure_finite(bb, "iono_klobuchar: beta must be finite");
    if (!std::isfinite(lat) || !std::isfinite(lon) || !std::isfinite(az) ||
        !std::isfinite(el) || !std::isfinite(gps_time)) {
      throw std::runtime_error("iono_klobuchar: scalar inputs must be finite");
    }
    return gnss_gpu::iono_klobuchar(static_cast<double*>(ba.ptr),
                                     static_cast<double*>(bb.ptr),
                                     lat, lon, az, el, gps_time);
  }, "Klobuchar ionospheric delay model (GPS broadcast)\n"
     "Returns delay in meters (L1 frequency)",
     py::arg("alpha"), py::arg("beta"),
     py::arg("lat"), py::arg("lon"), py::arg("az"), py::arg("el"),
     py::arg("gps_time"));

  m.def("tropo_correction_batch", [](DoubleArray rx_lla, DoubleArray sat_el) {
    auto blla = rx_lla.request();
    auto bel = sat_el.request();

    int n_epoch = 0;
    bool single_epoch = validate_rx_lla(blla, "tropo_correction_batch", n_epoch);
    ensure_finite(blla, "tropo_correction_batch: rx_lla must be finite");
    int n_sat = validate_sat_angles(
        bel, "tropo_correction_batch", "sat_el", single_epoch, n_epoch);

    auto corrections = py::array_t<double>({n_epoch, n_sat});
    gnss_gpu::tropo_correction_batch(
        static_cast<double*>(blla.ptr),
        static_cast<double*>(bel.ptr),
        static_cast<double*>(corrections.request().ptr),
        n_epoch, n_sat);

    if (single_epoch) {
      // Return 1D for single epoch input
      corrections.resize({n_sat});
    }
    return corrections;
  }, "Batch tropospheric correction (GPU)",
     py::arg("rx_lla"), py::arg("sat_el"));

  m.def("iono_correction_batch", [](DoubleArray rx_lla,
                                     DoubleArray sat_az,
                                     DoubleArray sat_el,
                                     DoubleArray alpha,
                                     DoubleArray beta,
                                     DoubleArray gps_times) {
    auto blla = rx_lla.request();
    auto baz = sat_az.request();
    auto bel = sat_el.request();
    auto ba = alpha.request();
    auto bb = beta.request();
    auto bt = gps_times.request();

    if (ba.ndim != 1 || ba.size != 4 || bb.ndim != 1 || bb.size != 4) {
      throw std::runtime_error(
          "iono_correction_batch: alpha and beta must have shape (4,)");
    }
    ensure_finite(ba, "iono_correction_batch: alpha must be finite");
    ensure_finite(bb, "iono_correction_batch: beta must be finite");

    int n_epoch = 0;
    bool single_epoch = validate_rx_lla(blla, "iono_correction_batch", n_epoch);
    ensure_finite(blla, "iono_correction_batch: rx_lla must be finite");
    int n_sat = validate_sat_angles(
        baz, "iono_correction_batch", "sat_az", single_epoch, n_epoch);
    int n_sat_el = validate_sat_angles(
        bel, "iono_correction_batch", "sat_el", single_epoch, n_epoch);
    if (n_sat_el != n_sat) {
      throw std::runtime_error(
          "iono_correction_batch: sat_az and sat_el must have matching shape");
    }
    validate_gps_times(bt, "iono_correction_batch", n_epoch);

    auto corrections = py::array_t<double>({n_epoch, n_sat});
    gnss_gpu::iono_correction_batch(
        static_cast<double*>(blla.ptr),
        static_cast<double*>(baz.ptr),
        static_cast<double*>(bel.ptr),
        static_cast<double*>(ba.ptr),
        static_cast<double*>(bb.ptr),
        static_cast<double*>(bt.ptr),
        static_cast<double*>(corrections.request().ptr),
        n_epoch, n_sat);

    if (single_epoch) {
      corrections.resize({n_sat});
    }
    return corrections;
  }, "Batch ionospheric correction (GPU)",
     py::arg("rx_lla"), py::arg("sat_az"), py::arg("sat_el"),
     py::arg("alpha"), py::arg("beta"), py::arg("gps_times"));
}
