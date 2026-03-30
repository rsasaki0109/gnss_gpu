#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/atmosphere.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_atmosphere, m) {
  m.doc() = "GPU-accelerated atmospheric delay correction models";

  m.def("tropo_saastamoinen", &gnss_gpu::tropo_saastamoinen,
        "Saastamoinen tropospheric delay model\n"
        "Returns delay in meters\n"
        "lat: receiver latitude [rad]\n"
        "alt: receiver altitude [m]\n"
        "el: satellite elevation [rad]",
        py::arg("lat"), py::arg("alt"), py::arg("el"));

  m.def("iono_klobuchar", [](py::array_t<double> alpha, py::array_t<double> beta,
                              double lat, double lon, double az, double el,
                              double gps_time) {
    auto ba = alpha.request();
    auto bb = beta.request();
    if (ba.size != 4 || bb.size != 4)
      throw std::runtime_error("alpha and beta must each have 4 elements");
    return gnss_gpu::iono_klobuchar(static_cast<double*>(ba.ptr),
                                     static_cast<double*>(bb.ptr),
                                     lat, lon, az, el, gps_time);
  }, "Klobuchar ionospheric delay model (GPS broadcast)\n"
     "Returns delay in meters (L1 frequency)",
     py::arg("alpha"), py::arg("beta"),
     py::arg("lat"), py::arg("lon"), py::arg("az"), py::arg("el"),
     py::arg("gps_time"));

  m.def("tropo_correction_batch", [](py::array_t<double> rx_lla,
                                      py::array_t<double> sat_el) {
    auto blla = rx_lla.request();
    auto bel = sat_el.request();

    int n_epoch, n_sat;
    if (blla.ndim == 1) {
      // Single epoch: lla is [3], sat_el is [n_sat]
      n_epoch = 1;
      n_sat = bel.size;
    } else {
      n_epoch = blla.shape[0];
      n_sat = bel.shape[1];
    }

    auto corrections = py::array_t<double>({n_epoch, n_sat});
    gnss_gpu::tropo_correction_batch(
        static_cast<double*>(blla.ptr),
        static_cast<double*>(bel.ptr),
        static_cast<double*>(corrections.request().ptr),
        n_epoch, n_sat);

    if (blla.ndim == 1) {
      // Return 1D for single epoch input
      corrections.resize({n_sat});
    }
    return corrections;
  }, "Batch tropospheric correction (GPU)",
     py::arg("rx_lla"), py::arg("sat_el"));

  m.def("iono_correction_batch", [](py::array_t<double> rx_lla,
                                     py::array_t<double> sat_az,
                                     py::array_t<double> sat_el,
                                     py::array_t<double> alpha,
                                     py::array_t<double> beta,
                                     py::array_t<double> gps_times) {
    auto blla = rx_lla.request();
    auto baz = sat_az.request();
    auto bel = sat_el.request();
    auto ba = alpha.request();
    auto bb = beta.request();
    auto bt = gps_times.request();

    if (ba.size != 4 || bb.size != 4)
      throw std::runtime_error("alpha and beta must each have 4 elements");

    int n_epoch, n_sat;
    if (blla.ndim == 1) {
      n_epoch = 1;
      n_sat = bel.size;
    } else {
      n_epoch = blla.shape[0];
      n_sat = bel.shape[1];
    }

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

    if (blla.ndim == 1) {
      corrections.resize({n_sat});
    }
    return corrections;
  }, "Batch ionospheric correction (GPU)",
     py::arg("rx_lla"), py::arg("sat_az"), py::arg("sat_el"),
     py::arg("alpha"), py::arg("beta"), py::arg("gps_times"));
}
