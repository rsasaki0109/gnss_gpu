#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/multipath.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_multipath, m) {
  m.doc() = "GPU-accelerated GNSS multipath simulation";

  m.def("simulate_multipath", [](py::array_t<double> rx_ecef,
                                  py::array_t<double> sat_ecef,
                                  py::array_t<double> reflector_planes,
                                  int n_rx, int n_sat, int n_ref,
                                  double carrier_freq_hz, double chip_rate) {
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto bref = reflector_planes.request();

    auto delays = py::array_t<double>(std::vector<ssize_t>{n_rx * n_sat});
    auto attenuations = py::array_t<double>(std::vector<ssize_t>{n_rx * n_sat});

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

  m.def("apply_multipath_error", [](py::array_t<double> clean_pr,
                                     py::array_t<double> rx_ecef,
                                     py::array_t<double> sat_ecef,
                                     py::array_t<double> reflector_planes,
                                     int n_epoch, int n_sat, int n_ref,
                                     double carrier_freq_hz, double chip_rate,
                                     double correlator_spacing) {
    auto bpr = clean_pr.request();
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto bref = reflector_planes.request();

    int total = n_epoch * n_sat;
    auto corrupted = py::array_t<double>(std::vector<ssize_t>{total});
    auto errors = py::array_t<double>(std::vector<ssize_t>{total});

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
