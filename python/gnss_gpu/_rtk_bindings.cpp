#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/rtk.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_rtk, m) {
  m.doc() = "RTK carrier phase positioning";

  m.def("rtk_float", [](py::array_t<double> base_ecef,
                         py::array_t<double> rover_pr, py::array_t<double> base_pr,
                         py::array_t<double> rover_carrier, py::array_t<double> base_carrier,
                         py::array_t<double> sat_ecef,
                         double wavelength, int max_iter, double tol) {
    auto bb = base_ecef.request();
    auto brpr = rover_pr.request(), bbpr = base_pr.request();
    auto brcp = rover_carrier.request(), bbcp = base_carrier.request();
    auto bsat = sat_ecef.request();
    int n_sat = brpr.size;
    int n_dd = n_sat - 1;

    auto result = py::array_t<double>({3}, {sizeof(double)});
    auto ambiguities = py::array_t<double>(std::vector<ssize_t>{n_dd});
    auto residuals = py::array_t<double>(std::vector<ssize_t>{2 * n_dd});

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

  m.def("rtk_float_batch", [](py::array_t<double> base_ecef,
                                py::array_t<double> rover_pr, py::array_t<double> base_pr,
                                py::array_t<double> rover_carrier, py::array_t<double> base_carrier,
                                py::array_t<double> sat_ecef,
                                double wavelength, int max_iter, double tol) {
    auto bb = base_ecef.request();
    auto brpr = rover_pr.request();
    int n_epoch = brpr.shape[0];
    int n_sat = brpr.shape[1];
    int n_dd = n_sat - 1;

    auto results = py::array_t<double>({n_epoch, 3});
    auto ambiguities = py::array_t<double>({n_epoch, n_dd});
    auto iters = py::array_t<int>(std::vector<ssize_t>{n_epoch});

    gnss_gpu::rtk_float_batch(
        static_cast<double*>(bb.ptr),
        static_cast<double*>(brpr.ptr),
        static_cast<double*>(base_pr.request().ptr),
        static_cast<double*>(rover_carrier.request().ptr),
        static_cast<double*>(base_carrier.request().ptr),
        static_cast<double*>(sat_ecef.request().ptr),
        static_cast<double*>(results.request().ptr),
        static_cast<double*>(ambiguities.request().ptr),
        static_cast<int*>(iters.request().ptr),
        n_epoch, n_sat, wavelength, max_iter, tol);

    return py::make_tuple(results, ambiguities, iters);
  }, "Batch RTK float solution (GPU parallel)",
     py::arg("base_ecef"), py::arg("rover_pr"), py::arg("base_pr"),
     py::arg("rover_carrier"), py::arg("base_carrier"), py::arg("sat_ecef"),
     py::arg("wavelength") = 0.19029, py::arg("max_iter") = 20, py::arg("tol") = 1e-4);

  m.def("lambda_integer", [](py::array_t<double> float_amb, py::array_t<double> Q_amb,
                               int n_candidates) {
    auto ba = float_amb.request(), bq = Q_amb.request();
    int n = ba.size;
    auto fixed_amb = py::array_t<int>(std::vector<ssize_t>{n});

    double ratio = gnss_gpu::lambda_integer(
        static_cast<double*>(ba.ptr),
        static_cast<double*>(bq.ptr),
        static_cast<int*>(fixed_amb.request().ptr),
        n, n_candidates);

    return py::make_tuple(fixed_amb, ratio);
  }, "LAMBDA integer ambiguity resolution",
     py::arg("float_amb"), py::arg("Q_amb"), py::arg("n_candidates") = 100);
}
