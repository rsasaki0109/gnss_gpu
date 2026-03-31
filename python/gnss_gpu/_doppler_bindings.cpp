#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/doppler.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_doppler, m) {
  m.doc() = "GPU-accelerated Doppler velocity estimation";

  m.def("doppler_velocity", [](py::array_t<double> sat_ecef,
                                py::array_t<double> sat_vel,
                                py::array_t<double> doppler,
                                py::array_t<double> rx_pos,
                                py::array_t<double> weights,
                                double wavelength, int max_iter, double tol) {
    auto bs = sat_ecef.request();
    auto bv = sat_vel.request();
    auto bd = doppler.request();
    auto br = rx_pos.request();
    auto bw = weights.request();
    // sat_ecef: accept (N,3) or (N*3,) flat
    if (br.size < 3)
      throw std::runtime_error("rx_pos must have at least 3 elements");
    int n_sat = bd.size;

    auto result = py::array_t<double>({4}, {sizeof(double)});
    double* result_ptr = result.mutable_data();
    int iters = gnss_gpu::doppler_velocity(
        static_cast<double*>(bs.ptr),
        static_cast<double*>(bv.ptr),
        static_cast<double*>(bd.ptr),
        static_cast<double*>(br.ptr),
        static_cast<double*>(bw.ptr),
        result_ptr, n_sat, wavelength, max_iter, tol);
    return py::make_tuple(result, iters);
  }, "Single-epoch Doppler velocity estimation (WLS)",
     py::arg("sat_ecef"), py::arg("sat_vel"), py::arg("doppler"),
     py::arg("rx_pos"), py::arg("weights"),
     py::arg("wavelength") = 0.19029367,
     py::arg("max_iter") = 10, py::arg("tol") = 1e-6);

  m.def("doppler_velocity_batch", [](py::array_t<double> sat_ecef,
                                      py::array_t<double> sat_vel,
                                      py::array_t<double> doppler,
                                      py::array_t<double> rx_pos,
                                      py::array_t<double> weights,
                                      double wavelength, int max_iter, double tol) {
    auto bs = sat_ecef.request();
    {
      auto bv = sat_vel.request();
    }
    int n_epoch = bs.shape[0];
    int n_sat = bs.shape[1];

    auto results = py::array_t<double>({n_epoch, 4});
    auto iters = py::array_t<int>(n_epoch);

    gnss_gpu::doppler_velocity_batch(
        static_cast<double*>(bs.ptr),
        static_cast<double*>(sat_vel.request().ptr),
        static_cast<double*>(doppler.request().ptr),
        static_cast<double*>(rx_pos.request().ptr),
        static_cast<double*>(weights.request().ptr),
        static_cast<double*>(results.request().ptr),
        static_cast<int*>(iters.request().ptr),
        n_epoch, n_sat, wavelength, max_iter, tol);
    return py::make_tuple(results, iters);
  }, "Batch Doppler velocity estimation (GPU parallel)",
     py::arg("sat_ecef"), py::arg("sat_vel"), py::arg("doppler"),
     py::arg("rx_pos"), py::arg("weights"),
     py::arg("wavelength") = 0.19029367,
     py::arg("max_iter") = 10, py::arg("tol") = 1e-6);
}
