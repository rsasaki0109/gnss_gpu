#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "gnss_gpu/doppler.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_doppler, m) {
  m.doc() = "GPU-accelerated Doppler velocity estimation";

  m.def("doppler_velocity", [](py::array_t<double, py::array::c_style | py::array::forcecast> sat_ecef,
                                py::array_t<double, py::array::c_style | py::array::forcecast> sat_vel,
                                py::array_t<double, py::array::c_style | py::array::forcecast> doppler,
                                py::array_t<double, py::array::c_style | py::array::forcecast> rx_pos,
                                py::array_t<double, py::array::c_style | py::array::forcecast> weights,
                                double wavelength, int max_iter, double tol) {
    auto bs = sat_ecef.request();
    auto bv = sat_vel.request();
    auto bd = doppler.request();
    auto br = rx_pos.request();
    auto bw = weights.request();
    if (bd.ndim != 1)
      throw std::runtime_error("doppler_velocity: doppler must have shape (n_sat,)");
    int n_sat = bd.size;
    if (n_sat < 1)
      throw std::runtime_error("doppler_velocity requires at least one satellite");
    if (!((bs.ndim == 1 && bs.size == n_sat * 3) ||
          (bs.ndim == 2 && bs.shape[0] == n_sat && bs.shape[1] == 3)))
      throw std::runtime_error("doppler_velocity: sat_ecef must have shape (n_sat, 3) or flat length n_sat*3");
    if (!((bv.ndim == 1 && bv.size == n_sat * 3) ||
          (bv.ndim == 2 && bv.shape[0] == n_sat && bv.shape[1] == 3)))
      throw std::runtime_error("doppler_velocity: sat_vel must have shape (n_sat, 3) or flat length n_sat*3");
    if (br.ndim != 1 || br.size != 3)
      throw std::runtime_error("doppler_velocity: rx_pos must have shape (3,)");
    if (bw.ndim != 1 || bw.size != n_sat)
      throw std::runtime_error("doppler_velocity: weights must have shape (n_sat,)");
    if (!std::isfinite(wavelength) || wavelength <= 0.0)
      throw std::runtime_error("doppler_velocity: wavelength must be positive and finite");
    if (max_iter < 1)
      throw std::runtime_error("doppler_velocity: max_iter must be >= 1");
    if (!std::isfinite(tol) || tol <= 0.0)
      throw std::runtime_error("doppler_velocity: tol must be positive and finite");
    const double* sat_ptr = static_cast<const double*>(bs.ptr);
    const double* vel_ptr = static_cast<const double*>(bv.ptr);
    const double* dop_ptr = static_cast<const double*>(bd.ptr);
    const double* rx_ptr = static_cast<const double*>(br.ptr);
    const double* weight_ptr = static_cast<const double*>(bw.ptr);
    for (int i = 0; i < n_sat * 3; i++) {
      if (!std::isfinite(sat_ptr[i]) || !std::isfinite(vel_ptr[i]))
        throw std::runtime_error("doppler_velocity: satellite positions and velocities must be finite");
    }
    for (int i = 0; i < n_sat; i++) {
      if (!std::isfinite(dop_ptr[i]))
        throw std::runtime_error("doppler_velocity: doppler values must be finite");
      if (!std::isfinite(weight_ptr[i]) || weight_ptr[i] < 0.0)
        throw std::runtime_error("doppler_velocity: weights must be finite and nonnegative");
    }
    for (int i = 0; i < 3; i++) {
      if (!std::isfinite(rx_ptr[i]))
        throw std::runtime_error("doppler_velocity: rx_pos must be finite");
    }

    auto result = py::array_t<double>({4}, {sizeof(double)});
    double* result_ptr = result.mutable_data();
    int iters = gnss_gpu::doppler_velocity(
        sat_ptr,
        vel_ptr,
        dop_ptr,
        rx_ptr,
        weight_ptr,
        result_ptr, n_sat, wavelength, max_iter, tol);
    return py::make_tuple(result, iters);
  }, "Single-epoch Doppler velocity estimation (WLS)",
     py::arg("sat_ecef"), py::arg("sat_vel"), py::arg("doppler"),
     py::arg("rx_pos"), py::arg("weights"),
     py::arg("wavelength") = 0.19029367,
     py::arg("max_iter") = 10, py::arg("tol") = 1e-6);

  m.def("doppler_velocity_batch", [](py::array_t<double, py::array::c_style | py::array::forcecast> sat_ecef,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> sat_vel,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> doppler,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> rx_pos,
                                      py::array_t<double, py::array::c_style | py::array::forcecast> weights,
                                      double wavelength, int max_iter, double tol) {
    auto bs = sat_ecef.request();
    auto bv = sat_vel.request();
    auto bd = doppler.request();
    auto br = rx_pos.request();
    auto bw = weights.request();
    if (bs.ndim != 3 || bs.shape[2] != 3)
      throw std::runtime_error("doppler_velocity_batch: sat_ecef must have shape (n_epoch, n_sat, 3)");
    int n_epoch = bs.shape[0];
    int n_sat = bs.shape[1];
    if (n_epoch < 1)
      throw std::runtime_error("doppler_velocity_batch: n_epoch must be >= 1");
    if (n_sat < 4)
      throw std::runtime_error("doppler_velocity_batch requires at least 4 satellites");
    if (bv.ndim != 3 || bv.shape[0] != n_epoch || bv.shape[1] != n_sat || bv.shape[2] != 3)
      throw std::runtime_error("doppler_velocity_batch: sat_vel shape must match sat_ecef");
    if (bd.ndim != 2 || bd.shape[0] != n_epoch || bd.shape[1] != n_sat)
      throw std::runtime_error("doppler_velocity_batch: doppler must have shape (n_epoch, n_sat)");
    if (br.ndim != 2 || br.shape[0] != n_epoch || br.shape[1] != 3)
      throw std::runtime_error("doppler_velocity_batch: rx_pos must have shape (n_epoch, 3)");
    if (bw.ndim != 2 || bw.shape[0] != n_epoch || bw.shape[1] != n_sat)
      throw std::runtime_error("doppler_velocity_batch: weights must have shape (n_epoch, n_sat)");
    if (!std::isfinite(wavelength) || wavelength <= 0.0)
      throw std::runtime_error("doppler_velocity_batch: wavelength must be positive and finite");
    if (max_iter < 1)
      throw std::runtime_error("doppler_velocity_batch: max_iter must be >= 1");
    if (!std::isfinite(tol) || tol <= 0.0)
      throw std::runtime_error("doppler_velocity_batch: tol must be positive and finite");
    const double* sat_ptr = static_cast<const double*>(bs.ptr);
    const double* vel_ptr = static_cast<const double*>(bv.ptr);
    const double* dop_ptr = static_cast<const double*>(bd.ptr);
    const double* rx_ptr = static_cast<const double*>(br.ptr);
    const double* weight_ptr = static_cast<const double*>(bw.ptr);
    for (int i = 0; i < n_epoch * n_sat * 3; i++) {
      if (!std::isfinite(sat_ptr[i]) || !std::isfinite(vel_ptr[i]))
        throw std::runtime_error("doppler_velocity_batch: satellite positions and velocities must be finite");
    }
    for (int i = 0; i < n_epoch * n_sat; i++) {
      if (!std::isfinite(dop_ptr[i]))
        throw std::runtime_error("doppler_velocity_batch: doppler values must be finite");
      if (!std::isfinite(weight_ptr[i]) || weight_ptr[i] < 0.0)
        throw std::runtime_error("doppler_velocity_batch: weights must be finite and nonnegative");
    }
    for (int i = 0; i < n_epoch * 3; i++) {
      if (!std::isfinite(rx_ptr[i]))
        throw std::runtime_error("doppler_velocity_batch: rx_pos must be finite");
    }

    auto results = py::array_t<double>({n_epoch, 4});
    auto iters = py::array_t<int>(std::vector<py::ssize_t>{n_epoch});

    gnss_gpu::doppler_velocity_batch(
        sat_ptr,
        vel_ptr,
        dop_ptr,
        rx_ptr,
        weight_ptr,
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
