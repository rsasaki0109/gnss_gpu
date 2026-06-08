#include <cmath>
#include <stdexcept>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "gnss_gpu/multi_gnss.h"

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

void validate_options(const char* name, int n_systems, int max_iter, double tol) {
  if (n_systems < 1 || n_systems > gnss_gpu::MAX_SYSTEMS)
    throw std::runtime_error(
        std::string(name) + ": n_systems must be in [1, " +
        std::to_string(gnss_gpu::MAX_SYSTEMS) + "]");
  if (max_iter < 1)
    throw std::runtime_error(std::string(name) + ": max_iter must be >= 1");
  if (!std::isfinite(tol) || tol <= 0.0)
    throw std::runtime_error(std::string(name) + ": tol must be positive and finite");
}

void validate_flat_values(const char* name, const double* sat, const double* pr, const double* weights,
                          const int* system_ids, int n_sat, int n_systems) {
  for (int i = 0; i < n_sat * 3; i++) {
    if (!std::isfinite(sat[i]))
      throw std::runtime_error(std::string(name) + ": satellite positions must be finite");
  }
  for (int i = 0; i < n_sat; i++) {
    if (!std::isfinite(pr[i]))
      throw std::runtime_error(std::string(name) + ": pseudoranges must be finite");
    if (!std::isfinite(weights[i]) || weights[i] < 0.0)
      throw std::runtime_error(std::string(name) + ": weights must be finite and nonnegative");
    if (system_ids[i] < 0 || system_ids[i] >= n_systems)
      throw std::runtime_error(std::string(name) + ": system_ids must be in [0, n_systems)");
  }
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_multi_gnss, m) {
  m.doc() = "Multi-GNSS WLS positioning with ISB estimation";

  m.def("wls_multi_gnss", [](DoubleArray sat_ecef,
                              DoubleArray pseudoranges,
                              DoubleArray weights,
                              IntArray system_ids,
                              int n_systems, int max_iter, double tol) {
    auto bs = sat_ecef.request();
    auto bp = pseudoranges.request();
    auto bw = weights.request();
    auto bsys = system_ids.request();
    validate_options("wls_multi_gnss", n_systems, max_iter, tol);
    if (bp.ndim != 1)
      throw std::runtime_error("wls_multi_gnss: pseudoranges must have shape (n_sat,)");
    int n_sat = static_cast<int>(bp.size);
    if (n_sat < 1)
      throw std::runtime_error("wls_multi_gnss requires at least one satellite");
    if (!((bs.ndim == 1 && bs.size == n_sat * 3) ||
          (bs.ndim == 2 && bs.shape[0] == n_sat && bs.shape[1] == 3)))
      throw std::runtime_error("wls_multi_gnss: sat_ecef must have shape (n_sat, 3) or flat length n_sat*3");
    if (bw.ndim != 1 || bw.size != n_sat)
      throw std::runtime_error("wls_multi_gnss: weights must have shape (n_sat,)");
    if (bsys.ndim != 1 || bsys.size != n_sat)
      throw std::runtime_error("wls_multi_gnss: system_ids must have shape (n_sat,)");
    int n_state = 3 + n_systems;
    const double* sat_ptr = static_cast<const double*>(bs.ptr);
    const double* pr_ptr = static_cast<const double*>(bp.ptr);
    const double* weights_ptr = static_cast<const double*>(bw.ptr);
    const int* system_ptr = static_cast<const int*>(bsys.ptr);
    validate_flat_values("wls_multi_gnss", sat_ptr, pr_ptr, weights_ptr, system_ptr, n_sat, n_systems);

    auto result = py::array_t<double>(std::vector<py::ssize_t>{n_state});
    double* result_ptr = static_cast<double*>(result.request().ptr);
    int iters = gnss_gpu::wls_multi_gnss(
        sat_ptr,
        pr_ptr,
        weights_ptr,
        system_ptr,
        result_ptr, n_sat, n_systems,
        max_iter, tol);
    return py::make_tuple(result, iters);
  }, "Single-epoch multi-GNSS WLS positioning with ISB estimation",
     py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights"),
     py::arg("system_ids"), py::arg("n_systems"),
     py::arg("max_iter") = 10, py::arg("tol") = 1e-4);

  m.def("wls_multi_gnss_batch", [](DoubleArray sat_ecef,
                                    DoubleArray pseudoranges,
                                    DoubleArray weights,
                                    IntArray system_ids,
                                    int n_systems, int max_iter, double tol) {
    auto bs = sat_ecef.request();
    auto bp = pseudoranges.request();
    auto bw = weights.request();
    auto bsys = system_ids.request();
    validate_options("wls_multi_gnss_batch", n_systems, max_iter, tol);
    if (bs.ndim != 3 || bs.shape[2] != 3)
      throw std::runtime_error("wls_multi_gnss_batch: sat_ecef must have shape (n_epoch, n_sat, 3)");
    int n_epoch = static_cast<int>(bs.shape[0]);
    int n_sat = static_cast<int>(bs.shape[1]);
    if (n_epoch < 1)
      throw std::runtime_error("wls_multi_gnss_batch: n_epoch must be >= 1");
    int n_state = 3 + n_systems;
    if (n_sat < n_state)
      throw std::runtime_error("wls_multi_gnss_batch requires at least 3 + n_systems satellites");
    if (bp.ndim != 2 || bp.shape[0] != n_epoch || bp.shape[1] != n_sat)
      throw std::runtime_error("wls_multi_gnss_batch: pseudoranges must have shape (n_epoch, n_sat)");
    if (bw.ndim != 2 || bw.shape[0] != n_epoch || bw.shape[1] != n_sat)
      throw std::runtime_error("wls_multi_gnss_batch: weights must have shape (n_epoch, n_sat)");
    if (bsys.ndim != 2 || bsys.shape[0] != n_epoch || bsys.shape[1] != n_sat)
      throw std::runtime_error("wls_multi_gnss_batch: system_ids must have shape (n_epoch, n_sat)");
    const double* sat_ptr = static_cast<const double*>(bs.ptr);
    const double* pr_ptr = static_cast<const double*>(bp.ptr);
    const double* weights_ptr = static_cast<const double*>(bw.ptr);
    const int* system_ptr = static_cast<const int*>(bsys.ptr);
    validate_flat_values("wls_multi_gnss_batch", sat_ptr, pr_ptr, weights_ptr, system_ptr,
                         n_epoch * n_sat, n_systems);

    auto results = py::array_t<double>({n_epoch, n_state});
    auto iters = py::array_t<int>(std::vector<py::ssize_t>{n_epoch});
    gnss_gpu::wls_multi_gnss_batch(
        sat_ptr,
        pr_ptr,
        weights_ptr,
        system_ptr,
        static_cast<double*>(results.request().ptr),
        static_cast<int*>(iters.request().ptr),
        n_epoch, n_sat, n_systems,
        max_iter, tol);
    return py::make_tuple(results, iters);
  }, "Batch multi-GNSS WLS positioning (GPU parallel)",
     py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights"),
     py::arg("system_ids"), py::arg("n_systems"),
     py::arg("max_iter") = 10, py::arg("tol") = 1e-4);
}
