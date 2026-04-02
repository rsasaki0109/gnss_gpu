#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/multi_gnss.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_multi_gnss, m) {
  m.doc() = "Multi-GNSS WLS positioning with ISB estimation";

  m.def("wls_multi_gnss", [](py::array_t<double> sat_ecef,
                              py::array_t<double> pseudoranges,
                              py::array_t<double> weights,
                              py::array_t<int> system_ids,
                              int n_systems, int max_iter, double tol) {
    auto bs = sat_ecef.request();
    auto bp = pseudoranges.request();
    auto bw = weights.request();
    auto bsys = system_ids.request();
    // sat_ecef: accept (N,3) or (N*3,) flat
    int n_sat = bp.size;
    int n_state = 3 + n_systems;
    auto result = py::array_t<double>(std::vector<ssize_t>{n_state});
    double* result_ptr = static_cast<double*>(result.request().ptr);
    int iters = gnss_gpu::wls_multi_gnss(
        static_cast<double*>(bs.ptr),
        static_cast<double*>(bp.ptr),
        static_cast<double*>(bw.ptr),
        static_cast<int*>(bsys.ptr),
        result_ptr, n_sat, n_systems,
        max_iter, tol);
    return py::make_tuple(result, iters);
  }, "Single-epoch multi-GNSS WLS positioning with ISB estimation",
     py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights"),
     py::arg("system_ids"), py::arg("n_systems"),
     py::arg("max_iter") = 10, py::arg("tol") = 1e-4);

  m.def("wls_multi_gnss_batch", [](py::array_t<double> sat_ecef,
                                    py::array_t<double> pseudoranges,
                                    py::array_t<double> weights,
                                    py::array_t<int> system_ids,
                                    int n_systems, int max_iter, double tol) {
    auto bs = sat_ecef.request();
    auto bp = pseudoranges.request();
    int n_epoch = bs.shape[0];
    int n_sat = bs.shape[1];
    int n_state = 3 + n_systems;
    if (n_sat < n_state)
      throw std::runtime_error("wls_multi_gnss_batch requires at least 3 + n_systems satellites");
    auto results = py::array_t<double>({n_epoch, n_state});
    auto iters = py::array_t<int>(std::vector<ssize_t>{n_epoch});
    gnss_gpu::wls_multi_gnss_batch(
        static_cast<double*>(bs.ptr),
        static_cast<double*>(bp.ptr),
        static_cast<double*>(weights.request().ptr),
        static_cast<int*>(system_ids.request().ptr),
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
