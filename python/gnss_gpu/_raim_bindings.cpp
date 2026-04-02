#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/raim.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_raim, m) {
  m.doc() = "RAIM / FDE integrity monitoring";

  py::class_<gnss_gpu::RAIMResult>(m, "RAIMResult")
    .def(py::init<>())
    .def_readwrite("integrity_ok", &gnss_gpu::RAIMResult::integrity_ok)
    .def_readwrite("hpl", &gnss_gpu::RAIMResult::hpl)
    .def_readwrite("vpl", &gnss_gpu::RAIMResult::vpl)
    .def_readwrite("test_statistic", &gnss_gpu::RAIMResult::test_statistic)
    .def_readwrite("threshold", &gnss_gpu::RAIMResult::threshold)
    .def_readwrite("excluded_sat", &gnss_gpu::RAIMResult::excluded_sat);

  m.def("raim_check", [](py::array_t<double> sat_ecef, py::array_t<double> pseudoranges,
                          py::array_t<double> weights, py::array_t<double> position,
                          double p_fa) {
    auto bs = sat_ecef.request();
    auto bp = pseudoranges.request();
    auto bw = weights.request();
    auto bpos = position.request();
    // sat_ecef: accept (N,3) or (N*3,) flat
    if (bpos.size < 4)
      throw std::runtime_error("position must have at least 4 elements [x, y, z, cb]");
    int n_sat = bp.size;
    if (n_sat < 4)
      throw std::runtime_error("raim_check requires at least 4 satellites");

    gnss_gpu::RAIMResult result;
    gnss_gpu::raim_check(static_cast<double*>(bs.ptr),
                         static_cast<double*>(bp.ptr),
                         static_cast<double*>(bw.ptr),
                         static_cast<double*>(bpos.ptr),
                         &result, n_sat, p_fa);
    return result;
  }, "RAIM chi-squared consistency check",
     py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights"),
     py::arg("position"), py::arg("p_fa") = 1e-5);

  m.def("raim_fde", [](py::array_t<double> sat_ecef, py::array_t<double> pseudoranges,
                        py::array_t<double> weights, py::array_t<double> position,
                        double p_fa) {
    auto bs = sat_ecef.request();
    auto bp = pseudoranges.request();
    auto bw = weights.request();
    auto bpos = position.request();
    // sat_ecef: accept (N,3) or (N*3,) flat
    if (bpos.size < 4)
      throw std::runtime_error("position must have at least 4 elements [x, y, z, cb]");
    int n_sat = bp.size;
    if (n_sat < 4)
      throw std::runtime_error("raim_fde requires at least 4 satellites");

    // Copy position so we can modify it
    auto pos_out = py::array_t<double>({4}, {sizeof(double)});
    double* pos_ptr = pos_out.mutable_data();
    const double* pos_in = static_cast<double*>(bpos.ptr);
    for (int i = 0; i < 4; i++) pos_ptr[i] = pos_in[i];

    gnss_gpu::RAIMResult result;
    gnss_gpu::raim_fde(static_cast<double*>(bs.ptr),
                       static_cast<double*>(bp.ptr),
                       static_cast<double*>(bw.ptr),
                       pos_ptr,
                       &result, n_sat, p_fa);
    return py::make_tuple(result, pos_out);
  }, "RAIM with Fault Detection and Exclusion",
     py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights"),
     py::arg("position"), py::arg("p_fa") = 1e-5);
}
