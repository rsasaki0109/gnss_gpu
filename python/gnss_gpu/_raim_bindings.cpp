#include <cmath>
#include <stdexcept>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/raim.h"

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

struct RAIMInputs {
  int n_sat;
  const double* sat_ptr;
  const double* pr_ptr;
  const double* weights_ptr;
  const double* position_ptr;
};

RAIMInputs validate_raim_inputs(const char* name, DoubleArray& sat_ecef, DoubleArray& pseudoranges,
                                DoubleArray& weights, DoubleArray& position, double p_fa) {
  auto bs = sat_ecef.request();
  auto bp = pseudoranges.request();
  auto bw = weights.request();
  auto bpos = position.request();

  if (bp.ndim != 1)
    throw std::runtime_error(std::string(name) + ": pseudoranges must have shape (n_sat,)");
  int n_sat = static_cast<int>(bp.size);
  if (n_sat < 4)
    throw std::runtime_error(std::string(name) + " requires at least 4 satellites");
  if (!((bs.ndim == 1 && bs.size == n_sat * 3) ||
        (bs.ndim == 2 && bs.shape[0] == n_sat && bs.shape[1] == 3)))
    throw std::runtime_error(std::string(name) + ": sat_ecef must have shape (n_sat, 3) or flat length n_sat*3");
  if (bw.ndim != 1 || bw.size != n_sat)
    throw std::runtime_error(std::string(name) + ": weights must have shape (n_sat,)");
  if (bpos.ndim != 1 || bpos.size != 4)
    throw std::runtime_error(std::string(name) + ": position must have shape (4,)");
  if (!std::isfinite(p_fa) || p_fa <= 0.0 || p_fa >= 1.0)
    throw std::runtime_error(std::string(name) + ": p_fa must be in (0, 1)");

  const double* sat_ptr = static_cast<const double*>(bs.ptr);
  const double* pr_ptr = static_cast<const double*>(bp.ptr);
  const double* weights_ptr = static_cast<const double*>(bw.ptr);
  const double* position_ptr = static_cast<const double*>(bpos.ptr);

  for (int i = 0; i < n_sat * 3; i++) {
    if (!std::isfinite(sat_ptr[i]))
      throw std::runtime_error(std::string(name) + ": satellite positions must be finite");
  }
  for (int i = 0; i < n_sat; i++) {
    if (!std::isfinite(pr_ptr[i]))
      throw std::runtime_error(std::string(name) + ": pseudoranges must be finite");
    if (!std::isfinite(weights_ptr[i]) || weights_ptr[i] < 0.0)
      throw std::runtime_error(std::string(name) + ": weights must be finite and nonnegative");
  }
  for (int i = 0; i < 4; i++) {
    if (!std::isfinite(position_ptr[i]))
      throw std::runtime_error(std::string(name) + ": position must be finite");
  }

  return RAIMInputs{n_sat, sat_ptr, pr_ptr, weights_ptr, position_ptr};
}

}  // namespace

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

  m.def("raim_check", [](DoubleArray sat_ecef, DoubleArray pseudoranges,
                          DoubleArray weights, DoubleArray position,
                          double p_fa) {
    auto inputs = validate_raim_inputs("raim_check", sat_ecef, pseudoranges, weights, position, p_fa);

    gnss_gpu::RAIMResult result;
    gnss_gpu::raim_check(inputs.sat_ptr,
                         inputs.pr_ptr,
                         inputs.weights_ptr,
                         inputs.position_ptr,
                         &result, inputs.n_sat, p_fa);
    return result;
  }, "RAIM chi-squared consistency check",
     py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights"),
     py::arg("position"), py::arg("p_fa") = 1e-5);

  m.def("raim_fde", [](DoubleArray sat_ecef, DoubleArray pseudoranges,
                        DoubleArray weights, DoubleArray position,
                        double p_fa) {
    auto inputs = validate_raim_inputs("raim_fde", sat_ecef, pseudoranges, weights, position, p_fa);
    if (inputs.n_sat > 64)
      throw std::runtime_error("raim_fde supports at most 64 satellites");

    // Copy position so we can modify it
    auto pos_out = py::array_t<double>({4}, {sizeof(double)});
    double* pos_ptr = pos_out.mutable_data();
    for (int i = 0; i < 4; i++) pos_ptr[i] = inputs.position_ptr[i];

    gnss_gpu::RAIMResult result;
    gnss_gpu::raim_fde(inputs.sat_ptr,
                       inputs.pr_ptr,
                       inputs.weights_ptr,
                       pos_ptr,
                       &result, inputs.n_sat, p_fa);
    return py::make_tuple(result, pos_out);
  }, "RAIM with Fault Detection and Exclusion",
     py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights"),
     py::arg("position"), py::arg("p_fa") = 1e-5);
}
