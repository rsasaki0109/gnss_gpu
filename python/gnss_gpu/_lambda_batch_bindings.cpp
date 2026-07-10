#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "gnss_gpu/lambda_batch.h"

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

}  // namespace

PYBIND11_MODULE(_gnss_gpu_lambda_batch, m) {
  m.doc() = "Batched LAMBDA/MLAMBDA integer ambiguity resolution (CUDA)";

  m.def("lambda_batch_max_n", &gnss_gpu::lambda_batch_max_n,
        "Maximum per-problem ambiguity count supported by the kernel");

  m.def("mlambda_batch", [](DoubleArray ahat_flat, DoubleArray Q_flat,
                             IntArray n_arr, int ncands, int parmode,
                             double P0) {
    auto ba = ahat_flat.request();
    auto bq = Q_flat.request();
    auto bn = n_arr.request();

    if (bn.ndim != 1 || bn.size < 1) {
      throw std::runtime_error("n_arr must be a non-empty 1D array");
    }
    const int n_prob = static_cast<int>(bn.size);
    const int* n_ptr = static_cast<const int*>(bn.ptr);

    long long atot = 0, qtot = 0;
    for (int p = 0; p < n_prob; p++) {
      if (n_ptr[p] < 1) {
        throw std::runtime_error("every n must be >= 1");
      }
      atot += n_ptr[p];
      qtot += static_cast<long long>(n_ptr[p]) * n_ptr[p];
    }
    if (ba.ndim != 1 || ba.size != atot) {
      throw std::runtime_error("ahat_flat size must equal sum(n)");
    }
    if (bq.ndim != 1 || bq.size != qtot) {
      throw std::runtime_error("Q_flat size must equal sum(n*n)");
    }
    if (ncands < 1 || ncands > 16) {
      throw std::runtime_error("ncands must be in [1, 16]");
    }
    if (parmode != 1 && parmode != 2) {
      throw std::runtime_error("parmode must be 1 or 2");
    }
    if (!std::isfinite(P0)) {
      throw std::runtime_error("P0 must be finite");
    }

    auto afix = py::array_t<double>(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(atot * ncands)});
    auto zfix = py::array_t<double>(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(atot * ncands)});
    auto s = py::array_t<double>(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(n_prob) * ncands});
    auto s_len = py::array_t<int>(std::vector<py::ssize_t>{n_prob});
    auto nfix = py::array_t<int>(std::vector<py::ssize_t>{n_prob});
    auto ps = py::array_t<double>(std::vector<py::ssize_t>{n_prob});
    auto status = py::array_t<int>(std::vector<py::ssize_t>{n_prob});

    gnss_gpu::lambda_batch(
        static_cast<const double*>(ba.ptr),
        static_cast<const double*>(bq.ptr), n_ptr, n_prob, ncands,
        parmode, P0, static_cast<double*>(afix.request().ptr),
        static_cast<double*>(zfix.request().ptr),
        static_cast<double*>(s.request().ptr),
        static_cast<int*>(s_len.request().ptr),
        static_cast<int*>(nfix.request().ptr),
        static_cast<double*>(ps.request().ptr),
        static_cast<int*>(status.request().ptr));

    return py::make_tuple(afix, zfix, s, s_len, nfix, ps, status);
  }, "Batched MLAMBDA over flattened (ahat, Qahat) problems",
     py::arg("ahat_flat"), py::arg("Q_flat"), py::arg("n_arr"),
     py::arg("ncands") = 2, py::arg("parmode") = 2,
     py::arg("P0") = 0.995);
}
