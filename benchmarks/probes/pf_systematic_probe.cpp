// Read-only benchmark diagnostics. Import the actual PF extension first so
// its global pybind type supplies the handle; this probe never owns PF state.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include "gnss_gpu/pf_device.h"

namespace py = pybind11;

static void check_cuda(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("PF probe: ") + cudaGetErrorString(error));
}

PYBIND11_MODULE(_pf_systematic_probe, m) {
  m.doc() = "Read-only diagnostic buffers after systematic resampling";
  m.def("read", [](const gnss_gpu::PFDeviceState* state) {
    if (!state || !state->allocated)
      throw std::runtime_error("PF probe: state is not allocated");
    check_cuda(cudaStreamSynchronize(state->stream));
    auto weights = py::array_t<double>(std::vector<py::ssize_t>{state->n_particles});
    auto cdf = py::array_t<double>(std::vector<py::ssize_t>{state->n_particles});
    auto max_sum = py::array_t<double>(std::vector<py::ssize_t>{2});
    size_t bytes = (size_t)state->n_particles * sizeof(double);
    check_cuda(cudaMemcpy(weights.mutable_data(), state->d_weights_norm, bytes, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(cdf.mutable_data(), state->d_cdf, bytes, cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(max_sum.mutable_data(), state->d_reduction_result, 2 * sizeof(double), cudaMemcpyDeviceToHost));
    return py::make_tuple(weights, cdf, max_sum);
  }, py::arg("state"));
}
