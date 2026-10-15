// Read-only benchmark diagnostics; production extensions own and run PF state.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "gnss_gpu/pf_device.h"

namespace py = pybind11;

static void check_cuda(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("PF scan probe: ") + cudaGetErrorString(error));
}

PYBIND11_MODULE(_pf_scan_cache_probe, m) {
  m.doc() = "Read-only scan diagnostics; cache metadata requires the candidate layout";
  m.def("read", [](const gnss_gpu::PFDeviceState* state) {
    if (!state || !state->allocated)
      throw std::runtime_error("PF scan probe: state is not allocated");
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
  // Never call this on the archived baseline, whose struct ends before these fields.
  m.def("cache_metadata", [](const gnss_gpu::PFDeviceState* state) {
    if (!state) throw std::runtime_error("PF scan probe: state is null");
    py::dict result;
    result["pointer"] = reinterpret_cast<std::uintptr_t>(state->d_scan_scratch);
    result["capacity_bytes"] = state->scan_scratch_capacity;
    result["in_use"] = state->scan_scratch_in_use;
    result["allocated"] = state->allocated;
    return result;
  }, py::arg("state"));
}
