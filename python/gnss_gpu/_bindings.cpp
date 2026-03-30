#include <pybind11/pybind11.h>
#include "gnss_gpu/gnss_gpu.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu, m) {
  m.doc() = "GPU-accelerated GNSS signal processing";
  m.def("hello", &gnss_gpu::hello, "Run a hello kernel on the GPU");
}
