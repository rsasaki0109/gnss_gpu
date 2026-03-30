#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/raytrace.h"

namespace py = pybind11;

PYBIND11_MODULE(_raytrace, m) {
  m.doc() = "GPU-accelerated ray tracing for GNSS NLOS detection";

  m.def("raytrace_los_check", [](py::array_t<double> rx_ecef,
                                  py::array_t<double> sat_ecef,
                                  py::array_t<double> triangles) {
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto btri = triangles.request();
    int n_sat = bsat.shape[0];
    int n_tri = btri.shape[0];

    // Use int array for CUDA kernel output
    auto is_los_int = py::array_t<int>(std::vector<ssize_t>{n_sat});
    int* int_ptr = is_los_int.mutable_data();

    gnss_gpu::raytrace_los_check(
        static_cast<double*>(brx.ptr),
        static_cast<double*>(bsat.ptr),
        reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr),
        int_ptr,
        n_sat, n_tri);

    // Convert to bool array
    auto is_los = py::array_t<bool>(std::vector<ssize_t>{n_sat});
    bool* bool_ptr = is_los.mutable_data();
    for (int i = 0; i < n_sat; i++) bool_ptr[i] = (int_ptr[i] != 0);
    return is_los;
  }, "Batch LOS check using ray tracing against building triangles",
     py::arg("rx_ecef"), py::arg("sat_ecef"), py::arg("triangles"));

  m.def("raytrace_multipath", [](py::array_t<double> rx_ecef,
                                  py::array_t<double> sat_ecef,
                                  py::array_t<double> triangles) {
    auto brx = rx_ecef.request();
    auto bsat = sat_ecef.request();
    auto btri = triangles.request();
    int n_sat = bsat.shape[0];
    int n_tri = btri.shape[0];

    auto reflection_points = py::array_t<double>({n_sat, 3});
    auto excess_delays = py::array_t<double>(std::vector<ssize_t>{n_sat});

    gnss_gpu::raytrace_multipath(
        static_cast<double*>(brx.ptr),
        static_cast<double*>(bsat.ptr),
        reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr),
        static_cast<double*>(reflection_points.request().ptr),
        static_cast<double*>(excess_delays.request().ptr),
        n_sat, n_tri);

    return py::make_tuple(reflection_points, excess_delays);
  }, "Compute first-order multipath reflections",
     py::arg("rx_ecef"), py::arg("sat_ecef"), py::arg("triangles"));
}
