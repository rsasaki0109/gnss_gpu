#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/pf_device.h"

namespace py = pybind11;

// Custom destructor for PFDeviceState managed by pybind11.
// When pybind11 deletes the Python wrapper, this frees GPU resources
// and then deletes the C++ object, preventing double-free.
struct PFDeviceStateDeleter {
    void operator()(gnss_gpu::PFDeviceState* state) {
        if (state) {
            gnss_gpu::pf_device_destroy(state);
        }
    }
};

PYBIND11_MODULE(_gnss_gpu_pf_device, m) {
    m.doc() = "GPU-accelerated Particle Filter with persistent device memory";

    // Expose PFDeviceState as opaque Python type with custom destructor.
    // The custom holder ensures pf_device_destroy is called exactly once
    // (either explicitly or when Python GC collects the object).
    py::class_<gnss_gpu::PFDeviceState, std::unique_ptr<gnss_gpu::PFDeviceState, PFDeviceStateDeleter>>(m, "PFDeviceState",
        "Opaque handle to device-resident particle state. "
        "Memory lives on GPU between calls.")
        .def_readonly("n_particles", &gnss_gpu::PFDeviceState::n_particles)
        .def_readonly("allocated", &gnss_gpu::PFDeviceState::allocated);

    m.def("pf_device_create", [](int n_particles) {
        return std::unique_ptr<gnss_gpu::PFDeviceState, PFDeviceStateDeleter>(
            gnss_gpu::pf_device_create(n_particles), PFDeviceStateDeleter());
    }, "Allocate persistent GPU memory for particle state",
       py::arg("n_particles"));

    m.def("pf_device_destroy", [](gnss_gpu::PFDeviceState* state) {
        // Only free GPU resources; pybind11 still owns the pointer via unique_ptr.
        // Mark as deallocated so the destructor won't double-free GPU memory.
        if (state && state->allocated) {
            gnss_gpu::pf_device_destroy_resources(state);
        }
    }, "Free all GPU memory for particle state",
       py::arg("state"));

    m.def("pf_device_initialize", [](gnss_gpu::PFDeviceState* state,
                                     double init_x, double init_y, double init_z, double init_cb,
                                     double spread_pos, double spread_cb,
                                     unsigned long long seed) {
        gnss_gpu::pf_device_initialize(state, init_x, init_y, init_z, init_cb,
                                       spread_pos, spread_cb, seed);
    }, "Initialize particles on device (no H2D copy)",
       py::arg("state"),
       py::arg("init_x"), py::arg("init_y"), py::arg("init_z"), py::arg("init_cb"),
       py::arg("spread_pos"), py::arg("spread_cb"), py::arg("seed"));

    m.def("pf_device_predict", [](gnss_gpu::PFDeviceState* state,
                                  double vx, double vy, double vz,
                                  double dt, double sigma_pos, double sigma_cb,
                                  unsigned long long seed, int step) {
        gnss_gpu::pf_device_predict(state, vx, vy, vz, dt, sigma_pos, sigma_cb, seed, step);
    }, "Predict step - operates entirely on device memory",
       py::arg("state"),
       py::arg("vx"), py::arg("vy"), py::arg("vz"),
       py::arg("dt"), py::arg("sigma_pos"), py::arg("sigma_cb"),
       py::arg("seed"), py::arg("step"));

    m.def("pf_device_weight", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef,
                                 py::array_t<double> pseudoranges,
                                 py::array_t<double> weights_sat,
                                 int n_sat, double sigma_pr) {
        gnss_gpu::pf_device_weight(state,
            static_cast<double*>(sat_ecef.request().ptr),
            static_cast<double*>(pseudoranges.request().ptr),
            static_cast<double*>(weights_sat.request().ptr),
            n_sat, sigma_pr);
    }, "Weight update - only satellite data transferred to device",
       py::arg("state"),
       py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights_sat"),
       py::arg("n_sat"), py::arg("sigma_pr"));

    m.def("pf_device_ess", [](const gnss_gpu::PFDeviceState* state) {
        return gnss_gpu::pf_device_ess(state);
    }, "Compute ESS on device, return scalar to host",
       py::arg("state"));

    m.def("pf_device_resample_systematic", [](gnss_gpu::PFDeviceState* state,
                                              unsigned long long seed) {
        gnss_gpu::pf_device_resample_systematic(state, seed);
    }, "Systematic resampling - operates entirely on device",
       py::arg("state"), py::arg("seed"));

    m.def("pf_device_resample_megopolis", [](gnss_gpu::PFDeviceState* state,
                                             int n_iterations,
                                             unsigned long long seed) {
        gnss_gpu::pf_device_resample_megopolis(state, n_iterations, seed);
    }, "Megopolis resampling - operates entirely on device",
       py::arg("state"), py::arg("n_iterations"), py::arg("seed"));

    m.def("pf_device_estimate", [](const gnss_gpu::PFDeviceState* state) {
        double result[4];
        gnss_gpu::pf_device_estimate(state, result);
        return py::array_t<double>({4}, {sizeof(double)}, result);
    }, "Compute weighted mean on device, return [4] to host",
       py::arg("state"));

    m.def("pf_device_get_particles", [](const gnss_gpu::PFDeviceState* state) {
        int N = state->n_particles;
        auto output = py::array_t<double>({N, 4});
        gnss_gpu::pf_device_get_particles(state,
            static_cast<double*>(output.request().ptr));
        return output;
    }, "Copy particles to host for visualization (only when needed)",
       py::arg("state"));

    m.def("pf_device_sync", [](gnss_gpu::PFDeviceState* state) {
        gnss_gpu::pf_device_sync(state);
    }, "Synchronize CUDA stream - wait for all pending operations to complete",
       py::arg("state"));
}
