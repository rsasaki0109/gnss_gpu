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
                                     unsigned long long seed,
                                     double init_vx, double init_vy, double init_vz,
                                     double spread_vel,
                                     double init_vel_sigma) {
        gnss_gpu::pf_device_initialize(state, init_x, init_y, init_z, init_cb,
                                       spread_pos, spread_cb, seed,
                                       init_vx, init_vy, init_vz, spread_vel,
                                       init_vel_sigma);
    }, "Initialize particles on device (no H2D copy)",
       py::arg("state"),
       py::arg("init_x"), py::arg("init_y"), py::arg("init_z"), py::arg("init_cb"),
       py::arg("spread_pos"), py::arg("spread_cb"), py::arg("seed"),
       py::arg("init_vx") = 0.0, py::arg("init_vy") = 0.0, py::arg("init_vz") = 0.0,
       py::arg("spread_vel") = 0.0,
       py::arg("init_vel_sigma") = 0.0);

    m.def("pf_device_predict", [](gnss_gpu::PFDeviceState* state,
                                  double vx, double vy, double vz,
                                  double dt, double sigma_pos, double sigma_cb,
                                  unsigned long long seed, int step,
                                  double sigma_vel,
                                  double velocity_guide_alpha,
                                  bool velocity_kf,
                                  double velocity_process_noise) {
        gnss_gpu::pf_device_predict(
            state, vx, vy, vz, dt, sigma_pos, sigma_cb, seed, step,
            sigma_vel, velocity_guide_alpha, velocity_kf,
            velocity_process_noise);
    }, "Predict step - operates entirely on device memory",
       py::arg("state"),
       py::arg("vx"), py::arg("vy"), py::arg("vz"),
       py::arg("dt"), py::arg("sigma_pos"), py::arg("sigma_cb"),
       py::arg("seed"), py::arg("step"),
       py::arg("sigma_vel") = 0.0,
       py::arg("velocity_guide_alpha") = 1.0,
       py::arg("velocity_kf") = false,
       py::arg("velocity_process_noise") = 0.0);

    m.def("pf_device_weight", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef,
                                 py::array_t<double> pseudoranges,
                                 py::array_t<double> weights_sat,
                                 int n_sat, double sigma_pr, double nu,
                                 double per_particle_nlos_threshold_m,
                                 bool per_particle_huber,
                                 double per_particle_huber_k) {
        py::buffer_info b_sat = sat_ecef.request();
        py::buffer_info b_pr = pseudoranges.request();
        py::buffer_info b_w = weights_sat.request();
        gnss_gpu::pf_device_weight(state,
            static_cast<double*>(b_sat.ptr),
            static_cast<double*>(b_pr.ptr),
            static_cast<double*>(b_w.ptr),
            n_sat, sigma_pr, nu, per_particle_nlos_threshold_m,
            per_particle_huber, per_particle_huber_k);
    }, "Weight update with optional robust Student's t or Huber likelihood",
       py::arg("state"),
       py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights_sat"),
       py::arg("n_sat"), py::arg("sigma_pr"), py::arg("nu") = 0.0,
       py::arg("per_particle_nlos_threshold_m") = 0.0,
       py::arg("per_particle_huber") = false,
       py::arg("per_particle_huber_k") = 1.5);

    m.def("pf_device_weight_dd_pseudorange", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef_k,
                                 py::array_t<double> ref_ecef,
                                 py::array_t<double> dd_pseudorange,
                                 py::array_t<double> base_range_k,
                                 py::array_t<double> base_range_ref,
                                 py::array_t<double> weights_dd,
                                 int n_dd, double sigma_pr,
                                 double per_particle_nlos_threshold_m,
                                 bool per_particle_huber,
                                 double per_particle_huber_k) {
        py::buffer_info b_sk = sat_ecef_k.request();
        py::buffer_info b_ref = ref_ecef.request();
        py::buffer_info b_dd = dd_pseudorange.request();
        py::buffer_info b_brk = base_range_k.request();
        py::buffer_info b_brr = base_range_ref.request();
        py::buffer_info b_w = weights_dd.request();
        gnss_gpu::pf_device_weight_dd_pseudorange(state,
            static_cast<double*>(b_sk.ptr),
            static_cast<double*>(b_ref.ptr),
            static_cast<double*>(b_dd.ptr),
            static_cast<double*>(b_brk.ptr),
            static_cast<double*>(b_brr.ptr),
            static_cast<double*>(b_w.ptr),
            n_dd, sigma_pr, per_particle_nlos_threshold_m,
            per_particle_huber, per_particle_huber_k);
    }, "Weight update using DD pseudorange likelihood (no clock bias needed)",
       py::arg("state"),
       py::arg("sat_ecef_k"), py::arg("ref_ecef"),
       py::arg("dd_pseudorange"), py::arg("base_range_k"),
       py::arg("base_range_ref"), py::arg("weights_dd"),
       py::arg("n_dd"), py::arg("sigma_pr"),
       py::arg("per_particle_nlos_threshold_m") = 0.0,
       py::arg("per_particle_huber") = false,
       py::arg("per_particle_huber_k") = 1.5);

    m.def("pf_device_weight_gmm", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef,
                                 py::array_t<double> pseudoranges,
                                 py::array_t<double> weights_sat,
                                 int n_sat, double sigma_pr,
                                 double w_los, double mu_nlos, double sigma_nlos) {
        py::buffer_info b_sat = sat_ecef.request();
        py::buffer_info b_pr = pseudoranges.request();
        py::buffer_info b_w = weights_sat.request();
        gnss_gpu::pf_device_weight_gmm(state,
            static_cast<double*>(b_sat.ptr),
            static_cast<double*>(b_pr.ptr),
            static_cast<double*>(b_w.ptr),
            n_sat, sigma_pr, w_los, mu_nlos, sigma_nlos);
    }, "Weight update using GMM likelihood (LOS + NLOS mixture)",
       py::arg("state"),
       py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights_sat"),
       py::arg("n_sat"), py::arg("sigma_pr"),
       py::arg("w_los") = 0.7, py::arg("mu_nlos") = 15.0, py::arg("sigma_nlos") = 30.0);

    m.def("pf_device_weight_carrier_afv", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef,
                                 py::array_t<double> carrier_phase,
                                 py::array_t<double> weights_sat,
                                 int n_sat, double wavelength, double sigma_cycles) {
        py::buffer_info b_sat = sat_ecef.request();
        py::buffer_info b_cp = carrier_phase.request();
        py::buffer_info b_w = weights_sat.request();
        gnss_gpu::pf_device_weight_carrier_afv(state,
            static_cast<double*>(b_sat.ptr),
            static_cast<double*>(b_cp.ptr),
            static_cast<double*>(b_w.ptr),
            n_sat, wavelength, sigma_cycles);
    }, "Weight update using carrier phase AFV likelihood (no ambiguity resolution needed)",
       py::arg("state"),
       py::arg("sat_ecef"), py::arg("carrier_phase"), py::arg("weights_sat"),
       py::arg("n_sat"), py::arg("wavelength") = 0.190293673, py::arg("sigma_cycles") = 0.05);

    m.def("pf_device_weight_dd_carrier_afv", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef_k,
                                 py::array_t<double> ref_ecef,
                                 py::array_t<double> dd_carrier,
                                 py::array_t<double> base_range_k,
                                 py::array_t<double> base_range_ref,
                                 py::array_t<double> weights_dd,
                                 py::array_t<double> wavelengths_m,
                                 int n_dd, double sigma_cycles,
                                 double per_particle_nlos_threshold_cycles,
                                 bool per_particle_huber,
                                 double per_particle_huber_k) {
        py::buffer_info b_sk = sat_ecef_k.request();
        py::buffer_info b_ref = ref_ecef.request();
        py::buffer_info b_dd = dd_carrier.request();
        py::buffer_info b_brk = base_range_k.request();
        py::buffer_info b_brr = base_range_ref.request();
        py::buffer_info b_w = weights_dd.request();
        py::buffer_info b_wl = wavelengths_m.request();
        gnss_gpu::pf_device_weight_dd_carrier_afv(state,
            static_cast<double*>(b_sk.ptr),
            static_cast<double*>(b_ref.ptr),
            static_cast<double*>(b_dd.ptr),
            static_cast<double*>(b_brk.ptr),
            static_cast<double*>(b_brr.ptr),
            static_cast<double*>(b_w.ptr),
            static_cast<double*>(b_wl.ptr),
            n_dd, sigma_cycles, per_particle_nlos_threshold_cycles,
            per_particle_huber, per_particle_huber_k);
    }, "Weight update using DD carrier phase AFV (no clock bias needed)",
       py::arg("state"),
       py::arg("sat_ecef_k"), py::arg("ref_ecef"),
       py::arg("dd_carrier"), py::arg("base_range_k"),
       py::arg("base_range_ref"), py::arg("weights_dd"),
       py::arg("wavelengths_m"),
       py::arg("n_dd"), py::arg("sigma_cycles") = 0.05,
       py::arg("per_particle_nlos_threshold_cycles") = 0.0,
       py::arg("per_particle_huber") = false,
       py::arg("per_particle_huber_k") = 1.5);

    m.def("pf_device_weight_doppler", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef,
                                 py::array_t<double> sat_vel,
                                 py::array_t<double> doppler_hz,
                                 py::array_t<double> weights_sat,
                                 int n_sat, double wavelength_m,
                                 double sigma_mps,
                                 double velocity_update_gain,
                                 double max_velocity_update_mps) {
        py::buffer_info b_sat = sat_ecef.request();
        py::buffer_info b_sat_vel = sat_vel.request();
        py::buffer_info b_doppler = doppler_hz.request();
        py::buffer_info b_weights = weights_sat.request();
        gnss_gpu::pf_device_weight_doppler(state,
            static_cast<double*>(b_sat.ptr),
            static_cast<double*>(b_sat_vel.ptr),
            static_cast<double*>(b_doppler.ptr),
            static_cast<double*>(b_weights.ptr),
            n_sat, wavelength_m, sigma_mps,
            velocity_update_gain, max_velocity_update_mps);
    }, "Velocity-domain update using Doppler observations (range-rate = -doppler * wavelength)",
       py::arg("state"),
       py::arg("sat_ecef"), py::arg("sat_vel"),
       py::arg("doppler_hz"), py::arg("weights_sat"),
       py::arg("n_sat"),
       py::arg("wavelength_m") = 0.19029367279836488,
       py::arg("sigma_mps") = 0.5,
       py::arg("velocity_update_gain") = 0.25,
       py::arg("max_velocity_update_mps") = 10.0);

    m.def("pf_device_position_update", [](gnss_gpu::PFDeviceState* state,
                                         double ref_x, double ref_y, double ref_z,
                                         double sigma_pos) {
        gnss_gpu::pf_device_position_update(state, ref_x, ref_y, ref_z, sigma_pos);
    }, "Apply position-domain soft constraint from external estimate",
       py::arg("state"),
       py::arg("ref_x"), py::arg("ref_y"), py::arg("ref_z"),
       py::arg("sigma_pos"));

    m.def("pf_device_shift_clock_bias", [](gnss_gpu::PFDeviceState* state,
                                           double shift) {
        gnss_gpu::pf_device_shift_clock_bias(state, shift);
    }, "Shift all particles' clock bias by a constant offset",
       py::arg("state"), py::arg("shift"));

    m.def("pf_device_ess", [](const gnss_gpu::PFDeviceState* state) {
        return gnss_gpu::pf_device_ess(state);
    }, "Compute ESS on device, return scalar to host",
       py::arg("state"));

    m.def("pf_device_position_spread", [](const gnss_gpu::PFDeviceState* state,
                                          double center_x, double center_y, double center_z) {
        return gnss_gpu::pf_device_position_spread(state, center_x, center_y, center_z);
    }, "Compute weighted RMS position spread around a reference point",
       py::arg("state"), py::arg("center_x"), py::arg("center_y"), py::arg("center_z"));

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

    m.def("pf_device_get_particle_states", [](const gnss_gpu::PFDeviceState* state) {
        int N = state->n_particles;
        auto output = py::array_t<double>({N, 16});
        gnss_gpu::pf_device_get_particle_states(state,
            static_cast<double*>(output.request().ptr));
        return output;
    }, "Copy full particle states [x,y,z,cb,mu_vx,mu_vy,mu_vz,Sigma_v(3x3)] to host",
       py::arg("state"));

    m.def("pf_device_get_log_weights", [](const gnss_gpu::PFDeviceState* state,
                                         py::array_t<double> out) {
        int N = state->n_particles;
        auto r = out.request();
        if (r.size != N) {
            throw std::runtime_error("pf_device_get_log_weights: output size must match n_particles");
        }
        gnss_gpu::pf_device_get_log_weights(state, static_cast<double*>(r.ptr));
    }, "Copy log-weights to host (D2H, synchronizes stream)",
       py::arg("state"), py::arg("out"));

    m.def("pf_device_get_resample_ancestors", [](const gnss_gpu::PFDeviceState* state,
                                                  py::array_t<int> out) {
        int N = state->n_particles;
        auto r = out.request();
        if (r.size != N) {
            throw std::runtime_error(
                "pf_device_get_resample_ancestors: output size must match n_particles");
        }
        gnss_gpu::pf_device_get_resample_ancestors(state, static_cast<int*>(r.ptr));
    }, "After systematic resample: copy ancestor indices out[j]=source (D2H, sync)",
       py::arg("state"), py::arg("out"));

    m.def("pf_device_sync", [](gnss_gpu::PFDeviceState* state) {
        gnss_gpu::pf_device_sync(state);
    }, "Synchronize CUDA stream - wait for all pending operations to complete",
       py::arg("state"));
}
