#pragma once
#include <cuda_runtime.h>

namespace gnss_gpu {

// Opaque handle to device-resident particle state
struct PFDeviceState {
    double* d_px;
    double* d_py;
    double* d_pz;
    double* d_vx;
    double* d_vy;
    double* d_vz;
    double* d_pcb;
    double* d_log_weights;

    // Double-buffer for resampling (persistent to avoid realloc)
    double* d_px_tmp;
    double* d_py_tmp;
    double* d_pz_tmp;
    double* d_vx_tmp;
    double* d_vy_tmp;
    double* d_vz_tmp;
    double* d_pcb_tmp;

    // Persistent temp buffers for reductions (ESS, estimate)
    // Sized to max grid blocks needed
    double* d_partial_a;  // grid * 4 for estimate, grid for ESS
    double* d_partial_b;  // grid for sum_w / sum_w2
    double* d_partial_c;  // grid for max_lw

    // For systematic resampling
    double* d_weights_norm;  // [N] normalized weights
    double* d_cdf;           // [N] CDF
    int* d_resample_ancestor;  // [N] last systematic resample: out[j]=source index i

    // Velocity buffer (3 doubles, persistent)
    double* d_vel;

    // CUDA stream for pipelined execution
    cudaStream_t stream;

    // Persistent device buffers for satellite data (avoids per-call cudaMalloc)
    double* d_sat_ecef;     // [pinned_capacity * 3]
    double* d_pseudoranges; // [pinned_capacity]
    double* d_weights_sat;  // [pinned_capacity]

    // Pinned host memory for async H2D transfers
    double* h_sat_pinned;    // pinned memory for satellite data
    double* h_result_pinned; // pinned memory for estimate result (4 doubles)
    // Scratch for ESS / estimate / systematic-resample CPU reductions (6 * grid doubles)
    double* h_reduction_pinned;
    int pinned_capacity;     // max satellites supported without realloc

    int n_particles;
    int grid_size;  // precomputed (n_particles + 255) / 256
    bool allocated;
};

// Lifecycle management
PFDeviceState* pf_device_create(int n_particles);
void pf_device_destroy(PFDeviceState* state);

// Free GPU resources without deleting the struct itself.
// Used by pybind11 custom holder to avoid double-free.
void pf_device_destroy_resources(PFDeviceState* state);

// Initialize particles on device (no H2D copy needed after this)
void pf_device_initialize(PFDeviceState* state,
    double init_x, double init_y, double init_z, double init_cb,
    double spread_pos, double spread_cb,
    unsigned long long seed,
    double init_vx = 0.0, double init_vy = 0.0, double init_vz = 0.0,
    double spread_vel = 0.0);

// Predict - operates entirely on device memory
void pf_device_predict(PFDeviceState* state,
    double vx, double vy, double vz,
    double dt, double sigma_pos, double sigma_cb,
    unsigned long long seed, int step,
    double sigma_vel = 0.0,
    double velocity_guide_alpha = 1.0);

// Weight update - satellite data is small, only that gets H2D copied
// nu: Student's t degrees of freedom. nu=0 means Gaussian (default).
//     nu=1 is Cauchy (most robust). nu=3-5 is moderately robust.
void pf_device_weight(PFDeviceState* state,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    int n_sat, double sigma_pr, double nu = 0.0,
    double per_particle_nlos_threshold_m = 0.0,
    bool per_particle_huber = false,
    double per_particle_huber_k = 1.5);

// Weight update using double-differenced pseudorange.
// Eliminates receiver clock bias; uses satellite differencing geometry.
// sat_ecef_k: [n_dd, 3] non-ref satellite positions
// ref_ecef: [n_dd, 3] reference satellite positions per DD pair
// dd_pseudorange: [n_dd] DD pseudorange observations in meters
// base_range_k: [n_dd] base-to-sat_k ranges [m]
// base_range_ref: [n_dd] base-to-ref ranges [m]
// weights_dd: [n_dd] per-DD-pair weights
void pf_device_weight_dd_pseudorange(PFDeviceState* state,
    const double* sat_ecef_k, const double* ref_ecef,
    const double* dd_pseudorange, const double* base_range_k,
    const double* base_range_ref, const double* weights_dd,
    int n_dd, double sigma_pr,
    double per_particle_nlos_threshold_m = 0.0,
    bool per_particle_huber = false,
    double per_particle_huber_k = 1.5);

// Weight update using GMM likelihood (LOS + NLOS mixture components)
// w_los: weight of LOS component (e.g. 0.7), sigma_pr is sigma_los
// mu_nlos: mean bias of NLOS component [m], sigma_nlos: std of NLOS component [m]
void pf_device_weight_gmm(PFDeviceState* state,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    int n_sat, double sigma_pr,
    double w_los = 0.7, double mu_nlos = 15.0, double sigma_nlos = 30.0);

// Weight update using carrier phase AFV (Ambiguity Function Value) likelihood.
// Uses fractional cycle residuals — no integer ambiguity resolution needed.
// Call AFTER pf_device_weight (pseudorange) for the MUPF algorithm.
// carrier_phase is in cycles, wavelength in meters, sigma_cycles in cycles.
void pf_device_weight_carrier_afv(PFDeviceState* state,
    const double* sat_ecef, const double* carrier_phase,
    const double* weights_sat,
    int n_sat, double wavelength = 0.190293673, double sigma_cycles = 0.05);

// Weight update using DD carrier phase AFV (Double-Differenced).
// Eliminates receiver clock bias; uses satellite differencing geometry.
// sat_ecef_k: [n_dd, 3] non-ref satellite positions
// ref_ecef: [n_dd, 3] reference satellite positions per DD pair
// dd_carrier: [n_dd] DD carrier phase observations in cycles
// base_range_k: [n_dd] base-to-sat_k ranges [m]
// base_range_ref: [n_dd] base-to-ref ranges [m]
// weights_dd: [n_dd] per-DD-pair weights
// wavelengths_m: [n_dd] carrier wavelengths [m]
void pf_device_weight_dd_carrier_afv(PFDeviceState* state,
    const double* sat_ecef_k, const double* ref_ecef,
    const double* dd_carrier, const double* base_range_k,
    const double* base_range_ref, const double* weights_dd,
    const double* wavelengths_m,
    int n_dd, double sigma_cycles = 0.05,
    double per_particle_nlos_threshold_cycles = 0.0,
    bool per_particle_huber = false,
    double per_particle_huber_k = 1.5);

// Velocity-domain update using Doppler observations.
// sat_ecef: [n_sat, 3] satellite positions [m]
// sat_vel: [n_sat, 3] satellite velocities [m/s]
// doppler_hz: [n_sat] Doppler observations [Hz]
// weights_sat: [n_sat] per-observation weights
// sigma_mps is Doppler range-rate noise [m/s].
// velocity_update_gain blends each particle velocity toward its per-particle
// Doppler WLS solution; set to 0 to apply likelihood only.
void pf_device_weight_doppler(PFDeviceState* state,
    const double* sat_ecef, const double* sat_vel,
    const double* doppler_hz, const double* weights_sat,
    int n_sat, double wavelength_m = 0.19029367279836488,
    double sigma_mps = 0.5,
    double velocity_update_gain = 0.25,
    double max_velocity_update_mps = 10.0);

// Position-domain update - apply soft constraint from external position estimate
void pf_device_position_update(PFDeviceState* state,
    double ref_x, double ref_y, double ref_z, double sigma_pos);

// Shift all particles' clock bias by a constant offset
void pf_device_shift_clock_bias(PFDeviceState* state, double shift);

// ESS - compute on device, return scalar to host
double pf_device_ess(const PFDeviceState* state);

// Position spread - weighted RMS distance from a reference position [m].
double pf_device_position_spread(
    const PFDeviceState* state,
    double center_x, double center_y, double center_z);

// Resample - operates entirely on device
void pf_device_resample_systematic(PFDeviceState* state, unsigned long long seed);
void pf_device_resample_megopolis(PFDeviceState* state, int n_iterations, unsigned long long seed);

// Estimate - compute weighted mean on device, return [4] to host
void pf_device_estimate(const PFDeviceState* state, double* result);

// Copy particles to host for visualization (only when needed)
void pf_device_get_particles(const PFDeviceState* state, double* output);

// Copy full particle states to host: [x, y, z, vx, vy, vz, cb].
// Synchronizes the stream.
void pf_device_get_particle_states(const PFDeviceState* state, double* output);

// Copy log-weights to host (for FFBSi / diagnostics). Synchronizes the stream.
void pf_device_get_log_weights(const PFDeviceState* state, double* output);

// Copy last systematic-resampling ancestor indices to host (out[j] = source idx).
// Only valid after pf_device_resample_systematic; synchronizes the stream.
void pf_device_get_resample_ancestors(const PFDeviceState* state, int* output);

// Explicit synchronization - wait for all stream operations to complete
void pf_device_sync(PFDeviceState* state);

}  // namespace gnss_gpu
