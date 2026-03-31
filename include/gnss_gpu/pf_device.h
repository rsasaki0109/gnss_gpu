#pragma once
#include <cuda_runtime.h>

namespace gnss_gpu {

// Opaque handle to device-resident particle state
struct PFDeviceState {
    double* d_px;
    double* d_py;
    double* d_pz;
    double* d_pcb;
    double* d_log_weights;

    // Double-buffer for resampling (persistent to avoid realloc)
    double* d_px_tmp;
    double* d_py_tmp;
    double* d_pz_tmp;
    double* d_pcb_tmp;

    // Persistent temp buffers for reductions (ESS, estimate)
    // Sized to max grid blocks needed
    double* d_partial_a;  // grid * 4 for estimate, grid for ESS
    double* d_partial_b;  // grid for sum_w / sum_w2
    double* d_partial_c;  // grid for max_lw

    // For systematic resampling
    double* d_weights_norm;  // [N] normalized weights
    double* d_cdf;           // [N] CDF

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
    unsigned long long seed);

// Predict - operates entirely on device memory
void pf_device_predict(PFDeviceState* state,
    double vx, double vy, double vz,
    double dt, double sigma_pos, double sigma_cb,
    unsigned long long seed, int step);

// Weight update - satellite data is small, only that gets H2D copied
void pf_device_weight(PFDeviceState* state,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    int n_sat, double sigma_pr);

// ESS - compute on device, return scalar to host
double pf_device_ess(const PFDeviceState* state);

// Resample - operates entirely on device
void pf_device_resample_systematic(PFDeviceState* state, unsigned long long seed);
void pf_device_resample_megopolis(PFDeviceState* state, int n_iterations, unsigned long long seed);

// Estimate - compute weighted mean on device, return [4] to host
void pf_device_estimate(const PFDeviceState* state, double* result);

// Copy particles to host for visualization (only when needed)
void pf_device_get_particles(const PFDeviceState* state, double* output);

// Explicit synchronization - wait for all stream operations to complete
void pf_device_sync(PFDeviceState* state);

}  // namespace gnss_gpu
