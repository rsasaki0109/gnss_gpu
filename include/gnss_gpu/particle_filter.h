#pragma once

namespace gnss_gpu {

struct ParticleFilterConfig {
    int n_particles;           // e.g., 1000000
    double sigma_pos;          // position noise std [m] (e.g., 1.0)
    double sigma_cb;           // clock bias noise std [m] (e.g., 300.0)
    double sigma_pr;           // pseudorange observation std [m] (e.g., 5.0)
    double ess_threshold;      // ESS ratio for resampling (e.g., 0.5)
    int resampling_method;     // 0=systematic, 1=megopolis
};

// Initialize particles around initial position
void pf_initialize(
    double* px, double* py, double* pz, double* pcb,  // [N] particle states (SoA)
    double init_x, double init_y, double init_z, double init_cb,
    double spread_pos, double spread_cb,
    int n_particles, unsigned long long seed);

// Predict step: constant velocity motion model + noise
void pf_predict(
    double* px, double* py, double* pz, double* pcb,
    const double* vx, const double* vy, const double* vz,  // velocity per particle or single
    double dt, double sigma_pos, double sigma_cb,
    int n_particles, unsigned long long seed, int step);

// Weight update: pseudorange likelihood
void pf_weight(
    const double* px, const double* py, const double* pz, const double* pcb,
    const double* sat_ecef,      // [n_sat * 3]
    const double* pseudoranges,  // [n_sat]
    const double* weights_sat,   // [n_sat] per-satellite weights (1/sigma^2)
    double* log_weights,         // [N] output log-weights
    int n_particles, int n_sat, double sigma_pr);

// Compute ESS (Effective Sample Size)
double pf_compute_ess(const double* log_weights, int n_particles);

// Systematic resampling using prefix-sum CDF
void pf_resample_systematic(
    double* px, double* py, double* pz, double* pcb,
    const double* log_weights,
    int n_particles, unsigned long long seed);

// Megopolis resampling (preferred for large N)
void pf_resample_megopolis(
    double* px, double* py, double* pz, double* pcb,
    const double* log_weights,
    int n_particles, int n_iterations,
    unsigned long long seed);

// Weighted mean estimate
void pf_estimate(
    const double* px, const double* py, const double* pz, const double* pcb,
    const double* log_weights,
    double* result,  // [4] output (x, y, z, cb)
    int n_particles);

// Get particle positions for visualization
void pf_get_particles(
    const double* px, const double* py, const double* pz, const double* pcb,
    double* output,  // [N * 4] output interleaved
    int n_particles);

}  // namespace gnss_gpu
