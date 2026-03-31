#pragma once

namespace gnss_gpu {

struct EKFState {
    double x[8];    // [x, y, z, vx, vy, vz, cb, cd] (pos, vel, clock bias, clock drift)
    double P[64];   // 8x8 covariance matrix (row-major)
};

struct EKFConfig {
    double sigma_pos;    // position process noise [m/s²] (e.g., 1.0)
    double sigma_vel;    // velocity process noise [m/s³] (e.g., 0.1)
    double sigma_clk;    // clock bias process noise [m/s] (e.g., 100.0)
    double sigma_drift;  // clock drift process noise [m/s²] (e.g., 10.0)
    double sigma_pr;     // pseudorange measurement noise [m] (e.g., 5.0)
};

// Initialize EKF state from WLS position
void ekf_initialize(EKFState* state, const double* initial_pos,
                    double initial_cb, double sigma_pos, double sigma_cb);

// EKF predict step (constant velocity + clock drift model)
void ekf_predict(EKFState* state, double dt, const EKFConfig& config);

// EKF update step with pseudorange measurements
// sat_ecef: [n_sat * 3] satellite ECEF positions
// pseudoranges: [n_sat] observed pseudoranges
// weights: [n_sat] measurement weights (1/sigma^2)
void ekf_update(EKFState* state, const double* sat_ecef,
                const double* pseudoranges, const double* weights,
                int n_sat);

// GPU batch EKF: process multiple independent EKF instances in parallel
// Useful for: testing, multi-receiver processing, Monte Carlo simulation
void ekf_batch(EKFState* states, const double* sat_ecef,
               const double* pseudoranges, const double* weights,
               double dt, const EKFConfig& config,
               int n_instances, int n_sat);

}  // namespace gnss_gpu
