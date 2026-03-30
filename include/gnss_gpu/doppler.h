#pragma once

namespace gnss_gpu {

// Doppler velocity estimation (WLS)
// sat_ecef: [n_sat*3], sat_vel: [n_sat*3] (satellite velocities in ECEF)
// doppler: [n_sat] Doppler measurements [Hz]
// rx_pos: [3] known receiver ECEF position
// weights: [n_sat] observation weights (1/sigma^2)
// result: [4] output (vx, vy, vz, clock_drift) in ECEF [m/s]
// returns number of iterations used
int doppler_velocity(const double* sat_ecef, const double* sat_vel,
                     const double* doppler, const double* rx_pos,
                     const double* weights, double* result,
                     int n_sat, double wavelength, int max_iter, double tol);

// Batch Doppler velocity estimation (GPU parallel)
// sat_ecef: [n_epoch * n_sat * 3]
// sat_vel: [n_epoch * n_sat * 3]
// doppler: [n_epoch * n_sat]
// rx_pos: [n_epoch * 3] known receiver ECEF positions
// weights: [n_epoch * n_sat]
// results: [n_epoch * 4] output (vx, vy, vz, clock_drift) per epoch
// iters: [n_epoch] iterations used per epoch (optional, can be nullptr)
void doppler_velocity_batch(const double* sat_ecef, const double* sat_vel,
                            const double* doppler, const double* rx_pos,
                            const double* weights, double* results, int* iters,
                            int n_epoch, int n_sat, double wavelength,
                            int max_iter, double tol);

}  // namespace gnss_gpu
