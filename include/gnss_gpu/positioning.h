#pragma once

namespace gnss_gpu {

// WLS single-epoch positioning
// sat_ecef: [n_sat * 3] satellite ECEF positions
// pseudoranges: [n_sat] observed pseudoranges
// weights: [n_sat] observation weights (1/sigma^2)
// result: [4] output (x, y, z, clock_bias) in ECEF [m]
// returns number of iterations used
int wls_position(const double* sat_ecef, const double* pseudoranges,
                 const double* weights, double* result,
                 int n_sat, int max_iter = 10, double tol = 1e-4);

// WLS batch positioning (GPU parallel)
// sat_ecef: [n_epoch * n_sat * 3]
// pseudoranges: [n_epoch * n_sat]
// weights: [n_epoch * n_sat]
// results: [n_epoch * 4] output
// iters: [n_epoch] iterations used per epoch (optional, can be nullptr)
void wls_batch(const double* sat_ecef, const double* pseudoranges,
               const double* weights, double* results, int* iters,
               int n_epoch, int n_sat, int max_iter = 10, double tol = 1e-4);

}  // namespace gnss_gpu
