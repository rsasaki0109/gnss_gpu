#pragma once

namespace gnss_gpu {

// Double-difference RTK positioning
// base_ecef: [3] base station ECEF position (known)
// rover_pr: [n_sat] rover pseudoranges
// base_pr: [n_sat] base pseudoranges
// rover_carrier: [n_sat] rover carrier phase observations [cycles]
// base_carrier: [n_sat] base carrier phase observations [cycles]
// sat_ecef: [n_sat * 3] satellite positions
// result: [3] output rover ECEF position
// ambiguities: [n_sat-1] output float ambiguities (DD)
// residuals: [2*(n_sat-1)] output residuals (DD pseudorange + DD carrier)
// Returns number of iterations
int rtk_float(const double* base_ecef,
              const double* rover_pr, const double* base_pr,
              const double* rover_carrier, const double* base_carrier,
              const double* sat_ecef, double* result,
              double* ambiguities, double* residuals,
              int n_sat, double wavelength, int max_iter, double tol);

// Batch RTK (GPU parallel over epochs)
// base_ecef: [3] base station ECEF position
// rover_pr: [n_epoch * n_sat]
// base_pr: [n_epoch * n_sat]
// rover_carrier: [n_epoch * n_sat]
// base_carrier: [n_epoch * n_sat]
// sat_ecef: [n_epoch * n_sat * 3]
// results: [n_epoch * 3] output rover positions
// ambiguities: [n_epoch * (n_sat-1)] output float ambiguities
// iters: [n_epoch] iterations used per epoch (optional, can be nullptr)
void rtk_float_batch(
    const double* base_ecef,
    const double* rover_pr, const double* base_pr,
    const double* rover_carrier, const double* base_carrier,
    const double* sat_ecef, double* results,
    double* ambiguities, int* iters,
    int n_epoch, int n_sat, double wavelength,
    int max_iter, double tol);

// LAMBDA integer ambiguity resolution
// float_amb: [n] float ambiguities
// Q_amb: [n * n] ambiguity covariance matrix
// fixed_amb: [n] output integer ambiguities
// n_candidates: number of candidates to evaluate
// Returns ratio of second-best to best candidate (ratio test value)
double lambda_integer(const double* float_amb, const double* Q_amb,
                      int* fixed_amb, int n, int n_candidates);

}  // namespace gnss_gpu
