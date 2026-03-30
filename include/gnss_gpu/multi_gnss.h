#pragma once

namespace gnss_gpu {

enum GNSSSystem { GPS = 0, GLONASS = 1, GALILEO = 2, BEIDOU = 3, QZSS = 4, MAX_SYSTEMS = 5 };

// Multi-GNSS WLS: estimates position + per-system clock bias (ISB)
// sat_ecef: [n_sat * 3] satellite ECEF positions
// pseudoranges: [n_sat] observed pseudoranges
// weights: [n_sat] observation weights (1/sigma^2)
// system_ids: [n_sat] system identifier for each satellite (0=GPS, 1=GLO, etc.)
// result: [3 + n_systems] output (x, y, z, cb_sys0, cb_sys1, ...)
// n_systems: number of distinct clock systems to estimate
// returns number of iterations used
int wls_multi_gnss(const double* sat_ecef, const double* pseudoranges,
                   const double* weights, const int* system_ids,
                   double* result, int n_sat, int n_systems,
                   int max_iter = 10, double tol = 1e-4);

// GPU batch version
// sat_ecef: [n_epoch * n_sat * 3]
// pseudoranges: [n_epoch * n_sat]
// weights: [n_epoch * n_sat]
// system_ids: [n_epoch * n_sat]
// results: [n_epoch * (3 + n_systems)]
// iters: [n_epoch] iterations used per epoch (optional, can be nullptr)
void wls_multi_gnss_batch(const double* sat_ecef, const double* pseudoranges,
                          const double* weights, const int* system_ids,
                          double* results, int* iters,
                          int n_epoch, int n_sat, int n_systems,
                          int max_iter = 10, double tol = 1e-4);

}  // namespace gnss_gpu
