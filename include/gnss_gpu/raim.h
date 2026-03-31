#pragma once

namespace gnss_gpu {

struct RAIMResult {
    bool integrity_ok;     // true if position is trustworthy
    double hpl;            // Horizontal Protection Level [m]
    double vpl;            // Vertical Protection Level [m]
    double test_statistic; // chi-squared test statistic
    double threshold;      // detection threshold
    int excluded_sat;      // -1 if none, satellite index if excluded
};

// RAIM chi-squared consistency check after WLS solution.
// sat_ecef: [n_sat * 3] satellite ECEF positions
// pseudoranges: [n_sat] observed pseudoranges
// weights: [n_sat] observation weights (1/sigma^2)
// position: [4] WLS solution (x, y, z, clock_bias)
// result: output RAIM result
// n_sat: number of satellites
// p_fa: probability of false alarm (e.g. 1e-5 for aviation)
void raim_check(const double* sat_ecef, const double* pseudoranges,
                const double* weights, const double* position,
                RAIMResult* result, int n_sat, double p_fa);

// RAIM with Fault Detection and Exclusion.
// If the chi-squared test fails, iteratively excludes each satellite,
// re-solves WLS, and selects the exclusion that yields the lowest SSE.
// position: [4] input initial WLS solution, overwritten with best solution on exclusion
void raim_fde(const double* sat_ecef, const double* pseudoranges,
              const double* weights, double* position,
              RAIMResult* result, int n_sat, double p_fa);

}  // namespace gnss_gpu
