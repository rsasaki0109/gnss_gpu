#pragma once

namespace gnss_gpu {

// Estimate RBF kernel bandwidth using median heuristic on a random subsample.
// Returns h such that k(x,y) = exp(-||x-y||^2 / (2*h^2))
double pf_estimate_bandwidth(
    const double* px, const double* py, const double* pz, const double* pcb,
    int n_particles, int n_subsample,
    unsigned long long seed);

// Perform one SVGD step: compute phi(x_i) and update particles.
// Uses K random neighbors per particle for O(N*K) complexity.
//
// The SVGD update rule:
//   phi(x_i) = (1/K) sum_j [ k(x_j, x_i) * grad_xj log p(x_j|y) + grad_xj k(x_j, x_i) ]
//   x_i <- x_i + step_size * phi(x_i)
//
// where the score function grad log p(x|y) is computed from pseudorange likelihood.
void pf_svgd_step(
    double* px, double* py, double* pz, double* pcb,
    const double* sat_ecef,        // [n_sat * 3]
    const double* pseudoranges,    // [n_sat]
    const double* weights_sat,     // [n_sat] per-satellite weights (1/sigma^2)
    int n_particles, int n_sat,
    double sigma_pr, double step_size,
    int n_neighbors,               // K random neighbors for kernel
    double bandwidth,              // RBF kernel bandwidth h
    unsigned long long seed, int step);

// Simple mean estimate (equal weights after SVGD, no log_weights needed)
void pf_svgd_estimate(
    const double* px, const double* py, const double* pz, const double* pcb,
    double* result,  // [4] output (x, y, z, cb)
    int n_particles);

}  // namespace gnss_gpu
