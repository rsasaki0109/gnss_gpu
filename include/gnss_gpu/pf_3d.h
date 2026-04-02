#pragma once
#include "gnss_gpu/raytrace.h"

namespace gnss_gpu {

/// 3D-aware weight kernel: uses ray tracing to determine LOS/NLOS per particle.
///
/// For each particle, casts rays to every satellite through the building mesh.
/// LOS satellites use a tight sigma; NLOS satellites use a loose sigma with
/// a positive-only bias correction for the expected multipath delay. The
/// ray-tracing result can also be treated as a soft prior on the LOS/NLOS
/// mixture instead of a hard switch.
///
/// @param px, py, pz, pcb  [N] particle states (SoA layout)
/// @param sat_ecef          [n_sat * 3] satellite ECEF positions
/// @param pseudoranges      [n_sat] observed pseudoranges [m]
/// @param weights_sat       [n_sat] per-satellite weights (e.g., elevation-based)
/// @param triangles         [n_tri] building mesh triangles
/// @param n_tri             number of triangles
/// @param log_weights       [N] output log-weights
/// @param n_particles       number of particles
/// @param n_sat             number of satellites
/// @param sigma_pr_los      sigma for LOS satellites [m] (tight, e.g., 3 m)
/// @param sigma_pr_nlos     sigma for NLOS satellites [m] (loose, e.g., 30 m)
/// @param nlos_bias         expected positive bias for NLOS [m] (e.g., 20 m)
/// @param blocked_nlos_prob P(NLOS | ray blocked), 0..1
/// @param clear_nlos_prob   P(NLOS | ray clear), 0..1
void pf_weight_3d(
    const double* px, const double* py, const double* pz, const double* pcb,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    const Triangle* triangles, int n_tri,
    double* log_weights,
    int n_particles, int n_sat,
    double sigma_pr_los,
    double sigma_pr_nlos,
    double nlos_bias,
    double blocked_nlos_prob,
    double clear_nlos_prob);

}  // namespace gnss_gpu
