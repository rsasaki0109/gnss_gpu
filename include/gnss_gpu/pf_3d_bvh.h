#pragma once
#include "gnss_gpu/bvh.h"

namespace gnss_gpu {

/// BVH-accelerated 3D-aware weight kernel.
///
/// Identical semantics to pf_weight_3d() but uses a pre-built BVH to perform
/// O(log n) ray-triangle intersection instead of the O(n) linear scan.  This
/// makes the kernel practical for urban meshes with tens of thousands of
/// triangles.
///
/// The BVH must have been built with bvh_build() and the triangles array must
/// be the reordered (sorted) triangle array produced by the same call.
///
/// @param px, py, pz, pcb  [N] particle states (SoA layout)
/// @param sat_ecef          [n_sat * 3] satellite ECEF positions
/// @param pseudoranges      [n_sat] observed pseudoranges [m]
/// @param weights_sat       [n_sat] per-satellite weights (e.g., elevation-based)
/// @param bvh_nodes         [n_nodes] flat BVH node array from bvh_build()
/// @param n_nodes           number of BVH nodes
/// @param sorted_tris       [n_tri] triangles reordered by bvh_build()
/// @param n_tri             number of triangles
/// @param log_weights       [N] output log-weights
/// @param n_particles       number of particles
/// @param n_sat             number of satellites
/// @param sigma_pr_los      sigma for LOS satellites [m] (tight, e.g., 3 m)
/// @param sigma_pr_nlos     sigma for NLOS satellites [m] (loose, e.g., 30 m)
/// @param nlos_bias         expected positive bias for NLOS [m] (applied only
///                          when the residual itself is positive)
/// @param blocked_nlos_prob P(NLOS | ray blocked), 0..1
/// @param clear_nlos_prob   P(NLOS | ray clear), 0..1
void pf_weight_3d_bvh(
    const double* px, const double* py, const double* pz, const double* pcb,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    const BVHNode* bvh_nodes, int n_nodes,
    const Triangle* sorted_tris, int n_tri,
    double* log_weights,
    int n_particles, int n_sat,
    double sigma_pr_los,
    double sigma_pr_nlos,
    double nlos_bias,
    double blocked_nlos_prob,
    double clear_nlos_prob);

}  // namespace gnss_gpu
