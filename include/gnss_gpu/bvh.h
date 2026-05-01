#pragma once

#include "gnss_gpu/raytrace.h"

namespace gnss_gpu {

struct AABB {
  double min[3];
  double max[3];
};

struct BVHNode {
  AABB bbox;
  int left;       // left child index (-1 if leaf)
  int right;      // right child index (-1 if leaf)
  int tri_start;  // first triangle index in sorted array (for leaf)
  int tri_count;  // number of triangles (for leaf, 0 for internal)
};

// Build BVH on CPU from triangles using SAH (Surface Area Heuristic).
// Outputs a flat array of BVH nodes and a reordered triangle index array.
// Caller must allocate nodes (2*n_tri - 1 max) and sorted_tri_indices (n_tri).
void bvh_build(const Triangle* triangles, int n_tri,
               BVHNode* nodes, int* n_nodes, int* sorted_tri_indices);

// LOS check using BVH traversal on GPU.
// sorted_tris must be triangles reordered by sorted_tri_indices from bvh_build.
void raytrace_los_check_bvh(const double* rx_ecef, const double* sat_ecef,
                             const BVHNode* bvh, const Triangle* sorted_tris,
                             int* is_los, int n_sat, int n_nodes);

// Multipath reflection computation using BVH traversal on GPU.
// For each satellite, finds the first-order reflection with minimum excess delay.
void raytrace_multipath_bvh(const double* rx_ecef, const double* sat_ecef,
                             const BVHNode* bvh, const Triangle* sorted_tris,
                             double* reflection_points, double* excess_delays,
                             int n_sat, int n_nodes);

// Batched LOS check across N epochs sharing one BVH.
// rx_ecef: N x 3 receiver positions, sat_ecef: N x n_sat x 3 satellite positions.
// is_los: N x n_sat int output (1 = LOS clear, 0 = blocked).
// Single CUDA launch over n_epoch * n_sat threads; BVH is uploaded once.
void raytrace_los_check_bvh_batch(const double* rx_ecef, const double* sat_ecef,
                                   const BVHNode* bvh, const Triangle* sorted_tris,
                                   int* is_los, int n_epoch, int n_sat, int n_nodes);

// Batched multipath reflection across N epochs sharing one BVH.
// rx_ecef: N x 3, sat_ecef: N x n_sat x 3.
// reflection_points: N x n_sat x 3, excess_delays: N x n_sat.
void raytrace_multipath_bvh_batch(const double* rx_ecef, const double* sat_ecef,
                                   const BVHNode* bvh, const Triangle* sorted_tris,
                                   double* reflection_points, double* excess_delays,
                                   int n_epoch, int n_sat, int n_nodes);

}  // namespace gnss_gpu
