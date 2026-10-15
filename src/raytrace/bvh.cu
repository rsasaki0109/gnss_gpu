#include "gnss_gpu/bvh.h"
#include "gnss_gpu/cuda_check.h"
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gnss_gpu {

// ============================================================
// CPU-side BVH construction
// ============================================================

static AABB triangle_aabb(const Triangle& tri) {
  AABB box;
  for (int d = 0; d < 3; d++) {
    double lo = tri.v0[d];
    double hi = lo;
    if (tri.v1[d] < lo) lo = tri.v1[d];
    if (tri.v2[d] < lo) lo = tri.v2[d];
    if (tri.v1[d] > hi) hi = tri.v1[d];
    if (tri.v2[d] > hi) hi = tri.v2[d];
    box.min[d] = lo;
    box.max[d] = hi;
  }
  return box;
}

static AABB union_aabb(const AABB& a, const AABB& b) {
  AABB c;
  for (int d = 0; d < 3; d++) {
    c.min[d] = (a.min[d] < b.min[d]) ? a.min[d] : b.min[d];
    c.max[d] = (a.max[d] > b.max[d]) ? a.max[d] : b.max[d];
  }
  return c;
}

static double aabb_surface_area(const AABB& b) {
  double dx = b.max[0] - b.min[0];
  double dy = b.max[1] - b.min[1];
  double dz = b.max[2] - b.min[2];
  return 2.0 * (dx * dy + dy * dz + dz * dx);
}

static double triangle_centroid(const Triangle& tri, int axis) {
  return (tri.v0[axis] + tri.v1[axis] + tri.v2[axis]) / 3.0;
}

static const int MAX_LEAF_TRIS = 4;
static const int SAH_BINS = 12;

// These values depend only on the input mesh. Keep the original FP64
// calculations, then reuse them throughout partitioning and centroid sorting.
struct BuildTriangle {
  AABB bounds;
  double centroid[3];
};

// Recursive top-down build. indices[start..end) are the triangle indices to partition.
static int build_recursive(const BuildTriangle* geometry,
                           std::vector<int>& indices, int start, int end,
                           std::vector<BVHNode>& nodes,
                           AABB* prefix_box, AABB* suffix_box) {
  int count = end - start;
  // Compute bounds of all triangles in range
  AABB bounds = geometry[indices[start]].bounds;
  for (int i = start + 1; i < end; i++) {
    bounds = union_aabb(bounds, geometry[indices[i]].bounds);
  }

  // Leaf node
  if (count <= MAX_LEAF_TRIS) {
    BVHNode node;
    node.bbox = bounds;
    node.left = -1;
    node.right = -1;
    node.tri_start = start;
    node.tri_count = count;
    int idx = (int)nodes.size();
    nodes.push_back(node);
    return idx;
  }

  // SAH split: try each axis, find best bin split
  double best_cost = DBL_MAX;
  int best_axis = 0;
  int best_split = start + count / 2;  // fallback: midpoint

  double parent_sa = aabb_surface_area(bounds);
  if (parent_sa < 1e-30) parent_sa = 1e-30;

  for (int axis = 0; axis < 3; axis++) {
    double axis_min = bounds.min[axis];
    double axis_max = bounds.max[axis];
    if (axis_max - axis_min < 1e-15) continue;

    double bin_width = (axis_max - axis_min) / SAH_BINS;

    // Count triangles per bin and compute bin bounds
    int bin_count[SAH_BINS] = {};
    AABB bin_bounds[SAH_BINS];
    for (int b = 0; b < SAH_BINS; b++) {
      bin_bounds[b].min[0] = bin_bounds[b].min[1] = bin_bounds[b].min[2] = DBL_MAX;
      bin_bounds[b].max[0] = bin_bounds[b].max[1] = bin_bounds[b].max[2] = -DBL_MAX;
    }

    for (int i = start; i < end; i++) {
      double c = geometry[indices[i]].centroid[axis];
      int b = (int)((c - axis_min) / bin_width);
      if (b < 0) b = 0;
      if (b >= SAH_BINS) b = SAH_BINS - 1;
      bin_count[b]++;
      AABB tb = geometry[indices[i]].bounds;
      bin_bounds[b] = union_aabb(bin_bounds[b], tb);
    }

    // Sweep from left: prefix counts and bounds
    AABB left_bounds[SAH_BINS - 1];
    int left_count[SAH_BINS - 1];
    {
      AABB running = bin_bounds[0];
      int cnt = bin_count[0];
      for (int s = 0; s < SAH_BINS - 1; s++) {
        if (s > 0) {
          running = union_aabb(running, bin_bounds[s]);
          cnt += bin_count[s];
        }
        left_bounds[s] = running;
        left_count[s] = cnt;
      }
    }

    // Sweep from right
    AABB right_bounds[SAH_BINS - 1];
    int right_count[SAH_BINS - 1];
    {
      AABB running = bin_bounds[SAH_BINS - 1];
      int cnt = bin_count[SAH_BINS - 1];
      for (int s = SAH_BINS - 2; s >= 0; s--) {
        right_bounds[s] = running;
        right_count[s] = cnt;
        if (s > 0) {
          running = union_aabb(running, bin_bounds[s]);
          cnt += bin_count[s];
        }
      }
    }

    // Evaluate SAH cost for each split plane
    for (int s = 0; s < SAH_BINS - 1; s++) {
      if (left_count[s] == 0 || right_count[s] == 0) continue;
      double cost = 1.0 +
          (aabb_surface_area(left_bounds[s]) * left_count[s] +
           aabb_surface_area(right_bounds[s]) * right_count[s]) / parent_sa;
      if (cost < best_cost) {
        best_cost = cost;
        best_axis = axis;
        // We need to compute the actual split position
        // Sort-and-partition approach: partition by centroid vs split plane
        best_split = -1;  // will be set below via partition
        // Store the split plane value
      }
    }

    // If this axis had the best cost, do the partition
    if (best_cost < DBL_MAX) {
      // Re-find the best split index for this axis
      // (We'll do a single clean pass below after choosing the axis)
    }
  }

  // Partition indices along best_axis using nth_element-style approach
  // Sort the range by centroid along best_axis
  int axis = best_axis;
  std::sort(indices.begin() + start, indices.begin() + end,
            [&](int a, int b) {
              return geometry[a].centroid[axis] <
                     geometry[b].centroid[axis];
            });

  // Find best SAH split on sorted array
  // These buffers are consumed before descending into either child, so
  // recursive calls can overwrite them without keeping per-node allocations.
  // Build prefix AABBs
  prefix_box[0] = geometry[indices[start]].bounds;
  for (int i = 1; i < count; i++) {
    prefix_box[i] = union_aabb(prefix_box[i - 1],
                                geometry[indices[start + i]].bounds);
  }

  suffix_box[count - 1] = geometry[indices[start + count - 1]].bounds;
  for (int i = count - 2; i >= 0; i--) {
    suffix_box[i] = union_aabb(suffix_box[i + 1],
                                geometry[indices[start + i]].bounds);
  }

  best_cost = DBL_MAX;
  best_split = start + count / 2;
  for (int i = 1; i < count; i++) {
    double cost = (aabb_surface_area(prefix_box[i - 1]) * i +
                   aabb_surface_area(suffix_box[i]) * (count - i)) / parent_sa;
    if (cost < best_cost) {
      best_cost = cost;
      best_split = start + i;
    }
  }

  // Prevent degenerate splits
  if (best_split <= start) best_split = start + 1;
  if (best_split >= end) best_split = end - 1;

  // Allocate internal node (reserve index)
  int node_idx = (int)nodes.size();
  nodes.push_back(BVHNode());  // placeholder

  int left_child = build_recursive(geometry, indices, start, best_split, nodes,
                                   prefix_box, suffix_box);
  int right_child = build_recursive(geometry, indices, best_split, end, nodes,
                                    prefix_box, suffix_box);

  nodes[node_idx].bbox = bounds;
  nodes[node_idx].left = left_child;
  nodes[node_idx].right = right_child;
  nodes[node_idx].tri_start = 0;
  nodes[node_idx].tri_count = 0;

  return node_idx;
}

void bvh_build(const Triangle* triangles, int n_tri,
               BVHNode* nodes, int* n_nodes, int* sorted_tri_indices) {
  if (n_tri <= 0) {
    *n_nodes = 0;
    return;
  }

  std::vector<int> indices(n_tri);
  for (int i = 0; i < n_tri; i++) indices[i] = i;

  std::vector<BVHNode> node_vec;
  node_vec.reserve(2 * n_tri);

  std::vector<BuildTriangle> geometry(n_tri);
  for (int i = 0; i < n_tri; i++) {
    geometry[i].bounds = triangle_aabb(triangles[i]);
    for (int axis = 0; axis < 3; axis++)
      geometry[i].centroid[axis] = triangle_centroid(triangles[i], axis);
  }

  std::vector<AABB> prefix_box(n_tri > MAX_LEAF_TRIS ? n_tri : 0);
  std::vector<AABB> suffix_box(prefix_box.size());
  build_recursive(geometry.data(), indices, 0, n_tri, node_vec,
                  prefix_box.data(), suffix_box.data());

  *n_nodes = (int)node_vec.size();
  for (int i = 0; i < *n_nodes; i++) {
    nodes[i] = node_vec[i];
  }
  for (int i = 0; i < n_tri; i++) {
    sorted_tri_indices[i] = indices[i];
  }
}

// ============================================================
// GPU-side BVH traversal
// ============================================================

// Moller-Trumbore (same as raytrace.cu but accessible here)
__device__ static bool mt_intersect(const double origin[3], const double dir[3],
                                    const double v0[3], const double v1[3],
                                    const double v2[3], double& t) {
  const double EPSILON = 1e-12;
  double e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  double e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
  double h[3] = {dir[1] * e2[2] - dir[2] * e2[1],
                  dir[2] * e2[0] - dir[0] * e2[2],
                  dir[0] * e2[1] - dir[1] * e2[0]};
  double a = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
  if (fabs(a) < EPSILON) return false;
  double f = 1.0 / a;
  double s[3] = {origin[0] - v0[0], origin[1] - v0[1], origin[2] - v0[2]};
  double u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
  if (u < 0.0 || u > 1.0) return false;
  double q[3] = {s[1] * e1[2] - s[2] * e1[1],
                  s[2] * e1[0] - s[0] * e1[2],
                  s[0] * e1[1] - s[1] * e1[0]};
  double v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
  if (v < 0.0 || u + v > 1.0) return false;
  t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);
  return t > EPSILON;
}

// Ray-AABB intersection using slab method
// dir_inv = 1.0 / dir for each component
__device__ static bool ray_aabb_intersect(const double origin[3],
                                          const double dir_inv[3],
                                          double tmin, double tmax,
                                          const AABB& aabb) {
  for (int d = 0; d < 3; d++) {
    double t1 = (aabb.min[d] - origin[d]) * dir_inv[d];
    double t2 = (aabb.max[d] - origin[d]) * dir_inv[d];
    if (t1 > t2) { double tmp = t1; t1 = t2; t2 = tmp; }
    if (t1 > tmin) tmin = t1;
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return false;
  }
  return true;
}

static const int BVH_MAX_DEPTH = 64;

__global__ void los_check_bvh_kernel(const double* rx_ecef,
                                      const double* sat_ecef,
                                      const BVHNode* bvh,
                                      const Triangle* tris,
                                      int* is_los,
                                      int n_sat, int n_nodes) {
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= n_sat) return;

  double ox = rx_ecef[0], oy = rx_ecef[1], oz = rx_ecef[2];
  double sx = sat_ecef[sid * 3 + 0];
  double sy = sat_ecef[sid * 3 + 1];
  double sz = sat_ecef[sid * 3 + 2];

  double dir[3] = {sx - ox, sy - oy, sz - oz};
  double dist = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  if (dist < 1e-15) { is_los[sid] = 1; return; }
  dir[0] /= dist; dir[1] /= dist; dir[2] /= dist;

  double dir_inv[3];
  for (int d = 0; d < 3; d++) {
    dir_inv[d] = (fabs(dir[d]) > 1e-20) ? (1.0 / dir[d]) : ((dir[d] >= 0) ? 1e20 : -1e20);
  }

  double origin[3] = {ox, oy, oz};

  // Stack-based traversal
  int stack[BVH_MAX_DEPTH];
  int sp = 0;
  stack[sp++] = 0;  // root node index

  bool blocked = false;
  while (sp > 0 && !blocked) {
    int node_idx = stack[--sp];
    const BVHNode& node = bvh[node_idx];

    if (!ray_aabb_intersect(origin, dir_inv, 0.0, dist, node.bbox))
      continue;

    if (node.left == -1) {
      // Leaf: test triangles
      for (int i = 0; i < node.tri_count; i++) {
        double t;
        const Triangle& tri = tris[node.tri_start + i];
        if (mt_intersect(origin, dir, tri.v0, tri.v1, tri.v2, t)) {
          if (t < dist) {
            blocked = true;
            break;
          }
        }
      }
    } else {
      // Internal: push children
      if (sp < BVH_MAX_DEPTH - 1) {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    }
  }

  is_los[sid] = blocked ? 0 : 1;
}

__global__ void los_check_bvh_batch_kernel(const double* rx_ecef,
                                            const double* sat_ecef,
                                            const BVHNode* bvh,
                                            const Triangle* tris,
                                            int* is_los,
                                            int n_epoch, int n_sat, int n_nodes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_epoch * n_sat;
  if (tid >= total) return;

  int eid = tid / n_sat;
  int sid = tid % n_sat;

  double ox = rx_ecef[eid * 3 + 0];
  double oy = rx_ecef[eid * 3 + 1];
  double oz = rx_ecef[eid * 3 + 2];
  int sat_offset = (eid * n_sat + sid) * 3;
  double sx = sat_ecef[sat_offset + 0];
  double sy = sat_ecef[sat_offset + 1];
  double sz = sat_ecef[sat_offset + 2];

  if (!isfinite(sx) || !isfinite(sy) || !isfinite(sz) ||
      !isfinite(ox) || !isfinite(oy) || !isfinite(oz)) {
    is_los[tid] = 0;
    return;
  }

  double dir[3] = {sx - ox, sy - oy, sz - oz};
  double dist = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  if (dist < 1e-15) { is_los[tid] = 1; return; }
  dir[0] /= dist; dir[1] /= dist; dir[2] /= dist;

  double dir_inv[3];
  for (int d = 0; d < 3; d++) {
    dir_inv[d] = (fabs(dir[d]) > 1e-20) ? (1.0 / dir[d]) : ((dir[d] >= 0) ? 1e20 : -1e20);
  }

  double origin[3] = {ox, oy, oz};

  int stack[BVH_MAX_DEPTH];
  int sp = 0;
  stack[sp++] = 0;

  bool blocked = false;
  while (sp > 0 && !blocked) {
    int node_idx = stack[--sp];
    const BVHNode& node = bvh[node_idx];

    if (!ray_aabb_intersect(origin, dir_inv, 0.0, dist, node.bbox))
      continue;

    if (node.left == -1) {
      for (int i = 0; i < node.tri_count; i++) {
        double t;
        const Triangle& tri = tris[node.tri_start + i];
        if (mt_intersect(origin, dir, tri.v0, tri.v1, tri.v2, t)) {
          if (t < dist) { blocked = true; break; }
        }
      }
    } else {
      if (sp < BVH_MAX_DEPTH - 1) {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    }
  }

  is_los[tid] = blocked ? 0 : 1;
}

__global__ void multipath_bvh_batch_kernel(const double* rx_ecef,
                                            const double* sat_ecef,
                                            const BVHNode* bvh,
                                            const Triangle* tris,
                                            double* reflection_points,
                                            double* excess_delays,
                                            int n_epoch, int n_sat, int n_nodes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_epoch * n_sat;
  if (tid >= total) return;

  int eid = tid / n_sat;
  int sid = tid % n_sat;

  double rx[3] = {rx_ecef[eid * 3 + 0],
                   rx_ecef[eid * 3 + 1],
                   rx_ecef[eid * 3 + 2]};
  int sat_offset = (eid * n_sat + sid) * 3;
  double sat[3] = {sat_ecef[sat_offset + 0],
                    sat_ecef[sat_offset + 1],
                    sat_ecef[sat_offset + 2]};

  if (!isfinite(rx[0]) || !isfinite(rx[1]) || !isfinite(rx[2]) ||
      !isfinite(sat[0]) || !isfinite(sat[1]) || !isfinite(sat[2])) {
    excess_delays[tid] = 0.0;
    reflection_points[tid * 3 + 0] = 0.0;
    reflection_points[tid * 3 + 1] = 0.0;
    reflection_points[tid * 3 + 2] = 0.0;
    return;
  }

  double d_rx_sat = sqrt((rx[0] - sat[0]) * (rx[0] - sat[0]) +
                          (rx[1] - sat[1]) * (rx[1] - sat[1]) +
                          (rx[2] - sat[2]) * (rx[2] - sat[2]));

  double best_delay = 0.0;
  double best_refl[3] = {0.0, 0.0, 0.0};
  bool found = false;

  int stack[64];
  int sp = 0;
  stack[sp++] = 0;

  while (sp > 0) {
    int node_idx = stack[--sp];
    const BVHNode& node = bvh[node_idx];

    if (node.left == -1) {
      for (int i = 0; i < node.tri_count; i++) {
        const Triangle& tri = tris[node.tri_start + i];

        double e1[3] = {tri.v1[0] - tri.v0[0], tri.v1[1] - tri.v0[1], tri.v1[2] - tri.v0[2]};
        double e2[3] = {tri.v2[0] - tri.v0[0], tri.v2[1] - tri.v0[1], tri.v2[2] - tri.v0[2]};
        double n[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0]};
        double nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (nn < 1e-15) continue;
        n[0] /= nn; n[1] /= nn; n[2] /= nn;

        double dv[3] = {rx[0] - tri.v0[0], rx[1] - tri.v0[1], rx[2] - tri.v0[2]};
        double d = dv[0] * n[0] + dv[1] * n[1] + dv[2] * n[2];
        double mirror[3] = {rx[0] - 2.0 * d * n[0],
                             rx[1] - 2.0 * d * n[1],
                             rx[2] - 2.0 * d * n[2]};

        double dir[3] = {sat[0] - mirror[0], sat[1] - mirror[1], sat[2] - mirror[2]};
        double dir_len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        if (dir_len < 1e-15) continue;
        dir[0] /= dir_len; dir[1] /= dir_len; dir[2] /= dir_len;

        double t;
        if (!mt_intersect(mirror, dir, tri.v0, tri.v1, tri.v2, t)) continue;

        double refl[3] = {mirror[0] + t * dir[0],
                           mirror[1] + t * dir[1],
                           mirror[2] + t * dir[2]};

        double d_rx_refl = sqrt((rx[0] - refl[0]) * (rx[0] - refl[0]) +
                                 (rx[1] - refl[1]) * (rx[1] - refl[1]) +
                                 (rx[2] - refl[2]) * (rx[2] - refl[2]));
        double d_refl_sat = sqrt((refl[0] - sat[0]) * (refl[0] - sat[0]) +
                                  (refl[1] - sat[1]) * (refl[1] - sat[1]) +
                                  (refl[2] - sat[2]) * (refl[2] - sat[2]));
        double excess = (d_rx_refl + d_refl_sat) - d_rx_sat;

        if (excess > 0.0 && (!found || excess < best_delay)) {
          best_delay = excess;
          best_refl[0] = refl[0]; best_refl[1] = refl[1]; best_refl[2] = refl[2];
          found = true;
        }
      }
    } else {
      if (sp < 62) {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    }
  }

  excess_delays[tid] = best_delay;
  reflection_points[tid * 3 + 0] = best_refl[0];
  reflection_points[tid * 3 + 1] = best_refl[1];
  reflection_points[tid * 3 + 2] = best_refl[2];
}

// The scene owns these GPU-computed normals in traversal order. Keeping the
// original FP64 operations on the device preserves reflection arithmetic.
struct ReflectionNormal {
  double value[3];
  int valid;
};

__global__ void reflection_normals_kernel(const Triangle* tris, const int* order,
                                          ReflectionNormal* normals, int count) {
  int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= count) return;
  const Triangle& tri = tris[order[ordinal]];
  double e1[3] = {tri.v1[0] - tri.v0[0], tri.v1[1] - tri.v0[1], tri.v1[2] - tri.v0[2]};
  double e2[3] = {tri.v2[0] - tri.v0[0], tri.v2[1] - tri.v0[1], tri.v2[2] - tri.v0[2]};
  double n[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                  e1[2] * e2[0] - e1[0] * e2[2],
                  e1[0] * e2[1] - e1[1] * e2[0]};
  double nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  ReflectionNormal result = {};
  if (!(nn < 1e-15)) {
    n[0] /= nn; n[1] /= nn; n[2] /= nn;
    result.value[0] = n[0]; result.value[1] = n[1]; result.value[2] = n[2];
    result.valid = 1;
  }
  normals[ordinal] = result;
}

// Large ordered batches reuse immutable GPU normals with one thread per ray.
__global__ void multipath_cached_serial_kernel(const double* rx_ecef,
                                            const double* sat_ecef,
                                            const int* triangle_order,
                                            const ReflectionNormal* normals,
                                            const Triangle* tris,
                                            double* reflection_points,
                                            double* excess_delays,
                                            int n_epoch, int n_sat, int n_tri) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_epoch * n_sat;
  if (tid >= total) return;

  int eid = tid / n_sat;
  int sid = tid % n_sat;

  double rx[3] = {rx_ecef[eid * 3 + 0],
                   rx_ecef[eid * 3 + 1],
                   rx_ecef[eid * 3 + 2]};
  int sat_offset = (eid * n_sat + sid) * 3;
  double sat[3] = {sat_ecef[sat_offset + 0],
                    sat_ecef[sat_offset + 1],
                    sat_ecef[sat_offset + 2]};

  if (!isfinite(rx[0]) || !isfinite(rx[1]) || !isfinite(rx[2]) ||
      !isfinite(sat[0]) || !isfinite(sat[1]) || !isfinite(sat[2])) {
    excess_delays[tid] = 0.0;
    reflection_points[tid * 3 + 0] = 0.0;
    reflection_points[tid * 3 + 1] = 0.0;
    reflection_points[tid * 3 + 2] = 0.0;
    return;
  }

  double d_rx_sat = sqrt((rx[0] - sat[0]) * (rx[0] - sat[0]) +
                          (rx[1] - sat[1]) * (rx[1] - sat[1]) +
                          (rx[2] - sat[2]) * (rx[2] - sat[2]));

  double best_delay = 0.0;
  double best_refl[3] = {0.0, 0.0, 0.0};
  bool found = false;

  // The owner captured the exact right-first traversal, including the
  // legacy stack limit. Equal delays still retain the first visited triangle.
  for (int ordinal = 0; ordinal < n_tri; ++ordinal) {
    const Triangle& tri = tris[triangle_order[ordinal]];
    const ReflectionNormal& normal = normals[ordinal];
    if (!normal.valid) continue;
    double n[3] = {normal.value[0], normal.value[1], normal.value[2]};

    double dv[3] = {rx[0] - tri.v0[0], rx[1] - tri.v0[1], rx[2] - tri.v0[2]};
    double d = dv[0] * n[0] + dv[1] * n[1] + dv[2] * n[2];
    double mirror[3] = {rx[0] - 2.0 * d * n[0],
                         rx[1] - 2.0 * d * n[1],
                         rx[2] - 2.0 * d * n[2]};

    double dir[3] = {sat[0] - mirror[0], sat[1] - mirror[1], sat[2] - mirror[2]};
    double dir_len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    if (dir_len < 1e-15) continue;
    dir[0] /= dir_len; dir[1] /= dir_len; dir[2] /= dir_len;

    double t;
    if (!mt_intersect(mirror, dir, tri.v0, tri.v1, tri.v2, t)) continue;

    double refl[3] = {mirror[0] + t * dir[0],
                       mirror[1] + t * dir[1],
                       mirror[2] + t * dir[2]};

    double d_rx_refl = sqrt((rx[0] - refl[0]) * (rx[0] - refl[0]) +
                             (rx[1] - refl[1]) * (rx[1] - refl[1]) +
                             (rx[2] - refl[2]) * (rx[2] - refl[2]));
    double d_refl_sat = sqrt((refl[0] - sat[0]) * (refl[0] - sat[0]) +
                              (refl[1] - sat[1]) * (refl[1] - sat[1]) +
                              (refl[2] - sat[2]) * (refl[2] - sat[2]));
    double excess = (d_rx_refl + d_refl_sat) - d_rx_sat;

    if (excess > 0.0 && (!found || excess < best_delay)) {
      best_delay = excess;
      best_refl[0] = refl[0]; best_refl[1] = refl[1]; best_refl[2] = refl[2];
      found = true;
    }
  }

  excess_delays[tid] = best_delay;
  reflection_points[tid * 3 + 0] = best_refl[0];
  reflection_points[tid * 3 + 1] = best_refl[1];
  reflection_points[tid * 3 + 2] = best_refl[2];
}

struct ReflectionMirror {
  double point[3];
};

// All satellites in an epoch use the same receiver mirror on each triangle.
// Recompute on every call from the current device rays, using the same FP64
// operations as candidate evaluation. No receiver-dependent data stays valid
// across calls without this preparation.
__global__ void reflection_mirrors_kernel(const double* rx_ecef,
                                           const Triangle* tris, const int* order,
                                           const ReflectionNormal* normals,
                                           ReflectionMirror* mirrors, int count) {
  int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
  if (ordinal >= count) return;
  int epoch = blockIdx.y;
  ReflectionMirror result = {};
  const ReflectionNormal& normal = normals[ordinal];
  if (normal.valid) {
    const Triangle& tri = tris[order[ordinal]];
    double rx[3] = {rx_ecef[epoch * 3], rx_ecef[epoch * 3 + 1], rx_ecef[epoch * 3 + 2]};
    double n[3] = {normal.value[0], normal.value[1], normal.value[2]};
    double dv[3] = {rx[0] - tri.v0[0], rx[1] - tri.v0[1], rx[2] - tri.v0[2]};
    double d = dv[0] * n[0] + dv[1] * n[1] + dv[2] * n[2];
    result.point[0] = rx[0] - 2.0 * d * n[0];
    result.point[1] = rx[1] - 2.0 * d * n[1];
    result.point[2] = rx[2] - 2.0 * d * n[2];
  }
  mirrors[(size_t)epoch * count + ordinal] = result;
}

struct MultipathCandidate {
  double delay;
  double point[3];
  int order;  // -1 means no reflection; otherwise original traversal ordinal.
};

__device__ static bool better_multipath(const MultipathCandidate& a,
                                        const MultipathCandidate& b) {
  return a.order >= 0 &&
      (b.order < 0 || a.delay < b.delay ||
       (a.delay == b.delay && a.order < b.order));
}

// Keep the arithmetic and intersection thresholds identical to the serial
// kernel above. Only independent candidate evaluation and selection are split.
template <bool CACHED_NORMALS, bool SHARED_MIRRORS>
__device__ static bool multipath_candidate(const double rx[3], const double sat[3],
                                           double d_rx_sat, const Triangle& tri,
                                           MultipathCandidate& result,
                                           const ReflectionNormal* normal,
                                           const ReflectionMirror* prepared_mirror) {
  double n[3];
  if constexpr (CACHED_NORMALS) {
    if (!normal->valid) return false;
    n[0] = normal->value[0]; n[1] = normal->value[1]; n[2] = normal->value[2];
  } else {
    double e1[3] = {tri.v1[0] - tri.v0[0], tri.v1[1] - tri.v0[1], tri.v1[2] - tri.v0[2]};
    double e2[3] = {tri.v2[0] - tri.v0[0], tri.v2[1] - tri.v0[1], tri.v2[2] - tri.v0[2]};
    n[0] = e1[1] * e2[2] - e1[2] * e2[1];
    n[1] = e1[2] * e2[0] - e1[0] * e2[2];
    n[2] = e1[0] * e2[1] - e1[1] * e2[0];
    double nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (nn < 1e-15) return false;
    n[0] /= nn; n[1] /= nn; n[2] /= nn;
  }

  double mirror[3];
  if constexpr (SHARED_MIRRORS) {
    mirror[0] = prepared_mirror->point[0];
    mirror[1] = prepared_mirror->point[1];
    mirror[2] = prepared_mirror->point[2];
  } else {
    double dv[3] = {rx[0] - tri.v0[0], rx[1] - tri.v0[1], rx[2] - tri.v0[2]};
    double d = dv[0] * n[0] + dv[1] * n[1] + dv[2] * n[2];
    mirror[0] = rx[0] - 2.0 * d * n[0];
    mirror[1] = rx[1] - 2.0 * d * n[1];
    mirror[2] = rx[2] - 2.0 * d * n[2];
  }

  double dir[3] = {sat[0] - mirror[0], sat[1] - mirror[1], sat[2] - mirror[2]};
  double dir_len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  if (dir_len < 1e-15) return false;
  dir[0] /= dir_len; dir[1] /= dir_len; dir[2] /= dir_len;

  double t;
  if (!mt_intersect(mirror, dir, tri.v0, tri.v1, tri.v2, t)) return false;

  double refl[3] = {mirror[0] + t * dir[0],
                     mirror[1] + t * dir[1],
                     mirror[2] + t * dir[2]};
  double d_rx_refl = sqrt((rx[0] - refl[0]) * (rx[0] - refl[0]) +
                           (rx[1] - refl[1]) * (rx[1] - refl[1]) +
                           (rx[2] - refl[2]) * (rx[2] - refl[2]));
  double d_refl_sat = sqrt((refl[0] - sat[0]) * (refl[0] - sat[0]) +
                            (refl[1] - sat[1]) * (refl[1] - sat[1]) +
                            (refl[2] - sat[2]) * (refl[2] - sat[2]));
  double excess = (d_rx_refl + d_refl_sat) - d_rx_sat;
  if (!(excess > 0.0)) return false;
  result.delay = excess;
  result.point[0] = refl[0]; result.point[1] = refl[1]; result.point[2] = refl[2];
  return true;
}

static const int MULTIPATH_BLOCK_SIZE = 128;

template <bool CACHED_NORMALS, bool SHARED_MIRRORS>
__global__ void multipath_candidates_kernel(const double* rx_ecef,
                                            const double* sat_ecef,
                                            const Triangle* tris,
                                            const int* triangle_order,
                                            MultipathCandidate* partial,
                                            int n_sat, int n_tri, int tiles,
                                            const ReflectionNormal* normals,
                                            const ReflectionMirror* mirrors) {
  int ray = blockIdx.x / tiles;
  int tile = blockIdx.x % tiles;
  int eid = ray / n_sat;
  double rx[3] = {rx_ecef[eid * 3], rx_ecef[eid * 3 + 1], rx_ecef[eid * 3 + 2]};
  double sat[3] = {sat_ecef[ray * 3], sat_ecef[ray * 3 + 1], sat_ecef[ray * 3 + 2]};
  MultipathCandidate best = {};
  best.order = -1;
  if (!isfinite(rx[0]) || !isfinite(rx[1]) || !isfinite(rx[2]) ||
      !isfinite(sat[0]) || !isfinite(sat[1]) || !isfinite(sat[2])) {
    if (threadIdx.x == 0) partial[blockIdx.x] = best;
    return;
  }
  double d_rx_sat = sqrt((rx[0] - sat[0]) * (rx[0] - sat[0]) +
                          (rx[1] - sat[1]) * (rx[1] - sat[1]) +
                          (rx[2] - sat[2]) * (rx[2] - sat[2]));
  for (int ordinal = tile * blockDim.x + threadIdx.x;
       ordinal < n_tri; ordinal += tiles * blockDim.x) {
    MultipathCandidate candidate = {};
    candidate.order = ordinal;
    const ReflectionNormal* normal = nullptr;
    if constexpr (CACHED_NORMALS) normal = normals + ordinal;
    const ReflectionMirror* mirror = nullptr;
    if constexpr (SHARED_MIRRORS) mirror = mirrors + (size_t)eid * n_tri + ordinal;
    if (multipath_candidate<CACHED_NORMALS, SHARED_MIRRORS>(
            rx, sat, d_rx_sat, tris[triangle_order[ordinal]], candidate, normal, mirror) &&
        better_multipath(candidate, best)) best = candidate;
  }
  __shared__ MultipathCandidate winners[MULTIPATH_BLOCK_SIZE];
  winners[threadIdx.x] = best;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride &&
        better_multipath(winners[threadIdx.x + stride], winners[threadIdx.x])) {
      winners[threadIdx.x] = winners[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = winners[0];
}

__global__ void multipath_finalize_kernel(const MultipathCandidate* partial,
                                          double* reflection_points,
                                          double* excess_delays, int n_rays, int tiles) {
  int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= n_rays) return;
  MultipathCandidate best = {};
  best.order = -1;
  for (int tile = 0; tile < tiles; tile++) {
    const MultipathCandidate& candidate = partial[ray * tiles + tile];
    if (better_multipath(candidate, best)) best = candidate;
  }
  excess_delays[ray] = best.delay;
  reflection_points[ray * 3] = best.point[0];
  reflection_points[ray * 3 + 1] = best.point[1];
  reflection_points[ray * 3 + 2] = best.point[2];
}

// Mirror the serial kernel's right-first leaf traversal and stack limit. A
// triangle's array index alone does not preserve ties across different leaves.
static bool multipath_triangle_order(const BVHNode* bvh, int n_nodes, int n_tri,
                                      std::vector<int>& order) {
  int stack[64];
  int sp = 0;
  int visited = 0;
  stack[sp++] = 0;
  order.reserve(n_tri);
  while (sp > 0) {
    int node_idx = stack[--sp];
    if (node_idx < 0 || node_idx >= n_nodes || ++visited > n_nodes) return false;
    const BVHNode& node = bvh[node_idx];
    if (node.left == -1) {
      if (node.tri_start < 0 || node.tri_count < 0 ||
          node.tri_start > n_tri || node.tri_count > n_tri - node.tri_start) return false;
      // Overlapping custom leaves can visit more entries than the triangle
      // buffer contains. Keep their serial behavior instead of growing scratch.
      if (order.size() > (size_t)(n_tri - node.tri_count)) return false;
      for (int i = 0; i < node.tri_count; i++) order.push_back(node.tri_start + i);
    } else if (sp < 62) {
      stack[sp++] = node.left;
      stack[sp++] = node.right;
    }
  }
  return !order.empty();
}

namespace {

template <typename T>
struct DeviceBuffer {
  T* data = nullptr;
  size_t capacity = 0;

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { reset(); }
  void reset() noexcept {
    if (data) cudaFree(data);
    data = nullptr;
    capacity = 0;
  }
  void reserve(size_t count) {
    if (count <= capacity) return;
    T* replacement = nullptr;
    CUDA_CHECK(cudaMalloc(&replacement, count * sizeof(T)));
    reset();
    data = replacement;
    capacity = count;
  }
  void upload(const T* source, size_t count) {
    reserve(count);
    if (count) CUDA_CHECK(cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice));
  }
};

class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    CUDA_CHECK(cudaGetDevice(&previous_));
    changed_ = previous_ != device;
    if (changed_) CUDA_CHECK(cudaSetDevice(device));
  }
  ~DeviceScope() { if (changed_) cudaSetDevice(previous_); }
 private:
  int previous_ = 0;
  bool changed_ = false;
};

// Validate ranges and reject reachable cycles before any GPU allocation. Shared
// or overlapping leaves remain valid and use the legacy multipath path as needed.
int validate_device_bvh(const BVHNode* nodes, int n_nodes, int n_tri) {
  int referenced = 0;
  for (int i = 0; i < n_nodes; ++i) {
    const auto& node = nodes[i];
    if (node.left == -1) {
      if (node.tri_start < 0 || node.tri_count < 0 || node.tri_start > n_tri ||
          node.tri_count > n_tri - node.tri_start)
        throw std::invalid_argument("DeviceBVH: leaf triangle range is out of bounds");
      referenced = std::max(referenced, node.tri_start + node.tri_count);
    } else if (node.left < 0 || node.right < 0 || node.left >= n_nodes || node.right >= n_nodes) {
      throw std::invalid_argument("DeviceBVH: child index is out of bounds");
    }
  }
  std::vector<unsigned char> colors(n_nodes, 0);
  std::vector<std::pair<int, bool>> pending{{0, false}};
  while (!pending.empty()) {
    auto [index, leaving] = pending.back();
    pending.pop_back();
    if (leaving) {
      colors[index] = 2;
      continue;
    }
    if (colors[index] == 1) throw std::invalid_argument("DeviceBVH: BVH contains a cycle");
    if (colors[index] == 2) continue;
    colors[index] = 1;
    pending.emplace_back(index, true);
    if (nodes[index].left != -1) {
      pending.emplace_back(nodes[index].left, false);
      pending.emplace_back(nodes[index].right, false);
    }
  }
  return referenced;
}

}  // namespace

struct DeviceBVH::Impl {
  int device = 0;
  int n_nodes = 0;
  int n_order = 0;
  bool ordered = false;
  std::vector<int> host_order;
  DeviceBuffer<BVHNode> nodes;
  DeviceBuffer<Triangle> triangles;
  DeviceBuffer<int> order, los;
  DeviceBuffer<double> rx, sat, points, delays;
  DeviceBuffer<MultipathCandidate> partial;
  DeviceBuffer<ReflectionNormal> normals;
  bool normals_ready = false;
  DeviceBuffer<ReflectionMirror> mirrors;
  bool mirrors_active = false;
  int mirror_epochs = 0;
  bool cached_serial = false;

  ~Impl() {
    int previous = 0;
    bool known = cudaGetDevice(&previous) == cudaSuccess;
    if (!known || previous != device) cudaSetDevice(device);
    nodes.reset(); triangles.reset(); order.reset(); los.reset();
    rx.reset(); sat.reset(); points.reset(); delays.reset(); partial.reset(); normals.reset();
    mirrors.reset();
    if (known && previous != device) cudaSetDevice(previous);
  }

  int upload_rays(const double* h_rx, const double* h_sat, int n_epoch, int n_sat) {
    // Kernel offsets use int, including the three coordinates of each ray.
    if (!h_rx || !h_sat || n_epoch <= 0 || n_sat <= 0 ||
        n_epoch > (std::numeric_limits<int>::max() / 3) / n_sat)
      throw std::invalid_argument("DeviceBVH: invalid or oversized ray batch");
    int total = n_epoch * n_sat;
    rx.upload(h_rx, (size_t)n_epoch * 3);
    sat.upload(h_sat, (size_t)total * 3);
    return total;
  }

  void launch_los(int n_epoch, int n_sat, bool batch) {
    int total = n_epoch * n_sat;
    if (batch) {
      los_check_bvh_batch_kernel<<<(total + 255) / 256, 256>>>(
          rx.data, sat.data, nodes.data, triangles.data, los.data, n_epoch, n_sat, n_nodes);
    } else {
      los_check_bvh_kernel<<<(total + 255) / 256, 256>>>(
          rx.data, sat.data, nodes.data, triangles.data, los.data, n_sat, n_nodes);
    }
    CUDA_CHECK_LAST();
  }

  int prepare_multipath(int total, int n_epoch, int n_sat) {
    mirrors_active = false;
    mirror_epochs = 0;
    cached_serial = false;
    points.reserve((size_t)total * 3);
    delays.reserve((size_t)total);
    if (!ordered) return 0;
    // LOS-only users do not need this device allocation/upload. Keep the host
    // order until the first successful parallel reflection call, then release it.
    if (!host_order.empty()) {
      order.upload(host_order.data(), host_order.size());
      std::vector<int>().swap(host_order);
    }
    // Substantial meshes amortize one winner reduction per ray. Keep the
    // candidate scratch bounded to 16,384 winners (640 KiB).
    bool large_parallel = n_order >= 64 && total <= 16384;
    if (total > 4096 && !large_parallel) {
      if (!normals_ready) {
        normals.reserve((size_t)n_order);
        reflection_normals_kernel<<<(n_order + 255) / 256, 256>>>(
            triangles.data, order.data, normals.data, n_order);
        CUDA_CHECK_LAST();
        normals_ready = true;
      }
      cached_serial = true;
      return 0;
    }
    int tiles = std::min(32, std::min(
        (n_order + MULTIPATH_BLOCK_SIZE - 1) / MULTIPATH_BLOCK_SIZE,
        std::max(1, 256 / total)));
    partial.reserve((size_t)total * tiles);
    // Prepare only for the parallel reflection path. LOS-only, small meshes and
    // serial fallback batches do not allocate or launch normal preparation.
    if (!normals_ready) {
      normals.reserve((size_t)n_order);
      reflection_normals_kernel<<<(n_order + 255) / 256, 256>>>(
          triangles.data, order.data, normals.data, n_order);
      CUDA_CHECK_LAST();
      normals_ready = true;
    }
    // Share receiver-dependent work only when a substantial mesh and satellite
    // group can amortize the extra launch. Reuse at most32 MiB across epoch
    // chunks; the stream finishes each consumer before overwriting its mirrors.
    size_t max_epochs = ((32u * 1024u * 1024u) / sizeof(ReflectionMirror)) / n_order;
    mirrors_active = n_order >= 8192 && n_sat >= 16 && max_epochs > 0;
    if (mirrors_active) {
      mirror_epochs = (int)std::min((size_t)n_epoch, max_epochs);
      mirrors.reserve((size_t)mirror_epochs * n_order);
    }
    return tiles;
  }

  void launch_multipath(int n_epoch, int n_sat, int tiles) {
    int total = n_epoch * n_sat;
    if (tiles) {
      if (mirrors_active) {
        for (int first_epoch = 0; first_epoch < n_epoch; first_epoch += mirror_epochs) {
          int epochs = std::min(mirror_epochs, n_epoch - first_epoch);
          size_t ray_offset = (size_t)first_epoch * n_sat;
          reflection_mirrors_kernel<<<dim3((n_order + 255) / 256, epochs), 256>>>(
              rx.data + (size_t)first_epoch * 3, triangles.data, order.data,
              normals.data, mirrors.data, n_order);
          CUDA_CHECK_LAST();
          multipath_candidates_kernel<true, true><<<epochs * n_sat * tiles, MULTIPATH_BLOCK_SIZE>>>(
              rx.data + (size_t)first_epoch * 3, sat.data + ray_offset * 3,
              triangles.data, order.data, partial.data + ray_offset * tiles,
              n_sat, n_order, tiles, normals.data, mirrors.data);
          CUDA_CHECK_LAST();
        }
      } else {
        multipath_candidates_kernel<true, false><<<total * tiles, MULTIPATH_BLOCK_SIZE>>>(
            rx.data, sat.data, triangles.data, order.data, partial.data, n_sat, n_order, tiles,
            normals.data, nullptr);
      }
      CUDA_CHECK_LAST();
      multipath_finalize_kernel<<<(total + MULTIPATH_BLOCK_SIZE - 1) / MULTIPATH_BLOCK_SIZE,
                                  MULTIPATH_BLOCK_SIZE>>>(
          partial.data, points.data, delays.data, total, tiles);
    } else if (cached_serial) {
      multipath_cached_serial_kernel<<<(total + 255) / 256, 256>>>(
          rx.data, sat.data, order.data, normals.data, triangles.data,
          points.data, delays.data, n_epoch, n_sat, n_order);
    } else {
      multipath_bvh_batch_kernel<<<(total + 255) / 256, 256>>>(
          rx.data, sat.data, nodes.data, triangles.data, points.data, delays.data,
          n_epoch, n_sat, n_nodes);
    }
    CUDA_CHECK_LAST();
  }

  void copy_multipath(double* h_points, double* h_delays, int total) {
    CUDA_CHECK(cudaMemcpy(h_points, points.data, (size_t)total * 3 * sizeof(double),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_delays, delays.data, (size_t)total * sizeof(double),
                           cudaMemcpyDeviceToHost));
  }

  void trace(const double* h_rx, const double* h_sat, int* h_los, double* h_points,
             double* h_delays, int n_epoch, int n_sat, bool batch) {
    if (!h_los || !h_points || !h_delays)
      throw std::invalid_argument("DeviceBVH: output must not be null");
    DeviceScope scope(device);
    int total = upload_rays(h_rx, h_sat, n_epoch, n_sat);
    los.reserve((size_t)total);
    // Prepare all allocations before either launch, including first-use order
    // upload. Both kernels then use the same rays on the default stream.
    int tiles = prepare_multipath(total, n_epoch, n_sat);
    launch_los(n_epoch, n_sat, batch);
    launch_multipath(n_epoch, n_sat, tiles);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_los, los.data, (size_t)total * sizeof(int), cudaMemcpyDeviceToHost));
    copy_multipath(h_points, h_delays, total);
  }
};

DeviceBVH::DeviceBVH(const BVHNode* h_nodes, int n_nodes, const Triangle* h_tri, int n_tri) {
  if (!h_nodes || !h_tri || n_nodes <= 0 || n_tri <= 0)
    throw std::invalid_argument("DeviceBVH: mesh must contain nodes and triangles");
  int referenced = validate_device_bvh(h_nodes, n_nodes, n_tri);
  // Keep this owner local until all uploads succeed, so failed construction also
  // releases any buffers already allocated.
  auto state = std::make_unique<Impl>();
  CUDA_CHECK(cudaGetDevice(&state->device));
  state->n_nodes = n_nodes;
  state->nodes.upload(h_nodes, (size_t)n_nodes);
  state->triangles.upload(h_tri, (size_t)referenced);
  if (referenced >= 64) {
    state->ordered = multipath_triangle_order(h_nodes, n_nodes, referenced, state->host_order);
    if (state->ordered) {
      state->n_order = (int)state->host_order.size();
    } else {
      std::vector<int>().swap(state->host_order);
    }
  }
  impl_ = std::move(state);
}

DeviceBVH::~DeviceBVH() = default;
void DeviceBVH::close() noexcept { impl_.reset(); }

std::size_t DeviceBVH::allocated_bytes() const noexcept {
  if (!impl_) return 0;
  const auto& s = *impl_;
  return s.nodes.capacity * sizeof(BVHNode) + s.triangles.capacity * sizeof(Triangle) +
      (s.order.capacity + s.los.capacity) * sizeof(int) +
      (s.rx.capacity + s.sat.capacity + s.points.capacity + s.delays.capacity) * sizeof(double) +
      s.partial.capacity * sizeof(MultipathCandidate) +
      s.normals.capacity * sizeof(ReflectionNormal) +
      s.mirrors.capacity * sizeof(ReflectionMirror);
}

void DeviceBVH::check_los(const double* h_rx, const double* h_sat, int* h_los, int n_sat) {
  if (!impl_) throw std::runtime_error("DeviceBVH is closed");
  if (!h_los) throw std::invalid_argument("DeviceBVH: output must not be null");
  auto& state = *impl_;
  DeviceScope scope(state.device);
  int total = state.upload_rays(h_rx, h_sat, 1, n_sat);
  state.los.reserve((size_t)total);
  state.launch_los(1, n_sat, false);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_los, state.los.data, (size_t)total * sizeof(int), cudaMemcpyDeviceToHost));
}

void DeviceBVH::check_los_batch(const double* h_rx, const double* h_sat, int* h_los,
                               int n_epoch, int n_sat) {
  if (!impl_) throw std::runtime_error("DeviceBVH is closed");
  if (!h_los) throw std::invalid_argument("DeviceBVH: output must not be null");
  auto& state = *impl_;
  DeviceScope scope(state.device);
  int total = state.upload_rays(h_rx, h_sat, n_epoch, n_sat);
  state.los.reserve((size_t)total);
  state.launch_los(n_epoch, n_sat, true);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_los, state.los.data, (size_t)total * sizeof(int), cudaMemcpyDeviceToHost));
}

void DeviceBVH::multipath_batch(const double* h_rx, const double* h_sat, double* h_points,
                               double* h_delays, int n_epoch, int n_sat) {
  if (!impl_) throw std::runtime_error("DeviceBVH is closed");
  if (!h_points || !h_delays) throw std::invalid_argument("DeviceBVH: output must not be null");
  auto& state = *impl_;
  DeviceScope scope(state.device);
  int total = state.upload_rays(h_rx, h_sat, n_epoch, n_sat);
  int tiles = state.prepare_multipath(total, n_epoch, n_sat);
  state.launch_multipath(n_epoch, n_sat, tiles);
  CUDA_CHECK(cudaDeviceSynchronize());
  state.copy_multipath(h_points, h_delays, total);
}

void DeviceBVH::trace(const double* h_rx, const double* h_sat, int* h_los,
                     double* h_points, double* h_delays, int n_sat) {
  if (!impl_) throw std::runtime_error("DeviceBVH is closed");
  impl_->trace(h_rx, h_sat, h_los, h_points, h_delays, 1, n_sat, false);
}

void DeviceBVH::trace_batch(const double* h_rx, const double* h_sat, int* h_los,
                           double* h_points, double* h_delays, int n_epoch, int n_sat) {
  if (!impl_) throw std::runtime_error("DeviceBVH is closed");
  impl_->trace(h_rx, h_sat, h_los, h_points, h_delays, n_epoch, n_sat, true);
}

void raytrace_los_check_bvh_batch(const double* rx_ecef, const double* sat_ecef,
                                   const BVHNode* bvh, const Triangle* sorted_tris,
                                   int* is_los, int n_epoch, int n_sat, int n_nodes) {
  size_t sz_rx = (size_t)n_epoch * 3 * sizeof(double);
  size_t sz_sat = (size_t)n_epoch * n_sat * 3 * sizeof(double);
  size_t sz_bvh = (size_t)n_nodes * sizeof(BVHNode);
  size_t sz_los = (size_t)n_epoch * n_sat * sizeof(int);

  int total_tris = 0;
  for (int i = 0; i < n_nodes; i++) {
    if (bvh[i].left == -1) {
      int end = bvh[i].tri_start + bvh[i].tri_count;
      if (end > total_tris) total_tris = end;
    }
  }
  size_t sz_tri = (size_t)total_tris * sizeof(Triangle);

  double *d_rx, *d_sat;
  BVHNode* d_bvh;
  Triangle* d_tri;
  int* d_los;

  CUDA_CHECK(cudaMalloc(&d_rx, sz_rx));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_bvh, sz_bvh));
  CUDA_CHECK(cudaMalloc(&d_tri, sz_tri));
  CUDA_CHECK(cudaMalloc(&d_los, sz_los));

  CUDA_CHECK(cudaMemcpy(d_rx, rx_ecef, sz_rx, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bvh, bvh, sz_bvh, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tri, sorted_tris, sz_tri, cudaMemcpyHostToDevice));

  int total = n_epoch * n_sat;
  int block = 256;
  int grid = (total + block - 1) / block;
  los_check_bvh_batch_kernel<<<grid, block>>>(d_rx, d_sat, d_bvh, d_tri, d_los,
                                                n_epoch, n_sat, n_nodes);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(is_los, d_los, sz_los, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_rx));
  CUDA_CHECK(cudaFree(d_sat));
  CUDA_CHECK(cudaFree(d_bvh));
  CUDA_CHECK(cudaFree(d_tri));
  CUDA_CHECK(cudaFree(d_los));
}

void raytrace_multipath_bvh_batch(const double* rx_ecef, const double* sat_ecef,
                                   const BVHNode* bvh, const Triangle* sorted_tris,
                                   double* reflection_points, double* excess_delays,
                                   int n_epoch, int n_sat, int n_nodes) {
  size_t sz_rx = (size_t)n_epoch * 3 * sizeof(double);
  size_t sz_sat = (size_t)n_epoch * n_sat * 3 * sizeof(double);
  size_t sz_bvh = (size_t)n_nodes * sizeof(BVHNode);
  size_t sz_refl = (size_t)n_epoch * n_sat * 3 * sizeof(double);
  size_t sz_delay = (size_t)n_epoch * n_sat * sizeof(double);

  int total_tris = 0;
  for (int i = 0; i < n_nodes; i++) {
    if (bvh[i].left == -1) {
      int end = bvh[i].tri_start + bvh[i].tri_count;
      if (end > total_tris) total_tris = end;
    }
  }
  size_t sz_tri = (size_t)total_tris * sizeof(Triangle);

  double *d_rx, *d_sat, *d_refl, *d_delay;
  int total = n_epoch * n_sat;
  std::vector<int> triangle_order;
  // A few rays leave the serial kernel mostly idle. Keep the original path
  // for tiny meshes and larger ray batches; bounded tiling avoids a
  // ray-by-triangle output buffer and limits scratch to at most 4096 winners.
  bool parallel = total > 0 && total <= 4096 && total_tris >= 512 &&
      multipath_triangle_order(bvh, n_nodes, total_tris, triangle_order);
  int tiles = parallel ? std::min(32, std::min(
      ((int)triangle_order.size() + MULTIPATH_BLOCK_SIZE - 1) / MULTIPATH_BLOCK_SIZE,
      std::max(1, 256 / total))) : 1;
  BVHNode* d_bvh = nullptr;
  int* d_order = nullptr;
  MultipathCandidate* d_partial = nullptr;
  Triangle* d_tri;

  CUDA_CHECK(cudaMalloc(&d_rx, sz_rx));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  if (!parallel) CUDA_CHECK(cudaMalloc(&d_bvh, sz_bvh));
  CUDA_CHECK(cudaMalloc(&d_tri, sz_tri));
  CUDA_CHECK(cudaMalloc(&d_refl, sz_refl));
  CUDA_CHECK(cudaMalloc(&d_delay, sz_delay));

  if (parallel) {
    size_t sz_order = triangle_order.size() * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_order, sz_order));
    CUDA_CHECK(cudaMalloc(&d_partial, (size_t)total * tiles * sizeof(MultipathCandidate)));
    CUDA_CHECK(cudaMemcpy(d_order, triangle_order.data(), sz_order, cudaMemcpyHostToDevice));
  }

  CUDA_CHECK(cudaMemcpy(d_rx, rx_ecef, sz_rx, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  if (!parallel) CUDA_CHECK(cudaMemcpy(d_bvh, bvh, sz_bvh, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tri, sorted_tris, sz_tri, cudaMemcpyHostToDevice));

  if (parallel) {
    multipath_candidates_kernel<false, false><<<total * tiles, MULTIPATH_BLOCK_SIZE>>>(
        d_rx, d_sat, d_tri, d_order, d_partial,
        n_sat, (int)triangle_order.size(), tiles, nullptr, nullptr);
    CUDA_CHECK_LAST();
    int grid = (total + MULTIPATH_BLOCK_SIZE - 1) / MULTIPATH_BLOCK_SIZE;
    multipath_finalize_kernel<<<grid, MULTIPATH_BLOCK_SIZE>>>(
        d_partial, d_refl, d_delay, total, tiles);
  } else {
    int block = 256;
    int grid = (total + block - 1) / block;
    multipath_bvh_batch_kernel<<<grid, block>>>(d_rx, d_sat, d_bvh, d_tri,
                                                  d_refl, d_delay,
                                                  n_epoch, n_sat, n_nodes);
  }
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(reflection_points, d_refl, sz_refl, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(excess_delays, d_delay, sz_delay, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_rx));
  CUDA_CHECK(cudaFree(d_sat));
  if (d_bvh) CUDA_CHECK(cudaFree(d_bvh));
  if (d_order) CUDA_CHECK(cudaFree(d_order));
  if (d_partial) CUDA_CHECK(cudaFree(d_partial));
  CUDA_CHECK(cudaFree(d_tri));
  CUDA_CHECK(cudaFree(d_refl));
  CUDA_CHECK(cudaFree(d_delay));
}

void raytrace_los_check_bvh(const double* rx_ecef, const double* sat_ecef,
                             const BVHNode* bvh, const Triangle* sorted_tris,
                             int* is_los, int n_sat, int n_nodes) {
  size_t sz_rx = 3 * sizeof(double);
  size_t sz_sat = (size_t)n_sat * 3 * sizeof(double);
  size_t sz_bvh = (size_t)n_nodes * sizeof(BVHNode);
  size_t sz_los = (size_t)n_sat * sizeof(int);

  // Count total triangles from leaf nodes
  int total_tris = 0;
  for (int i = 0; i < n_nodes; i++) {
    if (bvh[i].left == -1) {
      int end = bvh[i].tri_start + bvh[i].tri_count;
      if (end > total_tris) total_tris = end;
    }
  }
  size_t sz_tri = (size_t)total_tris * sizeof(Triangle);

  double *d_rx, *d_sat;
  BVHNode* d_bvh;
  Triangle* d_tri;
  int* d_los;

  CUDA_CHECK(cudaMalloc(&d_rx, sz_rx));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_bvh, sz_bvh));
  CUDA_CHECK(cudaMalloc(&d_tri, sz_tri));
  CUDA_CHECK(cudaMalloc(&d_los, sz_los));

  CUDA_CHECK(cudaMemcpy(d_rx, rx_ecef, sz_rx, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bvh, bvh, sz_bvh, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tri, sorted_tris, sz_tri, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_sat + block - 1) / block;
  los_check_bvh_kernel<<<grid, block>>>(d_rx, d_sat, d_bvh, d_tri, d_los,
                                         n_sat, n_nodes);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(is_los, d_los, sz_los, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_rx));
  CUDA_CHECK(cudaFree(d_sat));
  CUDA_CHECK(cudaFree(d_bvh));
  CUDA_CHECK(cudaFree(d_tri));
  CUDA_CHECK(cudaFree(d_los));
}

}  // namespace gnss_gpu
