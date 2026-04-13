#include "gnss_gpu/bvh.h"
#include "gnss_gpu/cuda_check.h"
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <algorithm>
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

// Recursive top-down build. indices[start..end) are the triangle indices to partition.
static int build_recursive(const Triangle* triangles,
                           std::vector<int>& indices, int start, int end,
                           std::vector<BVHNode>& nodes) {
  int count = end - start;
  // Compute bounds of all triangles in range
  AABB bounds = triangle_aabb(triangles[indices[start]]);
  for (int i = start + 1; i < end; i++) {
    bounds = union_aabb(bounds, triangle_aabb(triangles[indices[i]]));
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
      double c = triangle_centroid(triangles[indices[i]], axis);
      int b = (int)((c - axis_min) / bin_width);
      if (b < 0) b = 0;
      if (b >= SAH_BINS) b = SAH_BINS - 1;
      bin_count[b]++;
      AABB tb = triangle_aabb(triangles[indices[i]]);
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
              return triangle_centroid(triangles[a], axis) <
                     triangle_centroid(triangles[b], axis);
            });

  // Find best SAH split on sorted array
  // Build prefix AABBs
  std::vector<AABB> prefix_box(count);
  prefix_box[0] = triangle_aabb(triangles[indices[start]]);
  for (int i = 1; i < count; i++) {
    prefix_box[i] = union_aabb(prefix_box[i - 1],
                                triangle_aabb(triangles[indices[start + i]]));
  }

  std::vector<AABB> suffix_box(count);
  suffix_box[count - 1] = triangle_aabb(triangles[indices[start + count - 1]]);
  for (int i = count - 2; i >= 0; i--) {
    suffix_box[i] = union_aabb(suffix_box[i + 1],
                                triangle_aabb(triangles[indices[start + i]]));
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

  int left_child = build_recursive(triangles, indices, start, best_split, nodes);
  int right_child = build_recursive(triangles, indices, best_split, end, nodes);

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

  build_recursive(triangles, indices, 0, n_tri, node_vec);

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

// ============================================================
// BVH-accelerated multipath reflection
// ============================================================

// 1 thread per satellite.  Traverses ALL BVH leaves (no early exit) to find
// the first-order reflection with the smallest positive excess delay.
__global__ void multipath_bvh_kernel(const double* rx_ecef,
                                      const double* sat_ecef,
                                      const BVHNode* bvh,
                                      const Triangle* tris,
                                      double* reflection_points,
                                      double* excess_delays,
                                      int n_sat, int n_nodes) {
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= n_sat) return;

  double rx[3] = {rx_ecef[0], rx_ecef[1], rx_ecef[2]};
  double sat[3] = {sat_ecef[sid * 3 + 0], sat_ecef[sid * 3 + 1], sat_ecef[sid * 3 + 2]};

  double d_rx_sat = sqrt((rx[0] - sat[0]) * (rx[0] - sat[0]) +
                          (rx[1] - sat[1]) * (rx[1] - sat[1]) +
                          (rx[2] - sat[2]) * (rx[2] - sat[2]));

  double best_delay = 0.0;
  double best_refl[3] = {0.0, 0.0, 0.0};
  bool found = false;

  // Stack-based BVH traversal — visit ALL leaves
  int stack[64];
  int sp = 0;
  stack[sp++] = 0;

  while (sp > 0) {
    int node_idx = stack[--sp];
    const BVHNode& node = bvh[node_idx];

    // Skip subtrees that are too far (use a generous AABB check)
    // We need to visit triangles near the rx-sat line, so check AABB
    // against a bounding box of {rx, sat} expanded a bit
    // For simplicity, skip AABB culling here — traverse all leaves

    if (node.left == -1) {
      // Leaf: test each triangle for mirror reflection
      for (int i = 0; i < node.tri_count; i++) {
        const Triangle& tri = tris[node.tri_start + i];

        // Compute triangle plane normal
        double e1[3] = {tri.v1[0] - tri.v0[0], tri.v1[1] - tri.v0[1], tri.v1[2] - tri.v0[2]};
        double e2[3] = {tri.v2[0] - tri.v0[0], tri.v2[1] - tri.v0[1], tri.v2[2] - tri.v0[2]};
        double n[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0]};
        double nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (nn < 1e-15) continue;
        n[0] /= nn; n[1] /= nn; n[2] /= nn;

        // Mirror rx across triangle plane
        double dv[3] = {rx[0] - tri.v0[0], rx[1] - tri.v0[1], rx[2] - tri.v0[2]};
        double d = dv[0] * n[0] + dv[1] * n[1] + dv[2] * n[2];
        double mirror[3] = {rx[0] - 2.0 * d * n[0],
                             rx[1] - 2.0 * d * n[1],
                             rx[2] - 2.0 * d * n[2]};

        // Ray from mirror to satellite
        double dir[3] = {sat[0] - mirror[0], sat[1] - mirror[1], sat[2] - mirror[2]};
        double dir_len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        if (dir_len < 1e-15) continue;
        dir[0] /= dir_len; dir[1] /= dir_len; dir[2] /= dir_len;

        // Intersect mirror->sat ray with this triangle
        double t;
        if (!mt_intersect(mirror, dir, tri.v0, tri.v1, tri.v2, t)) continue;

        // Reflection point
        double refl[3] = {mirror[0] + t * dir[0],
                           mirror[1] + t * dir[1],
                           mirror[2] + t * dir[2]};

        // Excess delay = (|rx->refl| + |refl->sat|) - |rx->sat|
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
      // Internal: push children
      if (sp < 62) {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    }
  }

  excess_delays[sid] = best_delay;
  reflection_points[sid * 3 + 0] = best_refl[0];
  reflection_points[sid * 3 + 1] = best_refl[1];
  reflection_points[sid * 3 + 2] = best_refl[2];
}

void raytrace_multipath_bvh(const double* rx_ecef, const double* sat_ecef,
                             const BVHNode* bvh, const Triangle* sorted_tris,
                             double* reflection_points, double* excess_delays,
                             int n_sat, int n_nodes) {
  size_t sz_rx = 3 * sizeof(double);
  size_t sz_sat = (size_t)n_sat * 3 * sizeof(double);
  size_t sz_bvh = (size_t)n_nodes * sizeof(BVHNode);
  size_t sz_refl = (size_t)n_sat * 3 * sizeof(double);
  size_t sz_delay = (size_t)n_sat * sizeof(double);

  int total_tris = 0;
  for (int i = 0; i < n_nodes; i++) {
    if (bvh[i].left == -1) {
      int end = bvh[i].tri_start + bvh[i].tri_count;
      if (end > total_tris) total_tris = end;
    }
  }
  size_t sz_tri = (size_t)total_tris * sizeof(Triangle);

  double *d_rx, *d_sat, *d_refl, *d_delay;
  BVHNode* d_bvh;
  Triangle* d_tri;

  CUDA_CHECK(cudaMalloc(&d_rx, sz_rx));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_bvh, sz_bvh));
  CUDA_CHECK(cudaMalloc(&d_tri, sz_tri));
  CUDA_CHECK(cudaMalloc(&d_refl, sz_refl));
  CUDA_CHECK(cudaMalloc(&d_delay, sz_delay));

  CUDA_CHECK(cudaMemcpy(d_rx, rx_ecef, sz_rx, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bvh, bvh, sz_bvh, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tri, sorted_tris, sz_tri, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_sat + block - 1) / block;
  multipath_bvh_kernel<<<grid, block>>>(d_rx, d_sat, d_bvh, d_tri,
                                         d_refl, d_delay, n_sat, n_nodes);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(reflection_points, d_refl, sz_refl, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(excess_delays, d_delay, sz_delay, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_rx));
  CUDA_CHECK(cudaFree(d_sat));
  CUDA_CHECK(cudaFree(d_bvh));
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
