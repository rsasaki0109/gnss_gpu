#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/pf_3d_bvh.h"
#include <cmath>
#include <cstdio>

namespace gnss_gpu {

// Maximum satellites supported in shared memory
static constexpr int MAX_SATS_BVH = 64;

// Maximum BVH traversal stack depth
static constexpr int BVH_STACK_DEPTH = 64;

// -----------------------------------------------------------------------
// Device helpers: ray-AABB slab test and Moller-Trumbore intersection.
// These are inlined directly (no separate translation unit dependency needed
// on device side).
// -----------------------------------------------------------------------

__device__ static bool bvh_ray_aabb(const double origin[3],
                                     const double dir_inv[3],
                                     double tmin, double tmax,
                                     const AABB& box) {
  for (int d = 0; d < 3; d++) {
    double t1 = (box.min[d] - origin[d]) * dir_inv[d];
    double t2 = (box.max[d] - origin[d]) * dir_inv[d];
    if (t1 > t2) { double tmp = t1; t1 = t2; t2 = tmp; }
    if (t1 > tmin) tmin = t1;
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return false;
  }
  return true;
}

__device__ static bool bvh_mt_hit(const double origin[3], const double dir[3],
                                   const double v0[3], const double v1[3],
                                   const double v2[3], double max_t) {
  const double EPS = 1e-12;
  double e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  double e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
  double h[3]  = {dir[1]*e2[2] - dir[2]*e2[1],
                   dir[2]*e2[0] - dir[0]*e2[2],
                   dir[0]*e2[1] - dir[1]*e2[0]};
  double a = e1[0]*h[0] + e1[1]*h[1] + e1[2]*h[2];
  if (fabs(a) < EPS) return false;
  double f = 1.0 / a;
  double s[3] = {origin[0] - v0[0], origin[1] - v0[1], origin[2] - v0[2]};
  double u = f * (s[0]*h[0] + s[1]*h[1] + s[2]*h[2]);
  if (u < 0.0 || u > 1.0) return false;
  double q[3] = {s[1]*e1[2] - s[2]*e1[1],
                  s[2]*e1[0] - s[0]*e1[2],
                  s[0]*e1[1] - s[1]*e1[0]};
  double v = f * (dir[0]*q[0] + dir[1]*q[1] + dir[2]*q[2]);
  if (v < 0.0 || u + v > 1.0) return false;
  double t = f * (e2[0]*q[0] + e2[1]*q[1] + e2[2]*q[2]);
  return (t > 1e-6) && (t < max_t);
}

__device__ static double logaddexp2(double a, double b) {
  double m = (a > b) ? a : b;
  return m + log(exp(a - m) + exp(b - m));
}

__device__ static double mixed_log_likelihood(
    double los_loglik,
    double nlos_loglik,
    double nlos_prob) {
  if (nlos_prob <= 0.0) return los_loglik;
  if (nlos_prob >= 1.0) return nlos_loglik;
  return logaddexp2(log(1.0 - nlos_prob) + los_loglik,
                    log(nlos_prob) + nlos_loglik);
}

// -----------------------------------------------------------------------
// BVH traversal: returns true if any triangle blocks the ray from origin
// to origin + dir*dist (exclusive endpoints).
// Traversal uses an explicit stack of max depth BVH_STACK_DEPTH.
// -----------------------------------------------------------------------
__device__ static bool bvh_is_blocked(const double origin[3],
                                       const double dir[3],
                                       const double dir_inv[3],
                                       double dist,
                                       const BVHNode* __restrict__ bvh,
                                       const Triangle* __restrict__ tris) {
  int stack[BVH_STACK_DEPTH];
  int sp = 0;
  stack[sp++] = 0;  // root node

  while (sp > 0) {
    int node_idx = stack[--sp];
    const BVHNode& node = bvh[node_idx];

    if (!bvh_ray_aabb(origin, dir_inv, 0.0, dist, node.bbox))
      continue;

    if (node.left == -1) {
      // Leaf: test each triangle
      for (int i = 0; i < node.tri_count; i++) {
        const Triangle& tri = tris[node.tri_start + i];
        if (bvh_mt_hit(origin, dir, tri.v0, tri.v1, tri.v2, dist))
          return true;
      }
    } else {
      // Internal: push children (guard against stack overflow)
      if (sp < BVH_STACK_DEPTH - 1) {
        stack[sp++] = node.left;
        stack[sp++] = node.right;
      }
    }
  }
  return false;
}

// -----------------------------------------------------------------------
// Main kernel: one thread per particle.
//
// For each particle the kernel loops over all satellites and performs a
// BVH-accelerated ray cast to determine LOS/NLOS.  The appropriate
// Gaussian log-likelihood (tight sigma for LOS, loose sigma for NLOS with
// positive-only bias correction) is accumulated into log_weights[pid].
//
// Satellite data (small, invariant across particles) is loaded cooperatively
// into shared memory.  BVH nodes and triangles are accessed from read-only
// global memory (L1/L2 cache friendly because all threads share the same
// BVH pointer).
// -----------------------------------------------------------------------
__global__ void weight_3d_bvh_kernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const double* __restrict__ pz,
    const double* __restrict__ pcb,
    const double* __restrict__ sat_ecef,
    const double* __restrict__ pseudoranges,
    const double* __restrict__ weights_sat,
    const BVHNode* __restrict__ bvh,
    const Triangle* __restrict__ sorted_tris,
    int n_nodes,
    double* __restrict__ log_weights,
    int N, int n_sat,
    double sigma_los, double sigma_nlos, double nlos_bias,
    double blocked_nlos_prob, double clear_nlos_prob) {

  // Shared memory for satellite data (invariant across the block)
  __shared__ double s_sat[MAX_SATS_BVH * 3];
  __shared__ double s_pr[MAX_SATS_BVH];
  __shared__ double s_ws[MAX_SATS_BVH];

  for (int i = threadIdx.x; i < n_sat; i += blockDim.x) {
    s_sat[i * 3 + 0] = sat_ecef[i * 3 + 0];
    s_sat[i * 3 + 1] = sat_ecef[i * 3 + 1];
    s_sat[i * 3 + 2] = sat_ecef[i * 3 + 2];
    s_pr[i] = pseudoranges[i];
    s_ws[i] = weights_sat[i];
  }
  __syncthreads();

  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= N) return;

  double x  = px[pid];
  double y  = py[pid];
  double z  = pz[pid];
  double cb = pcb[pid];

  double inv_sigma_los2  = 1.0 / (sigma_los  * sigma_los);
  double inv_sigma_nlos2 = 1.0 / (sigma_nlos * sigma_nlos);

  double log_w = 0.0;

  for (int s = 0; s < n_sat; s++) {
    double sx = s_sat[s * 3 + 0];
    double sy = s_sat[s * 3 + 1];
    double sz = s_sat[s * 3 + 2];

    double dx = sx - x;
    double dy = sy - y;
    double dz = sz - z;
    double dist = sqrt(dx*dx + dy*dy + dz*dz);

    double pred_pr = dist + cb;
    double obs_pr  = s_pr[s];

    // Unit direction from particle to satellite
    double inv_dist = 1.0 / dist;
    double dir[3]     = {dx * inv_dist, dy * inv_dist, dz * inv_dist};
    double dir_inv[3];
    for (int d = 0; d < 3; d++) {
      dir_inv[d] = (fabs(dir[d]) > 1e-20)
                   ? (1.0 / dir[d])
                   : ((dir[d] >= 0.0) ? 1e20 : -1e20);
    }
    double origin[3] = {x, y, z};

    // BVH traversal to determine LOS/NLOS
    // With an empty mesh (n_nodes == 0) every satellite is treated as LOS.
    bool is_nlos = (n_nodes > 0) &&
                   bvh_is_blocked(origin, dir, dir_inv, dist, bvh, sorted_tris);

    // Compute likelihood contribution
    double residual = obs_pr - pred_pr;
    double los_loglik = -0.5 * s_ws[s] * residual * residual * inv_sigma_los2;

    double residual_nlos = residual;
    if (residual_nlos > 0.0) {
      residual_nlos -= nlos_bias;
    }
    double nlos_loglik =
        -0.5 * s_ws[s] * residual_nlos * residual_nlos * inv_sigma_nlos2;

    if (is_nlos) {
      log_w += mixed_log_likelihood(
          los_loglik, nlos_loglik, blocked_nlos_prob);
    } else {
      log_w += mixed_log_likelihood(
          los_loglik, nlos_loglik, clear_nlos_prob);
    }
  }

  log_weights[pid] = log_w;
}

// -----------------------------------------------------------------------
// Host wrapper
// -----------------------------------------------------------------------
void pf_weight_3d_bvh(
    const double* px, const double* py, const double* pz, const double* pcb,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    const BVHNode* bvh_nodes, int n_nodes,
    const Triangle* sorted_tris, int n_tri,
    double* log_weights,
    int n_particles, int n_sat,
    double sigma_pr_los, double sigma_pr_nlos, double nlos_bias,
    double blocked_nlos_prob, double clear_nlos_prob) {

  const size_t sz      = (size_t)n_particles * sizeof(double);
  const size_t sz_sat  = (size_t)n_sat * 3 * sizeof(double);
  const size_t sz_obs  = (size_t)n_sat * sizeof(double);
  const size_t sz_bvh  = (size_t)n_nodes * sizeof(BVHNode);
  const size_t sz_tri  = (size_t)n_tri * sizeof(Triangle);

  double *d_px, *d_py, *d_pz, *d_pcb;
  double *d_sat, *d_pr, *d_ws, *d_lw;
  BVHNode*  d_bvh = nullptr;
  Triangle* d_tri = nullptr;

  CUDA_CHECK(cudaMalloc(&d_px,  sz));
  CUDA_CHECK(cudaMalloc(&d_py,  sz));
  CUDA_CHECK(cudaMalloc(&d_pz,  sz));
  CUDA_CHECK(cudaMalloc(&d_pcb, sz));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_pr,  sz_obs));
  CUDA_CHECK(cudaMalloc(&d_ws,  sz_obs));
  CUDA_CHECK(cudaMalloc(&d_lw,  sz));

  CUDA_CHECK(cudaMemcpy(d_px,  px,           sz,     cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_py,  py,           sz,     cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pz,  pz,           sz,     cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pcb, pcb,          sz,     cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef,     sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pr,  pseudoranges, sz_obs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ws,  weights_sat,  sz_obs, cudaMemcpyHostToDevice));

  if (n_nodes > 0) {
    CUDA_CHECK(cudaMalloc(&d_bvh, sz_bvh));
    CUDA_CHECK(cudaMemcpy(d_bvh, bvh_nodes, sz_bvh, cudaMemcpyHostToDevice));
  }
  if (n_tri > 0) {
    CUDA_CHECK(cudaMalloc(&d_tri, sz_tri));
    CUDA_CHECK(cudaMemcpy(d_tri, sorted_tris, sz_tri, cudaMemcpyHostToDevice));
  }

  const int block = 256;
  const int grid  = (n_particles + block - 1) / block;

  weight_3d_bvh_kernel<<<grid, block>>>(
      d_px, d_py, d_pz, d_pcb,
      d_sat, d_pr, d_ws,
      d_bvh, d_tri, n_nodes,
      d_lw,
      n_particles, n_sat,
      sigma_pr_los, sigma_pr_nlos, nlos_bias,
      blocked_nlos_prob, clear_nlos_prob);

  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(log_weights, d_lw, sz, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_px));
  CUDA_CHECK(cudaFree(d_py));
  CUDA_CHECK(cudaFree(d_pz));
  CUDA_CHECK(cudaFree(d_pcb));
  CUDA_CHECK(cudaFree(d_sat));
  CUDA_CHECK(cudaFree(d_pr));
  CUDA_CHECK(cudaFree(d_ws));
  CUDA_CHECK(cudaFree(d_lw));
  if (d_bvh) CUDA_CHECK(cudaFree(d_bvh));
  if (d_tri) CUDA_CHECK(cudaFree(d_tri));
}

}  // namespace gnss_gpu
