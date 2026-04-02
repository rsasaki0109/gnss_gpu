#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/pf_3d.h"
#include <cmath>
#include <cstdio>

namespace gnss_gpu {

// Maximum satellites supported in shared memory (must match weight.cu)
static constexpr int MAX_SATS = 64;

// -----------------------------------------------------------------------
// Inline Moller-Trumbore intersection for the combined weight+raytrace kernel.
// Returns true if ray(origin, dir) intersects triangle(v0, v1, v2) and
// the hit distance t is in (eps, max_t).
// -----------------------------------------------------------------------
__device__ static bool moller_trumbore_inline(
    const double origin[3], const double dir[3],
    const double v0[3], const double v1[3], const double v2[3],
    double max_t) {
  const double EPSILON = 1e-12;

  double e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  double e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

  // h = dir x e2
  double h[3] = {dir[1] * e2[2] - dir[2] * e2[1],
                  dir[2] * e2[0] - dir[0] * e2[2],
                  dir[0] * e2[1] - dir[1] * e2[0]};

  double a = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
  if (fabs(a) < EPSILON) return false;

  double f = 1.0 / a;
  double s[3] = {origin[0] - v0[0], origin[1] - v0[1], origin[2] - v0[2]};

  double u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
  if (u < 0.0 || u > 1.0) return false;

  // q = s x e1
  double q[3] = {s[1] * e1[2] - s[2] * e1[1],
                  s[2] * e1[0] - s[0] * e1[2],
                  s[0] * e1[1] - s[1] * e1[0]};

  double v = f * (dir[0] * q[0] + dir[1] * q[1] + dir[2] * q[2]);
  if (v < 0.0 || u + v > 1.0) return false;

  double t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);
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
// Combined weight + ray tracing kernel.
//
// Each thread processes one particle.  For every satellite the thread:
//   1. Computes predicted pseudorange.
//   2. Casts a ray from the particle to the satellite.
//   3. Tests intersections against building triangles (read from global
//      memory via L1/L2 cache -- triangle data is read-only and shared
//      across all threads, so the cache is effective).
//   4. Applies an appropriate Gaussian likelihood: tight sigma for LOS,
//      loose sigma for NLOS, with positive bias correction applied only when
//      the residual itself is positive.
//
// Note: Shared memory is used only for satellite data (small, fits easily).
// Triangle data is accessed from global memory to avoid __syncthreads()
// hazards when threads have different early-exit patterns.  For large
// triangle counts a BVH acceleration structure would be the next step.
// -----------------------------------------------------------------------
__global__ void weight_3d_kernel(
    const double* __restrict__ px,
    const double* __restrict__ py,
    const double* __restrict__ pz,
    const double* __restrict__ pcb,
    const double* __restrict__ sat_ecef,
    const double* __restrict__ pseudoranges,
    const double* __restrict__ weights_sat,
    const Triangle* __restrict__ triangles,
    int n_tri,
    double* __restrict__ log_weights,
    int N, int n_sat,
    double sigma_los, double sigma_nlos, double nlos_bias,
    double blocked_nlos_prob, double clear_nlos_prob) {

  // Shared memory for satellite data only
  __shared__ double s_sat[MAX_SATS * 3];
  __shared__ double s_pr[MAX_SATS];
  __shared__ double s_ws[MAX_SATS];

  // Cooperative load of satellite data (invariant across particles)
  for (int i = threadIdx.x; i < n_sat; i += blockDim.x) {
    s_sat[i * 3 + 0] = sat_ecef[i * 3 + 0];
    s_sat[i * 3 + 1] = sat_ecef[i * 3 + 1];
    s_sat[i * 3 + 2] = sat_ecef[i * 3 + 2];
    s_pr[i]  = pseudoranges[i];
    s_ws[i]  = weights_sat[i];
  }
  __syncthreads();

  int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= N) return;

  double x  = px[pid];
  double y  = py[pid];
  double z  = pz[pid];
  double cb = pcb[pid];

  double inv_sigma_los2  = 1.0 / (sigma_los * sigma_los);
  double inv_sigma_nlos2 = 1.0 / (sigma_nlos * sigma_nlos);

  double log_w = 0.0;

  for (int s = 0; s < n_sat; s++) {
    double sx = s_sat[s * 3 + 0];
    double sy = s_sat[s * 3 + 1];
    double sz = s_sat[s * 3 + 2];

    double dx = x - sx;
    double dy = y - sy;
    double dz = z - sz;
    double dist = sqrt(dx * dx + dy * dy + dz * dz);

    double pred_pr = dist + cb;
    double obs_pr  = s_pr[s];

    // Ray direction: particle -> satellite (unit vector)
    double inv_dist = 1.0 / dist;
    double dir[3] = {(sx - x) * inv_dist,
                     (sy - y) * inv_dist,
                     (sz - z) * inv_dist};
    double origin[3] = {x, y, z};

    // Determine LOS/NLOS by testing against all triangles
    bool is_nlos = false;
    for (int t = 0; t < n_tri; t++) {
      if (moller_trumbore_inline(origin, dir,
                                  triangles[t].v0, triangles[t].v1,
                                  triangles[t].v2, dist)) {
        is_nlos = true;
        break;
      }
    }

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
void pf_weight_3d(
    const double* px, const double* py, const double* pz, const double* pcb,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    const Triangle* triangles, int n_tri,
    double* log_weights,
    int n_particles, int n_sat,
    double sigma_pr_los, double sigma_pr_nlos, double nlos_bias,
    double blocked_nlos_prob, double clear_nlos_prob) {

  size_t sz     = (size_t)n_particles * sizeof(double);
  size_t sz_sat = (size_t)n_sat * 3 * sizeof(double);
  size_t sz_obs = (size_t)n_sat * sizeof(double);
  size_t sz_tri = (size_t)n_tri * sizeof(Triangle);

  double *d_px, *d_py, *d_pz, *d_pcb;
  double *d_sat, *d_pr, *d_ws, *d_lw;
  Triangle* d_tri = nullptr;

  CUDA_CHECK(cudaMalloc(&d_px, sz));
  CUDA_CHECK(cudaMalloc(&d_py, sz));
  CUDA_CHECK(cudaMalloc(&d_pz, sz));
  CUDA_CHECK(cudaMalloc(&d_pcb, sz));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_pr, sz_obs));
  CUDA_CHECK(cudaMalloc(&d_ws, sz_obs));
  CUDA_CHECK(cudaMalloc(&d_lw, sz));

  CUDA_CHECK(cudaMemcpy(d_px,  px,          sz,     cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_py,  py,          sz,     cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pz,  pz,          sz,     cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pcb, pcb,         sz,     cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef,    sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pr,  pseudoranges,sz_obs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ws,  weights_sat, sz_obs, cudaMemcpyHostToDevice));

  if (n_tri > 0) {
    CUDA_CHECK(cudaMalloc(&d_tri, sz_tri));
    CUDA_CHECK(cudaMemcpy(d_tri, triangles, sz_tri, cudaMemcpyHostToDevice));
  }

  int block = 256;
  int grid  = (n_particles + block - 1) / block;

  weight_3d_kernel<<<grid, block>>>(
      d_px, d_py, d_pz, d_pcb,
      d_sat, d_pr, d_ws,
      d_tri, n_tri,
      d_lw,
      n_particles, n_sat,
      sigma_pr_los, sigma_pr_nlos, nlos_bias,
      blocked_nlos_prob, clear_nlos_prob);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "weight_3d_kernel launch error: %s\n",
            cudaGetErrorString(err));
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "weight_3d_kernel sync error: %s\n",
            cudaGetErrorString(err));
  }

  CUDA_CHECK(cudaMemcpy(log_weights, d_lw, sz, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_px));
  CUDA_CHECK(cudaFree(d_py));
  CUDA_CHECK(cudaFree(d_pz));
  CUDA_CHECK(cudaFree(d_pcb));
  CUDA_CHECK(cudaFree(d_sat));
  CUDA_CHECK(cudaFree(d_pr));
  CUDA_CHECK(cudaFree(d_ws));
  CUDA_CHECK(cudaFree(d_lw));
  if (d_tri) CUDA_CHECK(cudaFree(d_tri));
}

}  // namespace gnss_gpu
