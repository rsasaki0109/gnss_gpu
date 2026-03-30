#include "gnss_gpu/raytrace.h"
#include "gnss_gpu/cuda_check.h"
#include <cmath>
#include <cfloat>
#include <cstdio>

namespace gnss_gpu {

// Moller-Trumbore ray-triangle intersection
// Returns true if ray(origin, dir) hits triangle(v0, v1, v2), sets t to hit distance
__device__ bool moller_trumbore(const double origin[3], const double dir[3],
                                const double v0[3], const double v1[3],
                                const double v2[3], double& t) {
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

  t = f * (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]);
  return t > EPSILON;
}

// LOS check kernel: 1 thread per satellite
__global__ void los_check_kernel(const double* rx_ecef, const double* sat_ecef,
                                  const Triangle* triangles, int* is_los,
                                  int n_sat, int n_tri) {
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= n_sat) return;

  double ox = rx_ecef[0], oy = rx_ecef[1], oz = rx_ecef[2];
  double sx = sat_ecef[sid * 3 + 0];
  double sy = sat_ecef[sid * 3 + 1];
  double sz = sat_ecef[sid * 3 + 2];

  double dir[3] = {sx - ox, sy - oy, sz - oz};
  double dist = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  dir[0] /= dist; dir[1] /= dist; dir[2] /= dist;

  double origin[3] = {ox, oy, oz};
  bool blocked = false;


  for (int i = 0; i < n_tri; i++) {
    double t;
    if (moller_trumbore(origin, dir, triangles[i].v0, triangles[i].v1,
                        triangles[i].v2, t)) {
      if (t < dist) {
        blocked = true;
        break;
      }
    }
  }

  is_los[sid] = blocked ? 0 : 1;
}

// Multipath kernel: 1 thread per (satellite x triangle) pair
// Writes per-thread results to temporary arrays (no cross-thread races)
__global__ void multipath_kernel(const double* rx_ecef, const double* sat_ecef,
                                  const Triangle* triangles,
                                  double* temp_refl, double* temp_delay,
                                  int n_sat, int n_tri) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_sat * n_tri;
  if (idx >= total) return;

  int sid = idx / n_tri;
  int tid = idx % n_tri;

  // Default: no reflection (delay = 0)
  temp_delay[idx] = 0.0;
  temp_refl[idx * 3 + 0] = 0.0;
  temp_refl[idx * 3 + 1] = 0.0;
  temp_refl[idx * 3 + 2] = 0.0;

  double rx[3] = {rx_ecef[0], rx_ecef[1], rx_ecef[2]};
  double sat[3] = {sat_ecef[sid * 3 + 0], sat_ecef[sid * 3 + 1], sat_ecef[sid * 3 + 2]};

  const Triangle& tri = triangles[tid];

  // Compute triangle plane normal
  double e1[3] = {tri.v1[0] - tri.v0[0], tri.v1[1] - tri.v0[1], tri.v1[2] - tri.v0[2]};
  double e2[3] = {tri.v2[0] - tri.v0[0], tri.v2[1] - tri.v0[1], tri.v2[2] - tri.v0[2]};
  double n[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                  e1[2] * e2[0] - e1[0] * e2[2],
                  e1[0] * e2[1] - e1[1] * e2[0]};
  double nn = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (nn < 1e-15) return;
  n[0] /= nn; n[1] /= nn; n[2] /= nn;

  // Mirror rx across triangle plane
  // d = dot(rx - v0, n)
  double dv[3] = {rx[0] - tri.v0[0], rx[1] - tri.v0[1], rx[2] - tri.v0[2]};
  double d = dv[0] * n[0] + dv[1] * n[1] + dv[2] * n[2];

  double mirror[3] = {rx[0] - 2.0 * d * n[0],
                       rx[1] - 2.0 * d * n[1],
                       rx[2] - 2.0 * d * n[2]};

  // Ray from mirror to satellite
  double dir[3] = {sat[0] - mirror[0], sat[1] - mirror[1], sat[2] - mirror[2]};
  double dir_len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  if (dir_len < 1e-15) return;
  dir[0] /= dir_len; dir[1] /= dir_len; dir[2] /= dir_len;

  // Find reflection point: intersect mirror->sat ray with triangle
  double t;
  if (!moller_trumbore(mirror, dir, tri.v0, tri.v1, tri.v2, t)) return;

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
  double d_rx_sat = sqrt((rx[0] - sat[0]) * (rx[0] - sat[0]) +
                          (rx[1] - sat[1]) * (rx[1] - sat[1]) +
                          (rx[2] - sat[2]) * (rx[2] - sat[2]));

  double excess = (d_rx_refl + d_refl_sat) - d_rx_sat;

  if (excess > 0.0) {
    temp_delay[idx] = excess;
    temp_refl[idx * 3 + 0] = refl[0];
    temp_refl[idx * 3 + 1] = refl[1];
    temp_refl[idx * 3 + 2] = refl[2];
  }
}

// Reduction kernel: find minimum positive delay per satellite
// 1 thread per satellite
__global__ void multipath_reduce_kernel(const double* temp_refl,
                                         const double* temp_delay,
                                         double* reflection_points,
                                         double* excess_delays,
                                         int n_sat, int n_tri) {
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= n_sat) return;

  double best_delay = 0.0;
  int best_tid = -1;

  for (int tid = 0; tid < n_tri; tid++) {
    int idx = sid * n_tri + tid;
    double d = temp_delay[idx];
    if (d > 0.0 && (best_tid < 0 || d < best_delay)) {
      best_delay = d;
      best_tid = tid;
    }
  }

  if (best_tid >= 0) {
    int idx = sid * n_tri + best_tid;
    excess_delays[sid] = best_delay;
    reflection_points[sid * 3 + 0] = temp_refl[idx * 3 + 0];
    reflection_points[sid * 3 + 1] = temp_refl[idx * 3 + 1];
    reflection_points[sid * 3 + 2] = temp_refl[idx * 3 + 2];
  } else {
    excess_delays[sid] = 0.0;
    reflection_points[sid * 3 + 0] = 0.0;
    reflection_points[sid * 3 + 1] = 0.0;
    reflection_points[sid * 3 + 2] = 0.0;
  }
}

// Host function: batch LOS check
void raytrace_los_check(const double* rx_ecef, const double* sat_ecef,
                        const Triangle* triangles, int* is_los,
                        int n_sat, int n_tri) {
  size_t sz_rx = 3 * sizeof(double);
  size_t sz_sat = (size_t)n_sat * 3 * sizeof(double);
  size_t sz_tri = (size_t)n_tri * sizeof(Triangle);
  size_t sz_los = (size_t)n_sat * sizeof(int);

  double *d_rx, *d_sat;
  Triangle* d_tri;
  int* d_los;

  CUDA_CHECK(cudaMalloc(&d_rx, sz_rx));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_tri, sz_tri));
  CUDA_CHECK(cudaMalloc(&d_los, sz_los));

  CUDA_CHECK(cudaMemcpy(d_rx, rx_ecef, sz_rx, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tri, triangles, sz_tri, cudaMemcpyHostToDevice));
  // Initialize all to 1 (LOS) - kernel will set to 0 if blocked
  for (int i = 0; i < n_sat; i++) is_los[i] = 1;

  int block = 256;
  int grid_dim = (n_sat + block - 1) / block;
  los_check_kernel<<<grid_dim, block>>>(d_rx, d_sat, d_tri, d_los, n_sat, n_tri);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(is_los, d_los, sz_los, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_rx)); CUDA_CHECK(cudaFree(d_sat));
}

// Host function: multipath reflection computation
void raytrace_multipath(const double* rx_ecef, const double* sat_ecef,
                        const Triangle* triangles, double* reflection_points,
                        double* excess_delays, int n_sat, int n_tri) {
  size_t sz_rx = 3 * sizeof(double);
  size_t sz_sat = (size_t)n_sat * 3 * sizeof(double);
  size_t sz_tri = (size_t)n_tri * sizeof(Triangle);
  size_t sz_refl = (size_t)n_sat * 3 * sizeof(double);
  size_t sz_delay = (size_t)n_sat * sizeof(double);
  size_t total = (size_t)n_sat * n_tri;
  size_t sz_temp_refl = total * 3 * sizeof(double);
  size_t sz_temp_delay = total * sizeof(double);

  double *d_rx, *d_sat, *d_refl, *d_delay, *d_temp_refl, *d_temp_delay;
  Triangle* d_tri;

  CUDA_CHECK(cudaMalloc(&d_rx, sz_rx));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_tri, sz_tri));
  CUDA_CHECK(cudaMalloc(&d_refl, sz_refl));
  CUDA_CHECK(cudaMalloc(&d_delay, sz_delay));
  CUDA_CHECK(cudaMalloc(&d_temp_refl, sz_temp_refl));
  CUDA_CHECK(cudaMalloc(&d_temp_delay, sz_temp_delay));

  CUDA_CHECK(cudaMemcpy(d_rx, rx_ecef, sz_rx, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_tri, triangles, sz_tri, cudaMemcpyHostToDevice));

  // Phase 1: each thread writes to its own slot in temp arrays (no races)
  int block = 256;
  int grid = ((int)total + block - 1) / block;
  multipath_kernel<<<grid, block>>>(d_rx, d_sat, d_tri, d_temp_refl, d_temp_delay,
                                     n_sat, n_tri);
  CUDA_CHECK_LAST();

  // Phase 2: reduce per satellite to find minimum positive delay
  int grid2 = (n_sat + block - 1) / block;
  multipath_reduce_kernel<<<grid2, block>>>(d_temp_refl, d_temp_delay,
                                             d_refl, d_delay, n_sat, n_tri);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaMemcpy(reflection_points, d_refl, sz_refl, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(excess_delays, d_delay, sz_delay, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_rx)); CUDA_CHECK(cudaFree(d_sat));
  CUDA_CHECK(cudaFree(d_tri));
  CUDA_CHECK(cudaFree(d_refl)); CUDA_CHECK(cudaFree(d_delay));
  CUDA_CHECK(cudaFree(d_temp_refl)); CUDA_CHECK(cudaFree(d_temp_delay));
}

}  // namespace gnss_gpu
