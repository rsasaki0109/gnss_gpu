#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/svgd.h"
#include <curand_kernel.h>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <vector>

namespace gnss_gpu {

// Maximum satellites supported in shared memory (matches weight.cu)
static constexpr int SVGD_MAX_SATS = 64;

// ============================================================================
// Bandwidth estimation via median heuristic on random subsample
// ============================================================================

__global__ void compute_pairwise_distances_subsample(
    const double* px, const double* py, const double* pz, const double* pcb,
    double* distances, int N, int M, unsigned long long seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= M) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, tid, 0, &state);

  // Pick two random distinct particles
  int i = (int)(curand_uniform_double(&state) * N) % N;
  int j = (int)(curand_uniform_double(&state) * (N - 1)) % N;
  if (j >= i) j++;

  double dx = px[i] - px[j];
  double dy = py[i] - py[j];
  double dz = pz[i] - pz[j];
  double dcb = pcb[i] - pcb[j];
  distances[tid] = sqrt(dx * dx + dy * dy + dz * dz + dcb * dcb);
}

// Simple bitonic-style partial sort on CPU is fine for M=1000
static double compute_median_cpu(double* data, int n) {
  std::sort(data, data + n);
  if (n % 2 == 0) {
    return 0.5 * (data[n / 2 - 1] + data[n / 2]);
  }
  return data[n / 2];
}

double pf_estimate_bandwidth(
    const double* px, const double* py, const double* pz, const double* pcb,
    int n_particles, int n_subsample,
    unsigned long long seed) {
  size_t sz = (size_t)n_particles * sizeof(double);
  size_t sz_dist = (size_t)n_subsample * sizeof(double);

  double *d_px, *d_py, *d_pz, *d_pcb, *d_distances;
  CUDA_CHECK(cudaMalloc(&d_px, sz));
  CUDA_CHECK(cudaMalloc(&d_py, sz));
  CUDA_CHECK(cudaMalloc(&d_pz, sz));
  CUDA_CHECK(cudaMalloc(&d_pcb, sz));
  CUDA_CHECK(cudaMalloc(&d_distances, sz_dist));

  CUDA_CHECK(cudaMemcpy(d_px, px, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_py, py, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pz, pz, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pcb, pcb, sz, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_subsample + block - 1) / block;
  compute_pairwise_distances_subsample<<<grid, block>>>(
      d_px, d_py, d_pz, d_pcb, d_distances, n_particles, n_subsample, seed);

  double* h_distances = new double[n_subsample];
  CUDA_CHECK(cudaMemcpy(h_distances, d_distances, sz_dist, cudaMemcpyDeviceToHost));

  double median_dist = compute_median_cpu(h_distances, n_subsample);

  // Median heuristic: h^2 = median^2 / (2 * log(N))
  // This is the standard SVGD bandwidth from Liu & Wang (2016).
  double log_n = log((double)n_particles);
  double bandwidth = median_dist / sqrt(2.0 * log_n);

  // Clamp to avoid degenerate values
  if (bandwidth < 1e-6) bandwidth = 1.0;

  delete[] h_distances;
  CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py));
  CUDA_CHECK(cudaFree(d_pz)); CUDA_CHECK(cudaFree(d_pcb));
  CUDA_CHECK(cudaFree(d_distances));

  return bandwidth;
}

// ============================================================================
// SVGD gradient computation and update
// ============================================================================

// Each thread computes the SVGD update direction for one particle,
// using K random neighbors.
//
// SVGD update for particle i:
//   phi(x_i) = (1/K) sum_{j in neighbors} [
//       k(x_j, x_i) * score(x_j)          // attraction term
//     + grad_{x_j} k(x_j, x_i)            // repulsion term
//   ]
//
// Score function: grad_x log p(x|y) = grad_x log p(y|x)
//   = sum_sat [ (pr_obs - pr_pred) / sigma^2 ] * (x - sat) / r
//
// RBF kernel: k(x, y) = exp(-||x - y||^2 / (2*h^2))
// grad_x k(x, y) = -k(x, y) * (x - y) / h^2

__global__ void svgd_gradient_kernel(
    const double* px, const double* py, const double* pz, const double* pcb,
    const double* sat_ecef, const double* pseudoranges, const double* weights_sat,
    double* grad_x, double* grad_y, double* grad_z, double* grad_cb,
    int N, int n_sat, double sigma_pr, double bandwidth,
    int n_neighbors, unsigned long long seed, int step) {

  // Load satellite data into shared memory
  __shared__ double s_sat[SVGD_MAX_SATS * 3];
  __shared__ double s_pr[SVGD_MAX_SATS];
  __shared__ double s_ws[SVGD_MAX_SATS];

  for (int i = threadIdx.x; i < n_sat; i += blockDim.x) {
    s_sat[i * 3 + 0] = sat_ecef[i * 3 + 0];
    s_sat[i * 3 + 1] = sat_ecef[i * 3 + 1];
    s_sat[i * 3 + 2] = sat_ecef[i * 3 + 2];
    s_pr[i] = pseudoranges[i];
    s_ws[i] = weights_sat[i];
  }
  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  double xi = px[tid];
  double yi = py[tid];
  double zi = pz[tid];
  double cbi = pcb[tid];

  double inv_sigma2 = 1.0 / (sigma_pr * sigma_pr);
  double inv_h2 = 1.0 / (bandwidth * bandwidth);

  // Initialize accumulators for phi(x_i)
  double phi_x = 0.0, phi_y = 0.0, phi_z = 0.0, phi_cb = 0.0;

  // Random number generator for selecting neighbors
  curandStatePhilox4_32_10_t rng_state;
  curand_init(seed, tid, step, &rng_state);

  for (int k = 0; k < n_neighbors; k++) {
    // Pick random neighbor j
    int offset = (int)(curand_uniform_double(&rng_state) * (N - 1)) + 1;
    int j = (tid + offset) % N;

    double xj = px[j];
    double yj = py[j];
    double zj = pz[j];
    double cbj = pcb[j];

    // --- Compute score function at x_j: grad_{x_j} log p(y | x_j) ---
    double score_x = 0.0, score_y = 0.0, score_z = 0.0, score_cb = 0.0;

    // First pass: compute mean weighted residual (common-mode clock bias error)
    double mean_coeff = 0.0;
    double total_w = 0.0;
    for (int s = 0; s < n_sat; s++) {
      double dx_s = xj - s_sat[s * 3 + 0];
      double dy_s = yj - s_sat[s * 3 + 1];
      double dz_s = zj - s_sat[s * 3 + 2];
      double r = sqrt(dx_s * dx_s + dy_s * dy_s + dz_s * dz_s);
      if (r < 1e-10) continue;
      double pred_pr = r + cbj;
      double residual = s_pr[s] - pred_pr;
      mean_coeff += s_ws[s] * residual * inv_sigma2;
      total_w += s_ws[s];
    }
    if (total_w > 0.0) mean_coeff /= total_w;

    // Second pass: use de-meaned residual for position, full for clock bias
    for (int s = 0; s < n_sat; s++) {
      double dx_s = xj - s_sat[s * 3 + 0];
      double dy_s = yj - s_sat[s * 3 + 1];
      double dz_s = zj - s_sat[s * 3 + 2];
      double r = sqrt(dx_s * dx_s + dy_s * dy_s + dz_s * dz_s);

      if (r < 1e-10) continue;

      double pred_pr = r + cbj;
      double residual = s_pr[s] - pred_pr;  // obs - pred
      double coeff = s_ws[s] * residual * inv_sigma2;

      // For position: remove common-mode residual (clock bias leakage)
      // This prevents the clock-bias error from creating a spurious
      // position gradient via the net direction-cosine bias.
      double coeff_pos = coeff - s_ws[s] * mean_coeff;

      // Score derivation:
      // log p(y|x) = -0.5 * w_sat * (obs - pred)^2 / sigma^2
      // pred = r + cb, where r = ||x - sat||
      // d/dx = w_sat * residual / sigma^2 * (x - sat) / r
      // d/dcb = w_sat * residual / sigma^2
      score_x += coeff_pos * dx_s / r;
      score_y += coeff_pos * dy_s / r;
      score_z += coeff_pos * dz_s / r;
      score_cb += coeff;
    }

    // --- Compute RBF kernel k(x_j, x_i) ---
    double dxji = xj - xi;
    double dyji = yj - yi;
    double dzji = zj - zi;
    double dcbji = cbj - cbi;
    double dist2 = dxji * dxji + dyji * dyji + dzji * dzji + dcbji * dcbji;
    double kval = exp(-0.5 * dist2 * inv_h2);

    // --- Compute grad_{x_j} k(x_j, x_i) ---
    // k = exp(-||x_j - x_i||^2 / (2*h^2))
    // grad_{x_j} k = k * (-(x_j - x_i) / h^2)
    double grad_k_x = kval * (-dxji * inv_h2);
    double grad_k_y = kval * (-dyji * inv_h2);
    double grad_k_z = kval * (-dzji * inv_h2);
    double grad_k_cb = kval * (-dcbji * inv_h2);

    // --- Accumulate SVGD update ---
    // phi(x_i) += k(x_j, x_i) * score(x_j) + grad_{x_j} k(x_j, x_i)
    phi_x += kval * score_x + grad_k_x;
    phi_y += kval * score_y + grad_k_y;
    phi_z += kval * score_z + grad_k_z;
    phi_cb += kval * score_cb + grad_k_cb;
  }

  // Average over K neighbors
  double inv_k = 1.0 / (double)n_neighbors;
  grad_x[tid] = phi_x * inv_k;
  grad_y[tid] = phi_y * inv_k;
  grad_z[tid] = phi_z * inv_k;
  grad_cb[tid] = phi_cb * inv_k;
}

__global__ void svgd_update_kernel(
    double* px, double* py, double* pz, double* pcb,
    const double* grad_x, const double* grad_y,
    const double* grad_z, const double* grad_cb,
    double step_size, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  px[tid] += step_size * grad_x[tid];
  py[tid] += step_size * grad_y[tid];
  pz[tid] += step_size * grad_z[tid];
  pcb[tid] += step_size * grad_cb[tid];
}

void pf_svgd_step(
    double* px, double* py, double* pz, double* pcb,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    int n_particles, int n_sat,
    double sigma_pr, double step_size,
    int n_neighbors, double bandwidth,
    unsigned long long seed, int step) {

  size_t sz = (size_t)n_particles * sizeof(double);
  size_t sz_sat = (size_t)n_sat * 3 * sizeof(double);
  size_t sz_obs = (size_t)n_sat * sizeof(double);

  double *d_px, *d_py, *d_pz, *d_pcb;
  double *d_sat, *d_pr, *d_ws;
  double *d_gx, *d_gy, *d_gz, *d_gcb;

  CUDA_CHECK(cudaMalloc(&d_px, sz));
  CUDA_CHECK(cudaMalloc(&d_py, sz));
  CUDA_CHECK(cudaMalloc(&d_pz, sz));
  CUDA_CHECK(cudaMalloc(&d_pcb, sz));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_pr, sz_obs));
  CUDA_CHECK(cudaMalloc(&d_ws, sz_obs));
  CUDA_CHECK(cudaMalloc(&d_gx, sz));
  CUDA_CHECK(cudaMalloc(&d_gy, sz));
  CUDA_CHECK(cudaMalloc(&d_gz, sz));
  CUDA_CHECK(cudaMalloc(&d_gcb, sz));

  CUDA_CHECK(cudaMemcpy(d_px, px, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_py, py, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pz, pz, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pcb, pcb, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pr, pseudoranges, sz_obs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ws, weights_sat, sz_obs, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_particles + block - 1) / block;

  // Step 1: Compute SVGD gradient for all particles
  svgd_gradient_kernel<<<grid, block>>>(
      d_px, d_py, d_pz, d_pcb,
      d_sat, d_pr, d_ws,
      d_gx, d_gy, d_gz, d_gcb,
      n_particles, n_sat, sigma_pr, bandwidth,
      n_neighbors, seed, step);

  // Step 2: Apply update
  svgd_update_kernel<<<grid, block>>>(
      d_px, d_py, d_pz, d_pcb,
      d_gx, d_gy, d_gz, d_gcb,
      step_size, n_particles);

  // Copy results back
  CUDA_CHECK(cudaMemcpy(px, d_px, sz, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(py, d_py, sz, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(pz, d_pz, sz, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(pcb, d_pcb, sz, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py));
  CUDA_CHECK(cudaFree(d_pz)); CUDA_CHECK(cudaFree(d_pcb));
  CUDA_CHECK(cudaFree(d_sat)); CUDA_CHECK(cudaFree(d_pr));
  CUDA_CHECK(cudaFree(d_ws));
  CUDA_CHECK(cudaFree(d_gx)); CUDA_CHECK(cudaFree(d_gy));
  CUDA_CHECK(cudaFree(d_gz)); CUDA_CHECK(cudaFree(d_gcb));
}

// ============================================================================
// Simple mean estimate (equal weights after SVGD)
// ============================================================================

__global__ void svgd_mean_kernel(
    const double* px, const double* py, const double* pz, const double* pcb,
    double* partial_results, int N) {
  extern __shared__ double sdata[];
  double* s_x = sdata;
  double* s_y = sdata + blockDim.x;
  double* s_z = sdata + 2 * blockDim.x;
  double* s_cb = sdata + 3 * blockDim.x;

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  s_x[tid] = (gid < N) ? px[gid] : 0.0;
  s_y[tid] = (gid < N) ? py[gid] : 0.0;
  s_z[tid] = (gid < N) ? pz[gid] : 0.0;
  s_cb[tid] = (gid < N) ? pcb[gid] : 0.0;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_x[tid] += s_x[tid + s];
      s_y[tid] += s_y[tid + s];
      s_z[tid] += s_z[tid + s];
      s_cb[tid] += s_cb[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_results[blockIdx.x * 4 + 0] = s_x[0];
    partial_results[blockIdx.x * 4 + 1] = s_y[0];
    partial_results[blockIdx.x * 4 + 2] = s_z[0];
    partial_results[blockIdx.x * 4 + 3] = s_cb[0];
  }
}

void pf_svgd_estimate(
    const double* px, const double* py, const double* pz, const double* pcb,
    double* result, int n_particles) {
  size_t sz = (size_t)n_particles * sizeof(double);

  double *d_px, *d_py, *d_pz, *d_pcb;
  CUDA_CHECK(cudaMalloc(&d_px, sz));
  CUDA_CHECK(cudaMalloc(&d_py, sz));
  CUDA_CHECK(cudaMalloc(&d_pz, sz));
  CUDA_CHECK(cudaMalloc(&d_pcb, sz));

  CUDA_CHECK(cudaMemcpy(d_px, px, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_py, py, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pz, pz, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pcb, pcb, sz, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_particles + block - 1) / block;

  double* d_partial;
  CUDA_CHECK(cudaMalloc(&d_partial, grid * 4 * sizeof(double)));

  size_t smem = 4 * block * sizeof(double);
  svgd_mean_kernel<<<grid, block, smem>>>(d_px, d_py, d_pz, d_pcb,
                                          d_partial, n_particles);

  double* h_partial = new double[grid * 4];
  CUDA_CHECK(cudaMemcpy(h_partial, d_partial, grid * 4 * sizeof(double), cudaMemcpyDeviceToHost));

  double sx = 0, sy = 0, sz_val = 0, scb = 0;
  for (int i = 0; i < grid; i++) {
    sx += h_partial[i * 4 + 0];
    sy += h_partial[i * 4 + 1];
    sz_val += h_partial[i * 4 + 2];
    scb += h_partial[i * 4 + 3];
  }

  double inv_n = 1.0 / (double)n_particles;
  result[0] = sx * inv_n;
  result[1] = sy * inv_n;
  result[2] = sz_val * inv_n;
  result[3] = scb * inv_n;

  delete[] h_partial;
  CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py));
  CUDA_CHECK(cudaFree(d_pz)); CUDA_CHECK(cudaFree(d_pcb));
  CUDA_CHECK(cudaFree(d_partial));
}

}  // namespace gnss_gpu
