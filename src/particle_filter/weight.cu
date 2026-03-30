#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/particle_filter.h"
#include <cmath>
#include <cstring>
#include <cfloat>
#include <algorithm>

namespace gnss_gpu {

// Maximum satellites supported in shared memory
static constexpr int MAX_SATS = 64;

__global__ void weight_kernel(const double* px, const double* py,
                              const double* pz, const double* pcb,
                              const double* sat_ecef,
                              const double* pseudoranges,
                              const double* weights_sat,
                              double* log_weights,
                              int N, int n_sat, double sigma_pr) {
  // Load satellite data into shared memory
  __shared__ double s_sat[MAX_SATS * 3];  // x, y, z
  __shared__ double s_pr[MAX_SATS];
  __shared__ double s_ws[MAX_SATS];

  // Cooperative load of satellite data
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

  double x = px[tid];
  double y = py[tid];
  double z = pz[tid];
  double cb = pcb[tid];

  double inv_sigma2 = 1.0 / (sigma_pr * sigma_pr);
  double log_w = 0.0;

  for (int s = 0; s < n_sat; s++) {
    double dx = x - s_sat[s * 3 + 0];
    double dy = y - s_sat[s * 3 + 1];
    double dz = z - s_sat[s * 3 + 2];
    double r = sqrt(dx * dx + dy * dy + dz * dz);
    double pred_pr = r + cb;
    double residual = s_pr[s] - pred_pr;
    log_w += -0.5 * s_ws[s] * residual * residual * inv_sigma2;
  }

  log_weights[tid] = log_w;
}

// Parallel reduction for ESS computation
// Uses log-sum-exp trick for numerical stability
__global__ void compute_ess_kernel(const double* log_weights,
                                   double* partial_log_sum_w,
                                   double* partial_log_sum_w2,
                                   double* partial_max_lw,
                                   int N) {
  extern __shared__ double sdata[];
  double* s_logw = sdata;                          // blockDim.x
  double* s_logw2 = sdata + blockDim.x;            // blockDim.x
  double* s_max = sdata + 2 * blockDim.x;          // blockDim.x

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  // Load and find local max
  double lw = (gid < N) ? log_weights[gid] : -INFINITY;
  s_max[tid] = lw;
  __syncthreads();

  // Reduce to find block max
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
    }
    __syncthreads();
  }
  double block_max = s_max[0];

  // Compute exp(lw - max) and exp(2*(lw - max))
  double w_shifted = (gid < N) ? exp(lw - block_max) : 0.0;
  s_logw[tid] = w_shifted;
  s_logw2[tid] = w_shifted * w_shifted;
  __syncthreads();

  // Parallel sum reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_logw[tid] += s_logw[tid + s];
      s_logw2[tid] += s_logw2[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_log_sum_w[blockIdx.x] = s_logw[0];
    partial_log_sum_w2[blockIdx.x] = s_logw2[0];
    partial_max_lw[blockIdx.x] = block_max;
  }
}

double pf_compute_ess(const double* log_weights, int n_particles) {
  double *d_lw;
  size_t sz = (size_t)n_particles * sizeof(double);
  CUDA_CHECK(cudaMalloc(&d_lw, sz));
  CUDA_CHECK(cudaMemcpy(d_lw, log_weights, sz, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_particles + block - 1) / block;

  double *d_sum_w, *d_sum_w2, *d_max_lw;
  CUDA_CHECK(cudaMalloc(&d_sum_w, grid * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_sum_w2, grid * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_max_lw, grid * sizeof(double)));

  size_t smem = 3 * block * sizeof(double);
  compute_ess_kernel<<<grid, block, smem>>>(d_lw, d_sum_w, d_sum_w2, d_max_lw, n_particles);

  // Copy partial results back and finalize on CPU
  double* h_sum_w = new double[grid];
  double* h_sum_w2 = new double[grid];
  double* h_max_lw = new double[grid];
  CUDA_CHECK(cudaMemcpy(h_sum_w, d_sum_w, grid * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_sum_w2, d_sum_w2, grid * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_max_lw, d_max_lw, grid * sizeof(double), cudaMemcpyDeviceToHost));

  // Find global max
  double global_max = h_max_lw[0];
  for (int i = 1; i < grid; i++) {
    global_max = std::max(global_max, h_max_lw[i]);
  }

  // Combine partial sums with correction for different block maxima
  double total_w = 0.0, total_w2 = 0.0;
  for (int i = 0; i < grid; i++) {
    double correction = exp(h_max_lw[i] - global_max);
    total_w += h_sum_w[i] * correction;
    total_w2 += h_sum_w2[i] * correction * correction;
  }

  double ess = (total_w * total_w) / total_w2;

  delete[] h_sum_w;
  delete[] h_sum_w2;
  delete[] h_max_lw;
  CUDA_CHECK(cudaFree(d_lw));
  CUDA_CHECK(cudaFree(d_sum_w));
  CUDA_CHECK(cudaFree(d_sum_w2));
  CUDA_CHECK(cudaFree(d_max_lw));

  return ess;
}

// Weighted mean estimate via parallel reduction
__global__ void estimate_kernel(const double* px, const double* py,
                                const double* pz, const double* pcb,
                                const double* log_weights,
                                double* partial_results,
                                double* partial_sum_w,
                                double* partial_max_lw,
                                int N) {
  extern __shared__ double sdata[];
  // Layout: [wx, wy, wz, wcb, sw, max_lw] * blockDim.x
  double* s_wx = sdata;
  double* s_wy = sdata + blockDim.x;
  double* s_wz = sdata + 2 * blockDim.x;
  double* s_wcb = sdata + 3 * blockDim.x;
  double* s_sw = sdata + 4 * blockDim.x;
  double* s_max = sdata + 5 * blockDim.x;

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  double lw = (gid < N) ? log_weights[gid] : -INFINITY;
  s_max[tid] = lw;
  __syncthreads();

  // Find block max
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
    __syncthreads();
  }
  double block_max = s_max[0];

  double w = (gid < N) ? exp(lw - block_max) : 0.0;
  s_wx[tid] = (gid < N) ? w * px[gid] : 0.0;
  s_wy[tid] = (gid < N) ? w * py[gid] : 0.0;
  s_wz[tid] = (gid < N) ? w * pz[gid] : 0.0;
  s_wcb[tid] = (gid < N) ? w * pcb[gid] : 0.0;
  s_sw[tid] = w;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_wx[tid] += s_wx[tid + s];
      s_wy[tid] += s_wy[tid + s];
      s_wz[tid] += s_wz[tid + s];
      s_wcb[tid] += s_wcb[tid + s];
      s_sw[tid] += s_sw[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_results[blockIdx.x * 4 + 0] = s_wx[0];
    partial_results[blockIdx.x * 4 + 1] = s_wy[0];
    partial_results[blockIdx.x * 4 + 2] = s_wz[0];
    partial_results[blockIdx.x * 4 + 3] = s_wcb[0];
    partial_sum_w[blockIdx.x] = s_sw[0];
    partial_max_lw[blockIdx.x] = block_max;
  }
}

void pf_weight(const double* px, const double* py, const double* pz, const double* pcb,
               const double* sat_ecef, const double* pseudoranges,
               const double* weights_sat, double* log_weights,
               int n_particles, int n_sat, double sigma_pr) {
  size_t sz = (size_t)n_particles * sizeof(double);
  size_t sz_sat = (size_t)n_sat * 3 * sizeof(double);
  size_t sz_obs = (size_t)n_sat * sizeof(double);

  double *d_px, *d_py, *d_pz, *d_pcb;
  double *d_sat, *d_pr, *d_ws, *d_lw;

  CUDA_CHECK(cudaMalloc(&d_px, sz));
  CUDA_CHECK(cudaMalloc(&d_py, sz));
  CUDA_CHECK(cudaMalloc(&d_pz, sz));
  CUDA_CHECK(cudaMalloc(&d_pcb, sz));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_pr, sz_obs));
  CUDA_CHECK(cudaMalloc(&d_ws, sz_obs));
  CUDA_CHECK(cudaMalloc(&d_lw, sz));

  CUDA_CHECK(cudaMemcpy(d_px, px, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_py, py, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pz, pz, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pcb, pcb, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pr, pseudoranges, sz_obs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ws, weights_sat, sz_obs, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_particles + block - 1) / block;
  weight_kernel<<<grid, block>>>(d_px, d_py, d_pz, d_pcb,
                                 d_sat, d_pr, d_ws, d_lw,
                                 n_particles, n_sat, sigma_pr);

  CUDA_CHECK(cudaMemcpy(log_weights, d_lw, sz, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py));
  CUDA_CHECK(cudaFree(d_sat)); CUDA_CHECK(cudaFree(d_pr));
}

void pf_estimate(const double* px, const double* py, const double* pz, const double* pcb,
                 const double* log_weights, double* result, int n_particles) {
  size_t sz = (size_t)n_particles * sizeof(double);

  double *d_px, *d_py, *d_pz, *d_pcb, *d_lw;
  CUDA_CHECK(cudaMalloc(&d_px, sz));
  CUDA_CHECK(cudaMalloc(&d_py, sz));
  CUDA_CHECK(cudaMalloc(&d_pz, sz));
  CUDA_CHECK(cudaMalloc(&d_pcb, sz));
  CUDA_CHECK(cudaMalloc(&d_lw, sz));

  CUDA_CHECK(cudaMemcpy(d_px, px, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_py, py, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pz, pz, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pcb, pcb, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lw, log_weights, sz, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_particles + block - 1) / block;

  double *d_partial, *d_sum_w, *d_max_lw;
  CUDA_CHECK(cudaMalloc(&d_partial, grid * 4 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_sum_w, grid * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_max_lw, grid * sizeof(double)));

  size_t smem = 6 * block * sizeof(double);
  estimate_kernel<<<grid, block, smem>>>(d_px, d_py, d_pz, d_pcb, d_lw,
                                         d_partial, d_sum_w, d_max_lw,
                                         n_particles);

  double* h_partial = new double[grid * 4];
  double* h_sum_w = new double[grid];
  double* h_max_lw = new double[grid];
  CUDA_CHECK(cudaMemcpy(h_partial, d_partial, grid * 4 * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_sum_w, d_sum_w, grid * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_max_lw, d_max_lw, grid * sizeof(double), cudaMemcpyDeviceToHost));

  // Find global max
  double global_max = h_max_lw[0];
  for (int i = 1; i < grid; i++) {
    global_max = std::max(global_max, h_max_lw[i]);
  }

  // Combine with correction
  double wx = 0, wy = 0, wz = 0, wcb = 0, sw = 0;
  for (int i = 0; i < grid; i++) {
    double correction = exp(h_max_lw[i] - global_max);
    wx += h_partial[i * 4 + 0] * correction;
    wy += h_partial[i * 4 + 1] * correction;
    wz += h_partial[i * 4 + 2] * correction;
    wcb += h_partial[i * 4 + 3] * correction;
    sw += h_sum_w[i] * correction;
  }

  result[0] = wx / sw;
  result[1] = wy / sw;
  result[2] = wz / sw;
  result[3] = wcb / sw;

  delete[] h_partial;
  delete[] h_sum_w;
  delete[] h_max_lw;
  CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py));
  CUDA_CHECK(cudaFree(d_lw)); CUDA_CHECK(cudaFree(d_partial));
}

// Get particles kernel
__global__ void get_particles_kernel(const double* px, const double* py,
                                     const double* pz, const double* pcb,
                                     double* output, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  output[tid * 4 + 0] = px[tid];
  output[tid * 4 + 1] = py[tid];
  output[tid * 4 + 2] = pz[tid];
  output[tid * 4 + 3] = pcb[tid];
}

void pf_get_particles(const double* px, const double* py,
                      const double* pz, const double* pcb,
                      double* output, int n_particles) {
  size_t sz = (size_t)n_particles * sizeof(double);
  size_t sz_out = (size_t)n_particles * 4 * sizeof(double);

  double *d_px, *d_py, *d_pz, *d_pcb, *d_out;
  CUDA_CHECK(cudaMalloc(&d_px, sz));
  CUDA_CHECK(cudaMalloc(&d_py, sz));
  CUDA_CHECK(cudaMalloc(&d_pz, sz));
  CUDA_CHECK(cudaMalloc(&d_pcb, sz));
  CUDA_CHECK(cudaMalloc(&d_out, sz_out));

  CUDA_CHECK(cudaMemcpy(d_px, px, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_py, py, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pz, pz, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pcb, pcb, sz, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_particles + block - 1) / block;
  get_particles_kernel<<<grid, block>>>(d_px, d_py, d_pz, d_pcb, d_out, n_particles);

  CUDA_CHECK(cudaMemcpy(output, d_out, sz_out, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py));
  CUDA_CHECK(cudaFree(d_out));
}

}  // namespace gnss_gpu
