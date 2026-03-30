#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/particle_filter.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <curand_kernel.h>
#include <cmath>
#include <cfloat>
#include <algorithm>

namespace gnss_gpu {

// --- Helpers: normalize log_weights to weights via log-sum-exp ---

__global__ void log_sum_exp_max_kernel(const double* log_weights, double* block_max, int N) {
  extern __shared__ double sdata[];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  sdata[tid] = (gid < N) ? log_weights[gid] : -INFINITY;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) block_max[blockIdx.x] = sdata[0];
}

__global__ void exp_and_normalize_kernel(const double* log_weights, double* weights,
                                         double max_lw, double sum_w, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  weights[tid] = exp(log_weights[tid] - max_lw) / sum_w;
}

__global__ void exp_shift_kernel(const double* log_weights, double* weights,
                                 double max_lw, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  weights[tid] = exp(log_weights[tid] - max_lw);
}

__global__ void sum_reduce_kernel(const double* data, double* block_sums, int N) {
  extern __shared__ double sdata[];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  sdata[tid] = (gid < N) ? data[gid] : 0.0;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) block_sums[blockIdx.x] = sdata[0];
}

// --- Systematic Resampling ---

__global__ void systematic_resample_kernel(const double* cdf,
                                           const double* px_in, const double* py_in,
                                           const double* pz_in, const double* pcb_in,
                                           double* px_out, double* py_out,
                                           double* pz_out, double* pcb_out,
                                           int N, double u0) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  double target = (u0 + (double)tid) / (double)N;

  // Binary search in CDF
  int lo = 0, hi = N - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (cdf[mid] < target) lo = mid + 1;
    else hi = mid;
  }

  px_out[tid] = px_in[lo];
  py_out[tid] = py_in[lo];
  pz_out[tid] = pz_in[lo];
  pcb_out[tid] = pcb_in[lo];
}

void pf_resample_systematic(double* px, double* py, double* pz, double* pcb,
                            const double* log_weights,
                            int n_particles, unsigned long long seed) {
  size_t sz = (size_t)n_particles * sizeof(double);
  int block = 256;
  int grid = (n_particles + block - 1) / block;

  // Allocate device memory
  double *d_lw, *d_weights, *d_cdf;
  double *d_px, *d_py, *d_pz, *d_pcb;
  double *d_px_out, *d_py_out, *d_pz_out, *d_pcb_out;

  CUDA_CHECK(cudaMalloc(&d_lw, sz));
  CUDA_CHECK(cudaMalloc(&d_weights, sz));
  CUDA_CHECK(cudaMalloc(&d_cdf, sz));
  CUDA_CHECK(cudaMalloc(&d_px, sz));
  CUDA_CHECK(cudaMalloc(&d_py, sz));
  CUDA_CHECK(cudaMalloc(&d_pz, sz));
  CUDA_CHECK(cudaMalloc(&d_pcb, sz));
  CUDA_CHECK(cudaMalloc(&d_px_out, sz));
  CUDA_CHECK(cudaMalloc(&d_py_out, sz));
  CUDA_CHECK(cudaMalloc(&d_pz_out, sz));
  CUDA_CHECK(cudaMalloc(&d_pcb_out, sz));

  CUDA_CHECK(cudaMemcpy(d_lw, log_weights, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_px, px, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_py, py, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pz, pz, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pcb, pcb, sz, cudaMemcpyHostToDevice));

  // Step 1: Find max log_weight
  double* d_block_max;
  CUDA_CHECK(cudaMalloc(&d_block_max, grid * sizeof(double)));
  log_sum_exp_max_kernel<<<grid, block, block * sizeof(double)>>>(d_lw, d_block_max, n_particles);

  double* h_block_max = new double[grid];
  CUDA_CHECK(cudaMemcpy(h_block_max, d_block_max, grid * sizeof(double), cudaMemcpyDeviceToHost));
  double max_lw = h_block_max[0];
  for (int i = 1; i < grid; i++) max_lw = std::max(max_lw, h_block_max[i]);
  delete[] h_block_max;
  CUDA_CHECK(cudaFree(d_block_max));

  // Step 2: exp(lw - max) into weights
  exp_shift_kernel<<<grid, block>>>(d_lw, d_weights, max_lw, n_particles);

  // Step 3: Compute sum of weights
  double* d_block_sums;
  CUDA_CHECK(cudaMalloc(&d_block_sums, grid * sizeof(double)));
  sum_reduce_kernel<<<grid, block, block * sizeof(double)>>>(d_weights, d_block_sums, n_particles);

  double* h_block_sums = new double[grid];
  CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums, grid * sizeof(double), cudaMemcpyDeviceToHost));
  double sum_w = 0;
  for (int i = 0; i < grid; i++) sum_w += h_block_sums[i];
  delete[] h_block_sums;
  CUDA_CHECK(cudaFree(d_block_sums));

  // Step 4: Normalize weights
  exp_and_normalize_kernel<<<grid, block>>>(d_lw, d_weights, max_lw, sum_w, n_particles);

  // Step 5: Inclusive scan (CDF) using thrust with double precision
  thrust::device_ptr<double> w_ptr(d_weights);
  thrust::device_ptr<double> cdf_ptr(d_cdf);
  thrust::inclusive_scan(w_ptr, w_ptr + n_particles, cdf_ptr);

  // Step 6: Generate uniform random u0 on CPU
  // Simple LCG from seed for u0 in [0, 1/N)
  double u0 = (double)(seed % 1000000) / (double)(1000000 * n_particles);

  // Step 7: Resample
  systematic_resample_kernel<<<grid, block>>>(d_cdf, d_px, d_py, d_pz, d_pcb,
                                              d_px_out, d_py_out, d_pz_out, d_pcb_out,
                                              n_particles, u0);

  // Copy results back
  CUDA_CHECK(cudaMemcpy(px, d_px_out, sz, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(py, d_py_out, sz, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(pz, d_pz_out, sz, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(pcb, d_pcb_out, sz, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_lw)); CUDA_CHECK(cudaFree(d_weights));
  CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py));
  CUDA_CHECK(cudaFree(d_px_out)); CUDA_CHECK(cudaFree(d_py_out));
}

// --- Megopolis Resampling ---
// Reference: Chesser et al., "Megopolis Resampler", arXiv:2109.13504
// No prefix-sum needed, numerically stable for large N.
// Uses double-buffering to avoid race conditions.

__global__ void megopolis_kernel(double* px_a, double* py_a, double* pz_a, double* pcb_a,
                                double* px_b, double* py_b, double* pz_b, double* pcb_b,
                                const double* log_weights,
                                int N, unsigned long long seed, int iteration,
                                int src_buf) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, tid, iteration, &state);

  // Select source and destination buffers
  const double* src_px = (src_buf == 0) ? px_a : px_b;
  const double* src_py = (src_buf == 0) ? py_a : py_b;
  const double* src_pz = (src_buf == 0) ? pz_a : pz_b;
  const double* src_pcb = (src_buf == 0) ? pcb_a : pcb_b;
  double* dst_px = (src_buf == 0) ? px_b : px_a;
  double* dst_py = (src_buf == 0) ? py_b : py_a;
  double* dst_pz = (src_buf == 0) ? pz_b : pz_a;
  double* dst_pcb = (src_buf == 0) ? pcb_b : pcb_a;

  // Propose swap partner: random index j != tid
  int offset = (int)(curand_uniform_double(&state) * (N - 1)) + 1;
  int j = (tid + offset) % N;

  // Acceptance ratio (in log-space)
  double log_alpha = log_weights[j] - log_weights[tid];
  double alpha = fmin(1.0, exp(log_alpha));

  double u = curand_uniform_double(&state);

  if (u < alpha) {
    // Accept: copy from j
    dst_px[tid] = src_px[j];
    dst_py[tid] = src_py[j];
    dst_pz[tid] = src_pz[j];
    dst_pcb[tid] = src_pcb[j];
  } else {
    // Reject: keep current
    dst_px[tid] = src_px[tid];
    dst_py[tid] = src_py[tid];
    dst_pz[tid] = src_pz[tid];
    dst_pcb[tid] = src_pcb[tid];
  }
}

void pf_resample_megopolis(double* px, double* py, double* pz, double* pcb,
                           const double* log_weights,
                           int n_particles, int n_iterations,
                           unsigned long long seed) {
  size_t sz = (size_t)n_particles * sizeof(double);
  int block = 256;
  int grid = (n_particles + block - 1) / block;

  // Allocate double buffers on device
  double *d_px_a, *d_py_a, *d_pz_a, *d_pcb_a;
  double *d_px_b, *d_py_b, *d_pz_b, *d_pcb_b;
  double *d_lw;

  CUDA_CHECK(cudaMalloc(&d_px_a, sz)); CUDA_CHECK(cudaMalloc(&d_py_a, sz));
  CUDA_CHECK(cudaMalloc(&d_pz_a, sz)); CUDA_CHECK(cudaMalloc(&d_pcb_a, sz));
  CUDA_CHECK(cudaMalloc(&d_px_b, sz)); CUDA_CHECK(cudaMalloc(&d_py_b, sz));
  CUDA_CHECK(cudaMalloc(&d_pz_b, sz)); CUDA_CHECK(cudaMalloc(&d_pcb_b, sz));
  CUDA_CHECK(cudaMalloc(&d_lw, sz));

  CUDA_CHECK(cudaMemcpy(d_px_a, px, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_py_a, py, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pz_a, pz, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pcb_a, pcb, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lw, log_weights, sz, cudaMemcpyHostToDevice));

  // Iterate Megopolis mixing steps with double-buffering
  for (int iter = 0; iter < n_iterations; iter++) {
    int src_buf = iter % 2;
    megopolis_kernel<<<grid, block>>>(d_px_a, d_py_a, d_pz_a, d_pcb_a,
                                     d_px_b, d_py_b, d_pz_b, d_pcb_b,
                                     d_lw, n_particles, seed, iter, src_buf);
  }

  // Copy final result (depends on which buffer is destination of last iteration)
  int final_buf = n_iterations % 2;  // last write went to buf (n_iterations-1)%2 == 0 ? b : a
  if (final_buf == 0) {
    // Last iteration had src_buf = n_iterations-1 which is odd, dst = a
    CUDA_CHECK(cudaMemcpy(px, d_px_a, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(py, d_py_a, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pz, d_pz_a, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pcb, d_pcb_a, sz, cudaMemcpyDeviceToHost));
  } else {
    CUDA_CHECK(cudaMemcpy(px, d_px_b, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(py, d_py_b, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pz, d_pz_b, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pcb, d_pcb_b, sz, cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaFree(d_px_a)); CUDA_CHECK(cudaFree(d_py_a));
  CUDA_CHECK(cudaFree(d_px_b)); CUDA_CHECK(cudaFree(d_py_b));
  CUDA_CHECK(cudaFree(d_lw));
}

}  // namespace gnss_gpu
