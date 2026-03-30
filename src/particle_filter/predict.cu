#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/particle_filter.h"
#include <curand_kernel.h>

namespace gnss_gpu {

__global__ void init_kernel(double* px, double* py, double* pz, double* pcb,
                            double init_x, double init_y, double init_z, double init_cb,
                            double spread_pos, double spread_cb,
                            int N, unsigned long long seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, tid, 0, &state);

  px[tid] = init_x + curand_normal_double(&state) * spread_pos;
  py[tid] = init_y + curand_normal_double(&state) * spread_pos;
  pz[tid] = init_z + curand_normal_double(&state) * spread_pos;
  pcb[tid] = init_cb + curand_normal_double(&state) * spread_cb;
}

__global__ void predict_kernel(double* px, double* py, double* pz, double* pcb,
                               const double* vx, const double* vy, const double* vz,
                               double dt, double sigma_pos, double sigma_cb,
                               int N, unsigned long long seed, int step) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, tid, step, &state);

  // Apply velocity (single velocity vector broadcast to all particles)
  px[tid] += vx[0] * dt + curand_normal_double(&state) * sigma_pos;
  py[tid] += vy[0] * dt + curand_normal_double(&state) * sigma_pos;
  pz[tid] += vz[0] * dt + curand_normal_double(&state) * sigma_pos;
  pcb[tid] += curand_normal_double(&state) * sigma_cb;
}

void pf_initialize(double* px, double* py, double* pz, double* pcb,
                   double init_x, double init_y, double init_z, double init_cb,
                   double spread_pos, double spread_cb,
                   int n_particles, unsigned long long seed) {
  double *d_px, *d_py, *d_pz, *d_pcb;
  size_t sz = (size_t)n_particles * sizeof(double);

  CUDA_CHECK(cudaMalloc(&d_px, sz));
  CUDA_CHECK(cudaMalloc(&d_py, sz));
  CUDA_CHECK(cudaMalloc(&d_pz, sz));
  CUDA_CHECK(cudaMalloc(&d_pcb, sz));

  int block = 256;
  int grid = (n_particles + block - 1) / block;
  init_kernel<<<grid, block>>>(d_px, d_py, d_pz, d_pcb,
                               init_x, init_y, init_z, init_cb,
                               spread_pos, spread_cb,
                               n_particles, seed);

  CUDA_CHECK(cudaMemcpy(px, d_px, sz, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(py, d_py, sz, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(pz, d_pz, sz, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(pcb, d_pcb, sz, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_px));
  CUDA_CHECK(cudaFree(d_py));
  CUDA_CHECK(cudaFree(d_pz));
  CUDA_CHECK(cudaFree(d_pcb));
}

void pf_predict(double* px, double* py, double* pz, double* pcb,
                const double* vx, const double* vy, const double* vz,
                double dt, double sigma_pos, double sigma_cb,
                int n_particles, unsigned long long seed, int step) {
  double *d_px, *d_py, *d_pz, *d_pcb;
  double *d_vx, *d_vy, *d_vz;
  size_t sz = (size_t)n_particles * sizeof(double);

  CUDA_CHECK(cudaMalloc(&d_px, sz));
  CUDA_CHECK(cudaMalloc(&d_py, sz));
  CUDA_CHECK(cudaMalloc(&d_pz, sz));
  CUDA_CHECK(cudaMalloc(&d_pcb, sz));
  CUDA_CHECK(cudaMalloc(&d_vx, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_vy, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_vz, sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_px, px, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_py, py, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pz, pz, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pcb, pcb, sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vx, vx, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vy, vy, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vz, vz, sizeof(double), cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_particles + block - 1) / block;
  predict_kernel<<<grid, block>>>(d_px, d_py, d_pz, d_pcb,
                                  d_vx, d_vy, d_vz,
                                  dt, sigma_pos, sigma_cb,
                                  n_particles, seed, step);

  CUDA_CHECK(cudaMemcpy(px, d_px, sz, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(py, d_py, sz, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(pz, d_pz, sz, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(pcb, d_pcb, sz, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_px));
  CUDA_CHECK(cudaFree(d_py));
  CUDA_CHECK(cudaFree(d_pz));
  CUDA_CHECK(cudaFree(d_pcb));
  CUDA_CHECK(cudaFree(d_vx));
  CUDA_CHECK(cudaFree(d_vy));
  CUDA_CHECK(cudaFree(d_vz));
}

}  // namespace gnss_gpu
