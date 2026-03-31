#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/pf_device.h"
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace gnss_gpu {

static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_SATS = 64;

// ============================================================
// Kernels (self-contained, no extern dependencies)
// ============================================================

__global__ void pfd_init_kernel(double* px, double* py, double* pz, double* pcb,
                                double* log_weights,
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
    log_weights[tid] = 0.0;
}

__global__ void pfd_predict_kernel(double* px, double* py, double* pz, double* pcb,
                                   const double* vel,  // [3]: vx, vy, vz
                                   double dt, double sigma_pos, double sigma_cb,
                                   int N, unsigned long long seed, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, step, &state);

    px[tid] += vel[0] * dt + curand_normal_double(&state) * sigma_pos;
    py[tid] += vel[1] * dt + curand_normal_double(&state) * sigma_pos;
    pz[tid] += vel[2] * dt + curand_normal_double(&state) * sigma_pos;
    pcb[tid] += curand_normal_double(&state) * sigma_cb;
}

__global__ void pfd_weight_kernel(const double* px, const double* py,
                                  const double* pz, const double* pcb,
                                  const double* sat_ecef,
                                  const double* pseudoranges,
                                  const double* weights_sat,
                                  double* log_weights,
                                  int N, int n_sat, double sigma_pr) {
    // Dynamic shared memory layout: [sat_ecef: n_sat*3] [pr: n_sat] [ws: n_sat]
    extern __shared__ double s_data[];
    double* s_sat = s_data;
    double* s_pr = s_data + n_sat * 3;
    double* s_ws = s_data + n_sat * 4;

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

// --- ESS reduction kernel (same logic as weight.cu) ---
__global__ void pfd_ess_kernel(const double* log_weights,
                               double* partial_sum_w,
                               double* partial_sum_w2,
                               double* partial_max_lw,
                               int N) {
    extern __shared__ double sdata[];
    double* s_sum = sdata;
    double* s_sum2 = sdata + blockDim.x;
    double* s_max = sdata + 2 * blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double lw = (gid < N) ? log_weights[gid] : -INFINITY;
    s_max[tid] = lw;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
    double block_max = s_max[0];

    double w_shifted = (gid < N) ? exp(lw - block_max) : 0.0;
    s_sum[tid] = w_shifted;
    s_sum2[tid] = w_shifted * w_shifted;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum2[tid] += s_sum2[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sum_w[blockIdx.x] = s_sum[0];
        partial_sum_w2[blockIdx.x] = s_sum2[0];
        partial_max_lw[blockIdx.x] = block_max;
    }
}

// --- Estimate reduction kernel ---
__global__ void pfd_estimate_kernel(const double* px, const double* py,
                                    const double* pz, const double* pcb,
                                    const double* log_weights,
                                    double* partial_results,  // [grid * 4]
                                    double* partial_sum_w,    // [grid]
                                    double* partial_max_lw,   // [grid]
                                    int N) {
    extern __shared__ double sdata[];
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

// --- Get particles kernel ---
__global__ void pfd_get_particles_kernel(const double* px, const double* py,
                                         const double* pz, const double* pcb,
                                         double* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    output[tid * 4 + 0] = px[tid];
    output[tid * 4 + 1] = py[tid];
    output[tid * 4 + 2] = pz[tid];
    output[tid * 4 + 3] = pcb[tid];
}

// --- Resampling helper kernels ---
__global__ void pfd_find_max_kernel(const double* log_weights, double* block_max, int N) {
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

__global__ void pfd_exp_shift_kernel(const double* log_weights, double* weights,
                                     double max_lw, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    weights[tid] = exp(log_weights[tid] - max_lw);
}

__global__ void pfd_sum_reduce_kernel(const double* data, double* block_sums, int N) {
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

__global__ void pfd_normalize_kernel(const double* log_weights, double* weights,
                                     double max_lw, double sum_w, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    weights[tid] = exp(log_weights[tid] - max_lw) / sum_w;
}

__global__ void pfd_systematic_resample_kernel(const double* cdf,
                                               const double* px_in, const double* py_in,
                                               const double* pz_in, const double* pcb_in,
                                               double* px_out, double* py_out,
                                               double* pz_out, double* pcb_out,
                                               int N, double u0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double target = (u0 + (double)tid) / (double)N;

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

// --- Reset log weights to zero ---
__global__ void pfd_reset_weights_kernel(double* log_weights, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    log_weights[tid] = 0.0;
}

// --- Megopolis kernel ---
__global__ void pfd_megopolis_kernel(double* px_a, double* py_a, double* pz_a, double* pcb_a,
                                    double* px_b, double* py_b, double* pz_b, double* pcb_b,
                                    const double* log_weights,
                                    int N, unsigned long long seed, int iteration,
                                    int src_buf) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, iteration, &state);

    const double* src_px = (src_buf == 0) ? px_a : px_b;
    const double* src_py = (src_buf == 0) ? py_a : py_b;
    const double* src_pz = (src_buf == 0) ? pz_a : pz_b;
    const double* src_pcb = (src_buf == 0) ? pcb_a : pcb_b;
    double* dst_px = (src_buf == 0) ? px_b : px_a;
    double* dst_py = (src_buf == 0) ? py_b : py_a;
    double* dst_pz = (src_buf == 0) ? pz_b : pz_a;
    double* dst_pcb = (src_buf == 0) ? pcb_b : pcb_a;

    int offset = (int)(curand_uniform_double(&state) * (N - 1)) + 1;
    int j = (tid + offset) % N;

    double log_alpha = log_weights[j] - log_weights[tid];
    double alpha = fmin(1.0, exp(log_alpha));
    double u = curand_uniform_double(&state);

    if (u < alpha) {
        dst_px[tid] = src_px[j];
        dst_py[tid] = src_py[j];
        dst_pz[tid] = src_pz[j];
        dst_pcb[tid] = src_pcb[j];
    } else {
        dst_px[tid] = src_px[tid];
        dst_py[tid] = src_py[tid];
        dst_pz[tid] = src_pz[tid];
        dst_pcb[tid] = src_pcb[tid];
    }
}

// CUDA_CHECK macro is provided by gnss_gpu/cuda_check.h (included above)

// ============================================================
// Lifecycle
// ============================================================

PFDeviceState* pf_device_create(int n_particles) {
    PFDeviceState* state = new PFDeviceState();
    state->n_particles = n_particles;
    state->grid_size = (n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    state->allocated = false;

    size_t sz = (size_t)n_particles * sizeof(double);
    int grid = state->grid_size;

    // Main particle arrays
    CUDA_CHECK(cudaMalloc(&state->d_px, sz));
    CUDA_CHECK(cudaMalloc(&state->d_py, sz));
    CUDA_CHECK(cudaMalloc(&state->d_pz, sz));
    CUDA_CHECK(cudaMalloc(&state->d_pcb, sz));
    CUDA_CHECK(cudaMalloc(&state->d_log_weights, sz));

    // Double-buffer for resampling
    CUDA_CHECK(cudaMalloc(&state->d_px_tmp, sz));
    CUDA_CHECK(cudaMalloc(&state->d_py_tmp, sz));
    CUDA_CHECK(cudaMalloc(&state->d_pz_tmp, sz));
    CUDA_CHECK(cudaMalloc(&state->d_pcb_tmp, sz));

    // Reduction temporaries
    // partial_a: max(grid * 4, grid) doubles - used for estimate (grid*4) or ESS (grid)
    CUDA_CHECK(cudaMalloc(&state->d_partial_a, grid * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_partial_b, grid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_partial_c, grid * sizeof(double)));

    // Systematic resampling buffers
    CUDA_CHECK(cudaMalloc(&state->d_weights_norm, sz));
    CUDA_CHECK(cudaMalloc(&state->d_cdf, sz));

    // Velocity buffer (3 doubles)
    CUDA_CHECK(cudaMalloc(&state->d_vel, 3 * sizeof(double)));

    // CUDA stream for pipelined execution
    CUDA_CHECK(cudaStreamCreate(&state->stream));

    // Persistent device buffers for satellite data (pre-allocate for MAX_SATS)
    state->pinned_capacity = MAX_SATS;
    CUDA_CHECK(cudaMalloc(&state->d_sat_ecef, MAX_SATS * 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_pseudoranges, MAX_SATS * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_weights_sat, MAX_SATS * sizeof(double)));

    // Pinned host memory for async transfers
    CUDA_CHECK(cudaMallocHost(&state->h_sat_pinned, MAX_SATS * 5 * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&state->h_result_pinned, 4 * sizeof(double)));

    state->allocated = true;
    return state;
}

void pf_device_destroy_resources(PFDeviceState* state) {
    if (!state || !state->allocated) return;

    // Synchronize stream before freeing resources
    cudaStreamSynchronize(state->stream);
    cudaStreamDestroy(state->stream);

    cudaFree(state->d_px);
    cudaFree(state->d_py);
    cudaFree(state->d_pz);
    cudaFree(state->d_pcb);
    cudaFree(state->d_log_weights);
    cudaFree(state->d_px_tmp);
    cudaFree(state->d_py_tmp);
    cudaFree(state->d_pz_tmp);
    cudaFree(state->d_pcb_tmp);
    cudaFree(state->d_partial_a);
    cudaFree(state->d_partial_b);
    cudaFree(state->d_partial_c);
    cudaFree(state->d_weights_norm);
    cudaFree(state->d_cdf);
    cudaFree(state->d_vel);

    // Free persistent satellite device buffers
    cudaFree(state->d_sat_ecef);
    cudaFree(state->d_pseudoranges);
    cudaFree(state->d_weights_sat);

    // Free pinned host memory
    cudaFreeHost(state->h_sat_pinned);
    cudaFreeHost(state->h_result_pinned);

    // Null out all pointers to prevent use-after-free
    state->d_px = nullptr;
    state->d_py = nullptr;
    state->d_pz = nullptr;
    state->d_pcb = nullptr;
    state->d_log_weights = nullptr;
    state->d_px_tmp = nullptr;
    state->d_py_tmp = nullptr;
    state->d_pz_tmp = nullptr;
    state->d_pcb_tmp = nullptr;
    state->d_partial_a = nullptr;
    state->d_partial_b = nullptr;
    state->d_partial_c = nullptr;
    state->d_weights_norm = nullptr;
    state->d_cdf = nullptr;
    state->d_vel = nullptr;
    state->d_sat_ecef = nullptr;
    state->d_pseudoranges = nullptr;
    state->d_weights_sat = nullptr;
    state->h_sat_pinned = nullptr;
    state->h_result_pinned = nullptr;

    state->allocated = false;
}

void pf_device_destroy(PFDeviceState* state) {
    if (!state) return;
    pf_device_destroy_resources(state);
    delete state;
}

// ============================================================
// Initialize
// ============================================================

void pf_device_initialize(PFDeviceState* state,
    double init_x, double init_y, double init_z, double init_cb,
    double spread_pos, double spread_cb,
    unsigned long long seed) {

    int N = state->n_particles;
    int grid = state->grid_size;

    pfd_init_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz, state->d_pcb,
        state->d_log_weights,
        init_x, init_y, init_z, init_cb,
        spread_pos, spread_cb,
        N, seed);
    CUDA_CHECK_LAST();
}

// ============================================================
// Predict
// ============================================================

void pf_device_predict(PFDeviceState* state,
    double vx, double vy, double vz,
    double dt, double sigma_pos, double sigma_cb,
    unsigned long long seed, int step) {

    int N = state->n_particles;
    int grid = state->grid_size;

    // Ensure previous async transfer from pinned buffer is complete before overwriting
    CUDA_CHECK(cudaStreamSynchronize(state->stream));

    // Copy velocity to pinned memory, then async transfer to device
    // h_result_pinned has 4 doubles; we reuse first 3 for velocity (non-overlapping use)
    state->h_result_pinned[0] = vx;
    state->h_result_pinned[1] = vy;
    state->h_result_pinned[2] = vz;
    CUDA_CHECK(cudaMemcpyAsync(state->d_vel, state->h_result_pinned,
                               3 * sizeof(double), cudaMemcpyHostToDevice, state->stream));

    pfd_predict_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz, state->d_pcb,
        state->d_vel,
        dt, sigma_pos, sigma_cb,
        N, seed, step);
    CUDA_CHECK_LAST();
}

// ============================================================
// Weight
// ============================================================

void pf_device_weight(PFDeviceState* state,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    int n_sat, double sigma_pr) {

    int N = state->n_particles;
    int grid = state->grid_size;

    // Satellite data is small: n_sat * 5 doubles typically < 1KB
    // Use persistent device buffers and pinned host memory for async transfer
    size_t sz_sat = (size_t)n_sat * 3 * sizeof(double);
    size_t sz_obs = (size_t)n_sat * sizeof(double);

    // If n_sat exceeds pinned_capacity, reallocate (rare path)
    if (n_sat > state->pinned_capacity) {
        CUDA_CHECK(cudaStreamSynchronize(state->stream));
        CUDA_CHECK(cudaFreeHost(state->h_sat_pinned));
        CUDA_CHECK(cudaFree(state->d_sat_ecef));
        CUDA_CHECK(cudaFree(state->d_pseudoranges));
        CUDA_CHECK(cudaFree(state->d_weights_sat));

        state->pinned_capacity = n_sat;
        CUDA_CHECK(cudaMallocHost(&state->h_sat_pinned, n_sat * 5 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&state->d_sat_ecef, n_sat * 3 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&state->d_pseudoranges, n_sat * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&state->d_weights_sat, n_sat * sizeof(double)));
    }

    // Ensure previous async transfers from pinned buffer are complete before overwriting
    CUDA_CHECK(cudaStreamSynchronize(state->stream));

    // Stage satellite data into contiguous pinned buffer: [sat_ecef | pseudoranges | weights]
    double* h_sat = state->h_sat_pinned;
    memcpy(h_sat, sat_ecef, sz_sat);
    memcpy(h_sat + n_sat * 3, pseudoranges, sz_obs);
    memcpy(h_sat + n_sat * 4, weights_sat, sz_obs);

    // Async H2D transfer on the stream (overlaps with previous kernel if any)
    CUDA_CHECK(cudaMemcpyAsync(state->d_sat_ecef, h_sat,
                               sz_sat, cudaMemcpyHostToDevice, state->stream));
    CUDA_CHECK(cudaMemcpyAsync(state->d_pseudoranges, h_sat + n_sat * 3,
                               sz_obs, cudaMemcpyHostToDevice, state->stream));
    CUDA_CHECK(cudaMemcpyAsync(state->d_weights_sat, h_sat + n_sat * 4,
                               sz_obs, cudaMemcpyHostToDevice, state->stream));

    // Launch weight kernel on the same stream (waits for async copies to complete)
    // Dynamic shared memory: n_sat*3 (sat_ecef) + n_sat (pr) + n_sat (ws) = n_sat*5 doubles
    size_t smem_weight = (size_t)n_sat * 5 * sizeof(double);
    pfd_weight_kernel<<<grid, BLOCK_SIZE, smem_weight, state->stream>>>(
        state->d_px, state->d_py, state->d_pz, state->d_pcb,
        state->d_sat_ecef, state->d_pseudoranges, state->d_weights_sat,
        state->d_log_weights,
        N, n_sat, sigma_pr);
    CUDA_CHECK_LAST();
}

// ============================================================
// ESS
// ============================================================

double pf_device_ess(const PFDeviceState* state) {
    int N = state->n_particles;
    int grid = state->grid_size;

    // Use persistent partial buffers:
    // d_partial_a -> partial_sum_w  (grid doubles, fits in grid*4)
    // d_partial_b -> partial_sum_w2
    // d_partial_c -> partial_max_lw
    size_t smem = 3 * BLOCK_SIZE * sizeof(double);
    pfd_ess_kernel<<<grid, BLOCK_SIZE, smem, state->stream>>>(
        state->d_log_weights,
        state->d_partial_a,   // sum_w
        state->d_partial_b,   // sum_w2
        state->d_partial_c,   // max_lw
        N);
    CUDA_CHECK_LAST();

    // Synchronize stream before reading results back
    CUDA_CHECK(cudaStreamSynchronize(state->stream));

    // Copy partial results back (small: grid * 3 doubles)
    double* h_sum_w = new double[grid];
    double* h_sum_w2 = new double[grid];
    double* h_max_lw = new double[grid];
    CUDA_CHECK(cudaMemcpy(h_sum_w, state->d_partial_a, grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_w2, state->d_partial_b, grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max_lw, state->d_partial_c, grid * sizeof(double), cudaMemcpyDeviceToHost));

    // Find global max
    double global_max = h_max_lw[0];
    for (int i = 1; i < grid; i++) {
        global_max = std::max(global_max, h_max_lw[i]);
    }

    // Combine partial sums with correction
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

    return ess;
}

// ============================================================
// Resample - Systematic
// ============================================================

void pf_device_resample_systematic(PFDeviceState* state, unsigned long long seed) {
    int N = state->n_particles;
    int grid = state->grid_size;

    // Step 1: Find max log_weight using persistent buffers
    pfd_find_max_kernel<<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), state->stream>>>(
        state->d_log_weights, state->d_partial_c, N);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    double* h_block_max = new double[grid];
    CUDA_CHECK(cudaMemcpy(h_block_max, state->d_partial_c, grid * sizeof(double), cudaMemcpyDeviceToHost));
    double max_lw = h_block_max[0];
    for (int i = 1; i < grid; i++) max_lw = std::max(max_lw, h_block_max[i]);
    delete[] h_block_max;

    // Step 2: exp(lw - max) into weights_norm
    pfd_exp_shift_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_log_weights, state->d_weights_norm, max_lw, N);

    // Step 3: Compute sum of weights
    pfd_sum_reduce_kernel<<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), state->stream>>>(
        state->d_weights_norm, state->d_partial_b, N);

    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    double* h_block_sums = new double[grid];
    CUDA_CHECK(cudaMemcpy(h_block_sums, state->d_partial_b, grid * sizeof(double), cudaMemcpyDeviceToHost));
    double sum_w = 0;
    for (int i = 0; i < grid; i++) sum_w += h_block_sums[i];
    delete[] h_block_sums;

    // Step 4: Normalize weights
    pfd_normalize_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_log_weights, state->d_weights_norm, max_lw, sum_w, N);

    // Step 5: Inclusive scan (CDF) using thrust on the custom stream
    thrust::device_ptr<double> w_ptr(state->d_weights_norm);
    thrust::device_ptr<double> cdf_ptr(state->d_cdf);
    thrust::inclusive_scan(thrust::cuda::par.on(state->stream), w_ptr, w_ptr + N, cdf_ptr);

    // Step 6: Generate u0
    double u0 = (double)(seed % 1000000) / (double)(1000000 * N);

    // Step 7: Resample into tmp buffers
    pfd_systematic_resample_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_cdf,
        state->d_px, state->d_py, state->d_pz, state->d_pcb,
        state->d_px_tmp, state->d_py_tmp, state->d_pz_tmp, state->d_pcb_tmp,
        N, u0);
    CUDA_CHECK_LAST();

    // Step 8: Swap pointers (tmp becomes primary)
    std::swap(state->d_px, state->d_px_tmp);
    std::swap(state->d_py, state->d_py_tmp);
    std::swap(state->d_pz, state->d_pz_tmp);
    std::swap(state->d_pcb, state->d_pcb_tmp);

    // Step 9: Reset log weights to uniform
    pfd_reset_weights_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(state->d_log_weights, N);
    CUDA_CHECK_LAST();
}

// ============================================================
// Resample - Megopolis
// ============================================================

void pf_device_resample_megopolis(PFDeviceState* state, int n_iterations, unsigned long long seed) {
    int N = state->n_particles;
    int grid = state->grid_size;

    // Use d_px/d_py/d_pz/d_pcb as buffer A
    // Use d_px_tmp/d_py_tmp/d_pz_tmp/d_pcb_tmp as buffer B
    // Megopolis kernel uses double-buffering directly on device memory

    for (int iter = 0; iter < n_iterations; iter++) {
        int src_buf = iter % 2;
        pfd_megopolis_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
            state->d_px, state->d_py, state->d_pz, state->d_pcb,
            state->d_px_tmp, state->d_py_tmp, state->d_pz_tmp, state->d_pcb_tmp,
            state->d_log_weights,
            N, seed, iter, src_buf);
    }
    CUDA_CHECK_LAST();

    // If final result is in buffer B (tmp), swap pointers
    int final_buf = n_iterations % 2;
    if (final_buf == 1) {
        // Last iteration wrote to buffer B (tmp), swap so primary has result
        std::swap(state->d_px, state->d_px_tmp);
        std::swap(state->d_py, state->d_py_tmp);
        std::swap(state->d_pz, state->d_pz_tmp);
        std::swap(state->d_pcb, state->d_pcb_tmp);
    }

    // Reset log weights to uniform
    pfd_reset_weights_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(state->d_log_weights, N);
    CUDA_CHECK_LAST();
}

// ============================================================
// Estimate
// ============================================================

void pf_device_estimate(const PFDeviceState* state, double* result) {
    int N = state->n_particles;
    int grid = state->grid_size;

    size_t smem = 6 * BLOCK_SIZE * sizeof(double);
    pfd_estimate_kernel<<<grid, BLOCK_SIZE, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz, state->d_pcb,
        state->d_log_weights,
        state->d_partial_a,   // [grid * 4] partial results
        state->d_partial_b,   // [grid] sum_w
        state->d_partial_c,   // [grid] max_lw
        N);
    CUDA_CHECK_LAST();

    // Synchronize stream before reading results back
    CUDA_CHECK(cudaStreamSynchronize(state->stream));

    // Copy partial results back (small)
    double* h_partial = new double[grid * 4];
    double* h_sum_w = new double[grid];
    double* h_max_lw = new double[grid];
    CUDA_CHECK(cudaMemcpy(h_partial, state->d_partial_a, grid * 4 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_w, state->d_partial_b, grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max_lw, state->d_partial_c, grid * sizeof(double), cudaMemcpyDeviceToHost));

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
}

// ============================================================
// Get particles (D2H only when explicitly requested)
// ============================================================

void pf_device_get_particles(const PFDeviceState* state, double* output) {
    int N = state->n_particles;
    size_t sz = (size_t)N * sizeof(double);

    // Use d_partial_a as temp output buffer if it's big enough, otherwise alloc
    // For N particles we need N*4 doubles. d_partial_a has grid*4 doubles.
    // grid = N/256 so grid*4 << N*4. We need a temporary.
    double* d_out;
    size_t sz_out = (size_t)N * 4 * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_out, sz_out));

    int grid = state->grid_size;
    pfd_get_particles_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz, state->d_pcb,
        d_out, N);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    CUDA_CHECK(cudaMemcpy(output, d_out, sz_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
}

// ============================================================
// Explicit synchronization
// ============================================================

void pf_device_sync(PFDeviceState* state) {
    if (!state || !state->allocated) return;
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
}

}  // namespace gnss_gpu
