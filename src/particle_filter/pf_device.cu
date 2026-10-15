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
#include <array>
#include <cstdint>
#if defined(_M_X64) || defined(__x86_64__)
#include <emmintrin.h>
#endif

namespace gnss_gpu {

static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_SATS = 64;

__device__ __forceinline__ double pfd_weighted_huber_cost(
    double residual, double sigma, double obs_weight, double huber_k) {
    double z = fabs(residual) / sigma;
    double rho = (z < huber_k)
        ? 0.5 * z * z
        : huber_k * z - 0.5 * huber_k * huber_k;
    return obs_weight * rho;
}

__device__ bool pfd_solve_4x4(double A[4][5], double* x) {
    for (int col = 0; col < 4; col++) {
        int max_row = col;
        for (int row = col + 1; row < 4; row++) {
            if (fabs(A[row][col]) > fabs(A[max_row][col])) {
                max_row = row;
            }
        }
        if (max_row != col) {
            for (int k = 0; k < 5; k++) {
                double tmp = A[col][k];
                A[col][k] = A[max_row][k];
                A[max_row][k] = tmp;
            }
        }
        if (fabs(A[col][col]) < 1e-12) {
            return false;
        }
        for (int row = col + 1; row < 4; row++) {
            double factor = A[row][col] / A[col][col];
            for (int k = col; k < 5; k++) {
                A[row][k] -= factor * A[col][k];
            }
        }
    }
    for (int row = 3; row >= 0; row--) {
        double value = A[row][4];
        for (int col = row + 1; col < 4; col++) {
            value -= A[row][col] * x[col];
        }
        x[row] = value / A[row][row];
    }
    return isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2]) && isfinite(x[3]);
}

// ============================================================
// Kernels (self-contained, no extern dependencies)
// ============================================================

__global__ void pfd_init_kernel(double* px, double* py, double* pz,
                                double* vx, double* vy, double* vz,
                                double* vcov,
                                double* pcb,
                                double* log_weights,
                                double init_x, double init_y, double init_z, double init_cb,
                                double init_vx, double init_vy, double init_vz,
                                double spread_pos, double spread_cb, double spread_vel,
                                double init_vel_sigma,
                                int N, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    px[tid] = init_x + curand_normal_double(&state) * spread_pos;
    py[tid] = init_y + curand_normal_double(&state) * spread_pos;
    pz[tid] = init_z + curand_normal_double(&state) * spread_pos;
    vx[tid] = init_vx;
    vy[tid] = init_vy;
    vz[tid] = init_vz;
    if (spread_vel > 0.0) {
        vx[tid] += curand_normal_double(&state) * spread_vel;
        vy[tid] += curand_normal_double(&state) * spread_vel;
        vz[tid] += curand_normal_double(&state) * spread_vel;
    }
    double init_vel_var = init_vel_sigma * init_vel_sigma;
    int cov_off = tid * 9;
    for (int k = 0; k < 9; k++) {
        vcov[cov_off + k] = 0.0;
    }
    vcov[cov_off + 0] = init_vel_var;
    vcov[cov_off + 4] = init_vel_var;
    vcov[cov_off + 8] = init_vel_var;
    pcb[tid] = init_cb + curand_normal_double(&state) * spread_cb;
    log_weights[tid] = 0.0;
}

// Separate the uniform mode choices at launch time so the common prediction
// does not retain the KF covariance/Cholesky register footprint. Philox calls
// and each FP64 expression remain in their preceding order.
template <bool VELOCITY_KF, bool VELOCITY_NOISE>
__global__ void pfd_predict_kernel(double* px, double* py, double* pz,
                                   double* vx, double* vy, double* vz,
                                   double* vcov,
                                   double* pcb,
                                   double guide_vx, double guide_vy, double guide_vz,
                                   double dt, double sigma_pos, double sigma_cb,
                                   double sigma_vel, double velocity_guide_alpha,
                                   double velocity_process_noise,
                                   int N, unsigned long long seed, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, step, &state);

    double alpha = fmin(1.0, fmax(0.0, velocity_guide_alpha));
    double vxi = vx[tid];
    double vyi = vy[tid];
    double vzi = vz[tid];

    // The preceding kernel rounds alpha*guide before fusing the weighted prior
    // velocity. Capture this contraction explicitly when the guide is a launch
    // parameter, so constant-memory scheduling cannot change the rounding.
    vxi = __fma_rn(1.0 - alpha, vxi, __dmul_rn(alpha, guide_vx));
    vyi = __fma_rn(1.0 - alpha, vyi, __dmul_rn(alpha, guide_vy));
    vzi = __fma_rn(1.0 - alpha, vzi, __dmul_rn(alpha, guide_vz));
    if constexpr (VELOCITY_NOISE) {
        vxi += curand_normal_double(&state) * sigma_vel;
        vyi += curand_normal_double(&state) * sigma_vel;
        vzi += curand_normal_double(&state) * sigma_vel;
    }
    vx[tid] = vxi;
    vy[tid] = vyi;
    vz[tid] = vzi;

    double nx = 0.0;
    double ny = 0.0;
    double nz = 0.0;
    if constexpr (VELOCITY_KF) {
        int cov_off = tid * 9;
        double dt2 = dt * dt;
        double sigma_pos2 = sigma_pos * sigma_pos;
        double a00 = sigma_pos2 + dt2 * vcov[cov_off + 0];
        double a01 = dt2 * vcov[cov_off + 1];
        double a02 = dt2 * vcov[cov_off + 2];
        double a11 = sigma_pos2 + dt2 * vcov[cov_off + 4];
        double a12 = dt2 * vcov[cov_off + 5];
        double a22 = sigma_pos2 + dt2 * vcov[cov_off + 8];

        double l00 = sqrt(fmax(a00, 0.0));
        double l10 = 0.0;
        double l20 = 0.0;
        if (l00 > 1e-12) {
            l10 = a01 / l00;
            l20 = a02 / l00;
        }
        double l11 = sqrt(fmax(a11 - l10 * l10, 0.0));
        double l21 = 0.0;
        if (l11 > 1e-12) {
            l21 = (a12 - l20 * l10) / l11;
        }
        double l22 = sqrt(fmax(a22 - l20 * l20 - l21 * l21, 0.0));

        double z0 = curand_normal_double(&state);
        double z1 = curand_normal_double(&state);
        double z2 = curand_normal_double(&state);
        nx = l00 * z0;
        ny = l10 * z0 + l11 * z1;
        nz = l20 * z0 + l21 * z1 + l22 * z2;

        double qdt = fmax(velocity_process_noise, 0.0) * fmax(dt, 0.0);
        vcov[cov_off + 0] += qdt;
        vcov[cov_off + 4] += qdt;
        vcov[cov_off + 8] += qdt;
    } else {
        nx = curand_normal_double(&state) * sigma_pos;
        ny = curand_normal_double(&state) * sigma_pos;
        nz = curand_normal_double(&state) * sigma_pos;
    }

    px[tid] += vxi * dt + nx;
    py[tid] += vyi * dt + ny;
    pz[tid] += vzi * dt + nz;
    pcb[tid] += curand_normal_double(&state) * sigma_cb;
}

// Uniform covariance is certified by initialization and preserved by common
// process noise/resampling. All original producers remain available.
template <bool VELOCITY_KF, bool VELOCITY_NOISE>
__global__ void pfd_predict_common_kernel(double* px, double* py, double* pz,
                                   double* vx, double* vy, double* vz,
                                   double* vcov,
                                   double* pcb,
                                   double guide_vx, double guide_vy, double guide_vz,
                                   double dt, double sigma_pos, double sigma_cb,
                                   double sigma_vel, double velocity_guide_alpha,
                                   double velocity_process_noise,
                                   int N, unsigned long long seed, int step) {
    static_assert(VELOCITY_KF && !VELOCITY_NOISE);
    __shared__ double factor[6];
    if (threadIdx.x == 0) {
        // Read this block's own first particle before any in-place updates.
        // Reading particle0 in every block would race with block0's Q update.
        int cov_off = (blockIdx.x * blockDim.x) * 9;
        double dt2 = dt * dt;
        double sigma_pos2 = sigma_pos * sigma_pos;
        double a00 = sigma_pos2 + dt2 * vcov[cov_off + 0];
        double a01 = dt2 * vcov[cov_off + 1];
        double a02 = dt2 * vcov[cov_off + 2];
        double a11 = sigma_pos2 + dt2 * vcov[cov_off + 4];
        double a12 = dt2 * vcov[cov_off + 5];
        double a22 = sigma_pos2 + dt2 * vcov[cov_off + 8];

        double l00 = sqrt(fmax(a00, 0.0));
        double l10 = 0.0;
        double l20 = 0.0;
        if (l00 > 1e-12) {
            l10 = a01 / l00;
            l20 = a02 / l00;
        }
        double l11 = sqrt(fmax(a11 - l10 * l10, 0.0));
        double l21 = 0.0;
        if (l11 > 1e-12) {
            l21 = (a12 - l20 * l10) / l11;
        }
        double l22 = sqrt(fmax(a22 - l20 * l20 - l21 * l21, 0.0));

        factor[0] = l00;
        factor[1] = l10;
        factor[2] = l20;
        factor[3] = l11;
        factor[4] = l21;
        factor[5] = l22;
    }
    // Every tail thread participates before the original early return.
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, step, &state);

    double alpha = fmin(1.0, fmax(0.0, velocity_guide_alpha));
    double vxi = vx[tid];
    double vyi = vy[tid];
    double vzi = vz[tid];

    // The preceding kernel rounds alpha*guide before fusing the weighted prior
    // velocity. Capture this contraction explicitly when the guide is a launch
    // parameter, so constant-memory scheduling cannot change the rounding.
    vxi = __fma_rn(1.0 - alpha, vxi, __dmul_rn(alpha, guide_vx));
    vyi = __fma_rn(1.0 - alpha, vyi, __dmul_rn(alpha, guide_vy));
    vzi = __fma_rn(1.0 - alpha, vzi, __dmul_rn(alpha, guide_vz));
    if constexpr (VELOCITY_NOISE) {
        vxi += curand_normal_double(&state) * sigma_vel;
        vyi += curand_normal_double(&state) * sigma_vel;
        vzi += curand_normal_double(&state) * sigma_vel;
    }
    vx[tid] = vxi;
    vy[tid] = vyi;
    vz[tid] = vzi;

    double nx = 0.0;
    double ny = 0.0;
    double nz = 0.0;
    if constexpr (VELOCITY_KF) {
        int cov_off = tid * 9;
        double l00 = factor[0];
        double l10 = factor[1];
        double l20 = factor[2];
        double l11 = factor[3];
        double l21 = factor[4];
        double l22 = factor[5];

        double z0 = curand_normal_double(&state);
        double z1 = curand_normal_double(&state);
        double z2 = curand_normal_double(&state);
        nx = l00 * z0;
        ny = l10 * z0 + l11 * z1;
        nz = l20 * z0 + l21 * z1 + l22 * z2;

        double qdt = fmax(velocity_process_noise, 0.0) * fmax(dt, 0.0);
        vcov[cov_off + 0] += qdt;
        vcov[cov_off + 4] += qdt;
        vcov[cov_off + 8] += qdt;
    } else {
        nx = curand_normal_double(&state) * sigma_pos;
        ny = curand_normal_double(&state) * sigma_pos;
        nz = curand_normal_double(&state) * sigma_pos;
    }

    px[tid] += vxi * dt + nx;
    py[tid] += vyi * dt + ny;
    pz[tid] += vzi * dt + nz;
    pcb[tid] += curand_normal_double(&state) * sigma_cb;
}

// The default Gaussian path retains the same FP64 expression and satellite
// accumulation order. Compute its common reciprocal once per block and allow
// two independent satellite geometries to overlap without robust-mode branches.
__global__ void pfd_weight_gaussian_kernel(const double* px, const double* py,
                                          const double* pz, const double* pcb,
                                          const double* sat_ecef,
                                          const double* pseudoranges,
                                          const double* weights_sat,
                                          double* log_weights,
                                          int N, int n_sat, double sigma_pr) {
    extern __shared__ double s_data[];
    __shared__ double s_inv_sigma2;
    double* s_sat = s_data;
    double* s_pr = s_data + n_sat * 3;
    double* s_ws = s_data + n_sat * 4;
    if (threadIdx.x == 0) s_inv_sigma2 = 1.0 / (sigma_pr * sigma_pr);
    for (int s = threadIdx.x; s < n_sat; s += blockDim.x) {
        s_sat[s * 3 + 0] = sat_ecef[s * 3 + 0];
        s_sat[s * 3 + 1] = sat_ecef[s * 3 + 1];
        s_sat[s * 3 + 2] = sat_ecef[s * 3 + 2];
        s_pr[s] = pseudoranges[s];
        s_ws[s] = weights_sat[s];
    }
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    double x = px[tid], y = py[tid], z = pz[tid], cb = pcb[tid];
    double inv_sigma2 = s_inv_sigma2;
    double log_w = 0.0;
    #pragma unroll 2
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

// Retain the preceding Student product and ordered log accumulation.
__global__ void pfd_weight_student_kernel(const double* px, const double* py,
                                         const double* pz, const double* pcb,
                                         const double* sat_ecef, const double* pseudoranges,
                                         const double* weights_sat, double* log_weights,
                                         int N, int n_sat, double sigma_pr, double nu) {
    extern __shared__ double s_data[];
    __shared__ double s_inv_scale;
    __shared__ double s_half_nup1;
    double* s_sat = s_data;
    double* s_pr = s_data + n_sat * 3;
    double* s_ws = s_data + n_sat * 4;
    if (threadIdx.x == 0) {
        s_inv_scale = 1.0 / (nu * sigma_pr * sigma_pr);
        s_half_nup1 = 0.5 * (nu + 1.0);
    }
    for (int s = threadIdx.x; s < n_sat; s += blockDim.x) {
        s_sat[s * 3 + 0] = sat_ecef[s * 3 + 0];
        s_sat[s * 3 + 1] = sat_ecef[s * 3 + 1];
        s_sat[s * 3 + 2] = sat_ecef[s * 3 + 2];
        s_pr[s] = pseudoranges[s];
        s_ws[s] = weights_sat[s];
    }
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    double x = px[tid], y = py[tid], z = pz[tid], cb = pcb[tid];
    double log_w = 0.0;
    #pragma unroll 2
    for (int s = 0; s < n_sat; s++) {
        double dx = x - s_sat[s * 3 + 0];
        double dy = y - s_sat[s * 3 + 1];
        double dz = z - s_sat[s * 3 + 2];
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double residual = s_pr[s] - (r + cb);
        log_w += -s_half_nup1 * log(1.0 + s_ws[s] * residual * residual * s_inv_scale);
    }
    log_weights[tid] = log_w;
}

__global__ void pfd_weight_kernel(const double* px, const double* py,
                                  const double* pz, const double* pcb,
                                  const double* sat_ecef,
                                  const double* pseudoranges,
                                  const double* weights_sat,
                                  double* log_weights,
                                  int N, int n_sat, double sigma_pr,
                                  double nu,
                                  double per_particle_nlos_threshold_m,
                                  bool huber_enabled,
                                  double huber_k) {
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

    // nu <= 0: Gaussian likelihood (default)
    // nu > 0:  Student's t likelihood (robust to NLOS outliers)
    //          nu=1 is Cauchy (most robust), nu=3-5 moderately robust
    int use_student_t = (nu > 0.0);
    double half_nup1 = 0.5 * (nu + 1.0);
    double inv_nu_sigma2 = (nu > 0.0) ? 1.0 / (nu * sigma_pr * sigma_pr) : 0.0;
    bool use_huber = huber_enabled && !use_student_t && huber_k > 0.0;
    bool use_per_particle_gate = per_particle_nlos_threshold_m > 0.0;
    if (use_per_particle_gate) {
        int kept = 0;
        for (int s = 0; s < n_sat; s++) {
            double dx = x - s_sat[s * 3 + 0];
            double dy = y - s_sat[s * 3 + 1];
            double dz = z - s_sat[s * 3 + 2];
            double r = sqrt(dx * dx + dy * dy + dz * dz);
            double residual = s_pr[s] - (r + cb);
            if (fabs(residual) <= per_particle_nlos_threshold_m) {
                kept++;
            }
        }
        int min_kept = (n_sat < 4) ? n_sat : 4;
        if (kept < min_kept) {
            use_per_particle_gate = false;
        }
    }

    for (int s = 0; s < n_sat; s++) {
        double dx = x - s_sat[s * 3 + 0];
        double dy = y - s_sat[s * 3 + 1];
        double dz = z - s_sat[s * 3 + 2];
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double pred_pr = r + cb;
        double residual = s_pr[s] - pred_pr;
        if (use_per_particle_gate &&
            fabs(residual) > per_particle_nlos_threshold_m) {
            continue;
        }

        if (use_student_t) {
            // Student's t: log p = -((nu+1)/2) * log(1 + r^2/(nu*sigma^2))
            log_w += -half_nup1 * log(1.0 + s_ws[s] * residual * residual * inv_nu_sigma2);
        } else if (use_huber) {
            log_w += -pfd_weighted_huber_cost(residual, sigma_pr, s_ws[s], huber_k);
        } else {
            // Gaussian: log p = -0.5 * r^2/sigma^2
            log_w += -0.5 * s_ws[s] * residual * residual * inv_sigma2;
        }
    }

    log_weights[tid] = log_w;
}

// Positive undifferenced PR gates retain the original observation order.
// Mode 0: Gaussian; 1: Huber; 2: Student (which takes priority over Huber).
template<int Mode>
__global__ void pfd_weight_pr_gate_kernel(
    const double* px, const double* py, const double* pz, const double* pcb,
    const double* sat_ecef, const double* pseudoranges, const double* weights_sat,
    double* log_weights, int N, int n_sat, double sigma_pr, double nu,
    double threshold, double huber_k) {
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
    double x = px[tid], y = py[tid], z = pz[tid], cb = pcb[tid];
    double inv_sigma2 = 1.0 / (sigma_pr * sigma_pr);
    double half_nup1 = 0.5 * (nu + 1.0);
    double inv_nu_sigma2 = 1.0 / (nu * sigma_pr * sigma_pr);
    double all_log_w = 0.0, kept_log_w = 0.0;
    int kept = 0;
    for (int s = 0; s < n_sat; ++s) {
        double dx = x - s_sat[s * 3 + 0];
        double dy = y - s_sat[s * 3 + 1];
        double dz = z - s_sat[s * 3 + 2];
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double residual = s_pr[s] - (r + cb);
        double magnitude = fabs(residual);
        kept += magnitude <= threshold;
        // Counting excludes NaNs, but the original skip-if-greater filter
        // retains them when enough other observations pass the gate.
        bool adopt = !(magnitude > threshold);
        if constexpr (Mode == 2) {
            double value = log(1.0 + s_ws[s] * residual * residual * inv_nu_sigma2);
            all_log_w = fma(-half_nup1, value, all_log_w);
            if (adopt) kept_log_w = fma(-half_nup1, value, kept_log_w);
        } else if constexpr (Mode == 1) {
            double rho = pfd_weighted_huber_cost(residual, sigma_pr, 1.0, huber_k);
            all_log_w = fma(-s_ws[s], rho, all_log_w);
            if (adopt) kept_log_w = fma(-s_ws[s], rho, kept_log_w);
        } else {
            double term = -0.5 * s_ws[s] * residual * residual;
            all_log_w = fma(term, inv_sigma2, all_log_w);
            if (adopt) kept_log_w = fma(term, inv_sigma2, kept_log_w);
        }
    }
    int min_kept = n_sat < 4 ? n_sat : 4;
    log_weights[tid] = kept >= min_kept ? kept_log_w : all_log_w;
}

// --- DD pseudorange weight kernel ---
// Double-differenced pseudorange eliminates receiver clock bias from both rover
// and base. The expected DD range is:
//   dd_expected = range_to_k - range_to_ref - base_range_k + base_range_ref
template<bool CommonReference, bool CachedSigma = false>
__global__ void pfd_weight_dd_pseudorange_kernel(
    const double* px, const double* py,
    const double* pz,
    const double* dd_data,      // layout: [sat_k: n_dd*3][ref: n_dd*3][dd_pr: n_dd][base_range_k: n_dd][base_range_ref: n_dd][weights: n_dd]
    double* log_weights,
    int N, int n_dd,
    double sigma_pr,
    double per_particle_nlos_threshold_m,
    bool huber_enabled,
    double huber_k) {

    extern __shared__ double s_data[];
    double* s_sat_k = s_data;                   // n_dd * 3
    double* s_ref = s_data + n_dd * 3;         // n_dd * 3
    double* s_dd_pr = s_data + n_dd * 6;       // n_dd
    double* s_br_k = s_data + n_dd * 7;        // n_dd
    double* s_br_ref = s_data + n_dd * 8;      // n_dd
    double* s_ws = s_data + n_dd * 9;          // n_dd

    int total_shared = n_dd * 10;
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x) {
        s_data[i] = dd_data[i];
    }
    if constexpr (CachedSigma) {
        if (threadIdx.x == 0) {
            s_data[total_shared + 0] = 1.0 / (sigma_pr * sigma_pr);
        }
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double x = px[tid];
    double y = py[tid];
    double z = pz[tid];

    // One exact FP64 reference range per particle when every pair uses the
    // same coordinates. Rebuilt on each call; mixed references keep the
    // original per-pair geometry and ordered residual/likelihood arithmetic.
    double common_ref_range = 0.0;
    if constexpr (CommonReference) {
        double dx_ref = x - s_ref[0];
        double dy_ref = y - s_ref[1];
        double dz_ref = z - s_ref[2];
        common_ref_range = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
    }

    double inv_sigma2 = CachedSigma ? s_data[total_shared + 0] : 1.0 / (sigma_pr * sigma_pr);
    double log_w = 0.0;
    bool use_huber = huber_enabled && huber_k > 0.0;
    bool use_per_particle_gate = per_particle_nlos_threshold_m > 0.0;
    if (use_per_particle_gate) {
        int kept = 0;
        for (int d = 0; d < n_dd; d++) {
            double dx_k = x - s_sat_k[d * 3 + 0];
            double dy_k = y - s_sat_k[d * 3 + 1];
            double dz_k = z - s_sat_k[d * 3 + 2];
            double r_k = sqrt(dx_k * dx_k + dy_k * dy_k + dz_k * dz_k);

            double r_ref = common_ref_range;
            if constexpr (!CommonReference) {
                double dx_ref = x - s_ref[d * 3 + 0];
                double dy_ref = y - s_ref[d * 3 + 1];
                double dz_ref = z - s_ref[d * 3 + 2];
                r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
            }

            double dd_expected = r_k - r_ref - s_br_k[d] + s_br_ref[d];
            double residual = s_dd_pr[d] - dd_expected;
            if (fabs(residual) <= per_particle_nlos_threshold_m) {
                kept++;
            }
        }
        int min_kept = (n_dd < 3) ? n_dd : 3;
        if (kept < min_kept) {
            use_per_particle_gate = false;
        }
    }

    for (int d = 0; d < n_dd; d++) {
        double dx_k = x - s_sat_k[d * 3 + 0];
        double dy_k = y - s_sat_k[d * 3 + 1];
        double dz_k = z - s_sat_k[d * 3 + 2];
        double r_k = sqrt(dx_k * dx_k + dy_k * dy_k + dz_k * dz_k);

        double r_ref = common_ref_range;
        if constexpr (!CommonReference) {
            double dx_ref = x - s_ref[d * 3 + 0];
            double dy_ref = y - s_ref[d * 3 + 1];
            double dz_ref = z - s_ref[d * 3 + 2];
            r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
        }

        double dd_expected = r_k - r_ref - s_br_k[d] + s_br_ref[d];
        double residual = s_dd_pr[d] - dd_expected;
        if (use_per_particle_gate &&
            fabs(residual) > per_particle_nlos_threshold_m) {
            continue;
        }
        if (use_huber) {
            log_w += -pfd_weighted_huber_cost(residual, sigma_pr, s_ws[d], huber_k);
        } else {
            log_w += -0.5 * s_ws[d] * residual * residual * inv_sigma2;
        }
    }

    log_weights[tid] = log_w;
}

// Individual Gaussian DD gates retain their original skip-if-greater
// semantics (including NaNs) and their overwrite/add update distinction.
template<bool Carrier>
__global__ void pfd_weight_dd_single_gated_kernel(
    const double* px, const double* py, const double* pz, const double* dd_data,
    double* log_weights, int N, int n_dd, double sigma, double threshold) {
    extern __shared__ double data[];
    constexpr int columns = Carrier ? 11 : 10;
    int total_shared = n_dd * columns;
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x) {
        if constexpr (Carrier)
            data[i] = i >= n_dd * 10 ? 1.0 / dd_data[i] : dd_data[i];
        else
            data[i] = dd_data[i];
    }
    if (threadIdx.x == 0) data[total_shared] = 1.0 / (sigma * sigma);
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    double x = px[tid], y = py[tid], z = pz[tid];
    double dx_ref = x - data[n_dd * 3];
    double dy_ref = y - data[n_dd * 3 + 1];
    double dz_ref = z - data[n_dd * 3 + 2];
    double r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
    double inv_sigma2 = data[total_shared];
    double all = 0.0, filtered = 0.0;
    int kept = 0;
    for (int d = 0; d < n_dd; ++d) {
        double dx = x - data[d * 3];
        double dy = y - data[d * 3 + 1];
        double dz = z - data[d * 3 + 2];
        double range = sqrt(dx * dx + dy * dy + dz * dz) - r_ref
            - data[n_dd * 7 + d] + data[n_dd * 8 + d];
        double residual;
        if constexpr (Carrier) {
            double cycles = data[n_dd * 6 + d] - range * data[n_dd * 10 + d];
            residual = cycles - rint(cycles);
        } else {
            residual = data[n_dd * 6 + d] - range;
        }
        if (fabs(residual) <= threshold) ++kept;
        double term;
        if constexpr (Carrier)
            term = -0.5 * data[n_dd * 9 + d] * (residual * residual);
        else
            term = -0.5 * data[n_dd * 9 + d] * residual * residual;
        all = __fma_rn(term, inv_sigma2, all);
        if (!(fabs(residual) > threshold))
            filtered = __fma_rn(term, inv_sigma2, filtered);
    }
    int minimum = n_dd < 3 ? n_dd : 3;
    double value = kept >= minimum ? filtered : all;
    if constexpr (Carrier)
        log_weights[tid] = __dadd_rn(log_weights[tid], value);
    else
        log_weights[tid] = value;
}

// Individual Huber gates retain the normalized cost and weight FMA,
// original skip-if-greater semantics, and the overwrite/add distinction.
template<bool Carrier>
__global__ void pfd_weight_dd_single_huber_gated_kernel(
    const double* px, const double* py, const double* pz, const double* dd_data,
    double* log_weights, int N, int n_dd, double sigma, double threshold, double huber_k) {
    extern __shared__ double data[];
    constexpr int columns = Carrier ? 11 : 10;
    int total_shared = n_dd * columns;
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x) {
        if constexpr (Carrier)
            data[i] = i >= n_dd * 10 ? 1.0 / dd_data[i] : dd_data[i];
        else
            data[i] = dd_data[i];
    }
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    double x = px[tid], y = py[tid], z = pz[tid];
    double dx_ref = x - data[n_dd * 3];
    double dy_ref = y - data[n_dd * 3 + 1];
    double dz_ref = z - data[n_dd * 3 + 2];
    double r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
    double all = 0.0, filtered = 0.0;
    int kept = 0;
    for (int d = 0; d < n_dd; ++d) {
        double dx = x - data[d * 3];
        double dy = y - data[d * 3 + 1];
        double dz = z - data[d * 3 + 2];
        double range = sqrt(dx * dx + dy * dy + dz * dz) - r_ref
            - data[n_dd * 7 + d] + data[n_dd * 8 + d];
        double residual;
        if constexpr (Carrier) {
            double cycles = data[n_dd * 6 + d] - range * data[n_dd * 10 + d];
            residual = cycles - rint(cycles);
        } else {
            residual = data[n_dd * 6 + d] - range;
        }
        if (fabs(residual) <= threshold) ++kept;
        double rho = pfd_weighted_huber_cost(residual, sigma, 1.0, huber_k);
        double weight = data[n_dd * 9 + d];
        all = __fma_rn(-weight, rho, all);
        if (!(fabs(residual) > threshold))
            filtered = __fma_rn(-weight, rho, filtered);
    }
    int minimum = n_dd < 3 ? n_dd : 3;
    double value = kept >= minimum ? filtered : all;
    if constexpr (Carrier)
        log_weights[tid] = __dadd_rn(log_weights[tid], value);
    else
        log_weights[tid] = value;
}

// Mixed references retain per-pair ranges but evaluate gate and cost in one loop.
template<bool Carrier, bool Huber>
__global__ void pfd_weight_dd_mixed_gated_kernel(
    const double* px, const double* py, const double* pz, const double* dd_data,
    double* log_weights, int N, int n_dd, double sigma, double threshold, double huber_k) {
    extern __shared__ double data[];
    constexpr int columns = Carrier ? 11 : 10;
    int total_shared = n_dd * columns;
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x) {
        if constexpr (Carrier)
            data[i] = i >= n_dd * 10 ? 1.0 / dd_data[i] : dd_data[i];
        else
            data[i] = dd_data[i];
    }
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    double x = px[tid], y = py[tid], z = pz[tid];
    double inv_sigma2 = 0.0;
    if constexpr (!Huber) inv_sigma2 = 1.0 / (sigma * sigma);
    double all = 0.0, filtered = 0.0;
    int kept = 0;
    for (int d = 0; d < n_dd; ++d) {
        double dx = x - data[d * 3];
        double dy = y - data[d * 3 + 1];
        double dz = z - data[d * 3 + 2];
        double dx_ref = x - data[n_dd * 3 + d * 3];
        double dy_ref = y - data[n_dd * 3 + d * 3 + 1];
        double dz_ref = z - data[n_dd * 3 + d * 3 + 2];
        double r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
        double range = sqrt(dx * dx + dy * dy + dz * dz) - r_ref
            - data[n_dd * 7 + d] + data[n_dd * 8 + d];
        double residual;
        if constexpr (Carrier) {
            double cycles = data[n_dd * 6 + d] - range * data[n_dd * 10 + d];
            residual = cycles - rint(cycles);
        } else {
            residual = data[n_dd * 6 + d] - range;
        }
        if (fabs(residual) <= threshold) ++kept;
        if constexpr (Huber) {
            double rho = pfd_weighted_huber_cost(residual, sigma, 1.0, huber_k);
            double weight = data[n_dd * 9 + d];
            all = __fma_rn(-weight, rho, all);
            if (!(fabs(residual) > threshold))
                filtered = __fma_rn(-weight, rho, filtered);
        } else {
            double term;
            if constexpr (Carrier)
                term = -0.5 * data[n_dd * 9 + d] * (residual * residual);
            else
                term = -0.5 * data[n_dd * 9 + d] * residual * residual;
            all = __fma_rn(term, inv_sigma2, all);
            if (!(fabs(residual) > threshold))
                filtered = __fma_rn(term, inv_sigma2, filtered);
        }
    }
    int minimum = n_dd < 3 ? n_dd : 3;
    double value = kept >= minimum ? filtered : all;
    if constexpr (Carrier)
        log_weights[tid] = __dadd_rn(log_weights[tid], value);
    else
        log_weights[tid] = value;
}

// --- GMM weight kernel ---
// GMM likelihood: p(residual) = w_los * N(0, sigma_los) + w_nlos * N(mu_nlos, sigma_nlos)
// Uses logsumexp trick for numerical stability.
__global__ void pfd_weight_gmm_kernel(const double* px, const double* py,
                                      const double* pz, const double* pcb,
                                      const double* sat_ecef,
                                      const double* pseudoranges,
                                      const double* weights_sat,
                                      double* log_weights,
                                      int N, int n_sat, double sigma_los,
                                      double w_los, double mu_nlos, double sigma_nlos) {
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

    double inv_sigma_los2 = 1.0 / (sigma_los * sigma_los);
    double inv_sigma_nlos2 = 1.0 / (sigma_nlos * sigma_nlos);
    double log_w_los = log(w_los);
    double log_w_nlos = log(1.0 - w_los);
    double log_sigma_los = log(sigma_los);
    double log_sigma_nlos = log(sigma_nlos);

    double log_w = 0.0;

    for (int s = 0; s < n_sat; s++) {
        double dx = x - s_sat[s * 3 + 0];
        double dy = y - s_sat[s * 3 + 1];
        double dz = z - s_sat[s * 3 + 2];
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double pred_pr = r + cb;
        double residual = s_pr[s] - pred_pr;

        // LOS component: log(w_los * N(0, sigma_los)) = log(w_los) - log(sigma_los) - 0.5*(r/sigma_los)^2
        double log_los = log_w_los - log_sigma_los - 0.5 * s_ws[s] * residual * residual * inv_sigma_los2;
        // NLOS component: log(w_nlos * N(mu_nlos, sigma_nlos))
        double r_nlos = residual - mu_nlos;
        double log_nlos = log_w_nlos - log_sigma_nlos - 0.5 * s_ws[s] * r_nlos * r_nlos * inv_sigma_nlos2;

        // logsumexp: log(exp(a) + exp(b)) = max(a,b) + log(exp(a-max) + exp(b-max))
        double mx = fmax(log_los, log_nlos);
        log_w += mx + log(exp(log_los - mx) + exp(log_nlos - mx));
    }

    log_weights[tid] = log_w;
}

// Block-shared exact GPU GMM coefficients and finite-logit single-exp logsumexp.
__global__ void pfd_weight_gmm_shared_kernel(const double* px, const double* py,
                                      const double* pz, const double* pcb,
                                      const double* sat_ecef,
                                      const double* pseudoranges,
                                      const double* weights_sat,
                                      double* log_weights,
                                      int N, int n_sat, double sigma_los,
                                      double w_los, double mu_nlos, double sigma_nlos) {
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
    if (threadIdx.x == 0) {
        int begin = n_sat * 5;
        s_data[begin] = 1.0 / (sigma_los * sigma_los);
        s_data[begin + 1] = 1.0 / (sigma_nlos * sigma_nlos);
        s_data[begin + 2] = log(w_los);
        s_data[begin + 3] = log(1.0 - w_los);
        s_data[begin + 4] = log(sigma_los);
        s_data[begin + 5] = log(sigma_nlos);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double x = px[tid];
    double y = py[tid];
    double z = pz[tid];
    double cb = pcb[tid];

    int coefficients = n_sat * 5;
    double inv_sigma_los2 = s_data[coefficients];
    double inv_sigma_nlos2 = s_data[coefficients + 1];
    double log_w_los = s_data[coefficients + 2];
    double log_w_nlos = s_data[coefficients + 3];
    double log_sigma_los = s_data[coefficients + 4];
    double log_sigma_nlos = s_data[coefficients + 5];

    double log_w = 0.0;

    for (int s = 0; s < n_sat; s++) {
        double dx = x - s_sat[s * 3 + 0];
        double dy = y - s_sat[s * 3 + 1];
        double dz = z - s_sat[s * 3 + 2];
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double pred_pr = r + cb;
        double residual = s_pr[s] - pred_pr;

        // LOS component: log(w_los * N(0, sigma_los)) = log(w_los) - log(sigma_los) - 0.5*(r/sigma_los)^2
        double log_los = log_w_los - log_sigma_los - 0.5 * s_ws[s] * residual * residual * inv_sigma_los2;
        // NLOS component: log(w_nlos * N(mu_nlos, sigma_nlos))
        double r_nlos = residual - mu_nlos;
        double log_nlos = log_w_nlos - log_sigma_nlos - 0.5 * s_ws[s] * r_nlos * r_nlos * inv_sigma_nlos2;

        // logsumexp: log(exp(a) + exp(b)) = max(a,b) + log(exp(a-max) + exp(b-max))
        double mx = fmax(log_los, log_nlos);
        double sum_exp;
        if (isfinite(log_los) && isfinite(log_nlos)) {
            // One component equals mx exactly, so its original exp(0) is 1.
            double difference = log_los < log_nlos ? log_los - log_nlos : log_nlos - log_los;
            sum_exp = 1.0 + exp(difference);
        } else {
            // Preserve the original IEEE behavior and operand order.
            sum_exp = exp(log_los - mx) + exp(log_nlos - mx);
        }
        log_w += mx + log(sum_exp);
    }

    log_weights[tid] = log_w;
}

// --- Carrier phase AFV weight kernel ---
// Ambiguity Function Value likelihood: uses fractional cycle residuals
// so integer ambiguity resolution is not needed. After pseudorange update
// clusters particles to ~3m, the correct integer peak dominates.
__global__ void pfd_weight_carrier_afv_kernel(
    const double* px, const double* py,
    const double* pz, const double* pcb,
    const double* sat_ecef,
    const double* carrier_phase,   // [n_sat] in cycles
    const double* weights_sat,
    double* log_weights,
    int N, int n_sat,
    double wavelength,      // L1 ~ 0.190293673 m
    double sigma_cycles) {  // ~ 0.05 cycles

    // Dynamic shared memory layout: [sat_ecef: n_sat*3] [cp: n_sat] [ws: n_sat]
    extern __shared__ double s_data[];
    double* s_sat = s_data;
    double* s_cp  = s_data + n_sat * 3;
    double* s_ws  = s_data + n_sat * 4;

    for (int i = threadIdx.x; i < n_sat; i += blockDim.x) {
        s_sat[i * 3 + 0] = sat_ecef[i * 3 + 0];
        s_sat[i * 3 + 1] = sat_ecef[i * 3 + 1];
        s_sat[i * 3 + 2] = sat_ecef[i * 3 + 2];
        s_cp[i] = carrier_phase[i];
        s_ws[i] = weights_sat[i];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double x  = px[tid];
    double y  = py[tid];
    double z  = pz[tid];
    double cb = pcb[tid];

    double inv_wl = 1.0 / wavelength;
    double inv_sigma2 = 1.0 / (sigma_cycles * sigma_cycles);
    double log_w = 0.0;

    for (int s = 0; s < n_sat; s++) {
        double dx = x - s_sat[s * 3 + 0];
        double dy = y - s_sat[s * 3 + 1];
        double dz = z - s_sat[s * 3 + 2];
        double r = sqrt(dx * dx + dy * dy + dz * dz);

        // Expected carrier phase in cycles
        double expected_cycles = (r + cb) * inv_wl;
        double residual_cycles = s_cp[s] - expected_cycles;

        // AFV: distance to nearest integer cycle (fractional part)
        double afv = residual_cycles - rint(residual_cycles);

        log_w += -0.5 * s_ws[s] * (afv * afv) * inv_sigma2;
    }

    log_weights[tid] += log_w;
}

// --- DD Carrier phase AFV weight kernel ---
// Double-differenced: eliminates receiver clock bias from both rover and base.
// For each DD pair (sat_k vs ref):
//   dd_expected = (range_to_k - range_to_ref - base_range_k + base_range_ref) / wavelength
//   dd_residual = dd_carrier[k] - dd_expected
//   afv = dd_residual - round(dd_residual)
template<bool CommonReference, bool CachedSigma = false>
__global__ void pfd_weight_dd_carrier_afv_kernel(
    const double* px, const double* py,
    const double* pz,
    const double* dd_data,      // layout: [sat_k: n_dd*3][ref: n_dd*3][dd_carrier: n_dd][base_range_k: n_dd][base_range_ref: n_dd][weights: n_dd][wavelengths: n_dd]
    double* log_weights,
    int N, int n_dd,
    double sigma_cycles,
    double per_particle_nlos_threshold_cycles,
    bool huber_enabled,
    double huber_k) {

    // Dynamic shared memory layout:
    // [sat_k_ecef: n_dd*3] [ref_ecef: n_dd*3] [dd_carrier: n_dd]
    // [base_range_k: n_dd] [base_range_ref: n_dd] [weights: n_dd] [wavelengths: n_dd]
    extern __shared__ double s_data[];
    double* s_sat_k   = s_data;                      // n_dd * 3
    double* s_ref     = s_data + n_dd * 3;            // n_dd * 3
    double* s_dd_cp   = s_data + n_dd * 6;            // n_dd
    double* s_br_k    = s_data + n_dd * 7;            // n_dd
    double* s_br_ref  = s_data + n_dd * 8;            // n_dd
    double* s_ws      = s_data + n_dd * 9;            // n_dd
    double* s_inv_wl      = s_data + n_dd * 10;           // n_dd

    int total_shared = n_dd * 11;
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x) {
        // The final column is reused for exact GPU FP64 reciprocal wavelengths.
        // Its original division is now evaluated once per observation/block,
        // before the existing barrier, instead of once per particle/pair.
        s_data[i] = i >= n_dd * 10 ? 1.0 / dd_data[i] : dd_data[i];
    }
    if constexpr (CachedSigma) {
        if (threadIdx.x == 0) {
            s_data[total_shared + 0] = 1.0 / (sigma_cycles * sigma_cycles);
        }
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double x = px[tid];
    double y = py[tid];
    double z = pz[tid];

    // One exact FP64 reference range per particle when every pair uses the
    // same coordinates. Rebuilt on each call; mixed references keep the
    // original per-pair geometry and ordered residual/likelihood arithmetic.
    double common_ref_range = 0.0;
    if constexpr (CommonReference) {
        double dx_ref = x - s_ref[0];
        double dy_ref = y - s_ref[1];
        double dz_ref = z - s_ref[2];
        common_ref_range = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
    }

    double inv_sigma2 = CachedSigma ? s_data[total_shared + 0] : 1.0 / (sigma_cycles * sigma_cycles);
    double log_w = 0.0;
    bool use_huber = huber_enabled && huber_k > 0.0;
    bool use_per_particle_gate = per_particle_nlos_threshold_cycles > 0.0;
    if (use_per_particle_gate) {
        int kept = 0;
        for (int d = 0; d < n_dd; d++) {
            double dx_k = x - s_sat_k[d * 3 + 0];
            double dy_k = y - s_sat_k[d * 3 + 1];
            double dz_k = z - s_sat_k[d * 3 + 2];
            double r_k = sqrt(dx_k * dx_k + dy_k * dy_k + dz_k * dz_k);

            double r_ref = common_ref_range;
            if constexpr (!CommonReference) {
                double dx_ref = x - s_ref[d * 3 + 0];
                double dy_ref = y - s_ref[d * 3 + 1];
                double dz_ref = z - s_ref[d * 3 + 2];
                r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
            }

            double inv_wl = s_inv_wl[d];
            double dd_expected = (r_k - r_ref - s_br_k[d] + s_br_ref[d]) * inv_wl;
            double dd_residual = s_dd_cp[d] - dd_expected;
            double afv = dd_residual - rint(dd_residual);
            if (fabs(afv) <= per_particle_nlos_threshold_cycles) {
                kept++;
            }
        }
        int min_kept = (n_dd < 3) ? n_dd : 3;
        if (kept < min_kept) {
            use_per_particle_gate = false;
        }
    }

    for (int d = 0; d < n_dd; d++) {
        // Range from particle to satellite k
        double dx_k = x - s_sat_k[d * 3 + 0];
        double dy_k = y - s_sat_k[d * 3 + 1];
        double dz_k = z - s_sat_k[d * 3 + 2];
        double r_k = sqrt(dx_k * dx_k + dy_k * dy_k + dz_k * dz_k);

        double r_ref = common_ref_range;
        if constexpr (!CommonReference) {
            double dx_ref = x - s_ref[d * 3 + 0];
            double dy_ref = y - s_ref[d * 3 + 1];
            double dz_ref = z - s_ref[d * 3 + 2];
            r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
        }

        double inv_wl = s_inv_wl[d];

        // DD expected range in cycles (no clock bias!)
        double dd_expected = (r_k - r_ref - s_br_k[d] + s_br_ref[d]) * inv_wl;

        // DD residual
        double dd_residual = s_dd_cp[d] - dd_expected;

        // AFV: fractional cycle
        double afv = dd_residual - rint(dd_residual);
        if (use_per_particle_gate &&
            fabs(afv) > per_particle_nlos_threshold_cycles) {
            continue;
        }

        if (use_huber) {
            log_w += -pfd_weighted_huber_cost(afv, sigma_cycles, s_ws[d], huber_k);
        } else {
            log_w += -0.5 * s_ws[d] * (afv * afv) * inv_sigma2;
        }
    }

    log_weights[tid] += log_w;
}

// Joint DD pseudorange + carrier update. Sharing the range geometry avoids a
// second kernel launch and the duplicate square roots in the common path.
template<bool CommonReference, bool CachedSigma = false>
__global__ void pfd_weight_dd_joint_kernel(
    const double* px, const double* py, const double* pz,
    const double* dd_data,
    double* log_weights, int N, int n_dd,
    double sigma_pr, double sigma_cycles,
    double nlos_threshold_m, double nlos_threshold_cycles,
    bool huber_enabled, double huber_pr_k, double huber_carrier_k) {

    extern __shared__ double s_data[];
    double* s_sat_k = s_data;
    double* s_ref = s_data + n_dd * 3;
    double* s_dd_pr = s_data + n_dd * 6;
    double* s_dd_cp = s_data + n_dd * 7;
    double* s_br_k = s_data + n_dd * 8;
    double* s_br_ref = s_data + n_dd * 9;
    double* s_ws = s_data + n_dd * 10;
    double* s_inv_wl = s_data + n_dd * 11;

    int total_shared = n_dd * 12;
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x) {
        // The final column is reused for exact GPU FP64 reciprocal wavelengths.
        // Its original division is now evaluated once per observation/block,
        // before the existing barrier, instead of once per particle/pair.
        s_data[i] = i >= n_dd * 11 ? 1.0 / dd_data[i] : dd_data[i];
    }
    if constexpr (CachedSigma) {
        if (threadIdx.x == 0) {
            s_data[total_shared + 0] = 1.0 / (sigma_pr * sigma_pr);
            s_data[total_shared + 1] = 1.0 / (sigma_cycles * sigma_cycles);
        }
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    double x = px[tid];
    double y = py[tid];
    double z = pz[tid];

    double common_ref_range = 0.0;
    if constexpr (CommonReference) {
        double dx_ref = x - s_ref[0];
        double dy_ref = y - s_ref[1];
        double dz_ref = z - s_ref[2];
        common_ref_range = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
    }
    bool gate_pr = nlos_threshold_m > 0.0;
    bool gate_cp = nlos_threshold_cycles > 0.0;

    if (gate_pr || gate_cp) {
        int kept_pr = 0;
        int kept_cp = 0;
        for (int d = 0; d < n_dd; d++) {
            double dx_k = x - s_sat_k[d * 3 + 0];
            double dy_k = y - s_sat_k[d * 3 + 1];
            double dz_k = z - s_sat_k[d * 3 + 2];
            double dx_ref = x - s_ref[d * 3 + 0];
            double dy_ref = y - s_ref[d * 3 + 1];
            double dz_ref = z - s_ref[d * 3 + 2];
            double r_k = sqrt(dx_k * dx_k + dy_k * dy_k + dz_k * dz_k);
            double r_ref = CommonReference ? common_ref_range
                : sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
            double dd_range = r_k - r_ref - s_br_k[d] + s_br_ref[d];
            if (gate_pr && fabs(s_dd_pr[d] - dd_range) <= nlos_threshold_m) {
                kept_pr++;
            }
            double inv_wl = s_inv_wl[d];
            double cp_residual = s_dd_cp[d] - dd_range * inv_wl;
            double afv = cp_residual - rint(cp_residual);
            if (gate_cp && fabs(afv) <= nlos_threshold_cycles) {
                kept_cp++;
            }
        }
        int min_kept = (n_dd < 3) ? n_dd : 3;
        if (gate_pr && kept_pr < min_kept) gate_pr = false;
        if (gate_cp && kept_cp < min_kept) gate_cp = false;
    }

    double pr_log_w = 0.0;
    double cp_log_w = 0.0;
    double inv_pr_sigma2 = CachedSigma ? s_data[total_shared + 0] : 1.0 / (sigma_pr * sigma_pr);
    double inv_cp_sigma2 = CachedSigma ? s_data[total_shared + 1] : 1.0 / (sigma_cycles * sigma_cycles);
    bool huber_pr = huber_enabled && huber_pr_k > 0.0;
    bool huber_cp = huber_enabled && huber_carrier_k > 0.0;
    for (int d = 0; d < n_dd; d++) {
        double dx_k = x - s_sat_k[d * 3 + 0];
        double dy_k = y - s_sat_k[d * 3 + 1];
        double dz_k = z - s_sat_k[d * 3 + 2];
        double dx_ref = x - s_ref[d * 3 + 0];
        double dy_ref = y - s_ref[d * 3 + 1];
        double dz_ref = z - s_ref[d * 3 + 2];
        double r_k = sqrt(dx_k * dx_k + dy_k * dy_k + dz_k * dz_k);
        double r_ref = CommonReference ? common_ref_range
                : sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
        double dd_range = r_k - r_ref - s_br_k[d] + s_br_ref[d];
        double pr_residual = s_dd_pr[d] - dd_range;
        if (!gate_pr || fabs(pr_residual) <= nlos_threshold_m) {
            pr_log_w += huber_pr
                ? -pfd_weighted_huber_cost(pr_residual, sigma_pr, s_ws[d], huber_pr_k)
                : -0.5 * s_ws[d] * pr_residual * pr_residual * inv_pr_sigma2;
        }
        double inv_wl = s_inv_wl[d];
        double cp_residual = s_dd_cp[d] - dd_range * inv_wl;
        double afv = cp_residual - rint(cp_residual);
        if (!gate_cp || fabs(afv) <= nlos_threshold_cycles) {
            cp_log_w += huber_cp
                ? -pfd_weighted_huber_cost(afv, sigma_cycles, s_ws[d], huber_carrier_k)
                // The preceding joint kernel rounds this final multiply
                // before its CP accumulator add, including odd loop tails.
                // Keep that rounding when geometry reuse changes scheduling.
                : (CommonReference
                    ? __dmul_rn(-0.5 * s_ws[d] * afv * afv, inv_cp_sigma2)
                    : -0.5 * s_ws[d] * afv * afv * inv_cp_sigma2);
        }
    }
    log_weights[tid] = pr_log_w + cp_log_w;
}

// Gaussian gated DD: compute geometry once, accumulate both candidate sums
// in input order, then retain the original minimum-kept fallback independently.
__global__ void pfd_weight_dd_joint_gated_kernel(
    const double* px, const double* py, const double* pz, const double* dd_data,
    double* log_weights, int N, int n_dd, double sigma_pr, double sigma_cycles,
    double threshold_pr, double threshold_cp) {
    extern __shared__ double data[];
    const int size = n_dd * 12;
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        data[i] = i >= n_dd * 11 ? 1.0 / dd_data[i] : dd_data[i];
    if (threadIdx.x == 0) {
        data[size] = 1.0 / (sigma_pr * sigma_pr);
        data[size + 1] = 1.0 / (sigma_cycles * sigma_cycles);
    }
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    double x = px[tid], y = py[tid], z = pz[tid];
    double dx_ref = x - data[n_dd * 3];
    double dy_ref = y - data[n_dd * 3 + 1];
    double dz_ref = z - data[n_dd * 3 + 2];
    double r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
    double inv_pr = data[size], inv_cp = data[size + 1];
    double pr_all = 0.0, cp_all = 0.0, pr_kept = 0.0, cp_kept = 0.0;
    int count_pr = 0, count_cp = 0;
    bool gate_pr = threshold_pr > 0.0, gate_cp = threshold_cp > 0.0;
    for (int d = 0; d < n_dd; ++d) {
        double dx = x - data[d * 3];
        double dy = y - data[d * 3 + 1];
        double dz = z - data[d * 3 + 2];
        double range = sqrt(dx * dx + dy * dy + dz * dz) - r_ref
            - data[n_dd * 8 + d] + data[n_dd * 9 + d];
        double residual = data[n_dd * 6 + d] - range;
        double cp_residual = data[n_dd * 7 + d] - range * data[n_dd * 11 + d];
        double afv = cp_residual - rint(cp_residual);
        if (gate_pr && fabs(residual) <= threshold_pr) ++count_pr;
        if (gate_cp && fabs(afv) <= threshold_cp) ++count_cp;
        double pr_term = -0.5 * data[n_dd * 10 + d] * residual * residual;
        double cp_term = __dmul_rn(-0.5 * data[n_dd * 10 + d] * afv * afv, inv_cp);
        pr_all = __fma_rn(pr_term, inv_pr, pr_all);
        cp_all = __dadd_rn(cp_all, cp_term);
        // The joint likelihood keeps only <= threshold after a successful
        // gate. NaN residuals do not enter its filtered sum.
        if (gate_pr && fabs(residual) <= threshold_pr)
            pr_kept = __fma_rn(pr_term, inv_pr, pr_kept);
        if (gate_cp && fabs(afv) <= threshold_cp)
            cp_kept = __dadd_rn(cp_kept, cp_term);
    }
    int minimum = n_dd < 3 ? n_dd : 3;
    double pr = gate_pr && count_pr >= minimum ? pr_kept : pr_all;
    double cp = gate_cp && count_cp >= minimum ? cp_kept : cp_all;
    log_weights[tid] = __dadd_rn(pr, cp);
}

// Both-Huber gated DD retains the original normalized Huber cost expression
// and accumulates all/kept likelihoods in observation order in one geometry loop.
__global__ void pfd_weight_dd_joint_huber_gated_kernel(
    const double* px, const double* py, const double* pz, const double* dd_data,
    double* log_weights, int N, int n_dd, double sigma_pr, double sigma_cycles,
    double threshold_pr, double threshold_cp, double huber_pr_k, double huber_cp_k) {
    extern __shared__ double data[];
    const int size = n_dd * 12;
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        data[i] = i >= n_dd * 11 ? 1.0 / dd_data[i] : dd_data[i];
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    double x = px[tid], y = py[tid], z = pz[tid];
    double dx_ref = x - data[n_dd * 3];
    double dy_ref = y - data[n_dd * 3 + 1];
    double dz_ref = z - data[n_dd * 3 + 2];
    double r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
    double pr_all = 0.0, cp_all = 0.0, pr_kept = 0.0, cp_kept = 0.0;
    int count_pr = 0, count_cp = 0;
    bool gate_pr = threshold_pr > 0.0, gate_cp = threshold_cp > 0.0;
    for (int d = 0; d < n_dd; ++d) {
        double dx = x - data[d * 3];
        double dy = y - data[d * 3 + 1];
        double dz = z - data[d * 3 + 2];
        double range = sqrt(dx * dx + dy * dy + dz * dz) - r_ref
            - data[n_dd * 8 + d] + data[n_dd * 9 + d];
        double residual = data[n_dd * 6 + d] - range;
        double cp_residual = data[n_dd * 7 + d] - range * data[n_dd * 11 + d];
        double afv = cp_residual - rint(cp_residual);
        if (gate_pr && fabs(residual) <= threshold_pr) ++count_pr;
        if (gate_cp && fabs(afv) <= threshold_cp) ++count_cp;
        double weight = data[n_dd * 10 + d];
        double pr_rho = pfd_weighted_huber_cost(residual, sigma_pr, 1.0, huber_pr_k);
        double cp_rho = pfd_weighted_huber_cost(afv, sigma_cycles, 1.0, huber_cp_k);
        // The preceding joint PR contracts weight/cost into its accumulator,
        // while CP rounds the weighted cost before its accumulator addition.
        double cp_term = -__dmul_rn(weight, cp_rho);
        pr_all = __fma_rn(-weight, pr_rho, pr_all);
        cp_all = __dadd_rn(cp_all, cp_term);
        // The joint likelihood keeps only <= threshold after a successful
        // gate. NaN residuals do not enter its filtered sum.
        if (gate_pr && fabs(residual) <= threshold_pr)
            pr_kept = __fma_rn(-weight, pr_rho, pr_kept);
        if (gate_cp && fabs(afv) <= threshold_cp)
            cp_kept = __dadd_rn(cp_kept, cp_term);
    }
    int minimum = n_dd < 3 ? n_dd : 3;
    double pr = gate_pr && count_pr >= minimum ? pr_kept : pr_all;
    double cp = gate_cp && count_cp >= minimum ? cp_kept : cp_all;
    log_weights[tid] = __dadd_rn(pr, cp);
}

// Mixed joint gates retain per-pair ranges, independent filters and PR/CP rounding.
template<bool Huber>
__global__ void pfd_weight_dd_joint_mixed_gated_kernel(
    const double* px, const double* py, const double* pz, const double* dd_data,
    double* log_weights, int N, int n_dd, double sigma_pr, double sigma_cycles,
    double threshold_pr, double threshold_cp, double huber_pr_k, double huber_cp_k) {
    extern __shared__ double data[];
    const int size = n_dd * 12;
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        data[i] = i >= n_dd * 11 ? 1.0 / dd_data[i] : dd_data[i];
    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    double x = px[tid], y = py[tid], z = pz[tid];
    double inv_pr = 0.0, inv_cp = 0.0;
    if constexpr (!Huber) {
        inv_pr = 1.0 / (sigma_pr * sigma_pr);
        inv_cp = 1.0 / (sigma_cycles * sigma_cycles);
    }
    double pr_all = 0.0, cp_all = 0.0, pr_kept = 0.0, cp_kept = 0.0;
    int count_pr = 0, count_cp = 0;
    bool gate_pr = threshold_pr > 0.0, gate_cp = threshold_cp > 0.0;
    for (int d = 0; d < n_dd; ++d) {
        double dx = x - data[d * 3];
        double dy = y - data[d * 3 + 1];
        double dz = z - data[d * 3 + 2];
        double dx_ref = x - data[n_dd * 3 + d * 3];
        double dy_ref = y - data[n_dd * 3 + d * 3 + 1];
        double dz_ref = z - data[n_dd * 3 + d * 3 + 2];
        double r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);
        double range = sqrt(dx * dx + dy * dy + dz * dz) - r_ref
            - data[n_dd * 8 + d] + data[n_dd * 9 + d];
        double residual = data[n_dd * 6 + d] - range;
        double cp_residual = data[n_dd * 7 + d] - range * data[n_dd * 11 + d];
        double afv = cp_residual - rint(cp_residual);
        if (gate_pr && fabs(residual) <= threshold_pr) ++count_pr;
        if (gate_cp && fabs(afv) <= threshold_cp) ++count_cp;
        double weight = data[n_dd * 10 + d];
        if constexpr (Huber) {
            double pr_rho = pfd_weighted_huber_cost(residual, sigma_pr, 1.0, huber_pr_k);
            double cp_rho = pfd_weighted_huber_cost(afv, sigma_cycles, 1.0, huber_cp_k);
            double cp_term = -__dmul_rn(weight, cp_rho);
            pr_all = __fma_rn(-weight, pr_rho, pr_all);
            cp_all = __dadd_rn(cp_all, cp_term);
            if (gate_pr && fabs(residual) <= threshold_pr)
                pr_kept = __fma_rn(-weight, pr_rho, pr_kept);
            if (gate_cp && fabs(afv) <= threshold_cp)
                cp_kept = __dadd_rn(cp_kept, cp_term);
        } else {
            double pr_term = -0.5 * weight * residual * residual;
            double cp_term = __dmul_rn(-0.5 * weight * afv * afv, inv_cp);
            pr_all = __fma_rn(pr_term, inv_pr, pr_all);
            cp_all = __dadd_rn(cp_all, cp_term);
            if (gate_pr && fabs(residual) <= threshold_pr)
                pr_kept = __fma_rn(pr_term, inv_pr, pr_kept);
            if (gate_cp && fabs(afv) <= threshold_cp)
                cp_kept = __dadd_rn(cp_kept, cp_term);
        }
    }
    int minimum = n_dd < 3 ? n_dd : 3;
    double pr = gate_pr && count_pr >= minimum ? pr_kept : pr_all;
    double cp = gate_cp && count_cp >= minimum ? cp_kept : cp_all;
    log_weights[tid] = __dadd_rn(pr, cp);
}

// --- Doppler velocity-domain update kernel ---
// Applies a per-particle Doppler likelihood using each particle's velocity
// state, then optionally nudges velocity toward a per-particle WLS solution.
// RINEX/receiver Doppler convention: observed range-rate = -doppler * wavelength.
__global__ void pfd_weight_doppler_kernel(
    const double* px, const double* py, const double* pz,
    double* vx, double* vy, double* vz,
    const double* doppler_data,  // [sat_ecef: n*3][sat_vel: n*3][doppler_hz: n][weights: n]
    double* log_weights,
    int N, int n_sat,
    double wavelength_m,
    double sigma_mps,
    double velocity_update_gain,
    double max_velocity_update_mps) {

    extern __shared__ double s_data[];
    double* s_sat = s_data;                 // n_sat * 3
    double* s_sat_vel = s_data + n_sat * 3; // n_sat * 3
    double* s_doppler = s_data + n_sat * 6; // n_sat
    double* s_weights = s_data + n_sat * 7; // n_sat

    int total_shared = n_sat * 8;
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x) {
        s_data[i] = doppler_data[i];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double x = px[tid];
    double y = py[tid];
    double z = pz[tid];
    double vxi = vx[tid];
    double vyi = vy[tid];
    double vzi = vz[tid];

    double weighted_cd = 0.0;
    double weight_sum = 0.0;
    for (int s = 0; s < n_sat; s++) {
        double dx = s_sat[s * 3 + 0] - x;
        double dy = s_sat[s * 3 + 1] - y;
        double dz = s_sat[s * 3 + 2] - z;
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1.0) continue;
        double lx = dx / r;
        double ly = dy / r;
        double lz = dz / r;
        double obs = -s_doppler[s] * wavelength_m;
        double sv_radial =
            s_sat_vel[s * 3 + 0] * lx +
            s_sat_vel[s * 3 + 1] * ly +
            s_sat_vel[s * 3 + 2] * lz;
        double rx_radial = vxi * lx + vyi * ly + vzi * lz;
        double w = fmax(s_weights[s], 0.0);
        weighted_cd += w * (obs - sv_radial + rx_radial);
        weight_sum += w;
    }
    double cd = (weight_sum > 0.0) ? (weighted_cd / weight_sum) : 0.0;
    if (weight_sum <= 0.0) {
        return;
    }

    double inv_sigma2 = 1.0 / (sigma_mps * sigma_mps);
    double log_w = 0.0;
    for (int s = 0; s < n_sat; s++) {
        double dx = s_sat[s * 3 + 0] - x;
        double dy = s_sat[s * 3 + 1] - y;
        double dz = s_sat[s * 3 + 2] - z;
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1.0) continue;
        double lx = dx / r;
        double ly = dy / r;
        double lz = dz / r;
        double obs = -s_doppler[s] * wavelength_m;
        double pred =
            (s_sat_vel[s * 3 + 0] - vxi) * lx +
            (s_sat_vel[s * 3 + 1] - vyi) * ly +
            (s_sat_vel[s * 3 + 2] - vzi) * lz +
            cd;
        double residual = obs - pred;
        log_w += -0.5 * fmax(s_weights[s], 0.0) * residual * residual * inv_sigma2;
    }
    log_weights[tid] += log_w;

    double gain = fmin(1.0, fmax(0.0, velocity_update_gain));
    if (gain <= 0.0 || n_sat < 4) {
        return;
    }

    double normal[4][5];
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 5; col++) {
            normal[row][col] = 0.0;
        }
    }
    for (int s = 0; s < n_sat; s++) {
        double dx = s_sat[s * 3 + 0] - x;
        double dy = s_sat[s * 3 + 1] - y;
        double dz = s_sat[s * 3 + 2] - z;
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1.0) continue;
        double lx = dx / r;
        double ly = dy / r;
        double lz = dz / r;
        double h[4] = {-lx, -ly, -lz, 1.0};
        double sv_radial =
            s_sat_vel[s * 3 + 0] * lx +
            s_sat_vel[s * 3 + 1] * ly +
            s_sat_vel[s * 3 + 2] * lz;
        double rhs = -s_doppler[s] * wavelength_m - sv_radial;
        double w = fmax(s_weights[s], 0.0);
        for (int a = 0; a < 4; a++) {
            normal[a][4] += h[a] * w * rhs;
            for (int b = 0; b < 4; b++) {
                normal[a][b] += h[a] * w * h[b];
            }
        }
    }
    for (int d = 0; d < 4; d++) {
        normal[d][d] += 1e-9;
    }

    double solution[4] = {0.0, 0.0, 0.0, 0.0};
    if (!pfd_solve_4x4(normal, solution)) {
        return;
    }

    double dvx = solution[0] - vxi;
    double dvy = solution[1] - vyi;
    double dvz = solution[2] - vzi;
    double dv_norm = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
    if (max_velocity_update_mps > 0.0 && dv_norm > max_velocity_update_mps) {
        double scale = max_velocity_update_mps / fmax(dv_norm, 1e-9);
        dvx *= scale;
        dvy *= scale;
        dvz *= scale;
    }

    vx[tid] = vxi + gain * dvx;
    vy[tid] = vyi + gain * dvy;
    vz[tid] = vzi + gain * dvz;
}

// Bounded per-particle geometry keeps the original ordered Doppler/WLS sums.
// Refresh on each update; validity uses the original range cutoff, including NaN.
template<int CACHE_SATS>
__global__ void pfd_weight_doppler_cached_kernel(
    const double* px, const double* py, const double* pz,
    double* vx, double* vy, double* vz,
    const double* doppler_data,  // [sat_ecef: n*3][sat_vel: n*3][doppler_hz: n][weights: n]
    double* log_weights,
    int N, int n_sat,
    double wavelength_m,
    double sigma_mps,
    double velocity_update_gain,
    double max_velocity_update_mps) {

    extern __shared__ double s_data[];
    double* s_sat = s_data;                 // n_sat * 3
    double* s_sat_vel = s_data + n_sat * 3; // n_sat * 3
    double* s_doppler = s_data + n_sat * 6; // n_sat
    double* s_weights = s_data + n_sat * 7; // n_sat

    int total_shared = n_sat * 8;
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x) {
        s_data[i] = doppler_data[i];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double x = px[tid];
    double y = py[tid];
    double z = pz[tid];
    double vxi = vx[tid];
    double vyi = vy[tid];
    double vzi = vz[tid];

    double geometry[CACHE_SATS * 3];
    unsigned int valid_geometry = 0;

    double weighted_cd = 0.0;
    double weight_sum = 0.0;
    for (int s = 0; s < n_sat; s++) {
        double dx = s_sat[s * 3 + 0] - x;
        double dy = s_sat[s * 3 + 1] - y;
        double dz = s_sat[s * 3 + 2] - z;
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1.0) continue;
        double lx = dx / r;
        double ly = dy / r;
        double lz = dz / r;
        geometry[s * 3 + 0] = lx;
        geometry[s * 3 + 1] = ly;
        geometry[s * 3 + 2] = lz;
        valid_geometry |= 1u << s;
        double obs = -s_doppler[s] * wavelength_m;
        double sv_radial =
            s_sat_vel[s * 3 + 0] * lx +
            s_sat_vel[s * 3 + 1] * ly +
            s_sat_vel[s * 3 + 2] * lz;
        double rx_radial = vxi * lx + vyi * ly + vzi * lz;
        double w = fmax(s_weights[s], 0.0);
        weighted_cd += w * (obs - sv_radial + rx_radial);
        weight_sum += w;
    }
    double cd = (weight_sum > 0.0) ? (weighted_cd / weight_sum) : 0.0;
    if (weight_sum <= 0.0) {
        return;
    }

    double inv_sigma2 = 1.0 / (sigma_mps * sigma_mps);
    double log_w = 0.0;
    for (int s = 0; s < n_sat; s++) {
        if (!(valid_geometry & (1u << s))) continue;
        double lx = geometry[s * 3 + 0];
        double ly = geometry[s * 3 + 1];
        double lz = geometry[s * 3 + 2];
        double obs = -s_doppler[s] * wavelength_m;
        double pred =
            (s_sat_vel[s * 3 + 0] - vxi) * lx +
            (s_sat_vel[s * 3 + 1] - vyi) * ly +
            (s_sat_vel[s * 3 + 2] - vzi) * lz +
            cd;
        double residual = obs - pred;
        log_w += -0.5 * fmax(s_weights[s], 0.0) * residual * residual * inv_sigma2;
    }
    log_weights[tid] += log_w;

    double gain = fmin(1.0, fmax(0.0, velocity_update_gain));
    if (gain <= 0.0 || n_sat < 4) {
        return;
    }

    double normal[4][5];
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 5; col++) {
            normal[row][col] = 0.0;
        }
    }
    for (int s = 0; s < n_sat; s++) {
        if (!(valid_geometry & (1u << s))) continue;
        double lx = geometry[s * 3 + 0];
        double ly = geometry[s * 3 + 1];
        double lz = geometry[s * 3 + 2];
        double h[4] = {-lx, -ly, -lz, 1.0};
        double sv_radial =
            s_sat_vel[s * 3 + 0] * lx +
            s_sat_vel[s * 3 + 1] * ly +
            s_sat_vel[s * 3 + 2] * lz;
        double rhs = -s_doppler[s] * wavelength_m - sv_radial;
        double w = fmax(s_weights[s], 0.0);
        for (int a = 0; a < 4; a++) {
            normal[a][4] += h[a] * w * rhs;
            for (int b = 0; b < 4; b++) {
                normal[a][b] += h[a] * w * h[b];
            }
        }
    }
    for (int d = 0; d < 4; d++) {
        normal[d][d] += 1e-9;
    }

    double solution[4] = {0.0, 0.0, 0.0, 0.0};
    if (!pfd_solve_4x4(normal, solution)) {
        return;
    }

    double dvx = solution[0] - vxi;
    double dvy = solution[1] - vyi;
    double dvz = solution[2] - vzi;
    double dv_norm = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
    if (max_velocity_update_mps > 0.0 && dv_norm > max_velocity_update_mps) {
        double scale = max_velocity_update_mps / fmax(dv_norm, 1e-9);
        dvx *= scale;
        dvy *= scale;
        dvz *= scale;
    }

    vx[tid] = vxi + gain * dvx;
    vy[tid] = vyi + gain * dvy;
    vz[tid] = vzi + gain * dvz;
}

// Wider observation groups retain the same FP64 body and use all64 mask bits.
// The cache remains bounded; larger groups keep the uncached implementation.
template<int CACHE_SATS>
__global__ void pfd_weight_doppler_wide_cached_kernel(
    const double* px, const double* py, const double* pz,
    double* vx, double* vy, double* vz,
    const double* doppler_data,  // [sat_ecef: n*3][sat_vel: n*3][doppler_hz: n][weights: n]
    double* log_weights,
    int N, int n_sat,
    double wavelength_m,
    double sigma_mps,
    double velocity_update_gain,
    double max_velocity_update_mps) {

    extern __shared__ double s_data[];
    double* s_sat = s_data;                 // n_sat * 3
    double* s_sat_vel = s_data + n_sat * 3; // n_sat * 3
    double* s_doppler = s_data + n_sat * 6; // n_sat
    double* s_weights = s_data + n_sat * 7; // n_sat

    int total_shared = n_sat * 8;
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x) {
        s_data[i] = doppler_data[i];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double x = px[tid];
    double y = py[tid];
    double z = pz[tid];
    double vxi = vx[tid];
    double vyi = vy[tid];
    double vzi = vz[tid];

    double geometry[CACHE_SATS * 3];
    unsigned long long valid_geometry = 0;

    double weighted_cd = 0.0;
    double weight_sum = 0.0;
    for (int s = 0; s < n_sat; s++) {
        double dx = s_sat[s * 3 + 0] - x;
        double dy = s_sat[s * 3 + 1] - y;
        double dz = s_sat[s * 3 + 2] - z;
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1.0) continue;
        double lx = dx / r;
        double ly = dy / r;
        double lz = dz / r;
        geometry[s * 3 + 0] = lx;
        geometry[s * 3 + 1] = ly;
        geometry[s * 3 + 2] = lz;
        valid_geometry |= 1ULL << s;
        double obs = -s_doppler[s] * wavelength_m;
        double sv_radial =
            s_sat_vel[s * 3 + 0] * lx +
            s_sat_vel[s * 3 + 1] * ly +
            s_sat_vel[s * 3 + 2] * lz;
        double rx_radial = vxi * lx + vyi * ly + vzi * lz;
        double w = fmax(s_weights[s], 0.0);
        weighted_cd += w * (obs - sv_radial + rx_radial);
        weight_sum += w;
    }
    double cd = (weight_sum > 0.0) ? (weighted_cd / weight_sum) : 0.0;
    if (weight_sum <= 0.0) {
        return;
    }

    double inv_sigma2 = 1.0 / (sigma_mps * sigma_mps);
    double log_w = 0.0;
    for (int s = 0; s < n_sat; s++) {
        if (!(valid_geometry & (1ULL << s))) continue;
        double lx = geometry[s * 3 + 0];
        double ly = geometry[s * 3 + 1];
        double lz = geometry[s * 3 + 2];
        double obs = -s_doppler[s] * wavelength_m;
        double pred =
            (s_sat_vel[s * 3 + 0] - vxi) * lx +
            (s_sat_vel[s * 3 + 1] - vyi) * ly +
            (s_sat_vel[s * 3 + 2] - vzi) * lz +
            cd;
        double residual = obs - pred;
        log_w += -0.5 * fmax(s_weights[s], 0.0) * residual * residual * inv_sigma2;
    }
    log_weights[tid] += log_w;

    double gain = fmin(1.0, fmax(0.0, velocity_update_gain));
    if (gain <= 0.0 || n_sat < 4) {
        return;
    }

    double normal[4][5];
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 5; col++) {
            normal[row][col] = 0.0;
        }
    }
    for (int s = 0; s < n_sat; s++) {
        if (!(valid_geometry & (1ULL << s))) continue;
        double lx = geometry[s * 3 + 0];
        double ly = geometry[s * 3 + 1];
        double lz = geometry[s * 3 + 2];
        double h[4] = {-lx, -ly, -lz, 1.0};
        double sv_radial =
            s_sat_vel[s * 3 + 0] * lx +
            s_sat_vel[s * 3 + 1] * ly +
            s_sat_vel[s * 3 + 2] * lz;
        double rhs = -s_doppler[s] * wavelength_m - sv_radial;
        double w = fmax(s_weights[s], 0.0);
        for (int a = 0; a < 4; a++) {
            normal[a][4] += h[a] * w * rhs;
            for (int b = 0; b < 4; b++) {
                normal[a][b] += h[a] * w * h[b];
            }
        }
    }
    for (int d = 0; d < 4; d++) {
        normal[d][d] += 1e-9;
    }

    double solution[4] = {0.0, 0.0, 0.0, 0.0};
    if (!pfd_solve_4x4(normal, solution)) {
        return;
    }

    double dvx = solution[0] - vxi;
    double dvy = solution[1] - vyi;
    double dvz = solution[2] - vzi;
    double dv_norm = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
    if (max_velocity_update_mps > 0.0 && dv_norm > max_velocity_update_mps) {
        double scale = max_velocity_update_mps / fmax(dv_norm, 1e-9);
        dvx *= scale;
        dvy *= scale;
        dvz *= scale;
    }

    vx[tid] = vxi + gain * dvx;
    vy[tid] = vyi + gain * dvy;
    vz[tid] = vzi + gain * dvz;
}

template<int CACHE_SATS>
__global__ void pfd_doppler_kf_update_kernel(
    const double* px, const double* py, const double* pz,
    double* vx, double* vy, double* vz,
    double* vcov,
    const double* doppler_data,  // [sat_ecef: n*3][sat_vel: n*3][doppler_hz: n][weights: n]
    double* log_weights,
    int N, int n_sat,
    double wavelength_m,
    double sigma_mps) {

    extern __shared__ double s_data[];
    double* s_sat = s_data;                 // n_sat * 3
    double* s_sat_vel = s_data + n_sat * 3; // n_sat * 3
    double* s_doppler = s_data + n_sat * 6; // n_sat
    double* s_weights = s_data + n_sat * 7; // n_sat
    double* s_meas_var = s_data + n_sat * 8; // n_sat, particle-independent

    int total_shared = n_sat * 8;
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x) {
        s_data[i] = doppler_data[i];
    }
    double sigma2 = sigma_mps * sigma_mps;
    // Keep the original shared-memory requirement for unusually large sets.
    bool shared_variance = n_sat <= MAX_SATS;
    if (shared_variance) {
        for (int s = threadIdx.x; s < n_sat; s += blockDim.x) {
            double w = fmax(doppler_data[n_sat * 7 + s], 0.0);
            s_meas_var[s] = w > 0.0 ? sigma2 / w : 0.0;
        }
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double x = px[tid];
    double y = py[tid];
    double z = pz[tid];
    double mui[3] = {vx[tid], vy[tid], vz[tid]};

    int cov_off = tid * 9;
    double P[9];
    for (int k = 0; k < 9; k++) {
        P[k] = vcov[cov_off + k];
    }

    double h_mean[3] = {0.0, 0.0, 0.0};
    double y_mean = 0.0;
    double weight_sum = 0.0;
    // Geometry is unchanged while the second pass updates velocity/covariance.
    // Small observation sets reuse the exact first-pass FP64 values. Dynamic
    // indexing keeps this bounded cache in local memory instead of forcing a
    // large unrolled update/register footprint for every satellite count.
    double geometry[CACHE_SATS > 0 ? CACHE_SATS * 4 : 1];
    unsigned int valid_geometry = 0;
    for (int s = 0; s < n_sat; s++) {
        double w = fmax(s_weights[s], 0.0);
        if (w <= 0.0) continue;
        double dx = s_sat[s * 3 + 0] - x;
        double dy = s_sat[s * 3 + 1] - y;
        double dz = s_sat[s * 3 + 2] - z;
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1.0) continue;
        double lx = dx / r;
        double ly = dy / r;
        double lz = dz / r;
        double h0 = -lx;
        double h1 = -ly;
        double h2 = -lz;
        double sv_radial =
            s_sat_vel[s * 3 + 0] * lx +
            s_sat_vel[s * 3 + 1] * ly +
            s_sat_vel[s * 3 + 2] * lz;
        double obs = -s_doppler[s] * wavelength_m;
        double y_obs = obs - sv_radial;
        if constexpr (CACHE_SATS > 0) {
            geometry[s * 4 + 0] = lx;
            geometry[s * 4 + 1] = ly;
            geometry[s * 4 + 2] = lz;
            geometry[s * 4 + 3] = y_obs;
            valid_geometry |= 1u << s;
        }
        h_mean[0] += w * h0;
        h_mean[1] += w * h1;
        h_mean[2] += w * h2;
        y_mean += w * y_obs;
        weight_sum += w;
    }
    if (weight_sum <= 0.0) {
        return;
    }
    h_mean[0] /= weight_sum;
    h_mean[1] /= weight_sum;
    h_mean[2] /= weight_sum;
    y_mean /= weight_sum;

    double log_w = 0.0;
    for (int s = 0; s < n_sat; s++) {
        double obs_weight = fmax(s_weights[s], 0.0);
        if (obs_weight <= 0.0) continue;
        double lx, ly, lz, y_obs;
        if constexpr (CACHE_SATS > 0) {
            if (!(valid_geometry & (1u << s))) continue;
            lx = geometry[s * 4 + 0];
            ly = geometry[s * 4 + 1];
            lz = geometry[s * 4 + 2];
            y_obs = geometry[s * 4 + 3];
        } else {
            double dx = s_sat[s * 3 + 0] - x;
            double dy = s_sat[s * 3 + 1] - y;
            double dz = s_sat[s * 3 + 2] - z;
            double r = sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1.0) continue;
            lx = dx / r;
            ly = dy / r;
            lz = dz / r;
            double sv_radial =
                s_sat_vel[s * 3 + 0] * lx +
                s_sat_vel[s * 3 + 1] * ly +
                s_sat_vel[s * 3 + 2] * lz;
            y_obs = -s_doppler[s] * wavelength_m - sv_radial;
        }
        double h[3] = {-lx - h_mean[0], -ly - h_mean[1], -lz - h_mean[2]};
        double y_centered = y_obs - y_mean;
        double ph0 = P[0] * h[0] + P[1] * h[1] + P[2] * h[2];
        double ph1 = P[3] * h[0] + P[4] * h[1] + P[5] * h[2];
        double ph2 = P[6] * h[0] + P[7] * h[1] + P[8] * h[2];
        double meas_var = shared_variance ? s_meas_var[s] : sigma2 / obs_weight;
        double S = h[0] * ph0 + h[1] * ph1 + h[2] * ph2 + meas_var;
        if (!(S > 1e-18) || !isfinite(S)) {
            continue;
        }
        double pred = h[0] * mui[0] + h[1] * mui[1] + h[2] * mui[2];
        double innov = y_centered - pred;
        double K[3] = {ph0 / S, ph1 / S, ph2 / S};
        mui[0] += K[0] * innov;
        mui[1] += K[1] * innov;
        mui[2] += K[2] * innov;

        double hp0 = h[0] * P[0] + h[1] * P[3] + h[2] * P[6];
        double hp1 = h[0] * P[1] + h[1] * P[4] + h[2] * P[7];
        double hp2 = h[0] * P[2] + h[1] * P[5] + h[2] * P[8];
        P[0] -= K[0] * hp0; P[1] -= K[0] * hp1; P[2] -= K[0] * hp2;
        P[3] -= K[1] * hp0; P[4] -= K[1] * hp1; P[5] -= K[1] * hp2;
        P[6] -= K[2] * hp0; P[7] -= K[2] * hp1; P[8] -= K[2] * hp2;

        double p01 = 0.5 * (P[1] + P[3]);
        double p02 = 0.5 * (P[2] + P[6]);
        double p12 = 0.5 * (P[5] + P[7]);
        P[1] = P[3] = isfinite(p01) ? p01 : 0.0;
        P[2] = P[6] = isfinite(p02) ? p02 : 0.0;
        P[5] = P[7] = isfinite(p12) ? p12 : 0.0;
        P[0] = (isfinite(P[0]) && P[0] > 1e-12) ? P[0] : 1e-12;
        P[4] = (isfinite(P[4]) && P[4] > 1e-12) ? P[4] : 1e-12;
        P[8] = (isfinite(P[8]) && P[8] > 1e-12) ? P[8] : 1e-12;

        log_w += -0.5 * (innov * innov / S + log(S));
    }

    vx[tid] = mui[0];
    vy[tid] = mui[1];
    vz[tid] = mui[2];
    for (int k = 0; k < 9; k++) {
        vcov[cov_off + k] = P[k];
    }
    log_weights[tid] += log_w;
}

template<int CACHE_SATS>
__global__ void pfd_doppler_kf_wide_update_kernel(
    const double* px, const double* py, const double* pz,
    double* vx, double* vy, double* vz,
    double* vcov,
    const double* doppler_data,  // [sat_ecef: n*3][sat_vel: n*3][doppler_hz: n][weights: n]
    double* log_weights,
    int N, int n_sat,
    double wavelength_m,
    double sigma_mps) {

    extern __shared__ double s_data[];
    double* s_sat = s_data;                 // n_sat * 3
    double* s_sat_vel = s_data + n_sat * 3; // n_sat * 3
    double* s_doppler = s_data + n_sat * 6; // n_sat
    double* s_weights = s_data + n_sat * 7; // n_sat
    double* s_meas_var = s_data + n_sat * 8; // n_sat, particle-independent

    int total_shared = n_sat * 8;
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x) {
        s_data[i] = doppler_data[i];
    }
    double sigma2 = sigma_mps * sigma_mps;
    // Keep the original shared-memory requirement for unusually large sets.
    bool shared_variance = n_sat <= MAX_SATS;
    if (shared_variance) {
        for (int s = threadIdx.x; s < n_sat; s += blockDim.x) {
            double w = fmax(doppler_data[n_sat * 7 + s], 0.0);
            s_meas_var[s] = w > 0.0 ? sigma2 / w : 0.0;
        }
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double x = px[tid];
    double y = py[tid];
    double z = pz[tid];
    double mui[3] = {vx[tid], vy[tid], vz[tid]};

    int cov_off = tid * 9;
    double P[9];
    for (int k = 0; k < 9; k++) {
        P[k] = vcov[cov_off + k];
    }

    double h_mean[3] = {0.0, 0.0, 0.0};
    double y_mean = 0.0;
    double weight_sum = 0.0;
    // Geometry is unchanged while the second pass updates velocity/covariance.
    // Small observation sets reuse the exact first-pass FP64 values. Dynamic
    // indexing keeps this bounded cache in local memory instead of forcing a
    // large unrolled update/register footprint for every satellite count.
    double geometry[CACHE_SATS > 0 ? CACHE_SATS * 4 : 1];
    unsigned long long valid_geometry = 0;
    for (int s = 0; s < n_sat; s++) {
        double w = fmax(s_weights[s], 0.0);
        if (w <= 0.0) continue;
        double dx = s_sat[s * 3 + 0] - x;
        double dy = s_sat[s * 3 + 1] - y;
        double dz = s_sat[s * 3 + 2] - z;
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1.0) continue;
        double lx = dx / r;
        double ly = dy / r;
        double lz = dz / r;
        double h0 = -lx;
        double h1 = -ly;
        double h2 = -lz;
        double sv_radial =
            s_sat_vel[s * 3 + 0] * lx +
            s_sat_vel[s * 3 + 1] * ly +
            s_sat_vel[s * 3 + 2] * lz;
        double obs = -s_doppler[s] * wavelength_m;
        double y_obs = obs - sv_radial;
        if constexpr (CACHE_SATS > 0) {
            geometry[s * 4 + 0] = lx;
            geometry[s * 4 + 1] = ly;
            geometry[s * 4 + 2] = lz;
            geometry[s * 4 + 3] = y_obs;
            valid_geometry |= 1ULL << s;
        }
        h_mean[0] += w * h0;
        h_mean[1] += w * h1;
        h_mean[2] += w * h2;
        y_mean += w * y_obs;
        weight_sum += w;
    }
    if (weight_sum <= 0.0) {
        return;
    }
    h_mean[0] /= weight_sum;
    h_mean[1] /= weight_sum;
    h_mean[2] /= weight_sum;
    y_mean /= weight_sum;

    double log_w = 0.0;
    for (int s = 0; s < n_sat; s++) {
        double obs_weight = fmax(s_weights[s], 0.0);
        if (obs_weight <= 0.0) continue;
        double lx, ly, lz, y_obs;
        if constexpr (CACHE_SATS > 0) {
            if (!(valid_geometry & (1ULL << s))) continue;
            lx = geometry[s * 4 + 0];
            ly = geometry[s * 4 + 1];
            lz = geometry[s * 4 + 2];
            y_obs = geometry[s * 4 + 3];
        } else {
            double dx = s_sat[s * 3 + 0] - x;
            double dy = s_sat[s * 3 + 1] - y;
            double dz = s_sat[s * 3 + 2] - z;
            double r = sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1.0) continue;
            lx = dx / r;
            ly = dy / r;
            lz = dz / r;
            double sv_radial =
                s_sat_vel[s * 3 + 0] * lx +
                s_sat_vel[s * 3 + 1] * ly +
                s_sat_vel[s * 3 + 2] * lz;
            y_obs = -s_doppler[s] * wavelength_m - sv_radial;
        }
        double h[3] = {-lx - h_mean[0], -ly - h_mean[1], -lz - h_mean[2]};
        double y_centered = y_obs - y_mean;
        double ph0 = P[0] * h[0] + P[1] * h[1] + P[2] * h[2];
        double ph1 = P[3] * h[0] + P[4] * h[1] + P[5] * h[2];
        double ph2 = P[6] * h[0] + P[7] * h[1] + P[8] * h[2];
        double meas_var = shared_variance ? s_meas_var[s] : sigma2 / obs_weight;
        double S = h[0] * ph0 + h[1] * ph1 + h[2] * ph2 + meas_var;
        if (!(S > 1e-18) || !isfinite(S)) {
            continue;
        }
        double pred = h[0] * mui[0] + h[1] * mui[1] + h[2] * mui[2];
        double innov = y_centered - pred;
        double K[3] = {ph0 / S, ph1 / S, ph2 / S};
        mui[0] += K[0] * innov;
        mui[1] += K[1] * innov;
        mui[2] += K[2] * innov;

        double hp0 = h[0] * P[0] + h[1] * P[3] + h[2] * P[6];
        double hp1 = h[0] * P[1] + h[1] * P[4] + h[2] * P[7];
        double hp2 = h[0] * P[2] + h[1] * P[5] + h[2] * P[8];
        P[0] -= K[0] * hp0; P[1] -= K[0] * hp1; P[2] -= K[0] * hp2;
        P[3] -= K[1] * hp0; P[4] -= K[1] * hp1; P[5] -= K[1] * hp2;
        P[6] -= K[2] * hp0; P[7] -= K[2] * hp1; P[8] -= K[2] * hp2;

        double p01 = 0.5 * (P[1] + P[3]);
        double p02 = 0.5 * (P[2] + P[6]);
        double p12 = 0.5 * (P[5] + P[7]);
        P[1] = P[3] = isfinite(p01) ? p01 : 0.0;
        P[2] = P[6] = isfinite(p02) ? p02 : 0.0;
        P[5] = P[7] = isfinite(p12) ? p12 : 0.0;
        P[0] = (isfinite(P[0]) && P[0] > 1e-12) ? P[0] : 1e-12;
        P[4] = (isfinite(P[4]) && P[4] > 1e-12) ? P[4] : 1e-12;
        P[8] = (isfinite(P[8]) && P[8] > 1e-12) ? P[8] : 1e-12;

        log_w += -0.5 * (innov * innov / S + log(S));
    }

    vx[tid] = mui[0];
    vy[tid] = mui[1];
    vz[tid] = mui[2];
    for (int k = 0; k < 9; k++) {
        vcov[cov_off + k] = P[k];
    }
    log_weights[tid] += log_w;
}

// --- Position-domain update kernel ---
// Applies a Gaussian likelihood based on distance to a reference position.
// log_w += -0.5 * ||particle_pos - ref_pos||^2 / sigma^2
__global__ void pfd_position_update_kernel(
    const double* px, const double* py, const double* pz,
    double* log_weights,
    double ref_x, double ref_y, double ref_z,
    double inv_sigma2, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    double dx = px[tid] - ref_x;
    double dy = py[tid] - ref_y;
    double dz = pz[tid] - ref_z;
    log_weights[tid] += -0.5 * (dx*dx + dy*dy + dz*dz) * inv_sigma2;
}

// --- Clock bias shift kernel ---
// Shifts all particles' clock bias by a constant offset.
// Used to re-center cb around an external estimate each epoch.
__global__ void pfd_shift_clock_bias_kernel(
    double* pcb, double shift, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    pcb[tid] += shift;
}

__global__ void pfd_shift_position_kernel(
    double* px, double* py, double* pz,
    double dx, double dy, double dz,
    int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    px[tid] += dx;
    py[tid] += dy;
    pz[tid] += dz;
}

// --- Position reset kernel ---
// Redraws positions around an external reference and resets weights. The RNG
// stream is offset from predict's so the same (seed, step) does not reuse its
// noise.
__global__ void pfd_reset_position_kernel(
    double* px, double* py, double* pz,
    double* log_weights,
    double ref_x, double ref_y, double ref_z, double sigma_pos,
    int N, unsigned long long seed, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed ^ 0x9E3779B97F4A7C15ULL, tid, step, &state);
    px[tid] = ref_x + curand_normal_double(&state) * sigma_pos;
    py[tid] = ref_y + curand_normal_double(&state) * sigma_pos;
    pz[tid] = ref_z + curand_normal_double(&state) * sigma_pos;
    log_weights[tid] = 0.0;
}

// Retain the original 256-thread tree: shared-memory strides 128/64/32,
// followed by warp strides 16/8/4/2/1. Only the storage and synchronization of
// the last five stages change; operand order and FP64 arithmetic stay the same.
template <int COLUMNS, bool MAXIMUM = false>
__device__ __forceinline__ void pfd_reduce_query_block(double* columns) {
    const int tid = threadIdx.x;
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            for (int col = 0; col < COLUMNS; ++col) {
                double* values = columns + col * blockDim.x;
                values[tid] = MAXIMUM
                    ? fmax(values[tid], values[tid + stride])
                    : values[tid] + values[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid < 32) {
        double values[COLUMNS];
        for (int col = 0; col < COLUMNS; ++col)
            values[col] = columns[col * blockDim.x + tid];
        for (int stride = 16; stride > 0; stride >>= 1) {
            for (int col = 0; col < COLUMNS; ++col) {
                const double other = __shfl_down_sync(0xffffffffu, values[col], stride);
                if (tid < stride)
                    values[col] = MAXIMUM ? fmax(values[col], other) : values[col] + other;
            }
        }
        if (tid == 0) {
            for (int col = 0; col < COLUMNS; ++col)
                columns[col * blockDim.x] = values[col];
        }
    }
    // All threads consume the block maximum in the next phase. The sum phase
    // only needs thread 0, but this common barrier keeps the helper reusable.
    __syncthreads();
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

    pfd_reduce_query_block<1, true>(s_max);
    double block_max = s_max[0];

    double w_shifted = (gid < N) ? exp(lw - block_max) : 0.0;
    s_sum[tid] = w_shifted;
    s_sum2[tid] = w_shifted * w_shifted;
    __syncthreads();

    pfd_reduce_query_block<2>(sdata);

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

    pfd_reduce_query_block<1, true>(s_max);
    double block_max = s_max[0];

    double w = (gid < N) ? exp(lw - block_max) : 0.0;
    s_wx[tid] = (gid < N) ? w * px[gid] : 0.0;
    s_wy[tid] = (gid < N) ? w * py[gid] : 0.0;
    s_wz[tid] = (gid < N) ? w * pz[gid] : 0.0;
    s_wcb[tid] = (gid < N) ? w * pcb[gid] : 0.0;
    s_sw[tid] = w;
    __syncthreads();

    pfd_reduce_query_block<5>(sdata);

    if (tid == 0) {
        partial_results[blockIdx.x * 4 + 0] = s_wx[0];
        partial_results[blockIdx.x * 4 + 1] = s_wy[0];
        partial_results[blockIdx.x * 4 + 2] = s_wz[0];
        partial_results[blockIdx.x * 4 + 3] = s_wcb[0];
        partial_sum_w[blockIdx.x] = s_sw[0];
        partial_max_lw[blockIdx.x] = block_max;
    }
}

__global__ void pfd_position_spread_kernel(const double* px, const double* py,
                                           const double* pz,
                                           const double* log_weights,
                                           double center_x, double center_y, double center_z,
                                           double* partial_sum_wd2,
                                           double* partial_sum_w,
                                           double* partial_max_lw,
                                           int N) {
    extern __shared__ double sdata[];
    double* s_wd2 = sdata;
    double* s_sw = sdata + blockDim.x;
    double* s_max = sdata + 2 * blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double lw = (gid < N) ? log_weights[gid] : -INFINITY;
    s_max[tid] = lw;
    __syncthreads();

    pfd_reduce_query_block<1, true>(s_max);
    double block_max = s_max[0];

    double w = (gid < N) ? exp(lw - block_max) : 0.0;
    double wd2 = 0.0;
    if (gid < N) {
        double dx = px[gid] - center_x;
        double dy = py[gid] - center_y;
        double dz = pz[gid] - center_z;
        wd2 = w * (dx * dx + dy * dy + dz * dz);
    }
    s_wd2[tid] = wd2;
    s_sw[tid] = w;
    __syncthreads();

    pfd_reduce_query_block<2>(sdata);

    if (tid == 0) {
        partial_sum_wd2[blockIdx.x] = s_wd2[0];
        partial_sum_w[blockIdx.x] = s_sw[0];
        partial_max_lw[blockIdx.x] = block_max;
    }
}

__global__ void pfd_finalize_ess_kernel(const double* partial_sum_w,
                                        const double* partial_sum_w2,
                                        const double* partial_max_lw,
                                        double* result, int n_partial) {
    extern __shared__ double sdata[];
    double* s_max = sdata;
    double* s_sum = sdata + blockDim.x;
    double* s_sum2 = sdata + 2 * blockDim.x;
    const int tid = threadIdx.x;

    double local_max = -INFINITY;
    for (int i = tid; i < n_partial; i += blockDim.x) {
        local_max = fmax(local_max, partial_max_lw[i]);
    }
    s_max[tid] = local_max;
    __syncthreads();
    pfd_reduce_query_block<1, true>(s_max);
    const double global_max = s_max[0];

    double local_sum = 0.0;
    double local_sum2 = 0.0;
    for (int i = tid; i < n_partial; i += blockDim.x) {
        const double correction = exp(partial_max_lw[i] - global_max);
        local_sum += partial_sum_w[i] * correction;
        local_sum2 += partial_sum_w2[i] * correction * correction;
    }
    s_sum[tid] = local_sum;
    s_sum2[tid] = local_sum2;
    __syncthreads();
    pfd_reduce_query_block<2>(sdata + blockDim.x);
    if (tid == 0) {
        result[0] = (s_sum[0] * s_sum[0]) / s_sum2[0];
    }
}

__global__ void pfd_finalize_spread_kernel(const double* partial_sum_wd2,
                                           const double* partial_sum_w,
                                           const double* partial_max_lw,
                                           double* result, int n_partial) {
    extern __shared__ double sdata[];
    double* s_max = sdata;
    double* s_wd2 = sdata + blockDim.x;
    double* s_sum = sdata + 2 * blockDim.x;
    const int tid = threadIdx.x;

    double local_max = -INFINITY;
    for (int i = tid; i < n_partial; i += blockDim.x) {
        local_max = fmax(local_max, partial_max_lw[i]);
    }
    s_max[tid] = local_max;
    __syncthreads();
    pfd_reduce_query_block<1, true>(s_max);
    const double global_max = s_max[0];

    double local_wd2 = 0.0;
    double local_sum = 0.0;
    for (int i = tid; i < n_partial; i += blockDim.x) {
        const double correction = exp(partial_max_lw[i] - global_max);
        local_wd2 += partial_sum_wd2[i] * correction;
        local_sum += partial_sum_w[i] * correction;
    }
    s_wd2[tid] = local_wd2;
    s_sum[tid] = local_sum;
    __syncthreads();
    pfd_reduce_query_block<2>(sdata + blockDim.x);
    if (tid == 0) {
        result[0] = (s_sum[0] > 0.0 && isfinite(s_wd2[0]))
                        ? sqrt(fmax(s_wd2[0] / s_sum[0], 0.0))
                        : 0.0;
    }
}

__global__ void pfd_finalize_estimate_kernel(const double* partial_results,
                                             const double* partial_sum_w,
                                             const double* partial_max_lw,
                                             double* result, int n_partial) {
    extern __shared__ double sdata[];
    double* s_max = sdata;
    double* s_x = sdata + blockDim.x;
    double* s_y = sdata + 2 * blockDim.x;
    double* s_z = sdata + 3 * blockDim.x;
    double* s_cb = sdata + 4 * blockDim.x;
    double* s_w = sdata + 5 * blockDim.x;
    const int tid = threadIdx.x;

    double local_max = -INFINITY;
    for (int i = tid; i < n_partial; i += blockDim.x) {
        local_max = fmax(local_max, partial_max_lw[i]);
    }
    s_max[tid] = local_max;
    __syncthreads();
    pfd_reduce_query_block<1, true>(s_max);
    const double global_max = s_max[0];

    double x = 0.0, y = 0.0, z = 0.0, cb = 0.0, w = 0.0;
    for (int i = tid; i < n_partial; i += blockDim.x) {
        const double correction = exp(partial_max_lw[i] - global_max);
        x += partial_results[i * 4 + 0] * correction;
        y += partial_results[i * 4 + 1] * correction;
        z += partial_results[i * 4 + 2] * correction;
        cb += partial_results[i * 4 + 3] * correction;
        w += partial_sum_w[i] * correction;
    }
    s_x[tid] = x;
    s_y[tid] = y;
    s_z[tid] = z;
    s_cb[tid] = cb;
    s_w[tid] = w;
    __syncthreads();
    pfd_reduce_query_block<5>(sdata + blockDim.x);
    if (tid == 0) {
        result[0] = s_x[0] / s_w[0];
        result[1] = s_y[0] / s_w[0];
        result[2] = s_z[0] / s_w[0];
        result[3] = s_cb[0] / s_w[0];
    }
}

// Inline kernel parameters own the nine coefficients for this launch.
// Every block copies them before its tail threads may leave the barrier.
struct PFDPositionMixtureInput {
    double values[16];  // Up to four [ref_x, ref_y, ref_z, log_mix] records.
};

template<int COMPONENTS>
__global__ void pfd_position_mixture_components_kernel(
    const double* px, const double* py, const double* pz,
    double* output, int* nonfinite, PFDPositionMixtureInput input,
    double sigma_squared, int N) {
    __shared__ double coefficients[16];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) coefficients[i] = input.values[i];
    }
    __syncthreads();
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)N * COMPONENTS) return;
    int particle = (int)(index / COMPONENTS);
    int component = (int)(index % COMPONENTS);
    double x = px[particle], y = py[particle], z = pz[particle];
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
        atomicExch(nonfinite, 1);
        return;
    }
    int offset = component * 4;
    // Match NumPy's individual FP64 nodes, including the initial +0 sum.
    // Explicit rounding intrinsics prevent multiplication/addition contraction.
    double dx = __dsub_rn(x, coefficients[offset]);
    double dy = __dsub_rn(y, coefficients[offset + 1]);
    double dz = __dsub_rn(z, coefficients[offset + 2]);
    double distance = __dadd_rn(0.0, __dmul_rn(dx, dx));
    distance = __dadd_rn(distance, __dmul_rn(dy, dy));
    distance = __dadd_rn(distance, __dmul_rn(dz, dz));
    double penalty = __ddiv_rn(__dmul_rn(0.5, distance), sigma_squared);
    output[index] = __dsub_rn(coefficients[offset + 3], penalty);
}

struct PFDUniformVelocityCovariance {
    double values[9];
};

__global__ void pfd_set_velocity_covariance_kernel(
    double* covariance, PFDUniformVelocityCovariance input, int N) {
    __shared__ double coefficients[9];
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int k = 0; k < 9; ++k) coefficients[k] = input.values[k];
    }
    __syncthreads();
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < (size_t)N * 9) covariance[index] = coefficients[index % 9];
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

__global__ void pfd_get_particle_states_kernel(
    const double* px, const double* py, const double* pz,
    const double* vx, const double* vy, const double* vz,
    const double* vcov,
    const double* pcb, double* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int out_off = tid * 16;
    int cov_off = tid * 9;
    output[out_off + 0] = px[tid];
    output[out_off + 1] = py[tid];
    output[out_off + 2] = pz[tid];
    output[out_off + 3] = pcb[tid];
    output[out_off + 4] = vx[tid];
    output[out_off + 5] = vy[tid];
    output[out_off + 6] = vz[tid];
    for (int k = 0; k < 9; k++) {
        output[out_off + 7 + k] = vcov[cov_off + k];
    }
}

__global__ void pfd_set_particle_states_kernel(
    double* px, double* py, double* pz,
    double* vx, double* vy, double* vz,
    double* vcov,
    double* pcb,
    const double* input,
    int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int in_off = tid * 16;
    int cov_off = tid * 9;
    px[tid] = input[in_off + 0];
    py[tid] = input[in_off + 1];
    pz[tid] = input[in_off + 2];
    pcb[tid] = input[in_off + 3];
    vx[tid] = input[in_off + 4];
    vy[tid] = input[in_off + 5];
    vz[tid] = input[in_off + 6];
    for (int k = 0; k < 9; k++) {
        vcov[cov_off + k] = input[in_off + 7 + k];
    }
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

// Store the shifted weights once and sum them with the preceding reduction
// tree. Keeping the double store and the same additions preserves the values
// consumed by normalization/CDF, while removing a launch and an N-element read.
__global__ void pfd_exp_sum_kernel(const double* log_weights, double* weights,
                                   const double* max_lw, double* block_sums, int N) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    double w = (gid < N) ? exp(log_weights[gid] - max_lw[0]) : 0.0;
    if (gid < N) weights[gid] = w;
    sdata[tid] = w;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) block_sums[blockIdx.x] = sdata[0];
}

__global__ void pfd_normalize_kernel(double* weights, const double* sum_w, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    weights[tid] = weights[tid] / sum_w[0];
}

__global__ void pfd_finalize_max_kernel(const double* partial, double* result,
                                        int n_partial) {
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    double value = -INFINITY;
    for (int i = tid; i < n_partial; i += blockDim.x) {
        value = fmax(value, partial[i]);
    }
    sdata[tid] = value;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) result[0] = sdata[0];
}

__global__ void pfd_finalize_sum_kernel(const double* partial, double* result,
                                        int n_partial) {
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    double value = 0.0;
    for (int i = tid; i < n_partial; i += blockDim.x) value += partial[i];
    sdata[tid] = value;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) result[0] = sdata[0];
}

__global__ void pfd_systematic_resample_kernel(const double* cdf,
                                               const double* px_in, const double* py_in,
                                               const double* pz_in,
                                               const double* vx_in, const double* vy_in,
                                               const double* vz_in,
                                               const double* vcov_in,
                                               const double* pcb_in,
                                               double* px_out, double* py_out,
                                               double* pz_out,
                                               double* vx_out, double* vy_out,
                                               double* vz_out,
                                               double* vcov_out,
                                               double* pcb_out,
                                               int* ancestor_out,
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
    vx_out[tid] = vx_in[lo];
    vy_out[tid] = vy_in[lo];
    vz_out[tid] = vz_in[lo];
    int dst_cov = tid * 9;
    int src_cov = lo * 9;
    for (int k = 0; k < 9; k++) {
        vcov_out[dst_cov + k] = vcov_in[src_cov + k];
    }
    pcb_out[tid] = pcb_in[lo];
    if (ancestor_out != nullptr) {
        ancestor_out[tid] = lo;
    }
}

// --- Reset log weights to zero ---
__global__ void pfd_reset_weights_kernel(double* log_weights, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    log_weights[tid] = 0.0;
}

// Copy the caller's FP64 scalar bits to every log-weight without arithmetic.
__global__ void pfd_fill_weights_kernel(double* log_weights, int N, double value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    log_weights[tid] = value;
}

// --- Megopolis kernels ---
// Each iteration lets slot tid take over the content of slot j with an
// acceptance ratio that depends only on the slots' (fixed) log-weights and the
// per-(tid, iteration) random draws, never on the particle content. So the
// iterations can run on an ancestor-index array, followed by one gather of the
// 16 doubles per particle. This is bit-identical to copying the full state on
// every iteration, but moves 4-byte indices instead of 128-byte states.
__global__ void pfd_megopolis_index_init_kernel(int* idx, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    idx[tid] = tid;
}

__global__ void pfd_megopolis_index_kernel(const int* idx_src, int* idx_dst,
                                          const double* log_weights,
                                          int N, unsigned long long seed, int iteration) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, iteration, &state);

    int offset = (int)(curand_uniform_double(&state) * (N - 1)) + 1;
    int j = (tid + offset) % N;

    double log_alpha = log_weights[j] - log_weights[tid];
    double alpha = fmin(1.0, exp(log_alpha));
    double u = curand_uniform_double(&state);

    idx_dst[tid] = (u < alpha) ? idx_src[j] : idx_src[tid];
}

// Megopolis resampling as in Chesser et al. (2021), opt-in: each particle
// keeps a current ancestor k and accepts a proposal j with probability
// min(1, w_j / w_k). All threads in a block share the per-iteration offset, so
// the proposal reads log_weights[(i + offset) % N] are coalesced. The marginal
// proposal for every particle stays uniform. All B iterations run in one
// launch with k held in a register.
__global__ void pfd_megopolis_coalesced_kernel(const double* log_weights, int* ancestors,
                                              int N, unsigned long long seed,
                                              int n_iterations) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Block-shared offset stream (subsequence N + block) and per-particle
    // acceptance stream (subsequence i) are disjoint Philox subsequences.
    curandStatePhilox4_32_10_t block_rng;
    curand_init(seed, (unsigned long long)N + blockIdx.x, 0, &block_rng);
    curandStatePhilox4_32_10_t own_rng;
    curand_init(seed, (unsigned long long)i, 0, &own_rng);

    // Draws and the acceptance test use single precision: consumer GPUs run
    // FP64 at 1/32 rate, and float resolution is ample for u <= w_j / w_k.
    int k = i;
    double lw_k = log_weights[i];
    int b = 0;
    int full_iterations = n_iterations > 0 ? (n_iterations / 4) * 4 : 0;
    for (; b < full_iterations; b += 4) {
        uint4 proposals = curand4(&block_rng);
        float4 uniforms = curand_uniform4(&own_rng);
        int offsets[4] = {
            (int)(proposals.x % (unsigned int)N),
            (int)(proposals.y % (unsigned int)N),
            (int)(proposals.z % (unsigned int)N),
            (int)(proposals.w % (unsigned int)N)};
        float draws[4] = {uniforms.x, uniforms.y, uniforms.z, uniforms.w};
        int indices[4];
        double proposal_weights[4];
        // Proposal addresses do not depend on the accepted ancestor. Fetch
        // four ahead, then retain the original sequential acceptance order.
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int wrap = N - offsets[t];
            indices[t] = (i >= wrap) ? i - wrap : i + offsets[t];
            proposal_weights[t] = log_weights[indices[t]];
        }
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            if (__logf(draws[t]) <= (float)(proposal_weights[t] - lw_k)) {
                k = indices[t];
                lw_k = proposal_weights[t];
            }
        }
    }
    for (; b < n_iterations; b++) {
        int offset = (int)(curand(&block_rng) % (unsigned int)N);
        int wrap = N - offset;
        int j = (i >= wrap) ? i - wrap : i + offset;
        double lw_j = log_weights[j];
        float u = curand_uniform(&own_rng);
        if (__logf(u) <= (float)(lw_j - lw_k)) {
            k = j;
            lw_k = lw_j;
        }
    }
    ancestors[i] = k;
}

__global__ void pfd_pool_cdf_indices_kernel(const double* cdf, int source_count,
                                             int* indices, int target_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= target_count) return;
    // Match NumPy's separate FP64 addition and division, and side="left".
    double threshold = __ddiv_rn(__dadd_rn((double)tid, 0.5), (double)target_count);
    int low = 0, high = source_count;
    while (low < high) {
        int middle = low + (high - low) / 2;
        if (cdf[middle] < threshold) low = middle + 1;
        else high = middle;
    }
    indices[tid] = low < source_count ? low : source_count - 1;
}

__global__ void pfd_megopolis_gather_kernel(const double* src_px, const double* src_py,
                                           const double* src_pz,
                                           const double* src_vx, const double* src_vy,
                                           const double* src_vz,
                                           const double* src_vcov, const double* src_pcb,
                                           double* dst_px, double* dst_py, double* dst_pz,
                                           double* dst_vx, double* dst_vy, double* dst_vz,
                                           double* dst_vcov, double* dst_pcb,
                                           const int* idx, int N, bool copy_covariance) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int k = idx[tid];
        dst_px[tid] = src_px[k];
        dst_py[tid] = src_py[k];
        dst_pz[tid] = src_pz[k];
        dst_vx[tid] = src_vx[k];
        dst_vy[tid] = src_vy[k];
        dst_vz[tid] = src_vz[k];
        dst_pcb[tid] = src_pcb[k];
    }
    if (!copy_covariance) return;

    // Covariance is packed as nine doubles per particle. Assign consecutive
    // components to consecutive lanes so stores are contiguous and neighboring
    // lanes read the same ancestor's covariance, rather than nine strided
    // warp accesses to unrelated ancestors. Tail threads still copy components
    // belonging to valid particles in their block; no early return is allowed.
    int first = blockIdx.x * blockDim.x;
    int count = min((int)blockDim.x, N - first);
    for (int local = threadIdx.x; local < count * 9; local += blockDim.x) {
        int particle = first + local / 9;
        int component = local % 9;
        dst_vcov[(size_t)first * 9 + local] =
            src_vcov[(size_t)idx[particle] * 9 + component];
    }
}

// Copy covariance before reusing its former source for packed state rows.
// The component mapping and lane ordering match the original full gather.
__global__ void pfd_gather_resample_covariance_kernel(
    const double* src, double* dst, const int* indices, int N) {
    int first = blockIdx.x * blockDim.x;
    int count = min((int)blockDim.x, N - first);
    for (int local = threadIdx.x; local < count * 9; local += blockDim.x) {
        int particle = first + local / 9;
        int component = local % 9;
        dst[(size_t)first * 9 + local] =
            src[(size_t)indices[particle] * 9 + component];
    }
}

// The unused destination covariance has9*N doubles when covariance is
// invariant. Use8*N of them as temporary aligned rows of the seven changing
// state components; no allocation or persistent representation is added.
__global__ void pfd_pack_common_resample_kernel(
    const double* px, const double* py, const double* pz,
    const double* vx, const double* vy, const double* vz, const double* pcb,
    double* packed, int N) {
    // Pad each shared column to avoid transpose bank conflicts.
    __shared__ double tile[8 * (BLOCK_SIZE + 1)];
    int lane = threadIdx.x;
    int first = blockIdx.x * blockDim.x;
    int tid = first + lane;
    if (tid < N) {
        tile[0 * (BLOCK_SIZE + 1) + lane] = px[tid];
        tile[1 * (BLOCK_SIZE + 1) + lane] = py[tid];
        tile[2 * (BLOCK_SIZE + 1) + lane] = pz[tid];
        tile[3 * (BLOCK_SIZE + 1) + lane] = vx[tid];
        tile[4 * (BLOCK_SIZE + 1) + lane] = vy[tid];
        tile[5 * (BLOCK_SIZE + 1) + lane] = vz[tid];
        tile[6 * (BLOCK_SIZE + 1) + lane] = pcb[tid];
        tile[7 * (BLOCK_SIZE + 1) + lane] = 0.0;
    }
    __syncthreads();
    int count = min((int)blockDim.x, N - first);
    for (int local = lane; local < count * 8; local += blockDim.x) {
        packed[(size_t)first * 8 + local] =
            tile[(local % 8) * (BLOCK_SIZE + 1) + local / 8];
    }
}

__global__ void pfd_gather_common_packed_kernel(
    const double* packed, const int* indices,
    double* px, double* py, double* pz,
    double* vx, double* vy, double* vz, double* pcb,
    double* log_weights, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    const double2* rows = reinterpret_cast<const double2*>(packed);
    size_t row = (size_t)indices[tid] * 4;
    double2 xy = rows[row + 0];
    double2 zvx = rows[row + 1];
    double2 vyvz = rows[row + 2];
    double2 cbpad = rows[row + 3];
    px[tid] = xy.x;
    py[tid] = xy.y;
    pz[tid] = zvx.x;
    vx[tid] = zvx.y;
    vy[tid] = vyvz.x;
    vz[tid] = vyvz.y;
    pcb[tid] = cbpad.x;
    // Ancestor selection finished earlier on this stream; it can no longer
    // read these weights. Preserve the original positive-zero reset.
    log_weights[tid] = 0.0;
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
    state->covariance_uniform = false;
    state->ess_cached = false;

    size_t sz = (size_t)n_particles * sizeof(double);
    int grid = state->grid_size;

    // Main particle arrays
    CUDA_CHECK(cudaMalloc(&state->d_px, sz));
    CUDA_CHECK(cudaMalloc(&state->d_py, sz));
    CUDA_CHECK(cudaMalloc(&state->d_pz, sz));
    CUDA_CHECK(cudaMalloc(&state->d_vx, sz));
    CUDA_CHECK(cudaMalloc(&state->d_vy, sz));
    CUDA_CHECK(cudaMalloc(&state->d_vz, sz));
    CUDA_CHECK(cudaMalloc(&state->d_vcov, (size_t)n_particles * 9 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_pcb, sz));
    CUDA_CHECK(cudaMalloc(&state->d_log_weights, sz));

    // Double-buffer for resampling
    CUDA_CHECK(cudaMalloc(&state->d_px_tmp, sz));
    CUDA_CHECK(cudaMalloc(&state->d_py_tmp, sz));
    CUDA_CHECK(cudaMalloc(&state->d_pz_tmp, sz));
    CUDA_CHECK(cudaMalloc(&state->d_vx_tmp, sz));
    CUDA_CHECK(cudaMalloc(&state->d_vy_tmp, sz));
    CUDA_CHECK(cudaMalloc(&state->d_vz_tmp, sz));
    CUDA_CHECK(cudaMalloc(&state->d_vcov_tmp, (size_t)n_particles * 9 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_pcb_tmp, sz));

    // Reduction temporaries
    // partial_a: max(grid * 4, grid) doubles - used for estimate (grid*4) or ESS (grid)
    CUDA_CHECK(cudaMalloc(&state->d_partial_a, grid * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_partial_b, grid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_partial_c, grid * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&state->d_reduction_result, 6 * sizeof(double)));

    // Systematic resampling buffers
    CUDA_CHECK(cudaMalloc(&state->d_weights_norm, sz));
    CUDA_CHECK(cudaMalloc(&state->d_cdf, sz));
    CUDA_CHECK(cudaMalloc(&state->d_resample_ancestor, (size_t)n_particles * sizeof(int)));

    // Megopolis resampling index buffers
    CUDA_CHECK(cudaMalloc(&state->d_megopolis_idx_a, (size_t)n_particles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state->d_megopolis_idx_b, (size_t)n_particles * sizeof(int)));

    // Velocity buffer (3 doubles)
    CUDA_CHECK(cudaMalloc(&state->d_vel, 3 * sizeof(double)));

    // CUDA stream for pipelined execution
    CUDA_CHECK(cudaStreamCreate(&state->stream));

    // Persistent device buffers for satellite data (pre-allocate for MAX_SATS)
    state->pinned_capacity = MAX_SATS;
    CUDA_CHECK(cudaMalloc(&state->d_sat_ecef, MAX_SATS * 5 * sizeof(double)));
    state->d_pseudoranges = state->d_sat_ecef + MAX_SATS * 3;
    state->d_weights_sat = state->d_sat_ecef + MAX_SATS * 4;
    state->obs_scratch_capacity = MAX_SATS * 11;
    CUDA_CHECK(cudaMalloc(
        &state->d_obs_scratch,
        (size_t)state->obs_scratch_capacity * sizeof(double)));

    // Pinned host memory for async transfers
    CUDA_CHECK(cudaMallocHost(&state->h_sat_pinned, MAX_SATS * 5 * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&state->h_result_pinned, 4 * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&state->h_reduction_pinned, grid * 6 * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(
        &state->h_obs_scratch,
        (size_t)state->obs_scratch_capacity * sizeof(double)));

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
    cudaFree(state->d_vx);
    cudaFree(state->d_vy);
    cudaFree(state->d_vz);
    cudaFree(state->d_vcov);
    cudaFree(state->d_pcb);
    cudaFree(state->d_log_weights);
    cudaFree(state->d_px_tmp);
    cudaFree(state->d_py_tmp);
    cudaFree(state->d_pz_tmp);
    cudaFree(state->d_vx_tmp);
    cudaFree(state->d_vy_tmp);
    cudaFree(state->d_vz_tmp);
    cudaFree(state->d_vcov_tmp);
    cudaFree(state->d_pcb_tmp);
    cudaFree(state->d_partial_a);
    cudaFree(state->d_partial_b);
    cudaFree(state->d_partial_c);
    cudaFree(state->d_reduction_result);
    cudaFree(state->d_weights_norm);
    cudaFree(state->d_cdf);
    cudaFree(state->d_resample_ancestor);
    cudaFree(state->d_megopolis_idx_a);
    cudaFree(state->d_megopolis_idx_b);
    cudaFree(state->d_vel);

    // Free persistent satellite device buffers
    // Observation/weight pointers alias the packed satellite allocation.
    cudaFree(state->d_sat_ecef);
    cudaFree(state->d_obs_scratch);

    // Free pinned host memory
    cudaFreeHost(state->h_sat_pinned);
    cudaFreeHost(state->h_result_pinned);
    cudaFreeHost(state->h_reduction_pinned);
    cudaFreeHost(state->h_obs_scratch);

    // Null out all pointers to prevent use-after-free
    state->d_px = nullptr;
    state->d_py = nullptr;
    state->d_pz = nullptr;
    state->d_vx = nullptr;
    state->d_vy = nullptr;
    state->d_vz = nullptr;
    state->d_vcov = nullptr;
    state->d_pcb = nullptr;
    state->d_log_weights = nullptr;
    state->d_obs_scratch = nullptr;
    state->h_obs_scratch = nullptr;
    state->d_px_tmp = nullptr;
    state->d_py_tmp = nullptr;
    state->d_pz_tmp = nullptr;
    state->d_vx_tmp = nullptr;
    state->d_vy_tmp = nullptr;
    state->d_vz_tmp = nullptr;
    state->d_vcov_tmp = nullptr;
    state->d_pcb_tmp = nullptr;
    state->d_partial_a = nullptr;
    state->d_partial_b = nullptr;
    state->d_partial_c = nullptr;
    state->d_weights_norm = nullptr;
    state->d_cdf = nullptr;
    state->d_resample_ancestor = nullptr;
    state->d_megopolis_idx_a = nullptr;
    state->d_megopolis_idx_b = nullptr;
    state->d_reduction_result = nullptr;
    state->d_vel = nullptr;
    state->d_sat_ecef = nullptr;
    state->d_pseudoranges = nullptr;
    state->d_weights_sat = nullptr;
    state->h_sat_pinned = nullptr;
    state->h_result_pinned = nullptr;
    state->h_reduction_pinned = nullptr;

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
    unsigned long long seed,
    double init_vx, double init_vy, double init_vz,
    double spread_vel,
    double init_vel_sigma) {
    state->ess_cached = false;

    int N = state->n_particles;
    int grid = state->grid_size;

    pfd_init_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz, state->d_vcov, state->d_pcb,
        state->d_log_weights,
        init_x, init_y, init_z, init_cb,
        init_vx, init_vy, init_vz,
        spread_pos, spread_cb, spread_vel, init_vel_sigma,
        N, seed);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    state->covariance_uniform = true;
}

// ============================================================
// Predict
// ============================================================

void pf_device_predict(PFDeviceState* state,
    double vx, double vy, double vz,
    double dt, double sigma_pos, double sigma_cb,
    unsigned long long seed, int step,
    double sigma_vel,
    double velocity_guide_alpha,
    bool velocity_kf,
    double velocity_process_noise) {

    int N = state->n_particles;
    int grid = state->grid_size;

    // Each launch captures the guide by value. This removes the separate H2D
    // copy and pinned-result-buffer reuse while earlier predicts are pending.
    if (velocity_kf) {
        if (state->covariance_uniform) {
            pfd_predict_common_kernel<true, false><<<grid, BLOCK_SIZE, 0, state->stream>>>(
                state->d_px, state->d_py, state->d_pz,
                state->d_vx, state->d_vy, state->d_vz, state->d_vcov, state->d_pcb,
                vx, vy, vz, dt, sigma_pos, sigma_cb, sigma_vel, velocity_guide_alpha,
                velocity_process_noise, N, seed, step);
        } else {
            pfd_predict_kernel<true, false><<<grid, BLOCK_SIZE, 0, state->stream>>>(
                state->d_px, state->d_py, state->d_pz,
                state->d_vx, state->d_vy, state->d_vz, state->d_vcov, state->d_pcb,
                vx, vy, vz, dt, sigma_pos, sigma_cb, sigma_vel, velocity_guide_alpha,
                velocity_process_noise, N, seed, step);
        }
    } else if (sigma_vel > 0.0) {
        pfd_predict_kernel<false, true><<<grid, BLOCK_SIZE, 0, state->stream>>>(
            state->d_px, state->d_py, state->d_pz,
            state->d_vx, state->d_vy, state->d_vz, state->d_vcov, state->d_pcb,
            vx, vy, vz, dt, sigma_pos, sigma_cb, sigma_vel, velocity_guide_alpha,
            velocity_process_noise, N, seed, step);
    } else {
        pfd_predict_kernel<false, false><<<grid, BLOCK_SIZE, 0, state->stream>>>(
            state->d_px, state->d_py, state->d_pz,
            state->d_vx, state->d_vy, state->d_vz, state->d_vcov, state->d_pcb,
            vx, vy, vz, dt, sigma_pos, sigma_cb, sigma_vel, velocity_guide_alpha,
            velocity_process_noise, N, seed, step);
    }
    CUDA_CHECK_LAST();
}

// ============================================================
// Weight
// ============================================================

static void ensure_obs_scratch(PFDeviceState* state, int required_doubles) {
    // The host buffer is pinned and may still be the source of an async copy.
    // Synchronizing here makes it safe to overwrite while allowing unrelated
    // kernels to remain asynchronous until another packed-observation update.
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    if (required_doubles <= state->obs_scratch_capacity) return;

    cudaFree(state->d_obs_scratch);
    cudaFreeHost(state->h_obs_scratch);
    CUDA_CHECK(cudaMalloc(
        &state->d_obs_scratch, (size_t)required_doubles * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(
        &state->h_obs_scratch, (size_t)required_doubles * sizeof(double)));
    state->obs_scratch_capacity = required_doubles;
}

// PR, GMM and undifferenced carrier updates share the same five-column
// layout. Keep one owning allocation and refresh the current-length aliases.
static void pfd_stage_satellite_observations(PFDeviceState* state,
    const double* sat_ecef, const double* observations, const double* weights,
    int n_sat) {
    // Both the preceding kernel and pinned-source H2D must finish before the
    // shared buffers are overwritten or grown, exactly as for the old staging.
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    if (n_sat > state->pinned_capacity) {
        CUDA_CHECK(cudaFreeHost(state->h_sat_pinned));
        state->h_sat_pinned = nullptr;
        CUDA_CHECK(cudaFree(state->d_sat_ecef));
        state->d_sat_ecef = nullptr;
        state->d_pseudoranges = nullptr;
        state->d_weights_sat = nullptr;
        CUDA_CHECK(cudaMallocHost(&state->h_sat_pinned, (size_t)n_sat * 5 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&state->d_sat_ecef, (size_t)n_sat * 5 * sizeof(double)));
        state->pinned_capacity = n_sat;
    }
    size_t coordinates = (size_t)n_sat * 3;
    size_t values = (size_t)n_sat;
    memcpy(state->h_sat_pinned, sat_ecef, coordinates * sizeof(double));
    memcpy(state->h_sat_pinned + coordinates, observations, values * sizeof(double));
    memcpy(state->h_sat_pinned + coordinates + values, weights, values * sizeof(double));
    state->d_pseudoranges = state->d_sat_ecef + coordinates;
    state->d_weights_sat = state->d_pseudoranges + values;
    CUDA_CHECK(cudaMemcpyAsync(state->d_sat_ecef, state->h_sat_pinned,
        (coordinates + 2 * values) * sizeof(double), cudaMemcpyHostToDevice, state->stream));
}

void pf_device_weight(PFDeviceState* state,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    int n_sat, double sigma_pr, double nu,
    double per_particle_nlos_threshold_m,
    bool per_particle_huber,
    double per_particle_huber_k) {
    state->ess_cached = false;

    int N = state->n_particles;
    int grid = state->grid_size;

    pfd_stage_satellite_observations(state, sat_ecef, pseudoranges, weights_sat, n_sat);

    // Launch weight kernel on the same stream (waits for async copies to complete)
    // Dynamic shared memory: n_sat*3 (sat_ecef) + n_sat (pr) + n_sat (ws) = n_sat*5 doubles
    size_t smem_weight = (size_t)n_sat * 5 * sizeof(double);
    bool gaussian_only = !(nu > 0.0) && !(per_particle_nlos_threshold_m > 0.0) &&
        !(per_particle_huber && per_particle_huber_k > 0.0);
    if (nu > 0.0 && !(per_particle_nlos_threshold_m > 0.0)) {
        pfd_weight_student_kernel<<<grid, BLOCK_SIZE, smem_weight, state->stream>>>(
            state->d_px, state->d_py, state->d_pz, state->d_pcb,
            state->d_sat_ecef, state->d_pseudoranges, state->d_weights_sat,
            state->d_log_weights, N, n_sat, sigma_pr, nu);
    } else if (gaussian_only) {
        pfd_weight_gaussian_kernel<<<grid, BLOCK_SIZE, smem_weight, state->stream>>>(
            state->d_px, state->d_py, state->d_pz, state->d_pcb,
            state->d_sat_ecef, state->d_pseudoranges, state->d_weights_sat,
            state->d_log_weights, N, n_sat, sigma_pr);
    } else if (per_particle_nlos_threshold_m > 0.0) {
        if (nu > 0.0) {
            pfd_weight_pr_gate_kernel<2><<<grid, BLOCK_SIZE, smem_weight, state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_pcb,
                state->d_sat_ecef, state->d_pseudoranges, state->d_weights_sat,
                state->d_log_weights, N, n_sat, sigma_pr, nu,
                per_particle_nlos_threshold_m, per_particle_huber_k);
        } else if (per_particle_huber && per_particle_huber_k > 0.0) {
            pfd_weight_pr_gate_kernel<1><<<grid, BLOCK_SIZE, smem_weight, state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_pcb,
                state->d_sat_ecef, state->d_pseudoranges, state->d_weights_sat,
                state->d_log_weights, N, n_sat, sigma_pr, nu,
                per_particle_nlos_threshold_m, per_particle_huber_k);
        } else {
            pfd_weight_pr_gate_kernel<0><<<grid, BLOCK_SIZE, smem_weight, state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_pcb,
                state->d_sat_ecef, state->d_pseudoranges, state->d_weights_sat,
                state->d_log_weights, N, n_sat, sigma_pr, nu,
                per_particle_nlos_threshold_m, per_particle_huber_k);
        }
    } else {
        pfd_weight_kernel<<<grid, BLOCK_SIZE, smem_weight, state->stream>>>(
            state->d_px, state->d_py, state->d_pz, state->d_pcb,
            state->d_sat_ecef, state->d_pseudoranges, state->d_weights_sat,
            state->d_log_weights,
            N, n_sat, sigma_pr, nu, per_particle_nlos_threshold_m,
            per_particle_huber, per_particle_huber_k);
    }
    CUDA_CHECK_LAST();
}

// Detect the common-reference case from this call's host input. Bitwise
// comparison preserves signed-zero distinctions; nonfinite coordinates and
// single pairs use the unchanged mixed-reference specialization.
static bool pfd_has_common_dd_reference(const double* ref_ecef, int n_dd) {
    if (n_dd < 2) return false;
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(ref_ecef[axis])) return false;
    }
    for (int d = 1; d < n_dd; ++d) {
        if (memcmp(ref_ecef, ref_ecef + d * 3, 3 * sizeof(double)) != 0) return false;
    }
    return true;
}

// Exceptional inputs retain the original mixed-reference kernels.
static bool pfd_can_fuse_mixed_dd_gate(const double* packed, int n_dd, int columns,
    double threshold, bool huber, double k) {
    if (n_dd < 2 || !(threshold > 0.0) || (huber && (!(k > 0.0) || !std::isfinite(k)))) return false;
    for (int i = 0; i < n_dd * columns; ++i) {
        if (!std::isfinite(packed[i])) return false;
        if (columns == 11 && i >= n_dd * 10 && packed[i] < DBL_MIN) return false;
    }
    return true;
}

void pf_device_weight_dd_pseudorange(PFDeviceState* state,
    const double* sat_ecef_k, const double* ref_ecef,
    const double* dd_pseudorange, const double* base_range_k,
    const double* base_range_ref, const double* weights_dd,
    int n_dd, double sigma_pr,
    double per_particle_nlos_threshold_m,
    bool per_particle_huber,
    double per_particle_huber_k) {
    state->ess_cached = false;

    int N = state->n_particles;
    int grid = state->grid_size;

    // Pack all DD data into a contiguous buffer for a single H2D transfer.
    // Layout: [sat_k: n_dd*3][ref: n_dd*3][dd_pr: n_dd][base_range_k: n_dd][base_range_ref: n_dd][weights: n_dd]
    int total_doubles = n_dd * 10;
    size_t total_bytes = (size_t)total_doubles * sizeof(double);

    ensure_obs_scratch(state, total_doubles);
    double* h_buf = state->h_obs_scratch;

    int off = 0;
    memcpy(h_buf + off, sat_ecef_k, n_dd * 3 * sizeof(double)); off += n_dd * 3;
    memcpy(h_buf + off, ref_ecef, n_dd * 3 * sizeof(double)); off += n_dd * 3;
    memcpy(h_buf + off, dd_pseudorange, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, base_range_k, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, base_range_ref, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, weights_dd, n_dd * sizeof(double)); off += n_dd;

    CUDA_CHECK(cudaMemcpyAsync(
        state->d_obs_scratch, h_buf, total_bytes,
        cudaMemcpyHostToDevice, state->stream));

    size_t smem = (size_t)total_doubles * sizeof(double);
    if (pfd_has_common_dd_reference(ref_ecef, n_dd)) {
        bool fused_huber = per_particle_huber && per_particle_huber_k > 0.0 &&
            std::isfinite(per_particle_huber_k);
        bool fused_gate = (!per_particle_huber || fused_huber) && per_particle_nlos_threshold_m > 0.0;
        if (fused_gate) {
            for (int i = 0; i < total_doubles; ++i) {
                if (!std::isfinite(h_buf[i])) { fused_gate = false; break; }
            }
        }
        if (fused_gate && fused_huber) {
            pfd_weight_dd_single_huber_gated_kernel<false><<<grid, BLOCK_SIZE, smem, state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_pr, per_particle_nlos_threshold_m, per_particle_huber_k);
        } else if (fused_gate) {
            pfd_weight_dd_single_gated_kernel<false><<<grid, BLOCK_SIZE, smem + sizeof(double), state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_pr, per_particle_nlos_threshold_m);
        } else {
            pfd_weight_dd_pseudorange_kernel<true, true><<<grid, BLOCK_SIZE, smem + sizeof(double), state->stream>>>(
                state->d_px, state->d_py, state->d_pz,
                state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_pr, per_particle_nlos_threshold_m,
                per_particle_huber, per_particle_huber_k);
        }
    } else if (pfd_can_fuse_mixed_dd_gate(h_buf, n_dd, 10, per_particle_nlos_threshold_m,
        per_particle_huber, per_particle_huber_k)) {
        if (per_particle_huber) {
            pfd_weight_dd_mixed_gated_kernel<false, true><<<grid, BLOCK_SIZE, smem, state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_pr, per_particle_nlos_threshold_m, per_particle_huber_k);
        } else {
            pfd_weight_dd_mixed_gated_kernel<false, false><<<grid, BLOCK_SIZE, smem, state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_pr, per_particle_nlos_threshold_m, per_particle_huber_k);
        }
    } else {
        pfd_weight_dd_pseudorange_kernel<false><<<grid, BLOCK_SIZE, smem, state->stream>>>(
            state->d_px, state->d_py, state->d_pz,
            state->d_obs_scratch, state->d_log_weights,
            N, n_dd, sigma_pr, per_particle_nlos_threshold_m,
            per_particle_huber, per_particle_huber_k);
    }
    CUDA_CHECK_LAST();

}

void pf_device_weight_gmm(PFDeviceState* state,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    int n_sat, double sigma_pr,
    double w_los, double mu_nlos, double sigma_nlos) {
    state->ess_cached = false;

    int N = state->n_particles;
    int grid = state->grid_size;

    pfd_stage_satellite_observations(state, sat_ecef, pseudoranges, weights_sat, n_sat);

    // Launch GMM weight kernel on the same stream
    size_t smem_weight = (size_t)n_sat * 5 * sizeof(double);
    bool shared_coefficients = n_sat > 0 && n_sat <= 1227 &&
        std::isfinite(sigma_pr) && sigma_pr >= DBL_MIN &&
        std::isfinite(sigma_nlos) && sigma_nlos >= DBL_MIN &&
        std::isfinite(w_los) && w_los > 0.0 && w_los < 1.0 && std::isfinite(mu_nlos);
    if (shared_coefficients) {
        pfd_weight_gmm_shared_kernel<<<grid, BLOCK_SIZE, smem_weight + 6 * sizeof(double), state->stream>>>(
            state->d_px, state->d_py, state->d_pz, state->d_pcb,
            state->d_sat_ecef, state->d_pseudoranges, state->d_weights_sat,
            state->d_log_weights, N, n_sat, sigma_pr, w_los, mu_nlos, sigma_nlos);
    } else {
        pfd_weight_gmm_kernel<<<grid, BLOCK_SIZE, smem_weight, state->stream>>>(
            state->d_px, state->d_py, state->d_pz, state->d_pcb,
            state->d_sat_ecef, state->d_pseudoranges, state->d_weights_sat,
            state->d_log_weights, N, n_sat, sigma_pr, w_los, mu_nlos, sigma_nlos);
    }
    CUDA_CHECK_LAST();
}

// ============================================================
// Position-domain update (soft constraint from external position)
// ============================================================

void pf_device_position_update(PFDeviceState* state,
    double ref_x, double ref_y, double ref_z, double sigma_pos) {
    state->ess_cached = false;
    int N = state->n_particles;
    int grid = state->grid_size;
    double inv_sigma2 = 1.0 / (sigma_pos * sigma_pos);

    pfd_position_update_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_log_weights,
        ref_x, ref_y, ref_z, inv_sigma2, N);
    CUDA_CHECK_LAST();
}

// ============================================================
// Clock bias shift
// ============================================================

void pf_device_shift_clock_bias(PFDeviceState* state, double shift) {
    int N = state->n_particles;
    int grid = state->grid_size;

    pfd_shift_clock_bias_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_pcb, shift, N);
    CUDA_CHECK_LAST();
}

void pf_device_shift_position(PFDeviceState* state, double dx, double dy, double dz) {
    int N = state->n_particles;
    int grid = state->grid_size;

    pfd_shift_position_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        dx, dy, dz, N);
    CUDA_CHECK_LAST();
}

void pf_device_reset_position(PFDeviceState* state,
    double ref_x, double ref_y, double ref_z, double sigma_pos,
    unsigned long long seed, int step) {
    state->ess_cached = false;
    int N = state->n_particles;
    int grid = state->grid_size;

    pfd_reset_position_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_log_weights,
        ref_x, ref_y, ref_z, sigma_pos, N, seed, step);
    CUDA_CHECK_LAST();
}

// ============================================================
// ESS
// ============================================================

double pf_device_ess(const PFDeviceState* state) {
    if (state->ess_cached) {
        // Preserve the completion/error boundary even after weight-preserving
        // asynchronous operations such as predict or position shifts.
        CUDA_CHECK(cudaStreamSynchronize(state->stream));
        return state->cached_ess;
    }
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

    pfd_finalize_ess_kernel<<<1, BLOCK_SIZE, 3 * BLOCK_SIZE * sizeof(double),
                              state->stream>>>(
        state->d_partial_a, state->d_partial_b, state->d_partial_c,
        state->d_reduction_result, grid);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpyAsync(
        state->h_result_pinned, state->d_reduction_result, sizeof(double),
        cudaMemcpyDeviceToHost, state->stream));
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    // This separate host slot survives estimate/spread reuse of result buffers.
    state->cached_ess = state->h_result_pinned[0];
    state->ess_cached = true;
    return state->cached_ess;
}

double pf_device_position_spread(
    const PFDeviceState* state,
    double center_x, double center_y, double center_z) {
    int N = state->n_particles;
    int grid = state->grid_size;

    size_t smem = 3 * BLOCK_SIZE * sizeof(double);
    pfd_position_spread_kernel<<<grid, BLOCK_SIZE, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_log_weights,
        center_x, center_y, center_z,
        state->d_partial_a,
        state->d_partial_b,
        state->d_partial_c,
        N);
    CUDA_CHECK_LAST();

    pfd_finalize_spread_kernel<<<1, BLOCK_SIZE, 3 * BLOCK_SIZE * sizeof(double),
                                 state->stream>>>(
        state->d_partial_a, state->d_partial_b, state->d_partial_c,
        state->d_reduction_result, grid);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpyAsync(
        state->h_result_pinned, state->d_reduction_result, sizeof(double),
        cudaMemcpyDeviceToHost, state->stream));
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    return state->h_result_pinned[0];
}

// ============================================================
// Resample - Systematic
// ============================================================

void pf_device_resample_systematic(PFDeviceState* state, unsigned long long seed) {
    state->ess_cached = false;
    int N = state->n_particles;
    int grid = state->grid_size;

    // Step 1: Find max log_weight using persistent buffers
    pfd_find_max_kernel<<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), state->stream>>>(
        state->d_log_weights, state->d_partial_c, N);
    CUDA_CHECK_LAST();

    pfd_finalize_max_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double),
                              state->stream>>>(
        state->d_partial_c, state->d_reduction_result, grid);

    // Steps 2-3: exp(lw - max), store it, and sum with the same block tree.
    pfd_exp_sum_kernel<<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), state->stream>>>(
        state->d_log_weights, state->d_weights_norm,
        state->d_reduction_result, state->d_partial_b, N);

    pfd_finalize_sum_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double),
                              state->stream>>>(
        state->d_partial_b, state->d_reduction_result + 1, grid);

    // Step 4: Normalize weights
    pfd_normalize_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_weights_norm, state->d_reduction_result + 1, N);

    // Step 5: Inclusive scan (CDF) using thrust on the custom stream
    thrust::device_ptr<double> w_ptr(state->d_weights_norm);
    thrust::device_ptr<double> cdf_ptr(state->d_cdf);
    thrust::inclusive_scan(thrust::cuda::par.on(state->stream), w_ptr, w_ptr + N, cdf_ptr);

    // Step 6: Generate u0
    double u0 = (double)(seed % 1000000) / (double)(1000000 * N);

    // Step 7: Resample into tmp buffers
    pfd_systematic_resample_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_cdf,
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz, state->d_vcov, state->d_pcb,
        state->d_px_tmp, state->d_py_tmp, state->d_pz_tmp,
        state->d_vx_tmp, state->d_vy_tmp, state->d_vz_tmp, state->d_vcov_tmp, state->d_pcb_tmp,
        state->d_resample_ancestor,
        N, u0);
    CUDA_CHECK_LAST();

    // Step 8: Swap pointers (tmp becomes primary)
    std::swap(state->d_px, state->d_px_tmp);
    std::swap(state->d_py, state->d_py_tmp);
    std::swap(state->d_pz, state->d_pz_tmp);
    std::swap(state->d_vx, state->d_vx_tmp);
    std::swap(state->d_vy, state->d_vy_tmp);
    std::swap(state->d_vz, state->d_vz_tmp);
    std::swap(state->d_vcov, state->d_vcov_tmp);
    std::swap(state->d_pcb, state->d_pcb_tmp);

    // Step 9: Reset log weights to uniform
    pfd_reset_weights_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(state->d_log_weights, N);
    CUDA_CHECK_LAST();
}

// ============================================================
// Resample - Megopolis
// ============================================================

void pf_device_resample_megopolis(PFDeviceState* state, int n_iterations, unsigned long long seed) {
    state->ess_cached = false;
    int N = state->n_particles;
    int grid = state->grid_size;

    // Mix ancestor indices with double-buffering, then gather the particle
    // state once into the tmp arrays (see pfd_megopolis_index_kernel).
    int* idx_src = state->d_megopolis_idx_a;
    int* idx_dst = state->d_megopolis_idx_b;
    pfd_megopolis_index_init_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(idx_src, N);
    for (int iter = 0; iter < n_iterations; iter++) {
        pfd_megopolis_index_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
            idx_src, idx_dst, state->d_log_weights, N, seed, iter);
        std::swap(idx_src, idx_dst);
    }
    pfd_megopolis_gather_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz, state->d_vcov, state->d_pcb,
        state->d_px_tmp, state->d_py_tmp, state->d_pz_tmp,
        state->d_vx_tmp, state->d_vy_tmp, state->d_vz_tmp, state->d_vcov_tmp, state->d_pcb_tmp,
        idx_src, N, true);
    CUDA_CHECK_LAST();

    // The gather wrote to the tmp arrays; swap so the primary arrays hold it.
    std::swap(state->d_px, state->d_px_tmp);
    std::swap(state->d_py, state->d_py_tmp);
    std::swap(state->d_pz, state->d_pz_tmp);
    std::swap(state->d_vx, state->d_vx_tmp);
    std::swap(state->d_vy, state->d_vy_tmp);
    std::swap(state->d_vz, state->d_vz_tmp);
    std::swap(state->d_vcov, state->d_vcov_tmp);
    std::swap(state->d_pcb, state->d_pcb_tmp);

    // Reset log weights to uniform
    pfd_reset_weights_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(state->d_log_weights, N);
    CUDA_CHECK_LAST();
}

void pf_device_resample_megopolis_coalesced(PFDeviceState* state, int n_iterations,
                                           unsigned long long seed) {
    state->ess_cached = false;
    int N = state->n_particles;
    int grid = state->grid_size;
    // Extra covariance copy is amortized only for large, well-mixed gathers.
    const bool packed_gather = n_iterations > 0 &&
        (state->covariance_uniform || (N >= 100000 && n_iterations >= 60));

    pfd_megopolis_coalesced_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_log_weights, state->d_megopolis_idx_a, N, seed, n_iterations);
    if (packed_gather) {
        if (!state->covariance_uniform) {
            pfd_gather_resample_covariance_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
                state->d_vcov, state->d_vcov_tmp, state->d_megopolis_idx_a, N);
            CUDA_CHECK_LAST();
        }
        // All active covariance values now belong to the destination. The
        // former source may be overwritten after that copy on this stream.
        double* gather_scratch = state->covariance_uniform ? state->d_vcov_tmp : state->d_vcov;
        pfd_pack_common_resample_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
            state->d_px, state->d_py, state->d_pz,
            state->d_vx, state->d_vy, state->d_vz, state->d_pcb,
            gather_scratch, N);
        CUDA_CHECK_LAST();
        pfd_gather_common_packed_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
            gather_scratch, state->d_megopolis_idx_a,
            state->d_px_tmp, state->d_py_tmp, state->d_pz_tmp,
            state->d_vx_tmp, state->d_vy_tmp, state->d_vz_tmp, state->d_pcb_tmp,
            state->d_log_weights, N);
        CUDA_CHECK_LAST();
    } else {
    pfd_megopolis_gather_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz, state->d_vcov, state->d_pcb,
        state->d_px_tmp, state->d_py_tmp, state->d_pz_tmp,
        state->d_vx_tmp, state->d_vy_tmp, state->d_vz_tmp, state->d_vcov_tmp, state->d_pcb_tmp,
        state->d_megopolis_idx_a, N, !state->covariance_uniform);
    CUDA_CHECK_LAST();
    }

    std::swap(state->d_px, state->d_px_tmp);
    std::swap(state->d_py, state->d_py_tmp);
    std::swap(state->d_pz, state->d_pz_tmp);
    std::swap(state->d_vx, state->d_vx_tmp);
    std::swap(state->d_vy, state->d_vy_tmp);
    std::swap(state->d_vz, state->d_vz_tmp);
    // An invariant covariance already belongs to every new ancestor. Retain
    // its current array; the unused destination covariance need not be valid.
    if (!state->covariance_uniform) std::swap(state->d_vcov, state->d_vcov_tmp);
    std::swap(state->d_pcb, state->d_pcb_tmp);

    if (!packed_gather) {
    pfd_reset_weights_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(state->d_log_weights, N);
    CUDA_CHECK_LAST();
    }
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

    pfd_finalize_estimate_kernel<<<1, BLOCK_SIZE, 6 * BLOCK_SIZE * sizeof(double),
                                   state->stream>>>(
        state->d_partial_a, state->d_partial_b, state->d_partial_c,
        state->d_reduction_result, grid);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpyAsync(
        state->h_result_pinned, state->d_reduction_result, 4 * sizeof(double),
        cudaMemcpyDeviceToHost, state->stream));
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    std::memcpy(result, state->h_result_pinned, 4 * sizeof(double));
}

// ============================================================
// Get particles (D2H only when explicitly requested)
// ============================================================

bool pf_device_position_mixture_components(const PFDeviceState* state,
    const double* refs, const double* log_mix, int n_components,
    double sigma_squared, double* output) {
    PFDPositionMixtureInput input = {};
    for (int i = 0; i < n_components; ++i) {
        std::memcpy(input.values + 4 * i, refs + 3 * i, 3 * sizeof(double));
        input.values[4 * i + 3] = log_mix[i];
    }
    // Resampling fully overwrites the inactive covariance before reading it.
    // Up to 4N components fit in the existing unused 9N destination array.
    int* nonfinite = reinterpret_cast<int*>(state->d_reduction_result);
    CUDA_CHECK(cudaMemsetAsync(nonfinite, 0, sizeof(int), state->stream));
    size_t count = (size_t)state->n_particles * n_components;
    unsigned int grid = (unsigned int)((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (n_components == 2) {
        pfd_position_mixture_components_kernel<2><<<grid, BLOCK_SIZE, 0, state->stream>>>(
            state->d_px, state->d_py, state->d_pz, state->d_vcov_tmp,
            nonfinite, input, sigma_squared, state->n_particles);
    } else if (n_components == 3) {
        pfd_position_mixture_components_kernel<3><<<grid, BLOCK_SIZE, 0, state->stream>>>(
            state->d_px, state->d_py, state->d_pz, state->d_vcov_tmp,
            nonfinite, input, sigma_squared, state->n_particles);
    } else {
        pfd_position_mixture_components_kernel<4><<<grid, BLOCK_SIZE, 0, state->stream>>>(
            state->d_px, state->d_py, state->d_pz, state->d_vcov_tmp,
            nonfinite, input, sigma_squared, state->n_particles);
    }
    CUDA_CHECK_LAST();
    int* flag = reinterpret_cast<int*>(state->h_result_pinned);
    CUDA_CHECK(cudaMemcpyAsync(flag, nonfinite, sizeof(int), cudaMemcpyDeviceToHost, state->stream));
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    if (*flag != 0) return false;
    CUDA_CHECK(cudaMemcpyAsync(output, state->d_vcov_tmp, count * sizeof(double),
                               cudaMemcpyDeviceToHost, state->stream));
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    return true;
}

void pf_device_get_particles(const PFDeviceState* state, double* output) {
    int N = state->n_particles;

    // The inactive covariance has 9N doubles; the packed output needs 4N.
    // Resampling overwrites this scratch before reading its destination.
    // Keep the copy and completion semantics of the original getter.
    double* d_out = state->d_vcov_tmp;
    size_t sz_out = (size_t)N * 4 * sizeof(double);

    int grid = state->grid_size;
    pfd_get_particles_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz, state->d_pcb,
        d_out, N);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    CUDA_CHECK(cudaMemcpy(output, d_out, sz_out, cudaMemcpyDeviceToHost));
}

void pf_device_get_particle_states(const PFDeviceState* state, double* output) {
    int N = state->n_particles;
    double* d_out;
    size_t sz_out = (size_t)N * 16 * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_out, sz_out));

    int grid = state->grid_size;
    pfd_get_particle_states_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz, state->d_vcov, state->d_pcb,
        d_out, N);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    CUDA_CHECK(cudaMemcpy(output, d_out, sz_out, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
}

bool pf_device_gather_from(PFDeviceState* target, const PFDeviceState* source,
                           const int* ancestors) {
    // In-place or cross-device copies retain the host migration fallback.
    if (target == source) return false;
    cudaPointerAttributes src_attributes, dst_attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&src_attributes, source->d_px));
    CUDA_CHECK(cudaPointerGetAttributes(&dst_attributes, target->d_px));
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (src_attributes.device != dst_attributes.device ||
        dst_attributes.device != current_device) return false;

    // The source is read on the target's stream. Finish source producers first.
    CUDA_CHECK(cudaStreamSynchronize(source->stream));
    target->ess_cached = false;
    target->covariance_uniform = false;
    int N = target->n_particles;
    CUDA_CHECK(cudaMemcpyAsync(target->d_megopolis_idx_a, ancestors,
        (size_t)N * sizeof(int), cudaMemcpyHostToDevice, target->stream));
    pfd_megopolis_gather_kernel<<<target->grid_size, BLOCK_SIZE, 0, target->stream>>>(
        source->d_px, source->d_py, source->d_pz,
        source->d_vx, source->d_vy, source->d_vz, source->d_vcov, source->d_pcb,
        target->d_px, target->d_py, target->d_pz,
        target->d_vx, target->d_vy, target->d_vz, target->d_vcov, target->d_pcb,
        target->d_megopolis_idx_a, N, true);
    CUDA_CHECK_LAST();
    pfd_reset_weights_kernel<<<target->grid_size, BLOCK_SIZE, 0, target->stream>>>(target->d_log_weights, N);
    CUDA_CHECK_LAST();
    // Input indices and both states can be reused as soon as this call returns.
    CUDA_CHECK(cudaStreamSynchronize(target->stream));
    return true;
}

bool pf_device_gather_from_cdf(PFDeviceState* target, const PFDeviceState* source,
                               const double* cdf) {
    if (target == source) return false;
    cudaPointerAttributes src_attributes, dst_attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&src_attributes, source->d_px));
    CUDA_CHECK(cudaPointerGetAttributes(&dst_attributes, target->d_px));
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (src_attributes.device != dst_attributes.device ||
        dst_attributes.device != current_device) return false;

    // Source scratch is large enough even when the target has fewer particles.
    // Finish its producers before borrowing d_cdf on the target's stream.
    CUDA_CHECK(cudaStreamSynchronize(source->stream));
    target->ess_cached = false;
    target->covariance_uniform = false;
    int N = target->n_particles;
    CUDA_CHECK(cudaMemcpyAsync(source->d_cdf, cdf,
        (size_t)source->n_particles * sizeof(double), cudaMemcpyHostToDevice, target->stream));
    pfd_pool_cdf_indices_kernel<<<target->grid_size, BLOCK_SIZE, 0, target->stream>>>(
        source->d_cdf, source->n_particles, target->d_megopolis_idx_a, N);
    CUDA_CHECK_LAST();
    pfd_megopolis_gather_kernel<<<target->grid_size, BLOCK_SIZE, 0, target->stream>>>(
        source->d_px, source->d_py, source->d_pz,
        source->d_vx, source->d_vy, source->d_vz, source->d_vcov, source->d_pcb,
        target->d_px, target->d_py, target->d_pz,
        target->d_vx, target->d_vy, target->d_vz, target->d_vcov, target->d_pcb,
        target->d_megopolis_idx_a, N, true);
    CUDA_CHECK_LAST();
    pfd_reset_weights_kernel<<<target->grid_size, BLOCK_SIZE, 0, target->stream>>>(target->d_log_weights, N);
    CUDA_CHECK_LAST();
    // CDF input and both states, including source scratch, are reusable now.
    CUDA_CHECK(cudaStreamSynchronize(target->stream));
    return true;
}

// An exact run of repeated positive FP64 additions has a constant ULP
// increment inside one exponent bin. A tie starting on an odd significand
// needs one explicit step; every exponent crossing is evaluated in FP64.
struct pfd_uniform_cdf_segment {
    int first;
    int last;
    std::uint64_t base;
    std::uint64_t stride;
};
static_assert(sizeof(pfd_uniform_cdf_segment) == 24);

static bool pfd_uniform_cdf_segments(int N,
    std::array<pfd_uniform_cdf_segment, 64>& segments, int& count) {
#if defined(_M_X64) || defined(__x86_64__)
    if ((_mm_getcsr() & _MM_ROUND_MASK) != _MM_ROUND_NEAREST) return false;
#else
    return false;
#endif
    const double step = 1.0 / static_cast<double>(N);
    std::uint64_t step_bits;
    std::memcpy(&step_bits, &step, sizeof(step_bits));
    const int step_exponent = static_cast<int>(step_bits >> 52);
    const std::uint64_t significand = (step_bits & ((1ULL << 52) - 1)) | (1ULL << 52);
    std::uint64_t current = step_bits;
    int index = 0;
    count = 0;
    while (index < N) {
        if (count >= static_cast<int>(segments.size())) return false;
        const int exponent = static_cast<int>(current >> 52);
        const int shift = exponent - step_exponent;
        if (shift < 0 || shift >= 53) return false;
        const std::uint64_t quotient = significand >> shift;
        const std::uint64_t remainder = significand & ((1ULL << shift) - 1);
        const bool tie = shift > 0 && remainder == (1ULL << (shift - 1));
        if (tie && (current & 1)) {
            segments[count++] = {index, index, current, 0};
            ++index;
            if (index < N) {
                double previous;
                std::memcpy(&previous, &current, sizeof(previous));
                const double next = previous + step;
                std::memcpy(&current, &next, sizeof(current));
            }
            continue;
        }
        std::uint64_t delta = quotient;
        if (shift > 0) {
            const std::uint64_t half = 1ULL << (shift - 1);
            if (remainder > half || (remainder == half && (quotient & 1))) ++delta;
        }
        if (delta == 0) return false;
        const std::uint64_t boundary = static_cast<std::uint64_t>(exponent + 1) << 52;
        const std::uint64_t advances = std::min<std::uint64_t>(
            static_cast<std::uint64_t>(N - index - 1), (boundary - 1 - current) / delta);
        const int last = index + static_cast<int>(advances);
        segments[count++] = {index, last, current, delta};
        index = last + 1;
        if (index < N) {
            const std::uint64_t last_bits = current + delta * advances;
            double previous;
            std::memcpy(&previous, &last_bits, sizeof(previous));
            const double next = previous + step;
            std::memcpy(&current, &next, sizeof(current));
        }
    }
    return true;
}

__global__ void pfd_uniform_cdf_kernel(double* cdf, int N,
    const pfd_uniform_cdf_segment* segments, int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int low = 0;
    int high = count;
    while (low + 1 < high) {
        const int mid = low + (high - low) / 2;
        if (tid >= segments[mid].first) low = mid;
        else high = mid;
    }
    const pfd_uniform_cdf_segment segment = segments[low];
    const std::uint64_t bits = segment.base + segment.stride *
        static_cast<std::uint64_t>(tid - segment.first);
    cdf[tid] = __longlong_as_double(static_cast<long long>(bits));
}

// Invert the exact monotone CDF bit spans directly. Positive finite FP64
// values have the same order as their uint64 representations. Find the
// first span whose final value reaches the unchanged FP64 threshold, then
// compute the first element reaching it with exact integer ceiling division.
__global__ void pfd_uniform_cdf_indices_kernel(int* indices, int target_count,
    int source_count, const pfd_uniform_cdf_segment* segments,
    const std::uint64_t* reciprocals, int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= target_count) return;
    const double threshold = __ddiv_rn(__dadd_rn((double)tid, 0.5), (double)target_count);
    const std::uint64_t threshold_bits = static_cast<std::uint64_t>(__double_as_longlong(threshold));
    int low = 0;
    int high = count;
    while (low < high) {
        const int mid = low + (high - low) / 2;
        const pfd_uniform_cdf_segment span = segments[mid];
        const std::uint64_t end = span.base + span.stride *
            static_cast<std::uint64_t>(span.last - span.first);
        if (end < threshold_bits) low = mid + 1;
        else high = mid;
    }
    if (low == count) {
        indices[tid] = source_count - 1;
        return;
    }
    const pfd_uniform_cdf_segment span = segments[low];
    int selected = span.first;
    if (threshold_bits > span.base && span.stride != 0) {
        const std::uint64_t distance = threshold_bits - span.base;
        // reciprocal = ceil(2^64 / stride). Since distance < 2^52,
        // its high product overestimates floor(distance/stride) by at
        // most one. The exact product comparison yields the ceiling,
        // including exact ties, without a per-particle integer division.
        std::uint64_t offset = span.stride == 1 ? distance :
            __umul64hi(distance, reciprocals[low]);
        if (offset * span.stride < distance) ++offset;
        selected += static_cast<int>(offset);
    }
    indices[tid] = selected;
}

// Keep the exact uniform-CDF inverse, then copy the same seven scalar
// fields and nine covariance components and reset weights in one launch.
// Every lane initializes its ancestor and reaches the block barrier, including
// the tail. Covariance keeps the original contiguous component/lane mapping.
__global__ void pfd_uniform_cdf_gather_reset_kernel(
    const double* src_px, const double* src_py, const double* src_pz,
    const double* src_vx, const double* src_vy, const double* src_vz,
    const double* src_vcov, const double* src_pcb,
    double* dst_px, double* dst_py, double* dst_pz,
    double* dst_vx, double* dst_vy, double* dst_vz,
    double* dst_vcov, double* dst_pcb, double* log_weights,
    int target_count, int source_count, const pfd_uniform_cdf_segment* segments,
    const std::uint64_t* reciprocals, int count) {
    __shared__ int ancestors[BLOCK_SIZE];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int selected = 0;
    if (tid < target_count) {
        const double threshold = __ddiv_rn(__dadd_rn((double)tid, 0.5), (double)target_count);
        const std::uint64_t threshold_bits = static_cast<std::uint64_t>(__double_as_longlong(threshold));
        int low = 0;
        int high = count;
        while (low < high) {
            const int mid = low + (high - low) / 2;
            const pfd_uniform_cdf_segment span = segments[mid];
            const std::uint64_t end = span.base + span.stride *
                static_cast<std::uint64_t>(span.last - span.first);
            if (end < threshold_bits) low = mid + 1;
            else high = mid;
        }
        selected = source_count - 1;
        if (low != count) {
            const pfd_uniform_cdf_segment span = segments[low];
            selected = span.first;
            if (threshold_bits > span.base && span.stride != 0) {
                const std::uint64_t distance = threshold_bits - span.base;
                std::uint64_t offset = span.stride == 1 ? distance :
                    __umul64hi(distance, reciprocals[low]);
                if (offset * span.stride < distance) ++offset;
                selected += static_cast<int>(offset);
            }
        }
    }
    ancestors[threadIdx.x] = selected;
    __syncthreads();
    if (tid < target_count) {
        dst_px[tid] = src_px[selected];
        dst_py[tid] = src_py[selected];
        dst_pz[tid] = src_pz[selected];
        dst_vx[tid] = src_vx[selected];
        dst_vy[tid] = src_vy[selected];
        dst_vz[tid] = src_vz[selected];
        dst_pcb[tid] = src_pcb[selected];
        log_weights[tid] = 0.0;
    }
    const int first = blockIdx.x * blockDim.x;
    const int valid = min((int)blockDim.x, target_count - first);
    for (int local = threadIdx.x; local < valid * 9; local += blockDim.x) {
        const int component = local % 9;
        dst_vcov[(size_t)first * 9 + local] =
            src_vcov[(size_t)ancestors[local / 9] * 9 + component];
    }
}

static bool pfd_gather_from_uniform_cdf_inverse(PFDeviceState* target, const PFDeviceState* source,
                                      double* cdf_output) {
    if (target == source || source->n_particles < 65536) return false;
    std::array<pfd_uniform_cdf_segment, 64> segments;
    int count;
    if (!pfd_uniform_cdf_segments(source->n_particles, segments, count)) return false;
    const size_t segment_bytes = static_cast<size_t>(count) * sizeof(segments[0]);
    const size_t capacity_bytes = static_cast<size_t>(target->n_particles) * sizeof(int);
    if (segment_bytes > capacity_bytes) return false;
    const size_t direct_bytes = segment_bytes + static_cast<size_t>(count) * sizeof(std::uint64_t);
    const bool direct = direct_bytes <= capacity_bytes;
    std::array<std::uint64_t, 64 * 4> metadata;
    std::memcpy(metadata.data(), segments.data(), segment_bytes);
    if (direct) {
        const size_t offset = segment_bytes / sizeof(std::uint64_t);
        for (int i = 0; i < count; ++i) {
            const std::uint64_t stride = segments[i].stride;
            metadata[offset + i] = stride > 1 ? (~0ULL) / stride + 1 : 0;
        }
    }
    cudaPointerAttributes src_attributes, dst_attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&src_attributes, source->d_px));
    CUDA_CHECK(cudaPointerGetAttributes(&dst_attributes, target->d_px));
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (src_attributes.device != dst_attributes.device ||
        dst_attributes.device != current_device) return false;
    CUDA_CHECK(cudaStreamSynchronize(source->stream));
    target->ess_cached = false;
    target->covariance_uniform = false;
    CUDA_CHECK(cudaMemcpyAsync(target->d_megopolis_idx_b, metadata.data(),
        direct ? direct_bytes : segment_bytes, cudaMemcpyHostToDevice, target->stream));
    if (cdf_output != nullptr || !direct) {
        // Keep the original full-CDF path when the optional reciprocal table
        // does not fit, preserving the existing small-target acceptance.
        pfd_uniform_cdf_kernel<<<source->grid_size, BLOCK_SIZE, 0, target->stream>>>(
            source->d_cdf, source->n_particles,
            reinterpret_cast<const pfd_uniform_cdf_segment*>(target->d_megopolis_idx_b), count);
        CUDA_CHECK_LAST();
    }
    const int N = target->n_particles;
    if (direct) {
        pfd_uniform_cdf_gather_reset_kernel<<<target->grid_size, BLOCK_SIZE, 0, target->stream>>>(
            source->d_px, source->d_py, source->d_pz,
            source->d_vx, source->d_vy, source->d_vz, source->d_vcov, source->d_pcb,
            target->d_px, target->d_py, target->d_pz,
            target->d_vx, target->d_vy, target->d_vz, target->d_vcov, target->d_pcb,
            target->d_log_weights, N, source->n_particles,
            reinterpret_cast<const pfd_uniform_cdf_segment*>(target->d_megopolis_idx_b),
            reinterpret_cast<const std::uint64_t*>(
                reinterpret_cast<const char*>(target->d_megopolis_idx_b) + segment_bytes), count);
        CUDA_CHECK_LAST();
    } else {
        pfd_pool_cdf_indices_kernel<<<target->grid_size, BLOCK_SIZE, 0, target->stream>>>(
            source->d_cdf, source->n_particles, target->d_megopolis_idx_a, N);
        CUDA_CHECK_LAST();
        pfd_megopolis_gather_kernel<<<target->grid_size, BLOCK_SIZE, 0, target->stream>>>(
            source->d_px, source->d_py, source->d_pz,
            source->d_vx, source->d_vy, source->d_vz, source->d_vcov, source->d_pcb,
            target->d_px, target->d_py, target->d_pz,
            target->d_vx, target->d_vy, target->d_vz, target->d_vcov, target->d_pcb,
            target->d_megopolis_idx_a, N, true);
        CUDA_CHECK_LAST();
        pfd_reset_weights_kernel<<<target->grid_size, BLOCK_SIZE, 0, target->stream>>>(target->d_log_weights, N);
        CUDA_CHECK_LAST();
    }
    if (cdf_output != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(cdf_output, source->d_cdf,
            static_cast<size_t>(source->n_particles) * sizeof(double),
            cudaMemcpyDeviceToHost, target->stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(target->stream));
    return true;
}

bool pf_device_gather_from_uniform_cdf(PFDeviceState* target, const PFDeviceState* source,
                                      double* cdf_output) {
    // Small-source measurements did not show a stable whole-operation gain.
    // Keep D201's original host/CDF/search body below this measured gate.
    if (source->n_particles >= 262144) {
        return pfd_gather_from_uniform_cdf_inverse(target, source, cdf_output);
    }
    if (target == source || source->n_particles < 65536) return false;
    std::array<pfd_uniform_cdf_segment, 64> segments;
    int count;
    if (!pfd_uniform_cdf_segments(source->n_particles, segments, count)) return false;
    const size_t segment_bytes = static_cast<size_t>(count) * sizeof(segments[0]);
    if (segment_bytes > static_cast<size_t>(target->n_particles) * sizeof(int)) return false;
    cudaPointerAttributes src_attributes, dst_attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&src_attributes, source->d_px));
    CUDA_CHECK(cudaPointerGetAttributes(&dst_attributes, target->d_px));
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (src_attributes.device != dst_attributes.device ||
        dst_attributes.device != current_device) return false;
    CUDA_CHECK(cudaStreamSynchronize(source->stream));
    target->ess_cached = false;
    target->covariance_uniform = false;
    CUDA_CHECK(cudaMemcpyAsync(target->d_megopolis_idx_b, segments.data(), segment_bytes,
        cudaMemcpyHostToDevice, target->stream));
    pfd_uniform_cdf_kernel<<<source->grid_size, BLOCK_SIZE, 0, target->stream>>>(
        source->d_cdf, source->n_particles,
        reinterpret_cast<const pfd_uniform_cdf_segment*>(target->d_megopolis_idx_b), count);
    CUDA_CHECK_LAST();
    const int N = target->n_particles;
    pfd_pool_cdf_indices_kernel<<<target->grid_size, BLOCK_SIZE, 0, target->stream>>>(
        source->d_cdf, source->n_particles, target->d_megopolis_idx_a, N);
    CUDA_CHECK_LAST();
    pfd_megopolis_gather_kernel<<<target->grid_size, BLOCK_SIZE, 0, target->stream>>>(
        source->d_px, source->d_py, source->d_pz,
        source->d_vx, source->d_vy, source->d_vz, source->d_vcov, source->d_pcb,
        target->d_px, target->d_py, target->d_pz,
        target->d_vx, target->d_vy, target->d_vz, target->d_vcov, target->d_pcb,
        target->d_megopolis_idx_a, N, true);
    CUDA_CHECK_LAST();
    pfd_reset_weights_kernel<<<target->grid_size, BLOCK_SIZE, 0, target->stream>>>(target->d_log_weights, N);
    CUDA_CHECK_LAST();
    if (cdf_output != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(cdf_output, source->d_cdf,
            static_cast<size_t>(source->n_particles) * sizeof(double),
            cudaMemcpyDeviceToHost, target->stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(target->stream));
    return true;
}

// Classify every current device weight, including signed zeros, without
// FP arithmetic/comparison on NaNs or subnormals. Finite FP64 equality is
// bit equality except for the two zero representations.
__global__ void pfd_uniform_weights_flags_kernel(const double* weights, int N, int* partial) {
    __shared__ int flags[BLOCK_SIZE];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int invalid = 0;
    if (tid < N) {
        const unsigned long long bits = __double_as_longlong(weights[tid]);
        const unsigned long long first = __double_as_longlong(weights[0]);
        const unsigned long long magnitude = 0x7fffffffffffffffULL;
        const bool both_zero = (bits & magnitude) == 0 && (first & magnitude) == 0;
        invalid = ((bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL) ||
            (bits != first && !both_zero);
    }
    flags[threadIdx.x] = invalid;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) flags[threadIdx.x] |= flags[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = flags[0];
}

__global__ void pfd_uniform_weights_finish_kernel(const int* partial, int count, int* result) {
    __shared__ int flags[BLOCK_SIZE];
    int invalid = 0;
    for (int i = threadIdx.x; i < count; i += BLOCK_SIZE) invalid |= partial[i];
    flags[threadIdx.x] = invalid;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) flags[threadIdx.x] |= flags[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) result[0] = flags[0] == 0;
}

bool pf_device_try_gather_uniform_weights(PFDeviceState* target, const PFDeviceState* source) {
    if (target == nullptr || source == nullptr || !target->allocated || !source->allocated ||
        target == source || source->n_particles < 65536) return false;
    cudaPointerAttributes src_attributes, dst_attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&src_attributes, source->d_px));
    CUDA_CHECK(cudaPointerGetAttributes(&dst_attributes, target->d_px));
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (src_attributes.device != dst_attributes.device || dst_attributes.device != current_device) return false;
    // Source stream ordering observes every preceding producer. Integer flags
    // borrow existing reduction scratch; no particles, weights or ESS cache
    // change, and later reductions overwrite their complete scratch as usual.
    int* partial = reinterpret_cast<int*>(source->d_partial_c);
    int* result = reinterpret_cast<int*>(source->d_reduction_result);
    pfd_uniform_weights_flags_kernel<<<source->grid_size, BLOCK_SIZE, 0, source->stream>>>(
        source->d_log_weights, source->n_particles, partial);
    CUDA_CHECK_LAST();
    pfd_uniform_weights_finish_kernel<<<1, BLOCK_SIZE, 0, source->stream>>>(partial, source->grid_size, result);
    CUDA_CHECK_LAST();
    int accepted = 0;
    CUDA_CHECK(cudaMemcpyAsync(&accepted, result, sizeof(accepted), cudaMemcpyDeviceToHost, source->stream));
    CUDA_CHECK(cudaStreamSynchronize(source->stream));
    if (!accepted) return false;
    // Keep D199's exact sequential CDF and original left search/gather/reset.
    return pf_device_gather_from_uniform_cdf(target, source);
}

void pf_device_set_particle_states(PFDeviceState* state, const double* input) {
    state->ess_cached = false;
    // Imported covariance can differ by particle, even if positions match.
    state->covariance_uniform = false;
    int N = state->n_particles;
    if (N >= 65536) {
        // The copy and original import kernel share the state stream, so the
        // next chunk cannot overwrite scratch still consumed by the kernel.
        int chunk_rows = (int)(((size_t)N * 9) / 16);
        for (int offset = 0; offset < N;) {
            int rows = std::min(chunk_rows, N - offset);
            CUDA_CHECK(cudaMemcpyAsync(state->d_vcov_tmp, input + (size_t)offset * 16,
                (size_t)rows * 16 * sizeof(double), cudaMemcpyHostToDevice, state->stream));
            int grid = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
            pfd_set_particle_states_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
                state->d_px + offset, state->d_py + offset, state->d_pz + offset,
                state->d_vx + offset, state->d_vy + offset, state->d_vz + offset,
                state->d_vcov + (size_t)offset * 9, state->d_pcb + offset,
                state->d_vcov_tmp, rows);
            CUDA_CHECK_LAST();
            offset += rows;
        }
        pfd_reset_weights_kernel<<<state->grid_size, BLOCK_SIZE, 0, state->stream>>>(state->d_log_weights, N);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaStreamSynchronize(state->stream));
        return;
    }
    size_t sz_in = (size_t)N * 16 * sizeof(double);
    double* d_in;
    CUDA_CHECK(cudaMalloc(&d_in, sz_in));
    CUDA_CHECK(cudaMemcpyAsync(d_in, input, sz_in, cudaMemcpyHostToDevice, state->stream));

    int grid = state->grid_size;
    pfd_set_particle_states_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz, state->d_vcov, state->d_pcb,
        d_in, N);
    CUDA_CHECK_LAST();
    pfd_reset_weights_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(state->d_log_weights, N);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    CUDA_CHECK(cudaFree(d_in));
}

void pf_device_set_velocity_covariance(PFDeviceState* state, const double* covariance) {
    PFDUniformVelocityCovariance input;
    std::memcpy(input.values, covariance, sizeof(input.values));
    size_t count = (size_t)state->n_particles * 9;
    unsigned int grid = (unsigned int)((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
    pfd_set_velocity_covariance_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_vcov, input, state->n_particles);
    CUDA_CHECK_LAST();
    // The preceding Python setter synchronized its final weight restore.
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    state->covariance_uniform = true;
}

void pf_device_get_log_weights(const PFDeviceState* state, double* output) {
    int N = state->n_particles;
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    CUDA_CHECK(cudaMemcpy(output, state->d_log_weights,
                          (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));
}

void pf_device_set_log_weights(PFDeviceState* state, const double* input) {
    state->ess_cached = false;
    int N = state->n_particles;
    CUDA_CHECK(cudaMemcpyAsync(
        state->d_log_weights,
        input,
        (size_t)N * sizeof(double),
        cudaMemcpyHostToDevice,
        state->stream));
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
}

void pf_device_fill_log_weights(PFDeviceState* state, double value) {
    state->ess_cached = false;
    pfd_fill_weights_kernel<<<state->grid_size, BLOCK_SIZE, 0, state->stream>>>(
        state->d_log_weights, state->n_particles, value);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
}

void pf_device_get_resample_ancestors(const PFDeviceState* state, int* output) {
    int N = state->n_particles;
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    CUDA_CHECK(cudaMemcpy(output, state->d_resample_ancestor,
                          (size_t)N * sizeof(int), cudaMemcpyDeviceToHost));
}

// ============================================================
// Weight update: Carrier phase AFV
// ============================================================

void pf_device_weight_carrier_afv(PFDeviceState* state,
    const double* sat_ecef, const double* carrier_phase,
    const double* weights_sat,
    int n_sat, double wavelength, double sigma_cycles) {
    state->ess_cached = false;

    int N = state->n_particles;
    int grid = state->grid_size;

    pfd_stage_satellite_observations(state, sat_ecef, carrier_phase, weights_sat, n_sat);

    // Launch AFV weight kernel on the same stream
    // Dynamic shared memory: n_sat*3 (sat_ecef) + n_sat (cp) + n_sat (ws) = n_sat*5 doubles
    size_t smem_weight = (size_t)n_sat * 5 * sizeof(double);
    pfd_weight_carrier_afv_kernel<<<grid, BLOCK_SIZE, smem_weight, state->stream>>>(
        state->d_px, state->d_py, state->d_pz, state->d_pcb,
        state->d_sat_ecef, state->d_pseudoranges, state->d_weights_sat,
        state->d_log_weights,
        N, n_sat, wavelength, sigma_cycles);
    CUDA_CHECK_LAST();
}

// ============================================================
// Weight update: DD Carrier phase AFV
// ============================================================

void pf_device_weight_dd_carrier_afv(PFDeviceState* state,
    const double* sat_ecef_k, const double* ref_ecef,
    const double* dd_carrier, const double* base_range_k,
    const double* base_range_ref, const double* weights_dd,
    const double* wavelengths_m,
    int n_dd, double sigma_cycles,
    double per_particle_nlos_threshold_cycles,
    bool per_particle_huber,
    double per_particle_huber_k) {
    state->ess_cached = false;

    int N = state->n_particles;
    int grid = state->grid_size;

    // Pack all DD data into a contiguous buffer for a single H2D transfer.
    // Layout: [sat_k: n_dd*3][ref: n_dd*3][dd_carrier: n_dd][base_range_k: n_dd][base_range_ref: n_dd][weights: n_dd][wavelengths: n_dd]
    int total_doubles = n_dd * 11;
    size_t total_bytes = (size_t)total_doubles * sizeof(double);

    ensure_obs_scratch(state, total_doubles);
    double* h_buf = state->h_obs_scratch;

    // Pack
    int off = 0;
    memcpy(h_buf + off, sat_ecef_k, n_dd * 3 * sizeof(double)); off += n_dd * 3;
    memcpy(h_buf + off, ref_ecef, n_dd * 3 * sizeof(double)); off += n_dd * 3;
    memcpy(h_buf + off, dd_carrier, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, base_range_k, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, base_range_ref, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, weights_dd, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, wavelengths_m, n_dd * sizeof(double)); off += n_dd;

    CUDA_CHECK(cudaMemcpyAsync(state->d_obs_scratch, h_buf, total_bytes,
                                cudaMemcpyHostToDevice, state->stream));

    // Launch DD-AFV kernel
    size_t smem = (size_t)total_doubles * sizeof(double);
    if (pfd_has_common_dd_reference(ref_ecef, n_dd)) {
        bool normal_wavelengths = true;
        for (int d = 0; d < n_dd; ++d) {
            if (!(wavelengths_m[d] >= DBL_MIN) || !std::isfinite(wavelengths_m[d])) {
                normal_wavelengths = false;
                break;
            }
        }
        bool fused_huber = per_particle_huber && per_particle_huber_k > 0.0 &&
            std::isfinite(per_particle_huber_k);
        bool fused_gate = normal_wavelengths && (!per_particle_huber || fused_huber) &&
            per_particle_nlos_threshold_cycles > 0.0;
        if (fused_gate) {
            for (int i = 0; i < total_doubles; ++i) {
                if (!std::isfinite(h_buf[i])) { fused_gate = false; break; }
            }
        }
        if (fused_gate && fused_huber) {
            pfd_weight_dd_single_huber_gated_kernel<true><<<grid, BLOCK_SIZE, smem, state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_cycles, per_particle_nlos_threshold_cycles, per_particle_huber_k);
        } else if (fused_gate) {
            pfd_weight_dd_single_gated_kernel<true><<<grid, BLOCK_SIZE, smem + sizeof(double), state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_cycles, per_particle_nlos_threshold_cycles);
        } else if (normal_wavelengths) {
            pfd_weight_dd_carrier_afv_kernel<true, true><<<grid, BLOCK_SIZE, smem + sizeof(double), state->stream>>>(
                state->d_px, state->d_py, state->d_pz,
                state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_cycles, per_particle_nlos_threshold_cycles,
                per_particle_huber, per_particle_huber_k);
        } else {
            pfd_weight_dd_carrier_afv_kernel<true><<<grid, BLOCK_SIZE, smem, state->stream>>>(
                state->d_px, state->d_py, state->d_pz,
                state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_cycles, per_particle_nlos_threshold_cycles,
                per_particle_huber, per_particle_huber_k);
        }
    } else if (pfd_can_fuse_mixed_dd_gate(h_buf, n_dd, 11, per_particle_nlos_threshold_cycles,
        per_particle_huber, per_particle_huber_k)) {
        if (per_particle_huber) {
            pfd_weight_dd_mixed_gated_kernel<true, true><<<grid, BLOCK_SIZE, smem, state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_cycles, per_particle_nlos_threshold_cycles, per_particle_huber_k);
        } else {
            pfd_weight_dd_mixed_gated_kernel<true, false><<<grid, BLOCK_SIZE, smem, state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_cycles, per_particle_nlos_threshold_cycles, per_particle_huber_k);
        }
    } else {
        pfd_weight_dd_carrier_afv_kernel<false><<<grid, BLOCK_SIZE, smem, state->stream>>>(
            state->d_px, state->d_py, state->d_pz,
            state->d_obs_scratch, state->d_log_weights,
            N, n_dd, sigma_cycles, per_particle_nlos_threshold_cycles,
            per_particle_huber, per_particle_huber_k);
    }
    CUDA_CHECK_LAST();

}

void pf_device_weight_dd_joint(PFDeviceState* state,
    const double* sat_ecef_k, const double* ref_ecef,
    const double* dd_pseudorange, const double* dd_carrier,
    const double* base_range_k, const double* base_range_ref,
    const double* weights_dd, const double* wavelengths_m,
    int n_dd, double sigma_pr, double sigma_cycles,
    double per_particle_nlos_threshold_m,
    double per_particle_nlos_threshold_cycles,
    bool per_particle_huber,
    double per_particle_huber_dd_pr_k,
    double per_particle_huber_dd_carrier_k) {
    state->ess_cached = false;

    if (n_dd <= 0 || sigma_pr <= 0.0 || sigma_cycles <= 0.0) return;

    int N = state->n_particles;
    int grid = state->grid_size;
    int total_doubles = n_dd * 12;
    size_t total_bytes = (size_t)total_doubles * sizeof(double);

    ensure_obs_scratch(state, total_doubles);
    double* h_buf = state->h_obs_scratch;
    int off = 0;
    memcpy(h_buf + off, sat_ecef_k, n_dd * 3 * sizeof(double)); off += n_dd * 3;
    memcpy(h_buf + off, ref_ecef, n_dd * 3 * sizeof(double)); off += n_dd * 3;
    memcpy(h_buf + off, dd_pseudorange, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, dd_carrier, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, base_range_k, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, base_range_ref, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, weights_dd, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, wavelengths_m, n_dd * sizeof(double));

    CUDA_CHECK(cudaMemcpyAsync(
        state->d_obs_scratch, h_buf, total_bytes,
        cudaMemcpyHostToDevice, state->stream));
    // Carrier-only Huber can change Gaussian PR tail contraction in the
    // common-reference specialization. Keep its original instruction body.
    bool carrier_only_huber = per_particle_huber &&
        !(per_particle_huber_dd_pr_k > 0.0) && per_particle_huber_dd_carrier_k > 0.0;
    if (!carrier_only_huber && pfd_has_common_dd_reference(ref_ecef, n_dd)) {
        bool fused_huber_gate = per_particle_huber &&
            per_particle_huber_dd_pr_k > 0.0 && per_particle_huber_dd_carrier_k > 0.0 &&
            std::isfinite(per_particle_huber_dd_pr_k) && std::isfinite(per_particle_huber_dd_carrier_k);
        bool fused_gate = (!per_particle_huber || fused_huber_gate) && n_dd < 512 &&
            (per_particle_nlos_threshold_m > 0.0 || per_particle_nlos_threshold_cycles > 0.0);
        if (fused_gate) {
            // Retain the preceding instruction path for exceptional observations,
            // including its NaN payload/sign and invalid-wavelength behavior.
            for (int i = 0; i < total_doubles; ++i) {
                if (!std::isfinite(h_buf[i]) ||
                    (i >= n_dd * 11 && !(h_buf[i] >= DBL_MIN))) {
                    fused_gate = false;
                    break;
                }
            }
        }
        if (fused_gate && fused_huber_gate) {
            pfd_weight_dd_joint_huber_gated_kernel<<<grid, BLOCK_SIZE,
                (size_t)total_doubles * sizeof(double), state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_pr, sigma_cycles,
                per_particle_nlos_threshold_m, per_particle_nlos_threshold_cycles,
                per_particle_huber_dd_pr_k, per_particle_huber_dd_carrier_k);
        } else if (fused_gate) {
            pfd_weight_dd_joint_gated_kernel<<<grid, BLOCK_SIZE,
                ((size_t)total_doubles + 2) * sizeof(double), state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_pr, sigma_cycles,
                per_particle_nlos_threshold_m, per_particle_nlos_threshold_cycles);
        } else if (n_dd < 512) {
            pfd_weight_dd_joint_kernel<true, true><<<
                grid, BLOCK_SIZE, ((size_t)total_doubles + 2) * sizeof(double), state->stream>>>(
                state->d_px, state->d_py, state->d_pz,
                state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_pr, sigma_cycles,
                per_particle_nlos_threshold_m, per_particle_nlos_threshold_cycles,
                per_particle_huber, per_particle_huber_dd_pr_k,
                per_particle_huber_dd_carrier_k);
        } else {
            pfd_weight_dd_joint_kernel<true><<<
                grid, BLOCK_SIZE, (size_t)total_doubles * sizeof(double), state->stream>>>(
                state->d_px, state->d_py, state->d_pz,
                state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_pr, sigma_cycles,
                per_particle_nlos_threshold_m, per_particle_nlos_threshold_cycles,
                per_particle_huber, per_particle_huber_dd_pr_k,
                per_particle_huber_dd_carrier_k);
        }
    } else {
        bool fused_huber = per_particle_huber &&
            per_particle_huber_dd_pr_k > 0.0 && per_particle_huber_dd_carrier_k > 0.0 &&
            std::isfinite(per_particle_huber_dd_pr_k) && std::isfinite(per_particle_huber_dd_carrier_k);
        bool fused_gate = n_dd >= 2 && (!per_particle_huber || fused_huber) &&
            (per_particle_nlos_threshold_m > 0.0 || per_particle_nlos_threshold_cycles > 0.0);
        if (fused_gate) {
            for (int i = 0; i < total_doubles; ++i) {
                if (!std::isfinite(h_buf[i]) || (i >= n_dd * 11 && !(h_buf[i] >= DBL_MIN))) {
                    fused_gate = false;
                    break;
                }
            }
        }
        if (fused_gate && fused_huber) {
            pfd_weight_dd_joint_mixed_gated_kernel<true><<<
                grid, BLOCK_SIZE, (size_t)total_doubles * sizeof(double), state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_pr, sigma_cycles,
                per_particle_nlos_threshold_m, per_particle_nlos_threshold_cycles,
                per_particle_huber_dd_pr_k, per_particle_huber_dd_carrier_k);
        } else if (fused_gate) {
            pfd_weight_dd_joint_mixed_gated_kernel<false><<<
                grid, BLOCK_SIZE, (size_t)total_doubles * sizeof(double), state->stream>>>(
                state->d_px, state->d_py, state->d_pz, state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_pr, sigma_cycles,
                per_particle_nlos_threshold_m, per_particle_nlos_threshold_cycles,
                per_particle_huber_dd_pr_k, per_particle_huber_dd_carrier_k);
        } else {
            pfd_weight_dd_joint_kernel<false><<<
                grid, BLOCK_SIZE, (size_t)total_doubles * sizeof(double), state->stream>>>(
                state->d_px, state->d_py, state->d_pz,
                state->d_obs_scratch, state->d_log_weights,
                N, n_dd, sigma_pr, sigma_cycles,
                per_particle_nlos_threshold_m, per_particle_nlos_threshold_cycles,
                per_particle_huber, per_particle_huber_dd_pr_k,
                per_particle_huber_dd_carrier_k);
        }
    }
    CUDA_CHECK_LAST();
}

// ============================================================
// Weight/update: Doppler velocity
// ============================================================

void pf_device_weight_doppler(PFDeviceState* state,
    const double* sat_ecef, const double* sat_vel,
    const double* doppler_hz, const double* weights_sat,
    int n_sat, double wavelength_m,
    double sigma_mps,
    double velocity_update_gain,
    double max_velocity_update_mps) {
    state->ess_cached = false;

    if (n_sat <= 0 || sigma_mps <= 0.0 || wavelength_m <= 0.0) {
        return;
    }

    int N = state->n_particles;
    int grid = state->grid_size;

    int total_doubles = n_sat * 8;
    size_t total_bytes = (size_t)total_doubles * sizeof(double);
    ensure_obs_scratch(state, total_doubles);
    double* h_buf = state->h_obs_scratch;

    int off = 0;
    memcpy(h_buf + off, sat_ecef, n_sat * 3 * sizeof(double)); off += n_sat * 3;
    memcpy(h_buf + off, sat_vel, n_sat * 3 * sizeof(double)); off += n_sat * 3;
    memcpy(h_buf + off, doppler_hz, n_sat * sizeof(double)); off += n_sat;
    memcpy(h_buf + off, weights_sat, n_sat * sizeof(double)); off += n_sat;

    CUDA_CHECK(cudaMemcpyAsync(
        state->d_obs_scratch, h_buf, total_bytes,
        cudaMemcpyHostToDevice, state->stream));

    size_t smem = (size_t)total_doubles * sizeof(double);
    if (n_sat <= 8) {
    pfd_weight_doppler_cached_kernel<8><<<(N + 127) / 128, 128, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz,
        state->d_obs_scratch, state->d_log_weights,
        N, n_sat, wavelength_m, sigma_mps,
        velocity_update_gain, max_velocity_update_mps);
    } else if (n_sat <= 16) {
    pfd_weight_doppler_cached_kernel<16><<<(N + 127) / 128, 128, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz,
        state->d_obs_scratch, state->d_log_weights,
        N, n_sat, wavelength_m, sigma_mps,
        velocity_update_gain, max_velocity_update_mps);
    } else if (n_sat <= 24) {
    pfd_weight_doppler_cached_kernel<24><<<(N + 127) / 128, 128, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz,
        state->d_obs_scratch, state->d_log_weights,
        N, n_sat, wavelength_m, sigma_mps,
        velocity_update_gain, max_velocity_update_mps);
    } else if (n_sat <= 32) {
    pfd_weight_doppler_cached_kernel<32><<<(N + 127) / 128, 128, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz,
        state->d_obs_scratch, state->d_log_weights,
        N, n_sat, wavelength_m, sigma_mps,
        velocity_update_gain, max_velocity_update_mps);
    } else if (n_sat <= 48) {
    pfd_weight_doppler_wide_cached_kernel<48><<<(N + 127) / 128, 128, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz,
        state->d_obs_scratch, state->d_log_weights,
        N, n_sat, wavelength_m, sigma_mps,
        velocity_update_gain, max_velocity_update_mps);
    } else if (n_sat <= 64) {
    pfd_weight_doppler_wide_cached_kernel<64><<<(N + 127) / 128, 128, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz,
        state->d_obs_scratch, state->d_log_weights,
        N, n_sat, wavelength_m, sigma_mps,
        velocity_update_gain, max_velocity_update_mps);
    } else {
    pfd_weight_doppler_kernel<<<grid, BLOCK_SIZE, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz,
        state->d_obs_scratch, state->d_log_weights,
        N, n_sat, wavelength_m, sigma_mps,
        velocity_update_gain, max_velocity_update_mps);
    }
    CUDA_CHECK_LAST();

}

void pf_device_doppler_kf_update(PFDeviceState* state,
    const double* sat_ecef, const double* sat_vel,
    const double* doppler_hz, const double* weights_sat,
    int n_sat, double wavelength_m,
    double sigma_mps) {
    state->ess_cached = false;

    if (n_sat <= 0 || sigma_mps <= 0.0 || wavelength_m <= 0.0) {
        return;
    }

    int N = state->n_particles;
    // The bounded geometry caches use a larger register footprint. Smaller
    // blocks pack their warps more tightly without changing per-particle math.
    int block = n_sat <= 64 ? 128 : BLOCK_SIZE;
    int grid = (N + block - 1) / block;

    int total_doubles = n_sat * 8;
    size_t total_bytes = (size_t)total_doubles * sizeof(double);
    ensure_obs_scratch(state, total_doubles);
    double* h_buf = state->h_obs_scratch;

    int off = 0;
    memcpy(h_buf + off, sat_ecef, n_sat * 3 * sizeof(double)); off += n_sat * 3;
    memcpy(h_buf + off, sat_vel, n_sat * 3 * sizeof(double)); off += n_sat * 3;
    memcpy(h_buf + off, doppler_hz, n_sat * sizeof(double)); off += n_sat;
    memcpy(h_buf + off, weights_sat, n_sat * sizeof(double)); off += n_sat;

    CUDA_CHECK(cudaMemcpyAsync(
        state->d_obs_scratch, h_buf, total_bytes,
        cudaMemcpyHostToDevice, state->stream));

    size_t smem = (size_t)n_sat * (n_sat <= MAX_SATS ? 9 : 8) * sizeof(double);
    // The measurement geometry can now produce distinct particle covariances.
    state->covariance_uniform = false;
    if (n_sat <= 16) {
        pfd_doppler_kf_update_kernel<16><<<grid, block, smem, state->stream>>>(
            state->d_px, state->d_py, state->d_pz,
            state->d_vx, state->d_vy, state->d_vz, state->d_vcov,
            state->d_obs_scratch, state->d_log_weights,
            N, n_sat, wavelength_m, sigma_mps);
    } else if (n_sat <= 32) {
        pfd_doppler_kf_update_kernel<32><<<grid, block, smem, state->stream>>>(
            state->d_px, state->d_py, state->d_pz,
            state->d_vx, state->d_vy, state->d_vz, state->d_vcov,
            state->d_obs_scratch, state->d_log_weights,
            N, n_sat, wavelength_m, sigma_mps);
    } else if (n_sat <= 48) {
        pfd_doppler_kf_wide_update_kernel<48><<<grid, block, smem, state->stream>>>(
            state->d_px, state->d_py, state->d_pz,
            state->d_vx, state->d_vy, state->d_vz, state->d_vcov,
            state->d_obs_scratch, state->d_log_weights,
            N, n_sat, wavelength_m, sigma_mps);
    } else if (n_sat <= 64) {
        pfd_doppler_kf_wide_update_kernel<64><<<grid, block, smem, state->stream>>>(
            state->d_px, state->d_py, state->d_pz,
            state->d_vx, state->d_vy, state->d_vz, state->d_vcov,
            state->d_obs_scratch, state->d_log_weights,
            N, n_sat, wavelength_m, sigma_mps);
    } else {
        pfd_doppler_kf_update_kernel<0><<<grid, block, smem, state->stream>>>(
            state->d_px, state->d_py, state->d_pz,
            state->d_vx, state->d_vy, state->d_vz, state->d_vcov,
            state->d_obs_scratch, state->d_log_weights,
            N, n_sat, wavelength_m, sigma_mps);
    }
    CUDA_CHECK_LAST();

}

// ============================================================
// Explicit synchronization
// ============================================================

void pf_device_sync(PFDeviceState* state) {
    if (!state || !state->allocated) return;
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
}

}  // namespace gnss_gpu
