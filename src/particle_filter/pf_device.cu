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

__global__ void pfd_predict_kernel(double* px, double* py, double* pz,
                                   double* vx, double* vy, double* vz,
                                   double* vcov,
                                   double* pcb,
                                   const double* vel_guide,  // [3]: vx, vy, vz
                                   double dt, double sigma_pos, double sigma_cb,
                                   double sigma_vel, double velocity_guide_alpha,
                                   bool velocity_kf, double velocity_process_noise,
                                   int N, unsigned long long seed, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, step, &state);

    double alpha = fmin(1.0, fmax(0.0, velocity_guide_alpha));
    double vxi = vx[tid];
    double vyi = vy[tid];
    double vzi = vz[tid];

    vxi = (1.0 - alpha) * vxi + alpha * vel_guide[0];
    vyi = (1.0 - alpha) * vyi + alpha * vel_guide[1];
    vzi = (1.0 - alpha) * vzi + alpha * vel_guide[2];
    if (!velocity_kf && sigma_vel > 0.0) {
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
    if (velocity_kf) {
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

// --- DD pseudorange weight kernel ---
// Double-differenced pseudorange eliminates receiver clock bias from both rover
// and base. The expected DD range is:
//   dd_expected = range_to_k - range_to_ref - base_range_k + base_range_ref
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
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double x = px[tid];
    double y = py[tid];
    double z = pz[tid];

    double inv_sigma2 = 1.0 / (sigma_pr * sigma_pr);
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

            double dx_ref = x - s_ref[d * 3 + 0];
            double dy_ref = y - s_ref[d * 3 + 1];
            double dz_ref = z - s_ref[d * 3 + 2];
            double r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);

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

        double dx_ref = x - s_ref[d * 3 + 0];
        double dy_ref = y - s_ref[d * 3 + 1];
        double dz_ref = z - s_ref[d * 3 + 2];
        double r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);

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
    double* s_wl      = s_data + n_dd * 10;           // n_dd

    int total_shared = n_dd * 11;
    for (int i = threadIdx.x; i < total_shared; i += blockDim.x) {
        s_data[i] = dd_data[i];
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    double x = px[tid];
    double y = py[tid];
    double z = pz[tid];

    double inv_sigma2 = 1.0 / (sigma_cycles * sigma_cycles);
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

            double dx_ref = x - s_ref[d * 3 + 0];
            double dy_ref = y - s_ref[d * 3 + 1];
            double dz_ref = z - s_ref[d * 3 + 2];
            double r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);

            double inv_wl = 1.0 / s_wl[d];
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

        double dx_ref = x - s_ref[d * 3 + 0];
        double dy_ref = y - s_ref[d * 3 + 1];
        double dz_ref = z - s_ref[d * 3 + 2];
        double r_ref = sqrt(dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref);

        double inv_wl = 1.0 / s_wl[d];

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
    double mui[3] = {vx[tid], vy[tid], vz[tid]};

    int cov_off = tid * 9;
    double P[9];
    for (int k = 0; k < 9; k++) {
        P[k] = vcov[cov_off + k];
    }

    double h_mean[3] = {0.0, 0.0, 0.0};
    double y_mean = 0.0;
    double weight_sum = 0.0;
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
    double sigma2 = sigma_mps * sigma_mps;
    for (int s = 0; s < n_sat; s++) {
        double obs_weight = fmax(s_weights[s], 0.0);
        if (obs_weight <= 0.0) continue;
        double dx = s_sat[s * 3 + 0] - x;
        double dy = s_sat[s * 3 + 1] - y;
        double dz = s_sat[s * 3 + 2] - z;
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1.0) continue;
        double lx = dx / r;
        double ly = dy / r;
        double lz = dz / r;
        double h[3] = {-lx - h_mean[0], -ly - h_mean[1], -lz - h_mean[2]};
        double sv_radial =
            s_sat_vel[s * 3 + 0] * lx +
            s_sat_vel[s * 3 + 1] * ly +
            s_sat_vel[s * 3 + 2] * lz;
        double y_centered = (-s_doppler[s] * wavelength_m - sv_radial) - y_mean;
        double ph0 = P[0] * h[0] + P[1] * h[1] + P[2] * h[2];
        double ph1 = P[3] * h[0] + P[4] * h[1] + P[5] * h[2];
        double ph2 = P[6] * h[0] + P[7] * h[1] + P[8] * h[2];
        double meas_var = sigma2 / obs_weight;
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

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
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

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_wd2[tid] += s_wd2[tid + s];
            s_sw[tid] += s_sw[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sum_wd2[blockIdx.x] = s_wd2[0];
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

// --- Megopolis kernel ---
__global__ void pfd_megopolis_kernel(double* px_a, double* py_a, double* pz_a,
                                    double* vx_a, double* vy_a, double* vz_a,
                                    double* vcov_a,
                                    double* pcb_a,
                                    double* px_b, double* py_b, double* pz_b,
                                    double* vx_b, double* vy_b, double* vz_b,
                                    double* vcov_b,
                                    double* pcb_b,
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
    const double* src_vx = (src_buf == 0) ? vx_a : vx_b;
    const double* src_vy = (src_buf == 0) ? vy_a : vy_b;
    const double* src_vz = (src_buf == 0) ? vz_a : vz_b;
    const double* src_vcov = (src_buf == 0) ? vcov_a : vcov_b;
    const double* src_pcb = (src_buf == 0) ? pcb_a : pcb_b;
    double* dst_px = (src_buf == 0) ? px_b : px_a;
    double* dst_py = (src_buf == 0) ? py_b : py_a;
    double* dst_pz = (src_buf == 0) ? pz_b : pz_a;
    double* dst_vx = (src_buf == 0) ? vx_b : vx_a;
    double* dst_vy = (src_buf == 0) ? vy_b : vy_a;
    double* dst_vz = (src_buf == 0) ? vz_b : vz_a;
    double* dst_vcov = (src_buf == 0) ? vcov_b : vcov_a;
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
        dst_vx[tid] = src_vx[j];
        dst_vy[tid] = src_vy[j];
        dst_vz[tid] = src_vz[j];
        int dst_cov = tid * 9;
        int src_cov = j * 9;
        for (int k = 0; k < 9; k++) {
            dst_vcov[dst_cov + k] = src_vcov[src_cov + k];
        }
        dst_pcb[tid] = src_pcb[j];
    } else {
        dst_px[tid] = src_px[tid];
        dst_py[tid] = src_py[tid];
        dst_pz[tid] = src_pz[tid];
        dst_vx[tid] = src_vx[tid];
        dst_vy[tid] = src_vy[tid];
        dst_vz[tid] = src_vz[tid];
        int cov_off = tid * 9;
        for (int k = 0; k < 9; k++) {
            dst_vcov[cov_off + k] = src_vcov[cov_off + k];
        }
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

    // Systematic resampling buffers
    CUDA_CHECK(cudaMalloc(&state->d_weights_norm, sz));
    CUDA_CHECK(cudaMalloc(&state->d_cdf, sz));
    CUDA_CHECK(cudaMalloc(&state->d_resample_ancestor, (size_t)n_particles * sizeof(int)));

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
    CUDA_CHECK(cudaMallocHost(&state->h_reduction_pinned, grid * 6 * sizeof(double)));

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
    cudaFree(state->d_weights_norm);
    cudaFree(state->d_cdf);
    cudaFree(state->d_resample_ancestor);
    cudaFree(state->d_vel);

    // Free persistent satellite device buffers
    cudaFree(state->d_sat_ecef);
    cudaFree(state->d_pseudoranges);
    cudaFree(state->d_weights_sat);

    // Free pinned host memory
    cudaFreeHost(state->h_sat_pinned);
    cudaFreeHost(state->h_result_pinned);
    cudaFreeHost(state->h_reduction_pinned);

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

    // Copy velocity to pinned memory, then async transfer to device
    // h_result_pinned has 4 doubles; we reuse first 3 for velocity (non-overlapping use)
    state->h_result_pinned[0] = vx;
    state->h_result_pinned[1] = vy;
    state->h_result_pinned[2] = vz;
    CUDA_CHECK(cudaMemcpyAsync(state->d_vel, state->h_result_pinned,
                               3 * sizeof(double), cudaMemcpyHostToDevice, state->stream));

    pfd_predict_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz, state->d_vcov, state->d_pcb,
        state->d_vel,
        dt, sigma_pos, sigma_cb, sigma_vel, velocity_guide_alpha,
        velocity_kf, velocity_process_noise,
        N, seed, step);
    CUDA_CHECK_LAST();
}

// ============================================================
// Weight
// ============================================================

void pf_device_weight(PFDeviceState* state,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    int n_sat, double sigma_pr, double nu,
    double per_particle_nlos_threshold_m,
    bool per_particle_huber,
    double per_particle_huber_k) {

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
        N, n_sat, sigma_pr, nu, per_particle_nlos_threshold_m,
        per_particle_huber, per_particle_huber_k);
    CUDA_CHECK_LAST();
}

void pf_device_weight_dd_pseudorange(PFDeviceState* state,
    const double* sat_ecef_k, const double* ref_ecef,
    const double* dd_pseudorange, const double* base_range_k,
    const double* base_range_ref, const double* weights_dd,
    int n_dd, double sigma_pr,
    double per_particle_nlos_threshold_m,
    bool per_particle_huber,
    double per_particle_huber_k) {

    int N = state->n_particles;
    int grid = state->grid_size;

    // Pack all DD data into a contiguous buffer for a single H2D transfer.
    // Layout: [sat_k: n_dd*3][ref: n_dd*3][dd_pr: n_dd][base_range_k: n_dd][base_range_ref: n_dd][weights: n_dd]
    int total_doubles = n_dd * 10;
    size_t total_bytes = (size_t)total_doubles * sizeof(double);

    double* h_buf = (double*)malloc(total_bytes);
    if (!h_buf) return;

    int off = 0;
    memcpy(h_buf + off, sat_ecef_k, n_dd * 3 * sizeof(double)); off += n_dd * 3;
    memcpy(h_buf + off, ref_ecef, n_dd * 3 * sizeof(double)); off += n_dd * 3;
    memcpy(h_buf + off, dd_pseudorange, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, base_range_k, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, base_range_ref, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, weights_dd, n_dd * sizeof(double)); off += n_dd;

    double* d_dd_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dd_data, total_bytes));
    CUDA_CHECK(cudaMemcpyAsync(
        d_dd_data, h_buf, total_bytes, cudaMemcpyHostToDevice, state->stream));

    size_t smem = (size_t)total_doubles * sizeof(double);
    pfd_weight_dd_pseudorange_kernel<<<grid, BLOCK_SIZE, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        d_dd_data, state->d_log_weights,
        N, n_dd, sigma_pr, per_particle_nlos_threshold_m,
        per_particle_huber, per_particle_huber_k);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    CUDA_CHECK(cudaFree(d_dd_data));
    free(h_buf);
}

void pf_device_weight_gmm(PFDeviceState* state,
    const double* sat_ecef, const double* pseudoranges,
    const double* weights_sat,
    int n_sat, double sigma_pr,
    double w_los, double mu_nlos, double sigma_nlos) {

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

    // Async H2D transfer on the stream
    CUDA_CHECK(cudaMemcpyAsync(state->d_sat_ecef, h_sat,
                               sz_sat, cudaMemcpyHostToDevice, state->stream));
    CUDA_CHECK(cudaMemcpyAsync(state->d_pseudoranges, h_sat + n_sat * 3,
                               sz_obs, cudaMemcpyHostToDevice, state->stream));
    CUDA_CHECK(cudaMemcpyAsync(state->d_weights_sat, h_sat + n_sat * 4,
                               sz_obs, cudaMemcpyHostToDevice, state->stream));

    // Launch GMM weight kernel on the same stream
    size_t smem_weight = (size_t)n_sat * 5 * sizeof(double);
    pfd_weight_gmm_kernel<<<grid, BLOCK_SIZE, smem_weight, state->stream>>>(
        state->d_px, state->d_py, state->d_pz, state->d_pcb,
        state->d_sat_ecef, state->d_pseudoranges, state->d_weights_sat,
        state->d_log_weights,
        N, n_sat, sigma_pr, w_los, mu_nlos, sigma_nlos);
    CUDA_CHECK_LAST();
}

// ============================================================
// Position-domain update (soft constraint from external position)
// ============================================================

void pf_device_position_update(PFDeviceState* state,
    double ref_x, double ref_y, double ref_z, double sigma_pos) {
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

    CUDA_CHECK(cudaStreamSynchronize(state->stream));

    double* h_sum_w = state->h_reduction_pinned + 0;
    double* h_sum_w2 = state->h_reduction_pinned + grid;
    double* h_max_lw = state->h_reduction_pinned + 2 * grid;
    CUDA_CHECK(cudaMemcpy(h_sum_w, state->d_partial_a, grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_w2, state->d_partial_b, grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max_lw, state->d_partial_c, grid * sizeof(double), cudaMemcpyDeviceToHost));

    double global_max = h_max_lw[0];
    for (int i = 1; i < grid; i++) {
        global_max = std::max(global_max, h_max_lw[i]);
    }

    double total_w = 0.0, total_w2 = 0.0;
    for (int i = 0; i < grid; i++) {
        double correction = exp(h_max_lw[i] - global_max);
        total_w += h_sum_w[i] * correction;
        total_w2 += h_sum_w2[i] * correction * correction;
    }

    double ess = (total_w * total_w) / total_w2;
    return ess;
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

    CUDA_CHECK(cudaStreamSynchronize(state->stream));

    double* h_sum_wd2 = state->h_reduction_pinned + 0;
    double* h_sum_w = state->h_reduction_pinned + grid;
    double* h_max_lw = state->h_reduction_pinned + 2 * grid;
    CUDA_CHECK(cudaMemcpy(h_sum_wd2, state->d_partial_a, grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_w, state->d_partial_b, grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max_lw, state->d_partial_c, grid * sizeof(double), cudaMemcpyDeviceToHost));

    double global_max = h_max_lw[0];
    for (int i = 1; i < grid; i++) {
        global_max = std::max(global_max, h_max_lw[i]);
    }

    double total_wd2 = 0.0;
    double total_w = 0.0;
    for (int i = 0; i < grid; i++) {
        double correction = exp(h_max_lw[i] - global_max);
        total_wd2 += h_sum_wd2[i] * correction;
        total_w += h_sum_w[i] * correction;
    }

    if (!(total_w > 0.0) || !std::isfinite(total_wd2)) {
        return 0.0;
    }
    return sqrt(std::max(total_wd2 / total_w, 0.0));
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
    double* h_block_max = state->h_reduction_pinned + 0;
    CUDA_CHECK(cudaMemcpy(h_block_max, state->d_partial_c, grid * sizeof(double), cudaMemcpyDeviceToHost));
    double max_lw = h_block_max[0];
    for (int i = 1; i < grid; i++) max_lw = std::max(max_lw, h_block_max[i]);

    // Step 2: exp(lw - max) into weights_norm
    pfd_exp_shift_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
        state->d_log_weights, state->d_weights_norm, max_lw, N);

    // Step 3: Compute sum of weights
    pfd_sum_reduce_kernel<<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), state->stream>>>(
        state->d_weights_norm, state->d_partial_b, N);

    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    double* h_block_sums = state->h_reduction_pinned + grid;
    CUDA_CHECK(cudaMemcpy(h_block_sums, state->d_partial_b, grid * sizeof(double), cudaMemcpyDeviceToHost));
    double sum_w = 0;
    for (int i = 0; i < grid; i++) sum_w += h_block_sums[i];

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
    int N = state->n_particles;
    int grid = state->grid_size;

    // Use primary arrays as buffer A and tmp arrays as buffer B.
    // Megopolis kernel uses double-buffering directly on device memory

    for (int iter = 0; iter < n_iterations; iter++) {
        int src_buf = iter % 2;
        pfd_megopolis_kernel<<<grid, BLOCK_SIZE, 0, state->stream>>>(
            state->d_px, state->d_py, state->d_pz,
            state->d_vx, state->d_vy, state->d_vz, state->d_vcov, state->d_pcb,
            state->d_px_tmp, state->d_py_tmp, state->d_pz_tmp,
            state->d_vx_tmp, state->d_vy_tmp, state->d_vz_tmp, state->d_vcov_tmp, state->d_pcb_tmp,
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
        std::swap(state->d_vx, state->d_vx_tmp);
        std::swap(state->d_vy, state->d_vy_tmp);
        std::swap(state->d_vz, state->d_vz_tmp);
        std::swap(state->d_vcov, state->d_vcov_tmp);
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

    double* h_partial = state->h_reduction_pinned + 0;
    double* h_sum_w = state->h_reduction_pinned + 4 * grid;
    double* h_max_lw = state->h_reduction_pinned + 5 * grid;
    CUDA_CHECK(cudaMemcpy(h_partial, state->d_partial_a, grid * 4 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sum_w, state->d_partial_b, grid * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max_lw, state->d_partial_c, grid * sizeof(double), cudaMemcpyDeviceToHost));

    double global_max = h_max_lw[0];
    for (int i = 1; i < grid; i++) {
        global_max = std::max(global_max, h_max_lw[i]);
    }

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
}

// ============================================================
// Get particles (D2H only when explicitly requested)
// ============================================================

void pf_device_get_particles(const PFDeviceState* state, double* output) {
    int N = state->n_particles;

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

void pf_device_get_log_weights(const PFDeviceState* state, double* output) {
    int N = state->n_particles;
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    CUDA_CHECK(cudaMemcpy(output, state->d_log_weights,
                          (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));
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

    int N = state->n_particles;
    int grid = state->grid_size;

    // Satellite data is small: n_sat * 5 doubles typically < 1KB
    // Reuse persistent device buffers (d_pseudoranges holds carrier_phase here)
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

    // Stage satellite data into contiguous pinned buffer: [sat_ecef | carrier_phase | weights]
    double* h_sat = state->h_sat_pinned;
    memcpy(h_sat, sat_ecef, sz_sat);
    memcpy(h_sat + n_sat * 3, carrier_phase, sz_obs);
    memcpy(h_sat + n_sat * 4, weights_sat, sz_obs);

    // Async H2D transfer on the stream
    CUDA_CHECK(cudaMemcpyAsync(state->d_sat_ecef, h_sat,
                               sz_sat, cudaMemcpyHostToDevice, state->stream));
    CUDA_CHECK(cudaMemcpyAsync(state->d_pseudoranges, h_sat + n_sat * 3,
                               sz_obs, cudaMemcpyHostToDevice, state->stream));
    CUDA_CHECK(cudaMemcpyAsync(state->d_weights_sat, h_sat + n_sat * 4,
                               sz_obs, cudaMemcpyHostToDevice, state->stream));

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

    int N = state->n_particles;
    int grid = state->grid_size;

    // Pack all DD data into a contiguous buffer for a single H2D transfer.
    // Layout: [sat_k: n_dd*3][ref: n_dd*3][dd_carrier: n_dd][base_range_k: n_dd][base_range_ref: n_dd][weights: n_dd][wavelengths: n_dd]
    int total_doubles = n_dd * 11;
    size_t total_bytes = (size_t)total_doubles * sizeof(double);

    // Check if we need to reallocate (reuse pinned capacity; need total_doubles <= pinned_capacity * 5)
    // For safety, just use malloc/free for the staging buffer since n_dd is small (~10)
    // and this is called once per epoch.
    double* h_buf = (double*)malloc(total_bytes);
    if (!h_buf) return;

    // Pack
    int off = 0;
    memcpy(h_buf + off, sat_ecef_k, n_dd * 3 * sizeof(double)); off += n_dd * 3;
    memcpy(h_buf + off, ref_ecef, n_dd * 3 * sizeof(double)); off += n_dd * 3;
    memcpy(h_buf + off, dd_carrier, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, base_range_k, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, base_range_ref, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, weights_dd, n_dd * sizeof(double)); off += n_dd;
    memcpy(h_buf + off, wavelengths_m, n_dd * sizeof(double)); off += n_dd;

    // Allocate device buffer for DD data (small, ~hundreds of bytes)
    double* d_dd_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dd_data, total_bytes));

    // Synchronous copy (data is tiny)
    CUDA_CHECK(cudaMemcpyAsync(d_dd_data, h_buf, total_bytes,
                                cudaMemcpyHostToDevice, state->stream));

    // Launch DD-AFV kernel
    size_t smem = (size_t)total_doubles * sizeof(double);
    pfd_weight_dd_carrier_afv_kernel<<<grid, BLOCK_SIZE, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        d_dd_data, state->d_log_weights,
        N, n_dd, sigma_cycles, per_particle_nlos_threshold_cycles,
        per_particle_huber, per_particle_huber_k);
    CUDA_CHECK_LAST();

    // Free device buffer after kernel completes
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    CUDA_CHECK(cudaFree(d_dd_data));
    free(h_buf);
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

    if (n_sat <= 0 || sigma_mps <= 0.0 || wavelength_m <= 0.0) {
        return;
    }

    int N = state->n_particles;
    int grid = state->grid_size;

    int total_doubles = n_sat * 8;
    size_t total_bytes = (size_t)total_doubles * sizeof(double);
    double* h_buf = (double*)malloc(total_bytes);
    if (!h_buf) return;

    int off = 0;
    memcpy(h_buf + off, sat_ecef, n_sat * 3 * sizeof(double)); off += n_sat * 3;
    memcpy(h_buf + off, sat_vel, n_sat * 3 * sizeof(double)); off += n_sat * 3;
    memcpy(h_buf + off, doppler_hz, n_sat * sizeof(double)); off += n_sat;
    memcpy(h_buf + off, weights_sat, n_sat * sizeof(double)); off += n_sat;

    double* d_doppler_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_doppler_data, total_bytes));
    CUDA_CHECK(cudaMemcpyAsync(
        d_doppler_data, h_buf, total_bytes, cudaMemcpyHostToDevice, state->stream));

    size_t smem = (size_t)total_doubles * sizeof(double);
    pfd_weight_doppler_kernel<<<grid, BLOCK_SIZE, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz,
        d_doppler_data, state->d_log_weights,
        N, n_sat, wavelength_m, sigma_mps,
        velocity_update_gain, max_velocity_update_mps);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    CUDA_CHECK(cudaFree(d_doppler_data));
    free(h_buf);
}

void pf_device_doppler_kf_update(PFDeviceState* state,
    const double* sat_ecef, const double* sat_vel,
    const double* doppler_hz, const double* weights_sat,
    int n_sat, double wavelength_m,
    double sigma_mps) {

    if (n_sat <= 0 || sigma_mps <= 0.0 || wavelength_m <= 0.0) {
        return;
    }

    int N = state->n_particles;
    int grid = state->grid_size;

    int total_doubles = n_sat * 8;
    size_t total_bytes = (size_t)total_doubles * sizeof(double);
    double* h_buf = (double*)malloc(total_bytes);
    if (!h_buf) return;

    int off = 0;
    memcpy(h_buf + off, sat_ecef, n_sat * 3 * sizeof(double)); off += n_sat * 3;
    memcpy(h_buf + off, sat_vel, n_sat * 3 * sizeof(double)); off += n_sat * 3;
    memcpy(h_buf + off, doppler_hz, n_sat * sizeof(double)); off += n_sat;
    memcpy(h_buf + off, weights_sat, n_sat * sizeof(double)); off += n_sat;

    double* d_doppler_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_doppler_data, total_bytes));
    CUDA_CHECK(cudaMemcpyAsync(
        d_doppler_data, h_buf, total_bytes, cudaMemcpyHostToDevice, state->stream));

    size_t smem = (size_t)total_doubles * sizeof(double);
    pfd_doppler_kf_update_kernel<<<grid, BLOCK_SIZE, smem, state->stream>>>(
        state->d_px, state->d_py, state->d_pz,
        state->d_vx, state->d_vy, state->d_vz, state->d_vcov,
        d_doppler_data, state->d_log_weights,
        N, n_sat, wavelength_m, sigma_mps);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    CUDA_CHECK(cudaFree(d_doppler_data));
    free(h_buf);
}

// ============================================================
// Explicit synchronization
// ============================================================

void pf_device_sync(PFDeviceState* state) {
    if (!state || !state->allocated) return;
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
}

}  // namespace gnss_gpu
