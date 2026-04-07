#include "gnss_gpu/fgo.h"
#include "gnss_gpu/cuda_check.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace gnss_gpu {

namespace {

constexpr double kC = 299792458.0;
constexpr double kOmegaE = 7.2921151467e-5;
constexpr int kMaxClock = 4;
constexpr double kDiagJitter = 1e-3;

__device__ __host__ void fill_hc_int(int nc, int sk, double* hc) {
  for (int i = 0; i < nc; i++) hc[i] = 0.0;
  hc[0] = 1.0;
  if (sk > 0 && sk < nc) hc[sk] = 1.0;
}

__global__ void fgo_assemble_pseudorange(
    int n_epoch, int n_sat, int nc, int ss, int n_state,
    const double* sat_ecef,
    const double* pseudorange,
    const double* weights,
    const int* sys_kind,
    const double* state,
    double* H,
    double* g) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_epoch) return;

  const double x = state[t * ss + 0];
  const double y = state[t * ss + 1];
  const double z = state[t * ss + 2];
  const double* cptr = state + t * ss + 3;

  double Hloc[kMaxClock + 3][kMaxClock + 3] = {};
  double gloc[kMaxClock + 3] = {};

  const double* my_sat = sat_ecef + (size_t)t * n_sat * 3;
  const double* my_pr = pseudorange + (size_t)t * n_sat;
  const double* my_w = weights + (size_t)t * n_sat;

  for (int s = 0; s < n_sat; s++) {
    double w = my_w[s];
    if (w <= 0.0) continue;

    int sk = sys_kind ? sys_kind[t * n_sat + s] : 0;
    if (sk < 0 || sk >= nc) continue;

    double sx = my_sat[s * 3 + 0];
    double sy = my_sat[s * 3 + 1];
    double sz = my_sat[s * 3 + 2];

    double dx0 = x - sx, dy0 = y - sy, dz0 = z - sz;
    double r0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);

    double transit = r0 / kC;
    double theta = kOmegaE * transit;
    double sx_rot = sx * cos(theta) + sy * sin(theta);
    double sy_rot = -sx * sin(theta) + sy * cos(theta);

    double dx = x - sx_rot, dy_v = y - sy_rot, dz = z - sz;
    double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
    if (r < 1e-6) continue;

    double hc[kMaxClock];
    fill_hc_int(nc, sk, hc);
    double clk = 0.0;
    for (int k = 0; k < nc; k++) clk += hc[k] * cptr[k];

    double pred = r + clk;
    double res = my_pr[s] - pred;

    double J[3 + kMaxClock];
    J[0] = dx / r;
    J[1] = dy_v / r;
    J[2] = dz / r;
    for (int k = 0; k < nc; k++) J[3 + k] = hc[k];

    double Jr = res * w;
    for (int a = 0; a < ss; a++) {
      gloc[a] += J[a] * Jr;
      for (int b = 0; b < ss; b++) Hloc[a][b] += w * J[a] * J[b];
    }
  }

  int o = ss * t;
  for (int a = 0; a < ss; a++) {
    for (int b = 0; b < ss; b++) {
      H[(size_t)(o + a) * n_state + (o + b)] += Hloc[a][b];
    }
    g[o + a] += gloc[a];
  }
}

void add_motion_rw_host(int n_epoch, int ss, int n_state, double w_motion, const double* state,
                        double* H, double* g) {
  if (w_motion <= 0.0) return;
  for (int t = 0; t < n_epoch - 1; t++) {
    int o0 = ss * t;
    int o1 = ss * (t + 1);
    for (int i = 0; i < 3; i++) {
      double d01 = state[o0 + i] - state[o1 + i];
      g[o0 + i] += w_motion * d01;
      g[o1 + i] += w_motion * (-d01);
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        double id = (i == j) ? w_motion : 0.0;
        double neg = (i == j) ? -w_motion : 0.0;
        H[(size_t)(o0 + i) * n_state + (o0 + j)] += id;
        H[(size_t)(o1 + i) * n_state + (o1 + j)] += id;
        H[(size_t)(o0 + i) * n_state + (o1 + j)] += neg;
        H[(size_t)(o1 + i) * n_state + (o0 + j)] += neg;
      }
    }
  }
}

bool cholesky_decompose_inplace(int n, double* A) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = A[(size_t)i * n + j];
      for (int k = 0; k < j; k++) sum -= A[(size_t)i * n + k] * A[(size_t)j * n + k];
      if (i == j) {
        if (sum <= 1e-18) return false;
        A[(size_t)i * n + j] = sqrt(sum);
      } else {
        A[(size_t)i * n + j] = sum / A[(size_t)j * n + j];
      }
    }
    for (int j = i + 1; j < n; j++) A[(size_t)i * n + j] = 0.0;
  }
  return true;
}

void cholesky_solve_lower(int n, const double* L, const double* b, double* x) {
  for (int i = 0; i < n; i++) {
    double sum = b[i];
    for (int k = 0; k < i; k++) sum -= L[(size_t)i * n + k] * x[k];
    x[i] = sum / L[(size_t)i * n + i];
  }
  for (int i = n - 1; i >= 0; i--) {
    double sum = x[i];
    for (int k = i + 1; k < n; k++) sum -= L[(size_t)k * n + i] * x[k];
    x[i] = sum / L[(size_t)i * n + i];
  }
}

double pr_cost_host(
    int n_epoch, int n_sat, int nc, int ss,
    const double* sat_ecef,
    const double* pseudorange,
    const double* weights,
    const int* sys_kind_host,
    const double* state,
    double huber_k) {
  double e = 0.0;
  for (int t = 0; t < n_epoch; t++) {
    const double x = state[t * ss + 0], y = state[t * ss + 1], z = state[t * ss + 2];
    const double* cptr = state + t * ss + 3;
    const double* my_sat = sat_ecef + (size_t)t * n_sat * 3;
    const double* my_pr = pseudorange + (size_t)t * n_sat;
    const double* my_w = weights + (size_t)t * n_sat;
    for (int s = 0; s < n_sat; s++) {
      double w = my_w[s];
      if (w <= 0.0) continue;
      int sk = sys_kind_host[t * n_sat + s];
      if (sk < 0 || sk >= nc) continue;
      double sx = my_sat[s * 3 + 0], sy = my_sat[s * 3 + 1], sz = my_sat[s * 3 + 2];
      double dx0 = x - sx, dy0 = y - sy, dz0 = z - sz;
      double r0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
      double transit = r0 / kC;
      double theta = kOmegaE * transit;
      double sx_rot = sx * cos(theta) + sy * sin(theta);
      double sy_rot = -sx * sin(theta) + sy * cos(theta);
      double dx = x - sx_rot, dy_v = y - sy_rot, dz = z - sz;
      double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      if (r < 1e-6) continue;
      double hc[kMaxClock];
      fill_hc_int(nc, sk, hc);
      double clk = 0.0;
      for (int k = 0; k < nc; k++) clk += hc[k] * cptr[k];
      double res = my_pr[s] - (r + clk);
      if (huber_k <= 0.0) {
        e += 0.5 * w * res * res;
      } else {
        double z_m = sqrt(w) * std::fabs(res);
        if (z_m <= huber_k)
          e += 0.5 * z_m * z_m;
        else
          e += huber_k * z_m - 0.5 * huber_k * huber_k;
      }
    }
  }
  return e;
}

void effective_pr_weights_huber_host(
    int n_epoch, int n_sat, int nc, int ss,
    const double* sat_ecef,
    const double* pseudorange,
    const double* weights,
    const int* sys_kind_host,
    const double* state,
    double huber_k,
    double* eff_w_out) {
  if (huber_k <= 0.0) {
    std::memcpy(eff_w_out, weights, (size_t)n_epoch * n_sat * sizeof(double));
    return;
  }
  for (int t = 0; t < n_epoch; t++) {
    const double x = state[t * ss + 0], y = state[t * ss + 1], z = state[t * ss + 2];
    const double* cptr = state + t * ss + 3;
    const double* my_sat = sat_ecef + (size_t)t * n_sat * 3;
    const double* my_pr = pseudorange + (size_t)t * n_sat;
    const double* my_w = weights + (size_t)t * n_sat;
    for (int s = 0; s < n_sat; s++) {
      double w = my_w[s];
      size_t idx = (size_t)t * n_sat + s;
      if (w <= 0.0) {
        eff_w_out[idx] = w;
        continue;
      }
      int sk = sys_kind_host[t * n_sat + s];
      if (sk < 0 || sk >= nc) {
        eff_w_out[idx] = 0.0;
        continue;
      }
      double sx = my_sat[s * 3 + 0], sy = my_sat[s * 3 + 1], sz = my_sat[s * 3 + 2];
      double dx0 = x - sx, dy0 = y - sy, dz0 = z - sz;
      double r0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
      double transit = r0 / kC;
      double theta = kOmegaE * transit;
      double sx_rot = sx * cos(theta) + sy * sin(theta);
      double sy_rot = -sx * sin(theta) + sy * cos(theta);
      double dx = x - sx_rot, dy_v = y - sy_rot, dz = z - sz;
      double r_geom = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      if (r_geom < 1e-6) {
        eff_w_out[idx] = 0.0;
        continue;
      }
      double hc[kMaxClock];
      fill_hc_int(nc, sk, hc);
      double clk = 0.0;
      for (int k = 0; k < nc; k++) clk += hc[k] * cptr[k];
      double res = my_pr[s] - (r_geom + clk);
      double z_m = sqrt(w) * std::fabs(res);
      double v = (z_m <= huber_k) ? 1.0 : (huber_k / z_m);
      eff_w_out[idx] = w * v;
    }
  }
}

double motion_cost_host(int n_epoch, int ss, double w_motion, const double* state) {
  if (w_motion <= 0.0) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch - 1; t++) {
    int o0 = ss * t, o1 = ss * (t + 1);
    for (int i = 0; i < 3; i++) {
      double d = state[o0 + i] - state[o1 + i];
      e += 0.5 * w_motion * d * d;
    }
  }
  return e;
}

double compute_pr_mse_host(
    int n_epoch, int n_sat, int nc, int ss,
    const double* sat_ecef,
    const double* pseudorange,
    const double* weights,
    const int* sys_kind_host,
    const double* state) {
  double sse = 0.0;
  int cnt = 0;
  for (int t = 0; t < n_epoch; t++) {
    double x = state[t * ss + 0], y = state[t * ss + 1], z = state[t * ss + 2];
    const double* cptr = state + t * ss + 3;
    const double* my_sat = sat_ecef + (size_t)t * n_sat * 3;
    const double* my_pr = pseudorange + (size_t)t * n_sat;
    const double* my_w = weights + (size_t)t * n_sat;
    for (int s = 0; s < n_sat; s++) {
      double w = my_w[s];
      if (w <= 0.0) continue;
      int sk = sys_kind_host[t * n_sat + s];
      if (sk < 0 || sk >= nc) continue;
      double sx = my_sat[s * 3 + 0], sy = my_sat[s * 3 + 1], sz = my_sat[s * 3 + 2];
      double dx0 = x - sx, dy0 = y - sy, dz0 = z - sz;
      double r0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
      double transit = r0 / kC;
      double theta = kOmegaE * transit;
      double sx_rot = sx * cos(theta) + sy * sin(theta);
      double sy_rot = -sx * sin(theta) + sy * cos(theta);
      double dx = x - sx_rot, dy_v = y - sy_rot, dz = z - sz;
      double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      if (r < 1e-6) continue;
      double hc[kMaxClock];
      fill_hc_int(nc, sk, hc);
      double clk = 0.0;
      for (int k = 0; k < nc; k++) clk += hc[k] * cptr[k];
      double res = my_pr[s] - (r + clk);
      sse += w * res * res;
      cnt++;
    }
  }
  return cnt > 0 ? sse / cnt : 0.0;
}

}  // namespace

int fgo_gnss_lm(const double* sat_ecef,
                const double* pseudorange,
                const double* weights,
                const std::int32_t* sys_kind,
                int n_clock,
                double* state_io,
                int n_epoch,
                int n_sat,
                double motion_sigma_m,
                int max_iter,
                double tol,
                double huber_k,
                int enable_line_search,
                double* out_mse_pr) {
  if (n_epoch < 1 || n_sat < 4 || !sat_ecef || !pseudorange || !weights || !state_io) return -1;
  if (n_clock < 1 || n_clock > kMaxClock) return -1;

  const int ss = 3 + n_clock;
  const int n_state = ss * n_epoch;
  if (n_state > 8192) return -1;

  std::vector<int> sys_buf((size_t)n_epoch * n_sat, 0);
  if (sys_kind != nullptr) {
    for (size_t i = 0; i < sys_buf.size(); i++) {
      sys_buf[i] = static_cast<int>(sys_kind[i]);
    }
  }
  const int* sys_host = sys_buf.data();

  size_t sz_state = (size_t)n_state * sizeof(double);
  size_t sz_sat = (size_t)n_epoch * n_sat * 3 * sizeof(double);
  size_t sz_ws = (size_t)n_epoch * n_sat * sizeof(double);
  size_t sz_H = (size_t)n_state * n_state * sizeof(double);
  size_t sz_sys = (size_t)n_epoch * n_sat * sizeof(int);

  double *d_state = nullptr, *d_sat = nullptr, *d_pr = nullptr, *d_w = nullptr;
  double *d_H = nullptr, *d_g = nullptr;
  int* d_sys = nullptr;

  CUDA_CHECK(cudaMalloc(&d_state, sz_state));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_pr, sz_ws));
  CUDA_CHECK(cudaMalloc(&d_w, sz_ws));
  CUDA_CHECK(cudaMalloc(&d_H, sz_H));
  CUDA_CHECK(cudaMalloc(&d_g, sz_state));
  CUDA_CHECK(cudaMalloc(&d_sys, sz_sys));
  CUDA_CHECK(cudaMemcpy(d_sys, sys_host, sz_sys, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(d_state, state_io, sz_state, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pr, pseudorange, sz_ws, cudaMemcpyHostToDevice));

  double* h_H = (double*)std::malloc(sz_H);
  double* h_g = (double*)std::malloc(sz_state);
  double* h_delta = (double*)std::malloc(sz_state);
  double* h_work = (double*)std::malloc(sz_H);
  double* trial = (double*)std::malloc(sz_state);
  double* h_eff_w = (double*)std::malloc(sz_ws);
  if (!h_H || !h_g || !h_delta || !h_work || !trial || !h_eff_w) {
    if (h_H) std::free(h_H);
    if (h_g) std::free(h_g);
    if (h_delta) std::free(h_delta);
    if (h_work) std::free(h_work);
    if (trial) std::free(trial);
    if (h_eff_w) std::free(h_eff_w);
    cudaFree(d_state); cudaFree(d_sat); cudaFree(d_pr); cudaFree(d_w);
    cudaFree(d_H); cudaFree(d_g); cudaFree(d_sys);
    return -1;
  }

  double w_motion = 0.0;
  if (motion_sigma_m > 0.0) w_motion = 1.0 / (motion_sigma_m * motion_sigma_m);

  int total_iters = 0;
  bool ok = false;
  const int block = 256;

  for (int it = 0; it < max_iter; it++) {
    effective_pr_weights_huber_host(
        n_epoch, n_sat, n_clock, ss, sat_ecef, pseudorange, weights, sys_host, state_io,
        huber_k, h_eff_w);
    CUDA_CHECK(cudaMemcpy(d_w, h_eff_w, sz_ws, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_H, 0, sz_H));
    CUDA_CHECK(cudaMemset(d_g, 0, sz_state));

    int grid_pr = (n_epoch + block - 1) / block;
    fgo_assemble_pseudorange<<<grid_pr, block>>>(
        n_epoch, n_sat, n_clock, ss, n_state, d_sat, d_pr, d_w, d_sys, d_state, d_H, d_g);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(h_H, d_H, sz_H, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_g, d_g, sz_state, cudaMemcpyDeviceToHost));

    add_motion_rw_host(n_epoch, ss, n_state, w_motion, state_io, h_H, h_g);

    double cost_before =
        pr_cost_host(n_epoch, n_sat, n_clock, ss, sat_ecef, pseudorange, weights, sys_host, state_io,
                     huber_k) +
        motion_cost_host(n_epoch, ss, w_motion, state_io);

    for (int i = 0; i < n_state; i++) h_g[i] = -h_g[i];

    std::memcpy(h_work, h_H, sz_H);
    for (int i = 0; i < n_state; i++) h_work[(size_t)i * n_state + i] += kDiagJitter;
    if (!cholesky_decompose_inplace(n_state, h_work)) {
      break;
    }
    cholesky_solve_lower(n_state, h_work, h_g, h_delta);

    double step_norm = 0.0;
    for (int i = 0; i < n_state; i++) step_norm += h_delta[i] * h_delta[i];
    step_norm = sqrt(step_norm);

    bool accepted = false;
    if (!enable_line_search) {
      for (int i = 0; i < n_state; i++) state_io[i] += h_delta[i];
      accepted = true;
    } else {
      double alpha = 1.0;
      for (int ls = 0; ls < 12; ls++) {
        for (int i = 0; i < n_state; i++) trial[i] = state_io[i] + alpha * h_delta[i];
        double ctry = pr_cost_host(n_epoch, n_sat, n_clock, ss, sat_ecef, pseudorange, weights,
                                    sys_host, trial, huber_k) 
                       + motion_cost_host(n_epoch, ss, w_motion, trial);
        if (ctry <= cost_before * (1.0 + 1e-12)) {
          std::memcpy(state_io, trial, sz_state);
          accepted = true;
          break;
        }
        alpha *= 0.5;
      }
    }

    CUDA_CHECK(cudaMemcpy(d_state, state_io, sz_state, cudaMemcpyHostToDevice));

    total_iters++;
    ok = true;
    if (accepted && step_norm < tol) break;
    if (!accepted) break;
  }

  if (out_mse_pr)
    *out_mse_pr = compute_pr_mse_host(n_epoch, n_sat, n_clock, ss, sat_ecef, pseudorange, weights, sys_host, state_io);

  std::free(h_H);
  std::free(h_g);
  std::free(h_delta);
  std::free(h_work);
  std::free(trial);
  std::free(h_eff_w);
  CUDA_CHECK(cudaFree(d_state));
  CUDA_CHECK(cudaFree(d_sat));
  CUDA_CHECK(cudaFree(d_pr));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_H));
  CUDA_CHECK(cudaFree(d_g));
  CUDA_CHECK(cudaFree(d_sys));

  return ok ? total_iters : -1;
}

}  // namespace gnss_gpu
