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
constexpr int kMaxClock = 7;
constexpr double kDiagJitter = 1e-3;

inline double sagnac_range_rate_mps(
    double sx, double sy, double svx, double svy,
    double x, double y, double vx, double vy) {
  return kOmegaE * (svx * y + sx * vy - svy * x - sy * vx) / kC;
}

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
                        const double* motion_disp, double* H, double* g) {
  if (w_motion <= 0.0) return;
  for (int t = 0; t < n_epoch - 1; t++) {
    int o0 = ss * t;
    int o1 = ss * (t + 1);
    for (int i = 0; i < 3; i++) {
      double pred = motion_disp ? motion_disp[t * 3 + i] : 0.0;
      double d01 = state[o0 + i] - state[o1 + i] + pred;
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

double motion_cost_host(int n_epoch, int ss, double w_motion, const double* state,
                        const double* motion_disp) {
  if (w_motion <= 0.0) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch - 1; t++) {
    int o0 = ss * t, o1 = ss * (t + 1);
    for (int i = 0; i < 3; i++) {
      double pred = motion_disp ? motion_disp[t * 3 + i] : 0.0;
      double d = state[o0 + i] - state[o1 + i] + pred;
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

// TDCP factor for 4-dim state: [x, y, z, c0, ..., c_{nc-1}]
// Residual: e_s^T * (x_{t+1} - x_t) + (clk_{t+1} - clk_t) - tdcp_meas
// where e_s = LOS unit vector from receiver to satellite (using mid-epoch satellite
// position, approximated by sat at epoch t+1).
// Jacobians: dR/dx_t = -e_s^T, dR/dx_{t+1} = +e_s^T, dR/dclk_t = -1, dR/dclk_{t+1} = +1
void add_tdcp_factor_host(
    int n_epoch, int n_sat, int nc, int ss, int n_state,
    const double* sat_ecef,
    const int* sys_kind_host,
    const double* tdcp_meas,
    const double* tdcp_weights,
    double tdcp_sigma_m,
    const double* state,
    double* H, double* g) {
  if (!tdcp_meas) return;

  for (int t = 0; t < n_epoch - 1; t++) {
    int o0 = ss * t;
    int o1 = ss * (t + 1);
    const double x1 = state[o1 + 0], y1 = state[o1 + 1], z1 = state[o1 + 2];

    // Use satellite positions at epoch t+1 for LOS computation
    const double* my_sat = sat_ecef + (size_t)(t + 1) * n_sat * 3;

    for (int s = 0; s < n_sat; s++) {
      double w = 0.0;
      if (tdcp_weights) {
        w = tdcp_weights[(size_t)t * n_sat + s];
      } else if (tdcp_sigma_m > 0.0) {
        w = 1.0 / (tdcp_sigma_m * tdcp_sigma_m);
      }
      if (w <= 0.0) continue;

      double meas = tdcp_meas[(size_t)t * n_sat + s];
      if (meas == 0.0 && !tdcp_weights) continue;  // unobserved
      // If explicit weights are given, w==0 already skipped above

      double sx = my_sat[s * 3 + 0], sy = my_sat[s * 3 + 1], sz = my_sat[s * 3 + 2];

      // Sagnac correction
      double dx0 = x1 - sx, dy0 = y1 - sy, dz0 = z1 - sz;
      double r0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
      double transit = r0 / kC;
      double theta = kOmegaE * transit;
      double sx_rot = sx * cos(theta) + sy * sin(theta);
      double sy_rot = -sx * sin(theta) + sy * cos(theta);

      double dx = x1 - sx_rot, dy_v = y1 - sy_rot, dz = z1 - sz;
      double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      if (r < 1e-6) continue;

      double ex = dx / r, ey = dy_v / r, ez = dz / r;
      int sk = sys_kind_host ? sys_kind_host[(t + 1) * n_sat + s] : 0;
      if (sk < 0 || sk >= nc) continue;
      double hc[kMaxClock];
      fill_hc_int(nc, sk, hc);

      // Residual: obs - pred where pred = e^T*(x1-x0) + (c1-c0)
      // Match pseudorange convention: g += J_pred * w * (obs - pred)
      double dx_t0 = state[o0 + 0], dy_t0 = state[o0 + 1], dz_t0 = state[o0 + 2];
      double pred_tdcp = ex * (x1 - dx_t0) + ey * (y1 - dy_t0) + ez * (z1 - dz_t0);
      for (int k = 0; k < nc; k++) {
        pred_tdcp += hc[k] * (state[o1 + 3 + k] - state[o0 + 3 + k]);
      }
      double res = meas - pred_tdcp;  // obs - pred

      // J_pred at x_t: d(pred)/d(x_t) = [-ex,-ey,-ez], d(pred)/d(clk_t) = -1
      // J_pred at x_{t+1}: d(pred)/d(x_{t+1}) = [+ex,+ey,+ez], d(pred)/d(clk_{t+1}) = +1
      double Jr = w * res;

      double Jt[3 + kMaxClock] = {};
      double Jt1[3 + kMaxClock] = {};
      Jt[0] = -ex;
      Jt[1] = -ey;
      Jt[2] = -ez;
      Jt1[0] = ex;
      Jt1[1] = ey;
      Jt1[2] = ez;
      for (int k = 0; k < nc; k++) {
        Jt[3 + k] = -hc[k];
        Jt1[3 + k] = hc[k];
      }

      for (int a = 0; a < ss; a++) {
        g[o0 + a] += Jt[a] * Jr;
        g[o1 + a] += Jt1[a] * Jr;
      }

      // Hessian: H += w * J_pred * J_pred^T (same regardless of residual sign)
      for (int a = 0; a < ss; a++) {
        for (int b = 0; b < ss; b++) {
          H[(size_t)(o0 + a) * n_state + (o0 + b)] += w * Jt[a] * Jt[b];
          H[(size_t)(o1 + a) * n_state + (o1 + b)] += w * Jt1[a] * Jt1[b];
          H[(size_t)(o0 + a) * n_state + (o1 + b)] += w * Jt[a] * Jt1[b];
          H[(size_t)(o1 + a) * n_state + (o0 + b)] += w * Jt1[a] * Jt[b];
        }
      }
    }
  }
}

double tdcp_cost_host(
    int n_epoch, int n_sat, int nc, int ss,
    const double* sat_ecef,
    const int* sys_kind_host,
    const double* tdcp_meas,
    const double* tdcp_weights,
    double tdcp_sigma_m,
    const double* state) {
  if (!tdcp_meas) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch - 1; t++) {
    int o0 = ss * t;
    int o1 = ss * (t + 1);
    const double x1 = state[o1 + 0], y1 = state[o1 + 1], z1 = state[o1 + 2];

    const double* my_sat = sat_ecef + (size_t)(t + 1) * n_sat * 3;

    for (int s = 0; s < n_sat; s++) {
      double w = 0.0;
      if (tdcp_weights) {
        w = tdcp_weights[(size_t)t * n_sat + s];
      } else if (tdcp_sigma_m > 0.0) {
        w = 1.0 / (tdcp_sigma_m * tdcp_sigma_m);
      }
      if (w <= 0.0) continue;

      double meas = tdcp_meas[(size_t)t * n_sat + s];
      if (meas == 0.0 && !tdcp_weights) continue;

      double sx = my_sat[s * 3 + 0], sy = my_sat[s * 3 + 1], sz = my_sat[s * 3 + 2];
      double dx0 = x1 - sx, dy0 = y1 - sy, dz0 = z1 - sz;
      double r0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
      double transit = r0 / kC;
      double theta = kOmegaE * transit;
      double sx_rot = sx * cos(theta) + sy * sin(theta);
      double sy_rot = -sx * sin(theta) + sy * cos(theta);
      double dx = x1 - sx_rot, dy_v = y1 - sy_rot, dz = z1 - sz;
      double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      if (r < 1e-6) continue;

      double ex = dx / r, ey = dy_v / r, ez = dz / r;
      int sk = sys_kind_host ? sys_kind_host[(t + 1) * n_sat + s] : 0;
      if (sk < 0 || sk >= nc) continue;
      double hc[kMaxClock];
      fill_hc_int(nc, sk, hc);
      double x0 = state[o0 + 0], y0 = state[o0 + 1], z0 = state[o0 + 2];
      double pred = ex * (x1 - x0) + ey * (y1 - y0) + ez * (z1 - z0);
      for (int k = 0; k < nc; k++) {
        pred += hc[k] * (state[o1 + 3 + k] - state[o0 + 3 + k]);
      }
      double res = pred - meas;
      e += 0.5 * w * res * res;
    }
  }
  return e;
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
                double* out_mse_pr,
                const double* motion_displacement,
                const double* tdcp_meas,
                const double* tdcp_weights,
                double tdcp_sigma_m) {
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

    add_motion_rw_host(n_epoch, ss, n_state, w_motion, state_io, motion_displacement, h_H, h_g);
    add_tdcp_factor_host(n_epoch, n_sat, n_clock, ss, n_state, sat_ecef, sys_host,
                         tdcp_meas, tdcp_weights, tdcp_sigma_m, state_io, h_H, h_g);

    double cost_before =
        pr_cost_host(n_epoch, n_sat, n_clock, ss, sat_ecef, pseudorange, weights, sys_host, state_io,
                     huber_k) +
        motion_cost_host(n_epoch, ss, w_motion, state_io, motion_displacement) +
        tdcp_cost_host(n_epoch, n_sat, n_clock, ss, sat_ecef, sys_host, tdcp_meas, tdcp_weights, tdcp_sigma_m, state_io);

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
                       + motion_cost_host(n_epoch, ss, w_motion, trial, motion_displacement)
                       + tdcp_cost_host(n_epoch, n_sat, n_clock, ss, sat_ecef, sys_host, tdcp_meas, tdcp_weights, tdcp_sigma_m, trial);
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

// ===========================================================================
// Extended FGO with velocity state + clock drift + Doppler factor
// ===========================================================================
// Per-epoch state layout:
//   [x, y, z, vx, vy, vz, c0, ..., c_{nc-1}, drift]
//   ss_vd = 3 + 3 + nc + 1 = 7 + nc
// Optional extended IMU layout appends accelerometer bias:
//   [x, y, z, vx, vy, vz, c0, ..., c_{nc-1}, drift, bax, bay, baz]
// ===========================================================================

namespace {

constexpr int kMaxClockVD = 7;
constexpr int kMaxSSVD = 10 + kMaxClockVD;  // max state size per epoch

// Pseudorange factor for VD state: touches position [0..2] and clock [6..6+nc-1].
// Jacobian columns: dx/r, dy/r, dz/r at [0,1,2]; hc[k] at [6+k].
__global__ void fgo_assemble_pseudorange_vd(
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
  // clock states start at index 6
  const double* cptr = state + t * ss + 6;

  // Local accumulation buffers for the used columns only: 3 pos + nc clock
  double Hloc[kMaxSSVD][kMaxSSVD] = {};
  double gloc[kMaxSSVD] = {};

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

    double hc[kMaxClockVD];
    fill_hc_int(nc, sk, hc);
    double clk = 0.0;
    for (int k = 0; k < nc; k++) clk += hc[k] * cptr[k];

    double pred = r + clk;
    double res = my_pr[s] - pred;

    // Build full-ss Jacobian: only position and clock columns are non-zero
    double J[kMaxSSVD] = {};
    J[0] = dx / r;
    J[1] = dy_v / r;
    J[2] = dz / r;
    for (int k = 0; k < nc; k++) J[6 + k] = hc[k];

    double Jr = res * w;
    for (int a = 0; a < ss; a++) {
      if (J[a] == 0.0) continue;
      gloc[a] += J[a] * Jr;
      for (int b = 0; b < ss; b++) {
        if (J[b] == 0.0) continue;
        Hloc[a][b] += w * J[a] * J[b];
      }
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

// Doppler factor: res = doppler_obs - (drift - geometric_range_rate)
// where geometric_range_rate follows the RTKLIB convention, including the
// first-order Sagnac range-rate correction and optional satellite clock drift.
// Jacobian w.r.t. state [x,y,z,vx,vy,vz,clk...,drift]:
//   d/dv = receiver-to-satellite LOS + d(Sagnac rate)/d(receiver velocity)
//   d/ddrift = 1  (index 6+nc)
// NOTE: we do not differentiate the unit vector w.r.t. position here (standard
// linearization around the current position estimate).
void add_doppler_factor_host(
    int n_epoch, int n_sat, int nc, int ss, int n_state,
    const double* sat_ecef,
    const double* sat_vel,
    const double* sat_clock_drift,
    const double* doppler,
    const double* doppler_weights,
    const int* sys_kind,
    const double* state,
    double* H, double* g) {
  if (!sat_vel || !doppler || !doppler_weights) return;
  const int drift_idx = 6 + nc;

  for (int t = 0; t < n_epoch; t++) {
    const double x = state[t * ss + 0];
    const double y = state[t * ss + 1];
    const double z = state[t * ss + 2];
    const double vx = state[t * ss + 3];
    const double vy = state[t * ss + 4];
    const double vz = state[t * ss + 5];
    const double drift = state[t * ss + drift_idx];

    const double* my_sat = sat_ecef + (size_t)t * n_sat * 3;
    const double* my_sv = sat_vel + (size_t)t * n_sat * 3;
    const double* my_scd = sat_clock_drift ? sat_clock_drift + (size_t)t * n_sat : nullptr;
    const double* my_dop = doppler + (size_t)t * n_sat;
    const double* my_dw = doppler_weights + (size_t)t * n_sat;

    int o = ss * t;

    for (int s = 0; s < n_sat; s++) {
      double w = my_dw[s];
      if (w <= 0.0) continue;

      double sx = my_sat[s * 3 + 0], sy = my_sat[s * 3 + 1], sz = my_sat[s * 3 + 2];
      double dx = sx - x, dy_v = sy - y, dz = sz - z;
      double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      if (r < 1e-6) continue;

      double los_x = dx / r, los_y = dy_v / r, los_z = dz / r;

      double svx = my_sv[s * 3 + 0], svy = my_sv[s * 3 + 1], svz = my_sv[s * 3 + 2];

      double euclidean_rate = los_x * (svx - vx) + los_y * (svy - vy) +
                              los_z * (svz - vz);
      double sag_rate = sagnac_range_rate_mps(sx, sy, svx, svy, x, y, vx, vy);
      double sat_clk_drift = (my_scd && std::isfinite(my_scd[s])) ? my_scd[s] : 0.0;
      double pred = drift - (euclidean_rate - sag_rate - sat_clk_drift);
      double res = my_dop[s] - pred;

      // Jacobian w.r.t. [vx, vy, vz] at indices [3,4,5]:
      // d(pred)/d(v) = LOS + d(Sagnac range-rate)/d(v)
      // d(pred)/d(drift) = 1
      // We build a sparse J for the full state vector
      double Jv[3] = {
          los_x - kOmegaE * sy / kC,
          los_y + kOmegaE * sx / kC,
          los_z,
      };
      double Jd = 1.0;

      // Gradient: g += J * w * res (standard J^T * W * r convention, same as PR and motion)
      double Jr = res * w;
      // Velocity-velocity block
      for (int a = 0; a < 3; a++) {
        g[o + 3 + a] += Jv[a] * Jr;
        for (int b = 0; b < 3; b++) {
          H[(size_t)(o + 3 + a) * n_state + (o + 3 + b)] += w * Jv[a] * Jv[b];
        }
        // Velocity-drift cross
        H[(size_t)(o + 3 + a) * n_state + (o + drift_idx)] += w * Jv[a] * Jd;
        H[(size_t)(o + drift_idx) * n_state + (o + 3 + a)] += w * Jd * Jv[a];
      }
      // Drift-drift
      g[o + drift_idx] += Jd * Jr;
      H[(size_t)(o + drift_idx) * n_state + (o + drift_idx)] += w * Jd * Jd;
    }
  }
}

// Motion factor: x_{t+1} = x_t + v_t * dt
// residual_i = x_{t,i} + v_{t,i} * dt - x_{t+1,i}  for i in {0,1,2}
// Couples position at t, velocity at t, and position at t+1.
void add_motion_factor_host(
    int n_epoch, int ss, int n_state, double w_motion,
    const double* state, const double* dt_arr, double* H, double* g) {
  if (w_motion <= 0.0 || !dt_arr) return;

  for (int t = 0; t < n_epoch - 1; t++) {
    double dt = dt_arr[t];
    if (dt <= 0.0) continue;

    int o0 = ss * t;
    int o1 = ss * (t + 1);

    for (int i = 0; i < 3; i++) {
      double x_t = state[o0 + i];
      double v_t = state[o0 + 3 + i];
      double x_t1 = state[o1 + i];
      double res = x_t + v_t * dt - x_t1;

      // Jacobian: d/d(x_t,i)=1, d/d(v_t,i)=dt, d/d(x_{t+1},i)=-1
      g[o0 + i]     -= w_motion * res;
      g[o0 + 3 + i] -= w_motion * res * dt;
      g[o1 + i]     += w_motion * res;

      // Hessian contributions: J^T W J
      // (x_t,i)-(x_t,i):  1*1 = 1
      H[(size_t)(o0 + i) * n_state + (o0 + i)] += w_motion;
      // (x_t,i)-(v_t,i):  1*dt
      H[(size_t)(o0 + i) * n_state + (o0 + 3 + i)] += w_motion * dt;
      H[(size_t)(o0 + 3 + i) * n_state + (o0 + i)] += w_motion * dt;
      // (x_t,i)-(x_{t+1},i):  1*(-1)
      H[(size_t)(o0 + i) * n_state + (o1 + i)] += -w_motion;
      H[(size_t)(o1 + i) * n_state + (o0 + i)] += -w_motion;
      // (v_t,i)-(v_t,i):  dt*dt
      H[(size_t)(o0 + 3 + i) * n_state + (o0 + 3 + i)] += w_motion * dt * dt;
      // (v_t,i)-(x_{t+1},i):  dt*(-1)
      H[(size_t)(o0 + 3 + i) * n_state + (o1 + i)] += -w_motion * dt;
      H[(size_t)(o1 + i) * n_state + (o0 + 3 + i)] += -w_motion * dt;
      // (x_{t+1},i)-(x_{t+1},i):  (-1)*(-1) = 1
      H[(size_t)(o1 + i) * n_state + (o1 + i)] += w_motion;
    }
  }
}

// Clock drift factor:
//   XXDD / CCDD parity: c0_{t+1} = c0_t + (drift_t + drift_{t+1}) * dt / 2
//   legacy VD mode:     c0_{t+1} = c0_t + drift_t * dt
// Clock index in VD state: 6 (first clock). Drift index: 6+nc.
void add_clock_drift_factor_host(
    int n_epoch, int nc, int ss, int n_state, double w_clkdrift,
    const double* state, const double* dt_arr, bool clock_use_average_drift, double* H, double* g) {
  if (w_clkdrift <= 0.0 || !dt_arr) return;
  const int clk_idx = 6;  // first clock
  const int drift_idx = 6 + nc;

  for (int t = 0; t < n_epoch - 1; t++) {
    double dt = dt_arr[t];
    if (dt <= 0.0) continue;

    int o0 = ss * t;
    int o1 = ss * (t + 1);

    double c_t = state[o0 + clk_idx];
    double d_t = state[o0 + drift_idx];
    double d_t1 = state[o1 + drift_idx];
    double c_t1 = state[o1 + clk_idx];
    double res = c_t - c_t1;
    if (clock_use_average_drift) {
      res += 0.5 * (d_t + d_t1) * dt;
    } else {
      res += d_t * dt;
    }

    g[o0 + clk_idx]   -= w_clkdrift * res;
    g[o1 + clk_idx]   += w_clkdrift * res;
    if (clock_use_average_drift) {
      g[o0 + drift_idx] -= w_clkdrift * res * (dt * 0.5);
      g[o1 + drift_idx] -= w_clkdrift * res * (dt * 0.5);
    } else {
      g[o0 + drift_idx] -= w_clkdrift * res * dt;
    }

    H[(size_t)(o0 + clk_idx) * n_state + (o0 + clk_idx)] += w_clkdrift;
    H[(size_t)(o0 + clk_idx) * n_state + (o1 + clk_idx)] += -w_clkdrift;
    H[(size_t)(o1 + clk_idx) * n_state + (o0 + clk_idx)] += -w_clkdrift;
    H[(size_t)(o1 + clk_idx) * n_state + (o1 + clk_idx)] += w_clkdrift;
    if (clock_use_average_drift) {
      const double half_dt = 0.5 * dt;
      H[(size_t)(o0 + clk_idx) * n_state + (o0 + drift_idx)] += w_clkdrift * half_dt;
      H[(size_t)(o0 + drift_idx) * n_state + (o0 + clk_idx)] += w_clkdrift * half_dt;
      H[(size_t)(o0 + clk_idx) * n_state + (o1 + drift_idx)] += w_clkdrift * half_dt;
      H[(size_t)(o1 + drift_idx) * n_state + (o0 + clk_idx)] += w_clkdrift * half_dt;
      H[(size_t)(o1 + clk_idx) * n_state + (o0 + drift_idx)] += -w_clkdrift * half_dt;
      H[(size_t)(o0 + drift_idx) * n_state + (o1 + clk_idx)] += -w_clkdrift * half_dt;
      H[(size_t)(o1 + clk_idx) * n_state + (o1 + drift_idx)] += -w_clkdrift * half_dt;
      H[(size_t)(o1 + drift_idx) * n_state + (o1 + clk_idx)] += -w_clkdrift * half_dt;
      H[(size_t)(o0 + drift_idx) * n_state + (o0 + drift_idx)] += w_clkdrift * half_dt * half_dt;
      H[(size_t)(o1 + drift_idx) * n_state + (o1 + drift_idx)] += w_clkdrift * half_dt * half_dt;
      H[(size_t)(o0 + drift_idx) * n_state + (o1 + drift_idx)] += w_clkdrift * half_dt * half_dt;
      H[(size_t)(o1 + drift_idx) * n_state + (o0 + drift_idx)] += w_clkdrift * half_dt * half_dt;
    } else {
      H[(size_t)(o0 + clk_idx) * n_state + (o0 + drift_idx)] += w_clkdrift * dt;
      H[(size_t)(o0 + drift_idx) * n_state + (o0 + clk_idx)] += w_clkdrift * dt;
      H[(size_t)(o0 + drift_idx) * n_state + (o0 + drift_idx)] += w_clkdrift * dt * dt;
      H[(size_t)(o0 + drift_idx) * n_state + (o1 + clk_idx)] += -w_clkdrift * dt;
      H[(size_t)(o1 + clk_idx) * n_state + (o0 + drift_idx)] += -w_clkdrift * dt;
    }
  }
}

void add_stop_velocity_factor_host(
    int n_epoch, int ss, int n_state, double w_stop_velocity,
    const std::uint8_t* stop_mask, const double* state, double* H, double* g) {
  if (w_stop_velocity <= 0.0 || !stop_mask) return;

  for (int t = 0; t < n_epoch; t++) {
    if (stop_mask[t] == 0) continue;
    const int o = ss * t;
    for (int i = 0; i < 3; i++) {
      const int idx = o + 3 + i;
      const double res = -state[idx];
      g[idx] += w_stop_velocity * res;
      H[(size_t)idx * n_state + idx] += w_stop_velocity;
    }
  }
}

void add_stop_position_factor_host(
    int n_epoch, int ss, int n_state, double w_stop_position,
    const std::uint8_t* stop_mask, const double* state, double* H, double* g) {
  if (w_stop_position <= 0.0 || !stop_mask) return;

  for (int t = 0; t < n_epoch - 1; t++) {
    if (stop_mask[t] == 0 || stop_mask[t + 1] == 0) continue;
    const int o0 = ss * t;
    const int o1 = ss * (t + 1);
    for (int i = 0; i < 3; i++) {
      const double res = state[o0 + i] - state[o1 + i];
      g[o0 + i] += -w_stop_position * res;
      g[o1 + i] += w_stop_position * res;
      H[(size_t)(o0 + i) * n_state + (o0 + i)] += w_stop_position;
      H[(size_t)(o1 + i) * n_state + (o1 + i)] += w_stop_position;
      H[(size_t)(o0 + i) * n_state + (o1 + i)] += -w_stop_position;
      H[(size_t)(o1 + i) * n_state + (o0 + i)] += -w_stop_position;
    }
  }
}

// PR cost for VD state (clock offset at index 6)
double pr_cost_host_vd(
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
    const double* cptr = state + t * ss + 6;
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
      double hc[kMaxClockVD];
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

// Motion factor cost
double motion_factor_cost_host(
    int n_epoch, int ss, double w_motion,
    const double* state, const double* dt_arr) {
  if (w_motion <= 0.0 || !dt_arr) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch - 1; t++) {
    double dt = dt_arr[t];
    if (dt <= 0.0) continue;
    int o0 = ss * t, o1 = ss * (t + 1);
    for (int i = 0; i < 3; i++) {
      double res = state[o0 + i] + state[o0 + 3 + i] * dt - state[o1 + i];
      e += 0.5 * w_motion * res * res;
    }
  }
  return e;
}

// Clock drift factor cost
double clock_drift_cost_host(
    int n_epoch, int nc, int ss, double w_clkdrift,
    const double* state, const double* dt_arr, bool clock_use_average_drift) {
  if (w_clkdrift <= 0.0 || !dt_arr) return 0.0;
  double e = 0.0;
  const int clk_idx = 6;
  const int drift_idx = 6 + nc;
  for (int t = 0; t < n_epoch - 1; t++) {
    double dt = dt_arr[t];
    if (dt <= 0.0) continue;
    int o0 = ss * t, o1 = ss * (t + 1);
    double res = state[o0 + clk_idx] - state[o1 + clk_idx];
    if (clock_use_average_drift) {
      res += 0.5 * dt * (state[o0 + drift_idx] + state[o1 + drift_idx]);
    } else {
      res += state[o0 + drift_idx] * dt;
    }
    e += 0.5 * w_clkdrift * res * res;
  }
  return e;
}

double stop_velocity_cost_host(
    int n_epoch, int ss, double w_stop_velocity,
    const std::uint8_t* stop_mask, const double* state) {
  if (w_stop_velocity <= 0.0 || !stop_mask) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch; t++) {
    if (stop_mask[t] == 0) continue;
    const int o = ss * t;
    for (int i = 0; i < 3; i++) {
      const double res = state[o + 3 + i];
      e += 0.5 * w_stop_velocity * res * res;
    }
  }
  return e;
}

double stop_position_cost_host(
    int n_epoch, int ss, double w_stop_position,
    const std::uint8_t* stop_mask, const double* state) {
  if (w_stop_position <= 0.0 || !stop_mask) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch - 1; t++) {
    if (stop_mask[t] == 0 || stop_mask[t + 1] == 0) continue;
    const int o0 = ss * t;
    const int o1 = ss * (t + 1);
    for (int i = 0; i < 3; i++) {
      const double res = state[o0 + i] - state[o1 + i];
      e += 0.5 * w_stop_position * res * res;
    }
  }
  return e;
}

void add_imu_prior_factor_host(
    int n_epoch, int ss, int n_state,
    double w_imu_pos, double w_imu_vel,
    const double* imu_delta_p, const double* imu_delta_v,
    const double* state, const double* dt_arr, int accel_bias_idx, double* H, double* g) {
  if ((!imu_delta_p || w_imu_pos <= 0.0) && (!imu_delta_v || w_imu_vel <= 0.0)) return;

  for (int t = 0; t < n_epoch - 1; t++) {
    const int o0 = ss * t;
    const int o1 = ss * (t + 1);
    const double dt = dt_arr ? dt_arr[t] : 0.0;
    const bool has_valid_dt = dt_arr && std::isfinite(dt) && dt > 0.0;

    if (imu_delta_p && w_imu_pos > 0.0 && has_valid_dt) {
      const double* dp = imu_delta_p + (size_t)t * 3;
      for (int i = 0; i < 3; i++) {
        if (!std::isfinite(dp[i])) continue;
        double res = state[o0 + i] + state[o0 + 3 + i] * dt + dp[i] - state[o1 + i];
        const int b0 = accel_bias_idx >= 0 ? o0 + accel_bias_idx + i : -1;
        const double Jb = b0 >= 0 ? -0.5 * dt * dt : 0.0;
        if (b0 >= 0) res += Jb * state[b0];
        const double Jr = w_imu_pos * res;
        g[o0 + i] -= Jr;
        g[o0 + 3 + i] -= dt * Jr;
        g[o1 + i] += Jr;
        if (b0 >= 0) g[b0] -= Jb * Jr;

        H[(size_t)(o0 + i) * n_state + (o0 + i)] += w_imu_pos;
        H[(size_t)(o0 + i) * n_state + (o0 + 3 + i)] += w_imu_pos * dt;
        H[(size_t)(o0 + 3 + i) * n_state + (o0 + i)] += w_imu_pos * dt;
        H[(size_t)(o0 + i) * n_state + (o1 + i)] -= w_imu_pos;
        H[(size_t)(o1 + i) * n_state + (o0 + i)] -= w_imu_pos;
        H[(size_t)(o0 + 3 + i) * n_state + (o0 + 3 + i)] += w_imu_pos * dt * dt;
        H[(size_t)(o0 + 3 + i) * n_state + (o1 + i)] -= w_imu_pos * dt;
        H[(size_t)(o1 + i) * n_state + (o0 + 3 + i)] -= w_imu_pos * dt;
        H[(size_t)(o1 + i) * n_state + (o1 + i)] += w_imu_pos;
        if (b0 >= 0) {
          H[(size_t)(o0 + i) * n_state + b0] += w_imu_pos * Jb;
          H[(size_t)b0 * n_state + (o0 + i)] += w_imu_pos * Jb;
          H[(size_t)(o0 + 3 + i) * n_state + b0] += w_imu_pos * dt * Jb;
          H[(size_t)b0 * n_state + (o0 + 3 + i)] += w_imu_pos * dt * Jb;
          H[(size_t)(o1 + i) * n_state + b0] -= w_imu_pos * Jb;
          H[(size_t)b0 * n_state + (o1 + i)] -= w_imu_pos * Jb;
          H[(size_t)b0 * n_state + b0] += w_imu_pos * Jb * Jb;
        }
      }
    }

    if (imu_delta_v && w_imu_vel > 0.0) {
      const double* dv = imu_delta_v + (size_t)t * 3;
      for (int i = 0; i < 3; i++) {
        if (!std::isfinite(dv[i])) continue;
        const int v0 = o0 + 3 + i;
        const int v1 = o1 + 3 + i;
        double res = state[v0] + dv[i] - state[v1];
        const int b0 = accel_bias_idx >= 0 ? o0 + accel_bias_idx + i : -1;
        const double Jb = b0 >= 0 ? -dt : 0.0;
        if (b0 >= 0) res += Jb * state[b0];
        const double Jr = w_imu_vel * res;
        g[v0] -= Jr;
        g[v1] += Jr;
        if (b0 >= 0) g[b0] -= Jb * Jr;

        H[(size_t)v0 * n_state + v0] += w_imu_vel;
        H[(size_t)v0 * n_state + v1] -= w_imu_vel;
        H[(size_t)v1 * n_state + v0] -= w_imu_vel;
        H[(size_t)v1 * n_state + v1] += w_imu_vel;
        if (b0 >= 0) {
          H[(size_t)v0 * n_state + b0] += w_imu_vel * Jb;
          H[(size_t)b0 * n_state + v0] += w_imu_vel * Jb;
          H[(size_t)v1 * n_state + b0] -= w_imu_vel * Jb;
          H[(size_t)b0 * n_state + v1] -= w_imu_vel * Jb;
          H[(size_t)b0 * n_state + b0] += w_imu_vel * Jb * Jb;
        }
      }
    }
  }
}

double imu_prior_cost_host(
    int n_epoch, int ss,
    double w_imu_pos, double w_imu_vel,
    const double* imu_delta_p, const double* imu_delta_v,
    const double* state, const double* dt_arr, int accel_bias_idx) {
  if ((!imu_delta_p || w_imu_pos <= 0.0) && (!imu_delta_v || w_imu_vel <= 0.0)) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch - 1; t++) {
    const int o0 = ss * t;
    const int o1 = ss * (t + 1);
    const double dt = dt_arr ? dt_arr[t] : 0.0;
    const bool has_valid_dt = dt_arr && std::isfinite(dt) && dt > 0.0;

    if (imu_delta_p && w_imu_pos > 0.0 && has_valid_dt) {
      const double* dp = imu_delta_p + (size_t)t * 3;
      for (int i = 0; i < 3; i++) {
        if (!std::isfinite(dp[i])) continue;
        double res = state[o0 + i] + state[o0 + 3 + i] * dt + dp[i] - state[o1 + i];
        if (accel_bias_idx >= 0) res -= 0.5 * dt * dt * state[o0 + accel_bias_idx + i];
        e += 0.5 * w_imu_pos * res * res;
      }
    }

    if (imu_delta_v && w_imu_vel > 0.0) {
      const double* dv = imu_delta_v + (size_t)t * 3;
      for (int i = 0; i < 3; i++) {
        if (!std::isfinite(dv[i])) continue;
        double res = state[o0 + 3 + i] + dv[i] - state[o1 + 3 + i];
        if (accel_bias_idx >= 0) res -= dt * state[o0 + accel_bias_idx + i];
        e += 0.5 * w_imu_vel * res * res;
      }
    }
  }
  return e;
}

void add_accel_bias_factor_host(
    int n_epoch, int ss, int n_state, int accel_bias_idx,
    double w_bias_prior, double w_bias_between,
    const double* state, double* H, double* g) {
  if (accel_bias_idx < 0) return;

  if (w_bias_prior > 0.0) {
    const int o0 = 0;
    for (int i = 0; i < 3; i++) {
      const int idx = o0 + accel_bias_idx + i;
      const double res = state[idx];
      g[idx] -= w_bias_prior * res;
      H[(size_t)idx * n_state + idx] += w_bias_prior;
    }
  }

  if (w_bias_between <= 0.0) return;
  for (int t = 0; t < n_epoch - 1; t++) {
    const int o0 = ss * t;
    const int o1 = ss * (t + 1);
    for (int i = 0; i < 3; i++) {
      const int b0 = o0 + accel_bias_idx + i;
      const int b1 = o1 + accel_bias_idx + i;
      const double res = state[b0] - state[b1];
      const double Jr = w_bias_between * res;
      g[b0] -= Jr;
      g[b1] += Jr;
      H[(size_t)b0 * n_state + b0] += w_bias_between;
      H[(size_t)b0 * n_state + b1] -= w_bias_between;
      H[(size_t)b1 * n_state + b0] -= w_bias_between;
      H[(size_t)b1 * n_state + b1] += w_bias_between;
    }
  }
}

double accel_bias_cost_host(
    int n_epoch, int ss, int accel_bias_idx,
    double w_bias_prior, double w_bias_between,
    const double* state) {
  if (accel_bias_idx < 0) return 0.0;
  double e = 0.0;
  if (w_bias_prior > 0.0) {
    for (int i = 0; i < 3; i++) {
      const double res = state[accel_bias_idx + i];
      e += 0.5 * w_bias_prior * res * res;
    }
  }
  if (w_bias_between > 0.0) {
    for (int t = 0; t < n_epoch - 1; t++) {
      const int o0 = ss * t;
      const int o1 = ss * (t + 1);
      for (int i = 0; i < 3; i++) {
        const double res = state[o0 + accel_bias_idx + i] - state[o1 + accel_bias_idx + i];
        e += 0.5 * w_bias_between * res * res;
      }
    }
  }
  return e;
}

// Relative height (ENU up) equality: residual = u·(x_i - x_j), u = unit ENU-up in ECEF.
void add_relative_height_factor_host(
    int n_epoch, int ss, int n_state, double w_rel_h,
    double ux, double uy, double uz,
    int n_edges, const std::int32_t* edge_i, const std::int32_t* edge_j,
    const double* state, double* H, double* g) {
  if (w_rel_h <= 0.0 || n_edges <= 0 || !edge_i || !edge_j || !state || !H || !g) return;
  double nrm = std::sqrt(ux * ux + uy * uy + uz * uz);
  if (nrm < 1e-12) return;
  ux /= nrm;
  uy /= nrm;
  uz /= nrm;

  for (int eidx = 0; eidx < n_edges; eidx++) {
    int i = edge_i[eidx];
    int j = edge_j[eidx];
    if (i < 0 || j < 0 || i >= n_epoch || j >= n_epoch || i == j) continue;
    int oi = ss * i;
    int oj = ss * j;
    double r = ux * (state[oi + 0] - state[oj + 0]) + uy * (state[oi + 1] - state[oj + 1]) +
               uz * (state[oi + 2] - state[oj + 2]);
    for (int k = 0; k < 3; k++) {
      double uk = (k == 0) ? ux : ((k == 1) ? uy : uz);
      g[oi + k] -= w_rel_h * r * uk;
      g[oj + k] += w_rel_h * r * uk;
    }
    for (int a = 0; a < 3; a++) {
      double ua = (a == 0) ? ux : ((a == 1) ? uy : uz);
      for (int b = 0; b < 3; b++) {
        double ub = (b == 0) ? ux : ((b == 1) ? uy : uz);
        double hij = w_rel_h * ua * ub;
        H[(size_t)(oi + a) * n_state + (oi + b)] += hij;
        H[(size_t)(oj + a) * n_state + (oj + b)] += hij;
        H[(size_t)(oi + a) * n_state + (oj + b)] -= hij;
        H[(size_t)(oj + a) * n_state + (oi + b)] -= hij;
      }
    }
  }
}

double relative_height_cost_host(
    int n_epoch, int ss, double w_rel_h,
    double ux, double uy, double uz,
    int n_edges, const std::int32_t* edge_i, const std::int32_t* edge_j,
    const double* state) {
  if (w_rel_h <= 0.0 || n_edges <= 0 || !edge_i || !edge_j || !state) return 0.0;
  double nrm = std::sqrt(ux * ux + uy * uy + uz * uz);
  if (nrm < 1e-12) return 0.0;
  ux /= nrm;
  uy /= nrm;
  uz /= nrm;
  double cost = 0.0;
  for (int eidx = 0; eidx < n_edges; eidx++) {
    int i = edge_i[eidx];
    int j = edge_j[eidx];
    if (i < 0 || j < 0 || i >= n_epoch || j >= n_epoch || i == j) continue;
    int oi = ss * i;
    int oj = ss * j;
    double r = ux * (state[oi + 0] - state[oj + 0]) + uy * (state[oi + 1] - state[oj + 1]) +
               uz * (state[oi + 2] - state[oj + 2]);
    cost += 0.5 * w_rel_h * r * r;
  }
  return cost;
}

// Absolute height prior: residual = u·(ref_t - x_t), u = unit ENU-up in ECEF.
void add_absolute_height_factor_host(
    int n_epoch, int ss, int n_state, double w_abs_h,
    double ux, double uy, double uz,
    const double* ref_ecef, const double* state, double* H, double* g) {
  if (w_abs_h <= 0.0 || !ref_ecef || !state || !H || !g) return;
  double nrm = std::sqrt(ux * ux + uy * uy + uz * uz);
  if (nrm < 1e-12) return;
  ux /= nrm;
  uy /= nrm;
  uz /= nrm;

  for (int t = 0; t < n_epoch; t++) {
    const int o = ss * t;
    const double* ref = ref_ecef + (size_t)t * 3;
    if (!std::isfinite(ref[0]) || !std::isfinite(ref[1]) || !std::isfinite(ref[2])) continue;
    if (!std::isfinite(state[o + 0]) || !std::isfinite(state[o + 1]) || !std::isfinite(state[o + 2])) continue;
    const double r = ux * (ref[0] - state[o + 0]) + uy * (ref[1] - state[o + 1]) +
                     uz * (ref[2] - state[o + 2]);
    const double u[3] = {ux, uy, uz};
    for (int a = 0; a < 3; a++) {
      g[o + a] += w_abs_h * r * u[a];
      for (int b = 0; b < 3; b++) {
        H[(size_t)(o + a) * n_state + (o + b)] += w_abs_h * u[a] * u[b];
      }
    }
  }
}

double absolute_height_cost_host(
    int n_epoch, int ss, double w_abs_h,
    double ux, double uy, double uz,
    const double* ref_ecef, const double* state) {
  if (w_abs_h <= 0.0 || !ref_ecef || !state) return 0.0;
  double nrm = std::sqrt(ux * ux + uy * uy + uz * uz);
  if (nrm < 1e-12) return 0.0;
  ux /= nrm;
  uy /= nrm;
  uz /= nrm;
  double cost = 0.0;
  for (int t = 0; t < n_epoch; t++) {
    const int o = ss * t;
    const double* ref = ref_ecef + (size_t)t * 3;
    if (!std::isfinite(ref[0]) || !std::isfinite(ref[1]) || !std::isfinite(ref[2])) continue;
    if (!std::isfinite(state[o + 0]) || !std::isfinite(state[o + 1]) || !std::isfinite(state[o + 2])) continue;
    const double r = ux * (ref[0] - state[o + 0]) + uy * (ref[1] - state[o + 1]) +
                     uz * (ref[2] - state[o + 2]);
    cost += 0.5 * w_abs_h * r * r;
  }
  return cost;
}

// Doppler factor cost
double doppler_cost_host(
    int n_epoch, int n_sat, int nc, int ss,
    const double* sat_ecef, const double* sat_vel,
    const double* sat_clock_drift, const double* doppler, const double* doppler_weights,
    const double* state) {
  if (!sat_vel || !doppler || !doppler_weights) return 0.0;
  double e = 0.0;
  const int drift_idx = 6 + nc;
  for (int t = 0; t < n_epoch; t++) {
    const double x = state[t * ss + 0], y = state[t * ss + 1], z = state[t * ss + 2];
    const double vx = state[t * ss + 3], vy = state[t * ss + 4], vz = state[t * ss + 5];
    const double drift = state[t * ss + drift_idx];
    const double* my_sat = sat_ecef + (size_t)t * n_sat * 3;
    const double* my_sv = sat_vel + (size_t)t * n_sat * 3;
    const double* my_scd = sat_clock_drift ? sat_clock_drift + (size_t)t * n_sat : nullptr;
    const double* my_dop = doppler + (size_t)t * n_sat;
    const double* my_dw = doppler_weights + (size_t)t * n_sat;
    for (int s = 0; s < n_sat; s++) {
      double w = my_dw[s];
      if (w <= 0.0) continue;
      double sx = my_sat[s * 3 + 0], sy = my_sat[s * 3 + 1], sz = my_sat[s * 3 + 2];
      double dx = sx - x, dy_v = sy - y, dz = sz - z;
      double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      if (r < 1e-6) continue;
      double los_x = dx / r, los_y = dy_v / r, los_z = dz / r;
      double svx = my_sv[s * 3 + 0], svy = my_sv[s * 3 + 1], svz = my_sv[s * 3 + 2];
      double euclidean_rate = los_x * (svx - vx) + los_y * (svy - vy) +
                              los_z * (svz - vz);
      double sag_rate = sagnac_range_rate_mps(sx, sy, svx, svy, x, y, vx, vy);
      double sat_clk_drift = (my_scd && std::isfinite(my_scd[s])) ? my_scd[s] : 0.0;
      double pred = drift - (euclidean_rate - sag_rate - sat_clk_drift);
      double res = my_dop[s] - pred;
      e += 0.5 * w * res * res;
    }
  }
  return e;
}

// Effective Huber weights for VD state (clock at index 6)
void effective_pr_weights_huber_host_vd(
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
    const double* cptr = state + t * ss + 6;
    const double* my_sat = sat_ecef + (size_t)t * n_sat * 3;
    const double* my_pr = pseudorange + (size_t)t * n_sat;
    const double* my_w = weights + (size_t)t * n_sat;
    for (int s = 0; s < n_sat; s++) {
      double w = my_w[s];
      size_t idx = (size_t)t * n_sat + s;
      if (w <= 0.0) { eff_w_out[idx] = w; continue; }
      int sk = sys_kind_host[t * n_sat + s];
      if (sk < 0 || sk >= nc) { eff_w_out[idx] = 0.0; continue; }
      double sx = my_sat[s * 3 + 0], sy = my_sat[s * 3 + 1], sz = my_sat[s * 3 + 2];
      double dx0 = x - sx, dy0 = y - sy, dz0 = z - sz;
      double r0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
      double transit = r0 / kC;
      double theta = kOmegaE * transit;
      double sx_rot = sx * cos(theta) + sy * sin(theta);
      double sy_rot = -sx * sin(theta) + sy * cos(theta);
      double dx = x - sx_rot, dy_v = y - sy_rot, dz = z - sz;
      double r_geom = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      if (r_geom < 1e-6) { eff_w_out[idx] = 0.0; continue; }
      double hc[kMaxClockVD];
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

// PR MSE for VD state
double compute_pr_mse_host_vd(
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
    const double* cptr = state + t * ss + 6;
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
      double hc[kMaxClockVD];
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

// TDCP factor for VD state: [x, y, z, vx, vy, vz, c0, ..., c_{nc-1}, drift]
// Position at indices [0..2], first clock at index 6.
// Residual: e_s^T * (x_{t+1} - x_t) + (clk_{t+1} - clk_t) - tdcp_meas
void add_tdcp_factor_host_vd(
    int n_epoch, int n_sat, int nc, int ss, int n_state,
    const double* sat_ecef,
    const int* sys_kind_host,
    const double* dt_arr,
    const double* tdcp_meas,
    const double* tdcp_weights,
    double tdcp_sigma_m,
    bool tdcp_use_drift,
    const double* state,
    double* H, double* g) {
  if (!tdcp_meas) return;
  const int clk_idx = 6;  // first clock index in VD state
  const int drift_idx = 6 + nc;

  for (int t = 0; t < n_epoch - 1; t++) {
    const double dt = dt_arr ? dt_arr[t] : 0.0;
    if (tdcp_use_drift && (!std::isfinite(dt) || dt <= 0.0)) continue;
    int o0 = ss * t;
    int o1 = ss * (t + 1);
    const double x1 = state[o1 + 0], y1 = state[o1 + 1], z1 = state[o1 + 2];

    const double* my_sat = sat_ecef + (size_t)(t + 1) * n_sat * 3;

    for (int s = 0; s < n_sat; s++) {
      double w = 0.0;
      if (tdcp_weights) {
        w = tdcp_weights[(size_t)t * n_sat + s];
      } else if (tdcp_sigma_m > 0.0) {
        w = 1.0 / (tdcp_sigma_m * tdcp_sigma_m);
      }
      if (w <= 0.0) continue;

      double meas = tdcp_meas[(size_t)t * n_sat + s];
      if (meas == 0.0 && !tdcp_weights) continue;

      double sx = my_sat[s * 3 + 0], sy = my_sat[s * 3 + 1], sz = my_sat[s * 3 + 2];
      double dx0 = x1 - sx, dy0 = y1 - sy, dz0 = z1 - sz;
      double r0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
      double transit = r0 / kC;
      double theta = kOmegaE * transit;
      double sx_rot = sx * cos(theta) + sy * sin(theta);
      double sy_rot = -sx * sin(theta) + sy * cos(theta);
      double dx = x1 - sx_rot, dy_v = y1 - sy_rot, dz = z1 - sz;
      double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      if (r < 1e-6) continue;

      double ex = dx / r, ey = dy_v / r, ez = dz / r;
      int sk = sys_kind_host ? sys_kind_host[(t + 1) * n_sat + s] : 0;
      if (sk < 0 || sk >= nc) continue;
      double hc[kMaxClock];
      fill_hc_int(nc, sk, hc);

      // Residual: obs - pred (match pseudorange convention)
      double x0 = state[o0 + 0], y0 = state[o0 + 1], z0 = state[o0 + 2];
      double pred_tdcp = ex * (x1 - x0) + ey * (y1 - y0) + ez * (z1 - z0);
      if (tdcp_use_drift) {
        pred_tdcp += 0.5 * dt * (state[o0 + drift_idx] + state[o1 + drift_idx]);
      } else {
        for (int k = 0; k < nc; k++) {
          pred_tdcp += hc[k] * (state[o1 + clk_idx + k] - state[o0 + clk_idx + k]);
        }
      }
      double res = meas - pred_tdcp;  // obs - pred

      // J_pred at t: [-ex,-ey,-ez,-1] / [+dt/2] for XXCC / XXDD.
      double Jr = w * res;
      if (tdcp_use_drift) {
        const double half_dt = 0.5 * dt;
        g[o0 + 0] += (-ex) * Jr;
        g[o0 + 1] += (-ey) * Jr;
        g[o0 + 2] += (-ez) * Jr;
        g[o0 + drift_idx] += half_dt * Jr;

        g[o1 + 0] += ex * Jr;
        g[o1 + 1] += ey * Jr;
        g[o1 + 2] += ez * Jr;
        g[o1 + drift_idx] += half_dt * Jr;

        int idx0[4] = {o0 + 0, o0 + 1, o0 + 2, o0 + drift_idx};
        int idx1[4] = {o1 + 0, o1 + 1, o1 + 2, o1 + drift_idx};
        double Jt[4] = {-ex, -ey, -ez, half_dt};
        double Jt1[4] = {ex, ey, ez, half_dt};

        for (int a = 0; a < 4; a++)
          for (int b = 0; b < 4; b++)
            H[(size_t)idx0[a] * n_state + idx0[b]] += w * Jt[a] * Jt[b];
        for (int a = 0; a < 4; a++)
          for (int b = 0; b < 4; b++)
            H[(size_t)idx1[a] * n_state + idx1[b]] += w * Jt1[a] * Jt1[b];
        for (int a = 0; a < 4; a++)
          for (int b = 0; b < 4; b++) {
            H[(size_t)idx0[a] * n_state + idx1[b]] += w * Jt[a] * Jt1[b];
            H[(size_t)idx1[a] * n_state + idx0[b]] += w * Jt1[a] * Jt[b];
          }
      } else {
        double Jt[7 + kMaxClock] = {};
        double Jt1[7 + kMaxClock] = {};
        Jt[0] = -ex;
        Jt[1] = -ey;
        Jt[2] = -ez;
        Jt1[0] = ex;
        Jt1[1] = ey;
        Jt1[2] = ez;
        for (int k = 0; k < nc; k++) {
          Jt[clk_idx + k] = -hc[k];
          Jt1[clk_idx + k] = hc[k];
        }

        for (int a = 0; a < ss; a++) {
          g[o0 + a] += Jt[a] * Jr;
          g[o1 + a] += Jt1[a] * Jr;
        }

        for (int a = 0; a < ss; a++)
          for (int b = 0; b < ss; b++) {
            H[(size_t)(o0 + a) * n_state + (o0 + b)] += w * Jt[a] * Jt[b];
            H[(size_t)(o1 + a) * n_state + (o1 + b)] += w * Jt1[a] * Jt1[b];
            H[(size_t)(o0 + a) * n_state + (o1 + b)] += w * Jt[a] * Jt1[b];
            H[(size_t)(o1 + a) * n_state + (o0 + b)] += w * Jt1[a] * Jt[b];
          }
      }
    }
  }
}

double tdcp_cost_host_vd(
    int n_epoch, int n_sat, int nc, int ss,
    const double* sat_ecef,
    const int* sys_kind_host,
    const double* dt_arr,
    const double* tdcp_meas,
    const double* tdcp_weights,
    double tdcp_sigma_m,
    bool tdcp_use_drift,
    const double* state) {
  if (!tdcp_meas) return 0.0;
  double e = 0.0;
  const int clk_idx = 6;
  const int drift_idx = 6 + nc;
  for (int t = 0; t < n_epoch - 1; t++) {
    const double dt = dt_arr ? dt_arr[t] : 0.0;
    if (tdcp_use_drift && (!std::isfinite(dt) || dt <= 0.0)) continue;
    int o0 = ss * t;
    int o1 = ss * (t + 1);
    const double x1 = state[o1 + 0], y1 = state[o1 + 1], z1 = state[o1 + 2];

    const double* my_sat = sat_ecef + (size_t)(t + 1) * n_sat * 3;

    for (int s = 0; s < n_sat; s++) {
      double w = 0.0;
      if (tdcp_weights) {
        w = tdcp_weights[(size_t)t * n_sat + s];
      } else if (tdcp_sigma_m > 0.0) {
        w = 1.0 / (tdcp_sigma_m * tdcp_sigma_m);
      }
      if (w <= 0.0) continue;

      double meas = tdcp_meas[(size_t)t * n_sat + s];
      if (meas == 0.0 && !tdcp_weights) continue;

      double sx = my_sat[s * 3 + 0], sy = my_sat[s * 3 + 1], sz = my_sat[s * 3 + 2];
      double dx0 = x1 - sx, dy0 = y1 - sy, dz0 = z1 - sz;
      double r0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
      double transit = r0 / kC;
      double theta = kOmegaE * transit;
      double sx_rot = sx * cos(theta) + sy * sin(theta);
      double sy_rot = -sx * sin(theta) + sy * cos(theta);
      double dx = x1 - sx_rot, dy_v = y1 - sy_rot, dz = z1 - sz;
      double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      if (r < 1e-6) continue;

      double ex = dx / r, ey = dy_v / r, ez = dz / r;
      int sk = sys_kind_host ? sys_kind_host[(t + 1) * n_sat + s] : 0;
      if (sk < 0 || sk >= nc) continue;
      double hc[kMaxClock];
      fill_hc_int(nc, sk, hc);
      double x0 = state[o0 + 0], y0 = state[o0 + 1], z0 = state[o0 + 2];
      double res = ex * (x1 - x0) + ey * (y1 - y0) + ez * (z1 - z0) - meas;
      if (tdcp_use_drift) {
        res += 0.5 * dt * (state[o0 + drift_idx] + state[o1 + drift_idx]);
      } else {
        for (int k = 0; k < nc; k++) {
          res += hc[k] * (state[o1 + clk_idx + k] - state[o0 + clk_idx + k]);
        }
      }
      e += 0.5 * w * res * res;
    }
  }
  return e;
}

}  // anonymous namespace

int fgo_gnss_lm_vd(const double* sat_ecef,
                   const double* pseudorange,
                   const double* weights,
                   const std::int32_t* sys_kind,
                   int n_clock,
                   double* state_io,
                   int n_epoch,
                   int n_sat,
                   double motion_sigma_m,
                   double clock_drift_sigma_m,
                   bool clock_use_average_drift,
                   double stop_velocity_sigma_mps,
                   double stop_position_sigma_m,
                   int max_iter,
                   double tol,
                   double huber_k,
                   int enable_line_search,
                   double* out_mse_pr,
                   const double* sat_vel,
                   const double* doppler,
                   const double* doppler_weights,
                   const double* dt,
                   const std::uint8_t* stop_mask,
                   const double* tdcp_meas,
                   const double* tdcp_weights,
                   double tdcp_sigma_m,
                   bool tdcp_use_drift,
                   double relative_height_sigma_m,
                   const double* enu_up_ecef,
                   int n_rel_height_edges,
                   const std::int32_t* rel_height_i,
                   const std::int32_t* rel_height_j,
                   const double* imu_delta_p,
                   const double* imu_delta_v,
                   double imu_position_sigma_m,
                   double imu_velocity_sigma_mps,
                   const double* sat_clock_drift,
                   const double* absolute_height_ref_ecef,
                   double absolute_height_sigma_m,
                   int state_stride,
                   double imu_accel_bias_prior_sigma_mps2,
                   double imu_accel_bias_between_sigma_mps2) {
  if (n_epoch < 1 || n_sat < 4 || !sat_ecef || !pseudorange || !weights || !state_io) return -1;
  if (n_clock < 1 || n_clock > kMaxClockVD) return -1;

  const int base_ss = 7 + n_clock;  // x,y,z,vx,vy,vz,clk...,drift
  const int ss = state_stride > 0 ? state_stride : base_ss;
  if (ss != base_ss && ss != base_ss + 3) return -1;
  const int accel_bias_idx = (ss == base_ss + 3) ? base_ss : -1;
  const int n_state = ss * n_epoch;
  if (n_state > 16384) return -1;  // larger limit for extended state

  std::vector<int> sys_buf((size_t)n_epoch * n_sat, 0);
  if (sys_kind != nullptr) {
    for (size_t i = 0; i < sys_buf.size(); i++)
      sys_buf[i] = static_cast<int>(sys_kind[i]);
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

  double w_clkdrift = 0.0;
  if (clock_drift_sigma_m > 0.0) w_clkdrift = 1.0 / (clock_drift_sigma_m * clock_drift_sigma_m);

  double w_stop_velocity = 0.0;
  if (stop_velocity_sigma_mps > 0.0) {
    w_stop_velocity = 1.0 / (stop_velocity_sigma_mps * stop_velocity_sigma_mps);
  }

  double w_stop_position = 0.0;
  if (stop_position_sigma_m > 0.0) {
    w_stop_position = 1.0 / (stop_position_sigma_m * stop_position_sigma_m);
  }

  double w_imu_pos = 0.0;
  if (imu_delta_p != nullptr && imu_position_sigma_m > 0.0) {
    w_imu_pos = 1.0 / (imu_position_sigma_m * imu_position_sigma_m);
  }

  double w_imu_vel = 0.0;
  if (imu_delta_v != nullptr && imu_velocity_sigma_mps > 0.0) {
    w_imu_vel = 1.0 / (imu_velocity_sigma_mps * imu_velocity_sigma_mps);
  }

  double w_imu_accel_bias_prior = 0.0;
  if (accel_bias_idx >= 0 && imu_accel_bias_prior_sigma_mps2 > 0.0) {
    w_imu_accel_bias_prior = 1.0 / (imu_accel_bias_prior_sigma_mps2 * imu_accel_bias_prior_sigma_mps2);
  }

  double w_imu_accel_bias_between = 0.0;
  if (accel_bias_idx >= 0 && imu_accel_bias_between_sigma_mps2 > 0.0) {
    w_imu_accel_bias_between =
        1.0 / (imu_accel_bias_between_sigma_mps2 * imu_accel_bias_between_sigma_mps2);
  }

  double w_rel_height = 0.0;
  double rh_ux = 0.0, rh_uy = 0.0, rh_uz = 0.0;
  const std::int32_t* rh_i_ptr = nullptr;
  const std::int32_t* rh_j_ptr = nullptr;
  int rh_n_edges = 0;
  if (enu_up_ecef != nullptr) {
    rh_ux = enu_up_ecef[0];
    rh_uy = enu_up_ecef[1];
    rh_uz = enu_up_ecef[2];
  }
  if (relative_height_sigma_m > 0.0 && enu_up_ecef != nullptr && n_rel_height_edges > 0 && rel_height_i != nullptr &&
      rel_height_j != nullptr) {
    w_rel_height = 1.0 / (relative_height_sigma_m * relative_height_sigma_m);
    rh_n_edges = n_rel_height_edges;
    rh_i_ptr = rel_height_i;
    rh_j_ptr = rel_height_j;
  }

  double w_abs_height = 0.0;
  const double* abs_height_ref_ptr = nullptr;
  if (absolute_height_sigma_m > 0.0 && enu_up_ecef != nullptr && absolute_height_ref_ecef != nullptr) {
    w_abs_height = 1.0 / (absolute_height_sigma_m * absolute_height_sigma_m);
    abs_height_ref_ptr = absolute_height_ref_ecef;
  }

  int total_iters = 0;
  bool ok = false;
  const int block = 256;

  for (int it = 0; it < max_iter; it++) {
    effective_pr_weights_huber_host_vd(
        n_epoch, n_sat, n_clock, ss, sat_ecef, pseudorange, weights, sys_host, state_io,
        huber_k, h_eff_w);
    CUDA_CHECK(cudaMemcpy(d_w, h_eff_w, sz_ws, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_H, 0, sz_H));
    CUDA_CHECK(cudaMemset(d_g, 0, sz_state));

    int grid_pr = (n_epoch + block - 1) / block;
    fgo_assemble_pseudorange_vd<<<grid_pr, block>>>(
        n_epoch, n_sat, n_clock, ss, n_state, d_sat, d_pr, d_w, d_sys, d_state, d_H, d_g);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(h_H, d_H, sz_H, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_g, d_g, sz_state, cudaMemcpyDeviceToHost));

    // Add host-side factors
    add_motion_factor_host(n_epoch, ss, n_state, w_motion, state_io, dt, h_H, h_g);
    add_clock_drift_factor_host(n_epoch, n_clock, ss, n_state, w_clkdrift, state_io, dt,
                                clock_use_average_drift, h_H, h_g);
    add_stop_velocity_factor_host(n_epoch, ss, n_state, w_stop_velocity, stop_mask, state_io, h_H, h_g);
    add_stop_position_factor_host(n_epoch, ss, n_state, w_stop_position, stop_mask, state_io, h_H, h_g);
    add_imu_prior_factor_host(n_epoch, ss, n_state, w_imu_pos, w_imu_vel, imu_delta_p, imu_delta_v, state_io, dt,
                              accel_bias_idx, h_H, h_g);
    add_accel_bias_factor_host(n_epoch, ss, n_state, accel_bias_idx, w_imu_accel_bias_prior,
                               w_imu_accel_bias_between, state_io, h_H, h_g);
    add_doppler_factor_host(n_epoch, n_sat, n_clock, ss, n_state,
                            sat_ecef, sat_vel, sat_clock_drift, doppler, doppler_weights, sys_host, state_io,
                            h_H, h_g);
    add_tdcp_factor_host_vd(n_epoch, n_sat, n_clock, ss, n_state, sat_ecef, sys_host, dt,
                            tdcp_meas, tdcp_weights, tdcp_sigma_m, tdcp_use_drift, state_io, h_H, h_g);
    add_relative_height_factor_host(n_epoch, ss, n_state, w_rel_height, rh_ux, rh_uy, rh_uz, rh_n_edges, rh_i_ptr,
                                    rh_j_ptr, state_io, h_H, h_g);
    add_absolute_height_factor_host(n_epoch, ss, n_state, w_abs_height, rh_ux, rh_uy, rh_uz, abs_height_ref_ptr,
                                    state_io, h_H, h_g);

    double cost_before =
        pr_cost_host_vd(n_epoch, n_sat, n_clock, ss, sat_ecef, pseudorange, weights, sys_host, state_io, huber_k)
        + motion_factor_cost_host(n_epoch, ss, w_motion, state_io, dt)
        + clock_drift_cost_host(n_epoch, n_clock, ss, w_clkdrift, state_io, dt, clock_use_average_drift)
        + stop_velocity_cost_host(n_epoch, ss, w_stop_velocity, stop_mask, state_io)
        + stop_position_cost_host(n_epoch, ss, w_stop_position, stop_mask, state_io)
        + imu_prior_cost_host(n_epoch, ss, w_imu_pos, w_imu_vel, imu_delta_p, imu_delta_v, state_io, dt,
                              accel_bias_idx)
        + accel_bias_cost_host(n_epoch, ss, accel_bias_idx, w_imu_accel_bias_prior, w_imu_accel_bias_between,
                               state_io)
        + doppler_cost_host(n_epoch, n_sat, n_clock, ss, sat_ecef, sat_vel, sat_clock_drift, doppler, doppler_weights,
                            state_io)
        + tdcp_cost_host_vd(n_epoch, n_sat, n_clock, ss, sat_ecef, sys_host, dt, tdcp_meas, tdcp_weights, tdcp_sigma_m,
                            tdcp_use_drift, state_io)
        + relative_height_cost_host(n_epoch, ss, w_rel_height, rh_ux, rh_uy, rh_uz, rh_n_edges, rh_i_ptr, rh_j_ptr,
                                    state_io)
        + absolute_height_cost_host(n_epoch, ss, w_abs_height, rh_ux, rh_uy, rh_uz, abs_height_ref_ptr, state_io);

    // NOTE: Unlike the original fgo_gnss_lm which negates h_g, the VD solver
    // solves H * delta = g directly. All factors accumulate g = J^T * W * r
    // (the RHS of the Gauss-Newton normal equation), so the correct step is
    // delta = H^{-1} * g without negation.

    std::memcpy(h_work, h_H, sz_H);
    // Diagonal regularization: stronger for velocity/drift to ensure
    // positive definiteness even when unconstrained by factors.
    // Velocity (indices 3,4,5) and drift (index 6+nc) get ~1e6 times
    // stronger regularization than position/clock, which is still very
    // weak (equivalent to a 1000 m/s sigma prior).
    {
      constexpr double kVelDriftJitter = 1e-6;  // weak prior ~1000 m/s sigma
      for (int t2 = 0; t2 < n_epoch; t2++) {
        int off = ss * t2;
        for (int d2 = 0; d2 < ss; d2++) {
          double jit = kDiagJitter;
          if (d2 >= 3 && d2 <= 5) jit = kVelDriftJitter;  // velocity
          if (d2 == 6 + n_clock) jit = kVelDriftJitter;    // drift
          if (accel_bias_idx >= 0 && d2 >= accel_bias_idx && d2 < accel_bias_idx + 3) {
            jit = kVelDriftJitter;  // accelerometer bias
          }
          h_work[(size_t)(off + d2) * n_state + (off + d2)] += jit;
        }
      }
    }
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
        double ctry =
            pr_cost_host_vd(n_epoch, n_sat, n_clock, ss, sat_ecef, pseudorange, weights, sys_host, trial, huber_k)
            + motion_factor_cost_host(n_epoch, ss, w_motion, trial, dt)
            + clock_drift_cost_host(n_epoch, n_clock, ss, w_clkdrift, trial, dt, clock_use_average_drift)
            + stop_velocity_cost_host(n_epoch, ss, w_stop_velocity, stop_mask, trial)
            + stop_position_cost_host(n_epoch, ss, w_stop_position, stop_mask, trial)
            + imu_prior_cost_host(n_epoch, ss, w_imu_pos, w_imu_vel, imu_delta_p, imu_delta_v, trial, dt,
                                  accel_bias_idx)
            + accel_bias_cost_host(n_epoch, ss, accel_bias_idx, w_imu_accel_bias_prior, w_imu_accel_bias_between,
                                   trial)
            + doppler_cost_host(n_epoch, n_sat, n_clock, ss, sat_ecef, sat_vel, sat_clock_drift, doppler,
                                doppler_weights, trial)
            + tdcp_cost_host_vd(n_epoch, n_sat, n_clock, ss, sat_ecef, sys_host, dt, tdcp_meas, tdcp_weights, tdcp_sigma_m,
                                tdcp_use_drift, trial)
            + relative_height_cost_host(n_epoch, ss, w_rel_height, rh_ux, rh_uy, rh_uz, rh_n_edges, rh_i_ptr,
                                        rh_j_ptr, trial)
            + absolute_height_cost_host(n_epoch, ss, w_abs_height, rh_ux, rh_uy, rh_uz, abs_height_ref_ptr, trial);
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
    *out_mse_pr = compute_pr_mse_host_vd(n_epoch, n_sat, n_clock, ss, sat_ecef, pseudorange, weights, sys_host, state_io);

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
