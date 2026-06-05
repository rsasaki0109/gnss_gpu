#include "gnss_gpu/fgo.h"
#include "gnss_gpu/cuda_check.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
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

inline double huber_effective_weight(double w, double res, double huber_k) {
  if (w <= 0.0 || huber_k <= 0.0) return w;
  const double z_m = std::sqrt(w) * std::fabs(res);
  if (z_m <= huber_k || z_m <= 0.0) return w;
  return w * (huber_k / z_m);
}

inline double huber_loss(double w, double res, double huber_k) {
  if (w <= 0.0) return 0.0;
  if (huber_k <= 0.0) return 0.5 * w * res * res;
  const double z_m = std::sqrt(w) * std::fabs(res);
  if (z_m <= huber_k) return 0.5 * z_m * z_m;
  return huber_k * z_m - 0.5 * huber_k * huber_k;
}

inline double huber_weight_scale_from_whitened_norm(double z_norm, double huber_k) {
  if (huber_k <= 0.0 || z_norm <= huber_k || z_norm <= 0.0) return 1.0;
  return huber_k / z_norm;
}

inline double huber_loss_from_whitened_norm(double z_norm, double huber_k) {
  if (z_norm <= 0.0) return 0.0;
  if (huber_k <= 0.0 || z_norm <= huber_k) return 0.5 * z_norm * z_norm;
  return huber_k * z_norm - 0.5 * huber_k * huber_k;
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
    double tdcp_huber_k,
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
      const double w_eff = huber_effective_weight(w, res, tdcp_huber_k);

      // J_pred at x_t: d(pred)/d(x_t) = [-ex,-ey,-ez], d(pred)/d(clk_t) = -1
      // J_pred at x_{t+1}: d(pred)/d(x_{t+1}) = [+ex,+ey,+ez], d(pred)/d(clk_{t+1}) = +1
      double Jr = w_eff * res;

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
          H[(size_t)(o0 + a) * n_state + (o0 + b)] += w_eff * Jt[a] * Jt[b];
          H[(size_t)(o1 + a) * n_state + (o1 + b)] += w_eff * Jt1[a] * Jt1[b];
          H[(size_t)(o0 + a) * n_state + (o1 + b)] += w_eff * Jt[a] * Jt1[b];
          H[(size_t)(o1 + a) * n_state + (o0 + b)] += w_eff * Jt1[a] * Jt[b];
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
    double tdcp_huber_k,
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
      e += huber_loss(w, res, tdcp_huber_k);
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
                double tdcp_sigma_m,
                double tdcp_huber_k) {
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
                         tdcp_meas, tdcp_weights, tdcp_sigma_m, tdcp_huber_k, state_io, h_H, h_g);

    double cost_before =
        pr_cost_host(n_epoch, n_sat, n_clock, ss, sat_ecef, pseudorange, weights, sys_host, state_io,
                     huber_k) +
        motion_cost_host(n_epoch, ss, w_motion, state_io, motion_displacement) +
        tdcp_cost_host(n_epoch, n_sat, n_clock, ss, sat_ecef, sys_host, tdcp_meas, tdcp_weights, tdcp_sigma_m,
                       tdcp_huber_k, state_io);

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
                       + tdcp_cost_host(n_epoch, n_sat, n_clock, ss, sat_ecef, sys_host, tdcp_meas, tdcp_weights,
                                        tdcp_sigma_m, tdcp_huber_k, trial);
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
constexpr int kMaxSSVD = 19 + kMaxClockVD;  // max VD state size per epoch
constexpr double kClockConstrainedWeight = 1000.0;  // GTSAM Constrained::MixedSigmas zero-sigma default mu.
constexpr double kPosePointConstrainedWeight = 1000.0;  // GTSAM Constrained::All default mu.

void dump_dense_matrix_csv(const std::string& path, const double* data, int rows, int cols) {
  std::ofstream out(path);
  if (!out) return;
  out.precision(17);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      if (c > 0) out << ',';
      out << data[(size_t)r * cols + c];
    }
    out << '\n';
  }
}

void dump_vector_csv(const std::string& path, const double* data, int n) {
  std::ofstream out(path);
  if (!out) return;
  out.precision(17);
  for (int i = 0; i < n; i++) out << data[i] << '\n';
}

bool pr_prediction_vd(
    int n_sat, int nc, int ss, int t, int s,
    const double* sat_ecef,
    const double* pr_linearization_ref_ecef,
    const double* pr_linearization_los_ecef,
    const int* sys_kind_host,
    const double* state,
    double* pred,
    double* j_pos,
    double* hc) {
  const int sk = sys_kind_host ? sys_kind_host[t * n_sat + s] : 0;
  if (sk < 0 || sk >= nc) return false;
  fill_hc_int(nc, sk, hc);
  const int o = ss * t;
  const double x = state[o + 0], y = state[o + 1], z = state[o + 2];
  const double* cptr = state + o + 6;
  double clk = 0.0;
  for (int k = 0; k < nc; k++) clk += hc[k] * cptr[k];

  if (pr_linearization_ref_ecef && pr_linearization_los_ecef) {
    const double* ref = pr_linearization_ref_ecef + (size_t)t * 3;
    const double* los = pr_linearization_los_ecef + ((size_t)t * n_sat + s) * 3;
    if (!std::isfinite(ref[0]) || !std::isfinite(ref[1]) || !std::isfinite(ref[2]) ||
        !std::isfinite(los[0]) || !std::isfinite(los[1]) || !std::isfinite(los[2])) {
      return false;
    }
    j_pos[0] = los[0];
    j_pos[1] = los[1];
    j_pos[2] = los[2];
    *pred = los[0] * (x - ref[0]) + los[1] * (y - ref[1]) + los[2] * (z - ref[2]) + clk;
    return true;
  }

  if (!sat_ecef) return false;
  const double* my_sat = sat_ecef + (size_t)t * n_sat * 3;
  const double sx = my_sat[s * 3 + 0], sy = my_sat[s * 3 + 1], sz = my_sat[s * 3 + 2];
  const double dx0 = x - sx, dy0 = y - sy, dz0 = z - sz;
  const double r0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
  const double transit = r0 / kC;
  const double theta = kOmegaE * transit;
  const double sx_rot = sx * cos(theta) + sy * sin(theta);
  const double sy_rot = -sx * sin(theta) + sy * cos(theta);
  const double dx = x - sx_rot, dy_v = y - sy_rot, dz = z - sz;
  const double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
  if (r < 1e-6) return false;
  j_pos[0] = dx / r;
  j_pos[1] = dy_v / r;
  j_pos[2] = dz / r;
  *pred = r + clk;
  return true;
}

bool doppler_prediction_vd(
    int n_sat, int nc, int ss, int t, int s,
    const double* sat_ecef,
    const double* sat_vel,
    const double* sat_clock_drift,
    const double* doppler_linearization_ref_vel,
    const double* doppler_linearization_los_ecef,
    const double* state,
    double* pred,
    double* j_vel) {
  const int o = ss * t;
  const int drift_idx = 6 + nc;
  const double x = state[o + 0], y = state[o + 1], z = state[o + 2];
  const double vx = state[o + 3], vy = state[o + 4], vz = state[o + 5];
  const double drift = state[o + drift_idx];

  if (doppler_linearization_ref_vel && doppler_linearization_los_ecef) {
    const double* ref = doppler_linearization_ref_vel + (size_t)t * 3;
    const double* los = doppler_linearization_los_ecef + ((size_t)t * n_sat + s) * 3;
    if (!std::isfinite(ref[0]) || !std::isfinite(ref[1]) || !std::isfinite(ref[2]) ||
        !std::isfinite(los[0]) || !std::isfinite(los[1]) || !std::isfinite(los[2])) {
      return false;
    }
    j_vel[0] = los[0];
    j_vel[1] = los[1];
    j_vel[2] = los[2];
    *pred = los[0] * (vx - ref[0]) + los[1] * (vy - ref[1]) + los[2] * (vz - ref[2]) + drift;
    return true;
  }

  if (!sat_ecef || !sat_vel) return false;
  const double* my_sat = sat_ecef + (size_t)t * n_sat * 3;
  const double* my_sv = sat_vel + (size_t)t * n_sat * 3;
  const double* my_scd = sat_clock_drift ? sat_clock_drift + (size_t)t * n_sat : nullptr;
  const double sx = my_sat[s * 3 + 0], sy = my_sat[s * 3 + 1], sz = my_sat[s * 3 + 2];
  const double dx = sx - x, dy_v = sy - y, dz = sz - z;
  const double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
  if (r < 1e-6) return false;
  const double los_x = dx / r, los_y = dy_v / r, los_z = dz / r;
  const double svx = my_sv[s * 3 + 0], svy = my_sv[s * 3 + 1], svz = my_sv[s * 3 + 2];
  const double euclidean_rate = los_x * (svx - vx) + los_y * (svy - vy) +
                                los_z * (svz - vz);
  const double sag_rate = sagnac_range_rate_mps(sx, sy, svx, svy, x, y, vx, vy);
  const double sat_clk_drift = (my_scd && std::isfinite(my_scd[s])) ? my_scd[s] : 0.0;
  *pred = drift + (euclidean_rate - sag_rate - sat_clk_drift);
  j_vel[0] = -los_x + kOmegaE * sy / kC;
  j_vel[1] = -los_y - kOmegaE * sx / kC;
  j_vel[2] = -los_z;
  return true;
}

// Doppler factor: res = doppler_obs - (drift + geometric_range_rate)
// where geometric_range_rate follows the RTKLIB convention, including the
// first-order Sagnac range-rate correction and optional satellite clock drift.
// Jacobian w.r.t. state [x,y,z,vx,vy,vz,clk...,drift]:
//   d/dv = d(geometric_range_rate)/d(receiver velocity)
//   d/ddrift = 1  (index 6+nc)
// NOTE: we do not differentiate the unit vector w.r.t. position here (standard
// linearization around the current position estimate).
void add_doppler_factor_host(
    int n_epoch, int n_sat, int nc, int ss, int n_state,
    const double* sat_ecef,
    const double* sat_vel,
    const double* sat_clock_drift,
    const double* doppler_linearization_ref_vel,
    const double* doppler_linearization_los_ecef,
    const double* doppler,
    const double* doppler_weights,
    const int* sys_kind,
    double doppler_huber_k,
    const double* state,
    double* H, double* g) {
  if (!doppler || !doppler_weights) return;
  const int drift_idx = 6 + nc;

  for (int t = 0; t < n_epoch; t++) {
    const double* my_dop = doppler + (size_t)t * n_sat;
    const double* my_dw = doppler_weights + (size_t)t * n_sat;

    int o = ss * t;

    for (int s = 0; s < n_sat; s++) {
      double w = my_dw[s];
      if (w <= 0.0) continue;

      double pred = 0.0;
      double Jv[3] = {};
      if (!doppler_prediction_vd(n_sat, nc, ss, t, s, sat_ecef, sat_vel, sat_clock_drift,
                                 doppler_linearization_ref_vel, doppler_linearization_los_ecef,
                                 state, &pred, Jv)) {
        continue;
      }
      double res = my_dop[s] - pred;
      const double w_eff = huber_effective_weight(w, res, doppler_huber_k);

      // Jacobian w.r.t. [vx, vy, vz] at indices [3,4,5] and drift.
      double Jd = 1.0;

      // Gradient: g += J * w * res (standard J^T * W * r convention, same as PR and motion)
      double Jr = res * w_eff;
      // Velocity-velocity block
      for (int a = 0; a < 3; a++) {
        g[o + 3 + a] += Jv[a] * Jr;
        for (int b = 0; b < 3; b++) {
          H[(size_t)(o + 3 + a) * n_state + (o + 3 + b)] += w_eff * Jv[a] * Jv[b];
        }
        // Velocity-drift cross
        H[(size_t)(o + 3 + a) * n_state + (o + drift_idx)] += w_eff * Jv[a] * Jd;
        H[(size_t)(o + drift_idx) * n_state + (o + 3 + a)] += w_eff * Jd * Jv[a];
      }
      // Drift-drift
      g[o + drift_idx] += Jd * Jr;
      H[(size_t)(o + drift_idx) * n_state + (o + drift_idx)] += w_eff * Jd * Jd;
    }
  }
}

// Motion factor matching gtsam_gnss MotionFactor_XXVV:
// residual_i = x_{t+1,i} - x_{t,i} - (v_{t,i} + v_{t+1,i}) * dt / 2
// Couples adjacent positions and velocities with a trapezoidal velocity model.
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
      double v_t1 = state[o1 + 3 + i];
      double x_t1 = state[o1 + i];
      double half_dt = 0.5 * dt;
      double res = x_t1 - x_t - (v_t + v_t1) * half_dt;

      // Jacobian: d/d(x_t,i)=-1, d/d(x_{t+1},i)=1,
      // d/d(v_t,i)=d/d(v_{t+1},i)=-dt/2.  The VD solver uses
      // H * delta = -J^T W r for zero-measurement residual factors.
      g[o0 + i]     += w_motion * res;
      g[o1 + i]     -= w_motion * res;
      g[o0 + 3 + i] += w_motion * res * half_dt;
      g[o1 + 3 + i] += w_motion * res * half_dt;

      // Hessian contributions: J^T W J
      H[(size_t)(o0 + i) * n_state + (o0 + i)] += w_motion;
      H[(size_t)(o0 + i) * n_state + (o1 + i)] += -w_motion;
      H[(size_t)(o1 + i) * n_state + (o0 + i)] += -w_motion;
      H[(size_t)(o1 + i) * n_state + (o1 + i)] += w_motion;

      H[(size_t)(o0 + i) * n_state + (o0 + 3 + i)] += w_motion * half_dt;
      H[(size_t)(o0 + 3 + i) * n_state + (o0 + i)] += w_motion * half_dt;
      H[(size_t)(o0 + i) * n_state + (o1 + 3 + i)] += w_motion * half_dt;
      H[(size_t)(o1 + 3 + i) * n_state + (o0 + i)] += w_motion * half_dt;

      H[(size_t)(o1 + i) * n_state + (o0 + 3 + i)] += -w_motion * half_dt;
      H[(size_t)(o0 + 3 + i) * n_state + (o1 + i)] += -w_motion * half_dt;
      H[(size_t)(o1 + i) * n_state + (o1 + 3 + i)] += -w_motion * half_dt;
      H[(size_t)(o1 + 3 + i) * n_state + (o1 + i)] += -w_motion * half_dt;

      H[(size_t)(o0 + 3 + i) * n_state + (o0 + 3 + i)] += w_motion * half_dt * half_dt;
      H[(size_t)(o0 + 3 + i) * n_state + (o1 + 3 + i)] += w_motion * half_dt * half_dt;
      H[(size_t)(o1 + 3 + i) * n_state + (o0 + 3 + i)] += w_motion * half_dt * half_dt;
      H[(size_t)(o1 + 3 + i) * n_state + (o1 + 3 + i)] += w_motion * half_dt * half_dt;
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

    if (clock_use_average_drift) {
      const double half_dt = 0.5 * dt;
      const double d_t = state[o0 + drift_idx];
      const double d_t1 = state[o1 + drift_idx];
      for (int k = 0; k < nc; k++) {
        const double w_clk = (k == 0) ? w_clkdrift : kClockConstrainedWeight;
        const int c0 = o0 + clk_idx + k;
        const int c1 = o1 + clk_idx + k;
        double res = state[c0] - state[c1];
        if (k == 0) {
          res += half_dt * (d_t + d_t1);
        }

        g[c0] -= w_clk * res;
        g[c1] += w_clk * res;

        H[(size_t)c0 * n_state + c0] += w_clk;
        H[(size_t)c0 * n_state + c1] += -w_clk;
        H[(size_t)c1 * n_state + c0] += -w_clk;
        H[(size_t)c1 * n_state + c1] += w_clk;

        if (k == 0) {
          g[o0 + drift_idx] -= w_clk * res * half_dt;
          g[o1 + drift_idx] -= w_clk * res * half_dt;

          H[(size_t)c0 * n_state + (o0 + drift_idx)] += w_clk * half_dt;
          H[(size_t)(o0 + drift_idx) * n_state + c0] += w_clk * half_dt;
          H[(size_t)c0 * n_state + (o1 + drift_idx)] += w_clk * half_dt;
          H[(size_t)(o1 + drift_idx) * n_state + c0] += w_clk * half_dt;
          H[(size_t)c1 * n_state + (o0 + drift_idx)] += -w_clk * half_dt;
          H[(size_t)(o0 + drift_idx) * n_state + c1] += -w_clk * half_dt;
          H[(size_t)c1 * n_state + (o1 + drift_idx)] += -w_clk * half_dt;
          H[(size_t)(o1 + drift_idx) * n_state + c1] += -w_clk * half_dt;
          H[(size_t)(o0 + drift_idx) * n_state + (o0 + drift_idx)] += w_clk * half_dt * half_dt;
          H[(size_t)(o1 + drift_idx) * n_state + (o1 + drift_idx)] += w_clk * half_dt * half_dt;
          H[(size_t)(o0 + drift_idx) * n_state + (o1 + drift_idx)] += w_clk * half_dt * half_dt;
          H[(size_t)(o1 + drift_idx) * n_state + (o0 + drift_idx)] += w_clk * half_dt * half_dt;
        }
      }
      continue;
    }

    double c_t = state[o0 + clk_idx];
    double d_t = state[o0 + drift_idx];
    double c_t1 = state[o1 + clk_idx];
    double res = c_t - c_t1 + d_t * dt;

    g[o0 + clk_idx]   -= w_clkdrift * res;
    g[o1 + clk_idx]   += w_clkdrift * res;
    g[o0 + drift_idx] -= w_clkdrift * res * dt;

    H[(size_t)(o0 + clk_idx) * n_state + (o0 + clk_idx)] += w_clkdrift;
    H[(size_t)(o0 + clk_idx) * n_state + (o1 + clk_idx)] += -w_clkdrift;
    H[(size_t)(o1 + clk_idx) * n_state + (o0 + clk_idx)] += -w_clkdrift;
    H[(size_t)(o1 + clk_idx) * n_state + (o1 + clk_idx)] += w_clkdrift;
    H[(size_t)(o0 + clk_idx) * n_state + (o0 + drift_idx)] += w_clkdrift * dt;
    H[(size_t)(o0 + drift_idx) * n_state + (o0 + clk_idx)] += w_clkdrift * dt;
    H[(size_t)(o0 + drift_idx) * n_state + (o0 + drift_idx)] += w_clkdrift * dt * dt;
    H[(size_t)(o0 + drift_idx) * n_state + (o1 + clk_idx)] += -w_clkdrift * dt;
    H[(size_t)(o1 + clk_idx) * n_state + (o0 + drift_idx)] += -w_clkdrift * dt;
  }
}

void add_stop_velocity_factor_host(
    int n_epoch, int ss, int n_state, double w_stop_velocity,
    double stop_velocity_huber_k,
    const std::uint8_t* stop_mask, const double* state, double* H, double* g) {
  if (w_stop_velocity <= 0.0 || !stop_mask) return;

  for (int t = 0; t < n_epoch; t++) {
    if (stop_mask[t] == 0) continue;
    const int o = ss * t;
    double z2 = 0.0;
    for (int i = 0; i < 3; i++) {
      const double res = -state[o + 3 + i];
      z2 += w_stop_velocity * res * res;
    }
    const double w_eff = w_stop_velocity *
                         huber_weight_scale_from_whitened_norm(std::sqrt(z2), stop_velocity_huber_k);
    for (int i = 0; i < 3; i++) {
      const int idx = o + 3 + i;
      const double res = -state[idx];
      g[idx] += w_eff * res;
      H[(size_t)idx * n_state + idx] += w_eff;
    }
  }
}

void add_stop_position_factor_host(
    int n_epoch, int ss, int n_state, double w_stop_position,
    double stop_position_huber_k,
    const std::uint8_t* stop_mask, const double* state, double* H, double* g) {
  if (w_stop_position <= 0.0 || !stop_mask) return;

  for (int t = 0; t < n_epoch - 1; t++) {
    if (stop_mask[t] == 0 || stop_mask[t + 1] == 0) continue;
    const int o0 = ss * t;
    const int o1 = ss * (t + 1);
    for (int i = 0; i < 3; i++) {
      const double res = state[o0 + i] - state[o1 + i];
      const double w_eff = huber_effective_weight(w_stop_position, res, stop_position_huber_k);
      g[o0 + i] += -w_eff * res;
      g[o1 + i] += w_eff * res;
      H[(size_t)(o0 + i) * n_state + (o0 + i)] += w_eff;
      H[(size_t)(o1 + i) * n_state + (o1 + i)] += w_eff;
      H[(size_t)(o0 + i) * n_state + (o1 + i)] += -w_eff;
      H[(size_t)(o1 + i) * n_state + (o0 + i)] += -w_eff;
    }
  }
}

void add_pr_factor_host_vd(
    int n_epoch, int n_sat, int nc, int ss, int n_state,
    const double* sat_ecef,
    const double* pr_linearization_ref_ecef,
    const double* pr_linearization_los_ecef,
    const double* pseudorange,
    const double* weights,
    const int* sys_kind_host,
    const double* state,
    double* H,
    double* g) {
  if (!pseudorange || !weights) return;
  for (int t = 0; t < n_epoch; t++) {
    const int o = ss * t;
    const double* my_pr = pseudorange + (size_t)t * n_sat;
    const double* my_w = weights + (size_t)t * n_sat;
    for (int s = 0; s < n_sat; s++) {
      const double w = my_w[s];
      if (w <= 0.0) continue;
      double hc[kMaxClockVD];
      double j_pos[3] = {};
      double pred = 0.0;
      if (!pr_prediction_vd(n_sat, nc, ss, t, s, sat_ecef, pr_linearization_ref_ecef,
                            pr_linearization_los_ecef, sys_kind_host, state, &pred, j_pos, hc)) {
        continue;
      }
      const double res = my_pr[s] - pred;

      double J[kMaxSSVD] = {};
      J[0] = j_pos[0];
      J[1] = j_pos[1];
      J[2] = j_pos[2];
      for (int k = 0; k < nc; k++) J[6 + k] = hc[k];

      const double Jr = res * w;
      for (int a = 0; a < ss; a++) {
        if (J[a] == 0.0) continue;
        g[o + a] += J[a] * Jr;
        for (int b = 0; b < ss; b++) {
          if (J[b] == 0.0) continue;
          H[(size_t)(o + a) * n_state + (o + b)] += w * J[a] * J[b];
        }
      }
    }
  }
}

// PR cost for VD state (clock offset at index 6)
double pr_cost_host_vd(
    int n_epoch, int n_sat, int nc, int ss,
    const double* sat_ecef,
    const double* pr_linearization_ref_ecef,
    const double* pr_linearization_los_ecef,
    const double* pseudorange,
    const double* weights,
    const int* sys_kind_host,
    const double* state,
    double huber_k) {
  double e = 0.0;
  for (int t = 0; t < n_epoch; t++) {
    const double* my_pr = pseudorange + (size_t)t * n_sat;
    const double* my_w = weights + (size_t)t * n_sat;
    for (int s = 0; s < n_sat; s++) {
      double w = my_w[s];
      if (w <= 0.0) continue;
      double hc[kMaxClockVD];
      double j_pos[3] = {};
      double pred = 0.0;
      if (!pr_prediction_vd(n_sat, nc, ss, t, s, sat_ecef, pr_linearization_ref_ecef,
                            pr_linearization_los_ecef, sys_kind_host, state, &pred, j_pos, hc)) {
        continue;
      }
      double res = my_pr[s] - pred;
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
      double res = state[o1 + i] - state[o0 + i]
                   - (state[o0 + 3 + i] + state[o1 + 3 + i]) * dt * 0.5;
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
    if (clock_use_average_drift) {
      const double half_dt = 0.5 * dt;
      for (int k = 0; k < nc; k++) {
        double res = state[o0 + clk_idx + k] - state[o1 + clk_idx + k];
        if (k == 0) {
          res += half_dt * (state[o0 + drift_idx] + state[o1 + drift_idx]);
        }
        const double w_clk = (k == 0) ? w_clkdrift : kClockConstrainedWeight;
        e += 0.5 * w_clk * res * res;
      }
      continue;
    }
    double res = state[o0 + clk_idx] - state[o1 + clk_idx] + state[o0 + drift_idx] * dt;
    e += 0.5 * w_clkdrift * res * res;
  }
  return e;
}

double stop_velocity_cost_host(
    int n_epoch, int ss, double w_stop_velocity,
    double stop_velocity_huber_k,
    const std::uint8_t* stop_mask, const double* state) {
  if (w_stop_velocity <= 0.0 || !stop_mask) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch; t++) {
    if (stop_mask[t] == 0) continue;
    const int o = ss * t;
    double z2 = 0.0;
    for (int i = 0; i < 3; i++) {
      const double res = state[o + 3 + i];
      z2 += w_stop_velocity * res * res;
    }
    e += huber_loss_from_whitened_norm(std::sqrt(z2), stop_velocity_huber_k);
  }
  return e;
}

double stop_position_cost_host(
    int n_epoch, int ss, double w_stop_position,
    double stop_position_huber_k,
    const std::uint8_t* stop_mask, const double* state) {
  if (w_stop_position <= 0.0 || !stop_mask) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch - 1; t++) {
    if (stop_mask[t] == 0 || stop_mask[t + 1] == 0) continue;
    const int o0 = ss * t;
    const int o1 = ss * (t + 1);
    for (int i = 0; i < 3; i++) {
      const double res = state[o0 + i] - state[o1 + i];
      e += huber_loss(w_stop_position, res, stop_position_huber_k);
    }
  }
  return e;
}

inline void add_linear_residual_host(
    int n_terms, const int* idx, const double* jac, double weight, double res,
    int n_state, double* H, double* g) {
  const double Jr = weight * res;
  for (int a = 0; a < n_terms; a++) {
    if (idx[a] < 0 || jac[a] == 0.0) continue;
    g[idx[a]] -= jac[a] * Jr;
    for (int b = 0; b < n_terms; b++) {
      if (idx[b] < 0 || jac[b] == 0.0) continue;
      H[(size_t)idx[a] * n_state + idx[b]] += weight * jac[a] * jac[b];
    }
  }
}

inline void skew3(const double* v, double K[9]) {
  K[0] = 0.0;
  K[1] = -v[2];
  K[2] = v[1];
  K[3] = v[2];
  K[4] = 0.0;
  K[5] = -v[0];
  K[6] = -v[1];
  K[7] = v[0];
  K[8] = 0.0;
}

inline void mat3_mul(const double A[9], const double B[9], double C[9]) {
  double tmp[9];
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      tmp[r * 3 + c] = 0.0;
      for (int k = 0; k < 3; k++) {
        tmp[r * 3 + c] += A[r * 3 + k] * B[k * 3 + c];
      }
    }
  }
  for (int i = 0; i < 9; i++) C[i] = tmp[i];
}

inline void mat3_transpose(const double A[9], double AT[9]) {
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      AT[r * 3 + c] = A[c * 3 + r];
    }
  }
}

inline void mat3_vec_mul(const double A[9], const double v[3], double out[3]) {
  for (int r = 0; r < 3; r++) {
    out[r] = A[r * 3 + 0] * v[0] + A[r * 3 + 1] * v[1] + A[r * 3 + 2] * v[2];
  }
}

inline bool finite3(const double* v) {
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

inline void rotvec_to_rotm_host(const double* rv, double R[9]) {
  const double theta = std::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
  double K[9];
  skew3(rv, K);
  double K2[9];
  mat3_mul(K, K, K2);
  double a = 1.0;
  double b = 0.5;
  if (theta >= 1e-12) {
    a = std::sin(theta) / theta;
    b = (1.0 - std::cos(theta)) / (theta * theta);
  }
  for (int i = 0; i < 9; i++) R[i] = a * K[i] + b * K2[i];
  R[0] += 1.0;
  R[4] += 1.0;
  R[8] += 1.0;
}

inline void rotm_to_rotvec_host(const double R[9], double rv[3]) {
  double cos_theta = 0.5 * (R[0] + R[4] + R[8] - 1.0);
  if (cos_theta > 1.0) cos_theta = 1.0;
  if (cos_theta < -1.0) cos_theta = -1.0;
  const double theta = std::acos(cos_theta);
  const double vee[3] = {
      R[7] - R[5],
      R[2] - R[6],
      R[3] - R[1],
  };
  if (theta < 1e-12) {
    rv[0] = 0.5 * vee[0];
    rv[1] = 0.5 * vee[1];
    rv[2] = 0.5 * vee[2];
    return;
  }
  const double sin_theta = std::sin(theta);
  if (std::fabs(sin_theta) < 1e-12) {
    rv[0] = 0.5 * vee[0];
    rv[1] = 0.5 * vee[1];
    rv[2] = 0.5 * vee[2];
    return;
  }
  const double scale = theta / (2.0 * sin_theta);
  rv[0] = scale * vee[0];
  rv[1] = scale * vee[1];
  rv[2] = scale * vee[2];
}

inline void retract_rotvec_right_host(const double base[3], const double delta[3], double out[3]) {
  double R_base[9], R_delta[9], R_out[9];
  rotvec_to_rotm_host(base, R_base);
  rotvec_to_rotm_host(delta, R_delta);
  mat3_mul(R_base, R_delta, R_out);
  rotm_to_rotvec_host(R_out, out);
}

inline void perturb_rotvec_right_axis_host(const double base[3], int axis, double eps, double out[3]) {
  double delta[3] = {0.0, 0.0, 0.0};
  if (axis >= 0 && axis < 3) delta[axis] = eps;
  retract_rotvec_right_host(base, delta, out);
}

inline void so3_left_jacobian_vec_host(const double w[3], const double v[3], double out[3]) {
  const double theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  double a = 0.5;
  double b = 1.0 / 6.0;
  if (theta2 >= 1e-12) {
    const double theta = std::sqrt(theta2);
    a = (1.0 - std::cos(theta)) / theta2;
    b = (theta - std::sin(theta)) / (theta2 * theta);
  }
  const double wxv[3] = {
      w[1] * v[2] - w[2] * v[1],
      w[2] * v[0] - w[0] * v[2],
      w[0] * v[1] - w[1] * v[0],
  };
  const double wxwxv[3] = {
      w[1] * wxv[2] - w[2] * wxv[1],
      w[2] * wxv[0] - w[0] * wxv[2],
      w[0] * wxv[1] - w[1] * wxv[0],
  };
  for (int i = 0; i < 3; i++) out[i] = v[i] + a * wxv[i] + b * wxwxv[i];
}

inline void so3_left_jacobian_inverse_vec_host(const double w[3], const double v[3], double out[3]) {
  const double theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  double a = 1.0 / 12.0;
  if (theta2 >= 1e-12) {
    const double theta = std::sqrt(theta2);
    const double sin_theta = std::sin(theta);
    const double cos_theta = std::cos(theta);
    if (std::fabs(sin_theta) >= 1e-12) {
      a = 1.0 / theta2 - (1.0 + cos_theta) / (2.0 * theta * sin_theta);
    }
  }
  const double wxv[3] = {
      w[1] * v[2] - w[2] * v[1],
      w[2] * v[0] - w[0] * v[2],
      w[0] * v[1] - w[1] * v[0],
  };
  const double wxwxv[3] = {
      w[1] * wxv[2] - w[2] * wxv[1],
      w[2] * wxv[0] - w[0] * wxv[2],
      w[0] * wxv[1] - w[1] * wxv[0],
  };
  for (int i = 0; i < 3; i++) out[i] = v[i] - 0.5 * wxv[i] + a * wxwxv[i];
}

inline void pose3_retract_right_host(
    const double base_p[3], const double base_att[3],
    const double delta_w[3], const double delta_v[3],
    double out_p[3], double out_att[3]) {
  double R_base[9], local_t[3], world_t[3];
  rotvec_to_rotm_host(base_att, R_base);
  so3_left_jacobian_vec_host(delta_w, delta_v, local_t);
  mat3_vec_mul(R_base, local_t, world_t);
  for (int i = 0; i < 3; i++) out_p[i] = base_p[i] + world_t[i];
  retract_rotvec_right_host(base_att, delta_w, out_att);
}

inline void perturb_pose_translation_axis_host(
    const double base_p[3], const double base_att[3], int axis, double eps, double out_p[3]) {
  double R_base[9];
  rotvec_to_rotm_host(base_att, R_base);
  for (int i = 0; i < 3; i++) out_p[i] = base_p[i];
  if (axis < 0 || axis >= 3) return;
  for (int r = 0; r < 3; r++) out_p[r] += R_base[r * 3 + axis] * eps;
}

inline bool pose3_between_log_residual_from_vectors(
    const double p0[3], const double att0[3],
    const double p1[3], const double att1[3],
    double res6[6]) {
  if (!finite3(p0) || !finite3(att0) || !finite3(p1) || !finite3(att1)) return false;
  double R0[9], R1[9], R0T[9], Rrel[9];
  rotvec_to_rotm_host(att0, R0);
  rotvec_to_rotm_host(att1, R1);
  mat3_transpose(R0, R0T);
  mat3_mul(R0T, R1, Rrel);
  double omega[3];
  rotm_to_rotvec_host(Rrel, omega);
  double dp_world[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
  double t_rel[3];
  mat3_vec_mul(R0T, dp_world, t_rel);
  double trans_log[3];
  so3_left_jacobian_inverse_vec_host(omega, t_rel, trans_log);
  for (int i = 0; i < 3; i++) res6[i] = omega[i];
  for (int i = 0; i < 3; i++) res6[3 + i] = trans_log[i];
  return finite3(res6) && finite3(res6 + 3);
}

inline void apply_vd_state_delta_host(
    int n_epoch, int ss, int pose_position_idx, int attitude_idx,
    const double* base, const double* delta, double scale, double* out) {
  const int n_state = n_epoch * ss;
  for (int i = 0; i < n_state; i++) {
    out[i] = base[i] + scale * delta[i];
  }
  if (attitude_idx < 0) return;
  for (int t = 0; t < n_epoch; t++) {
    const int epoch_off = ss * t;
    const int att_off = epoch_off + attitude_idx;
    const double base_att[3] = {base[att_off], base[att_off + 1], base[att_off + 2]};
    const double local_delta[3] = {
        scale * delta[att_off],
        scale * delta[att_off + 1],
        scale * delta[att_off + 2],
    };
    double retracted[3];
    if (pose_position_idx >= 0) {
      const int pos_off = epoch_off + pose_position_idx;
      const double base_p[3] = {base[pos_off], base[pos_off + 1], base[pos_off + 2]};
      const double local_translation_delta[3] = {
          scale * delta[pos_off],
          scale * delta[pos_off + 1],
          scale * delta[pos_off + 2],
      };
      double retracted_p[3];
      pose3_retract_right_host(base_p, base_att, local_delta, local_translation_delta, retracted_p, retracted);
      out[pos_off] = retracted_p[0];
      out[pos_off + 1] = retracted_p[1];
      out[pos_off + 2] = retracted_p[2];
    } else {
      retract_rotvec_right_host(base_att, local_delta, retracted);
    }
    out[att_off] = retracted[0];
    out[att_off + 1] = retracted[1];
    out[att_off + 2] = retracted[2];
  }
}

inline double component_weight(double base_weight, const double* weights, int t, int i) {
  if (weights == nullptr) return base_weight;
  const double w = weights[(size_t)t * 3 + i];
  return (std::isfinite(w) && w > 0.0) ? w : 0.0;
}

inline double imu_bias_jacobian_value(const double* jac, int t, int row, int col, double fallback_diag) {
  if (jac == nullptr) return row == col ? fallback_diag : 0.0;
  const double v = jac[(size_t)t * 9 + row * 3 + col];
  return std::isfinite(v) ? v : 0.0;
}

inline bool imu_bias_jacobian_interval_has_nonzero(const double* jac, int t) {
  if (jac == nullptr) return false;
  const size_t base = (size_t)t * 9;
  for (int k = 0; k < 9; k++) {
    const double v = jac[base + k];
    if (std::isfinite(v) && std::fabs(v) > 0.0) return true;
  }
  return false;
}

constexpr int kImuPvaMaxTerms = 24;

inline void corrected_imu_delta_angle(
    const double* da, const double* gyro_jac, int t, double fallback_diag,
    const double* gyro_bias_state, double corrected[3]) {
  corrected[0] = da[0];
  corrected[1] = da[1];
  corrected[2] = da[2];
  if (gyro_bias_state == nullptr) return;
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      corrected[row] -= imu_bias_jacobian_value(gyro_jac, t, row, col, fallback_diag) *
                        gyro_bias_state[col];
    }
  }
}

inline bool imu_rotation_residual_from_vectors(
    const double att0[3], const double att1[3], const double corrected_delta[3], double res[3]) {
  if (!finite3(att0) || !finite3(att1) || !finite3(corrected_delta)) return false;
  double R0[9], R1[9], R1T[9], Rd[9], tmp[9], Rerr[9];
  rotvec_to_rotm_host(att0, R0);
  rotvec_to_rotm_host(att1, R1);
  rotvec_to_rotm_host(corrected_delta, Rd);
  mat3_transpose(R1, R1T);
  mat3_mul(R1T, R0, tmp);
  mat3_mul(tmp, Rd, Rerr);
  rotm_to_rotvec_host(Rerr, res);
  return finite3(res);
}

inline bool imu_rotation_residual_for_state(
    int t, int ss, const double* imu_delta_angle, const double* imu_delta_angle_bias_gyro_jac,
    const double* state, const double* dt_arr, int attitude_idx, int gyro_bias_idx,
    bool imu_factor_use_next_bias, double res[3]) {
  if (!imu_delta_angle || attitude_idx < 0 || !dt_arr) return false;
  const double dt = dt_arr[t];
  if (!std::isfinite(dt) || dt <= 0.0) return false;
  const double* da = imu_delta_angle + (size_t)t * 3;
  if (!finite3(da)) return false;
  const int o0 = ss * t;
  const int o1 = ss * (t + 1);
  const int ob = imu_factor_use_next_bias ? o1 : o0;
  const double* bg = gyro_bias_idx >= 0 ? state + ob + gyro_bias_idx : nullptr;
  double corrected[3];
  corrected_imu_delta_angle(da, imu_delta_angle_bias_gyro_jac, t, dt, bg, corrected);
  return imu_rotation_residual_from_vectors(
      state + o0 + attitude_idx, state + o1 + attitude_idx, corrected, res);
}

inline void fill_imu_rotation_jacobian_terms(
    int t, int ss, const double* imu_delta_angle, const double* imu_delta_angle_bias_gyro_jac,
    const double* state, const double* dt_arr, int attitude_idx, int gyro_bias_idx,
    bool imu_factor_use_next_bias, int term_idx[kImuPvaMaxTerms], double J[3][kImuPvaMaxTerms]) {
  for (int k = 0; k < kImuPvaMaxTerms; k++) {
    term_idx[k] = -1;
    for (int r = 0; r < 3; r++) J[r][k] = 0.0;
  }
  if (!imu_delta_angle || attitude_idx < 0 || !dt_arr) return;
  const double dt = dt_arr[t];
  if (!std::isfinite(dt) || dt <= 0.0) return;
  const double* da = imu_delta_angle + (size_t)t * 3;
  if (!finite3(da)) return;

  const int o0 = ss * t;
  const int o1 = ss * (t + 1);
  const int ob = imu_factor_use_next_bias ? o1 : o0;
  double att0[3] = {
      state[o0 + attitude_idx],
      state[o0 + attitude_idx + 1],
      state[o0 + attitude_idx + 2],
  };
  double att1[3] = {
      state[o1 + attitude_idx],
      state[o1 + attitude_idx + 1],
      state[o1 + attitude_idx + 2],
  };
  double bg[3] = {0.0, 0.0, 0.0};
  const bool has_bg = gyro_bias_idx >= 0;
  if (has_bg) {
    bg[0] = state[ob + gyro_bias_idx];
    bg[1] = state[ob + gyro_bias_idx + 1];
    bg[2] = state[ob + gyro_bias_idx + 2];
  }
  if (!finite3(att0) || !finite3(att1) || (has_bg && !finite3(bg))) return;

  int n_terms = 0;
  for (int i = 0; i < 3; i++) term_idx[n_terms++] = o0 + attitude_idx + i;
  for (int i = 0; i < 3; i++) term_idx[n_terms++] = o1 + attitude_idx + i;
  if (has_bg) {
    for (int i = 0; i < 3; i++) term_idx[n_terms++] = ob + gyro_bias_idx + i;
  }

  constexpr double eps = 1e-6;
  for (int k = 0; k < n_terms; k++) {
    const int local_axis = k % 3;
    double plus_att0[3] = {att0[0], att0[1], att0[2]};
    double minus_att0[3] = {att0[0], att0[1], att0[2]};
    double plus_att1[3] = {att1[0], att1[1], att1[2]};
    double minus_att1[3] = {att1[0], att1[1], att1[2]};
    double plus_bg[3] = {bg[0], bg[1], bg[2]};
    double minus_bg[3] = {bg[0], bg[1], bg[2]};
    if (k < 3) {
      perturb_rotvec_right_axis_host(att0, local_axis, eps, plus_att0);
      perturb_rotvec_right_axis_host(att0, local_axis, -eps, minus_att0);
    } else if (k < 6) {
      perturb_rotvec_right_axis_host(att1, local_axis, eps, plus_att1);
      perturb_rotvec_right_axis_host(att1, local_axis, -eps, minus_att1);
    } else {
      plus_bg[local_axis] += eps;
      minus_bg[local_axis] -= eps;
    }
    double plus_delta[3], minus_delta[3];
    corrected_imu_delta_angle(da, imu_delta_angle_bias_gyro_jac, t, dt, has_bg ? plus_bg : nullptr, plus_delta);
    corrected_imu_delta_angle(da, imu_delta_angle_bias_gyro_jac, t, dt, has_bg ? minus_bg : nullptr, minus_delta);
    double r_plus[3], r_minus[3];
    if (!imu_rotation_residual_from_vectors(plus_att0, plus_att1, plus_delta, r_plus)) continue;
    if (!imu_rotation_residual_from_vectors(minus_att0, minus_att1, minus_delta, r_minus)) continue;
    for (int r = 0; r < 3; r++) {
      J[r][k] = (r_plus[r] - r_minus[r]) / (2.0 * eps);
    }
  }
}

inline bool stop_attitude_residual_from_vectors(
    const double att0[3], const double att1[3], double res[3]) {
  const double zero_delta[3] = {0.0, 0.0, 0.0};
  return imu_rotation_residual_from_vectors(att0, att1, zero_delta, res);
}

inline bool stop_attitude_residual_for_state(
    int t, int ss, const double* state, int attitude_idx, double res[3]) {
  if (attitude_idx < 0) return false;
  const int o0 = ss * t;
  const int o1 = ss * (t + 1);
  return stop_attitude_residual_from_vectors(
      state + o0 + attitude_idx, state + o1 + attitude_idx, res);
}

inline bool stop_pose3_between_residual_for_state(
    int t, int ss, const double* state, int pose_position_idx, int attitude_idx, double res6[6]) {
  if (pose_position_idx < 0 || attitude_idx < 0) return false;
  const int o0 = ss * t;
  const int o1 = ss * (t + 1);
  return pose3_between_log_residual_from_vectors(
      state + o0 + pose_position_idx, state + o0 + attitude_idx,
      state + o1 + pose_position_idx, state + o1 + attitude_idx,
      res6);
}

inline void fill_stop_attitude_jacobian_terms(
    int t, int ss, const double* state, int attitude_idx,
    int term_idx[6], double J[3][6]) {
  for (int k = 0; k < 6; k++) {
    term_idx[k] = -1;
    for (int r = 0; r < 3; r++) J[r][k] = 0.0;
  }
  if (attitude_idx < 0) return;
  const int o0 = ss * t;
  const int o1 = ss * (t + 1);
  double att0[3] = {
      state[o0 + attitude_idx],
      state[o0 + attitude_idx + 1],
      state[o0 + attitude_idx + 2],
  };
  double att1[3] = {
      state[o1 + attitude_idx],
      state[o1 + attitude_idx + 1],
      state[o1 + attitude_idx + 2],
  };
  if (!finite3(att0) || !finite3(att1)) return;

  for (int i = 0; i < 3; i++) term_idx[i] = o0 + attitude_idx + i;
  for (int i = 0; i < 3; i++) term_idx[3 + i] = o1 + attitude_idx + i;

  constexpr double eps = 1e-6;
  for (int k = 0; k < 6; k++) {
    const int local_axis = k % 3;
    double plus_att0[3] = {att0[0], att0[1], att0[2]};
    double minus_att0[3] = {att0[0], att0[1], att0[2]};
    double plus_att1[3] = {att1[0], att1[1], att1[2]};
    double minus_att1[3] = {att1[0], att1[1], att1[2]};
    if (k < 3) {
      perturb_rotvec_right_axis_host(att0, local_axis, eps, plus_att0);
      perturb_rotvec_right_axis_host(att0, local_axis, -eps, minus_att0);
    } else {
      perturb_rotvec_right_axis_host(att1, local_axis, eps, plus_att1);
      perturb_rotvec_right_axis_host(att1, local_axis, -eps, minus_att1);
    }
    double r_plus[3], r_minus[3];
    if (!stop_attitude_residual_from_vectors(plus_att0, plus_att1, r_plus)) continue;
    if (!stop_attitude_residual_from_vectors(minus_att0, minus_att1, r_minus)) continue;
    for (int r = 0; r < 3; r++) {
      J[r][k] = (r_plus[r] - r_minus[r]) / (2.0 * eps);
    }
  }
}

constexpr int kStopPose3MaxTerms = 12;

inline void fill_stop_pose3_between_jacobian_terms(
    int t, int ss, const double* state, int pose_position_idx, int attitude_idx,
    int term_idx[kStopPose3MaxTerms], double J[6][kStopPose3MaxTerms]) {
  for (int k = 0; k < kStopPose3MaxTerms; k++) {
    term_idx[k] = -1;
    for (int r = 0; r < 6; r++) J[r][k] = 0.0;
  }
  if (pose_position_idx < 0 || attitude_idx < 0) return;
  const int o0 = ss * t;
  const int o1 = ss * (t + 1);
  double p0[3] = {state[o0 + pose_position_idx], state[o0 + pose_position_idx + 1],
                  state[o0 + pose_position_idx + 2]};
  double att0[3] = {state[o0 + attitude_idx], state[o0 + attitude_idx + 1],
                    state[o0 + attitude_idx + 2]};
  double p1[3] = {state[o1 + pose_position_idx], state[o1 + pose_position_idx + 1],
                  state[o1 + pose_position_idx + 2]};
  double att1[3] = {state[o1 + attitude_idx], state[o1 + attitude_idx + 1],
                    state[o1 + attitude_idx + 2]};
  double base_res[6];
  if (!pose3_between_log_residual_from_vectors(p0, att0, p1, att1, base_res)) return;

  int n_terms = 0;
  for (int i = 0; i < 3; i++) term_idx[n_terms++] = o0 + pose_position_idx + i;
  for (int i = 0; i < 3; i++) term_idx[n_terms++] = o0 + attitude_idx + i;
  for (int i = 0; i < 3; i++) term_idx[n_terms++] = o1 + pose_position_idx + i;
  for (int i = 0; i < 3; i++) term_idx[n_terms++] = o1 + attitude_idx + i;

  constexpr double eps = 1e-6;
  const double zero[3] = {0.0, 0.0, 0.0};
  for (int k = 0; k < n_terms; k++) {
    const int category = k / 3;
    const int axis = k % 3;
    double plus_p0[3] = {p0[0], p0[1], p0[2]};
    double minus_p0[3] = {p0[0], p0[1], p0[2]};
    double plus_att0[3] = {att0[0], att0[1], att0[2]};
    double minus_att0[3] = {att0[0], att0[1], att0[2]};
    double plus_att1[3] = {att1[0], att1[1], att1[2]};
    double minus_att1[3] = {att1[0], att1[1], att1[2]};
    double plus_p1[3] = {p1[0], p1[1], p1[2]};
    double minus_p1[3] = {p1[0], p1[1], p1[2]};
    double delta_plus[3] = {0.0, 0.0, 0.0};
    double delta_minus[3] = {0.0, 0.0, 0.0};
    delta_plus[axis] = eps;
    delta_minus[axis] = -eps;
    if (category == 0) {
      pose3_retract_right_host(p0, att0, zero, delta_plus, plus_p0, plus_att0);
      pose3_retract_right_host(p0, att0, zero, delta_minus, minus_p0, minus_att0);
    } else if (category == 1) {
      pose3_retract_right_host(p0, att0, delta_plus, zero, plus_p0, plus_att0);
      pose3_retract_right_host(p0, att0, delta_minus, zero, minus_p0, minus_att0);
    } else if (category == 2) {
      pose3_retract_right_host(p1, att1, zero, delta_plus, plus_p1, plus_att1);
      pose3_retract_right_host(p1, att1, zero, delta_minus, minus_p1, minus_att1);
    } else {
      pose3_retract_right_host(p1, att1, delta_plus, zero, plus_p1, plus_att1);
      pose3_retract_right_host(p1, att1, delta_minus, zero, minus_p1, minus_att1);
    }
    double r_plus[6], r_minus[6];
    if (!pose3_between_log_residual_from_vectors(plus_p0, plus_att0, plus_p1, plus_att1, r_plus)) continue;
    if (!pose3_between_log_residual_from_vectors(minus_p0, minus_att0, minus_p1, minus_att1, r_minus)) continue;
    for (int r = 0; r < 6; r++) {
      J[r][k] = (r_plus[r] - r_minus[r]) / (2.0 * eps);
    }
  }
}

void add_stop_attitude_factor_host(
    int n_epoch, int ss, int n_state, double w_stop_attitude,
    double stop_position_huber_k,
    const std::uint8_t* stop_mask, const double* state, int attitude_idx,
    double* H, double* g) {
  if (w_stop_attitude <= 0.0 || !stop_mask || attitude_idx < 0) return;

  for (int t = 0; t < n_epoch - 1; t++) {
    if (stop_mask[t] == 0 || stop_mask[t + 1] == 0) continue;
    double res[3];
    if (!stop_attitude_residual_for_state(t, ss, state, attitude_idx, res)) continue;
    int term_idx[6];
    double J[3][6];
    fill_stop_attitude_jacobian_terms(t, ss, state, attitude_idx, term_idx, J);
    for (int i = 0; i < 3; i++) {
      const double w_eff = huber_effective_weight(w_stop_attitude, res[i], stop_position_huber_k);
      add_linear_residual_host(6, term_idx, J[i], w_eff, res[i], n_state, H, g);
    }
  }
}

double stop_attitude_cost_host(
    int n_epoch, int ss, double w_stop_attitude,
    double stop_position_huber_k,
    const std::uint8_t* stop_mask, const double* state, int attitude_idx) {
  if (w_stop_attitude <= 0.0 || !stop_mask || attitude_idx < 0) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch - 1; t++) {
    if (stop_mask[t] == 0 || stop_mask[t + 1] == 0) continue;
    double res[3];
    if (!stop_attitude_residual_for_state(t, ss, state, attitude_idx, res)) continue;
    for (int i = 0; i < 3; i++) {
      e += huber_loss(w_stop_attitude, res[i], stop_position_huber_k);
    }
  }
  return e;
}

void add_stop_pose_factor_host(
    int n_epoch, int ss, int n_state, double w_stop_position, double w_stop_attitude,
    double stop_position_huber_k, const std::uint8_t* stop_mask, const double* state,
    int pose_position_idx, int attitude_idx, double* H, double* g) {
  if ((w_stop_position <= 0.0 && w_stop_attitude <= 0.0) || !stop_mask) return;
  const int pos_idx = pose_position_idx >= 0 ? pose_position_idx : 0;
  if (pose_position_idx >= 0 && attitude_idx >= 0) {
    for (int t = 0; t < n_epoch - 1; t++) {
      if (stop_mask[t] == 0 || stop_mask[t + 1] == 0) continue;
      double res6[6];
      if (!stop_pose3_between_residual_for_state(t, ss, state, pose_position_idx, attitude_idx, res6)) continue;
      double z2 = 0.0;
      if (w_stop_attitude > 0.0) {
        for (int i = 0; i < 3; i++) z2 += w_stop_attitude * res6[i] * res6[i];
      }
      if (w_stop_position > 0.0) {
        for (int i = 0; i < 3; i++) z2 += w_stop_position * res6[3 + i] * res6[3 + i];
      }
      const double scale =
          huber_weight_scale_from_whitened_norm(std::sqrt(z2), stop_position_huber_k);
      int term_idx[kStopPose3MaxTerms];
      double J[6][kStopPose3MaxTerms];
      fill_stop_pose3_between_jacobian_terms(t, ss, state, pose_position_idx, attitude_idx, term_idx, J);
      if (w_stop_attitude > 0.0) {
        const double w_eff = w_stop_attitude * scale;
        for (int i = 0; i < 3; i++) {
          add_linear_residual_host(kStopPose3MaxTerms, term_idx, J[i], w_eff, res6[i], n_state, H, g);
        }
      }
      if (w_stop_position > 0.0) {
        const double w_eff = w_stop_position * scale;
        for (int i = 0; i < 3; i++) {
          add_linear_residual_host(kStopPose3MaxTerms, term_idx, J[3 + i], w_eff, res6[3 + i], n_state, H, g);
        }
      }
    }
    return;
  }

  for (int t = 0; t < n_epoch - 1; t++) {
    if (stop_mask[t] == 0 || stop_mask[t + 1] == 0) continue;
    const int o0 = ss * t;
    const int o1 = ss * (t + 1);
    double pos_res[3] = {0.0, 0.0, 0.0};
    double att_res[3] = {0.0, 0.0, 0.0};
    bool has_att = false;
    double z2 = 0.0;
    if (w_stop_position > 0.0) {
      for (int i = 0; i < 3; i++) {
        pos_res[i] = state[o0 + pos_idx + i] - state[o1 + pos_idx + i];
        z2 += w_stop_position * pos_res[i] * pos_res[i];
      }
    }
    if (w_stop_attitude > 0.0 && attitude_idx >= 0 &&
        stop_attitude_residual_for_state(t, ss, state, attitude_idx, att_res)) {
      has_att = true;
      for (int i = 0; i < 3; i++) {
        z2 += w_stop_attitude * att_res[i] * att_res[i];
      }
    }
    const double scale =
        huber_weight_scale_from_whitened_norm(std::sqrt(z2), stop_position_huber_k);
    if (w_stop_position > 0.0) {
      const double w_eff = w_stop_position * scale;
      for (int i = 0; i < 3; i++) {
        const double res = pos_res[i];
        const int p0 = o0 + pos_idx + i;
        const int p1 = o1 + pos_idx + i;
        g[p0] += -w_eff * res;
        g[p1] += w_eff * res;
        H[(size_t)p0 * n_state + p0] += w_eff;
        H[(size_t)p1 * n_state + p1] += w_eff;
        H[(size_t)p0 * n_state + p1] += -w_eff;
        H[(size_t)p1 * n_state + p0] += -w_eff;
      }
    }
    if (has_att) {
      int term_idx[6];
      double J[3][6];
      fill_stop_attitude_jacobian_terms(t, ss, state, attitude_idx, term_idx, J);
      const double w_eff = w_stop_attitude * scale;
      for (int i = 0; i < 3; i++) {
        add_linear_residual_host(6, term_idx, J[i], w_eff, att_res[i], n_state, H, g);
      }
    }
  }
}

double stop_pose_cost_host(
    int n_epoch, int ss, double w_stop_position, double w_stop_attitude,
    double stop_position_huber_k, const std::uint8_t* stop_mask, const double* state,
    int pose_position_idx, int attitude_idx) {
  if ((w_stop_position <= 0.0 && w_stop_attitude <= 0.0) || !stop_mask) return 0.0;
  const int pos_idx = pose_position_idx >= 0 ? pose_position_idx : 0;
  double e = 0.0;
  if (pose_position_idx >= 0 && attitude_idx >= 0) {
    for (int t = 0; t < n_epoch - 1; t++) {
      if (stop_mask[t] == 0 || stop_mask[t + 1] == 0) continue;
      double res6[6];
      if (!stop_pose3_between_residual_for_state(t, ss, state, pose_position_idx, attitude_idx, res6)) continue;
      double z2 = 0.0;
      if (w_stop_attitude > 0.0) {
        for (int i = 0; i < 3; i++) z2 += w_stop_attitude * res6[i] * res6[i];
      }
      if (w_stop_position > 0.0) {
        for (int i = 0; i < 3; i++) z2 += w_stop_position * res6[3 + i] * res6[3 + i];
      }
      e += huber_loss_from_whitened_norm(std::sqrt(z2), stop_position_huber_k);
    }
    return e;
  }
  for (int t = 0; t < n_epoch - 1; t++) {
    if (stop_mask[t] == 0 || stop_mask[t + 1] == 0) continue;
    const int o0 = ss * t;
    const int o1 = ss * (t + 1);
    double z2 = 0.0;
    if (w_stop_position > 0.0) {
      for (int i = 0; i < 3; i++) {
        const double res = state[o0 + pos_idx + i] - state[o1 + pos_idx + i];
        z2 += w_stop_position * res * res;
      }
    }
    if (w_stop_attitude > 0.0 && attitude_idx >= 0) {
      double res[3];
      if (stop_attitude_residual_for_state(t, ss, state, attitude_idx, res)) {
        for (int i = 0; i < 3; i++) {
          z2 += w_stop_attitude * res[i] * res[i];
        }
      }
    }
    e += huber_loss_from_whitened_norm(std::sqrt(z2), stop_position_huber_k);
  }
  return e;
}

void add_pose_point_factor_host(
    int n_epoch, int ss, int n_state, int pose_position_idx,
    int attitude_idx, const double* state, double* H, double* g) {
  if (pose_position_idx < 0) return;
  const double w = kPosePointConstrainedWeight;
  for (int t = 0; t < n_epoch; t++) {
    const int o = ss * t;
    if (attitude_idx < 0 || !finite3(state + o + attitude_idx)) continue;
    double R[9];
    rotvec_to_rotm_host(state + o + attitude_idx, R);
    int idx[6];
    for (int i = 0; i < 3; i++) idx[i] = o + pose_position_idx + i;
    for (int i = 0; i < 3; i++) idx[3 + i] = o + i;
    for (int row = 0; row < 3; row++) {
      double jac[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      for (int col = 0; col < 3; col++) jac[col] = R[row * 3 + col];
      jac[3 + row] = -1.0;
      const double res = state[o + pose_position_idx + row] - state[o + row];
      add_linear_residual_host(6, idx, jac, w, res, n_state, H, g);
    }
  }
}

double pose_point_cost_host(int n_epoch, int ss, int pose_position_idx, const double* state) {
  if (pose_position_idx < 0) return 0.0;
  const double w = kPosePointConstrainedWeight;
  double e = 0.0;
  for (int t = 0; t < n_epoch; t++) {
    const int o = ss * t;
    for (int i = 0; i < 3; i++) {
      const double res = state[o + pose_position_idx + i] - state[o + i];
      e += 0.5 * w * res * res;
    }
  }
  return e;
}

inline bool corrected_imu_delta_from_vectors(
    const double* delta, const double* accel_jac, double accel_fallback_diag,
    const double accel_bias[3], bool has_accel_bias,
    const double* gyro_jac, double gyro_fallback_diag,
    const double gyro_bias[3], bool has_gyro_bias, bool use_direct_gyro_bias,
    int t, double corrected[3]);

inline bool rotated_imu_delta_from_vectors(
    const double* delta, const double* accel_jac, double accel_fallback_diag,
    const double accel_bias[3], bool has_accel_bias,
    const double* gyro_jac, double gyro_fallback_diag,
    const double gyro_bias[3], bool has_gyro_bias, bool use_direct_gyro_bias,
    const double* angle_gyro_jac, double angle_fallback_diag,
    const double attitude[3], bool use_attitude, int t, double out[3]) {
  double corrected[3];
  if (!corrected_imu_delta_from_vectors(delta, accel_jac, accel_fallback_diag,
                                        accel_bias, has_accel_bias,
                                        gyro_jac, gyro_fallback_diag,
                                        gyro_bias, has_gyro_bias, use_direct_gyro_bias,
                                        t, corrected)) {
    return false;
  }
  if (use_attitude && !finite3(attitude)) return false;

  if (!use_attitude) {
    out[0] = corrected[0];
    out[1] = corrected[1];
    out[2] = corrected[2];
    return true;
  }

  double effective_att[3] = {attitude[0], attitude[1], attitude[2]};
  if (has_gyro_bias && !use_direct_gyro_bias) {
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        effective_att[row] -= imu_bias_jacobian_value(
                                  angle_gyro_jac, t, row, col, angle_fallback_diag) *
                              gyro_bias[col];
      }
    }
  }
  if (!finite3(effective_att)) return false;

  double R[9];
  rotvec_to_rotm_host(effective_att, R);
  mat3_vec_mul(R, corrected, out);
  return finite3(out);
}

inline bool corrected_imu_delta_from_vectors(
    const double* delta, const double* accel_jac, double accel_fallback_diag,
    const double accel_bias[3], bool has_accel_bias,
    const double* gyro_jac, double gyro_fallback_diag,
    const double gyro_bias[3], bool has_gyro_bias, bool use_direct_gyro_bias,
    int t, double corrected[3]) {
  if (!finite3(delta)) return false;
  if (has_accel_bias && !finite3(accel_bias)) return false;
  if (has_gyro_bias && !finite3(gyro_bias)) return false;

  corrected[0] = delta[0];
  corrected[1] = delta[1];
  corrected[2] = delta[2];
  if (has_accel_bias) {
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        corrected[row] -= imu_bias_jacobian_value(accel_jac, t, row, col, accel_fallback_diag) *
                          accel_bias[col];
      }
    }
  }
  if (has_gyro_bias && use_direct_gyro_bias) {
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        corrected[row] -= imu_bias_jacobian_value(gyro_jac, t, row, col, gyro_fallback_diag) *
                          gyro_bias[col];
      }
    }
  }
  if (!finite3(corrected)) return false;
  return true;
}

inline bool rotated_imu_delta_for_state(
    int t, int ss, const double* delta_arr,
    const double* accel_jac, double accel_fallback_diag,
    const double* gyro_jac, double gyro_fallback_diag,
    const double* angle_gyro_jac, double angle_fallback_diag,
    const double* state, int attitude_idx, int accel_bias_idx, int gyro_bias_idx,
    bool imu_factor_use_next_bias, bool use_direct_gyro_bias, double out[3]) {
  if (!delta_arr) return false;
  const int o0 = ss * t;
  const int o1 = ss * (t + 1);
  const int ob = imu_factor_use_next_bias ? o1 : o0;
  const bool use_att = attitude_idx >= 0;
  const bool has_accel_bias = accel_bias_idx >= 0;
  const bool has_gyro_bias = gyro_bias_idx >= 0 && (use_direct_gyro_bias || use_att);
  const double* delta = delta_arr + (size_t)t * 3;
  const double zero[3] = {0.0, 0.0, 0.0};
  const double* attitude = use_att ? state + o0 + attitude_idx : zero;
  const double* accel_bias = has_accel_bias ? state + ob + accel_bias_idx : zero;
  const double* gyro_bias = has_gyro_bias ? state + ob + gyro_bias_idx : zero;
  return rotated_imu_delta_from_vectors(
      delta, accel_jac, accel_fallback_diag, accel_bias, has_accel_bias,
      gyro_jac, gyro_fallback_diag, gyro_bias, has_gyro_bias, use_direct_gyro_bias,
      angle_gyro_jac, angle_fallback_diag, attitude, use_att, t, out);
}

inline void fill_rotated_imu_delta_jacobian_terms(
    int t, int ss, const double* delta_arr,
    const double* accel_jac, double accel_fallback_diag,
    const double* gyro_jac, double gyro_fallback_diag,
    const double* angle_gyro_jac, double angle_fallback_diag,
    const double* state, int attitude_idx, int accel_bias_idx, int gyro_bias_idx,
    bool imu_factor_use_next_bias, bool use_direct_gyro_bias,
    int term_idx[kImuPvaMaxTerms], double J[3][kImuPvaMaxTerms]) {
  for (int k = 0; k < kImuPvaMaxTerms; k++) {
    term_idx[k] = -1;
    for (int r = 0; r < 3; r++) J[r][k] = 0.0;
  }
  if (!delta_arr) return;
  const int o0 = ss * t;
  const int o1 = ss * (t + 1);
  const int ob = imu_factor_use_next_bias ? o1 : o0;
  const bool use_att = attitude_idx >= 0;
  const bool has_accel_bias = accel_bias_idx >= 0;
  const bool has_gyro_bias = gyro_bias_idx >= 0 && (use_direct_gyro_bias || use_att);
  const double* delta = delta_arr + (size_t)t * 3;

  double att[3] = {0.0, 0.0, 0.0};
  double ba[3] = {0.0, 0.0, 0.0};
  double bg[3] = {0.0, 0.0, 0.0};
  if (use_att) {
    att[0] = state[o0 + attitude_idx];
    att[1] = state[o0 + attitude_idx + 1];
    att[2] = state[o0 + attitude_idx + 2];
  }
  if (has_accel_bias) {
    ba[0] = state[ob + accel_bias_idx];
    ba[1] = state[ob + accel_bias_idx + 1];
    ba[2] = state[ob + accel_bias_idx + 2];
  }
  if (has_gyro_bias) {
    bg[0] = state[ob + gyro_bias_idx];
    bg[1] = state[ob + gyro_bias_idx + 1];
    bg[2] = state[ob + gyro_bias_idx + 2];
  }
  double base_out[3];
  if (!rotated_imu_delta_from_vectors(
          delta, accel_jac, accel_fallback_diag, ba, has_accel_bias,
          gyro_jac, gyro_fallback_diag, bg, has_gyro_bias, use_direct_gyro_bias,
          angle_gyro_jac, angle_fallback_diag, att, use_att, t, base_out)) {
    return;
  }

  int n_terms = 0;
  if (use_att) {
    for (int i = 0; i < 3; i++) term_idx[n_terms++] = o0 + attitude_idx + i;
  }
  if (has_accel_bias) {
    for (int i = 0; i < 3; i++) term_idx[n_terms++] = ob + accel_bias_idx + i;
  }
  if (has_gyro_bias) {
    for (int i = 0; i < 3; i++) term_idx[n_terms++] = ob + gyro_bias_idx + i;
  }

  constexpr double eps = 1e-6;
  for (int k = 0; k < n_terms; k++) {
    int category = 0;
    int category_start = 0;
    if (use_att && k < 3) {
      category = 0;
      category_start = 0;
    } else if (has_accel_bias && k < (use_att ? 6 : 3)) {
      category = 1;
      category_start = use_att ? 3 : 0;
    } else {
      category = 2;
      category_start = (use_att ? 3 : 0) + (has_accel_bias ? 3 : 0);
    }
    const int axis = k - category_start;
    double plus_att[3] = {att[0], att[1], att[2]};
    double minus_att[3] = {att[0], att[1], att[2]};
    double plus_ba[3] = {ba[0], ba[1], ba[2]};
    double minus_ba[3] = {ba[0], ba[1], ba[2]};
    double plus_bg[3] = {bg[0], bg[1], bg[2]};
    double minus_bg[3] = {bg[0], bg[1], bg[2]};
    if (category == 0) {
      perturb_rotvec_right_axis_host(att, axis, eps, plus_att);
      perturb_rotvec_right_axis_host(att, axis, -eps, minus_att);
    } else if (category == 1) {
      plus_ba[axis] += eps;
      minus_ba[axis] -= eps;
    } else {
      plus_bg[axis] += eps;
      minus_bg[axis] -= eps;
    }
    double r_plus[3], r_minus[3];
    if (!rotated_imu_delta_from_vectors(
            delta, accel_jac, accel_fallback_diag, plus_ba, has_accel_bias,
            gyro_jac, gyro_fallback_diag, plus_bg, has_gyro_bias, use_direct_gyro_bias,
            angle_gyro_jac, angle_fallback_diag, plus_att, use_att, t, r_plus)) {
      continue;
    }
    if (!rotated_imu_delta_from_vectors(
            delta, accel_jac, accel_fallback_diag, minus_ba, has_accel_bias,
            gyro_jac, gyro_fallback_diag, minus_bg, has_gyro_bias, use_direct_gyro_bias,
            angle_gyro_jac, angle_fallback_diag, minus_att, use_att, t, r_minus)) {
      continue;
    }
    for (int r = 0; r < 3; r++) {
      J[r][k] = (r_plus[r] - r_minus[r]) / (2.0 * eps);
    }
  }
}

inline bool imu_body_delta_residual_from_vectors(
    const double* delta, const double* accel_jac, double accel_fallback_diag,
    const double* gyro_jac, double gyro_fallback_diag,
    const double* delta_angle, const double* angle_gyro_jac, double angle_fallback_diag,
    const double accel_bias[3], bool has_accel_bias,
    const double gyro_bias[3], bool has_delta_gyro_bias, bool has_angle_gyro_bias,
    const double p0[3], const double v0[3], const double att0[3],
    const double p1[3], const double v1[3], const double att1[3], const double gravity[3],
    double dt, bool is_position, int t, double res[3]) {
  (void)delta_angle;
  (void)angle_gyro_jac;
  (void)angle_fallback_diag;
  (void)has_angle_gyro_bias;
  if (!std::isfinite(dt) || dt <= 0.0) return false;
  if (!finite3(att0) || !finite3(att1) || !finite3(v0) || !finite3(gravity)) return false;
  if (is_position && (!finite3(p0) || !finite3(p1))) return false;
  if (!is_position && !finite3(v1)) return false;

  double corrected[3];
  if (!corrected_imu_delta_from_vectors(delta, accel_jac, accel_fallback_diag,
                                        accel_bias, has_accel_bias,
                                        gyro_jac, gyro_fallback_diag,
                                        gyro_bias, has_delta_gyro_bias, true,
                                        t, corrected)) {
    return false;
  }

  double R0[9], R1[9], R1T[9], predicted_body_delta[3], predicted_minus_actual[3];
  rotvec_to_rotm_host(att0, R0);
  rotvec_to_rotm_host(att1, R1);
  mat3_transpose(R1, R1T);
  mat3_vec_mul(R0, corrected, predicted_body_delta);

  if (is_position) {
    for (int i = 0; i < 3; i++) {
      const double predicted = p0[i] + v0[i] * dt + 0.5 * gravity[i] * dt * dt + predicted_body_delta[i];
      predicted_minus_actual[i] = predicted - p1[i];
    }
  } else {
    for (int i = 0; i < 3; i++) {
      const double predicted = v0[i] + gravity[i] * dt + predicted_body_delta[i];
      predicted_minus_actual[i] = predicted - v1[i];
    }
  }

  mat3_vec_mul(R1T, predicted_minus_actual, res);
  return finite3(res);
}

inline bool imu_body_delta_residual_for_state(
    int t, int ss, const double* delta_arr, const double* imu_gravity,
    const double* accel_jac, double accel_fallback_diag,
    const double* gyro_jac, double gyro_fallback_diag,
    const double* delta_angle_arr, const double* angle_gyro_jac, double angle_fallback_diag,
    const double* state, const double* dt_arr, int pose_position_idx, int attitude_idx,
    int accel_bias_idx, int gyro_bias_idx, bool imu_factor_use_next_bias,
    bool is_position, double res[3]) {
  if (!delta_arr || !imu_gravity || attitude_idx < 0 || !dt_arr) return false;
  const double dt = dt_arr[t];
  if (!std::isfinite(dt) || dt <= 0.0) return false;
  const int o0 = ss * t;
  const int o1 = ss * (t + 1);
  const int pos_idx = pose_position_idx >= 0 ? pose_position_idx : 0;
  const int ob = imu_factor_use_next_bias ? o1 : o0;
  const bool has_accel_bias = accel_bias_idx >= 0;
  const bool has_delta_gyro_bias =
      gyro_bias_idx >= 0 && imu_bias_jacobian_interval_has_nonzero(gyro_jac, t);
  const bool needs_gyro_bias = has_delta_gyro_bias;

  const double zero[3] = {0.0, 0.0, 0.0};
  const double* accel_bias = has_accel_bias ? state + ob + accel_bias_idx : zero;
  const double* gyro_bias = needs_gyro_bias ? state + ob + gyro_bias_idx : zero;
  const double* delta = delta_arr + (size_t)t * 3;
  const double* delta_angle = delta_angle_arr ? delta_angle_arr + (size_t)t * 3 : nullptr;
  const double* gravity = imu_gravity + (size_t)t * 3;
  return imu_body_delta_residual_from_vectors(
      delta, accel_jac, accel_fallback_diag, gyro_jac, gyro_fallback_diag,
      delta_angle, angle_gyro_jac, angle_fallback_diag,
      accel_bias, has_accel_bias, gyro_bias, has_delta_gyro_bias, false,
      state + o0 + pos_idx, state + o0 + 3, state + o0 + attitude_idx,
      state + o1 + pos_idx, state + o1 + 3, state + o1 + attitude_idx,
      gravity, dt, is_position, t, res);
}

inline void fill_body_imu_delta_jacobian_terms(
    int t, int ss, const double* delta_arr, const double* imu_gravity,
    const double* accel_jac, double accel_fallback_diag,
    const double* gyro_jac, double gyro_fallback_diag,
    const double* delta_angle_arr, const double* angle_gyro_jac, double angle_fallback_diag,
    const double* state, const double* dt_arr, int pose_position_idx, int attitude_idx,
    int accel_bias_idx, int gyro_bias_idx, bool imu_factor_use_next_bias,
    bool is_position, int term_idx[kImuPvaMaxTerms], double J[3][kImuPvaMaxTerms]) {
  for (int k = 0; k < kImuPvaMaxTerms; k++) {
    term_idx[k] = -1;
    for (int r = 0; r < 3; r++) J[r][k] = 0.0;
  }
  if (!delta_arr || !imu_gravity || attitude_idx < 0 || !dt_arr) return;
  const double dt = dt_arr[t];
  if (!std::isfinite(dt) || dt <= 0.0) return;

  const int o0 = ss * t;
  const int o1 = ss * (t + 1);
  const int pos_idx = pose_position_idx >= 0 ? pose_position_idx : 0;
  const int ob = imu_factor_use_next_bias ? o1 : o0;
  const bool has_accel_bias = accel_bias_idx >= 0;
  const bool has_delta_gyro_bias =
      gyro_bias_idx >= 0 && imu_bias_jacobian_interval_has_nonzero(gyro_jac, t);
  const bool needs_gyro_bias = has_delta_gyro_bias;

  double p0[3] = {state[o0 + pos_idx], state[o0 + pos_idx + 1], state[o0 + pos_idx + 2]};
  double v0[3] = {state[o0 + 3], state[o0 + 4], state[o0 + 5]};
  double att0[3] = {
      state[o0 + attitude_idx],
      state[o0 + attitude_idx + 1],
      state[o0 + attitude_idx + 2],
  };
  double att1[3] = {
      state[o1 + attitude_idx],
      state[o1 + attitude_idx + 1],
      state[o1 + attitude_idx + 2],
  };
  double p1[3] = {state[o1 + pos_idx], state[o1 + pos_idx + 1], state[o1 + pos_idx + 2]};
  double v1[3] = {state[o1 + 3], state[o1 + 4], state[o1 + 5]};
  const bool use_pose3_translation = pose_position_idx >= 0;
  if (use_pose3_translation && (!finite3(att0) || !finite3(att1))) return;
  double ba[3] = {0.0, 0.0, 0.0};
  double bg[3] = {0.0, 0.0, 0.0};
  if (has_accel_bias) {
    ba[0] = state[ob + accel_bias_idx];
    ba[1] = state[ob + accel_bias_idx + 1];
    ba[2] = state[ob + accel_bias_idx + 2];
  }
  if (needs_gyro_bias) {
    bg[0] = state[ob + gyro_bias_idx];
    bg[1] = state[ob + gyro_bias_idx + 1];
    bg[2] = state[ob + gyro_bias_idx + 2];
  }

  const double* delta = delta_arr + (size_t)t * 3;
  const double* delta_angle = delta_angle_arr ? delta_angle_arr + (size_t)t * 3 : nullptr;
  const double* gravity = imu_gravity + (size_t)t * 3;
  double base_res[3];
  if (!imu_body_delta_residual_from_vectors(
          delta, accel_jac, accel_fallback_diag, gyro_jac, gyro_fallback_diag,
          delta_angle, angle_gyro_jac, angle_fallback_diag,
          ba, has_accel_bias, bg, has_delta_gyro_bias, false, p0, v0, att0, p1, v1, att1, gravity,
          dt, is_position, t, base_res)) {
    return;
  }

  constexpr int kTermP0 = 0;
  constexpr int kTermV0 = 1;
  constexpr int kTermAtt0 = 2;
  constexpr int kTermAccelBias = 3;
  constexpr int kTermGyroBias = 4;
  constexpr int kTermP1 = 5;
  constexpr int kTermV1 = 6;
  constexpr int kTermAtt1 = 7;
  int term_category[kImuPvaMaxTerms];
  int term_axis[kImuPvaMaxTerms];
  int n_terms = 0;
  if (is_position) {
    for (int i = 0; i < 3 && n_terms < kImuPvaMaxTerms; i++) {
      term_idx[n_terms] = o0 + pos_idx + i;
      term_category[n_terms] = kTermP0;
      term_axis[n_terms++] = i;
    }
  }
  for (int i = 0; i < 3 && n_terms < kImuPvaMaxTerms; i++) {
    term_idx[n_terms] = o0 + 3 + i;
    term_category[n_terms] = kTermV0;
    term_axis[n_terms++] = i;
  }
  for (int i = 0; i < 3 && n_terms < kImuPvaMaxTerms; i++) {
    term_idx[n_terms] = o0 + attitude_idx + i;
    term_category[n_terms] = kTermAtt0;
    term_axis[n_terms++] = i;
  }
  if (has_accel_bias) {
    for (int i = 0; i < 3 && n_terms < kImuPvaMaxTerms; i++) {
      term_idx[n_terms] = ob + accel_bias_idx + i;
      term_category[n_terms] = kTermAccelBias;
      term_axis[n_terms++] = i;
    }
  }
  if (needs_gyro_bias) {
    for (int i = 0; i < 3 && n_terms < kImuPvaMaxTerms; i++) {
      term_idx[n_terms] = ob + gyro_bias_idx + i;
      term_category[n_terms] = kTermGyroBias;
      term_axis[n_terms++] = i;
    }
  }
  if (is_position) {
    for (int i = 0; i < 3 && n_terms < kImuPvaMaxTerms; i++) {
      term_idx[n_terms] = o1 + pos_idx + i;
      term_category[n_terms] = kTermP1;
      term_axis[n_terms++] = i;
    }
  } else {
    for (int i = 0; i < 3 && n_terms < kImuPvaMaxTerms; i++) {
      term_idx[n_terms] = o1 + 3 + i;
      term_category[n_terms] = kTermV1;
      term_axis[n_terms++] = i;
    }
  }
  for (int i = 0; i < 3 && n_terms < kImuPvaMaxTerms; i++) {
    term_idx[n_terms] = o1 + attitude_idx + i;
    term_category[n_terms] = kTermAtt1;
    term_axis[n_terms++] = i;
  }

  constexpr double eps = 1e-6;
  for (int k = 0; k < n_terms; k++) {
    double plus_p0[3] = {p0[0], p0[1], p0[2]};
    double minus_p0[3] = {p0[0], p0[1], p0[2]};
    double plus_v0[3] = {v0[0], v0[1], v0[2]};
    double minus_v0[3] = {v0[0], v0[1], v0[2]};
    double plus_att0[3] = {att0[0], att0[1], att0[2]};
    double minus_att0[3] = {att0[0], att0[1], att0[2]};
    double plus_att1[3] = {att1[0], att1[1], att1[2]};
    double minus_att1[3] = {att1[0], att1[1], att1[2]};
    double plus_p1[3] = {p1[0], p1[1], p1[2]};
    double minus_p1[3] = {p1[0], p1[1], p1[2]};
    double plus_v1[3] = {v1[0], v1[1], v1[2]};
    double minus_v1[3] = {v1[0], v1[1], v1[2]};
    double plus_ba[3] = {ba[0], ba[1], ba[2]};
    double minus_ba[3] = {ba[0], ba[1], ba[2]};
    double plus_bg[3] = {bg[0], bg[1], bg[2]};
    double minus_bg[3] = {bg[0], bg[1], bg[2]};
    const int axis = term_axis[k];
    switch (term_category[k]) {
      case kTermP0:
        if (use_pose3_translation) {
          perturb_pose_translation_axis_host(p0, att0, axis, eps, plus_p0);
          perturb_pose_translation_axis_host(p0, att0, axis, -eps, minus_p0);
        } else {
          plus_p0[axis] += eps;
          minus_p0[axis] -= eps;
        }
        break;
      case kTermV0:
        plus_v0[axis] += eps;
        minus_v0[axis] -= eps;
        break;
      case kTermAtt0:
        perturb_rotvec_right_axis_host(att0, axis, eps, plus_att0);
        perturb_rotvec_right_axis_host(att0, axis, -eps, minus_att0);
        break;
      case kTermAccelBias:
        plus_ba[axis] += eps;
        minus_ba[axis] -= eps;
        break;
      case kTermGyroBias:
        plus_bg[axis] += eps;
        minus_bg[axis] -= eps;
        break;
      case kTermP1:
        if (use_pose3_translation) {
          perturb_pose_translation_axis_host(p1, att1, axis, eps, plus_p1);
          perturb_pose_translation_axis_host(p1, att1, axis, -eps, minus_p1);
        } else {
          plus_p1[axis] += eps;
          minus_p1[axis] -= eps;
        }
        break;
      case kTermV1:
        plus_v1[axis] += eps;
        minus_v1[axis] -= eps;
        break;
      case kTermAtt1:
        perturb_rotvec_right_axis_host(att1, axis, eps, plus_att1);
        perturb_rotvec_right_axis_host(att1, axis, -eps, minus_att1);
        break;
    }
    double r_plus[3], r_minus[3];
    if (!imu_body_delta_residual_from_vectors(
            delta, accel_jac, accel_fallback_diag, gyro_jac, gyro_fallback_diag,
            delta_angle, angle_gyro_jac, angle_fallback_diag,
            plus_ba, has_accel_bias, plus_bg, has_delta_gyro_bias, false,
            plus_p0, plus_v0, plus_att0, plus_p1, plus_v1, plus_att1, gravity, dt,
            is_position, t, r_plus)) {
      continue;
    }
    if (!imu_body_delta_residual_from_vectors(
            delta, accel_jac, accel_fallback_diag, gyro_jac, gyro_fallback_diag,
            delta_angle, angle_gyro_jac, angle_fallback_diag,
            minus_ba, has_accel_bias, minus_bg, has_delta_gyro_bias, false,
            minus_p0, minus_v0, minus_att0, minus_p1, minus_v1, minus_att1, gravity, dt,
            is_position, t, r_minus)) {
      continue;
    }
    for (int r = 0; r < 3; r++) {
      J[r][k] = (r_plus[r] - r_minus[r]) / (2.0 * eps);
    }
  }
}

inline void clear_imu_pva_sparse(int idx[9][kImuPvaMaxTerms], double jac[9][kImuPvaMaxTerms]) {
  for (int r = 0; r < 9; r++) {
    for (int k = 0; k < kImuPvaMaxTerms; k++) {
      idx[r][k] = -1;
      jac[r][k] = 0.0;
    }
  }
}

void fill_imu_pva_interval_residuals(
    int t, int ss,
    const double* imu_delta_p, const double* imu_delta_v, const double* imu_delta_angle,
    const double* imu_delta_p_bias_accel_jac, const double* imu_delta_v_bias_accel_jac,
    const double* imu_delta_p_bias_gyro_jac, const double* imu_delta_v_bias_gyro_jac,
    const double* imu_delta_angle_bias_gyro_jac, const double* imu_gravity,
    const double* state, const double* dt_arr, int pose_position_idx, int attitude_idx,
    int accel_bias_idx, int gyro_bias_idx,
    bool imu_factor_use_next_bias,
    double res[9], bool valid[9], int idx[9][kImuPvaMaxTerms], double jac[9][kImuPvaMaxTerms]) {
  for (int r = 0; r < 9; r++) {
    res[r] = 0.0;
    valid[r] = false;
  }
  clear_imu_pva_sparse(idx, jac);

  const int o0 = ss * t;
  const int o1 = ss * (t + 1);
  const int pos_idx = pose_position_idx >= 0 ? pose_position_idx : 0;
  const double dt = dt_arr ? dt_arr[t] : 0.0;
  const bool has_valid_dt = dt_arr && std::isfinite(dt) && dt > 0.0;
  if (!has_valid_dt) return;

  if (imu_delta_p) {
    const double* dp = imu_delta_p + (size_t)t * 3;
    if (imu_gravity && attitude_idx >= 0) {
      double body_res[3];
      if (imu_body_delta_residual_for_state(
              t, ss, imu_delta_p, imu_gravity,
              imu_delta_p_bias_accel_jac, 0.5 * dt * dt,
              imu_delta_p_bias_gyro_jac, 0.0,
              imu_delta_angle, imu_delta_angle_bias_gyro_jac, dt,
              state, dt_arr, pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
              imu_factor_use_next_bias, true, body_res)) {
        int body_idx[kImuPvaMaxTerms];
        double Jbody[3][kImuPvaMaxTerms];
        fill_body_imu_delta_jacobian_terms(
            t, ss, imu_delta_p, imu_gravity,
            imu_delta_p_bias_accel_jac, 0.5 * dt * dt,
            imu_delta_p_bias_gyro_jac, 0.0,
            imu_delta_angle, imu_delta_angle_bias_gyro_jac, dt,
            state, dt_arr, pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
            imu_factor_use_next_bias, true, body_idx, Jbody);
        for (int i = 0; i < 3; i++) {
          if (!std::isfinite(body_res[i])) continue;
          const int r = 3 + i;
          res[r] = body_res[i];
          valid[r] = true;
          for (int j = 0; j < kImuPvaMaxTerms; j++) {
            if (body_idx[j] < 0 || Jbody[i][j] == 0.0) continue;
            idx[r][j] = body_idx[j];
            jac[r][j] = Jbody[i][j];
          }
        }
      }
    } else {
      const bool use_direct_gyro_bias = imu_bias_jacobian_interval_has_nonzero(imu_delta_p_bias_gyro_jac, t);
      double rot_dp[3];
      if (!rotated_imu_delta_for_state(t, ss, imu_delta_p,
                                       imu_delta_p_bias_accel_jac, 0.5 * dt * dt,
                                       imu_delta_p_bias_gyro_jac, 0.0,
                                       imu_delta_angle_bias_gyro_jac, dt,
                                       state, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                       imu_factor_use_next_bias, use_direct_gyro_bias, rot_dp)) {
        rot_dp[0] = dp[0];
        rot_dp[1] = dp[1];
        rot_dp[2] = dp[2];
      }
      int rot_idx[kImuPvaMaxTerms];
      double Jrot[3][kImuPvaMaxTerms];
      fill_rotated_imu_delta_jacobian_terms(t, ss, imu_delta_p,
                                            imu_delta_p_bias_accel_jac, 0.5 * dt * dt,
                                            imu_delta_p_bias_gyro_jac, 0.0,
                                            imu_delta_angle_bias_gyro_jac, dt,
                                            state, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                            imu_factor_use_next_bias, use_direct_gyro_bias,
                                            rot_idx, Jrot);
      for (int i = 0; i < 3; i++) {
        if (!std::isfinite(dp[i])) continue;
        const int r = 3 + i;
        res[r] = state[o0 + pos_idx + i] + state[o0 + 3 + i] * dt + rot_dp[i] - state[o1 + pos_idx + i];
        valid[r] = true;
        int k = 0;
        idx[r][k] = o0 + pos_idx + i;
        jac[r][k++] = 1.0;
        idx[r][k] = o0 + 3 + i;
        jac[r][k++] = dt;
        idx[r][k] = o1 + pos_idx + i;
        jac[r][k++] = -1.0;
        for (int j = 0; j < kImuPvaMaxTerms; j++) {
          if (rot_idx[j] < 0 || Jrot[i][j] == 0.0) continue;
          idx[r][k] = rot_idx[j];
          jac[r][k++] = Jrot[i][j];
        }
      }
    }
  }

  if (imu_delta_v) {
    const double* dv = imu_delta_v + (size_t)t * 3;
    if (imu_gravity && attitude_idx >= 0) {
      double body_res[3];
      if (imu_body_delta_residual_for_state(
              t, ss, imu_delta_v, imu_gravity,
              imu_delta_v_bias_accel_jac, dt,
              imu_delta_v_bias_gyro_jac, 0.0,
              imu_delta_angle, imu_delta_angle_bias_gyro_jac, dt,
              state, dt_arr, pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
              imu_factor_use_next_bias, false, body_res)) {
        int body_idx[kImuPvaMaxTerms];
        double Jbody[3][kImuPvaMaxTerms];
        fill_body_imu_delta_jacobian_terms(
            t, ss, imu_delta_v, imu_gravity,
            imu_delta_v_bias_accel_jac, dt,
            imu_delta_v_bias_gyro_jac, 0.0,
            imu_delta_angle, imu_delta_angle_bias_gyro_jac, dt,
            state, dt_arr, pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
            imu_factor_use_next_bias, false, body_idx, Jbody);
        for (int i = 0; i < 3; i++) {
          if (!std::isfinite(body_res[i])) continue;
          const int r = 6 + i;
          res[r] = body_res[i];
          valid[r] = true;
          for (int j = 0; j < kImuPvaMaxTerms; j++) {
            if (body_idx[j] < 0 || Jbody[i][j] == 0.0) continue;
            idx[r][j] = body_idx[j];
            jac[r][j] = Jbody[i][j];
          }
        }
      }
    } else {
      const bool use_direct_gyro_bias = imu_bias_jacobian_interval_has_nonzero(imu_delta_v_bias_gyro_jac, t);
      double rot_dv[3];
      if (!rotated_imu_delta_for_state(t, ss, imu_delta_v,
                                       imu_delta_v_bias_accel_jac, dt,
                                       imu_delta_v_bias_gyro_jac, 0.0,
                                       imu_delta_angle_bias_gyro_jac, dt,
                                       state, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                       imu_factor_use_next_bias, use_direct_gyro_bias, rot_dv)) {
        rot_dv[0] = dv[0];
        rot_dv[1] = dv[1];
        rot_dv[2] = dv[2];
      }
      int rot_idx[kImuPvaMaxTerms];
      double Jrot[3][kImuPvaMaxTerms];
      fill_rotated_imu_delta_jacobian_terms(t, ss, imu_delta_v,
                                            imu_delta_v_bias_accel_jac, dt,
                                            imu_delta_v_bias_gyro_jac, 0.0,
                                            imu_delta_angle_bias_gyro_jac, dt,
                                            state, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                            imu_factor_use_next_bias, use_direct_gyro_bias,
                                            rot_idx, Jrot);
      for (int i = 0; i < 3; i++) {
        if (!std::isfinite(dv[i])) continue;
        const int r = 6 + i;
        const int v0 = o0 + 3 + i;
        const int v1 = o1 + 3 + i;
        res[r] = state[v0] + rot_dv[i] - state[v1];
        valid[r] = true;
        int k = 0;
        idx[r][k] = v0;
        jac[r][k++] = 1.0;
        idx[r][k] = v1;
        jac[r][k++] = -1.0;
        for (int j = 0; j < kImuPvaMaxTerms; j++) {
          if (rot_idx[j] < 0 || Jrot[i][j] == 0.0) continue;
          idx[r][k] = rot_idx[j];
          jac[r][k++] = Jrot[i][j];
        }
      }
    }
  }

  if (imu_delta_angle && attitude_idx >= 0) {
    double rot_res[3];
    if (!imu_rotation_residual_for_state(t, ss, imu_delta_angle, imu_delta_angle_bias_gyro_jac,
                                         state, dt_arr, attitude_idx, gyro_bias_idx,
                                         imu_factor_use_next_bias, rot_res)) {
      return;
    }
    int term_idx[kImuPvaMaxTerms];
    double Jrot[3][kImuPvaMaxTerms];
    fill_imu_rotation_jacobian_terms(t, ss, imu_delta_angle, imu_delta_angle_bias_gyro_jac,
                                     state, dt_arr, attitude_idx, gyro_bias_idx,
                                     imu_factor_use_next_bias, term_idx, Jrot);
    for (int i = 0; i < 3; i++) {
      const int r = i;
      if (!std::isfinite(rot_res[i])) continue;
      res[r] = rot_res[i];
      valid[r] = true;
      for (int k = 0; k < kImuPvaMaxTerms; k++) {
        if (term_idx[k] < 0) continue;
        idx[r][k] = term_idx[k];
        jac[r][k] = Jrot[i][k];
      }
    }
  }
}

void add_imu_pva_factor_host(
    int n_epoch, int ss, int n_state,
    const double* imu_delta_p, const double* imu_delta_v, const double* imu_delta_angle,
    const double* imu_delta_p_bias_accel_jac, const double* imu_delta_v_bias_accel_jac,
    const double* imu_delta_p_bias_gyro_jac, const double* imu_delta_v_bias_gyro_jac,
    const double* imu_delta_angle_bias_gyro_jac,
    const double* imu_pva_information,
    const double* state, const double* dt_arr, const double* imu_gravity,
    int pose_position_idx, int attitude_idx, int accel_bias_idx, int gyro_bias_idx,
    bool imu_factor_use_next_bias,
    double* H, double* g) {
  if (!imu_pva_information || (!imu_delta_p && !imu_delta_v && !imu_delta_angle)) return;

  for (int t = 0; t < n_epoch - 1; t++) {
    double res[9];
    bool valid[9];
    int idx[9][kImuPvaMaxTerms];
    double jac[9][kImuPvaMaxTerms];
    fill_imu_pva_interval_residuals(t, ss, imu_delta_p, imu_delta_v, imu_delta_angle,
                                    imu_delta_p_bias_accel_jac, imu_delta_v_bias_accel_jac,
                                    imu_delta_p_bias_gyro_jac, imu_delta_v_bias_gyro_jac,
                                    imu_delta_angle_bias_gyro_jac, imu_gravity, state, dt_arr,
                                    pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                    imu_factor_use_next_bias,
                                    res, valid, idx, jac);
    const double* info = imu_pva_information + (size_t)t * 81;
    double weighted_res[9] = {};
    bool any = false;
    for (int a = 0; a < 9; a++) {
      if (!valid[a]) continue;
      for (int b = 0; b < 9; b++) {
        if (!valid[b]) continue;
        const double w = info[a * 9 + b];
        if (!std::isfinite(w) || w == 0.0) continue;
        weighted_res[a] += w * res[b];
        any = true;
      }
    }
    if (!any) continue;

    for (int a = 0; a < 9; a++) {
      if (!valid[a] || weighted_res[a] == 0.0) continue;
      for (int ia = 0; ia < kImuPvaMaxTerms; ia++) {
        if (idx[a][ia] < 0 || jac[a][ia] == 0.0) continue;
        g[idx[a][ia]] -= jac[a][ia] * weighted_res[a];
      }
    }

    for (int a = 0; a < 9; a++) {
      if (!valid[a]) continue;
      for (int b = 0; b < 9; b++) {
        if (!valid[b]) continue;
        const double w = info[a * 9 + b];
        if (!std::isfinite(w) || w == 0.0) continue;
        for (int ia = 0; ia < kImuPvaMaxTerms; ia++) {
          if (idx[a][ia] < 0 || jac[a][ia] == 0.0) continue;
          for (int ib = 0; ib < kImuPvaMaxTerms; ib++) {
            if (idx[b][ib] < 0 || jac[b][ib] == 0.0) continue;
            H[(size_t)idx[a][ia] * n_state + idx[b][ib]] += w * jac[a][ia] * jac[b][ib];
          }
        }
      }
    }
  }
}

double imu_pva_cost_host(
    int n_epoch, int ss,
    const double* imu_delta_p, const double* imu_delta_v, const double* imu_delta_angle,
    const double* imu_delta_p_bias_accel_jac, const double* imu_delta_v_bias_accel_jac,
    const double* imu_delta_p_bias_gyro_jac, const double* imu_delta_v_bias_gyro_jac,
    const double* imu_delta_angle_bias_gyro_jac,
    const double* imu_pva_information,
    const double* state, const double* dt_arr, const double* imu_gravity,
    int pose_position_idx, int attitude_idx, int accel_bias_idx, int gyro_bias_idx,
    bool imu_factor_use_next_bias) {
  if (!imu_pva_information || (!imu_delta_p && !imu_delta_v && !imu_delta_angle)) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch - 1; t++) {
    double res[9];
    bool valid[9];
    int idx[9][kImuPvaMaxTerms];
    double jac[9][kImuPvaMaxTerms];
    fill_imu_pva_interval_residuals(t, ss, imu_delta_p, imu_delta_v, imu_delta_angle,
                                    imu_delta_p_bias_accel_jac, imu_delta_v_bias_accel_jac,
                                    imu_delta_p_bias_gyro_jac, imu_delta_v_bias_gyro_jac,
                                    imu_delta_angle_bias_gyro_jac, imu_gravity, state, dt_arr,
                                    pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                    imu_factor_use_next_bias,
                                    res, valid, idx, jac);
    const double* info = imu_pva_information + (size_t)t * 81;
    double q = 0.0;
    for (int a = 0; a < 9; a++) {
      if (!valid[a]) continue;
      for (int b = 0; b < 9; b++) {
        if (!valid[b]) continue;
        const double w = info[a * 9 + b];
        if (!std::isfinite(w) || w == 0.0) continue;
        q += res[a] * w * res[b];
      }
    }
    e += 0.5 * q;
  }
  return e;
}

void add_imu_prior_factor_host(
    int n_epoch, int ss, int n_state,
    double w_imu_pos, double w_imu_vel,
    const double* imu_delta_p, const double* imu_delta_v, const double* imu_delta_angle,
    const double* imu_delta_p_bias_accel_jac, const double* imu_delta_v_bias_accel_jac,
    const double* imu_delta_p_bias_gyro_jac, const double* imu_delta_v_bias_gyro_jac,
    const double* imu_delta_angle_bias_gyro_jac,
    const double* imu_position_weights, const double* imu_velocity_weights,
    const double* state, const double* dt_arr, const double* imu_gravity,
    int pose_position_idx, int attitude_idx, int accel_bias_idx, int gyro_bias_idx,
    bool imu_factor_use_next_bias, double* H, double* g) {
  if (!imu_delta_p && !imu_delta_v) return;

  for (int t = 0; t < n_epoch - 1; t++) {
    const int o0 = ss * t;
    const int o1 = ss * (t + 1);
    const int pos_idx = pose_position_idx >= 0 ? pose_position_idx : 0;
    const double dt = dt_arr ? dt_arr[t] : 0.0;
    const bool has_valid_dt = dt_arr && std::isfinite(dt) && dt > 0.0;

    if (imu_delta_p && has_valid_dt && (w_imu_pos > 0.0 || imu_position_weights != nullptr)) {
      const double* dp = imu_delta_p + (size_t)t * 3;
      if (imu_gravity && attitude_idx >= 0) {
        double body_res[3];
        if (imu_body_delta_residual_for_state(
                t, ss, imu_delta_p, imu_gravity,
                imu_delta_p_bias_accel_jac, 0.5 * dt * dt,
                imu_delta_p_bias_gyro_jac, 0.0,
                imu_delta_angle, imu_delta_angle_bias_gyro_jac, dt,
                state, dt_arr, pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
                imu_factor_use_next_bias, true, body_res)) {
          int body_idx[kImuPvaMaxTerms];
          double Jbody[3][kImuPvaMaxTerms];
          fill_body_imu_delta_jacobian_terms(
              t, ss, imu_delta_p, imu_gravity,
              imu_delta_p_bias_accel_jac, 0.5 * dt * dt,
              imu_delta_p_bias_gyro_jac, 0.0,
              imu_delta_angle, imu_delta_angle_bias_gyro_jac, dt,
              state, dt_arr, pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
              imu_factor_use_next_bias, true, body_idx, Jbody);
          int n_terms = 0;
          while (n_terms < kImuPvaMaxTerms && body_idx[n_terms] >= 0) n_terms++;
          for (int i = 0; i < 3; i++) {
            const double wi = component_weight(w_imu_pos, imu_position_weights, t, i);
            if (wi <= 0.0 || !std::isfinite(body_res[i])) continue;
            add_linear_residual_host(n_terms, body_idx, Jbody[i], wi, body_res[i], n_state, H, g);
          }
        }
      } else {
        const bool use_direct_gyro_bias = imu_bias_jacobian_interval_has_nonzero(imu_delta_p_bias_gyro_jac, t);
        double rot_dp[3];
        if (!rotated_imu_delta_for_state(t, ss, imu_delta_p,
                                         imu_delta_p_bias_accel_jac, 0.5 * dt * dt,
                                         imu_delta_p_bias_gyro_jac, 0.0,
                                         imu_delta_angle_bias_gyro_jac, dt,
                                         state, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                         imu_factor_use_next_bias, use_direct_gyro_bias, rot_dp)) {
          rot_dp[0] = dp[0];
          rot_dp[1] = dp[1];
          rot_dp[2] = dp[2];
        }
        int rot_idx[kImuPvaMaxTerms];
        double Jrot[3][kImuPvaMaxTerms];
        fill_rotated_imu_delta_jacobian_terms(t, ss, imu_delta_p,
                                              imu_delta_p_bias_accel_jac, 0.5 * dt * dt,
                                              imu_delta_p_bias_gyro_jac, 0.0,
                                              imu_delta_angle_bias_gyro_jac, dt,
                                              state, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                              imu_factor_use_next_bias, use_direct_gyro_bias,
                                              rot_idx, Jrot);
        for (int i = 0; i < 3; i++) {
          if (!std::isfinite(dp[i])) continue;
          const double wi = component_weight(w_imu_pos, imu_position_weights, t, i);
          if (wi <= 0.0) continue;
          double res = state[o0 + pos_idx + i] + state[o0 + 3 + i] * dt + rot_dp[i] - state[o1 + pos_idx + i];
          int idx[kImuPvaMaxTerms];
          double jac[kImuPvaMaxTerms];
          for (int k = 0; k < kImuPvaMaxTerms; k++) {
            idx[k] = -1;
            jac[k] = 0.0;
          }
          int k = 0;
          idx[k] = o0 + pos_idx + i;
          jac[k++] = 1.0;
          idx[k] = o0 + 3 + i;
          jac[k++] = dt;
          idx[k] = o1 + pos_idx + i;
          jac[k++] = -1.0;
          for (int j = 0; j < kImuPvaMaxTerms; j++) {
            if (rot_idx[j] < 0 || Jrot[i][j] == 0.0) continue;
            idx[k] = rot_idx[j];
            jac[k++] = Jrot[i][j];
          }
          add_linear_residual_host(k, idx, jac, wi, res, n_state, H, g);
        }
      }
    }

    if (imu_delta_v && (w_imu_vel > 0.0 || imu_velocity_weights != nullptr)) {
      const double* dv = imu_delta_v + (size_t)t * 3;
      if (imu_gravity && attitude_idx >= 0) {
        double body_res[3];
        if (has_valid_dt &&
            imu_body_delta_residual_for_state(
                t, ss, imu_delta_v, imu_gravity,
                imu_delta_v_bias_accel_jac, dt,
                imu_delta_v_bias_gyro_jac, 0.0,
                imu_delta_angle, imu_delta_angle_bias_gyro_jac, dt,
                state, dt_arr, pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
                imu_factor_use_next_bias, false, body_res)) {
          int body_idx[kImuPvaMaxTerms];
          double Jbody[3][kImuPvaMaxTerms];
          fill_body_imu_delta_jacobian_terms(
              t, ss, imu_delta_v, imu_gravity,
              imu_delta_v_bias_accel_jac, dt,
              imu_delta_v_bias_gyro_jac, 0.0,
              imu_delta_angle, imu_delta_angle_bias_gyro_jac, dt,
              state, dt_arr, pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
              imu_factor_use_next_bias, false, body_idx, Jbody);
          int n_terms = 0;
          while (n_terms < kImuPvaMaxTerms && body_idx[n_terms] >= 0) n_terms++;
          for (int i = 0; i < 3; i++) {
            const double wi = component_weight(w_imu_vel, imu_velocity_weights, t, i);
            if (wi <= 0.0 || !std::isfinite(body_res[i])) continue;
            add_linear_residual_host(n_terms, body_idx, Jbody[i], wi, body_res[i], n_state, H, g);
          }
        }
      } else {
        const bool use_direct_gyro_bias = imu_bias_jacobian_interval_has_nonzero(imu_delta_v_bias_gyro_jac, t);
        double rot_dv[3];
        if (!rotated_imu_delta_for_state(t, ss, imu_delta_v,
                                         imu_delta_v_bias_accel_jac, dt,
                                         imu_delta_v_bias_gyro_jac, 0.0,
                                         imu_delta_angle_bias_gyro_jac, dt,
                                         state, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                         imu_factor_use_next_bias, use_direct_gyro_bias, rot_dv)) {
          rot_dv[0] = dv[0];
          rot_dv[1] = dv[1];
          rot_dv[2] = dv[2];
        }
        int rot_idx[kImuPvaMaxTerms];
        double Jrot[3][kImuPvaMaxTerms];
        fill_rotated_imu_delta_jacobian_terms(t, ss, imu_delta_v,
                                              imu_delta_v_bias_accel_jac, dt,
                                              imu_delta_v_bias_gyro_jac, 0.0,
                                              imu_delta_angle_bias_gyro_jac, dt,
                                              state, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                              imu_factor_use_next_bias, use_direct_gyro_bias,
                                              rot_idx, Jrot);
        for (int i = 0; i < 3; i++) {
          if (!std::isfinite(dv[i])) continue;
          const double wi = component_weight(w_imu_vel, imu_velocity_weights, t, i);
          if (wi <= 0.0) continue;
          const int v0 = o0 + 3 + i;
          const int v1 = o1 + 3 + i;
          double res = state[v0] + rot_dv[i] - state[v1];
          int idx[kImuPvaMaxTerms];
          double jac[kImuPvaMaxTerms];
          for (int k = 0; k < kImuPvaMaxTerms; k++) {
            idx[k] = -1;
            jac[k] = 0.0;
          }
          int k = 0;
          idx[k] = v0;
          jac[k++] = 1.0;
          idx[k] = v1;
          jac[k++] = -1.0;
          for (int j = 0; j < kImuPvaMaxTerms; j++) {
            if (rot_idx[j] < 0 || Jrot[i][j] == 0.0) continue;
            idx[k] = rot_idx[j];
            jac[k++] = Jrot[i][j];
          }
          add_linear_residual_host(k, idx, jac, wi, res, n_state, H, g);
        }
      }
    }
  }
}

double imu_prior_cost_host(
    int n_epoch, int ss,
    double w_imu_pos, double w_imu_vel,
    const double* imu_delta_p, const double* imu_delta_v, const double* imu_delta_angle,
    const double* imu_delta_p_bias_accel_jac, const double* imu_delta_v_bias_accel_jac,
    const double* imu_delta_p_bias_gyro_jac, const double* imu_delta_v_bias_gyro_jac,
    const double* imu_delta_angle_bias_gyro_jac,
    const double* imu_position_weights, const double* imu_velocity_weights,
    const double* state, const double* dt_arr, const double* imu_gravity,
    int pose_position_idx, int attitude_idx, int accel_bias_idx, int gyro_bias_idx,
    bool imu_factor_use_next_bias) {
  if (!imu_delta_p && !imu_delta_v) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch - 1; t++) {
    const int o0 = ss * t;
    const int o1 = ss * (t + 1);
    const int pos_idx = pose_position_idx >= 0 ? pose_position_idx : 0;
    const double dt = dt_arr ? dt_arr[t] : 0.0;
    const bool has_valid_dt = dt_arr && std::isfinite(dt) && dt > 0.0;

    if (imu_delta_p && has_valid_dt && (w_imu_pos > 0.0 || imu_position_weights != nullptr)) {
      const double* dp = imu_delta_p + (size_t)t * 3;
      if (imu_gravity && attitude_idx >= 0) {
        double body_res[3];
        if (imu_body_delta_residual_for_state(
                t, ss, imu_delta_p, imu_gravity,
                imu_delta_p_bias_accel_jac, 0.5 * dt * dt,
                imu_delta_p_bias_gyro_jac, 0.0,
                imu_delta_angle, imu_delta_angle_bias_gyro_jac, dt,
                state, dt_arr, pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
                imu_factor_use_next_bias, true, body_res)) {
          for (int i = 0; i < 3; i++) {
            const double wi = component_weight(w_imu_pos, imu_position_weights, t, i);
            if (wi <= 0.0 || !std::isfinite(body_res[i])) continue;
            e += 0.5 * wi * body_res[i] * body_res[i];
          }
        }
      } else {
        const bool use_direct_gyro_bias = imu_bias_jacobian_interval_has_nonzero(imu_delta_p_bias_gyro_jac, t);
        double rot_dp[3];
        if (!rotated_imu_delta_for_state(t, ss, imu_delta_p,
                                         imu_delta_p_bias_accel_jac, 0.5 * dt * dt,
                                         imu_delta_p_bias_gyro_jac, 0.0,
                                         imu_delta_angle_bias_gyro_jac, dt,
                                         state, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                         imu_factor_use_next_bias, use_direct_gyro_bias, rot_dp)) {
          rot_dp[0] = dp[0];
          rot_dp[1] = dp[1];
          rot_dp[2] = dp[2];
        }
        for (int i = 0; i < 3; i++) {
          if (!std::isfinite(dp[i])) continue;
          const double wi = component_weight(w_imu_pos, imu_position_weights, t, i);
          if (wi <= 0.0) continue;
          double res = state[o0 + pos_idx + i] + state[o0 + 3 + i] * dt + rot_dp[i] - state[o1 + pos_idx + i];
          e += 0.5 * wi * res * res;
        }
      }
    }

    if (imu_delta_v && (w_imu_vel > 0.0 || imu_velocity_weights != nullptr)) {
      const double* dv = imu_delta_v + (size_t)t * 3;
      if (imu_gravity && attitude_idx >= 0) {
        double body_res[3];
        if (has_valid_dt &&
            imu_body_delta_residual_for_state(
                t, ss, imu_delta_v, imu_gravity,
                imu_delta_v_bias_accel_jac, dt,
                imu_delta_v_bias_gyro_jac, 0.0,
                imu_delta_angle, imu_delta_angle_bias_gyro_jac, dt,
                state, dt_arr, pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
                imu_factor_use_next_bias, false, body_res)) {
          for (int i = 0; i < 3; i++) {
            const double wi = component_weight(w_imu_vel, imu_velocity_weights, t, i);
            if (wi <= 0.0 || !std::isfinite(body_res[i])) continue;
            e += 0.5 * wi * body_res[i] * body_res[i];
          }
        }
      } else {
        const bool use_direct_gyro_bias = imu_bias_jacobian_interval_has_nonzero(imu_delta_v_bias_gyro_jac, t);
        double rot_dv[3];
        if (!rotated_imu_delta_for_state(t, ss, imu_delta_v,
                                         imu_delta_v_bias_accel_jac, dt,
                                         imu_delta_v_bias_gyro_jac, 0.0,
                                         imu_delta_angle_bias_gyro_jac, dt,
                                         state, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                         imu_factor_use_next_bias, use_direct_gyro_bias, rot_dv)) {
          rot_dv[0] = dv[0];
          rot_dv[1] = dv[1];
          rot_dv[2] = dv[2];
        }
        for (int i = 0; i < 3; i++) {
          if (!std::isfinite(dv[i])) continue;
          const double wi = component_weight(w_imu_vel, imu_velocity_weights, t, i);
          if (wi <= 0.0) continue;
          double res = state[o0 + 3 + i] + rot_dv[i] - state[o1 + 3 + i];
          e += 0.5 * wi * res * res;
        }
      }
    }
  }
  return e;
}

void add_imu_attitude_factor_host(
    int n_epoch, int ss, int n_state, double w_imu_att,
    const double* imu_delta_angle, const double* imu_attitude_weights,
    const double* imu_delta_angle_bias_gyro_jac,
    const double* state, const double* dt_arr,
    int attitude_idx, int gyro_bias_idx, bool imu_factor_use_next_bias, double* H, double* g) {
  if (!imu_delta_angle || attitude_idx < 0) return;

  for (int t = 0; t < n_epoch - 1; t++) {
    double rot_res[3];
    if (!imu_rotation_residual_for_state(t, ss, imu_delta_angle, imu_delta_angle_bias_gyro_jac,
                                         state, dt_arr, attitude_idx, gyro_bias_idx,
                                         imu_factor_use_next_bias, rot_res)) {
      continue;
    }
    int term_idx[kImuPvaMaxTerms];
    double Jrot[3][kImuPvaMaxTerms];
    fill_imu_rotation_jacobian_terms(t, ss, imu_delta_angle, imu_delta_angle_bias_gyro_jac,
                                     state, dt_arr, attitude_idx, gyro_bias_idx,
                                     imu_factor_use_next_bias, term_idx, Jrot);
    int n_terms = 0;
    while (n_terms < kImuPvaMaxTerms && term_idx[n_terms] >= 0) n_terms++;
    if (n_terms == 0) continue;
    for (int i = 0; i < 3; i++) {
      const double wi = component_weight(w_imu_att, imu_attitude_weights, t, i);
      if (wi <= 0.0) continue;
      if (!std::isfinite(rot_res[i])) continue;
      add_linear_residual_host(n_terms, term_idx, Jrot[i], wi, rot_res[i], n_state, H, g);
    }
  }
}

double imu_attitude_cost_host(
    int n_epoch, int ss, double w_imu_att, const double* imu_delta_angle,
    const double* imu_attitude_weights, const double* imu_delta_angle_bias_gyro_jac,
    const double* state, const double* dt_arr,
    int attitude_idx, int gyro_bias_idx, bool imu_factor_use_next_bias) {
  if (!imu_delta_angle || attitude_idx < 0) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch - 1; t++) {
    double rot_res[3];
    if (!imu_rotation_residual_for_state(t, ss, imu_delta_angle, imu_delta_angle_bias_gyro_jac,
                                         state, dt_arr, attitude_idx, gyro_bias_idx,
                                         imu_factor_use_next_bias, rot_res)) {
      continue;
    }
    for (int i = 0; i < 3; i++) {
      const double wi = component_weight(w_imu_att, imu_attitude_weights, t, i);
      if (wi <= 0.0) continue;
      if (!std::isfinite(rot_res[i])) continue;
      e += 0.5 * wi * rot_res[i] * rot_res[i];
    }
  }
  return e;
}

void add_bias_triplet_factor_host(
    int n_epoch, int ss, int n_state, int bias_idx,
    double w_bias_prior, double w_bias_between,
    const double* bias_between_weights,
    const double* state, double* H, double* g) {
  if (bias_idx < 0) return;

  if (w_bias_prior > 0.0) {
    const int o0 = 0;
    for (int i = 0; i < 3; i++) {
      const int idx = o0 + bias_idx + i;
      const double res = state[idx];
      g[idx] -= w_bias_prior * res;
      H[(size_t)idx * n_state + idx] += w_bias_prior;
    }
  }

  if (w_bias_between <= 0.0 && !bias_between_weights) return;
  for (int t = 0; t < n_epoch - 1; t++) {
    const int o0 = ss * t;
    const int o1 = ss * (t + 1);
    for (int i = 0; i < 3; i++) {
      const double wi = component_weight(w_bias_between, bias_between_weights, t, i);
      if (wi <= 0.0) continue;
      const int b0 = o0 + bias_idx + i;
      const int b1 = o1 + bias_idx + i;
      const double res = state[b0] - state[b1];
      const double Jr = wi * res;
      g[b0] -= Jr;
      g[b1] += Jr;
      H[(size_t)b0 * n_state + b0] += wi;
      H[(size_t)b0 * n_state + b1] -= wi;
      H[(size_t)b1 * n_state + b0] -= wi;
      H[(size_t)b1 * n_state + b1] += wi;
    }
  }
}

double bias_triplet_cost_host(
    int n_epoch, int ss, int bias_idx,
    double w_bias_prior, double w_bias_between,
    const double* bias_between_weights,
    const double* state) {
  if (bias_idx < 0) return 0.0;
  double e = 0.0;
  if (w_bias_prior > 0.0) {
    for (int i = 0; i < 3; i++) {
      const double res = state[bias_idx + i];
      e += 0.5 * w_bias_prior * res * res;
    }
  }
  if (w_bias_between > 0.0 || bias_between_weights) {
    for (int t = 0; t < n_epoch - 1; t++) {
      const int o0 = ss * t;
      const int o1 = ss * (t + 1);
      for (int i = 0; i < 3; i++) {
        const double wi = component_weight(w_bias_between, bias_between_weights, t, i);
        if (wi <= 0.0) continue;
        const double res = state[o0 + bias_idx + i] - state[o1 + bias_idx + i];
        e += 0.5 * wi * res * res;
      }
    }
  }
  return e;
}

// Relative height (ENU up) equality: residual = u·(x_i - x_j), u = unit ENU-up in ECEF.
void add_relative_height_factor_host(
    int n_epoch, int ss, int n_state, double w_rel_h,
    double relative_height_huber_k,
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
    double w_eff = huber_effective_weight(w_rel_h, r, relative_height_huber_k);
    for (int k = 0; k < 3; k++) {
      double uk = (k == 0) ? ux : ((k == 1) ? uy : uz);
      g[oi + k] -= w_eff * r * uk;
      g[oj + k] += w_eff * r * uk;
    }
    for (int a = 0; a < 3; a++) {
      double ua = (a == 0) ? ux : ((a == 1) ? uy : uz);
      for (int b = 0; b < 3; b++) {
        double ub = (b == 0) ? ux : ((b == 1) ? uy : uz);
        double hij = w_eff * ua * ub;
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
    double relative_height_huber_k,
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
    cost += huber_loss(w_rel_h, r, relative_height_huber_k);
  }
  return cost;
}

// Absolute height prior: residual = u·(ref_t - x_t), u = unit ENU-up in ECEF.
void add_absolute_height_factor_host(
    int n_epoch, int ss, int n_state, double w_abs_h,
    double absolute_height_huber_k,
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
    const double w_eff = huber_effective_weight(w_abs_h, r, absolute_height_huber_k);
    const double u[3] = {ux, uy, uz};
    for (int a = 0; a < 3; a++) {
      g[o + a] += w_eff * r * u[a];
      for (int b = 0; b < 3; b++) {
        H[(size_t)(o + a) * n_state + (o + b)] += w_eff * u[a] * u[b];
      }
    }
  }
}

double absolute_height_cost_host(
    int n_epoch, int ss, double w_abs_h,
    double absolute_height_huber_k,
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
    cost += huber_loss(w_abs_h, r, absolute_height_huber_k);
  }
  return cost;
}

// Doppler factor cost
double doppler_cost_host(
    int n_epoch, int n_sat, int nc, int ss,
    const double* sat_ecef, const double* sat_vel,
    const double* sat_clock_drift,
    const double* doppler_linearization_ref_vel,
    const double* doppler_linearization_los_ecef,
    const double* doppler, const double* doppler_weights,
    double doppler_huber_k,
    const double* state) {
  if (!doppler || !doppler_weights) return 0.0;
  double e = 0.0;
  for (int t = 0; t < n_epoch; t++) {
    const double* my_dop = doppler + (size_t)t * n_sat;
    const double* my_dw = doppler_weights + (size_t)t * n_sat;
    for (int s = 0; s < n_sat; s++) {
      double w = my_dw[s];
      if (w <= 0.0) continue;
      double pred = 0.0;
      double j_vel[3] = {};
      if (!doppler_prediction_vd(n_sat, nc, ss, t, s, sat_ecef, sat_vel, sat_clock_drift,
                                 doppler_linearization_ref_vel, doppler_linearization_los_ecef,
                                 state, &pred, j_vel)) {
        continue;
      }
      double res = my_dop[s] - pred;
      e += huber_loss(w, res, doppler_huber_k);
    }
  }
  return e;
}

// Effective Huber weights for VD state (clock at index 6)
void effective_pr_weights_huber_host_vd(
    int n_epoch, int n_sat, int nc, int ss,
    const double* sat_ecef,
    const double* pr_linearization_ref_ecef,
    const double* pr_linearization_los_ecef,
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
    const double* my_pr = pseudorange + (size_t)t * n_sat;
    const double* my_w = weights + (size_t)t * n_sat;
    for (int s = 0; s < n_sat; s++) {
      double w = my_w[s];
      size_t idx = (size_t)t * n_sat + s;
      if (w <= 0.0) { eff_w_out[idx] = w; continue; }
      double hc[kMaxClockVD];
      double j_pos[3] = {};
      double pred = 0.0;
      if (!pr_prediction_vd(n_sat, nc, ss, t, s, sat_ecef, pr_linearization_ref_ecef,
                            pr_linearization_los_ecef, sys_kind_host, state, &pred, j_pos, hc)) {
        eff_w_out[idx] = 0.0;
        continue;
      }
      double res = my_pr[s] - pred;
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
    const double* pr_linearization_ref_ecef,
    const double* pr_linearization_los_ecef,
    const double* pseudorange,
    const double* weights,
    const int* sys_kind_host,
    const double* state) {
  double sse = 0.0;
  int cnt = 0;
  for (int t = 0; t < n_epoch; t++) {
    const double* my_pr = pseudorange + (size_t)t * n_sat;
    const double* my_w = weights + (size_t)t * n_sat;
    for (int s = 0; s < n_sat; s++) {
      double w = my_w[s];
      if (w <= 0.0) continue;
      double hc[kMaxClockVD];
      double j_pos[3] = {};
      double pred = 0.0;
      if (!pr_prediction_vd(n_sat, nc, ss, t, s, sat_ecef, pr_linearization_ref_ecef,
                            pr_linearization_los_ecef, sys_kind_host, state, &pred, j_pos, hc)) {
        continue;
      }
      double res = my_pr[s] - pred;
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
    const double* tdcp_linearization_ref_ecef,
    const int* sys_kind_host,
    const double* dt_arr,
    const double* tdcp_meas,
    const double* tdcp_weights,
    double tdcp_sigma_m,
    bool tdcp_use_drift,
    double tdcp_huber_k,
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
    const double state_x0 = state[o0 + 0], state_y0 = state[o0 + 1], state_z0 = state[o0 + 2];
    const double state_x1 = state[o1 + 0], state_y1 = state[o1 + 1], state_z1 = state[o1 + 2];
    double los_rx_x = state_x1, los_rx_y = state_y1, los_rx_z = state_z1;
    int los_epoch = t + 1;
    double ref_x0 = 0.0, ref_y0 = 0.0, ref_z0 = 0.0;
    double ref_x1 = 0.0, ref_y1 = 0.0, ref_z1 = 0.0;
    const bool use_ref = tdcp_linearization_ref_ecef != nullptr;
    if (use_ref) {
      const double* ref0 = tdcp_linearization_ref_ecef + (size_t)t * 3;
      const double* ref1 = tdcp_linearization_ref_ecef + (size_t)(t + 1) * 3;
      ref_x0 = ref0[0];
      ref_y0 = ref0[1];
      ref_z0 = ref0[2];
      ref_x1 = ref1[0];
      ref_y1 = ref1[1];
      ref_z1 = ref1[2];
      if (!std::isfinite(ref_x0) || !std::isfinite(ref_y0) || !std::isfinite(ref_z0) ||
          !std::isfinite(ref_x1) || !std::isfinite(ref_y1) || !std::isfinite(ref_z1)) {
        continue;
      }
      los_rx_x = ref_x1;
      los_rx_y = ref_y1;
      los_rx_z = ref_z1;
    }

    const double* my_sat = sat_ecef + (size_t)los_epoch * n_sat * 3;

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
      double dx0 = los_rx_x - sx, dy0 = los_rx_y - sy, dz0 = los_rx_z - sz;
      double r0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
      double dx = dx0, dy_v = dy0, dz = dz0;
      if (!use_ref) {
        double transit = r0 / kC;
        double theta = kOmegaE * transit;
        double sx_rot = sx * cos(theta) + sy * sin(theta);
        double sy_rot = -sx * sin(theta) + sy * cos(theta);
        dx = los_rx_x - sx_rot;
        dy_v = los_rx_y - sy_rot;
      }
      double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      if (r < 1e-6) continue;

      double ex = dx / r, ey = dy_v / r, ez = dz / r;
      int sk = sys_kind_host ? sys_kind_host[los_epoch * n_sat + s] : 0;
      if (sk < 0 || sk >= nc) continue;
      double hc[kMaxClock];
      fill_hc_int(nc, sk, hc);

      // Residual: obs - pred (match pseudorange convention)
      double pred_tdcp = 0.0;
      if (use_ref) {
        pred_tdcp = ex * ((state_x1 - ref_x1) - (state_x0 - ref_x0)) +
                    ey * ((state_y1 - ref_y1) - (state_y0 - ref_y0)) +
                    ez * ((state_z1 - ref_z1) - (state_z0 - ref_z0));
      } else {
        pred_tdcp = ex * (state_x1 - state_x0) + ey * (state_y1 - state_y0) + ez * (state_z1 - state_z0);
      }
      if (tdcp_use_drift) {
        pred_tdcp += 0.5 * dt * (state[o0 + drift_idx] + state[o1 + drift_idx]);
      } else if (use_ref) {
        pred_tdcp += state[o1 + clk_idx] - state[o0 + clk_idx];
      } else {
        for (int k = 0; k < nc; k++) {
          pred_tdcp += hc[k] * (state[o1 + clk_idx + k] - state[o0 + clk_idx + k]);
        }
      }
      double res = meas - pred_tdcp;  // obs - pred
      const double w_eff = huber_effective_weight(w, res, tdcp_huber_k);

      // J_pred at t: [-ex,-ey,-ez,-1] / [+dt/2] for XXCC / XXDD.
      double Jr = w_eff * res;
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
            H[(size_t)idx0[a] * n_state + idx0[b]] += w_eff * Jt[a] * Jt[b];
        for (int a = 0; a < 4; a++)
          for (int b = 0; b < 4; b++)
            H[(size_t)idx1[a] * n_state + idx1[b]] += w_eff * Jt1[a] * Jt1[b];
        for (int a = 0; a < 4; a++)
          for (int b = 0; b < 4; b++) {
            H[(size_t)idx0[a] * n_state + idx1[b]] += w_eff * Jt[a] * Jt1[b];
            H[(size_t)idx1[a] * n_state + idx0[b]] += w_eff * Jt1[a] * Jt[b];
          }
      } else {
        double Jt[kMaxSSVD] = {};
        double Jt1[kMaxSSVD] = {};
        Jt[0] = -ex;
        Jt[1] = -ey;
        Jt[2] = -ez;
        Jt1[0] = ex;
        Jt1[1] = ey;
        Jt1[2] = ez;
        if (use_ref) {
          Jt[clk_idx] = -1.0;
          Jt1[clk_idx] = 1.0;
        } else {
          for (int k = 0; k < nc; k++) {
            Jt[clk_idx + k] = -hc[k];
            Jt1[clk_idx + k] = hc[k];
          }
        }

        for (int a = 0; a < ss; a++) {
          g[o0 + a] += Jt[a] * Jr;
          g[o1 + a] += Jt1[a] * Jr;
        }

        for (int a = 0; a < ss; a++)
          for (int b = 0; b < ss; b++) {
            H[(size_t)(o0 + a) * n_state + (o0 + b)] += w_eff * Jt[a] * Jt[b];
            H[(size_t)(o1 + a) * n_state + (o1 + b)] += w_eff * Jt1[a] * Jt1[b];
            H[(size_t)(o0 + a) * n_state + (o1 + b)] += w_eff * Jt[a] * Jt1[b];
            H[(size_t)(o1 + a) * n_state + (o0 + b)] += w_eff * Jt1[a] * Jt[b];
          }
      }
    }
  }
}

double tdcp_cost_host_vd(
    int n_epoch, int n_sat, int nc, int ss,
    const double* sat_ecef,
    const double* tdcp_linearization_ref_ecef,
    const int* sys_kind_host,
    const double* dt_arr,
    const double* tdcp_meas,
    const double* tdcp_weights,
    double tdcp_sigma_m,
    bool tdcp_use_drift,
    double tdcp_huber_k,
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
    const double state_x0 = state[o0 + 0], state_y0 = state[o0 + 1], state_z0 = state[o0 + 2];
    const double state_x1 = state[o1 + 0], state_y1 = state[o1 + 1], state_z1 = state[o1 + 2];
    double los_rx_x = state_x1, los_rx_y = state_y1, los_rx_z = state_z1;
    int los_epoch = t + 1;
    double ref_x0 = 0.0, ref_y0 = 0.0, ref_z0 = 0.0;
    double ref_x1 = 0.0, ref_y1 = 0.0, ref_z1 = 0.0;
    const bool use_ref = tdcp_linearization_ref_ecef != nullptr;
    if (use_ref) {
      const double* ref0 = tdcp_linearization_ref_ecef + (size_t)t * 3;
      const double* ref1 = tdcp_linearization_ref_ecef + (size_t)(t + 1) * 3;
      ref_x0 = ref0[0];
      ref_y0 = ref0[1];
      ref_z0 = ref0[2];
      ref_x1 = ref1[0];
      ref_y1 = ref1[1];
      ref_z1 = ref1[2];
      if (!std::isfinite(ref_x0) || !std::isfinite(ref_y0) || !std::isfinite(ref_z0) ||
          !std::isfinite(ref_x1) || !std::isfinite(ref_y1) || !std::isfinite(ref_z1)) {
        continue;
      }
      los_rx_x = ref_x1;
      los_rx_y = ref_y1;
      los_rx_z = ref_z1;
    }

    const double* my_sat = sat_ecef + (size_t)los_epoch * n_sat * 3;

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
      double dx0 = los_rx_x - sx, dy0 = los_rx_y - sy, dz0 = los_rx_z - sz;
      double r0 = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
      double dx = dx0, dy_v = dy0, dz = dz0;
      if (!use_ref) {
        double transit = r0 / kC;
        double theta = kOmegaE * transit;
        double sx_rot = sx * cos(theta) + sy * sin(theta);
        double sy_rot = -sx * sin(theta) + sy * cos(theta);
        dx = los_rx_x - sx_rot;
        dy_v = los_rx_y - sy_rot;
      }
      double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      if (r < 1e-6) continue;

      double ex = dx / r, ey = dy_v / r, ez = dz / r;
      int sk = sys_kind_host ? sys_kind_host[los_epoch * n_sat + s] : 0;
      if (sk < 0 || sk >= nc) continue;
      double hc[kMaxClock];
      fill_hc_int(nc, sk, hc);
      double res = 0.0;
      if (use_ref) {
        res = ex * ((state_x1 - ref_x1) - (state_x0 - ref_x0)) +
              ey * ((state_y1 - ref_y1) - (state_y0 - ref_y0)) +
              ez * ((state_z1 - ref_z1) - (state_z0 - ref_z0)) - meas;
      } else {
        res = ex * (state_x1 - state_x0) + ey * (state_y1 - state_y0) + ez * (state_z1 - state_z0) - meas;
      }
      if (tdcp_use_drift) {
        res += 0.5 * dt * (state[o0 + drift_idx] + state[o1 + drift_idx]);
      } else if (use_ref) {
        res += state[o1 + clk_idx] - state[o0 + clk_idx];
      } else {
        for (int k = 0; k < nc; k++) {
          res += hc[k] * (state[o1 + clk_idx + k] - state[o0 + clk_idx + k]);
        }
      }
      e += huber_loss(w, res, tdcp_huber_k);
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
                   const double* imu_delta_angle,
                   const double* imu_delta_t,
                   const double* imu_delta_p_bias_accel_jac,
                   const double* imu_delta_v_bias_accel_jac,
                   const double* imu_delta_p_bias_gyro_jac,
                   const double* imu_delta_v_bias_gyro_jac,
                   const double* imu_delta_angle_bias_gyro_jac,
                   double imu_position_sigma_m,
                   double imu_velocity_sigma_mps,
                   double imu_attitude_sigma_rad,
                   const double* imu_position_weights,
                   const double* imu_velocity_weights,
                   const double* imu_attitude_weights,
                   const double* imu_preintegration_information,
                   bool imu_factor_use_next_bias,
                   const double* sat_clock_drift,
                   const double* absolute_height_ref_ecef,
                   double absolute_height_sigma_m,
                   int state_stride,
                   double imu_accel_bias_prior_sigma_mps2,
                   double imu_accel_bias_between_sigma_mps2,
                   const double* imu_accel_bias_between_weights,
                   double imu_gyro_bias_prior_sigma_radps,
                   double imu_gyro_bias_between_sigma_radps,
                   const double* imu_gyro_bias_between_weights,
                   double doppler_huber_k,
                   double tdcp_huber_k,
	                   const double* tdcp_linearization_ref_ecef,
	                   double stop_velocity_huber_k,
	                   double stop_position_huber_k,
	                   double relative_height_huber_k,
	                   double absolute_height_huber_k,
	                   const double* imu_gravity,
	                   const double* pr_linearization_ref_ecef,
	                   const double* pr_linearization_los_ecef,
	                   const double* doppler_linearization_ref_vel,
	                   const double* doppler_linearization_los_ecef,
	                   double stop_attitude_sigma_rad,
	                   double lm_damping) {
  if (n_epoch < 1 || n_sat < 4 || !sat_ecef || !pseudorange || !weights || !state_io) return -1;
  if (n_clock < 1 || n_clock > kMaxClockVD) return -1;

  const int base_ss = 7 + n_clock;  // x,y,z,vx,vy,vz,clk...,drift
  const int ss = state_stride > 0 ? state_stride : base_ss;
  if (ss != base_ss && ss != base_ss + 3 && ss != base_ss + 6 &&
      ss != base_ss + 9 && ss != base_ss + 12) return -1;
  const int pose_position_idx = (ss == base_ss + 12) ? base_ss : -1;
  const int attitude_idx = (ss == base_ss + 9) ? base_ss
                         : (ss == base_ss + 12) ? base_ss + 3
                                                : -1;
  const int accel_bias_idx = (ss == base_ss + 3 || ss == base_ss + 6) ? base_ss
                             : (ss == base_ss + 9) ? base_ss + 3
                             : (ss == base_ss + 12) ? base_ss + 6
                                                   : -1;
  const int gyro_bias_idx = (ss == base_ss + 6) ? base_ss + 3
                          : (ss == base_ss + 9) ? base_ss + 6
                          : (ss == base_ss + 12) ? base_ss + 9
                                                : -1;
  const double* imu_dt = imu_delta_t != nullptr ? imu_delta_t : dt;
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
  double* h_tied_work = (double*)std::malloc(sz_H);
  double* h_rhs = (double*)std::malloc(sz_state);
  double* h_tie_offset = (double*)std::malloc(sz_state);
  double* trial = (double*)std::malloc(sz_state);
  double* h_eff_w = (double*)std::malloc(sz_ws);
  if (!h_H || !h_g || !h_delta || !h_work || !h_tied_work || !h_rhs || !h_tie_offset ||
      !trial || !h_eff_w) {
    if (h_H) std::free(h_H);
    if (h_g) std::free(h_g);
    if (h_delta) std::free(h_delta);
    if (h_work) std::free(h_work);
    if (h_tied_work) std::free(h_tied_work);
    if (h_rhs) std::free(h_rhs);
    if (h_tie_offset) std::free(h_tie_offset);
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

	  double w_stop_attitude = 0.0;
	  if (attitude_idx >= 0 && stop_attitude_sigma_rad > 0.0) {
	    w_stop_attitude = 1.0 / (stop_attitude_sigma_rad * stop_attitude_sigma_rad);
	  }

  double w_imu_pos = 0.0;
  if (imu_delta_p != nullptr && imu_position_sigma_m > 0.0) {
    w_imu_pos = 1.0 / (imu_position_sigma_m * imu_position_sigma_m);
  }

  double w_imu_vel = 0.0;
  if (imu_delta_v != nullptr && imu_velocity_sigma_mps > 0.0) {
    w_imu_vel = 1.0 / (imu_velocity_sigma_mps * imu_velocity_sigma_mps);
  }

  double w_imu_att = 0.0;
  if (attitude_idx >= 0 && imu_delta_angle != nullptr && imu_attitude_sigma_rad > 0.0) {
    w_imu_att = 1.0 / (imu_attitude_sigma_rad * imu_attitude_sigma_rad);
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

  double w_imu_gyro_bias_prior = 0.0;
  if (gyro_bias_idx >= 0 && imu_gyro_bias_prior_sigma_radps > 0.0) {
    w_imu_gyro_bias_prior = 1.0 / (imu_gyro_bias_prior_sigma_radps * imu_gyro_bias_prior_sigma_radps);
  }

  double w_imu_gyro_bias_between = 0.0;
  if (gyro_bias_idx >= 0 && imu_gyro_bias_between_sigma_radps > 0.0) {
    w_imu_gyro_bias_between =
        1.0 / (imu_gyro_bias_between_sigma_radps * imu_gyro_bias_between_sigma_radps);
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
  const bool use_lm =
      std::isfinite(lm_damping) && lm_damping > 0.0;
  double lm_lambda = use_lm ? lm_damping : 0.0;
  constexpr double kLmFactor = 10.0;
  constexpr double kLmUpperBound = 1.0e5;
  constexpr double kLmLowerBound = 1.0e-12;
  constexpr int kLmMaxInnerAttempts = 16;

  constexpr int kMaxHardConstraintTerms = 3;
  std::vector<int> hard_term_count((size_t)n_state, 1);
  std::vector<int> hard_term_col((size_t)n_state * kMaxHardConstraintTerms, 0);
  std::vector<double> hard_term_coef((size_t)n_state * kMaxHardConstraintTerms, 0.0);
  std::vector<std::uint8_t> hard_representative((size_t)n_state, 1);
  const bool hard_tie_extra_clocks =
      clock_use_average_drift && n_clock > 1 && w_clkdrift > 0.0 && dt != nullptr;
  const bool hard_tie_pose_point = pose_position_idx >= 0 && attitude_idx >= 0;

  for (int it = 0; it < max_iter; it++) {
    effective_pr_weights_huber_host_vd(
        n_epoch, n_sat, n_clock, ss, sat_ecef, pr_linearization_ref_ecef,
        pr_linearization_los_ecef, pseudorange, weights, sys_host, state_io,
        huber_k, h_eff_w);
    std::memset(h_H, 0, sz_H);
    std::memset(h_g, 0, sz_state);
	    const char* debug_prefix_env =
	        (it == 0) ? std::getenv("GNSS_GPU_FGO_VD_DEBUG_LINEAR_SYSTEM_PREFIX") : nullptr;
	    const bool debug_linear_system =
	        debug_prefix_env && debug_prefix_env[0] != '\0';
	    const bool debug_components =
	        debug_linear_system &&
	        std::getenv("GNSS_GPU_FGO_VD_DEBUG_LINEAR_SYSTEM_COMPONENTS") != nullptr;
	    const std::string debug_prefix = debug_prefix_env ? std::string(debug_prefix_env) : std::string();
    auto dump_debug_component = [&](const char* stage) {
      if (!debug_components) return;
      const std::string suffix(stage);
      dump_dense_matrix_csv(debug_prefix + "_" + suffix + "_H.csv", h_H, n_state, n_state);
      dump_vector_csv(debug_prefix + "_" + suffix + "_g.csv", h_g, n_state);
    };
    add_pr_factor_host_vd(n_epoch, n_sat, n_clock, ss, n_state,
                          sat_ecef, pr_linearization_ref_ecef, pr_linearization_los_ecef,
                          pseudorange, h_eff_w, sys_host, state_io, h_H, h_g);
    dump_debug_component("01_pr");

    // Add host-side factors
    add_motion_factor_host(n_epoch, ss, n_state, w_motion, state_io, dt, h_H, h_g);
    dump_debug_component("02_motion");
    add_clock_drift_factor_host(n_epoch, n_clock, ss, n_state, w_clkdrift, state_io, dt,
                                clock_use_average_drift, h_H, h_g);
    dump_debug_component("03_clock");
    add_stop_velocity_factor_host(n_epoch, ss, n_state, w_stop_velocity, stop_velocity_huber_k, stop_mask, state_io,
                                  h_H, h_g);
    dump_debug_component("04_stop_velocity");
	    add_stop_pose_factor_host(n_epoch, ss, n_state, w_stop_position, w_stop_attitude, stop_position_huber_k,
	                              stop_mask, state_io, pose_position_idx, attitude_idx, h_H, h_g);
    dump_debug_component("05_stop_pose");
    add_pose_point_factor_host(n_epoch, ss, n_state, pose_position_idx, attitude_idx, state_io, h_H, h_g);
    dump_debug_component("06_pose_point");
    if (imu_preintegration_information) {
      add_imu_pva_factor_host(n_epoch, ss, n_state, imu_delta_p, imu_delta_v, imu_delta_angle,
                              imu_delta_p_bias_accel_jac, imu_delta_v_bias_accel_jac,
	                              imu_delta_p_bias_gyro_jac, imu_delta_v_bias_gyro_jac,
	                              imu_delta_angle_bias_gyro_jac,
	                              imu_preintegration_information, state_io, imu_dt, imu_gravity,
	                              pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx, imu_factor_use_next_bias,
	                              h_H, h_g);
	    } else {
	      add_imu_prior_factor_host(n_epoch, ss, n_state, w_imu_pos, w_imu_vel, imu_delta_p, imu_delta_v,
	                                imu_delta_angle,
	                                imu_delta_p_bias_accel_jac, imu_delta_v_bias_accel_jac,
	                                imu_delta_p_bias_gyro_jac, imu_delta_v_bias_gyro_jac,
	                                imu_delta_angle_bias_gyro_jac,
	                                imu_position_weights, imu_velocity_weights, state_io, imu_dt, imu_gravity,
	                                pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
	                                imu_factor_use_next_bias, h_H, h_g);
      add_imu_attitude_factor_host(n_epoch, ss, n_state, w_imu_att, imu_delta_angle, imu_attitude_weights,
                                   imu_delta_angle_bias_gyro_jac, state_io, imu_dt, attitude_idx, gyro_bias_idx,
                                   imu_factor_use_next_bias, h_H, h_g);
    }
    dump_debug_component("07_imu");
    add_bias_triplet_factor_host(n_epoch, ss, n_state, accel_bias_idx, w_imu_accel_bias_prior,
                                 w_imu_accel_bias_between, imu_accel_bias_between_weights, state_io, h_H, h_g);
    dump_debug_component("08_accel_bias");
    add_bias_triplet_factor_host(n_epoch, ss, n_state, gyro_bias_idx, w_imu_gyro_bias_prior,
                                 w_imu_gyro_bias_between, imu_gyro_bias_between_weights, state_io, h_H, h_g);
    dump_debug_component("09_gyro_bias");
    add_doppler_factor_host(n_epoch, n_sat, n_clock, ss, n_state,
                            sat_ecef, sat_vel, sat_clock_drift,
                            doppler_linearization_ref_vel, doppler_linearization_los_ecef,
                            doppler, doppler_weights, sys_host,
                            doppler_huber_k, state_io,
                            h_H, h_g);
    dump_debug_component("10_doppler");
    add_tdcp_factor_host_vd(n_epoch, n_sat, n_clock, ss, n_state, sat_ecef, tdcp_linearization_ref_ecef, sys_host, dt,
                            tdcp_meas, tdcp_weights, tdcp_sigma_m, tdcp_use_drift, tdcp_huber_k, state_io,
                            h_H, h_g);
    dump_debug_component("11_tdcp");
    add_relative_height_factor_host(n_epoch, ss, n_state, w_rel_height, relative_height_huber_k, rh_ux, rh_uy, rh_uz,
                                    rh_n_edges, rh_i_ptr, rh_j_ptr, state_io, h_H, h_g);
    dump_debug_component("12_relative_height");
    add_absolute_height_factor_host(n_epoch, ss, n_state, w_abs_height, absolute_height_huber_k, rh_ux, rh_uy, rh_uz,
                                    abs_height_ref_ptr, state_io, h_H, h_g);
    dump_debug_component("13_absolute_height");

	    if (it == 0) {
	      if (debug_linear_system) {
	        dump_dense_matrix_csv(debug_prefix + "_H.csv", h_H, n_state, n_state);
	        dump_vector_csv(debug_prefix + "_g.csv", h_g, n_state);
	        dump_vector_csv(debug_prefix + "_state.csv", state_io, n_state);
	      }
	    }

    auto total_cost = [&](const double* eval_state) {
      return pr_cost_host_vd(n_epoch, n_sat, n_clock, ss, sat_ecef, pr_linearization_ref_ecef,
                             pr_linearization_los_ecef, pseudorange, weights, sys_host, eval_state, huber_k)
          + motion_factor_cost_host(n_epoch, ss, w_motion, eval_state, dt)
          + clock_drift_cost_host(n_epoch, n_clock, ss, w_clkdrift, eval_state, dt, clock_use_average_drift)
          + stop_velocity_cost_host(n_epoch, ss, w_stop_velocity, stop_velocity_huber_k, stop_mask, eval_state)
          + stop_pose_cost_host(n_epoch, ss, w_stop_position, w_stop_attitude, stop_position_huber_k, stop_mask,
                                eval_state, pose_position_idx, attitude_idx)
          + pose_point_cost_host(n_epoch, ss, pose_position_idx, eval_state)
          + (imu_preintegration_information
                 ? imu_pva_cost_host(n_epoch, ss, imu_delta_p, imu_delta_v, imu_delta_angle,
                                     imu_delta_p_bias_accel_jac, imu_delta_v_bias_accel_jac,
                                     imu_delta_p_bias_gyro_jac, imu_delta_v_bias_gyro_jac,
                                     imu_delta_angle_bias_gyro_jac, imu_preintegration_information, eval_state,
                                     imu_dt, imu_gravity, pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                     imu_factor_use_next_bias)
                 : (imu_prior_cost_host(n_epoch, ss, w_imu_pos, w_imu_vel, imu_delta_p, imu_delta_v,
                                        imu_delta_angle, imu_delta_p_bias_accel_jac,
                                        imu_delta_v_bias_accel_jac, imu_delta_p_bias_gyro_jac,
                                        imu_delta_v_bias_gyro_jac, imu_delta_angle_bias_gyro_jac,
                                        imu_position_weights, imu_velocity_weights, eval_state, imu_dt,
                                        imu_gravity, pose_position_idx, attitude_idx, accel_bias_idx, gyro_bias_idx,
                                        imu_factor_use_next_bias)
                    + imu_attitude_cost_host(n_epoch, ss, w_imu_att, imu_delta_angle, imu_attitude_weights,
                                             imu_delta_angle_bias_gyro_jac, eval_state, imu_dt, attitude_idx,
                                             gyro_bias_idx, imu_factor_use_next_bias)))
          + bias_triplet_cost_host(n_epoch, ss, accel_bias_idx, w_imu_accel_bias_prior, w_imu_accel_bias_between,
                                   imu_accel_bias_between_weights, eval_state)
          + bias_triplet_cost_host(n_epoch, ss, gyro_bias_idx, w_imu_gyro_bias_prior, w_imu_gyro_bias_between,
                                   imu_gyro_bias_between_weights, eval_state)
          + doppler_cost_host(n_epoch, n_sat, n_clock, ss, sat_ecef, sat_vel, sat_clock_drift,
                              doppler_linearization_ref_vel, doppler_linearization_los_ecef, doppler,
                              doppler_weights, doppler_huber_k, eval_state)
          + tdcp_cost_host_vd(n_epoch, n_sat, n_clock, ss, sat_ecef, tdcp_linearization_ref_ecef, sys_host, dt,
                              tdcp_meas, tdcp_weights, tdcp_sigma_m, tdcp_use_drift, tdcp_huber_k, eval_state)
          + relative_height_cost_host(n_epoch, ss, w_rel_height, relative_height_huber_k, rh_ux, rh_uy, rh_uz,
                                      rh_n_edges, rh_i_ptr, rh_j_ptr, eval_state)
          + absolute_height_cost_host(n_epoch, ss, w_abs_height, absolute_height_huber_k, rh_ux, rh_uy, rh_uz,
                                      abs_height_ref_ptr, eval_state);
    };

    double cost_before = total_cost(state_io);

    // NOTE: Unlike the original fgo_gnss_lm which negates h_g, the VD solver
    // solves H * delta = g directly. All factors accumulate g = J^T * W * r
    // (the RHS of the Gauss-Newton normal equation), so the correct step is
    // delta = H^{-1} * g without negation.

    double step_norm = 0.0;
    auto solve_damped_step = [&](double lambda) {
      std::memcpy(h_work, h_H, sz_H);
      // Diagonal regularization floor for local VD-only states.  Keep the same
      // small positive floor as position/clock; using a weaker floor here makes
      // the extended IMU attitude/bias system intermittently fail Cholesky on
      // otherwise identical first-iteration systems.
      const double diag_jitter = use_lm ? 0.0 : kDiagJitter;
      const double kVelDriftJitter = diag_jitter;
      for (int t2 = 0; t2 < n_epoch; t2++) {
        int off = ss * t2;
        for (int d2 = 0; d2 < ss; d2++) {
          double jit = diag_jitter;
          if (d2 >= 3 && d2 <= 5) jit = kVelDriftJitter;  // velocity
          if (d2 == 6 + n_clock) jit = kVelDriftJitter;    // drift
          if (attitude_idx >= 0 && d2 >= attitude_idx && d2 < attitude_idx + 3) {
            jit = kVelDriftJitter;  // attitude error state
          }
          if (accel_bias_idx >= 0 && d2 >= accel_bias_idx && d2 < accel_bias_idx + 3) {
            jit = kVelDriftJitter;  // accelerometer bias
          }
          if (gyro_bias_idx >= 0 && d2 >= gyro_bias_idx && d2 < gyro_bias_idx + 3) {
            jit = kVelDriftJitter;  // gyroscope bias
          }
          h_work[(size_t)(off + d2) * n_state + (off + d2)] += jit + lambda;
        }
      }

      double* solve_matrix = h_work;
      const double* solve_rhs = h_g;
      bool hard_constraints_active = false;
      if (hard_tie_extra_clocks || hard_tie_pose_point) {
        for (int i = 0; i < n_state; i++) {
          hard_term_count[(size_t)i] = 1;
          hard_term_col[(size_t)i * kMaxHardConstraintTerms] = i;
          hard_term_coef[(size_t)i * kMaxHardConstraintTerms] = 1.0;
          for (int k = 1; k < kMaxHardConstraintTerms; k++) {
            hard_term_col[(size_t)i * kMaxHardConstraintTerms + k] = 0;
            hard_term_coef[(size_t)i * kMaxHardConstraintTerms + k] = 0.0;
          }
          hard_representative[(size_t)i] = 1;
          h_tie_offset[i] = 0.0;
        }

        if (hard_tie_extra_clocks) {
          for (int k = 1; k < n_clock; k++) {
            int root = 6 + k;
            for (int t = 0; t < n_epoch; t++) {
              if (t > 0) {
                const double dt_prev = dt[t - 1];
                if (!std::isfinite(dt_prev) || dt_prev <= 0.0) {
                  root = ss * t + 6 + k;
                }
              }
              const int idx = ss * t + 6 + k;
              if (idx == root) continue;
              hard_term_count[(size_t)idx] = 1;
              hard_term_col[(size_t)idx * kMaxHardConstraintTerms] = root;
              hard_term_coef[(size_t)idx * kMaxHardConstraintTerms] = 1.0;
              h_tie_offset[idx] = state_io[root] - state_io[idx];
              hard_representative[(size_t)idx] = 0;
              hard_constraints_active = true;
            }
          }
        }

        if (hard_tie_pose_point) {
          for (int t = 0; t < n_epoch; t++) {
            const int o = ss * t;
            const double* att = state_io + o + attitude_idx;
            if (!finite3(att)) continue;
            double R[9];
            rotvec_to_rotm_host(att, R);
            const double err[3] = {
                state_io[o + pose_position_idx + 0] - state_io[o + 0],
                state_io[o + pose_position_idx + 1] - state_io[o + 1],
                state_io[o + pose_position_idx + 2] - state_io[o + 2],
            };
            for (int axis = 0; axis < 3; axis++) {
              const int idx = o + pose_position_idx + axis;
              hard_term_count[(size_t)idx] = 3;
              for (int row = 0; row < 3; row++) {
                hard_term_col[(size_t)idx * kMaxHardConstraintTerms + row] = o + row;
                hard_term_coef[(size_t)idx * kMaxHardConstraintTerms + row] =
                    R[row * 3 + axis];
              }
              h_tie_offset[idx] =
                  -(R[0 * 3 + axis] * err[0] + R[1 * 3 + axis] * err[1] +
                    R[2 * 3 + axis] * err[2]);
              hard_representative[(size_t)idx] = 0;
              hard_constraints_active = true;
            }
          }
        }
      }

      if (hard_constraints_active) {
        std::memset(h_tied_work, 0, sz_H);
        std::memset(h_rhs, 0, sz_state);

        for (int i = 0; i < n_state; i++) {
          double hq = 0.0;
          const size_t row = (size_t)i * n_state;
          for (int j = 0; j < n_state; j++) {
            hq += h_work[row + j] * h_tie_offset[j];
          }
          const double rhs_i = h_g[i] - hq;
          const int ni = hard_term_count[(size_t)i];
          for (int ai = 0; ai < ni; ai++) {
            const int ri = hard_term_col[(size_t)i * kMaxHardConstraintTerms + ai];
            const double ci = hard_term_coef[(size_t)i * kMaxHardConstraintTerms + ai];
            h_rhs[ri] += ci * rhs_i;
          }
          for (int j = 0; j < n_state; j++) {
            const double hij = h_work[row + j];
            if (hij == 0.0) continue;
            const int nj = hard_term_count[(size_t)j];
            for (int ai = 0; ai < ni; ai++) {
              const int ri = hard_term_col[(size_t)i * kMaxHardConstraintTerms + ai];
              const double ci = hard_term_coef[(size_t)i * kMaxHardConstraintTerms + ai];
              for (int aj = 0; aj < nj; aj++) {
                const int rj = hard_term_col[(size_t)j * kMaxHardConstraintTerms + aj];
                const double cj = hard_term_coef[(size_t)j * kMaxHardConstraintTerms + aj];
                h_tied_work[(size_t)ri * n_state + rj] += ci * cj * hij;
              }
            }
          }
        }
        for (int i = 0; i < n_state; i++) {
          if (hard_representative[(size_t)i]) continue;
          for (int j = 0; j < n_state; j++) {
            h_tied_work[(size_t)i * n_state + j] = 0.0;
            h_tied_work[(size_t)j * n_state + i] = 0.0;
          }
          h_tied_work[(size_t)i * n_state + i] = 1.0;
          h_rhs[i] = 0.0;
        }
        solve_matrix = h_tied_work;
        solve_rhs = h_rhs;
      }

      if (!cholesky_decompose_inplace(n_state, solve_matrix)) {
        return false;
      }
      cholesky_solve_lower(n_state, solve_matrix, solve_rhs, h_delta);
      if (hard_constraints_active) {
        for (int i = 0; i < n_state; i++) {
          if (hard_representative[(size_t)i]) continue;
          double value = h_tie_offset[i];
          const int ni = hard_term_count[(size_t)i];
          for (int ai = 0; ai < ni; ai++) {
            const int col = hard_term_col[(size_t)i * kMaxHardConstraintTerms + ai];
            const double coef = hard_term_coef[(size_t)i * kMaxHardConstraintTerms + ai];
            value += coef * h_delta[col];
          }
          h_delta[i] = value;
        }
      }
      return true;
    };

	    bool accepted = false;
	    double accepted_lambda = 0.0;
	    double accepted_alpha = 1.0;
	    if (use_lm && !enable_line_search) {
	      if (!solve_damped_step(lm_lambda)) {
	        break;
	      }
      for (int i = 0; i < n_state; i++) step_norm += h_delta[i] * h_delta[i];
      step_norm = sqrt(step_norm);
	      apply_vd_state_delta_host(n_epoch, ss, pose_position_idx, attitude_idx, state_io, h_delta, 1.0, trial);
	      std::memcpy(state_io, trial, sz_state);
	      accepted_lambda = lm_lambda;
	      accepted = true;
	    } else if (use_lm) {
	      double lambda_try = lm_lambda;
      for (int lm_attempt = 0;
           lm_attempt < kLmMaxInnerAttempts && lambda_try <= kLmUpperBound;
           lm_attempt++) {
        if (!solve_damped_step(lambda_try)) {
          lambda_try *= kLmFactor;
          continue;
        }
        double trial_step_norm = 0.0;
        for (int i = 0; i < n_state; i++) trial_step_norm += h_delta[i] * h_delta[i];
        trial_step_norm = sqrt(trial_step_norm);
        apply_vd_state_delta_host(n_epoch, ss, pose_position_idx, attitude_idx, state_io, h_delta, 1.0, trial);
        double ctry = total_cost(trial);
	        if (ctry <= cost_before * (1.0 + 1e-12)) {
	          std::memcpy(state_io, trial, sz_state);
	          step_norm = trial_step_norm;
	          accepted_lambda = lambda_try;
	          lm_lambda = lambda_try / kLmFactor;
	          if (lm_lambda < kLmLowerBound) lm_lambda = kLmLowerBound;
	          accepted = true;
          break;
        }
        lambda_try *= kLmFactor;
      }
      if (!accepted) lm_lambda = lambda_try;
    } else if (!solve_damped_step(0.0)) {
      break;
    } else {
      for (int i = 0; i < n_state; i++) step_norm += h_delta[i] * h_delta[i];
      step_norm = sqrt(step_norm);
      if (!enable_line_search) {
      apply_vd_state_delta_host(n_epoch, ss, pose_position_idx, attitude_idx, state_io, h_delta, 1.0, trial);
      std::memcpy(state_io, trial, sz_state);
      accepted = true;
      } else {
      double alpha = 1.0;
      for (int ls = 0; ls < 12; ls++) {
        apply_vd_state_delta_host(n_epoch, ss, pose_position_idx, attitude_idx, state_io, h_delta, alpha, trial);
        double ctry = total_cost(trial);
	        if (ctry <= cost_before * (1.0 + 1e-12)) {
	          std::memcpy(state_io, trial, sz_state);
	          accepted_alpha = alpha;
	          accepted = true;
	          break;
	        }
        alpha *= 0.5;
      }
      }
	    }

	    if (debug_linear_system && accepted) {
	      const double cost_after = total_cost(state_io);
	      const double step_meta[] = {
	          1.0, accepted_lambda, accepted_alpha, step_norm, cost_before, cost_after};
	      dump_vector_csv(debug_prefix + "_accepted_delta.csv", h_delta, n_state);
	      dump_vector_csv(debug_prefix + "_accepted_state.csv", state_io, n_state);
	      dump_vector_csv(debug_prefix + "_accepted_meta.csv", step_meta, 6);
	    }

	    CUDA_CHECK(cudaMemcpy(d_state, state_io, sz_state, cudaMemcpyHostToDevice));

    total_iters++;
    ok = true;
    if (accepted && step_norm < tol) break;
    if (!accepted) break;
  }

  if (out_mse_pr)
    *out_mse_pr = compute_pr_mse_host_vd(n_epoch, n_sat, n_clock, ss, sat_ecef, pr_linearization_ref_ecef,
                                         pr_linearization_los_ecef, pseudorange, weights, sys_host, state_io);

  std::free(h_H);
  std::free(h_g);
  std::free(h_delta);
  std::free(h_work);
  std::free(h_tied_work);
  std::free(h_rhs);
  std::free(h_tie_offset);
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
