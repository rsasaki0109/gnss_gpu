#include "gnss_gpu/raim.h"
#include "gnss_gpu/positioning.h"
#include "gnss_gpu/coordinates.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace gnss_gpu {

static void sagnac_los(const double* sat_ecef, int s,
                       double x, double y, double z,
                       double* dx, double* dy, double* dz, double* r) {
  double sx = sat_ecef[s * 3 + 0];
  double sy = sat_ecef[s * 3 + 1];
  double sz = sat_ecef[s * 3 + 2];

  double dx0 = x - sx;
  double dy0 = y - sy;
  double dz0 = z - sz;
  double range_approx = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
  double transit_time = range_approx / 299792458.0;
  double theta = 7.2921151467e-5 * transit_time;
  double sx_rot = sx * cos(theta) + sy * sin(theta);
  double sy_rot = -sx * sin(theta) + sy * cos(theta);

  *dx = x - sx_rot;
  *dy = y - sy_rot;
  *dz = z - sz;
  *r = sqrt((*dx) * (*dx) + (*dy) * (*dy) + (*dz) * (*dz));
  if (*r < 1e-12) *r = 1e-12;
}

// Chi-squared critical values for degrees of freedom 1..20 at common p_fa levels.
// We use a simple lookup + interpolation for p_fa = {1e-3, 1e-5, 1e-7}.
// For other p_fa values, use Wilson-Hilferty approximation.
static double chi2_threshold(int dof, double p_fa) {
  if (dof <= 0) return 0.0;

  // Wilson-Hilferty approximation for chi-squared inverse CDF
  // Q(p) ~ dof * (1 - 2/(9*dof) + z_p * sqrt(2/(9*dof)))^3
  // where z_p is the standard normal quantile for (1-p_fa)

  // Approximate z_p from p_fa using rational approximation (Abramowitz & Stegun 26.2.23)
  double p = 1.0 - p_fa;
  double t;
  if (p_fa > 0.5) {
    t = sqrt(-2.0 * log(1.0 - p));
  } else {
    t = sqrt(-2.0 * log(p_fa));
  }
  double z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
                 (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
  if (p_fa > 0.5) z = -z;

  // Wilson-Hilferty
  double d = (double)dof;
  double a = 1.0 - 2.0 / (9.0 * d) + z * sqrt(2.0 / (9.0 * d));
  double chi2 = d * a * a * a;
  if (chi2 < 0.0) chi2 = 0.0;
  return chi2;
}

// Compute weighted SSE (sum of squared errors) given a position solution
static double compute_sse(const double* sat_ecef, const double* pseudoranges,
                          const double* weights, const double* position,
                          int n_sat) {
  double x = position[0], y = position[1], z = position[2], cb = position[3];
  double sse = 0.0;
  for (int s = 0; s < n_sat; s++) {
    double dx, dy, dz, r;
    sagnac_los(sat_ecef, s, x, y, z, &dx, &dy, &dz, &r);
    double residual = pseudoranges[s] - (r + cb);
    sse += weights[s] * residual * residual;
  }
  return sse;
}

// Compute SSE excluding one satellite
static double compute_sse_excluding(const double* sat_ecef, const double* pseudoranges,
                                    const double* weights, const double* position,
                                    int n_sat, int exclude) {
  double x = position[0], y = position[1], z = position[2], cb = position[3];
  double sse = 0.0;
  for (int s = 0; s < n_sat; s++) {
    if (s == exclude) continue;
    double dx, dy, dz, r;
    sagnac_los(sat_ecef, s, x, y, z, &dx, &dy, &dz, &r);
    double residual = pseudoranges[s] - (r + cb);
    sse += weights[s] * residual * residual;
  }
  return sse;
}

// Invert a 4x4 matrix (for covariance computation). Returns false if singular.
static bool invert_4x4(const double* M, double* Minv) {
  // Augmented matrix [M | I]
  double A[4][8];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      A[i][j] = M[i * 4 + j];
      A[i][j + 4] = (i == j) ? 1.0 : 0.0;
    }
  }

  for (int col = 0; col < 4; col++) {
    int max_row = col;
    for (int row = col + 1; row < 4; row++) {
      if (fabs(A[row][col]) > fabs(A[max_row][col])) max_row = row;
    }
    if (max_row != col) {
      for (int k = 0; k < 8; k++) {
        double tmp = A[col][k]; A[col][k] = A[max_row][k]; A[max_row][k] = tmp;
      }
    }
    if (fabs(A[col][col]) < 1e-15) return false;
    double pivot = A[col][col];
    for (int k = 0; k < 8; k++) A[col][k] /= pivot;
    for (int row = 0; row < 4; row++) {
      if (row == col) continue;
      double factor = A[row][col];
      for (int k = 0; k < 8; k++) A[row][k] -= factor * A[col][k];
    }
  }

  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      Minv[i * 4 + j] = A[i][j + 4];

  return true;
}

// Compute HPL and VPL from the geometry matrix and position.
// Uses the slope matrix approach: S = (H^T W H)^{-1} H^T W
// Then transforms to local ENU frame for horizontal/vertical separation.
static void compute_protection_levels(const double* sat_ecef, const double* weights,
                                      const double* position, int n_sat,
                                      double p_fa, double* hpl, double* vpl) {
  double x = position[0], y = position[1], z = position[2];

  // Build H^T W H
  double HTWH[16] = {};
  for (int s = 0; s < n_sat; s++) {
    double dx, dy, dz, r;
    sagnac_los(sat_ecef, s, x, y, z, &dx, &dy, &dz, &r);
    double H[4] = {dx / r, dy / r, dz / r, 1.0};
    double w = weights[s];
    for (int a = 0; a < 4; a++)
      for (int b = 0; b < 4; b++)
        HTWH[a * 4 + b] += H[a] * w * H[b];
  }

  // Invert to get covariance in ECEF
  double cov[16];
  if (!invert_4x4(HTWH, cov)) {
    *hpl = 1e9;
    *vpl = 1e9;
    return;
  }

  // Rotation matrix from ECEF to ENU at current position
  double r_pos = sqrt(x * x + y * y + z * z);
  double lat = asin(z / r_pos);
  double lon = atan2(y, x);
  double slat = sin(lat), clat = cos(lat);
  double slon = sin(lon), clon = cos(lon);

  // R: 3x3 rotation ECEF->ENU
  // East:  [-slon, clon, 0]
  // North: [-slat*clon, -slat*slon, clat]
  // Up:    [clat*clon, clat*slon, slat]
  double R[3][3] = {
    {-slon,        clon,         0.0},
    {-slat * clon, -slat * slon, clat},
    { clat * clon,  clat * slon, slat}
  };

  // Transform position covariance (3x3 upper-left of cov) to ENU
  // cov_enu = R * cov_xyz * R^T
  double cov_xyz[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      cov_xyz[i][j] = cov[i * 4 + j];

  double tmp[3][3] = {};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        tmp[i][j] += R[i][k] * cov_xyz[k][j];

  double cov_enu[3][3] = {};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++)
        cov_enu[i][j] += tmp[i][k] * R[j][k];

  // K-factor from p_fa (normal distribution quantile for one-sided)
  // For RAIM, typical K_H ~ 6.0 (p_fa=1e-9) or use chi2-derived value
  double t_val = sqrt(-2.0 * log(p_fa));
  double k_factor = t_val - (2.515517 + 0.802853 * t_val + 0.010328 * t_val * t_val) /
                            (1.0 + 1.432788 * t_val + 0.189269 * t_val * t_val +
                             0.001308 * t_val * t_val * t_val);

  // HPL = k * sqrt(var_east + var_north), VPL = k * sqrt(var_up)
  double var_e = fabs(cov_enu[0][0]);
  double var_n = fabs(cov_enu[1][1]);
  double var_u = fabs(cov_enu[2][2]);

  *hpl = k_factor * sqrt(var_e + var_n);
  *vpl = k_factor * sqrt(var_u);
}

void raim_check(const double* sat_ecef, const double* pseudoranges,
                const double* weights, const double* position,
                RAIMResult* result, int n_sat, double p_fa) {
  result->excluded_sat = -1;

  // Need redundancy: n_sat > 4 (degrees of freedom > 0)
  int dof = n_sat - 4;
  if (dof <= 0) {
    result->integrity_ok = (n_sat >= 4);
    result->test_statistic = 0.0;
    result->threshold = 0.0;
    result->hpl = 1e9;
    result->vpl = 1e9;
    return;
  }

  double sse = compute_sse(sat_ecef, pseudoranges, weights, position, n_sat);
  double threshold = chi2_threshold(dof, p_fa);

  result->test_statistic = sse;
  result->threshold = threshold;
  result->integrity_ok = (sse <= threshold);

  compute_protection_levels(sat_ecef, weights, position, n_sat, p_fa,
                            &result->hpl, &result->vpl);
}

void raim_fde(const double* sat_ecef, const double* pseudoranges,
              const double* weights, double* position,
              RAIMResult* result, int n_sat, double p_fa) {
  // First run the standard RAIM check
  raim_check(sat_ecef, pseudoranges, weights, position, result, n_sat, p_fa);

  if (result->integrity_ok) return;  // No fault detected

  // Need at least 6 satellites for FDE (5 after exclusion, giving 1 dof for re-check)
  if (n_sat < 6) return;

  // Try excluding each satellite
  double best_sse = 1e30;
  int best_exclude = -1;
  double best_pos[4];

  // Buffers for subset data
  int n_sub = n_sat - 1;
  double sub_sat[64 * 3];  // max 64 satellites
  double sub_pr[64];
  double sub_w[64];

  for (int ex = 0; ex < n_sat; ex++) {
    // Build subset excluding satellite ex
    int idx = 0;
    for (int s = 0; s < n_sat; s++) {
      if (s == ex) continue;
      sub_sat[idx * 3 + 0] = sat_ecef[s * 3 + 0];
      sub_sat[idx * 3 + 1] = sat_ecef[s * 3 + 1];
      sub_sat[idx * 3 + 2] = sat_ecef[s * 3 + 2];
      sub_pr[idx] = pseudoranges[s];
      sub_w[idx] = weights[s];
      idx++;
    }

    // Re-solve WLS with subset
    double sub_pos[4];
    int iters = wls_position(sub_sat, sub_pr, sub_w, sub_pos, n_sub);
    if (iters < 0) continue;  // WLS failed

    double sse = compute_sse(sub_sat, sub_pr, sub_w, sub_pos, n_sub);
    if (sse < best_sse) {
      best_sse = sse;
      best_exclude = ex;
      memcpy(best_pos, sub_pos, 4 * sizeof(double));
    }
  }

  if (best_exclude < 0) return;  // No valid exclusion found

  // Check if the best exclusion passes the chi-squared test
  int dof_sub = n_sub - 4;
  double threshold_sub = chi2_threshold(dof_sub, p_fa);

  if (best_sse <= threshold_sub) {
    // Exclusion succeeded
    result->integrity_ok = true;
    result->excluded_sat = best_exclude;
    result->test_statistic = best_sse;
    result->threshold = threshold_sub;
    memcpy(position, best_pos, 4 * sizeof(double));

    // Recompute protection levels with subset geometry
    int idx = 0;
    for (int s = 0; s < n_sat; s++) {
      if (s == best_exclude) continue;
      sub_sat[idx * 3 + 0] = sat_ecef[s * 3 + 0];
      sub_sat[idx * 3 + 1] = sat_ecef[s * 3 + 1];
      sub_sat[idx * 3 + 2] = sat_ecef[s * 3 + 2];
      sub_w[idx] = weights[s];
      idx++;
    }
    compute_protection_levels(sub_sat, sub_w, position, n_sub, p_fa,
                              &result->hpl, &result->vpl);
  } else {
    // Even best exclusion doesn't pass; report the best we found
    result->integrity_ok = false;
    result->excluded_sat = best_exclude;
    result->test_statistic = best_sse;
    result->threshold = threshold_sub;
  }
}

}  // namespace gnss_gpu
