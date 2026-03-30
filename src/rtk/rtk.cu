#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/rtk.h"
#include "gnss_gpu/coordinates.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <algorithm>

namespace gnss_gpu {

// Maximum satellites supported (compile-time bound for stack arrays)
static constexpr int MAX_SAT = 64;
static constexpr int MAX_DD = MAX_SAT - 1;  // max double-difference observations
static constexpr int MAX_STATE = 3 + MAX_DD; // 3 position + DD ambiguities

// GPU kernel uses smaller limits to reduce per-thread stack usage.
static constexpr int GPU_MAX_SAT = 16;
static constexpr int GPU_MAX_DD = GPU_MAX_SAT - 1;
static constexpr int GPU_MAX_STATE = 3 + GPU_MAX_DD;

// --------------------------------------------------------------------------
// Helper: compute geometric range
// --------------------------------------------------------------------------
__host__ __device__ static inline double
geo_range(double rx, double ry, double rz,
          double sx, double sy, double sz) {
  double dx = rx - sx, dy = ry - sy, dz = rz - sz;
  return sqrt(dx * dx + dy * dy + dz * dz);
}

// --------------------------------------------------------------------------
// Form double-difference observations
// ref_idx: index of reference satellite (highest elevation)
// dd_pr[n_dd], dd_cp[n_dd]: output DD pseudorange / carrier-phase
// dd_map[n_dd]: maps DD index -> satellite index (non-ref)
// n_dd = n_sat - 1
// --------------------------------------------------------------------------
__host__ __device__ static void
form_double_difference(const double* rover_pr, const double* base_pr,
                       const double* rover_cp, const double* base_cp,
                       int ref_idx, int n_sat,
                       double* dd_pr, double* dd_cp, int* dd_map, int& n_dd) {
  // Single-difference: SD = rover - base
  // Double-difference: DD_ij = SD_j - SD_ref
  double sd_pr_ref = rover_pr[ref_idx] - base_pr[ref_idx];
  double sd_cp_ref = rover_cp[ref_idx] - base_cp[ref_idx];

  n_dd = 0;
  for (int s = 0; s < n_sat; s++) {
    if (s == ref_idx) continue;
    double sd_pr_s = rover_pr[s] - base_pr[s];
    double sd_cp_s = rover_cp[s] - base_cp[s];
    dd_pr[n_dd] = sd_pr_s - sd_pr_ref;
    dd_cp[n_dd] = sd_cp_s - sd_cp_ref;
    dd_map[n_dd] = s;
    n_dd++;
  }
}

// --------------------------------------------------------------------------
// Select reference satellite: highest elevation seen from approximate position
// --------------------------------------------------------------------------
__host__ __device__ static int
select_ref_satellite(double rx, double ry, double rz,
                     const double* sat_ecef, int n_sat) {
  int best = 0;
  double best_el = -1e30;
  // Approximate "up" direction = normalized receiver position (works for Earth surface)
  double rn = sqrt(rx * rx + ry * ry + rz * rz);
  if (rn < 1.0) rn = 1.0;
  double ux = rx / rn, uy = ry / rn, uz = rz / rn;

  for (int s = 0; s < n_sat; s++) {
    double dx = sat_ecef[s * 3 + 0] - rx;
    double dy = sat_ecef[s * 3 + 1] - ry;
    double dz = sat_ecef[s * 3 + 2] - rz;
    double d = sqrt(dx * dx + dy * dy + dz * dz);
    if (d < 1.0) continue;
    double sin_el = (dx * ux + dy * uy + dz * uz) / d;
    if (sin_el > best_el) {
      best_el = sin_el;
      best = s;
    }
  }
  return best;
}

// --------------------------------------------------------------------------
// Gauss-Newton solver for N x N augmented system [A | b]
// A is n_state x (n_state+1), delta is n_state output
// Uses Gaussian elimination with partial pivoting.
// --------------------------------------------------------------------------
__host__ __device__ static void
solve_augmented(double* A, double* delta, int n) {
  // A is stored row-major as n x (n+1)
  int ncol = n + 1;
  for (int col = 0; col < n; col++) {
    // Partial pivoting
    int max_row = col;
    double max_val = fabs(A[col * ncol + col]);
    for (int row = col + 1; row < n; row++) {
      double v = fabs(A[row * ncol + col]);
      if (v > max_val) { max_val = v; max_row = row; }
    }
    if (max_row != col) {
      for (int k = 0; k < ncol; k++) {
        double tmp = A[col * ncol + k];
        A[col * ncol + k] = A[max_row * ncol + k];
        A[max_row * ncol + k] = tmp;
      }
    }
    if (fabs(A[col * ncol + col]) < 1e-15) continue;
    for (int row = col + 1; row < n; row++) {
      double factor = A[row * ncol + col] / A[col * ncol + col];
      for (int k = col; k < ncol; k++)
        A[row * ncol + k] -= factor * A[col * ncol + k];
    }
  }
  // Back substitution
  for (int row = n - 1; row >= 0; row--) {
    delta[row] = A[row * ncol + n];
    for (int col = row + 1; col < n; col++)
      delta[row] -= A[row * ncol + col] * delta[col];
    if (fabs(A[row * ncol + row]) > 1e-15)
      delta[row] /= A[row * ncol + row];
    else
      delta[row] = 0.0;
  }
}

// --------------------------------------------------------------------------
// CPU single-epoch RTK float solution
// --------------------------------------------------------------------------
int rtk_float(const double* base_ecef,
              const double* rover_pr, const double* base_pr,
              const double* rover_carrier, const double* base_carrier,
              const double* sat_ecef, double* result,
              double* ambiguities, double* residuals,
              int n_sat, double wavelength, int max_iter, double tol) {
  if (n_sat < 4) return 0;

  int n_dd = n_sat - 1;
  int n_state = 3 + n_dd;  // [dx, dy, dz, N1, ..., N_{n_dd}]

  // Initial rover position = base position (short baseline assumption)
  double x = base_ecef[0], y = base_ecef[1], z = base_ecef[2];

  // Select reference satellite (highest elevation from base)
  int ref_idx = select_ref_satellite(base_ecef[0], base_ecef[1], base_ecef[2],
                                     sat_ecef, n_sat);

  // Form double-difference observations
  double dd_pr[MAX_DD], dd_cp[MAX_DD];
  int dd_map[MAX_DD];
  int actual_dd;
  form_double_difference(rover_pr, base_pr, rover_carrier, base_carrier,
                         ref_idx, n_sat, dd_pr, dd_cp, dd_map, actual_dd);

  // Initialize float ambiguities from DD pseudorange and carrier phase
  // DD_carrier * wavelength ~ DD_range + N * wavelength
  // => N ~ DD_carrier - DD_range / wavelength
  // Approximate: N ~ DD_carrier - DD_pseudorange / wavelength
  double N[MAX_DD];
  for (int i = 0; i < actual_dd; i++) {
    N[i] = dd_cp[i] - dd_pr[i] / wavelength;
  }

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    // Build normal equation: H^T H delta = H^T dy
    // Total observations: 2 * n_dd (DD pseudorange + DD carrier)
    // State: [dx, dy, dz, N_0, ..., N_{n_dd-1}]

    // Allocate H^T H as n_state x n_state and H^T dy as n_state
    double HTH[MAX_STATE * MAX_STATE] = {};
    double HTdy[MAX_STATE] = {};

    // Precompute geometric ranges
    double r_base_ref = geo_range(base_ecef[0], base_ecef[1], base_ecef[2],
                                  sat_ecef[ref_idx * 3 + 0],
                                  sat_ecef[ref_idx * 3 + 1],
                                  sat_ecef[ref_idx * 3 + 2]);
    double r_rover_ref = geo_range(x, y, z,
                                   sat_ecef[ref_idx * 3 + 0],
                                   sat_ecef[ref_idx * 3 + 1],
                                   sat_ecef[ref_idx * 3 + 2]);

    for (int i = 0; i < actual_dd; i++) {
      int s = dd_map[i];
      double r_base_s = geo_range(base_ecef[0], base_ecef[1], base_ecef[2],
                                  sat_ecef[s * 3 + 0],
                                  sat_ecef[s * 3 + 1],
                                  sat_ecef[s * 3 + 2]);
      double r_rover_s = geo_range(x, y, z,
                                   sat_ecef[s * 3 + 0],
                                   sat_ecef[s * 3 + 1],
                                   sat_ecef[s * 3 + 2]);

      // DD geometric range = (r_rover_s - r_base_s) - (r_rover_ref - r_base_ref)
      double dd_geo = (r_rover_s - r_base_s) - (r_rover_ref - r_base_ref);

      // Direction cosines for satellite s from rover
      double dx_s = (x - sat_ecef[s * 3 + 0]) / r_rover_s;
      double dy_s = (x - sat_ecef[s * 3 + 1]) / r_rover_s;  // intentionally using direction to rover
      double dz_s = (x - sat_ecef[s * 3 + 2]) / r_rover_s;
      // Fix: proper direction cosines
      dx_s = (x - sat_ecef[s * 3 + 0]) / r_rover_s;
      dy_s = (y - sat_ecef[s * 3 + 1]) / r_rover_s;
      dz_s = (z - sat_ecef[s * 3 + 2]) / r_rover_s;

      // Direction cosines for reference satellite from rover
      double dx_r = (x - sat_ecef[ref_idx * 3 + 0]) / r_rover_ref;
      double dy_r = (y - sat_ecef[ref_idx * 3 + 1]) / r_rover_ref;
      double dz_r = (z - sat_ecef[ref_idx * 3 + 2]) / r_rover_ref;

      // DD Jacobian for position: d(DD_geo)/d(pos) = [e_s - e_ref]
      double H_pos[3] = {
        dx_s - dx_r,
        dy_s - dy_r,
        dz_s - dz_r
      };

      // --- DD pseudorange observation ---
      // Predicted: dd_geo
      // Observed: dd_pr[i]
      double res_pr = dd_pr[i] - dd_geo;

      // Pseudorange weight (lower than carrier)
      double w_pr = 1.0;

      // Jacobian row for pseudorange: [H_pos, 0...0]
      for (int a = 0; a < 3; a++) {
        HTdy[a] += H_pos[a] * w_pr * res_pr;
        for (int b = 0; b < 3; b++)
          HTH[a * n_state + b] += H_pos[a] * w_pr * H_pos[b];
      }

      // --- DD carrier phase observation ---
      // Predicted: dd_geo / wavelength + N[i]
      // Observed: dd_cp[i]
      double res_cp = dd_cp[i] - (dd_geo / wavelength + N[i]);

      // Carrier phase weight (higher precision)
      double w_cp = 10000.0;  // ~100x better than pseudorange

      // Jacobian row for carrier: [H_pos / wavelength, 0...1...0]
      double H_cp_pos[3] = {
        H_pos[0] / wavelength,
        H_pos[1] / wavelength,
        H_pos[2] / wavelength
      };

      for (int a = 0; a < 3; a++) {
        HTdy[a] += H_cp_pos[a] * w_cp * res_cp;
        for (int b = 0; b < 3; b++)
          HTH[a * n_state + b] += H_cp_pos[a] * w_cp * H_cp_pos[b];
        // Cross term: position x ambiguity
        HTH[a * n_state + (3 + i)] += H_cp_pos[a] * w_cp * 1.0;
        HTH[(3 + i) * n_state + a] += 1.0 * w_cp * H_cp_pos[a];
      }

      // Ambiguity diagonal
      HTH[(3 + i) * n_state + (3 + i)] += w_cp * 1.0;
      HTdy[3 + i] += w_cp * res_cp;
    }

    // Solve normal equations
    double aug[MAX_STATE * (MAX_STATE + 1)];
    for (int a = 0; a < n_state; a++) {
      for (int b = 0; b < n_state; b++)
        aug[a * (n_state + 1) + b] = HTH[a * n_state + b];
      aug[a * (n_state + 1) + n_state] = HTdy[a];
    }

    double delta[MAX_STATE];
    solve_augmented(aug, delta, n_state);

    x += delta[0]; y += delta[1]; z += delta[2];
    for (int i = 0; i < actual_dd; i++)
      N[i] += delta[3 + i];

    double norm = sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                       delta[2] * delta[2]);
    if (norm < tol) { iter++; break; }
  }

  result[0] = x; result[1] = y; result[2] = z;
  for (int i = 0; i < actual_dd; i++)
    ambiguities[i] = N[i];

  // Compute final residuals
  if (residuals) {
    double r_base_ref = geo_range(base_ecef[0], base_ecef[1], base_ecef[2],
                                  sat_ecef[ref_idx * 3 + 0],
                                  sat_ecef[ref_idx * 3 + 1],
                                  sat_ecef[ref_idx * 3 + 2]);
    double r_rover_ref = geo_range(x, y, z,
                                   sat_ecef[ref_idx * 3 + 0],
                                   sat_ecef[ref_idx * 3 + 1],
                                   sat_ecef[ref_idx * 3 + 2]);
    for (int i = 0; i < actual_dd; i++) {
      int s = dd_map[i];
      double r_base_s = geo_range(base_ecef[0], base_ecef[1], base_ecef[2],
                                  sat_ecef[s * 3 + 0], sat_ecef[s * 3 + 1],
                                  sat_ecef[s * 3 + 2]);
      double r_rover_s = geo_range(x, y, z,
                                   sat_ecef[s * 3 + 0], sat_ecef[s * 3 + 1],
                                   sat_ecef[s * 3 + 2]);
      double dd_geo = (r_rover_s - r_base_s) - (r_rover_ref - r_base_ref);
      residuals[i] = dd_pr[i] - dd_geo;
      residuals[actual_dd + i] = dd_cp[i] - (dd_geo / wavelength + N[i]);
    }
  }

  return iter;
}

// --------------------------------------------------------------------------
// GPU kernel: RTK float batch (one thread per epoch)
// --------------------------------------------------------------------------
__global__ void rtk_float_batch_kernel(
    const double* base_ecef,
    const double* rover_pr, const double* base_pr,
    const double* rover_carrier, const double* base_carrier,
    const double* sat_ecef, double* results,
    double* ambiguities, int* iters,
    int n_epoch, int n_sat, double wavelength,
    int max_iter, double tol) {

  int epoch = blockIdx.x * blockDim.x + threadIdx.x;
  if (epoch >= n_epoch) return;

  int n_dd = n_sat - 1;
  int n_state = 3 + n_dd;

  const double* my_rpr = rover_pr + epoch * n_sat;
  const double* my_bpr = base_pr + epoch * n_sat;
  const double* my_rcp = rover_carrier + epoch * n_sat;
  const double* my_bcp = base_carrier + epoch * n_sat;
  const double* my_sat = sat_ecef + epoch * n_sat * 3;

  double bx = base_ecef[0], by = base_ecef[1], bz = base_ecef[2];

  // Initial position = base
  double x = bx, y = by, z = bz;

  // Select reference satellite
  int ref_idx = select_ref_satellite(bx, by, bz, my_sat, n_sat);

  // Form DD
  double dd_pr_arr[GPU_MAX_DD], dd_cp_arr[GPU_MAX_DD];
  int dd_map_arr[GPU_MAX_DD];
  int actual_dd;
  form_double_difference(my_rpr, my_bpr, my_rcp, my_bcp,
                         ref_idx, n_sat, dd_pr_arr, dd_cp_arr, dd_map_arr, actual_dd);

  // Initialize ambiguities
  double N[GPU_MAX_DD];
  for (int i = 0; i < actual_dd; i++)
    N[i] = dd_cp_arr[i] - dd_pr_arr[i] / wavelength;

  int it;
  for (it = 0; it < max_iter; it++) {
    double HTH[GPU_MAX_STATE * GPU_MAX_STATE];
    double HTdy[GPU_MAX_STATE];
    for (int ii = 0; ii < n_state * n_state; ii++) HTH[ii] = 0.0;
    for (int ii = 0; ii < n_state; ii++) HTdy[ii] = 0.0;

    double r_base_ref = geo_range(bx, by, bz,
                                  my_sat[ref_idx * 3 + 0],
                                  my_sat[ref_idx * 3 + 1],
                                  my_sat[ref_idx * 3 + 2]);
    double r_rover_ref = geo_range(x, y, z,
                                   my_sat[ref_idx * 3 + 0],
                                   my_sat[ref_idx * 3 + 1],
                                   my_sat[ref_idx * 3 + 2]);

    for (int i = 0; i < actual_dd; i++) {
      int s = dd_map_arr[i];
      double r_base_s = geo_range(bx, by, bz,
                                  my_sat[s * 3 + 0], my_sat[s * 3 + 1],
                                  my_sat[s * 3 + 2]);
      double r_rover_s = geo_range(x, y, z,
                                   my_sat[s * 3 + 0], my_sat[s * 3 + 1],
                                   my_sat[s * 3 + 2]);

      double dd_geo = (r_rover_s - r_base_s) - (r_rover_ref - r_base_ref);

      double dx_s = (x - my_sat[s * 3 + 0]) / r_rover_s;
      double dy_s = (y - my_sat[s * 3 + 1]) / r_rover_s;
      double dz_s = (z - my_sat[s * 3 + 2]) / r_rover_s;

      double dx_r = (x - my_sat[ref_idx * 3 + 0]) / r_rover_ref;
      double dy_r = (y - my_sat[ref_idx * 3 + 1]) / r_rover_ref;
      double dz_r = (z - my_sat[ref_idx * 3 + 2]) / r_rover_ref;

      double H_pos[3] = {dx_s - dx_r, dy_s - dy_r, dz_s - dz_r};

      // DD pseudorange
      double res_pr = dd_pr_arr[i] - dd_geo;
      double w_pr = 1.0;
      for (int a = 0; a < 3; a++) {
        HTdy[a] += H_pos[a] * w_pr * res_pr;
        for (int b = 0; b < 3; b++)
          HTH[a * n_state + b] += H_pos[a] * w_pr * H_pos[b];
      }

      // DD carrier phase
      double res_cp = dd_cp_arr[i] - (dd_geo / wavelength + N[i]);
      double w_cp = 10000.0;
      double H_cp[3] = {H_pos[0] / wavelength, H_pos[1] / wavelength, H_pos[2] / wavelength};

      for (int a = 0; a < 3; a++) {
        HTdy[a] += H_cp[a] * w_cp * res_cp;
        for (int b = 0; b < 3; b++)
          HTH[a * n_state + b] += H_cp[a] * w_cp * H_cp[b];
        HTH[a * n_state + (3 + i)] += H_cp[a] * w_cp;
        HTH[(3 + i) * n_state + a] += w_cp * H_cp[a];
      }
      HTH[(3 + i) * n_state + (3 + i)] += w_cp;
      HTdy[3 + i] += w_cp * res_cp;
    }

    // Solve
    double aug[GPU_MAX_STATE * (GPU_MAX_STATE + 1)];
    for (int a = 0; a < n_state; a++) {
      for (int b = 0; b < n_state; b++)
        aug[a * (n_state + 1) + b] = HTH[a * n_state + b];
      aug[a * (n_state + 1) + n_state] = HTdy[a];
    }

    double delta[GPU_MAX_STATE];
    solve_augmented(aug, delta, n_state);

    x += delta[0]; y += delta[1]; z += delta[2];
    for (int i = 0; i < actual_dd; i++)
      N[i] += delta[3 + i];

    double norm = sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                       delta[2] * delta[2]);
    if (norm < tol) { it++; break; }
  }

  results[epoch * 3 + 0] = x;
  results[epoch * 3 + 1] = y;
  results[epoch * 3 + 2] = z;
  for (int i = 0; i < actual_dd; i++)
    ambiguities[epoch * n_dd + i] = N[i];
  if (iters) iters[epoch] = it;
}

// --------------------------------------------------------------------------
// CPU wrapper for batch RTK
// --------------------------------------------------------------------------
void rtk_float_batch(
    const double* base_ecef,
    const double* rover_pr, const double* base_pr,
    const double* rover_carrier, const double* base_carrier,
    const double* sat_ecef, double* results,
    double* ambiguities, int* iters,
    int n_epoch, int n_sat, double wavelength,
    int max_iter, double tol) {

  int n_dd = n_sat - 1;
  size_t sz_obs = (size_t)n_epoch * n_sat * sizeof(double);
  size_t sz_sat = (size_t)n_epoch * n_sat * 3 * sizeof(double);
  size_t sz_res = (size_t)n_epoch * 3 * sizeof(double);
  size_t sz_amb = (size_t)n_epoch * n_dd * sizeof(double);
  size_t sz_base = 3 * sizeof(double);
  size_t sz_it = (size_t)n_epoch * sizeof(int);

  // Each thread uses stack arrays for the solver; ensure sufficient stack.
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));

  double *d_base, *d_rpr, *d_bpr, *d_rcp, *d_bcp, *d_sat, *d_res, *d_amb;
  int *d_it = nullptr;

  CUDA_CHECK(cudaMalloc(&d_base, sz_base));
  CUDA_CHECK(cudaMalloc(&d_rpr, sz_obs));
  CUDA_CHECK(cudaMalloc(&d_bpr, sz_obs));
  CUDA_CHECK(cudaMalloc(&d_rcp, sz_obs));
  CUDA_CHECK(cudaMalloc(&d_bcp, sz_obs));
  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_res, sz_res));
  CUDA_CHECK(cudaMalloc(&d_amb, sz_amb));
  if (iters) CUDA_CHECK(cudaMalloc(&d_it, sz_it));

  CUDA_CHECK(cudaMemcpy(d_base, base_ecef, sz_base, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rpr, rover_pr, sz_obs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bpr, base_pr, sz_obs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rcp, rover_carrier, sz_obs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bcp, base_carrier, sz_obs, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));

  int block = 64;  // fewer threads per block due to large per-thread memory
  int grid = (n_epoch + block - 1) / block;
  rtk_float_batch_kernel<<<grid, block>>>(d_base, d_rpr, d_bpr, d_rcp, d_bcp,
                                           d_sat, d_res, d_amb, d_it,
                                           n_epoch, n_sat, wavelength,
                                           max_iter, tol);

  CUDA_CHECK(cudaMemcpy(results, d_res, sz_res, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ambiguities, d_amb, sz_amb, cudaMemcpyDeviceToHost));
  if (iters) {
    CUDA_CHECK(cudaMemcpy(iters, d_it, sz_it, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_it));
  }

  CUDA_CHECK(cudaFree(d_base)); CUDA_CHECK(cudaFree(d_rpr));
  CUDA_CHECK(cudaFree(d_rcp)); CUDA_CHECK(cudaFree(d_bcp));
  CUDA_CHECK(cudaFree(d_res)); CUDA_CHECK(cudaFree(d_amb));
}

// --------------------------------------------------------------------------
// LAMBDA integer ambiguity resolution (CPU)
// Simplified ILS: LDLT decorrelation + bounded search
// --------------------------------------------------------------------------

// LDLT decomposition of symmetric positive definite matrix Q (n x n)
// Q = L D L^T, L lower triangular with unit diagonal, D diagonal
static void ldlt_decompose(const double* Q, double* L, double* D, int n) {
  // Initialize L = identity, D = 0
  for (int i = 0; i < n * n; i++) L[i] = 0.0;
  for (int i = 0; i < n; i++) { L[i * n + i] = 1.0; D[i] = 0.0; }

  // Work on a copy
  double* W = new double[n * n];
  for (int i = 0; i < n * n; i++) W[i] = Q[i];

  for (int j = 0; j < n; j++) {
    D[j] = W[j * n + j];
    for (int k = 0; k < j; k++)
      D[j] -= L[j * n + k] * L[j * n + k] * D[k];

    if (D[j] < 1e-30) D[j] = 1e-30;

    for (int i = j + 1; i < n; i++) {
      L[i * n + j] = W[i * n + j];
      for (int k = 0; k < j; k++)
        L[i * n + j] -= L[i * n + k] * L[j * n + k] * D[k];
      L[i * n + j] /= D[j];
    }
  }

  delete[] W;
}

// Z-transform decorrelation (integer Gauss transform)
// Reduces the condition number of the ambiguity covariance
static void decorrelate(double* L, double* D, double* z_float, int* Z, int n) {
  // Initialize Z = identity
  for (int i = 0; i < n * n; i++) Z[i] = 0;
  for (int i = 0; i < n; i++) Z[i * n + i] = 1;

  // Iterative integer Gauss transforms
  for (int k = n - 2; k >= 0; k--) {
    for (int i = k + 1; i < n; i++) {
      int mu = (int)round(L[i * n + k]);
      if (mu == 0) continue;

      // Apply integer transformation
      for (int j = 0; j < n; j++)
        Z[i * n + j] -= mu * Z[k * n + j];

      for (int j = 0; j <= k; j++)
        L[i * n + j] -= mu * L[k * n + j];

      z_float[i] -= mu * z_float[k];
    }

    // Permutation step: swap k and k+1 if it improves conditioning
    double delta_val = D[k] + L[k + 1] * L[k + 1] * D[k + 1];  // wrong indexing
    // Actually use: L[(k+1)*n+k]
    double Lkk = L[(k + 1) * n + k];
    delta_val = D[k] + Lkk * Lkk * D[k + 1];

    if (delta_val < D[k + 1]) {
      double eta = D[k] / delta_val;
      double lambda_val = D[k + 1] * Lkk / delta_val;

      D[k] = D[k + 1] * D[k] / delta_val;
      D[k + 1] = delta_val;

      // Swap rows in L and Z
      double Lk_old = L[(k + 1) * n + k];
      L[(k + 1) * n + k] = lambda_val;

      for (int j = 0; j < k; j++) {
        double tmp = L[k * n + j];
        L[k * n + j] = L[(k + 1) * n + j];
        L[(k + 1) * n + j] = tmp;
      }

      for (int i = k + 2; i < n; i++) {
        double tmp = L[i * n + k];
        L[i * n + k] = L[i * n + k] * lambda_val + L[i * n + (k + 1)] * eta;
        L[i * n + (k + 1)] = L[i * n + (k + 1)] - tmp * Lk_old;
      }

      // Swap z_float
      double tmp = z_float[k];
      z_float[k] = z_float[k + 1];
      z_float[k + 1] = tmp;

      // Swap Z rows
      for (int j = 0; j < n; j++) {
        int itmp = Z[k * n + j];
        Z[k * n + j] = Z[(k + 1) * n + j];
        Z[(k + 1) * n + j] = itmp;
      }
    }
  }
}

// Schnorr-Euchner enumeration for integer least-squares search.
// Searches the tree of integer candidates in order of increasing
// distance from the float solution, using LDLT factored covariance
// for efficient pruning.
static double search_ils(const double* z_float, const double* L, const double* D,
                         int* best, int* second_best, double& chi2_best,
                         double& chi2_second, int n, int n_candidates) {
  chi2_best = DBL_MAX;
  chi2_second = DBL_MAX;

  // Working arrays for the tree search
  double* dist = new double[n];     // partial distances at each level
  double* z_cond = new double[n];   // conditional means
  int* z_cur = new int[n];          // current integer candidate
  int* se_step = new int[n];        // SE step counter (0=initial, 1=first alt, 2=second alt, ...)
  int* se_sign = new int[n];        // initial SE direction: +1 if round < z_cond, -1 otherwise

  // Start from the top level (k = n-1)
  int k = n - 1;

  // Initialize conditional mean at top level (no conditioning)
  z_cond[k] = z_float[k];
  z_cur[k] = (int)round(z_cond[k]);
  dist[k] = 0.0;
  se_step[k] = 0;
  se_sign[k] = ((double)z_cur[k] < z_cond[k]) ? 1 : -1;

  int evaluated = 0;
  double chi2_bound = DBL_MAX;  // current search bound

  // Helper lambda: advance SE counter and update z_cur
  // SE sequence from round(z_cond): round, round+sign*1, round-sign*1,
  //   round+sign*2, round-sign*2, ...
  auto se_next = [&](int level) {
    se_step[level]++;
    int s = se_step[level];
    int half = (s + 1) / 2;
    if (s % 2 == 1) {
      z_cur[level] = (int)round(z_cond[level]) + se_sign[level] * half;
    } else {
      z_cur[level] = (int)round(z_cond[level]) - se_sign[level] * half;
    }
  };

  while (true) {
    // Compute distance contribution at level k
    double diff = (double)z_cur[k] - z_cond[k];
    double new_dist = dist[k] + diff * diff / D[k];

    if (new_dist < chi2_bound) {
      if (k == 0) {
        // Complete candidate found
        evaluated++;

        if (new_dist < chi2_best) {
          // New best: old best becomes second best
          chi2_second = chi2_best;
          for (int i = 0; i < n; i++) second_best[i] = best[i];
          chi2_best = new_dist;
          for (int i = 0; i < n; i++) best[i] = z_cur[i];
        } else if (new_dist < chi2_second) {
          chi2_second = new_dist;
          for (int i = 0; i < n; i++) second_best[i] = z_cur[i];
        }

        // Update search bound to second-best distance
        chi2_bound = chi2_second;

        if (evaluated >= n_candidates) break;

        // Stay at level 0, try next candidate (Schnorr-Euchner order)
        se_next(k);
      } else {
        // Move deeper in tree
        k--;
        dist[k] = new_dist;

        // Compute conditional mean at level k, given z_cur[k+1..n-1]
        z_cond[k] = z_float[k];
        for (int j = k + 1; j < n; j++)
          z_cond[k] -= L[j * n + k] * ((double)z_cur[j] - z_float[j]);

        z_cur[k] = (int)round(z_cond[k]);
        se_step[k] = 0;
        se_sign[k] = ((double)z_cur[k] < z_cond[k]) ? 1 : -1;
      }
    } else {
      // Prune: backtrack up the tree
      k++;
      if (k >= n) break;  // exhausted search

      // Try next candidate at level k (Schnorr-Euchner order)
      se_next(k);
    }
  }

  delete[] dist;
  delete[] z_cond;
  delete[] z_cur;
  delete[] se_step;
  delete[] se_sign;

  return (chi2_best > 1e-30) ? chi2_second / chi2_best : 0.0;
}

double lambda_integer(const double* float_amb, const double* Q_amb,
                      int* fixed_amb, int n, int n_candidates) {
  if (n <= 0) return 0.0;

  double* L = new double[n * n];
  double* D = new double[n];
  int* Z = new int[n * n];
  double* z_float = new double[n];
  int* z_best = new int[n];
  int* z_second = new int[n];

  // Copy float ambiguities
  for (int i = 0; i < n; i++) z_float[i] = float_amb[i];

  // LDLT decomposition
  ldlt_decompose(Q_amb, L, D, n);

  // Decorrelate (Z-transform)
  decorrelate(L, D, z_float, Z, n);

  // Search for integer solution in decorrelated space
  double chi2_best, chi2_second;
  double ratio = search_ils(z_float, L, D, z_best, z_second,
                            chi2_best, chi2_second, n, n_candidates);

  // Transform back: fixed = Z^{-1} * z_best
  // Z is an integer unimodular matrix (det = +/-1), so Z^{-1} is also integer.
  // Use integer Gaussian elimination to compute Z^{-1} exactly.
  int* Zaug = new int[n * 2 * n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      Zaug[i * 2 * n + j] = Z[i * n + j];
      Zaug[i * 2 * n + n + j] = (i == j) ? 1 : 0;
    }
  }
  // Integer Gaussian elimination with partial pivoting
  for (int col = 0; col < n; col++) {
    // Find pivot with largest absolute value
    int max_row = col;
    for (int row = col + 1; row < n; row++) {
      if (abs(Zaug[row * 2 * n + col]) > abs(Zaug[max_row * 2 * n + col]))
        max_row = row;
    }
    if (max_row != col) {
      for (int k = 0; k < 2 * n; k++) {
        int tmp = Zaug[col * 2 * n + k];
        Zaug[col * 2 * n + k] = Zaug[max_row * 2 * n + k];
        Zaug[max_row * 2 * n + k] = tmp;
      }
    }
    int pivot = Zaug[col * 2 * n + col];
    if (pivot == 0) continue;

    // Eliminate other rows using integer operations
    for (int row = 0; row < n; row++) {
      if (row == col) continue;
      int val = Zaug[row * 2 * n + col];
      if (val == 0) continue;
      // Since Z is unimodular, pivot divides val exactly after reduction
      // Use: row = row * pivot - val * pivot_row, then divide by pivot
      // For unimodular matrices, we can use direct integer subtraction
      // when pivot is +/-1 (which it will be for unimodular matrices)
      if (pivot == 1) {
        for (int k = 0; k < 2 * n; k++)
          Zaug[row * 2 * n + k] -= val * Zaug[col * 2 * n + k];
      } else if (pivot == -1) {
        for (int k = 0; k < 2 * n; k++)
          Zaug[row * 2 * n + k] += val * Zaug[col * 2 * n + k];
      } else {
        // General case: multiply row by |pivot|, subtract, divide
        // For unimodular Z this path should not be needed, but handle gracefully
        int sign = (pivot > 0) ? 1 : -1;
        int ap = abs(pivot);
        for (int k = 0; k < 2 * n; k++)
          Zaug[row * 2 * n + k] = Zaug[row * 2 * n + k] * ap - val * sign * Zaug[col * 2 * n + k];
        // Divide by ap (should divide evenly for unimodular)
        for (int k = 0; k < 2 * n; k++)
          Zaug[row * 2 * n + k] /= ap;
      }
    }
    // Normalize pivot row so diagonal is 1
    if (pivot == -1) {
      for (int k = 0; k < 2 * n; k++)
        Zaug[col * 2 * n + k] = -Zaug[col * 2 * n + k];
    }
  }

  // fixed_amb = Z^{-1} * z_best (exact integer arithmetic)
  for (int i = 0; i < n; i++) {
    int val = 0;
    for (int j = 0; j < n; j++)
      val += Zaug[i * 2 * n + n + j] * z_best[j];
    fixed_amb[i] = val;
  }

  delete[] L; delete[] D; delete[] Z;
  delete[] z_float; delete[] z_best; delete[] z_second;
  delete[] Zaug;

  return ratio;
}

}  // namespace gnss_gpu
