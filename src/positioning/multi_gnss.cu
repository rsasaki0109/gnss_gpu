#include "gnss_gpu/multi_gnss.h"
#include "gnss_gpu/coordinates.h"
#include "gnss_gpu/cuda_check.h"
#include <cmath>
#include <cstring>

namespace gnss_gpu {

// Maximum state dimension: 3 (position) + MAX_SYSTEMS (clock biases)
static constexpr int MAX_STATE = 3 + MAX_SYSTEMS;

// Solve NxN augmented matrix [A|b] by Gaussian elimination with partial pivoting
// A_aug: [n][n+1] augmented matrix, overwritten in-place
// delta: [n] solution output
static void solve_NxN_cpu(double* A_aug, double* delta, int n) {
  // Forward elimination with partial pivoting
  for (int col = 0; col < n; col++) {
    int max_row = col;
    for (int row = col + 1; row < n; row++) {
      if (fabs(A_aug[row * (n + 1) + col]) > fabs(A_aug[max_row * (n + 1) + col]))
        max_row = row;
    }
    if (max_row != col) {
      for (int k = 0; k < n + 1; k++) {
        double tmp = A_aug[col * (n + 1) + k];
        A_aug[col * (n + 1) + k] = A_aug[max_row * (n + 1) + k];
        A_aug[max_row * (n + 1) + k] = tmp;
      }
    }
    if (fabs(A_aug[col * (n + 1) + col]) < 1e-15) continue;
    for (int row = col + 1; row < n; row++) {
      double factor = A_aug[row * (n + 1) + col] / A_aug[col * (n + 1) + col];
      for (int k = col; k < n + 1; k++)
        A_aug[row * (n + 1) + k] -= factor * A_aug[col * (n + 1) + k];
    }
  }
  // Back substitution
  for (int row = n - 1; row >= 0; row--) {
    delta[row] = A_aug[row * (n + 1) + n];
    for (int col = row + 1; col < n; col++)
      delta[row] -= A_aug[row * (n + 1) + col] * delta[col];
    if (fabs(A_aug[row * (n + 1) + row]) > 1e-15)
      delta[row] /= A_aug[row * (n + 1) + row];
    else
      delta[row] = 0.0;
  }
}

// CPU single-epoch multi-GNSS WLS (Gauss-Newton)
int wls_multi_gnss(const double* sat_ecef, const double* pseudoranges,
                   const double* weights, const int* system_ids,
                   double* result, int n_sat, int n_systems,
                   int max_iter, double tol) {
  int n_state = 3 + n_systems;
  // Need at least n_state satellites
  if (n_sat < n_state) {
    for (int i = 0; i < n_state; i++) result[i] = 0;
    return -1;
  }

  // Initial guess: project satellite centroid onto Earth surface
  double cx = 0, cy = 0, cz = 0;
  for (int s = 0; s < n_sat; s++) {
    cx += sat_ecef[s * 3 + 0];
    cy += sat_ecef[s * 3 + 1];
    cz += sat_ecef[s * 3 + 2];
  }
  cx /= n_sat; cy /= n_sat; cz /= n_sat;
  double cn = sqrt(cx * cx + cy * cy + cz * cz);
  if (cn < 1e-6) {
    for (int i = 0; i < n_state; i++) result[i] = 0;
    return -1;
  }
  double scale = WGS84_A / cn;
  double x = cx * scale, y = cy * scale, z = cz * scale;

  // Initialize per-system clock biases from mean pseudorange residuals
  double cb[MAX_SYSTEMS] = {};
  int cb_count[MAX_SYSTEMS] = {};
  for (int s = 0; s < n_sat; s++) {
    double dx = x - sat_ecef[s * 3 + 0];
    double dy_v = y - sat_ecef[s * 3 + 1];
    double dz = z - sat_ecef[s * 3 + 2];
    double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
    int sys = system_ids[s];
    if (sys >= 0 && sys < n_systems) {
      cb[sys] += pseudoranges[s] - r;
      cb_count[sys]++;
    }
  }
  for (int k = 0; k < n_systems; k++) {
    if (cb_count[k] > 0) cb[k] /= cb_count[k];
  }

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    // Normal equation: H^T W H and H^T W dy
    double HTWH[MAX_STATE * MAX_STATE] = {};
    double HTWdy[MAX_STATE] = {};

    for (int s = 0; s < n_sat; s++) {
      double sx = sat_ecef[s * 3 + 0];
      double sy = sat_ecef[s * 3 + 1];
      double sz = sat_ecef[s * 3 + 2];

      // Keep the single-epoch solver consistent with the batch/kernel and the
      // Python fallback: inputs are expected to already be corrected upstream.
      double dx = x - sx, dy_v = y - sy, dz = z - sz;
      double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      int sys = system_ids[s];
      double pr_pred = r + ((sys >= 0 && sys < n_systems) ? cb[sys] : cb[0]);
      double residual = pseudoranges[s] - pr_pred;
      double w = weights[s];

      // Jacobian row: [dx/r, dy/r, dz/r, 0, ..., 1(at col 3+sys), ..., 0]
      double H[MAX_STATE] = {};
      H[0] = dx / r;
      H[1] = dy_v / r;
      H[2] = dz / r;
      if (sys >= 0 && sys < n_systems) H[3 + sys] = 1.0;

      for (int a = 0; a < n_state; a++) {
        HTWdy[a] += H[a] * w * residual;
        for (int b = 0; b < n_state; b++) {
          HTWH[a * n_state + b] += H[a] * w * H[b];
        }
      }
    }

    // Build augmented matrix and solve
    double A_aug[MAX_STATE * (MAX_STATE + 1)] = {};
    for (int a = 0; a < n_state; a++) {
      for (int b = 0; b < n_state; b++)
        A_aug[a * (n_state + 1) + b] = HTWH[a * n_state + b];
      A_aug[a * (n_state + 1) + n_state] = HTWdy[a];
    }

    double delta[MAX_STATE] = {};
    solve_NxN_cpu(A_aug, delta, n_state);

    x += delta[0]; y += delta[1]; z += delta[2];
    for (int k = 0; k < n_systems; k++) cb[k] += delta[3 + k];

    double norm = 0;
    for (int i = 0; i < n_state; i++) norm += delta[i] * delta[i];
    norm = sqrt(norm);
    if (norm < tol) { iter++; break; }
  }

  result[0] = x; result[1] = y; result[2] = z;
  for (int k = 0; k < n_systems; k++) result[3 + k] = cb[k];
  return iter;
}

// GPU device solver: NxN with partial pivoting
// Uses flat array A_aug[n * (n+1)] and delta[n]
__device__ void solve_NxN_device(double* A_aug, double* delta, int n) {
  for (int col = 0; col < n; col++) {
    int max_row = col;
    for (int row = col + 1; row < n; row++) {
      if (fabs(A_aug[row * (n + 1) + col]) > fabs(A_aug[max_row * (n + 1) + col]))
        max_row = row;
    }
    if (max_row != col) {
      for (int k = 0; k < n + 1; k++) {
        double tmp = A_aug[col * (n + 1) + k];
        A_aug[col * (n + 1) + k] = A_aug[max_row * (n + 1) + k];
        A_aug[max_row * (n + 1) + k] = tmp;
      }
    }
    if (fabs(A_aug[col * (n + 1) + col]) < 1e-15) continue;
    for (int row = col + 1; row < n; row++) {
      double factor = A_aug[row * (n + 1) + col] / A_aug[col * (n + 1) + col];
      for (int k = col; k < n + 1; k++)
        A_aug[row * (n + 1) + k] -= factor * A_aug[col * (n + 1) + k];
    }
  }
  for (int row = n - 1; row >= 0; row--) {
    delta[row] = A_aug[row * (n + 1) + n];
    for (int col = row + 1; col < n; col++)
      delta[row] -= A_aug[row * (n + 1) + col] * delta[col];
    if (fabs(A_aug[row * (n + 1) + row]) > 1e-15)
      delta[row] /= A_aug[row * (n + 1) + row];
    else
      delta[row] = 0.0;
  }
}

static __device__ inline void sagnac_los_device(
    const double* sat_ecef, int s, double x, double y, double z,
    double* dx, double* dy, double* dz, double* r) {
  double sx = sat_ecef[s * 3 + 0];
  double sy = sat_ecef[s * 3 + 1];
  double sz = sat_ecef[s * 3 + 2];

  double dx0 = x - sx;
  double dy0 = y - sy;
  double dz0 = z - sz;
  double range_approx = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
  double theta = 7.2921151467e-5 * (range_approx / 299792458.0);
  double sx_rot = sx * cos(theta) + sy * sin(theta);
  double sy_rot = -sx * sin(theta) + sy * cos(theta);

  *dx = x - sx_rot;
  *dy = y - sy_rot;
  *dz = z - sz;
  *r = sqrt((*dx) * (*dx) + (*dy) * (*dy) + (*dz) * (*dz));
  if (*r < 1e-12) *r = 1e-12;
}

__global__ void wls_multi_gnss_batch_kernel(
    const double* sat_ecef, const double* pseudoranges,
    const double* weights, const int* system_ids,
    double* results, int* iters,
    int n_epoch, int n_sat, int n_systems,
    int max_iter, double tol) {

  int epoch = blockIdx.x * blockDim.x + threadIdx.x;
  if (epoch >= n_epoch) return;

  int n_state = 3 + n_systems;
  const double* my_sat = sat_ecef + epoch * n_sat * 3;
  const double* my_pr = pseudoranges + epoch * n_sat;
  const double* my_w = weights + epoch * n_sat;
  const int* my_sys = system_ids + epoch * n_sat;

  // Initial guess: project satellite centroid onto Earth surface
  double cx = 0, cy = 0, cz = 0;
  for (int s = 0; s < n_sat; s++) {
    cx += my_sat[s * 3 + 0];
    cy += my_sat[s * 3 + 1];
    cz += my_sat[s * 3 + 2];
  }
  cx /= n_sat; cy /= n_sat; cz /= n_sat;
  double cn = sqrt(cx * cx + cy * cy + cz * cz);
  double scale = WGS84_A / cn;
  double x = cx * scale, y = cy * scale, z = cz * scale;

  // Per-system clock bias initialization
  double cb[MAX_SYSTEMS] = {};
  int cb_count[MAX_SYSTEMS] = {};
  for (int s = 0; s < n_sat; s++) {
    double dx, dy_v, dz, r;
    sagnac_los_device(my_sat, s, x, y, z, &dx, &dy_v, &dz, &r);
    int sys = my_sys[s];
    if (sys >= 0 && sys < n_systems) {
      cb[sys] += my_pr[s] - r;
      cb_count[sys]++;
    }
  }
  for (int k = 0; k < n_systems; k++) {
    if (cb_count[k] > 0) cb[k] /= cb_count[k];
  }

  int it;
  for (it = 0; it < max_iter; it++) {
    double HTWH[MAX_STATE * MAX_STATE] = {};
    double HTWdy[MAX_STATE] = {};

    for (int s = 0; s < n_sat; s++) {
      double dx, dy_v, dz, r;
      sagnac_los_device(my_sat, s, x, y, z, &dx, &dy_v, &dz, &r);
      int sys = my_sys[s];
      double pr_pred = r + ((sys >= 0 && sys < n_systems) ? cb[sys] : cb[0]);
      double residual = my_pr[s] - (pr_pred);
      double w = my_w[s];

      double H[MAX_STATE] = {};
      H[0] = dx / r;
      H[1] = dy_v / r;
      H[2] = dz / r;
      if (sys >= 0 && sys < n_systems) H[3 + sys] = 1.0;

      for (int a = 0; a < n_state; a++) {
        HTWdy[a] += H[a] * w * residual;
        for (int b = 0; b < n_state; b++) {
          HTWH[a * n_state + b] += H[a] * w * H[b];
        }
      }
    }

    double A_aug[MAX_STATE * (MAX_STATE + 1)] = {};
    for (int a = 0; a < n_state; a++) {
      for (int b = 0; b < n_state; b++)
        A_aug[a * (n_state + 1) + b] = HTWH[a * n_state + b];
      A_aug[a * (n_state + 1) + n_state] = HTWdy[a];
    }

    double delta[MAX_STATE] = {};
    solve_NxN_device(A_aug, delta, n_state);

    x += delta[0]; y += delta[1]; z += delta[2];
    for (int k = 0; k < n_systems; k++) cb[k] += delta[3 + k];

    double norm = 0;
    for (int i = 0; i < n_state; i++) norm += delta[i] * delta[i];
    norm = sqrt(norm);
    if (norm < tol) { it++; break; }
  }

  double* my_res = results + epoch * n_state;
  my_res[0] = x; my_res[1] = y; my_res[2] = z;
  for (int k = 0; k < n_systems; k++) my_res[3 + k] = cb[k];
  if (iters) iters[epoch] = it;
}

void wls_multi_gnss_batch(const double* sat_ecef, const double* pseudoranges,
                          const double* weights, const int* system_ids,
                          double* results, int* iters,
                          int n_epoch, int n_sat, int n_systems,
                          int max_iter, double tol) {
  int n_state = 3 + n_systems;
  size_t sz_sat = (size_t)n_epoch * n_sat * 3 * sizeof(double);
  size_t sz_pr = (size_t)n_epoch * n_sat * sizeof(double);
  size_t sz_sys = (size_t)n_epoch * n_sat * sizeof(int);
  size_t sz_res = (size_t)n_epoch * n_state * sizeof(double);
  size_t sz_it = (size_t)n_epoch * sizeof(int);

  double *d_sat, *d_pr, *d_w, *d_res;
  int *d_sys, *d_it = nullptr;

  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_pr, sz_pr));
  CUDA_CHECK(cudaMalloc(&d_w, sz_pr));
  CUDA_CHECK(cudaMalloc(&d_sys, sz_sys));
  CUDA_CHECK(cudaMalloc(&d_res, sz_res));
  if (iters) CUDA_CHECK(cudaMalloc(&d_it, sz_it));

  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pr, pseudoranges, sz_pr, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, weights, sz_pr, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sys, system_ids, sz_sys, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_epoch + block - 1) / block;
  wls_multi_gnss_batch_kernel<<<grid, block>>>(d_sat, d_pr, d_w, d_sys, d_res, d_it,
                                                n_epoch, n_sat, n_systems,
                                                max_iter, tol);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaMemcpy(results, d_res, sz_res, cudaMemcpyDeviceToHost));
  if (iters) {
    CUDA_CHECK(cudaMemcpy(iters, d_it, sz_it, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_it));
  }

  CUDA_CHECK(cudaFree(d_sat)); CUDA_CHECK(cudaFree(d_pr));
  CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_sys));
  CUDA_CHECK(cudaFree(d_res));
}

}  // namespace gnss_gpu
