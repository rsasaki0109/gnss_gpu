#include "gnss_gpu/positioning.h"
#include "gnss_gpu/coordinates.h"
#include "gnss_gpu/cuda_check.h"
#include <cmath>
#include <cstring>

namespace gnss_gpu {

// CPU single-epoch WLS (Gauss-Newton)
int wls_position(const double* sat_ecef, const double* pseudoranges,
                 const double* weights, double* result,
                 int n_sat, int max_iter, double tol) {
  // Need at least 4 satellites for 3D position + clock bias
  if (n_sat < 4) {
    result[0] = 0; result[1] = 0; result[2] = 0; result[3] = 0;
    return -1;
  }

  // Initial guess using Bancroft-like approximation
  // Use first satellite direction, project to Earth surface (~6371 km)
  double cx = 0, cy = 0, cz = 0;
  for (int s = 0; s < n_sat; s++) {
    cx += sat_ecef[s * 3 + 0];
    cy += sat_ecef[s * 3 + 1];
    cz += sat_ecef[s * 3 + 2];
  }
  cx /= n_sat; cy /= n_sat; cz /= n_sat;
  double cn = sqrt(cx * cx + cy * cy + cz * cz);
  // Guard against degenerate satellite geometry (centroid at origin)
  if (cn < 1e-6) {
    result[0] = 0; result[1] = 0; result[2] = 0; result[3] = 0;
    return -1;
  }
  // Project centroid direction onto Earth surface
  double scale = WGS84_A / cn;  // ~6378 km / ~26000 km
  double x = cx * scale, y = cy * scale, z = cz * scale;
  // Estimate clock bias from mean pseudorange residual
  double cb = 0;
  for (int s = 0; s < n_sat; s++) {
    double dx = x - sat_ecef[s * 3 + 0];
    double dy_v = y - sat_ecef[s * 3 + 1];
    double dz = z - sat_ecef[s * 3 + 2];
    double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
    cb += pseudoranges[s] - r;
  }
  cb /= n_sat;

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    // H^T W H and H^T W dy (normal equation)
    double HTWHa[16] = {};  // 4x4
    double HTWdy[4] = {};

    for (int s = 0; s < n_sat; s++) {
      double sx = sat_ecef[s * 3 + 0];
      double sy = sat_ecef[s * 3 + 1];
      double sz = sat_ecef[s * 3 + 2];

      // Keep the single-epoch solver consistent with wls_batch() and the
      // Python fallback: inputs are expected to already be corrected upstream.
      double dx = x - sx, dy_v = y - sy, dz = z - sz;
      double r = sqrt(dx * dx + dy_v * dy_v + dz * dz);
      double pr_pred = r + cb;

      double residual = pseudoranges[s] - pr_pred;
      double w = weights[s];

      // Jacobian row: [(x-sx)/r, (y-sy)/r, (z-sz)/r, 1]
      double H[4] = {dx / r, dy_v / r, dz / r, 1.0};

      for (int a = 0; a < 4; a++) {
        HTWdy[a] += H[a] * w * residual;
        for (int b = 0; b < 4; b++) {
          HTWHa[a * 4 + b] += H[a] * w * H[b];
        }
      }
    }

    // Solve 4x4 system by Gaussian elimination
    double A[4][5];
    for (int a = 0; a < 4; a++) {
      for (int b = 0; b < 4; b++) A[a][b] = HTWHa[a * 4 + b];
      A[a][4] = HTWdy[a];
    }

    bool singular = false;
    for (int col = 0; col < 4; col++) {
      // Partial pivoting
      int max_row = col;
      for (int row = col + 1; row < 4; row++) {
        if (fabs(A[row][col]) > fabs(A[max_row][col])) max_row = row;
      }
      for (int k = 0; k < 5; k++) {
        double tmp = A[col][k]; A[col][k] = A[max_row][k]; A[max_row][k] = tmp;
      }
      if (fabs(A[col][col]) < 1e-15) { singular = true; break; }
      for (int row = col + 1; row < 4; row++) {
        double factor = A[row][col] / A[col][col];
        for (int k = col; k < 5; k++) A[row][k] -= factor * A[col][k];
      }
    }
    if (singular) {
      // Matrix is singular; return current best estimate
      result[0] = x; result[1] = y; result[2] = z; result[3] = cb;
      return iter;
    }
    double delta[4] = {};
    for (int row = 3; row >= 0; row--) {
      delta[row] = A[row][4];
      for (int col = row + 1; col < 4; col++) delta[row] -= A[row][col] * delta[col];
      delta[row] /= A[row][row];
    }

    x += delta[0]; y += delta[1]; z += delta[2]; cb += delta[3];

    double norm = sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                       delta[2] * delta[2] + delta[3] * delta[3]);
    if (norm < tol) { iter++; break; }
  }

  result[0] = x; result[1] = y; result[2] = z; result[3] = cb;
  return iter;
}

// GPU batch WLS kernel
__device__ void solve_4x4(double A[4][5], double* delta) {
  for (int col = 0; col < 4; col++) {
    int max_row = col;
    for (int row = col + 1; row < 4; row++) {
      if (fabs(A[row][col]) > fabs(A[max_row][col])) max_row = row;
    }
    if (max_row != col) {
      for (int k = 0; k < 5; k++) {
        double tmp = A[col][k]; A[col][k] = A[max_row][k]; A[max_row][k] = tmp;
      }
    }
    if (fabs(A[col][col]) < 1e-15) continue;
    for (int row = col + 1; row < 4; row++) {
      double factor = A[row][col] / A[col][col];
      for (int k = col; k < 5; k++) A[row][k] -= factor * A[col][k];
    }
  }
  for (int row = 3; row >= 0; row--) {
    delta[row] = A[row][4];
    for (int col = row + 1; col < 4; col++) delta[row] -= A[row][col] * delta[col];
    delta[row] /= A[row][row];
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

__global__ void wls_batch_kernel(const double* sat_ecef, const double* pseudoranges,
                                  const double* weights, double* results, int* iters,
                                  int n_epoch, int n_sat, int max_iter, double tol) {
  int epoch = blockIdx.x * blockDim.x + threadIdx.x;
  if (epoch >= n_epoch) return;

  const double* my_sat = sat_ecef + epoch * n_sat * 3;
  const double* my_pr = pseudoranges + epoch * n_sat;
  const double* my_w = weights + epoch * n_sat;

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
  double cb = 0;
  for (int s = 0; s < n_sat; s++) {
    double dx, dy_v, dz, r;
    sagnac_los_device(my_sat, s, x, y, z, &dx, &dy_v, &dz, &r);
    cb += my_pr[s] - r;
  }
  cb /= n_sat;

  int it;
  for (it = 0; it < max_iter; it++) {
    double HTWH[16] = {};
    double HTWdy[4] = {};

    for (int s = 0; s < n_sat; s++) {
      double dx, dy_v, dz, r;
      sagnac_los_device(my_sat, s, x, y, z, &dx, &dy_v, &dz, &r);
      double residual = my_pr[s] - (r + cb);
      double w = my_w[s];
      double H[4] = {dx / r, dy_v / r, dz / r, 1.0};

      for (int a = 0; a < 4; a++) {
        HTWdy[a] += H[a] * w * residual;
        for (int b = 0; b < 4; b++) {
          HTWH[a * 4 + b] += H[a] * w * H[b];
        }
      }
    }

    double A[4][5];
    for (int a = 0; a < 4; a++) {
      for (int b = 0; b < 4; b++) A[a][b] = HTWH[a * 4 + b];
      A[a][4] = HTWdy[a];
    }
    double delta[4];
    solve_4x4(A, delta);

    x += delta[0]; y += delta[1]; z += delta[2]; cb += delta[3];

    double norm = sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                       delta[2] * delta[2] + delta[3] * delta[3]);
    if (norm < tol) { it++; break; }
  }

  results[epoch * 4 + 0] = x;
  results[epoch * 4 + 1] = y;
  results[epoch * 4 + 2] = z;
  results[epoch * 4 + 3] = cb;
  if (iters) iters[epoch] = it;
}

void wls_batch(const double* sat_ecef, const double* pseudoranges,
               const double* weights, double* results, int* iters,
               int n_epoch, int n_sat, int max_iter, double tol) {
  size_t sz_sat = (size_t)n_epoch * n_sat * 3 * sizeof(double);
  size_t sz_pr = (size_t)n_epoch * n_sat * sizeof(double);
  size_t sz_res = (size_t)n_epoch * 4 * sizeof(double);
  size_t sz_it = (size_t)n_epoch * sizeof(int);

  double *d_sat, *d_pr, *d_w, *d_res;
  int *d_it = nullptr;

  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_pr, sz_pr));
  CUDA_CHECK(cudaMalloc(&d_w, sz_pr));
  CUDA_CHECK(cudaMalloc(&d_res, sz_res));
  if (iters) CUDA_CHECK(cudaMalloc(&d_it, sz_it));

  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pr, pseudoranges, sz_pr, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, weights, sz_pr, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_epoch + block - 1) / block;
  wls_batch_kernel<<<grid, block>>>(d_sat, d_pr, d_w, d_res, d_it,
                                     n_epoch, n_sat, max_iter, tol);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaMemcpy(results, d_res, sz_res, cudaMemcpyDeviceToHost));
  if (iters) {
    CUDA_CHECK(cudaMemcpy(iters, d_it, sz_it, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_it));
  }

  CUDA_CHECK(cudaFree(d_sat)); CUDA_CHECK(cudaFree(d_pr));
  CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_res));
}

}  // namespace gnss_gpu
