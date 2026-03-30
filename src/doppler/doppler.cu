#include "gnss_gpu/doppler.h"
#include "gnss_gpu/cuda_check.h"
#include <cmath>

namespace gnss_gpu {

// CPU single-epoch Doppler velocity estimation (Gauss-Newton)
//
// Observation model:
//   doppler_meas * lambda = (sat_vel - rx_vel) . LOS + clock_drift
// where LOS = (sat_pos - rx_pos) / |sat_pos - rx_pos|
//
// Rearranged residual:
//   y = doppler_meas * lambda - (sat_vel . LOS - clock_drift)
//   y = predicted - observed  => solve for rx_vel and clock_drift
//
// Jacobian H = [-LOS_x, -LOS_y, -LOS_z, 1] per satellite
// (negative because rx_vel appears with a minus sign in the observation)

int doppler_velocity(const double* sat_ecef, const double* sat_vel,
                     const double* doppler, const double* rx_pos,
                     const double* weights, double* result,
                     int n_sat, double wavelength, int max_iter, double tol) {
  if (n_sat < 4) {
    result[0] = 0; result[1] = 0; result[2] = 0; result[3] = 0;
    return -1;
  }

  double rx = rx_pos[0], ry = rx_pos[1], rz = rx_pos[2];

  // Initial guess: zero velocity, zero clock drift
  double vx = 0, vy = 0, vz = 0, cd = 0;

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    double HTWH[16] = {};  // 4x4
    double HTWdy[4] = {};

    for (int s = 0; s < n_sat; s++) {
      double sx = sat_ecef[s * 3 + 0];
      double sy = sat_ecef[s * 3 + 1];
      double sz = sat_ecef[s * 3 + 2];

      double dx = sx - rx, dy = sy - ry, dz = sz - rz;
      double r = sqrt(dx * dx + dy * dy + dz * dz);
      if (r < 1e-6) continue;

      // Line-of-sight unit vector (receiver -> satellite)
      double lx = dx / r, ly = dy / r, lz = dz / r;

      double svx = sat_vel[s * 3 + 0];
      double svy = sat_vel[s * 3 + 1];
      double svz = sat_vel[s * 3 + 2];

      // Predicted Doppler * lambda = (sat_vel - rx_vel) . LOS + clock_drift
      double pred = (svx - vx) * lx + (svy - vy) * ly + (svz - vz) * lz + cd;
      double obs = doppler[s] * wavelength;
      double residual = obs - pred;
      double w = weights[s];

      // Jacobian row: d(pred)/d(vx) = -lx, d(pred)/d(vy) = -ly,
      //               d(pred)/d(vz) = -lz, d(pred)/d(cd) = 1
      double H[4] = {-lx, -ly, -lz, 1.0};

      for (int a = 0; a < 4; a++) {
        HTWdy[a] += H[a] * w * residual;
        for (int b = 0; b < 4; b++) {
          HTWH[a * 4 + b] += H[a] * w * H[b];
        }
      }
    }

    // Solve 4x4 system by Gaussian elimination with partial pivoting
    double A[4][5];
    for (int a = 0; a < 4; a++) {
      for (int b = 0; b < 4; b++) A[a][b] = HTWH[a * 4 + b];
      A[a][4] = HTWdy[a];
    }

    bool singular = false;
    for (int col = 0; col < 4; col++) {
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
      result[0] = vx; result[1] = vy; result[2] = vz; result[3] = cd;
      return iter;
    }
    double delta[4] = {};
    for (int row = 3; row >= 0; row--) {
      delta[row] = A[row][4];
      for (int col = row + 1; col < 4; col++) delta[row] -= A[row][col] * delta[col];
      delta[row] /= A[row][row];
    }

    vx += delta[0]; vy += delta[1]; vz += delta[2]; cd += delta[3];

    double norm = sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                       delta[2] * delta[2] + delta[3] * delta[3]);
    if (norm < tol) { iter++; break; }
  }

  result[0] = vx; result[1] = vy; result[2] = vz; result[3] = cd;
  return iter;
}

// GPU kernel: solve 4x4 linear system (shared with wls.cu pattern)
__device__ void doppler_solve_4x4(double A[4][5], double* delta) {
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

__global__ void doppler_velocity_batch_kernel(
    const double* sat_ecef, const double* sat_vel,
    const double* doppler, const double* rx_pos,
    const double* weights, double* results, int* iters,
    int n_epoch, int n_sat, double wavelength, int max_iter, double tol) {

  int epoch = blockIdx.x * blockDim.x + threadIdx.x;
  if (epoch >= n_epoch) return;

  const double* my_sat = sat_ecef + epoch * n_sat * 3;
  const double* my_svel = sat_vel + epoch * n_sat * 3;
  const double* my_dop = doppler + epoch * n_sat;
  const double* my_rx = rx_pos + epoch * 3;
  const double* my_w = weights + epoch * n_sat;

  double rx = my_rx[0], ry = my_rx[1], rz = my_rx[2];
  double vx = 0, vy = 0, vz = 0, cd = 0;

  int it;
  for (it = 0; it < max_iter; it++) {
    double HTWH[16] = {};
    double HTWdy[4] = {};

    for (int s = 0; s < n_sat; s++) {
      double dx = my_sat[s * 3 + 0] - rx;
      double dy = my_sat[s * 3 + 1] - ry;
      double dz = my_sat[s * 3 + 2] - rz;
      double r = sqrt(dx * dx + dy * dy + dz * dz);
      if (r < 1e-6) continue;

      double lx = dx / r, ly = dy / r, lz = dz / r;

      double pred = (my_svel[s * 3 + 0] - vx) * lx +
                    (my_svel[s * 3 + 1] - vy) * ly +
                    (my_svel[s * 3 + 2] - vz) * lz + cd;
      double obs = my_dop[s] * wavelength;
      double residual = obs - pred;
      double w = my_w[s];

      double H[4] = {-lx, -ly, -lz, 1.0};

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
    doppler_solve_4x4(A, delta);

    vx += delta[0]; vy += delta[1]; vz += delta[2]; cd += delta[3];

    double norm = sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                       delta[2] * delta[2] + delta[3] * delta[3]);
    if (norm < tol) { it++; break; }
  }

  results[epoch * 4 + 0] = vx;
  results[epoch * 4 + 1] = vy;
  results[epoch * 4 + 2] = vz;
  results[epoch * 4 + 3] = cd;
  if (iters) iters[epoch] = it;
}

void doppler_velocity_batch(const double* sat_ecef, const double* sat_vel,
                            const double* doppler, const double* rx_pos,
                            const double* weights, double* results, int* iters,
                            int n_epoch, int n_sat, double wavelength,
                            int max_iter, double tol) {
  size_t sz_sat = (size_t)n_epoch * n_sat * 3 * sizeof(double);
  size_t sz_dop = (size_t)n_epoch * n_sat * sizeof(double);
  size_t sz_rx = (size_t)n_epoch * 3 * sizeof(double);
  size_t sz_res = (size_t)n_epoch * 4 * sizeof(double);
  size_t sz_it = (size_t)n_epoch * sizeof(int);

  double *d_sat, *d_svel, *d_dop, *d_rx, *d_w, *d_res;
  int *d_it = nullptr;

  CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_svel, sz_sat));
  CUDA_CHECK(cudaMalloc(&d_dop, sz_dop));
  CUDA_CHECK(cudaMalloc(&d_rx, sz_rx));
  CUDA_CHECK(cudaMalloc(&d_w, sz_dop));
  CUDA_CHECK(cudaMalloc(&d_res, sz_res));
  if (iters) CUDA_CHECK(cudaMalloc(&d_it, sz_it));

  CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_svel, sat_vel, sz_sat, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dop, doppler, sz_dop, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rx, rx_pos, sz_rx, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, weights, sz_dop, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = (n_epoch + block - 1) / block;
  doppler_velocity_batch_kernel<<<grid, block>>>(
      d_sat, d_svel, d_dop, d_rx, d_w, d_res, d_it,
      n_epoch, n_sat, wavelength, max_iter, tol);
  CUDA_CHECK_LAST();

  CUDA_CHECK(cudaMemcpy(results, d_res, sz_res, cudaMemcpyDeviceToHost));
  if (iters) {
    CUDA_CHECK(cudaMemcpy(iters, d_it, sz_it, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_it));
  }

  CUDA_CHECK(cudaFree(d_sat));
  CUDA_CHECK(cudaFree(d_svel));
  CUDA_CHECK(cudaFree(d_dop));
  CUDA_CHECK(cudaFree(d_rx));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_res));
}

}  // namespace gnss_gpu
