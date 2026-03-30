#include "gnss_gpu/ekf.h"
#include "gnss_gpu/cuda_check.h"
#include <cmath>
#include <cstring>

namespace gnss_gpu {

// ============================================================
// 8x8 matrix helpers (row-major, all on stack)
// ============================================================

static inline void mat8_zero(double* M) {
    memset(M, 0, 64 * sizeof(double));
}

static inline void mat8_identity(double* M) {
    mat8_zero(M);
    for (int i = 0; i < 8; i++) M[i * 8 + i] = 1.0;
}

// C = A * B  (all 8x8)
static void mat8_mul(const double* A, const double* B, double* C) {
    mat8_zero(C);
    for (int i = 0; i < 8; i++)
        for (int k = 0; k < 8; k++) {
            double a = A[i * 8 + k];
            for (int j = 0; j < 8; j++)
                C[i * 8 + j] += a * B[k * 8 + j];
        }
}

// C = A * B^T  (all 8x8)
static void mat8_mul_bt(const double* A, const double* B, double* C) {
    mat8_zero(C);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            double sum = 0;
            for (int k = 0; k < 8; k++)
                sum += A[i * 8 + k] * B[j * 8 + k];
            C[i * 8 + j] = sum;
        }
}

// A += B
static void mat8_add(double* A, const double* B) {
    for (int i = 0; i < 64; i++) A[i] += B[i];
}

// ============================================================
// Small NxN matrix helpers for measurement update
// ============================================================

// Solve NxN linear system A*X = B by Gaussian elimination with partial pivoting
// A is NxN (destroyed), B is NxM (overwritten with solution)
// Returns false if singular
static bool solve_linear(double* A, double* B, int N, int M) {
    for (int col = 0; col < N; col++) {
        int max_row = col;
        for (int row = col + 1; row < N; row++) {
            if (fabs(A[row * N + col]) > fabs(A[max_row * N + col]))
                max_row = row;
        }
        if (max_row != col) {
            for (int k = 0; k < N; k++) {
                double tmp = A[col * N + k]; A[col * N + k] = A[max_row * N + k]; A[max_row * N + k] = tmp;
            }
            for (int k = 0; k < M; k++) {
                double tmp = B[col * M + k]; B[col * M + k] = B[max_row * M + k]; B[max_row * M + k] = tmp;
            }
        }
        if (fabs(A[col * N + col]) < 1e-15) return false;
        double pivot = A[col * N + col];
        for (int row = col + 1; row < N; row++) {
            double factor = A[row * N + col] / pivot;
            for (int k = col; k < N; k++)
                A[row * N + k] -= factor * A[col * N + k];
            for (int k = 0; k < M; k++)
                B[row * M + k] -= factor * B[col * M + k];
        }
    }
    // Back substitution
    for (int row = N - 1; row >= 0; row--) {
        for (int k = 0; k < M; k++) {
            for (int col = row + 1; col < N; col++)
                B[row * M + k] -= A[row * N + col] * B[col * M + k];
            B[row * M + k] /= A[row * N + row];
        }
    }
    return true;
}

// ============================================================
// CPU EKF implementation
// ============================================================

void ekf_initialize(EKFState* state, const double* initial_pos,
                    double initial_cb, double sigma_pos, double sigma_cb) {
    memset(state, 0, sizeof(EKFState));
    state->x[0] = initial_pos[0];
    state->x[1] = initial_pos[1];
    state->x[2] = initial_pos[2];
    // x[3..5] = 0 (velocity)
    state->x[6] = initial_cb;
    // x[7] = 0 (clock drift)

    mat8_zero(state->P);
    double sp2 = sigma_pos * sigma_pos;
    double sv2 = 100.0 * 100.0;   // large initial velocity uncertainty
    double sc2 = sigma_cb * sigma_cb;
    double sd2 = 100.0 * 100.0;   // large initial clock drift uncertainty
    state->P[0 * 8 + 0] = sp2;
    state->P[1 * 8 + 1] = sp2;
    state->P[2 * 8 + 2] = sp2;
    state->P[3 * 8 + 3] = sv2;
    state->P[4 * 8 + 4] = sv2;
    state->P[5 * 8 + 5] = sv2;
    state->P[6 * 8 + 6] = sc2;
    state->P[7 * 8 + 7] = sd2;
}

void ekf_predict(EKFState* state, double dt, const EKFConfig& config) {
    // State transition: constant velocity + clock drift model
    // x_new = F * x
    // F = I + dt * off-diagonal blocks:
    //   pos += vel * dt
    //   cb  += cd  * dt
    double F[64];
    mat8_identity(F);
    F[0 * 8 + 3] = dt;  // x += vx*dt
    F[1 * 8 + 4] = dt;  // y += vy*dt
    F[2 * 8 + 5] = dt;  // z += vz*dt
    F[6 * 8 + 7] = dt;  // cb += cd*dt

    // Predict state
    double x_new[8] = {};
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            x_new[i] += F[i * 8 + j] * state->x[j];
    memcpy(state->x, x_new, 8 * sizeof(double));

    // Process noise Q
    double Q[64];
    mat8_zero(Q);
    double dt2 = dt * dt;
    double sp2 = config.sigma_pos * config.sigma_pos;
    double sv2 = config.sigma_vel * config.sigma_vel;
    double sc2 = config.sigma_clk * config.sigma_clk;
    double sd2 = config.sigma_drift * config.sigma_drift;
    // Position: sigma_pos^2 * dt^2
    Q[0 * 8 + 0] = sp2 * dt2;
    Q[1 * 8 + 1] = sp2 * dt2;
    Q[2 * 8 + 2] = sp2 * dt2;
    // Velocity: sigma_vel^2 * dt
    Q[3 * 8 + 3] = sv2 * dt;
    Q[4 * 8 + 4] = sv2 * dt;
    Q[5 * 8 + 5] = sv2 * dt;
    // Clock bias: sigma_clk^2 * dt^2
    Q[6 * 8 + 6] = sc2 * dt2;
    // Clock drift: sigma_drift^2 * dt
    Q[7 * 8 + 7] = sd2 * dt;

    // P = F*P*F^T + Q
    double FP[64];
    mat8_mul(F, state->P, FP);
    mat8_mul_bt(FP, F, state->P);
    mat8_add(state->P, Q);
}

void ekf_update(EKFState* state, const double* sat_ecef,
                const double* pseudoranges, const double* weights,
                int n_sat) {
    if (n_sat <= 0) return;

    // Build H matrix [n_sat x 8] and innovation vector y [n_sat]
    // Allocate dynamically for variable satellite count
    double* H = new double[n_sat * 8];
    double* y = new double[n_sat];
    double* R_diag = new double[n_sat];

    memset(H, 0, n_sat * 8 * sizeof(double));

    double rx = state->x[0], ry = state->x[1], rz = state->x[2];
    double cb = state->x[6];

    for (int s = 0; s < n_sat; s++) {
        double sx = sat_ecef[s * 3 + 0];
        double sy = sat_ecef[s * 3 + 1];
        double sz = sat_ecef[s * 3 + 2];

        double dx = rx - sx, dy = ry - sy, dz = rz - sz;
        double r = sqrt(dx * dx + dy * dy + dz * dz);

        if (r < 1e-6) { r = 1e-6; }

        double predicted_pr = r + cb;
        y[s] = pseudoranges[s] - predicted_pr;

        // H row: [-dx/r, -dy/r, -dz/r, 0, 0, 0, 1, 0]
        // Note: derivative of (r + cb) w.r.t. x is dx/r, but since
        // y = observed - predicted, H should be the Jacobian of h(x),
        // so H = [dx/r, dy/r, dz/r, 0, 0, 0, 1, 0]
        // and the update uses y - H*dx form. With standard EKF:
        // h(x) = ||x_pos - sat|| + cb
        // dh/dx = [dx/r, dy/r, dz/r, 0, 0, 0, 1, 0]
        H[s * 8 + 0] = dx / r;
        H[s * 8 + 1] = dy / r;
        H[s * 8 + 2] = dz / r;
        H[s * 8 + 6] = 1.0;

        // Measurement noise: 1/weight = sigma^2
        R_diag[s] = (weights[s] > 1e-15) ? 1.0 / weights[s] : 1e6;
    }

    // S = H * P * H^T + R  [n_sat x n_sat]
    // K = P * H^T * S^-1   [8 x n_sat]

    // Compute P * H^T  [8 x n_sat]
    double* PHt = new double[8 * n_sat];
    memset(PHt, 0, 8 * n_sat * sizeof(double));
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < n_sat; j++)
            for (int k = 0; k < 8; k++)
                PHt[i * n_sat + j] += state->P[i * 8 + k] * H[j * 8 + k];

    // S = H * PHt + R  [n_sat x n_sat]
    double* S = new double[n_sat * n_sat];
    memset(S, 0, n_sat * n_sat * sizeof(double));
    for (int i = 0; i < n_sat; i++) {
        for (int j = 0; j < n_sat; j++) {
            for (int k = 0; k < 8; k++)
                S[i * n_sat + j] += H[i * 8 + k] * PHt[k * n_sat + j];
        }
        S[i * n_sat + i] += R_diag[i];
    }

    // Compute K = PHt * S^-1 by solving S^T * K^T = PHt^T
    // Equivalently: solve S * X = PHt^T for X, then K = X^T
    // But simpler: solve S * K_col = PHt_col for each column of PHt
    // Actually, K = PHt * inv(S). We solve S * K^T_rows = PHt^T
    // Let's just solve: S * Z = I to get S^-1, then K = PHt * S^-1

    // For moderate n_sat (<30), direct inversion via solve is fine
    // Solve S * Sinv = I
    double* Sinv = new double[n_sat * n_sat];
    // Initialize Sinv as identity
    memset(Sinv, 0, n_sat * n_sat * sizeof(double));
    for (int i = 0; i < n_sat; i++) Sinv[i * n_sat + i] = 1.0;

    // S is destroyed by solve_linear, so copy
    double* S_copy = new double[n_sat * n_sat];
    memcpy(S_copy, S, n_sat * n_sat * sizeof(double));

    solve_linear(S_copy, Sinv, n_sat, n_sat);

    // K = PHt * Sinv  [8 x n_sat]
    double* K = new double[8 * n_sat];
    memset(K, 0, 8 * n_sat * sizeof(double));
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < n_sat; j++)
            for (int k = 0; k < n_sat; k++)
                K[i * n_sat + j] += PHt[i * n_sat + k] * Sinv[k * n_sat + j];

    // State update: x = x + K * y
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < n_sat; j++)
            state->x[i] += K[i * n_sat + j] * y[j];

    // Covariance update: P = (I - K*H) * P
    // Compute KH [8x8]
    double KH[64];
    memset(KH, 0, 64 * sizeof(double));
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            for (int k = 0; k < n_sat; k++)
                KH[i * 8 + j] += K[i * n_sat + k] * H[k * 8 + j];

    // IKH = I - KH
    double IKH[64];
    mat8_identity(IKH);
    for (int i = 0; i < 64; i++) IKH[i] -= KH[i];

    // P_new = IKH * P
    double P_new[64];
    mat8_mul(IKH, state->P, P_new);
    memcpy(state->P, P_new, 64 * sizeof(double));

    delete[] H;
    delete[] y;
    delete[] R_diag;
    delete[] PHt;
    delete[] S;
    delete[] S_copy;
    delete[] Sinv;
    delete[] K;
}

// ============================================================
// GPU batch EKF kernel
// ============================================================

// Device-side 8x8 matrix helpers
__device__ void d_mat8_zero(double* M) {
    for (int i = 0; i < 64; i++) M[i] = 0.0;
}

__device__ void d_mat8_identity(double* M) {
    d_mat8_zero(M);
    for (int i = 0; i < 8; i++) M[i * 8 + i] = 1.0;
}

__device__ void d_mat8_mul(const double* A, const double* B, double* C) {
    d_mat8_zero(C);
    for (int i = 0; i < 8; i++)
        for (int k = 0; k < 8; k++) {
            double a = A[i * 8 + k];
            for (int j = 0; j < 8; j++)
                C[i * 8 + j] += a * B[k * 8 + j];
        }
}

__device__ void d_mat8_mul_bt(const double* A, const double* B, double* C) {
    d_mat8_zero(C);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            double sum = 0;
            for (int k = 0; k < 8; k++)
                sum += A[i * 8 + k] * B[j * 8 + k];
            C[i * 8 + j] = sum;
        }
}

// Device-side small matrix inverse via Gauss-Jordan (n <= MAX_SAT)
// A: [n x n] input (destroyed), Ainv: [n x n] output
// Returns false if singular
__device__ bool d_mat_inv(double* A, double* Ainv, int n) {
    // Initialize Ainv = I
    for (int i = 0; i < n * n; i++) Ainv[i] = 0.0;
    for (int i = 0; i < n; i++) Ainv[i * n + i] = 1.0;

    for (int col = 0; col < n; col++) {
        int max_row = col;
        for (int row = col + 1; row < n; row++) {
            if (fabs(A[row * n + col]) > fabs(A[max_row * n + col]))
                max_row = row;
        }
        if (max_row != col) {
            for (int k = 0; k < n; k++) {
                double tmp = A[col * n + k]; A[col * n + k] = A[max_row * n + k]; A[max_row * n + k] = tmp;
                tmp = Ainv[col * n + k]; Ainv[col * n + k] = Ainv[max_row * n + k]; Ainv[max_row * n + k] = tmp;
            }
        }
        double pivot = A[col * n + col];
        if (fabs(pivot) < 1e-15) return false;
        for (int k = 0; k < n; k++) {
            A[col * n + k] /= pivot;
            Ainv[col * n + k] /= pivot;
        }
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double factor = A[row * n + col];
            for (int k = 0; k < n; k++) {
                A[row * n + k] -= factor * A[col * n + k];
                Ainv[row * n + k] -= factor * Ainv[col * n + k];
            }
        }
    }
    return true;
}

// Maximum number of satellites per epoch for GPU kernel (stack allocation)
#define EKF_MAX_SAT 32

__global__ void ekf_batch_kernel(EKFState* states, const double* sat_ecef,
                                  const double* pseudoranges, const double* weights,
                                  double dt, EKFConfig config,
                                  int n_instances, int n_sat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_instances) return;

    // Each instance shares the same satellite data
    EKFState& st = states[idx];

    // --- Predict ---
    double F[64];
    d_mat8_identity(F);
    F[0 * 8 + 3] = dt;
    F[1 * 8 + 4] = dt;
    F[2 * 8 + 5] = dt;
    F[6 * 8 + 7] = dt;

    double x_new[8] = {};
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            x_new[i] += F[i * 8 + j] * st.x[j];
    for (int i = 0; i < 8; i++) st.x[i] = x_new[i];

    double Q[64];
    d_mat8_zero(Q);
    double dt2 = dt * dt;
    Q[0 * 8 + 0] = config.sigma_pos * config.sigma_pos * dt2;
    Q[1 * 8 + 1] = config.sigma_pos * config.sigma_pos * dt2;
    Q[2 * 8 + 2] = config.sigma_pos * config.sigma_pos * dt2;
    Q[3 * 8 + 3] = config.sigma_vel * config.sigma_vel * dt;
    Q[4 * 8 + 4] = config.sigma_vel * config.sigma_vel * dt;
    Q[5 * 8 + 5] = config.sigma_vel * config.sigma_vel * dt;
    Q[6 * 8 + 6] = config.sigma_clk * config.sigma_clk * dt2;
    Q[7 * 8 + 7] = config.sigma_drift * config.sigma_drift * dt;

    double FP[64], P_new[64];
    d_mat8_mul(F, st.P, FP);
    d_mat8_mul_bt(FP, F, P_new);
    for (int i = 0; i < 64; i++) st.P[i] = P_new[i] + Q[i];

    // --- Update ---
    int ns = (n_sat < EKF_MAX_SAT) ? n_sat : EKF_MAX_SAT;
    if (ns <= 0) return;

    double H[EKF_MAX_SAT * 8];
    double y[EKF_MAX_SAT];
    double R_diag[EKF_MAX_SAT];

    for (int i = 0; i < ns * 8; i++) H[i] = 0.0;

    double rx = st.x[0], ry = st.x[1], rz = st.x[2], cb = st.x[6];

    for (int s = 0; s < ns; s++) {
        double sx = sat_ecef[s * 3 + 0];
        double sy = sat_ecef[s * 3 + 1];
        double sz = sat_ecef[s * 3 + 2];

        double dx = rx - sx, dy = ry - sy, dz = rz - sz;
        double r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1e-6) r = 1e-6;

        y[s] = pseudoranges[s] - (r + cb);

        H[s * 8 + 0] = dx / r;
        H[s * 8 + 1] = dy / r;
        H[s * 8 + 2] = dz / r;
        H[s * 8 + 6] = 1.0;

        R_diag[s] = (weights[s] > 1e-15) ? 1.0 / weights[s] : 1e6;
    }

    // PHt [8 x ns]
    double PHt[8 * EKF_MAX_SAT];
    for (int i = 0; i < 8 * ns; i++) PHt[i] = 0.0;
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < ns; j++)
            for (int k = 0; k < 8; k++)
                PHt[i * ns + j] += st.P[i * 8 + k] * H[j * 8 + k];

    // S = H * PHt + R [ns x ns]
    double S[EKF_MAX_SAT * EKF_MAX_SAT];
    for (int i = 0; i < ns * ns; i++) S[i] = 0.0;
    for (int i = 0; i < ns; i++) {
        for (int j = 0; j < ns; j++)
            for (int k = 0; k < 8; k++)
                S[i * ns + j] += H[i * 8 + k] * PHt[k * ns + j];
        S[i * ns + i] += R_diag[i];
    }

    // Invert S
    double Sinv[EKF_MAX_SAT * EKF_MAX_SAT];
    d_mat_inv(S, Sinv, ns);  // S is destroyed

    // K = PHt * Sinv [8 x ns]
    double K[8 * EKF_MAX_SAT];
    for (int i = 0; i < 8 * ns; i++) K[i] = 0.0;
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < ns; j++)
            for (int k = 0; k < ns; k++)
                K[i * ns + j] += PHt[i * ns + k] * Sinv[k * ns + j];

    // x += K * y
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < ns; j++)
            st.x[i] += K[i * ns + j] * y[j];

    // P = (I - K*H) * P
    double KH[64];
    for (int i = 0; i < 64; i++) KH[i] = 0.0;
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            for (int k = 0; k < ns; k++)
                KH[i * 8 + j] += K[i * ns + k] * H[k * 8 + j];

    double IKH[64];
    d_mat8_identity(IKH);
    for (int i = 0; i < 64; i++) IKH[i] -= KH[i];

    d_mat8_mul(IKH, st.P, P_new);
    for (int i = 0; i < 64; i++) st.P[i] = P_new[i];
}

void ekf_batch(EKFState* states, const double* sat_ecef,
               const double* pseudoranges, const double* weights,
               double dt, const EKFConfig& config,
               int n_instances, int n_sat) {
    size_t sz_states = (size_t)n_instances * sizeof(EKFState);
    size_t sz_sat = (size_t)n_sat * 3 * sizeof(double);
    size_t sz_pr = (size_t)n_sat * sizeof(double);

    EKFState* d_states;
    double *d_sat, *d_pr, *d_w;

    CUDA_CHECK(cudaMalloc(&d_states, sz_states));
    CUDA_CHECK(cudaMalloc(&d_sat, sz_sat));
    CUDA_CHECK(cudaMalloc(&d_pr, sz_pr));
    CUDA_CHECK(cudaMalloc(&d_w, sz_pr));

    CUDA_CHECK(cudaMemcpy(d_states, states, sz_states, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sat, sat_ecef, sz_sat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pr, pseudoranges, sz_pr, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, weights, sz_pr, cudaMemcpyHostToDevice));

    int block = 128;
    int grid = (n_instances + block - 1) / block;
    ekf_batch_kernel<<<grid, block>>>(d_states, d_sat, d_pr, d_w,
                                       dt, config, n_instances, n_sat);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaMemcpy(states, d_states, sz_states, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_sat));
    CUDA_CHECK(cudaFree(d_pr));
    CUDA_CHECK(cudaFree(d_w));
}

}  // namespace gnss_gpu
