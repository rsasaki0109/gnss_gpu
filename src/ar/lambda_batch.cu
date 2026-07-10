// Batched LAMBDA / MLAMBDA integer least-squares, one GPU thread per
// (ahat, Qahat) problem. Faithful transcription of cssrlib's mlambda.py
// (see include/gnss_gpu/lambda_batch.h for the parity contract).
//
// IMPORTANT: this file must be compiled with --fmad=false so that the
// floating-point operation order matches the numba njit CPU reference
// (no FMA contraction). See CMakeLists.txt.

#include "gnss_gpu/cuda_check.h"
#include "gnss_gpu/lambda_batch.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace gnss_gpu {

namespace {

constexpr int MAX_N = 64;
constexpr int MAX_CANDS = 16;  // WP17: 8 -> 16 (RB-FGO-PF top-K uses 12)
constexpr int LOOPMAX = 10000;  // cssrlib mlambda.py LOOPMAX (RTKLIB)

// Per-problem double workspace layout (all row-major, lda = n unless
// noted). Sized for MAX_N.
//   A     : MAX_N*MAX_N   (ldldecom working copy of Q)
//   L     : MAX_N*MAX_N
//   Z     : MAX_N*MAX_N
//   Qz    : MAX_N*MAX_N   (Qzhat)
//   iZt   : MAX_N*MAX_N
//   AUG   : MAX_N*2*MAX_N (Gauss-Jordan augmented scratch)
//   S     : MAX_N*MAX_N   (estimILS spine)
//   vecs  : d, zhat, dist, acond, left, QP, zfix_head scratch etc.
constexpr int WS_MAT = MAX_N * MAX_N;
constexpr int WS_DBL_PER_PROB =
    6 * WS_MAT + 2 * WS_MAT /* AUG */ + 16 * MAX_N +
    2 * MAX_N * MAX_CANDS;
constexpr int WS_INT_PER_PROB = 4 * MAX_N;

__device__ inline int d_round_to_int(double v) {
  // cssrlib _round_to_int: int(np.rint(v)) -- round half to even.
  return static_cast<int>(rint(v));
}

__device__ inline int d_signed_step(double v) {
  if (v > 0.0) return 1;
  if (v < 0.0) return -1;
  return 0;
}

__device__ inline int d_signed_step_i(int v) {
  if (v > 0) return 1;
  if (v < 0) return -1;
  return 0;
}

// cssrlib _sr_boost over d[i0 .. n-1].
__device__ double d_sr_boost(const double* d, int i0, int n) {
  const double inv_sqrt2 = 1.0 / sqrt(2.0);
  double prod = 1.0;
  for (int i = i0; i < n; i++) {
    double val = 0.5 / sqrt(d[i]);
    double cdf = 0.5 * (1.0 + erf(val * inv_sqrt2));
    prod *= 2.0 * cdf - 1.0;
  }
  return prod;
}

// cssrlib _ldldecom: Q (n x n, row-major) -> L (unit lower via reversed
// LDL), d. A is an n*n working buffer. Returns true when every
// d[i] >= 1e-10 (the ldldecom() wrapper raises LambdaError otherwise).
__device__ bool d_ldldecom(const double* Q, double* L, double* d,
                           double* A, int n) {
  for (int i = 0; i < n * n; i++) {
    A[i] = Q[i];
    L[i] = 0.0;
  }
  for (int i = 0; i < n; i++) d[i] = 0.0;
  for (int i = n - 1; i >= 0; i--) {
    d[i] = A[i * n + i];
    if (d[i] <= 0.0) continue;
    double sq = sqrt(d[i]);
    for (int c = 0; c <= i; c++) L[i * n + c] = A[i * n + c] / sq;
    for (int j = 0; j < i; j++) {
      double lij = L[i * n + j];
      for (int c = 0; c <= j; c++) A[j * n + c] -= L[i * n + c] * lij;
    }
    double lii = L[i * n + i];
    for (int c = 0; c <= i; c++) L[i * n + c] /= lii;
  }
  for (int i = 0; i < n; i++) {
    if (d[i] < 1e-10) return false;
  }
  return true;
}

// cssrlib _reduction (LLL / decorrelation). L, d in place; Z out (eye
// on entry is set here).
__device__ void d_reduction(double* L, double* d, double* Z, int n) {
  for (int r = 0; r < n; r++)
    for (int c = 0; c < n; c++) Z[r * n + c] = (r == c) ? 1.0 : 0.0;
  int j = n - 2;
  int k = n - 2;
  int loops = 0;
  while (j >= 0 && loops < LOOPMAX) {
    loops++;
    if (j <= k) {
      for (int i = j + 1; i < n; i++) {
        double mu = rint(L[i * n + j]);
        if (mu != 0.0) {
          for (int r = i; r < n; r++) L[r * n + j] -= mu * L[r * n + i];
          for (int r = 0; r < n; r++) Z[r * n + j] -= mu * Z[r * n + i];
        }
      }
    }
    double lj1j = L[(j + 1) * n + j];
    double delta = d[j] + lj1j * lj1j * d[j + 1];
    if (delta + 1e-6 < d[j + 1]) {
      double eta = d[j] / delta;
      double lam = d[j + 1] * lj1j / delta;
      d[j] = eta * d[j + 1];
      d[j + 1] = delta;
      if (j > 0) {
        for (int col = 0; col < j; col++) {
          double Lj_col = L[j * n + col];
          double Lj1_col = L[(j + 1) * n + col];
          double t0 = -lj1j * Lj_col + Lj1_col;
          double t1 = eta * Lj_col + lam * Lj1_col;
          L[j * n + col] = t0;
          L[(j + 1) * n + col] = t1;
        }
      }
      L[(j + 1) * n + j] = lam;
      for (int row = j + 2; row < n; row++) {
        double tmp = L[row * n + j];
        L[row * n + j] = L[row * n + j + 1];
        L[row * n + j + 1] = tmp;
      }
      for (int row = 0; row < n; row++) {
        double tmp = Z[row * n + j];
        Z[row * n + j] = Z[row * n + j + 1];
        Z[row * n + j + 1] = tmp;
      }
      k = j;
      j = n - 2;
    } else {
      j -= 1;
    }
  }
}

// Stable ascending insertion argsort (matches np.argsort on the tiny
// ncands arrays used here, including ties).
__device__ void d_stable_argsort(const double* v, int m, int* order) {
  for (int i = 0; i < m; i++) order[i] = i;
  for (int i = 1; i < m; i++) {
    int oi = order[i];
    double key = v[oi];
    int j = i - 1;
    while (j >= 0 && v[order[j]] > key) {
      order[j + 1] = order[j];
      j--;
    }
    order[j + 1] = oi;
  }
}

// cssrlib _estimILS (search-and-shrink ILS). Operates on the trailing
// (sub) problem of size n with L addressed at stride lda starting at
// (off, off), d/ahat at offset off. Outputs afixed (n x ncands,
// row-major) and sqnorm (ncands), both SORTED ascending by sqnorm
// exactly like the reference.
// Scratch: S (n*n), dist/acond/left (n dbl), zcond/step/path (n int),
// af_tmp (n*ncands), sq_tmp (ncands), order (ncands int).
__device__ void d_estimILS(const double* Lb, int lda, int off,
                           const double* db, const double* ahatb,
                           int n, int ncands,
                           double* afixed, double* sqnorm,
                           double* S, double* dist, double* acond,
                           double* left, int* zcond, int* step,
                           int* path, double* af_tmp, double* sq_tmp,
                           int* order) {
  const double* d = db + off;
  const double* ahat = ahatb + off;
#define LSUB(r, c) Lb[((r) + off) * lda + ((c) + off)]

  double Chi2 = 1e18;
  int loop_count = 0;
  bool aborted = false;

  int k0 = (ncands == 1 && n > 1) ? 1 : 0;

  for (int i = 0; i < n * ncands; i++) af_tmp[i] = 0.0;
  for (int i = 0; i < ncands; i++) sq_tmp[i] = 1e18;
  for (int i = 0; i < n; i++) {
    dist[i] = 0.0;
    acond[i] = 0.0;
    left[i] = 0.0;
    zcond[i] = 0;
    step[i] = 0;
    path[i] = n - 1;
  }
  for (int i = 0; i < n * n; i++) S[i] = 0.0;

  acond[n - 1] = ahat[n - 1];
  zcond[n - 1] = d_round_to_int(acond[n - 1]);
  left[n - 1] = acond[n - 1] - zcond[n - 1];
  step[n - 1] = d_signed_step(left[n - 1]);
  if (step[n - 1] == 0) step[n - 1] = 1;

  int count = -1;
  int imax = ncands - 1;
  bool endSearch = false;
  int k = n - 1;

  while (!endSearch) {
    double newdist = dist[k] + left[k] * left[k] / d[k];
    while (newdist < Chi2) {
      loop_count++;
      if (loop_count > LOOPMAX) {
        aborted = true;
        endSearch = true;
        break;
      }
      if (k != 0) {
        k -= 1;
        dist[k] = newdist;
        for (int j = path[k]; j > k; j--) {
          S[(j - 1) * n + k] = S[j * n + k] - left[j] * LSUB(j, k);
        }
        acond[k] = ahat[k] + S[k * n + k];
        zcond[k] = d_round_to_int(acond[k]);
        left[k] = acond[k] - zcond[k];
        step[k] = d_signed_step(left[k]);
        if (step[k] == 0) step[k] = 1;
      } else {
        if (count < ncands - 2) {
          count += 1;
          for (int i = 0; i < n; i++)
            af_tmp[i * ncands + count] = static_cast<double>(zcond[i]);
          sq_tmp[count] = newdist;
        } else {
          for (int i = 0; i < n; i++)
            af_tmp[i * ncands + imax] = static_cast<double>(zcond[i]);
          sq_tmp[imax] = newdist;
          // np.argmax: first index of the maximum.
          imax = 0;
          double vmax = sq_tmp[0];
          for (int i = 1; i < ncands; i++) {
            if (sq_tmp[i] > vmax) {
              vmax = sq_tmp[i];
              imax = i;
            }
          }
          Chi2 = sq_tmp[imax];
        }
        k = k0;
        zcond[k] += step[k];
        left[k] = acond[k] - zcond[k];
        step[k] = -step[k] - d_signed_step_i(step[k]);
      }
      newdist = dist[k] + left[k] * left[k] / d[k];
    }

    if (aborted) break;

    int ilevel = k;

    while (newdist >= Chi2) {
      loop_count++;
      if (loop_count > LOOPMAX) {
        aborted = true;
        endSearch = true;
        break;
      }
      if (k == n - 1) {
        endSearch = true;
        break;
      }
      k += 1;
      zcond[k] += step[k];
      left[k] = acond[k] - zcond[k];
      step[k] = -step[k] - d_signed_step_i(step[k]);
      newdist = dist[k] + left[k] * left[k] / d[k];
    }

    if (aborted) break;

    for (int i = ilevel; i < k; i++) path[i] = k;
    for (int j = ilevel - 1; j >= 0; j--) {
      if (path[j] < k) {
        path[j] = k;
      } else {
        break;
      }
    }
  }

  d_stable_argsort(sq_tmp, ncands, order);
  for (int idx = 0; idx < ncands; idx++) {
    sqnorm[idx] = sq_tmp[order[idx]];
    for (int i = 0; i < n; i++)
      afixed[i * ncands + idx] = af_tmp[i * ncands + order[idx]];
  }
#undef LSUB
}

// Gauss-Jordan inverse with partial pivoting (used for iZt = inv(Z.T)
// and the PAR k=1 cross-adjustment). AUG is m x 2m scratch. Returns
// false when a pivot vanishes (singular).
__device__ bool d_inv(const double* M, int m, double* out, double* AUG) {
  for (int r = 0; r < m; r++) {
    for (int c = 0; c < m; c++) {
      AUG[r * 2 * m + c] = M[r * m + c];
      AUG[r * 2 * m + m + c] = (r == c) ? 1.0 : 0.0;
    }
  }
  for (int col = 0; col < m; col++) {
    int piv = col;
    double vmax = fabs(AUG[col * 2 * m + col]);
    for (int r = col + 1; r < m; r++) {
      double v = fabs(AUG[r * 2 * m + col]);
      if (v > vmax) {
        vmax = v;
        piv = r;
      }
    }
    if (vmax == 0.0) return false;
    if (piv != col) {
      for (int c = 0; c < 2 * m; c++) {
        double tmp = AUG[col * 2 * m + c];
        AUG[col * 2 * m + c] = AUG[piv * 2 * m + c];
        AUG[piv * 2 * m + c] = tmp;
      }
    }
    double pivot = AUG[col * 2 * m + col];
    for (int c = 0; c < 2 * m; c++) AUG[col * 2 * m + c] /= pivot;
    for (int r = 0; r < m; r++) {
      if (r == col) continue;
      double f = AUG[r * 2 * m + col];
      if (f == 0.0) continue;
      for (int c = 0; c < 2 * m; c++)
        AUG[r * 2 * m + c] -= f * AUG[col * 2 * m + c];
    }
  }
  for (int r = 0; r < m; r++)
    for (int c = 0; c < m; c++) out[r * m + c] = AUG[r * 2 * m + m + c];
  return true;
}

__global__ void lambda_batch_kernel(
    const double* __restrict__ ahat_flat,
    const double* __restrict__ Q_flat,
    const int* __restrict__ n_arr,
    const long long* __restrict__ a_off,
    const long long* __restrict__ q_off,
    int n_prob, int ncands, int parmode, double P0,
    double* __restrict__ afix_flat,
    double* __restrict__ zfix_flat,
    double* __restrict__ s_out,
    int* __restrict__ s_len_out,
    int* __restrict__ nfix_out,
    double* __restrict__ ps_out,
    int* __restrict__ status_out,
    double* __restrict__ ws_dbl,
    int* __restrict__ ws_int) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_prob) return;

  const int n = n_arr[p];
  const double* ahat = ahat_flat + a_off[p];
  const double* Q = Q_flat + q_off[p];
  double* afix = afix_flat + a_off[p] * ncands;
  double* zfix = zfix_flat + a_off[p] * ncands;
  double* s = s_out + static_cast<long long>(p) * ncands;
  for (int i = 0; i < ncands; i++) s[i] = 0.0;
  s_len_out[p] = 0;
  nfix_out[p] = 0;
  ps_out[p] = nan("");

  if (n < 1 || n > MAX_N || ncands < 1 || ncands > MAX_CANDS ||
      (parmode != 1 && parmode != 2)) {
    status_out[p] = 2;
    return;
  }

  double* w = ws_dbl + static_cast<long long>(p) * WS_DBL_PER_PROB;
  int* wi = ws_int + static_cast<long long>(p) * WS_INT_PER_PROB;
  double* A = w;               // n*n
  double* L = A + WS_MAT;      // n*n
  double* Z = L + WS_MAT;      // n*n
  double* Qz = Z + WS_MAT;     // n*n (Qzhat)
  double* iZt = Qz + WS_MAT;   // n*n
  double* S = iZt + WS_MAT;    // n*n (estimILS spine)
  double* AUG = S + WS_MAT;    // n*2n scratch
  double* vec = AUG + 2 * WS_MAT;
  double* d = vec;             // n
  double* zhat = d + MAX_N;    // n
  double* dist = zhat + MAX_N;
  double* acond = dist + MAX_N;
  double* left = acond + MAX_N;
  double* QP = left + MAX_N;       // n (k=1 row)
  double* dz = QP + MAX_N;         // n scratch (zhat[k:] - zpar)
  double* sq_tmp = dz + MAX_N;     // MAX_CANDS (<= MAX_N)
  double* af_tmp = sq_tmp + MAX_N;         // n*ncands
  double* zp = af_tmp + MAX_N * MAX_CANDS; // n*ncands (zpar)
  int* zcond = wi;
  int* step = zcond + MAX_N;
  int* path = step + MAX_N;
  int* order = path + MAX_N;

  // ldldecom (LambdaError -> status 1, matching the CPU reject path).
  if (!d_ldldecom(Q, L, d, A, n)) {
    status_out[p] = 1;
    return;
  }

  // reduction.
  d_reduction(L, d, Z, n);

  // iZt = np.round(inv(Z.T)). Z entries are integer-valued doubles.
  // A is free now -- reuse it to hold Z.T.
  for (int r = 0; r < n; r++)
    for (int c = 0; c < n; c++) A[r * n + c] = Z[c * n + r];
  if (!d_inv(A, n, iZt, AUG)) {
    status_out[p] = 1;  // unimodular Z cannot be singular in practice
    return;
  }
  for (int i = 0; i < n * n; i++) iZt[i] = rint(iZt[i]);

  // zhat = Z.T @ ahat
  for (int i = 0; i < n; i++) {
    double acc = 0.0;
    for (int j = 0; j < n; j++) acc += Z[j * n + i] * ahat[j];
    zhat[i] = acc;
  }

  // Qzhat = L.T @ diag(d) @ L  (reuse A for diag(d) @ L)
  for (int r = 0; r < n; r++)
    for (int c = 0; c < n; c++) A[r * n + c] = d[r] * L[r * n + c];
  for (int r = 0; r < n; r++) {
    for (int c = 0; c < n; c++) {
      double acc = 0.0;
      for (int m = 0; m < n; m++) acc += L[m * n + r] * A[m * n + c];
      Qz[r * n + c] = acc;
    }
  }

  double Ps = d_sr_boost(d, 0, n);
  int nfix = 0;

  if (parmode == 1) {
    // zfix, s = estimILS(L, d, zhat, ncands); nfix = n
    d_estimILS(L, n, 0, d, zhat, n, ncands, zfix, s, S, dist, acond,
               left, zcond, step, path, af_tmp, sq_tmp, order);
    s_len_out[p] = ncands;
    nfix = n;
  } else {
    // parsearch (exclmax = 1, as called by cssrlib mlambda()).
    const int exclmax = 1;
    int k = 0;
    while (Ps < P0 && k < n - 1) {
      k += 1;
      Ps = d_sr_boost(d, k, n);
    }
    if (k <= exclmax && Ps > P0) {
      const int m = n - k;
      // zpar, sqnorm = estimILS(L[k:,k:], d[k:], zhat[k:], ncands)
      d_estimILS(L, n, k, d, zhat, m, ncands, zp, s, S, dist, acond,
                 left, zcond, step, path, af_tmp, sq_tmp, order);
      s_len_out[p] = ncands;
      if (k > 0) {
        // QP = Qzhat[:k, k:] @ inv(Qzhat[k:, k:])   (k == 1 here)
        // Reuse A for the m x m submatrix and S for its inverse.
        for (int r = 0; r < m; r++)
          for (int c = 0; c < m; c++)
            A[r * m + c] = Qz[(r + k) * n + (c + k)];
        if (!d_inv(A, m, S, AUG)) {
          status_out[p] = 1;  // matches np.linalg.inv LinAlgError
          return;
        }
        for (int c = 0; c < m; c++) {
          double acc = 0.0;
          for (int j = 0; j < m; j++)
            acc += Qz[0 * n + (j + k)] * S[j * m + c];
          QP[c] = acc;
        }
        for (int cand = 0; cand < ncands; cand++) {
          for (int j = 0; j < m; j++)
            dz[j] = zhat[k + j] - zp[j * ncands + cand];
          double acc = 0.0;
          for (int j = 0; j < m; j++) acc += QP[j] * dz[j];
          zfix[0 * ncands + cand] = zhat[0] - acc;
          for (int j = 0; j < m; j++)
            zfix[(k + j) * ncands + cand] = zp[j * ncands + cand];
        }
      } else {
        for (int i = 0; i < m * ncands; i++) zfix[i] = zp[i];
      }
      nfix = n - k;
    } else {
      // Rejection: cssrlib returns zfix = zhat (1-D), s = [], Ps = NaN.
      Ps = nan("");
      nfix = 0;
      s_len_out[p] = 0;
      for (int i = 0; i < n; i++) {
        for (int c = 0; c < ncands; c++)
          zfix[i * ncands + c] = (c == 0) ? zhat[i] : 0.0;
      }
    }
  }

  // afix = iZt @ zfix (column-wise)
  for (int i = 0; i < n; i++) {
    for (int c = 0; c < ncands; c++) {
      double acc = 0.0;
      for (int m = 0; m < n; m++) acc += iZt[i * n + m] * zfix[m * ncands + c];
      afix[i * ncands + c] = acc;
    }
  }

  nfix_out[p] = nfix;
  ps_out[p] = Ps;
  status_out[p] = 0;
}

// Growable cached device + pinned-host buffers (calls are serialized by
// the Python GIL; no concurrency expected). Everything for one launch
// is suballocated from four slabs so a launch costs 2 H2D + 2 D2H
// copies and ZERO cudaMalloc/cudaFree -- the per-launch overhead
// matters because the pipeline's cascade batches are small (10-30
// problems).
struct DeviceCache {
  double* dbl = nullptr;      // device workspace slab
  int* i32 = nullptr;         // device int workspace slab
  double* io_dbl = nullptr;   // device I/O doubles slab
  long long* io_i64 = nullptr;
  int* io_i32 = nullptr;
  double* h_dbl = nullptr;    // pinned host staging
  long long* h_i64 = nullptr;
  int* h_i32 = nullptr;
  size_t dbl_cap = 0, i32_cap = 0, io_dbl_cap = 0, io_i64_cap = 0,
         io_i32_cap = 0, h_dbl_cap = 0, h_i64_cap = 0, h_i32_cap = 0;

  static void grow_dev_dbl(double** p, size_t* cap, size_t n) {
    if (n > *cap) {
      if (*p) cudaFree(*p);
      CUDA_CHECK(cudaMalloc(p, n * sizeof(double)));
      *cap = n;
    }
  }
  void ensure(size_t nd_ws, size_t ni_ws, size_t nd_io, size_t ni64_io,
              size_t ni32_io) {
    grow_dev_dbl(&dbl, &dbl_cap, nd_ws);
    if (ni_ws > i32_cap) {
      if (i32) cudaFree(i32);
      CUDA_CHECK(cudaMalloc(&i32, ni_ws * sizeof(int)));
      i32_cap = ni_ws;
    }
    grow_dev_dbl(&io_dbl, &io_dbl_cap, nd_io);
    if (ni64_io > io_i64_cap) {
      if (io_i64) cudaFree(io_i64);
      CUDA_CHECK(cudaMalloc(&io_i64, ni64_io * sizeof(long long)));
      io_i64_cap = ni64_io;
    }
    if (ni32_io > io_i32_cap) {
      if (io_i32) cudaFree(io_i32);
      CUDA_CHECK(cudaMalloc(&io_i32, ni32_io * sizeof(int)));
      io_i32_cap = ni32_io;
    }
    if (nd_io > h_dbl_cap) {
      if (h_dbl) cudaFreeHost(h_dbl);
      CUDA_CHECK(cudaMallocHost(&h_dbl, nd_io * sizeof(double)));
      h_dbl_cap = nd_io;
    }
    if (ni64_io > h_i64_cap) {
      if (h_i64) cudaFreeHost(h_i64);
      CUDA_CHECK(cudaMallocHost(&h_i64, ni64_io * sizeof(long long)));
      h_i64_cap = ni64_io;
    }
    if (ni32_io > h_i32_cap) {
      if (h_i32) cudaFreeHost(h_i32);
      CUDA_CHECK(cudaMallocHost(&h_i32, ni32_io * sizeof(int)));
      h_i32_cap = ni32_io;
    }
  }
};

DeviceCache g_ws;

}  // namespace

int lambda_batch_max_n() { return MAX_N; }

void lambda_batch(const double* ahat_flat, const double* Q_flat,
                  const int* n_arr, int n_prob,
                  int ncands, int parmode, double P0,
                  double* afix_flat, double* zfix_flat,
                  double* s_out, int* s_len_out, int* nfix_out,
                  double* ps_out, int* status_out) {
  if (n_prob <= 0) return;

  std::vector<long long> a_off(n_prob), q_off(n_prob);
  long long atot = 0, qtot = 0;
  for (int p = 0; p < n_prob; p++) {
    a_off[p] = atot;
    q_off[p] = qtot;
    atot += n_arr[p];
    qtot += static_cast<long long>(n_arr[p]) * n_arr[p];
  }

  const size_t ws_dbl_n = static_cast<size_t>(n_prob) * WS_DBL_PER_PROB;
  const size_t ws_int_n = static_cast<size_t>(n_prob) * WS_INT_PER_PROB;

  // Single-slab I/O layout (suballocated; 2 H2D + 2 D2H copies total).
  // doubles in : [ahat(atot) | Q(qtot)]
  // doubles out: [afix(atot*nc) | zfix(atot*nc) | s(np*nc) | ps(np)]
  // i64 in     : [a_off(np) | q_off(np)]
  // i32 in/out : [n(np)] / [s_len(np) | nfix(np) | status(np)]
  const size_t in_dbl_n = static_cast<size_t>(atot) + qtot;
  const size_t out_dbl_n = 2 * static_cast<size_t>(atot) * ncands +
                           static_cast<size_t>(n_prob) * ncands + n_prob;
  const size_t io_dbl_n = in_dbl_n > out_dbl_n ? in_dbl_n : out_dbl_n;
  const size_t io_i64_n = 2 * static_cast<size_t>(n_prob);
  const size_t io_i32_n = 4 * static_cast<size_t>(n_prob);

  // in and out double regions live in the SAME device slab but must not
  // overlap (the kernel reads inputs while writing outputs), so size the
  // slab for both and place outputs after inputs. (io_dbl_n is the
  // pinned-host staging size: staging is reused for in then out.)
  (void)io_dbl_n;
  g_ws.ensure(ws_dbl_n, ws_int_n, in_dbl_n + out_dbl_n, io_i64_n,
              io_i32_n);

  double* d_in = g_ws.io_dbl;                    // ahat | Q
  double* d_out = g_ws.io_dbl + in_dbl_n;        // afix | zfix | s | ps
  double* d_ahat = d_in;
  double* d_Q = d_in + atot;
  double* d_afix = d_out;
  double* d_zfix = d_afix + static_cast<size_t>(atot) * ncands;
  double* d_s = d_zfix + static_cast<size_t>(atot) * ncands;
  double* d_ps = d_s + static_cast<size_t>(n_prob) * ncands;
  long long* d_aoff = g_ws.io_i64;
  long long* d_qoff = g_ws.io_i64 + n_prob;
  int* d_n = g_ws.io_i32;                              // input
  int* d_slen = g_ws.io_i32 + n_prob;                  // outputs follow
  int* d_nfix = g_ws.io_i32 + 2 * static_cast<size_t>(n_prob);
  int* d_status = g_ws.io_i32 + 3 * static_cast<size_t>(n_prob);

  // Stage inputs in pinned memory, 3 async H2D copies on the default
  // stream (i64 + i32 are tiny).
  std::memcpy(g_ws.h_dbl, ahat_flat, atot * sizeof(double));
  std::memcpy(g_ws.h_dbl + atot, Q_flat, qtot * sizeof(double));
  std::memcpy(g_ws.h_i64, a_off.data(), n_prob * sizeof(long long));
  std::memcpy(g_ws.h_i64 + n_prob, q_off.data(),
              n_prob * sizeof(long long));
  std::memcpy(g_ws.h_i32, n_arr, n_prob * sizeof(int));
  CUDA_CHECK(cudaMemcpyAsync(d_in, g_ws.h_dbl, in_dbl_n * sizeof(double),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(g_ws.io_i64, g_ws.h_i64,
                             io_i64_n * sizeof(long long),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(g_ws.io_i32, g_ws.h_i32,
                             n_prob * sizeof(int),
                             cudaMemcpyHostToDevice));

  const int block = 32;
  const int grid = (n_prob + block - 1) / block;
  lambda_batch_kernel<<<grid, block>>>(
      d_ahat, d_Q, d_n, d_aoff, d_qoff, n_prob, ncands, parmode, P0,
      d_afix, d_zfix, d_s, d_slen, d_nfix, d_ps, d_status, g_ws.dbl,
      g_ws.i32);
  CUDA_CHECK_LAST();

  // 2 D2H copies into pinned staging; the blocking cudaMemcpy provides
  // the synchronization (no explicit cudaDeviceSynchronize).
  CUDA_CHECK(cudaMemcpy(g_ws.h_dbl, d_out, out_dbl_n * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(g_ws.h_i32, g_ws.io_i32 + n_prob,
                        3 * static_cast<size_t>(n_prob) * sizeof(int),
                        cudaMemcpyDeviceToHost));

  const double* h_afix = g_ws.h_dbl;
  const double* h_zfix = h_afix + static_cast<size_t>(atot) * ncands;
  const double* h_s = h_zfix + static_cast<size_t>(atot) * ncands;
  const double* h_ps = h_s + static_cast<size_t>(n_prob) * ncands;
  std::memcpy(afix_flat, h_afix,
              static_cast<size_t>(atot) * ncands * sizeof(double));
  std::memcpy(zfix_flat, h_zfix,
              static_cast<size_t>(atot) * ncands * sizeof(double));
  std::memcpy(s_out, h_s,
              static_cast<size_t>(n_prob) * ncands * sizeof(double));
  std::memcpy(ps_out, h_ps, n_prob * sizeof(double));
  std::memcpy(s_len_out, g_ws.h_i32, n_prob * sizeof(int));
  std::memcpy(nfix_out, g_ws.h_i32 + n_prob, n_prob * sizeof(int));
  std::memcpy(status_out, g_ws.h_i32 + 2 * static_cast<size_t>(n_prob),
              n_prob * sizeof(int));
}  // NOLINT(readability/fn_size)

}  // namespace gnss_gpu
