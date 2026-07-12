#pragma once

namespace gnss_gpu {

// Batched LAMBDA / MLAMBDA integer least-squares ambiguity resolution.
//
// Faithful port of cssrlib's mlambda.py (ldldecom -> LLL reduction ->
// estimILS search, plus the parmode=2 PAR path via parsearch with
// exclmax=1), evaluated for MANY independent (ahat, Qahat) problems in a
// single kernel launch. Designed for the tc/-style subset/retry AR
// cascade, where one failing epoch fires 10-30 sequential LAMBDA calls
// on slightly different candidate subsets.
//
// Numerical-parity notes: the device code transcribes cssrlib's
// operation ORDER exactly and the translation unit is compiled with
// --fmad=false, so the LDL factorization, the LLL reduction and the
// integer search make bit-identical rounding decisions to the CPU
// (numba njit) reference on well-scaled inputs; the small matrix
// products surrounding them (zhat, Qzhat, iZt, the k=1 PAR
// cross-adjustment) use naive ascending-index accumulation, which can
// differ from BLAS by ~1 ulp -- integer outputs are unaffected unless a
// conditional float lands exactly on a rounding boundary. Parity is
// verified empirically against captured pipeline inputs in
// tests/test_lambda_batch.py.
//
// Inputs (flattened, problem i has n_i = n_arr[i] ambiguities):
//   ahat_flat: concatenated float ambiguity vectors, sum(n_i)
//   Q_flat:    concatenated row-major n_i x n_i covariances, sum(n_i^2)
//   n_arr:     [n_prob]
//   ncands:    number of integer candidates (cssrlib default 2)
//   parmode:   1 = full ILS (estimILS), 2 = PAR (parsearch, exclmax=1)
//   P0:        PAR bootstrapped success-rate threshold
// Outputs (flattened with the same offsets; caller allocates):
//   afix_flat: sum(n_i)*ncands, row-major (n_i, ncands) per problem --
//              iZt @ zfix, matching cssrlib mlambda()'s afix_. On a
//              parmode=2 rejection only column 0 is meaningful
//              (cssrlib returns the 1-D float vector there).
//   zfix_flat: sum(n_i)*ncands, row-major (n_i, ncands) per problem --
//              the decorrelated-domain solution BEFORE the iZt
//              back-transform; rows k..n-1 are exact integers.
//   s_out:     [n_prob*ncands] candidate squared norms (cssrlib s);
//              zero-filled when s_len_out[i] == 0
//   s_len_out: [n_prob] length of s (0 on a parmode=2 rejection,
//              matching cssrlib's empty list)
//   nfix_out:  [n_prob] number of fixed ambiguities (n, n-k, or 0)
//   ps_out:    [n_prob] bootstrapped success rate (NaN on PAR reject)
//   status_out:[n_prob] 0 = ok; 1 = non-positive-definite covariance
//              (cssrlib raises LambdaError); 2 = unsupported dims
// Throws std::runtime_error on CUDA failures.
void lambda_batch(const double* ahat_flat, const double* Q_flat,
                  const int* n_arr, int n_prob,
                  int ncands, int parmode, double P0,
                  double* afix_flat, double* zfix_flat,
                  double* s_out, int* s_len_out, int* nfix_out,
                  double* ps_out, int* status_out);

// Maximum per-problem ambiguity count supported by the kernel.
int lambda_batch_max_n();

}  // namespace gnss_gpu
