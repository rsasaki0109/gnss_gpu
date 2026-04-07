#pragma once

#include <cstdint>

namespace gnss_gpu {

// Batch GNSS factor-graph optimization (iterated Gauss–Newton; optional
// backtracking line search; dense Cholesky on the host) with GPU assembly of
// normal equations. Pseudorange factor follows PseudorangeFactor_XC in
// gtsam_gnss: prediction = rho + h_c^T c with h_c(0)=1 and h_c(sys)=1 (ISB /
// multi-clock). Optional position random walk between epochs.
//
// Per-epoch state layout: [x, y, z, c0, c1, ..., c(nc-1)] with nc = n_clock (1..4).
// sys_kind[m] in [0, nc) per measurement m = t*n_sat+s; sk=0 uses clock c0 only
// (after filling rule: h[0]=1; if sk>0 and sk<nc then h[sk]=1). Matches
// gtsam_gnss PseudorangeFactor_XC (receiver clock + inter-system on non-zero sk).
//
// n_state = (3 + n_clock) * n_epoch (limit 8192).
//
// huber_k: Huber threshold on Mahalanobis residual z = |sqrt(w)*res|; <=0 disables
// robust reweighting (pure WLS). When enabled, each GN iteration uses IRLS weights
// w_eff = w * min(1, huber_k / z) for the linearized normal equations; PR cost for
// line search follows the Huber loss.
//
// enable_line_search: if non-zero, backtracking on the GN step to reduce total cost.
//
// Returns Gauss–Newton iterations completed on success, or -1 on failure.
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
                double* out_mse_pr);

}  // namespace gnss_gpu
