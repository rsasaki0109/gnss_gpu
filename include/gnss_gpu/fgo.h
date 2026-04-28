#pragma once

#include <cstdint>

namespace gnss_gpu {

// Batch GNSS factor-graph optimization (iterated Gauss–Newton; optional
// backtracking line search; dense Cholesky on the host) with GPU assembly of
// normal equations. Pseudorange factor follows PseudorangeFactor_XC in
// gtsam_gnss: prediction = rho + h_c^T c with h_c(0)=1 and h_c(sys)=1 (ISB /
// multi-clock). Optional position random walk between epochs.
//
// Per-epoch state layout: [x, y, z, c0, c1, ..., c(nc-1)] with nc = n_clock (1..7).
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
                double* out_mse_pr,
                const double* motion_displacement = nullptr,
                const double* tdcp_meas = nullptr,
                const double* tdcp_weights = nullptr,
                double tdcp_sigma_m = 0.0);

// Extended FGO with velocity state + clock drift + Doppler factor.
//
// Per-epoch state layout:
//   [x, y, z, vx, vy, vz, c0, ..., c_{nc-1}, drift]
//   ss = 3 + 3 + n_clock + 1 = 7 + n_clock
//
// Pseudorange factor constrains position + clock (same model as fgo_gnss_lm).
// Motion factor: x_{t+1} ≈ x_t + v_t * dt (position-velocity coupling).
// Clock drift factor: c0_{t+1} ≈ c0_t + drift_t * dt.
// Doppler factor: doppler_obs ≈ drift - geometric_range_rate.
//   geometric_range_rate includes first-order Sagnac; optional sat_clock_drift
//   is subtracted from the geometric rate before forming the prediction.
//   Constrains velocity [vx,vy,vz] and drift.
//
// sat_vel: [T, S, 3] satellite velocity ECEF (m/s).
// doppler: [T, S] Doppler pseudorange-rate (m/s), 0 means unobserved.
// doppler_weights: [T, S] weights for Doppler observations.
// dt: [T] time differences between consecutive epochs (seconds); dt[T-1] unused.
// sat_clock_drift: optional [T,S] satellite clock drift in m/s.
// imu_delta_p: optional [T-1,3] preintegrated displacement prior in ECEF metres.
// imu_delta_v: optional [T-1,3] preintegrated velocity delta prior in ECEF m/s.
// state_stride: optional state width. 0 keeps legacy ss=7+n_clock. ss=10+n_clock
// enables a minimal accelerometer-bias extension:
//   [x, y, z, vx, vy, vz, c0, ..., c_{nc-1}, drift, bax, bay, baz]
// When enabled, IMU delta residuals include first-order accel-bias correction,
// plus optional initial zero-bias prior and between-epoch bias smoothness.
// absolute_height_ref_ecef: optional [T,3] reference ECEF positions. When
// absolute_height_sigma_m > 0 and enu_up_ecef is set, constrains only ENU-up
// height residual u·(x_t - ref_t); non-finite rows are skipped.
//
// Returns iterations completed on success, -1 on failure.
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
                   const double* sat_vel = nullptr,
                   const double* doppler = nullptr,
                   const double* doppler_weights = nullptr,
                   const double* dt = nullptr,
                   const std::uint8_t* stop_mask = nullptr,
                   const double* tdcp_meas = nullptr,
                   const double* tdcp_weights = nullptr,
                   double tdcp_sigma_m = 0.0,
                   bool tdcp_use_drift = false,
                   // Loop-aware relative height: soft equality of ENU "up" at epochs i,j.
                   double relative_height_sigma_m = 0.0,
                   const double* enu_up_ecef = nullptr,
                   int n_rel_height_edges = 0,
                   const std::int32_t* rel_height_i = nullptr,
                   const std::int32_t* rel_height_j = nullptr,
                   const double* imu_delta_p = nullptr,
                   const double* imu_delta_v = nullptr,
                   double imu_position_sigma_m = 0.0,
                   double imu_velocity_sigma_mps = 0.0,
                   const double* sat_clock_drift = nullptr,
                   const double* absolute_height_ref_ecef = nullptr,
                   double absolute_height_sigma_m = 0.0,
                   int state_stride = 0,
                   double imu_accel_bias_prior_sigma_mps2 = 0.0,
                   double imu_accel_bias_between_sigma_mps2 = 0.0);

}  // namespace gnss_gpu
