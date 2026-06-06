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
                double tdcp_sigma_m = 0.0,
                double tdcp_huber_k = 0.0);

// Extended FGO with velocity state + clock drift + Doppler factor.
//
// Per-epoch state layout:
//   [x, y, z, vx, vy, vz, c0, ..., c_{nc-1}, drift]
//   ss = 3 + 3 + n_clock + 1 = 7 + n_clock
//
// Pseudorange factor constrains position + clock (same model as fgo_gnss_lm).
// Motion factor: x_{t+1} - x_t ≈ (v_t + v_{t+1}) * dt / 2
// (gtsam_gnss MotionFactor_XXVV parity).
// Clock drift factor: c0_{t+1} ≈ c0_t + drift_t * dt.
// Doppler factor: doppler_obs ≈ drift + geometric_range_rate.
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
// imu_delta_angle: optional [T-1,3] preintegrated delta angle in radians.
// imu_delta_t: optional [T-1] preintegration interval duration in seconds.
//   When unset, IMU delta residuals fall back to graph dt.
// imu_delta_*_bias_*_jac: optional [T-1,3,3] positive preintegration
//   measurement Jacobians wrt accel/gyro bias. Native residuals subtract
//   these Jacobians times the active bias state. When unset, diagonal
//   constant-acceleration fallbacks are used.
// imu_*_weights: optional [T-1,3] diagonal component weights that override
// scalar IMU sigmas per interval/component; non-finite or non-positive entries
// skip the corresponding component.
// imu_preintegration_information: optional [T-1,9,9] dense information matrix
// for GTSAM residual order [delta_angle(3), delta_p(3), delta_v(3)]. When provided,
// it replaces the separate diagonal IMU delta factors and can encode p/v/angle
// cross-covariance terms.
// imu_gravity: optional [T-1,3] navigation-frame gravity acceleration used by
// the GTSAM-style body-frame IMU p/v residual when the attitude state is present.
// Native predicts state j from pose_i/vel_i/preintegration, then evaluates p/v
// in R_j^T with predicted-minus-actual sign, matching NavState::localCoordinates.
// When unset, the solver keeps the legacy navigation-frame p/v delta residual.
// doppler_huber_k / tdcp_huber_k: optional Huber thresholds on Mahalanobis
// residuals z=|sqrt(w)*res| for Doppler and TDCP. <=0 keeps pure L2.
// state_stride: optional state width. 0 keeps legacy ss=7+n_clock. ss=10+n_clock
// enables a minimal accelerometer-bias extension, ss=13+n_clock enables a
// Taroz-shaped accelerometer + gyroscope bias extension, ss=16+n_clock
// prepends a minimal attitude-error state before the 6D bias, and
// ss=19+n_clock prepends a separate Pose3 translation before attitude/bias
// with a constrained pose-translation-to-x factor:
//   [x, y, z, vx, vy, vz, c0, ..., c_{nc-1}, drift, bax, bay, baz]
//   [x, y, z, vx, vy, vz, c0, ..., c_{nc-1}, drift, bax, bay, baz, bgx, bgy, bgz]
//   [x, y, z, vx, vy, vz, c0, ..., c_{nc-1}, drift, ax, ay, az, bax, bay, baz, bgx, bgy, bgz]
//   [x, y, z, vx, vy, vz, c0, ..., c_{nc-1}, drift, px, py, pz, ax, ay, az, bax, bay, baz, bgx, bgy, bgz]
// When enabled, IMU delta residuals include accel-bias correction. The
// attitude-error state rotates delta position/velocity with SO(3) Exp, and can
// receive delta-angle factors with SO(3) Log residuals and gyro-bias correction.
// Accel and gyro bias states can also receive optional initial zero-bias priors
// and between-epoch smoothness.
// imu_factor_use_next_bias selects the bias epoch used inside IMU delta
// residuals. false uses epoch t, true uses epoch t+1, matching Taroz
// fgo_gnss_imu.m's ImuFactor(..., keyB2, ...).
// imu_*_bias_between_weights: optional [T-1,3] interval/component weights that
// override scalar bias-between sigmas; useful for Taroz sqrt(IMU sample count)
// bias random-walk scaling.
// absolute_height_ref_ecef: optional [T,3] reference ECEF positions. When
// absolute_height_sigma_m > 0 and enu_up_ecef is set, constrains only ENU-up
// height residual u·(x_t - ref_t); non-finite rows are skipped.
// tdcp_linearization_ref_ecef: optional [T,3] receiver ECEF positions used to
// evaluate TDCP LOS at the second epoch and predict origin-relative
// displacement e^T((x2-ref2)-(x1-ref1)). This matches Taroz gtsam_gnss TDCP
// residuals when tdcp_meas has already been geometry-corrected at the same
// reference.
// pr_linearization_ref_ecef/pr_linearization_los_ecef: optional [T,3] receiver
// references and [T,S,3] LOS vectors. When both are set, pseudorange uses the
// Taroz/GTSAM fixed-linearized residual los·(x-ref)+clock-pr instead of
// nonlinear satellite geometry.
// doppler_linearization_ref_vel/doppler_linearization_los_ecef: optional [T,3]
// receiver velocity references and [T,S,3] LOS vectors. When both are set,
// Doppler uses los·(v-ref_vel)+drift-doppler instead of nonlinear range-rate.
//
// lm_damping: when >0, enables Levenberg-Marquardt-style diagonal damping.
// With enable_line_search set, this is the initial lambda for adaptive LM:
// lambda is multiplied/divided by 10 on rejected/accepted trial steps,
// approximating GTSAM's default LM schedule. Without enable_line_search, the
// fixed lambda is applied once per iteration without trial rejection.
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
                   const double* imu_delta_angle = nullptr,
                   const double* imu_delta_t = nullptr,
                   const double* imu_delta_p_bias_accel_jac = nullptr,
                   const double* imu_delta_v_bias_accel_jac = nullptr,
                   const double* imu_delta_p_bias_gyro_jac = nullptr,
                   const double* imu_delta_v_bias_gyro_jac = nullptr,
                   const double* imu_delta_angle_bias_gyro_jac = nullptr,
                   double imu_position_sigma_m = 0.0,
                   double imu_velocity_sigma_mps = 0.0,
                   double imu_attitude_sigma_rad = 0.0,
                   const double* imu_position_weights = nullptr,
                   const double* imu_velocity_weights = nullptr,
                   const double* imu_attitude_weights = nullptr,
                   const double* imu_preintegration_information = nullptr,
                   bool imu_factor_use_next_bias = false,
                   const double* sat_clock_drift = nullptr,
                   const double* absolute_height_ref_ecef = nullptr,
                   double absolute_height_sigma_m = 0.0,
                   int state_stride = 0,
                   double imu_accel_bias_prior_sigma_mps2 = 0.0,
                   double imu_accel_bias_between_sigma_mps2 = 0.0,
                   const double* imu_accel_bias_between_weights = nullptr,
                   double imu_gyro_bias_prior_sigma_radps = 0.0,
                   double imu_gyro_bias_between_sigma_radps = 0.0,
                   const double* imu_gyro_bias_between_weights = nullptr,
                   double doppler_huber_k = 0.0,
                   double tdcp_huber_k = 0.0,
                   const double* tdcp_linearization_ref_ecef = nullptr,
	                   double stop_velocity_huber_k = 0.0,
	                   double stop_position_huber_k = 0.0,
	                   double relative_height_huber_k = 0.0,
	                   double absolute_height_huber_k = 0.0,
	                   const double* imu_gravity = nullptr,
                   const double* pr_linearization_ref_ecef = nullptr,
	                   const double* pr_linearization_los_ecef = nullptr,
                   const double* doppler_linearization_ref_vel = nullptr,
                   const double* doppler_linearization_los_ecef = nullptr,
                   double stop_attitude_sigma_rad = 0.0,
                   double lm_damping = 0.0);

}  // namespace gnss_gpu
