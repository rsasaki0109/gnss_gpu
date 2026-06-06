"""GPU-assembled GNSS batch factor-graph optimization.

References *PseudorangeFactor_XC* / multi-clock patterns from `gtsam_gnss`
(Taro Suzuki et al.). See ``fgo_gnss_lm`` parameters for ``n_clock`` and
``sys_kind``.
"""

from __future__ import annotations

import numpy as np

try:
    from gnss_gpu._gnss_gpu import fgo_gnss_lm as _fgo_gnss_lm
except ImportError:
    _fgo_gnss_lm = None

try:
    from gnss_gpu._gnss_gpu import fgo_gnss_lm_vd as _fgo_gnss_lm_vd
except ImportError:
    _fgo_gnss_lm_vd = None


def fgo_gnss_lm(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    *,
    sys_kind: np.ndarray | None = None,
    n_clock: int = 1,
    motion_sigma_m: float = 0.0,
    max_iter: int = 25,
    tol: float = 1e-3,
    huber_k: float = 0.0,
    line_search: bool = True,
    motion_displacement: np.ndarray | None = None,
    tdcp_meas: np.ndarray | None = None,
    tdcp_weights: np.ndarray | None = None,
    tdcp_sigma_m: float = 0.0,
    tdcp_huber_k: float = 0.0,
) -> tuple[int, float]:
    """Iterated Gauss-Newton with GPU-assembled normal equations (in-place ``state``).

    ``state`` has shape ``(T, 3 + n_clock)``: ``[x,y,z,c0,...,c_{K-1}]`` in metres.
    ``sys_kind`` is optional ``int32`` ``(T, S)`` with values in ``0..n_clock-1``.
    Row ``h`` for a measurement is ``h[0]=1`` and ``h[sk]=1`` if ``sk > 0``
    (gtsam_gnss clock + ISB pattern).
    The native solver accepts up to seven clocks, matching MATLAB's L1/L5
    signal-clock layout used by the GSDC2023 raw bridge.

    ``huber_k``: if > 0, apply IRLS Huber reweighting with threshold on Mahalanobis
    residuals ``z = |sqrt(w) * res|`` (same pattern as common robust GNSS solvers).

    ``motion_displacement``: optional ``(T, 3)`` array of predicted inter-epoch
    position changes (e.g. Doppler velocity * dt). When provided, the motion
    random-walk factor penalises ``(x_{t} - x_{t+1}) + disp[t]`` instead of
    ``(x_{t} - x_{t+1})``, equivalent to gtsam_gnss DopplerFactor_XXCC.

    ``tdcp_meas``: optional ``(T-1, S)`` TDCP measurements in metres (carrier phase
    difference between consecutive epochs). Zero means unobserved when
    ``tdcp_weights`` is not provided.

    ``tdcp_weights``: optional ``(T-1, S)`` per-observation weights for TDCP.
    When not provided but ``tdcp_sigma_m > 0``, uniform weight
    ``1/tdcp_sigma_m^2`` is used.

    ``tdcp_sigma_m``: uniform TDCP sigma in metres (used when ``tdcp_weights``
    is None).

    ``tdcp_huber_k``: optional TDCP Huber threshold on Mahalanobis residuals
    ``z = |sqrt(w) * res|``. ``0`` keeps pure L2 TDCP.
    """
    if _fgo_gnss_lm is None:
        raise RuntimeError("gnss_gpu native extension not built (fgo_gnss_lm unavailable)")
    sat_ecef = np.ascontiguousarray(sat_ecef, dtype=np.float64)
    pseudorange = np.ascontiguousarray(pseudorange, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    if state.dtype != np.float64 or not state.flags.writeable:
        raise ValueError("state must be float64 and writeable")
    state = np.ascontiguousarray(state, dtype=np.float64)
    if state.shape[1] != 3 + n_clock:
        raise ValueError(f"state columns {state.shape[1]} != 3 + n_clock ({3 + n_clock})")
    sk = None
    if sys_kind is not None:
        sk = np.ascontiguousarray(sys_kind, dtype=np.int32)
    md = None
    if motion_displacement is not None:
        md = np.ascontiguousarray(motion_displacement, dtype=np.float64).ravel()
    tm = None
    if tdcp_meas is not None:
        tm = np.ascontiguousarray(tdcp_meas, dtype=np.float64)
    tw = None
    if tdcp_weights is not None:
        tw = np.ascontiguousarray(tdcp_weights, dtype=np.float64)
    ls = 1 if line_search else 0
    native_args = (
        sat_ecef,
        pseudorange,
        weights,
        state,
        float(motion_sigma_m),
        int(max_iter),
        float(tol),
        float(huber_k),
        ls,
        sk,
        int(n_clock),
        md,
        tm,
        tw,
        float(tdcp_sigma_m),
        float(tdcp_huber_k),
    )
    try:
        return _fgo_gnss_lm(*native_args)
    except TypeError as exc:
        if float(tdcp_huber_k) > 0.0:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for TDCP Huber factors") from exc
        return _fgo_gnss_lm(*native_args[:-1])


def fgo_gnss_lm_vd(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    *,
    sys_kind: np.ndarray | None = None,
    n_clock: int = 1,
    motion_sigma_m: float = 0.0,
    clock_drift_sigma_m: float = 0.0,
    clock_use_average_drift: bool = False,
    stop_velocity_sigma_mps: float = 0.0,
    stop_position_sigma_m: float = 0.0,
    stop_attitude_sigma_rad: float = 0.0,
    max_iter: int = 25,
    tol: float = 1e-3,
    huber_k: float = 0.0,
    line_search: bool = True,
    lm_damping: float = 0.0,
    sat_vel: np.ndarray | None = None,
    doppler: np.ndarray | None = None,
    doppler_weights: np.ndarray | None = None,
    sat_clock_drift: np.ndarray | None = None,
    dt: np.ndarray | None = None,
    stop_mask: np.ndarray | None = None,
    tdcp_meas: np.ndarray | None = None,
    tdcp_weights: np.ndarray | None = None,
    tdcp_sigma_m: float = 0.0,
    tdcp_use_drift: bool = False,
    relative_height_sigma_m: float = 0.0,
    enu_up_ecef: np.ndarray | None = None,
    rel_height_edge_i: np.ndarray | None = None,
    rel_height_edge_j: np.ndarray | None = None,
    absolute_height_ref_ecef: np.ndarray | None = None,
    absolute_height_sigma_m: float = 0.0,
    imu_delta_p: np.ndarray | None = None,
    imu_delta_v: np.ndarray | None = None,
    imu_delta_angle: np.ndarray | None = None,
    imu_delta_t: np.ndarray | None = None,
    imu_delta_p_bias_accel_jac: np.ndarray | None = None,
    imu_delta_v_bias_accel_jac: np.ndarray | None = None,
    imu_delta_p_bias_gyro_jac: np.ndarray | None = None,
    imu_delta_v_bias_gyro_jac: np.ndarray | None = None,
    imu_delta_angle_bias_gyro_jac: np.ndarray | None = None,
    imu_position_sigma_m: float = 0.0,
    imu_velocity_sigma_mps: float = 0.0,
    imu_attitude_sigma_rad: float = 0.0,
    imu_position_weights: np.ndarray | None = None,
    imu_velocity_weights: np.ndarray | None = None,
    imu_attitude_weights: np.ndarray | None = None,
    imu_preintegration_information: np.ndarray | None = None,
    imu_gravity: np.ndarray | None = None,
    imu_factor_use_next_bias: bool = False,
    imu_accel_bias_prior_sigma_mps2: float = 0.0,
    imu_accel_bias_between_sigma_mps2: float = 0.0,
    imu_accel_bias_between_weights: np.ndarray | None = None,
    imu_gyro_bias_prior_sigma_radps: float = 0.0,
    imu_gyro_bias_between_sigma_radps: float = 0.0,
    imu_gyro_bias_between_weights: np.ndarray | None = None,
    doppler_huber_k: float = 0.0,
    tdcp_huber_k: float = 0.0,
    tdcp_linearization_ref_ecef: np.ndarray | None = None,
    pr_linearization_ref_ecef: np.ndarray | None = None,
    pr_linearization_los_ecef: np.ndarray | None = None,
    doppler_linearization_ref_vel: np.ndarray | None = None,
    doppler_linearization_los_ecef: np.ndarray | None = None,
    stop_velocity_huber_k: float = 0.0,
    stop_position_huber_k: float = 0.0,
    relative_height_huber_k: float = 0.0,
    absolute_height_huber_k: float = 0.0,
) -> tuple[int, float]:
    """Extended FGO with velocity state + Doppler factor + optional TDCP (in-place ``state``).

    ``state`` has shape ``(T, 7 + n_clock)``:
    ``[x, y, z, vx, vy, vz, c0, ..., c_{K-1}, drift]`` in metres / (m/s).
    Passing ``(T, 10 + n_clock)`` appends a minimal accelerometer-bias state
    ``[bax, bay, baz]``. Passing ``(T, 13 + n_clock)`` appends a Taroz-shaped
    6D IMU bias state ``[bax, bay, baz, bgx, bgy, bgz]``. Passing
    ``(T, 16 + n_clock)`` appends ``[attx, atty, attz, bax, bay, baz, bgx,
    bgy, bgz]``. Passing ``(T, 19 + n_clock)`` appends Taroz-style
    ``[pose_x, pose_y, pose_z, attx, atty, attz, bax, bay, baz, bgx, bgy,
    bgz]`` and adds a constrained ``pose translation - x`` factor. The native
    solver uses accel bias in IMU delta residuals,
    SO(3) Exp rotation for delta position/velocity when the attitude state is
    present, and SO(3) Log residuals for delta angle with gyro-bias correction.

    Motion factor couples position and velocity with gtsam_gnss
    ``MotionFactor_XXVV`` parity:
    ``x_{t+1} - x_t = (v_t + v_t+1) * dt / 2``.
    Clock drift factor: default ``clk_{t+1} = clk_t + drift_t * dt``.
    When ``clock_use_average_drift`` is true, use MATLAB CCDD parity
    ``clk_{t+1} = clk_t + (drift_t + drift_t+1) * dt / 2``.

    Doppler factor constrains velocity and clock drift from pseudorange-rate
    observations. Requires ``sat_vel`` (satellite velocity), ``doppler``
    (pseudorange-rate), ``doppler_weights``, and ``dt`` (inter-epoch time
    differences).

    ``stop_mask``: optional ``(T,)`` boolean mask of stop epochs. When provided,
    ``stop_velocity_sigma_mps`` adds per-epoch zero-velocity priors on stopped
    epochs, and ``stop_position_sigma_m`` adds ``x_t = x_t+1`` hold factors on
    consecutive stopped epochs. With an attitude-IMU state,
    ``stop_attitude_sigma_rad`` also adds zero-relative-rotation hold factors
    on consecutive stopped epochs.

    ``sat_vel``: ``(T, S, 3)`` satellite velocity in ECEF (m/s).
    ``doppler``: ``(T, S)`` pseudorange-rate (m/s); 0 = unobserved.
    ``doppler_weights``: ``(T, S)`` weights for Doppler observations.
    ``sat_clock_drift``: optional ``(T, S)`` satellite clock drift in m/s,
    subtracted from geometric range-rate before the Doppler prediction
    ``doppler_obs ~= drift + geometric_range_rate``.
    ``dt``: ``(T,)`` inter-epoch time differences in seconds; ``dt[T-1]`` unused.

    ``tdcp_meas``: optional ``(T-1, S)`` TDCP measurements in metres.
    ``tdcp_weights``: optional ``(T-1, S)`` per-observation weights for TDCP.
    ``tdcp_sigma_m``: uniform TDCP sigma in metres (used when ``tdcp_weights``
    is None).
    ``doppler_huber_k`` / ``tdcp_huber_k``: optional Huber thresholds for the
    Doppler and TDCP factors on Mahalanobis residuals. ``0`` keeps pure L2.
    ``stop_velocity_huber_k`` / ``stop_position_huber_k`` /
    ``relative_height_huber_k`` / ``absolute_height_huber_k`` do the same for
    stop and height factors.
    ``tdcp_use_drift``: when true, use the MATLAB XXDD variant
    ``e^T Δx + dt*(d_t + d_t+1)/2`` instead of the default XXCC clock-delta
    variant ``e^T Δx + (clk_t+1 - clk_t)``.
    ``tdcp_linearization_ref_ecef``: optional ``(T, 3)`` receiver ECEF
    positions used to evaluate TDCP LOS and predict origin-relative
    displacement. Use this when ``tdcp_meas`` has already had reference
    geometry removed, matching Taroz ``TDCPFactor_XXCC/XXDD``.
    ``pr_linearization_ref_ecef`` / ``pr_linearization_los_ecef``: optional
    ``(T, 3)`` receiver references and ``(T, S, 3)`` LOS vectors that replace
    the nonlinear pseudorange geometry with Taroz/GTSAM-style fixed
    linearization ``los·(x-ref)+clock``.
    ``doppler_linearization_ref_vel`` / ``doppler_linearization_los_ecef``:
    optional ``(T, 3)`` velocity references and ``(T, S, 3)`` LOS vectors that
    replace nonlinear Doppler geometry with ``los·(v-ref_vel)+drift``.

    ``relative_height_sigma_m``: optional std-dev (m) for loop-closure relative
    height equality in ENU-up: penalises ``u·(x_i - x_j)`` with unit ``u`` (ECEF).
    Requires ``enu_up_ecef`` (3,) and matching ``rel_height_edge_i`` /
    ``rel_height_edge_j`` int32 index pairs (local epoch indices).

    ``absolute_height_ref_ecef``: optional ``(T, 3)`` reference ECEF positions.
    With ``absolute_height_sigma_m > 0`` and ``enu_up_ecef`` set, constrains only
    ENU-up height via ``u·(x_t - ref_t)``. Non-finite reference rows are skipped.

    ``imu_delta_p`` / ``imu_delta_v`` / ``imu_delta_angle``: optional
    ``(T-1, 3)`` preintegrated ECEF displacement, velocity-delta, and
    delta-angle priors between adjacent epochs. ``imu_position_sigma_m``,
    ``imu_velocity_sigma_mps``, and ``imu_attitude_sigma_rad`` set their scalar
    standard deviations; non-positive sigmas disable the corresponding prior.
    ``imu_delta_t`` optionally provides the ``(T-1,)`` preintegration interval
    durations used inside IMU residuals and bias correction terms. If omitted,
    IMU residuals fall back to ``dt``.
    ``imu_delta_p_bias_accel_jac`` / ``imu_delta_v_bias_accel_jac`` /
    ``imu_delta_p_bias_gyro_jac`` / ``imu_delta_v_bias_gyro_jac`` /
    ``imu_delta_angle_bias_gyro_jac`` optionally provide positive measurement
    Jacobians with shape ``(T-1, 3, 3)``. Residuals subtract these Jacobians
    times the relevant accel/gyro bias state; when omitted, native fallback
    diagonal Jacobians derived from ``imu_delta_t``/``dt`` are used where
    available.
    ``imu_position_weights`` / ``imu_velocity_weights`` /
    ``imu_attitude_weights`` optionally provide diagonal per-interval weights
    with shape ``(T-1, 3)`` and override the scalar sigmas per component.
    ``imu_preintegration_information`` optionally provides dense per-interval
    information matrices with shape ``(T-1, 9, 9)`` for GTSAM residual order
    ``[delta_angle(3), delta_p(3), delta_v(3)]``. When supplied it replaces the
    separate diagonal IMU delta factors and can encode p/v/angle cross terms.
    ``imu_gravity`` optionally provides ``(T-1, 3)`` navigation-frame gravity
    acceleration and switches IMU p/v residuals to the GTSAM-style body-frame
    topology: predict state ``j`` from pose_i/vel_i/preintegration, then evaluate
    p/v in ``R_j.T`` with predicted-minus-actual sign. It requires the
    appended attitude/bias state and uses ``imu_delta_t`` when provided, else
    ``dt``.
    ``imu_factor_use_next_bias`` makes IMU delta residuals use the epoch ``t+1``
    bias state instead of epoch ``t``, matching Taroz ``ImuFactor(..., keyB2, ...)``.
    ``lm_damping`` enables Levenberg-Marquardt-style diagonal damping. With
    ``line_search=True`` the native solver uses this value as the initial
    adaptive LM lambda and multiplies or divides it by 10 on rejected or
    accepted trial steps. With ``line_search=False`` it applies this fixed
    lambda once per iteration without trial rejection, useful for parity
    diagnostics.
    ``imu_accel_bias_prior_sigma_mps2`` adds an initial zero-bias prior and
    ``imu_accel_bias_between_sigma_mps2`` adds between-epoch bias smoothness
    when the appended accelerometer-bias state is present.
    ``imu_gyro_bias_prior_sigma_radps`` / ``imu_gyro_bias_between_sigma_radps``
    do the same for the optional appended gyroscope-bias state.
    ``imu_accel_bias_between_weights`` / ``imu_gyro_bias_between_weights`` may
    provide ``(T-1, 3)`` per-interval bias-between information weights that
    override the scalar between sigmas.

    Maintains backward compatibility: if no Doppler/TDCP data is provided, only
    pseudorange + motion + clock drift factors are used.
    """
    if _fgo_gnss_lm_vd is None:
        raise RuntimeError("gnss_gpu native extension not built (fgo_gnss_lm_vd unavailable)")
    sat_ecef = np.ascontiguousarray(sat_ecef, dtype=np.float64)
    pseudorange = np.ascontiguousarray(pseudorange, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    if state.dtype != np.float64 or not state.flags.writeable:
        raise ValueError("state must be float64 and writeable")
    state = np.ascontiguousarray(state, dtype=np.float64)
    base_ss = 7 + n_clock
    if state.shape[1] not in (base_ss, base_ss + 3, base_ss + 6, base_ss + 9, base_ss + 12):
        raise ValueError(
            f"state columns {state.shape[1]} must be 7+n_clock ({base_ss}), "
            f"10+n_clock ({base_ss + 3}), 13+n_clock ({base_ss + 6}), "
            f"16+n_clock ({base_ss + 9}), or 19+n_clock ({base_ss + 12})"
        )
    if imu_gravity is not None and state.shape[1] not in (base_ss + 9, base_ss + 12):
        raise ValueError(
            f"imu_gravity requires state columns 16+n_clock ({base_ss + 9}) "
            f"or 19+n_clock ({base_ss + 12})"
        )

    sk = None
    if sys_kind is not None:
        sk = np.ascontiguousarray(sys_kind, dtype=np.int32)

    sv = None
    if sat_vel is not None:
        sv = np.ascontiguousarray(sat_vel, dtype=np.float64)

    dop = None
    if doppler is not None:
        dop = np.ascontiguousarray(doppler, dtype=np.float64)

    dw = None
    if doppler_weights is not None:
        dw = np.ascontiguousarray(doppler_weights, dtype=np.float64)

    scd = None
    if sat_clock_drift is not None:
        scd = np.ascontiguousarray(sat_clock_drift, dtype=np.float64)

    dt_arr = None
    if dt is not None:
        dt_arr = np.ascontiguousarray(dt, dtype=np.float64).ravel()

    stop_arr = None
    if stop_mask is not None:
        stop_arr = np.ascontiguousarray(stop_mask, dtype=np.uint8).ravel()

    tm = None
    if tdcp_meas is not None:
        tm = np.ascontiguousarray(tdcp_meas, dtype=np.float64)
    tw_arr = None
    if tdcp_weights is not None:
        tw_arr = np.ascontiguousarray(tdcp_weights, dtype=np.float64)
    tdcp_ref = None
    if tdcp_linearization_ref_ecef is not None:
        tdcp_ref = np.ascontiguousarray(tdcp_linearization_ref_ecef, dtype=np.float64)
    pr_ref = None
    if pr_linearization_ref_ecef is not None:
        pr_ref = np.ascontiguousarray(pr_linearization_ref_ecef, dtype=np.float64)
    pr_los = None
    if pr_linearization_los_ecef is not None:
        pr_los = np.ascontiguousarray(pr_linearization_los_ecef, dtype=np.float64)
    if (pr_ref is None) != (pr_los is None):
        raise ValueError("pr_linearization_ref_ecef and pr_linearization_los_ecef must be provided together")
    dop_ref_vel = None
    if doppler_linearization_ref_vel is not None:
        dop_ref_vel = np.ascontiguousarray(doppler_linearization_ref_vel, dtype=np.float64)
    dop_los = None
    if doppler_linearization_los_ecef is not None:
        dop_los = np.ascontiguousarray(doppler_linearization_los_ecef, dtype=np.float64)
    if (dop_ref_vel is None) != (dop_los is None):
        raise ValueError(
            "doppler_linearization_ref_vel and doppler_linearization_los_ecef must be provided together"
        )

    enu_up = None
    if enu_up_ecef is not None:
        enu_up = np.ascontiguousarray(enu_up_ecef, dtype=np.float64).ravel()
    rei = None
    rej = None
    if rel_height_edge_i is not None and rel_height_edge_j is not None:
        rei = np.ascontiguousarray(rel_height_edge_i, dtype=np.int32).ravel()
        rej = np.ascontiguousarray(rel_height_edge_j, dtype=np.int32).ravel()

    abs_h_ref = None
    if absolute_height_ref_ecef is not None:
        abs_h_ref = np.ascontiguousarray(absolute_height_ref_ecef, dtype=np.float64)

    imu_dp = None
    if imu_delta_p is not None:
        imu_dp = np.ascontiguousarray(imu_delta_p, dtype=np.float64)
    imu_dv = None
    if imu_delta_v is not None:
        imu_dv = np.ascontiguousarray(imu_delta_v, dtype=np.float64)
    imu_da = None
    if imu_delta_angle is not None:
        imu_da = np.ascontiguousarray(imu_delta_angle, dtype=np.float64)
    imu_dt = None
    if imu_delta_t is not None:
        imu_dt = np.ascontiguousarray(imu_delta_t, dtype=np.float64).ravel()
    imu_dp_ba_jac = None
    if imu_delta_p_bias_accel_jac is not None:
        imu_dp_ba_jac = np.ascontiguousarray(imu_delta_p_bias_accel_jac, dtype=np.float64)
    imu_dv_ba_jac = None
    if imu_delta_v_bias_accel_jac is not None:
        imu_dv_ba_jac = np.ascontiguousarray(imu_delta_v_bias_accel_jac, dtype=np.float64)
    imu_dp_bg_jac = None
    if imu_delta_p_bias_gyro_jac is not None:
        imu_dp_bg_jac = np.ascontiguousarray(imu_delta_p_bias_gyro_jac, dtype=np.float64)
    imu_dv_bg_jac = None
    if imu_delta_v_bias_gyro_jac is not None:
        imu_dv_bg_jac = np.ascontiguousarray(imu_delta_v_bias_gyro_jac, dtype=np.float64)
    imu_da_bg_jac = None
    if imu_delta_angle_bias_gyro_jac is not None:
        imu_da_bg_jac = np.ascontiguousarray(imu_delta_angle_bias_gyro_jac, dtype=np.float64)
    imu_pw = None
    if imu_position_weights is not None:
        imu_pw = np.ascontiguousarray(imu_position_weights, dtype=np.float64)
    imu_vw = None
    if imu_velocity_weights is not None:
        imu_vw = np.ascontiguousarray(imu_velocity_weights, dtype=np.float64)
    imu_aw = None
    if imu_attitude_weights is not None:
        imu_aw = np.ascontiguousarray(imu_attitude_weights, dtype=np.float64)
    imu_info = None
    if imu_preintegration_information is not None:
        imu_info = np.ascontiguousarray(imu_preintegration_information, dtype=np.float64)
    imu_grav = None
    if imu_gravity is not None:
        imu_grav = np.ascontiguousarray(imu_gravity, dtype=np.float64)
    imu_abw = None
    if imu_accel_bias_between_weights is not None:
        imu_abw = np.ascontiguousarray(imu_accel_bias_between_weights, dtype=np.float64)
    imu_gbw = None
    if imu_gyro_bias_between_weights is not None:
        imu_gbw = np.ascontiguousarray(imu_gyro_bias_between_weights, dtype=np.float64)

    ls = 1 if line_search else 0
    native_args_pre_imu_attitude = (
        sat_ecef,
        pseudorange,
        weights,
        state,
        float(motion_sigma_m),
        float(clock_drift_sigma_m),
        bool(clock_use_average_drift),
        float(stop_velocity_sigma_mps),
        float(stop_position_sigma_m),
        int(max_iter),
        float(tol),
        float(huber_k),
        ls,
        sk,
        int(n_clock),
        sv,
        dop,
        dw,
        dt_arr,
        stop_arr,
        tm,
        tw_arr,
        float(tdcp_sigma_m),
        bool(tdcp_use_drift),
        float(relative_height_sigma_m),
        enu_up,
        rei,
        rej,
        imu_dp,
        imu_dv,
    )
    native_args_pre_gyro_bias_no_attitude = native_args_pre_imu_attitude + (
        float(imu_position_sigma_m),
        float(imu_velocity_sigma_mps),
        scd,
        abs_h_ref,
        float(absolute_height_sigma_m),
        float(imu_accel_bias_prior_sigma_mps2),
        float(imu_accel_bias_between_sigma_mps2),
    )
    native_args_pre_gyro_bias = native_args_pre_imu_attitude + (
        imu_da,
        imu_dt,
        imu_dp_ba_jac,
        imu_dv_ba_jac,
        imu_dp_bg_jac,
        imu_dv_bg_jac,
        imu_da_bg_jac,
        float(imu_position_sigma_m),
        float(imu_velocity_sigma_mps),
        float(imu_attitude_sigma_rad),
        imu_pw,
        imu_vw,
        imu_aw,
        imu_info,
        bool(imu_factor_use_next_bias),
        scd,
        abs_h_ref,
        float(absolute_height_sigma_m),
        float(imu_accel_bias_prior_sigma_mps2),
        float(imu_accel_bias_between_sigma_mps2),
        imu_abw,
    )
    native_args_no_gyro_bias = native_args_pre_gyro_bias + (float(doppler_huber_k), float(tdcp_huber_k))
    native_args = native_args_pre_gyro_bias + (
        float(imu_gyro_bias_prior_sigma_radps),
        float(imu_gyro_bias_between_sigma_radps),
        imu_gbw,
    )
    native_args = native_args + (float(doppler_huber_k), float(tdcp_huber_k))
    native_args = native_args + (tdcp_ref,)
    native_args = native_args + (
        float(stop_velocity_huber_k),
        float(stop_position_huber_k),
        float(relative_height_huber_k),
        float(absolute_height_huber_k),
    )
    fixed_linearization_requested = (
        pr_ref is not None
        or pr_los is not None
        or dop_ref_vel is not None
        or dop_los is not None
    )
    stop_attitude_requested = float(stop_attitude_sigma_rad) > 0.0
    lm_damping_requested = float(lm_damping) > 0.0
    if imu_grav is not None or fixed_linearization_requested or stop_attitude_requested or lm_damping_requested:
        native_args = native_args + (imu_grav,)
    if fixed_linearization_requested:
        native_args = native_args + (pr_ref, pr_los, dop_ref_vel, dop_los)
    elif stop_attitude_requested or lm_damping_requested:
        native_args = native_args + (None, None, None, None)
    if stop_attitude_requested or lm_damping_requested:
        native_args = native_args + (float(stop_attitude_sigma_rad),)
    if lm_damping_requested:
        native_args = native_args + (float(lm_damping),)
    try:
        return _fgo_gnss_lm_vd(*native_args)
    except TypeError as exc:
        if lm_damping_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for LM damping") from exc
        if fixed_linearization_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for fixed-linearized P/D VD factors") from exc
        if imu_grav is not None:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for IMU gravity body-frame residuals") from exc
        if stop_attitude_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for stop attitude VD factors") from exc
        bias_jac_requested = (
            imu_dp_ba_jac is not None
            or imu_dv_ba_jac is not None
            or imu_dp_bg_jac is not None
            or imu_dv_bg_jac is not None
            or imu_da_bg_jac is not None
        )
        if bias_jac_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for IMU preintegration bias Jacobians") from exc
        imu_delta_t_requested = imu_dt is not None
        if imu_delta_t_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for IMU preintegration delta times") from exc
        pva_info_requested = imu_info is not None
        if pva_info_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for IMU preintegration information matrices") from exc
        weights_requested = imu_pw is not None or imu_vw is not None or imu_aw is not None
        if weights_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for IMU diagonal covariance weights") from exc
        bias_between_weights_requested = imu_abw is not None or imu_gbw is not None
        if bias_between_weights_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for IMU bias-between interval weights") from exc
        if bool(imu_factor_use_next_bias):
            raise RuntimeError("gnss_gpu native extension must be rebuilt for IMU next-bias factor mode") from exc
        native_args_pre_gyro_bias_no_weights = native_args_pre_imu_attitude + (
            imu_da,
            float(imu_position_sigma_m),
            float(imu_velocity_sigma_mps),
            float(imu_attitude_sigma_rad),
            scd,
            abs_h_ref,
            float(absolute_height_sigma_m),
            float(imu_accel_bias_prior_sigma_mps2),
            float(imu_accel_bias_between_sigma_mps2),
        )
        native_args_no_weights = native_args_pre_gyro_bias_no_weights + (
            float(imu_gyro_bias_prior_sigma_radps),
            float(imu_gyro_bias_between_sigma_radps),
            float(doppler_huber_k),
            float(tdcp_huber_k),
            tdcp_ref,
            float(stop_velocity_huber_k),
            float(stop_position_huber_k),
            float(relative_height_huber_k),
            float(absolute_height_huber_k),
        )
        try:
            return _fgo_gnss_lm_vd(*native_args_no_weights)
        except TypeError as exc_no_weights:
            exc = exc_no_weights
        attitude_requested = (
            state.shape[1] in (base_ss + 9, base_ss + 12)
            or imu_da is not None
            or float(imu_attitude_sigma_rad) > 0.0
        )
        if attitude_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for IMU attitude VD states") from exc
        native_args_no_attitude = native_args_pre_gyro_bias_no_attitude + (
            float(imu_gyro_bias_prior_sigma_radps),
            float(imu_gyro_bias_between_sigma_radps),
            float(doppler_huber_k),
            float(tdcp_huber_k),
            tdcp_ref,
            float(stop_velocity_huber_k),
            float(stop_position_huber_k),
            float(relative_height_huber_k),
            float(absolute_height_huber_k),
        )
        try:
            return _fgo_gnss_lm_vd(*native_args_no_attitude)
        except TypeError as exc_no_attitude:
            exc = exc_no_attitude
        gyro_bias_requested = (
            state.shape[1] == base_ss + 6
            or float(imu_gyro_bias_prior_sigma_radps) > 0.0
            or float(imu_gyro_bias_between_sigma_radps) > 0.0
        )
        if gyro_bias_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for IMU gyro-bias VD states") from exc
        native_args_no_gyro_bias = native_args_pre_gyro_bias_no_attitude + (
            float(doppler_huber_k),
            float(tdcp_huber_k),
        )
        native_args = native_args_no_gyro_bias + (tdcp_ref,)
        native_args = native_args + (
            float(stop_velocity_huber_k),
            float(stop_position_huber_k),
            float(relative_height_huber_k),
            float(absolute_height_huber_k),
        )
        try:
            return _fgo_gnss_lm_vd(*native_args)
        except TypeError as exc_no_gyro_bias:
            exc = exc_no_gyro_bias
        stop_height_huber_requested = (
            float(stop_velocity_huber_k) > 0.0
            or float(stop_position_huber_k) > 0.0
            or float(relative_height_huber_k) > 0.0
            or float(absolute_height_huber_k) > 0.0
        )
        if stop_height_huber_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for stop/height Huber VD factors") from exc
        native_args = native_args[:-4]
        try:
            return _fgo_gnss_lm_vd(*native_args)
        except TypeError as exc_no_stop_height_huber:
            exc = exc_no_stop_height_huber
        tdcp_ref_requested = tdcp_ref is not None
        if tdcp_ref_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for TDCP linearization reference") from exc
        native_args = native_args[:-1]
        factor_huber_requested = float(doppler_huber_k) > 0.0 or float(tdcp_huber_k) > 0.0
        if factor_huber_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for Doppler/TDCP Huber VD factors") from exc
        native_args = native_args[:-2]
        try:
            return _fgo_gnss_lm_vd(*native_args)
        except TypeError as exc_no_factor_huber:
            exc = exc_no_factor_huber
        accel_bias_requested = (
            state.shape[1] in (base_ss + 3, base_ss + 6)
            or float(imu_accel_bias_prior_sigma_mps2) > 0.0
            or float(imu_accel_bias_between_sigma_mps2) > 0.0
        )
        if accel_bias_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for IMU accel-bias VD states") from exc
        native_args_no_accel_bias = native_args[:-2]
        try:
            return _fgo_gnss_lm_vd(*native_args_no_accel_bias)
        except TypeError as exc_no_accel_bias:
            no_accel_bias_exc = exc_no_accel_bias
        absolute_height_requested = (
            abs_h_ref is not None
            and float(absolute_height_sigma_m) > 0.0
        )
        sat_clock_drift_requested = scd is not None
        imu_requested = (
            imu_dp is not None
            or imu_dv is not None
            or float(imu_position_sigma_m) > 0.0
            or float(imu_velocity_sigma_mps) > 0.0
        )
        if absolute_height_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for absolute-height VD factors") from no_accel_bias_exc
        native_args_no_abs_height = native_args_no_accel_bias[:-2]
        try:
            return _fgo_gnss_lm_vd(*native_args_no_abs_height)
        except TypeError as exc_no_abs_height:
            no_abs_height_exc = exc_no_abs_height
        if sat_clock_drift_requested:
            raise RuntimeError("gnss_gpu native extension must be rebuilt for Doppler satellite clock drift") from no_abs_height_exc
        try:
            return _fgo_gnss_lm_vd(*native_args_no_abs_height[:-1])
        except TypeError as exc_no_sat_clock:
            if imu_requested:
                raise RuntimeError("gnss_gpu native extension must be rebuilt for IMU VD factors") from exc_no_sat_clock
            return _fgo_gnss_lm_vd(*native_args_no_abs_height[:-5])


__all__ = ["fgo_gnss_lm", "fgo_gnss_lm_vd"]
