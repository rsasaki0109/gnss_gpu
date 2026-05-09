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
    return _fgo_gnss_lm(
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
    )


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
    max_iter: int = 25,
    tol: float = 1e-3,
    huber_k: float = 0.0,
    line_search: bool = True,
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
    imu_position_sigma_m: float = 0.0,
    imu_velocity_sigma_mps: float = 0.0,
    imu_accel_bias_prior_sigma_mps2: float = 0.0,
    imu_accel_bias_between_sigma_mps2: float = 0.0,
) -> tuple[int, float]:
    """Extended FGO with velocity state + Doppler factor + optional TDCP (in-place ``state``).

    ``state`` has shape ``(T, 7 + n_clock)``:
    ``[x, y, z, vx, vy, vz, c0, ..., c_{K-1}, drift]`` in metres / (m/s).
    Passing ``(T, 10 + n_clock)`` appends a minimal accelerometer-bias state
    ``[bax, bay, baz]``. When present, IMU delta residuals include first-order
    acceleration-bias correction.

    Motion factor couples position and velocity: ``x_{t+1} = x_t + v_t * dt``.
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
    consecutive stopped epochs.

    ``sat_vel``: ``(T, S, 3)`` satellite velocity in ECEF (m/s).
    ``doppler``: ``(T, S)`` pseudorange-rate (m/s); 0 = unobserved.
    ``doppler_weights``: ``(T, S)`` weights for Doppler observations.
    ``sat_clock_drift``: optional ``(T, S)`` satellite clock drift in m/s,
    subtracted from geometric range-rate in the Doppler prediction.
    ``dt``: ``(T,)`` inter-epoch time differences in seconds; ``dt[T-1]`` unused.

    ``tdcp_meas``: optional ``(T-1, S)`` TDCP measurements in metres.
    ``tdcp_weights``: optional ``(T-1, S)`` per-observation weights for TDCP.
    ``tdcp_sigma_m``: uniform TDCP sigma in metres (used when ``tdcp_weights``
    is None).
    ``tdcp_use_drift``: when true, use the MATLAB XXDD variant
    ``e^T Δx + dt*(d_t + d_t+1)/2`` instead of the default XXCC clock-delta
    variant ``e^T Δx + (clk_t+1 - clk_t)``.

    ``relative_height_sigma_m``: optional std-dev (m) for loop-closure relative
    height equality in ENU-up: penalises ``u·(x_i - x_j)`` with unit ``u`` (ECEF).
    Requires ``enu_up_ecef`` (3,) and matching ``rel_height_edge_i`` /
    ``rel_height_edge_j`` int32 index pairs (local epoch indices).

    ``absolute_height_ref_ecef``: optional ``(T, 3)`` reference ECEF positions.
    With ``absolute_height_sigma_m > 0`` and ``enu_up_ecef`` set, constrains only
    ENU-up height via ``u·(x_t - ref_t)``. Non-finite reference rows are skipped.

    ``imu_delta_p`` / ``imu_delta_v``: optional ``(T-1, 3)`` preintegrated
    ECEF displacement and velocity-delta priors between adjacent epochs.
    ``imu_position_sigma_m`` and ``imu_velocity_sigma_mps`` set their scalar
    standard deviations; non-positive sigmas disable the corresponding prior.
    ``imu_accel_bias_prior_sigma_mps2`` adds an initial zero-bias prior and
    ``imu_accel_bias_between_sigma_mps2`` adds between-epoch bias smoothness
    when the appended accelerometer-bias state is present.

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
    if state.shape[1] not in (base_ss, base_ss + 3):
        raise ValueError(f"state columns {state.shape[1]} must be 7+n_clock ({base_ss}) or 10+n_clock ({base_ss + 3})")

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

    ls = 1 if line_search else 0
    native_args = (
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
        float(imu_position_sigma_m),
        float(imu_velocity_sigma_mps),
        scd,
        abs_h_ref,
        float(absolute_height_sigma_m),
        float(imu_accel_bias_prior_sigma_mps2),
        float(imu_accel_bias_between_sigma_mps2),
    )
    try:
        return _fgo_gnss_lm_vd(*native_args)
    except TypeError as exc:
        accel_bias_requested = (
            state.shape[1] == base_ss + 3
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
