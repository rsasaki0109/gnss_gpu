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
) -> tuple[int, float]:
    """Iterated Gauss–Newton with GPU-assembled normal equations (in-place ``state``).

    ``state`` has shape ``(T, 3 + n_clock)``: ``[x,y,z,c0,...,c_{K-1}]`` in metres.
    ``sys_kind`` is optional ``int32`` ``(T, S)`` with values in ``0..n_clock-1``.
    Row ``h`` for a measurement is ``h[0]=1`` and ``h[sk]=1`` if ``sk > 0``
    (gtsam_gnss clock + ISB pattern).

    ``huber_k``: if > 0, apply IRLS Huber reweighting with threshold on Mahalanobis
    residuals ``z = |sqrt(w) * res|`` (same pattern as common robust GNSS solvers).

    ``motion_displacement``: optional ``(T, 3)`` array of predicted inter-epoch
    position changes (e.g. Doppler velocity * dt). When provided, the motion
    random-walk factor penalises ``(x_{t} - x_{t+1}) + disp[t]`` instead of
    ``(x_{t} - x_{t+1})``, equivalent to gtsam_gnss DopplerFactor_XXCC.
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
    max_iter: int = 25,
    tol: float = 1e-3,
    huber_k: float = 0.0,
    line_search: bool = True,
    sat_vel: np.ndarray | None = None,
    doppler: np.ndarray | None = None,
    doppler_weights: np.ndarray | None = None,
    dt: np.ndarray | None = None,
) -> tuple[int, float]:
    """Extended FGO with velocity state + Doppler factor (in-place ``state``).

    ``state`` has shape ``(T, 7 + n_clock)``:
    ``[x, y, z, vx, vy, vz, c0, ..., c_{K-1}, drift]`` in metres / (m/s).

    Motion factor couples position and velocity: ``x_{t+1} = x_t + v_t * dt``.
    Clock drift factor: ``clk_{t+1} = clk_t + drift_t * dt``.

    Doppler factor constrains velocity and clock drift from pseudorange-rate
    observations. Requires ``sat_vel`` (satellite velocity), ``doppler``
    (pseudorange-rate), ``doppler_weights``, and ``dt`` (inter-epoch time
    differences).

    ``sat_vel``: ``(T, S, 3)`` satellite velocity in ECEF (m/s).
    ``doppler``: ``(T, S)`` pseudorange-rate (m/s); 0 = unobserved.
    ``doppler_weights``: ``(T, S)`` weights for Doppler observations.
    ``dt``: ``(T,)`` inter-epoch time differences in seconds; ``dt[T-1]`` unused.

    Maintains backward compatibility: if no Doppler data is provided, only
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
    ss = 7 + n_clock
    if state.shape[1] != ss:
        raise ValueError(f"state columns {state.shape[1]} != 7 + n_clock ({ss})")

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

    dt_arr = None
    if dt is not None:
        dt_arr = np.ascontiguousarray(dt, dtype=np.float64).ravel()

    ls = 1 if line_search else 0
    return _fgo_gnss_lm_vd(
        sat_ecef,
        pseudorange,
        weights,
        state,
        float(motion_sigma_m),
        float(clock_drift_sigma_m),
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
    )


__all__ = ["fgo_gnss_lm", "fgo_gnss_lm_vd"]
