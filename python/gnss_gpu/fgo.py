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
) -> tuple[int, float]:
    """Iterated Gauss–Newton with GPU-assembled normal equations (in-place ``state``).

    ``state`` has shape ``(T, 3 + n_clock)``: ``[x,y,z,c0,...,c_{K-1}]`` in metres.
    ``sys_kind`` is optional ``int32`` ``(T, S)`` with values in ``0..n_clock-1``.
    Row ``h`` for a measurement is ``h[0]=1`` and ``h[sk]=1`` if ``sk > 0``
    (gtsam_gnss clock + ISB pattern).

    ``huber_k``: if > 0, apply IRLS Huber reweighting with threshold on Mahalanobis
    residuals ``z = |sqrt(w) * res|`` (same pattern as common robust GNSS solvers).
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
    )


__all__ = ["fgo_gnss_lm"]
