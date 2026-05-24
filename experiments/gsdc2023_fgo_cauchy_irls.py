"""Cauchy IRLS wrapper around ``fgo_gnss_lm_vd`` / ``fgo_gnss_lm``.

The native CUDA FGO solver only implements the Huber robust kernel.  This
module adds a Python-level **Iteratively Reweighted Least Squares (IRLS)**
loop with a **Cauchy** influence function, useful for NLOS-heavy GSDC2023
trips where Huber under-rejects outliers (e.g. ``lax-o``/``mtv-h``).

The Cauchy IRLS step computes per-observation effective weights from
the current residuals:

::

    z = sqrt(w_user) * |residual|
    w_eff = w_user / (1 + (z / c)^2)

then re-runs the native solver with the updated ``weights`` (and
``huber_k=0`` so the inner Huber path is disabled).  A small number of
outer iterations (typically 2-3) is sufficient to converge on the
Cauchy minimum from a Huber/L2 warm start.

Design notes
------------

* No CUDA changes required — Cauchy reweighting is computed on the
  host using ``sat_ecef`` and the current ``state`` (Sagnac correction
  matches ``pr_cost_host`` in ``src/positioning/fgo.cu``).
* The wrapper assumes the multi-clock VD state layout ``[x, y, z, vx,
  vy, vz, c0..c_{nc-1}, drift]`` used by ``fgo_gnss_lm_vd``.  See
  ``python/gnss_gpu/fgo.py``.
* The shape of the optional acceleration-bias state (``[bax, bay,
  baz]`` appended at the end) is preserved transparently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnss_gpu.fgo import fgo_gnss_lm_vd


LIGHT_SPEED_MPS = 299_792_458.0
OMEGA_E = 7.2921151467e-5


CAUCHY_C_DEFAULT_M = 4.0
CAUCHY_MAX_OUTER_ITERS_DEFAULT = 3
CAUCHY_WEIGHT_FLOOR = 1e-6


@dataclass(frozen=True)
class CauchyIRLSDiagnostics:
    inner_iters: int  # Cumulative inner LM iterations across outer rounds.
    outer_iters: int  # Number of outer reweight passes executed.
    final_mean_weight_ratio: float  # mean(weights_eff / weights_user).


def _hc(nc: int, sk: int) -> np.ndarray:
    """Mirror ``fill_hc_int`` from ``fgo.cu``: ``hc[0]=1, hc[sk]=1`` if sk>0."""
    hc = np.zeros(nc, dtype=np.float64)
    hc[0] = 1.0
    if 0 < sk < nc:
        hc[sk] = 1.0
    return hc


def _pr_residuals_vd(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    sys_kind: np.ndarray | None,
    state: np.ndarray,
    n_clock: int,
) -> np.ndarray:
    """Reproduce per-(t, s) Sagnac-corrected PR residuals."""
    n_epoch, n_sat = pseudorange.shape
    pos = state[:, :3]
    clks = state[:, 6 : 6 + n_clock]
    residuals = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        x, y, z = pos[t]
        for s in range(n_sat):
            sx, sy, sz = sat_ecef[t, s]
            dx0 = x - sx
            dy0 = y - sy
            dz0 = z - sz
            r0 = np.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0)
            if r0 < 1e-6:
                continue
            theta = OMEGA_E * (r0 / LIGHT_SPEED_MPS)
            ct = np.cos(theta)
            st = np.sin(theta)
            sx_rot = sx * ct + sy * st
            sy_rot = -sx * st + sy * ct
            dx = x - sx_rot
            dy_v = y - sy_rot
            dz = z - sz
            r_geom = np.sqrt(dx * dx + dy_v * dy_v + dz * dz)
            if r_geom < 1e-6:
                continue
            sk = int(sys_kind[t, s]) if sys_kind is not None else 0
            if sk < 0 or sk >= n_clock:
                continue
            hc = _hc(n_clock, sk)
            clk = float(np.dot(hc, clks[t]))
            residuals[t, s] = pseudorange[t, s] - (r_geom + clk)
    return residuals


def _apply_cauchy_weights(
    user_weights: np.ndarray,
    residuals: np.ndarray,
    cauchy_c: float,
) -> tuple[np.ndarray, float]:
    """Cauchy IRLS weight: ``w_eff = w_user / (1 + (z/c)^2)``."""
    w = np.asarray(user_weights, dtype=np.float64)
    res = np.asarray(residuals, dtype=np.float64)
    valid = (w > 0.0) & np.isfinite(res)
    z_m = np.zeros_like(w)
    z_m[valid] = np.sqrt(w[valid]) * np.abs(res[valid])
    denom = 1.0 + (z_m / max(float(cauchy_c), 1e-9)) ** 2
    eff_w = np.zeros_like(w)
    eff_w[valid] = w[valid] / denom[valid]
    eff_w = np.where(eff_w < CAUCHY_WEIGHT_FLOOR, 0.0, eff_w)
    pos = (w > 0.0)
    ratio = float(np.mean(eff_w[pos] / w[pos])) if np.any(pos) else 1.0
    return eff_w, ratio


def fgo_gnss_lm_vd_cauchy(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    *,
    cauchy_c_m: float = CAUCHY_C_DEFAULT_M,
    max_outer_iters: int = CAUCHY_MAX_OUTER_ITERS_DEFAULT,
    huber_k_warmstart: float = 0.0,
    **fgo_kwargs,
) -> tuple[int, float, CauchyIRLSDiagnostics]:
    """Run ``fgo_gnss_lm_vd`` under a Cauchy IRLS outer loop.

    Returns ``(iters_inner_total, mse_final, diagnostics)``.

    ``cauchy_c_m``: scale parameter ``c`` of the Cauchy influence
    function, expressed in metres (same units as Mahalanobis-scaled PR
    residuals).  Smaller ``c`` is more aggressive outlier rejection;
    taroz uses ``cauchy(c=2)`` in some configurations.

    ``max_outer_iters``: number of reweight passes (>=1).  When ``1``
    behaves like a single Huber-style call (or L2 if ``huber_k_warmstart=0``).
    ``2-3`` is the practical sweet spot.

    ``huber_k_warmstart``: huber threshold for the first inner run, to
    keep the state from diverging when initialised from a poor seed.
    ``0`` (default) means pure L2 warmstart.  After the first outer
    iteration the user weights are replaced by Cauchy IRLS weights and
    ``huber_k`` is forced to ``0`` so we do not stack two robust kernels.
    """
    if max_outer_iters < 1:
        raise ValueError("max_outer_iters must be >= 1")
    n_clock = int(fgo_kwargs.get("n_clock", 1))
    sys_kind = fgo_kwargs.get("sys_kind")

    user_weights = np.asarray(weights, dtype=np.float64).copy()
    current_weights = user_weights.copy()
    total_inner = 0
    last_mse = float("nan")
    last_ratio = 1.0

    for outer in range(max_outer_iters):
        kwargs = dict(fgo_kwargs)
        kwargs["huber_k"] = (
            float(huber_k_warmstart) if outer == 0 else 0.0
        )
        iters, mse = fgo_gnss_lm_vd(
            sat_ecef,
            pseudorange,
            current_weights,
            state,
            **kwargs,
        )
        total_inner += int(iters)
        last_mse = float(mse)
        if outer == max_outer_iters - 1:
            break
        residuals = _pr_residuals_vd(
            sat_ecef,
            pseudorange,
            np.asarray(sys_kind, dtype=np.int32) if sys_kind is not None else None,
            state,
            n_clock,
        )
        current_weights, last_ratio = _apply_cauchy_weights(
            user_weights, residuals, cauchy_c_m
        )

    diagnostics = CauchyIRLSDiagnostics(
        inner_iters=total_inner,
        outer_iters=int(max_outer_iters),
        final_mean_weight_ratio=last_ratio,
    )
    return total_inner, last_mse, diagnostics


__all__ = [
    "CAUCHY_C_DEFAULT_M",
    "CAUCHY_MAX_OUTER_ITERS_DEFAULT",
    "CauchyIRLSDiagnostics",
    "fgo_gnss_lm_vd_cauchy",
]
