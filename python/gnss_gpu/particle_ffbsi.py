"""FFBSi-style backward index sampling for particle trajectories.

Matches the ``ParticleFilterDevice`` dynamics: predict adds per-axis Gaussian noise
``sigma_pos`` on x,y,z and ``sigma_cb`` on clock bias (no drift on cb beyond noise).
Backward weights combine normalized filtering log-weights at time t with the
transition density from x_t to x_{t+1}.
"""

from __future__ import annotations

import numpy as np


def transition_logpdf(
    x_next: np.ndarray,
    x_t: np.ndarray,
    vel: np.ndarray,
    dt: float,
    sigma_pos: float,
    sigma_cb: float,
) -> np.ndarray:
    """Log p(x_next | x_t) under one predict step. x_t shape (N, 4), x_next (4,)."""
    v = np.asarray(vel, dtype=np.float64).ravel()
    xt = np.asarray(x_t, dtype=np.float64)
    dp = x_next[:3] - (xt[:, :3] + v * float(dt))
    dcb = x_next[3] - xt[:, 3]
    inv_sp2 = 1.0 / (float(sigma_pos) ** 2)
    inv_sc2 = 1.0 / (float(sigma_cb) ** 2)
    return -0.5 * np.sum(dp * dp, axis=1) * inv_sp2 - 0.5 * (dcb * dcb) * inv_sc2


def ffbsi_sample_indices(
    log_weights: np.ndarray,
    X: np.ndarray,
    vel: np.ndarray,
    dt: np.ndarray,
    sigma_pos: np.ndarray,
    sigma_cb: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One FFBSi backward trajectory as ancestor indices I[0..T-1].

    Parameters
    ----------
    log_weights : (T, N)
        Log filtering weights after measurement / PU, before resample.
    X : (T, N, 4)
        Particle states [x,y,z,cb] aligned with ``log_weights``.
    vel, dt, sigma_pos : (T,)
        Predict parameters at the **start** of each epoch (row k applies to
        transition from end of k-1 to end of k). Row 0 may encode the first
        predict; transition t -> t+1 uses row t+1.
    sigma_cb : float
        Clock bias predict sigma (same as PF).
    """
    lw = np.asarray(log_weights, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    T, N = lw.shape
    if X.shape != (T, N, 4):
        raise ValueError(f"X shape {X.shape} != ({T},{N},4)")

    vel = np.asarray(vel, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    sigma_pos = np.asarray(sigma_pos, dtype=np.float64)
    if vel.shape[0] != T or dt.shape[0] != T or sigma_pos.shape[0] != T:
        raise ValueError("vel, dt, sigma_pos must have length T")

    idx = np.empty(T, dtype=np.int64)

    mx = lw[T - 1].max()
    w = np.exp(lw[T - 1] - mx)
    s = w.sum()
    if s <= 0.0 or not np.isfinite(s):
        idx[:] = 0
        return idx
    w = w / s
    idx[T - 1] = int(rng.choice(N, p=w))

    sig_cb = float(sigma_cb)

    for t in range(T - 2, -1, -1):
        mxt = lw[t].max()
        log_norm = mxt + np.log(np.sum(np.exp(lw[t] - mxt)))
        log_fil = lw[t] - log_norm

        lf = transition_logpdf(
            X[t + 1, idx[t + 1]],
            X[t],
            vel[t + 1],
            float(dt[t + 1]),
            float(sigma_pos[t + 1]),
            sig_cb,
        )
        logp = log_fil + lf
        mp = float(np.max(logp))
        w_b = np.exp(logp - mp)
        sb = float(np.sum(w_b))
        if sb <= 0.0 or not np.isfinite(sb):
            idx[t] = int(rng.integers(0, N))
            continue
        w_b = w_b / sb
        idx[t] = int(rng.choice(N, p=w_b))

    return idx


def ffbsi_smooth_sample(
    log_weights: np.ndarray,
    X: np.ndarray,
    vel: np.ndarray,
    dt: np.ndarray,
    sigma_pos: np.ndarray,
    sigma_cb: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One smoothed path (T, 3) ECEF from FFBSi indices."""
    idx = ffbsi_sample_indices(log_weights, X, vel, dt, sigma_pos, sigma_cb, rng)
    return np.asarray(X[np.arange(X.shape[0]), idx, :3], dtype=np.float64)


def genealogy_smooth_indices(
    log_weights: np.ndarray,
    ancestors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Backward trace along resampling genealogy (one ancestral lineage).

    ``ancestors[t, j]`` is the source particle index *i* such that, after
    resampling at the end of epoch *t*, slot *j* holds a copy of the pre-resample
    particle *i* (same convention as ``get_resample_ancestors``).

    Row ``T-1`` is ignored; it may be ``np.arange(N)``.
    """
    lw = np.asarray(log_weights, dtype=np.float64)
    anc = np.asarray(ancestors, dtype=np.int64)
    T, N = lw.shape
    if anc.shape != (T, N):
        raise ValueError(f"ancestors shape {anc.shape} != ({T},{N})")

    idx = np.empty(T, dtype=np.int64)
    mx = lw[T - 1].max()
    w = np.exp(lw[T - 1] - mx)
    s = float(w.sum())
    if s <= 0.0 or not np.isfinite(s):
        idx[:] = 0
        return idx
    w = w / s
    idx[T - 1] = int(rng.choice(N, p=w))
    for t in range(T - 2, -1, -1):
        idx[t] = int(anc[t, idx[t + 1]])
    return idx


def genealogy_smooth_sample(
    log_weights: np.ndarray,
    X: np.ndarray,
    ancestors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """One smoothed path (T, 3) ECEF via genealogy backward sampling."""
    idx = genealogy_smooth_indices(log_weights, ancestors, rng)
    X = np.asarray(X, dtype=np.float64)
    return np.asarray(X[np.arange(X.shape[0]), idx, :3], dtype=np.float64)
