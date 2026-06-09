"""Doppler-based velocity estimation module for GNSS."""

import numpy as np

try:
    from gnss_gpu._gnss_gpu_doppler import (
        doppler_velocity as _doppler_velocity,
        doppler_velocity_batch as _doppler_velocity_batch,
    )
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False

# L1 C/A wavelength [m] (GPS L1: 1575.42 MHz)
L1_WAVELENGTH = 0.19029367279836488


def _validate_solver_options(name, wavelength, max_iter, tol):
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise RuntimeError(f"{name}: wavelength must be positive and finite")
    if max_iter < 1:
        raise RuntimeError(f"{name}: max_iter must be >= 1")
    if not np.isfinite(tol) or tol <= 0.0:
        raise RuntimeError(f"{name}: tol must be positive and finite")


def _sat_matrix(name, label, values, n_sat):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1 and arr.size == n_sat * 3:
        arr = arr.reshape(n_sat, 3)
    elif not (arr.ndim == 2 and arr.shape == (n_sat, 3)):
        raise RuntimeError(
            f"{name}: {label} must have shape (n_sat, 3) or flat length n_sat*3"
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_doppler_single(sat_ecef, sat_vel, doppler, rx_pos, weights, wavelength, max_iter, tol):
    name = "doppler_velocity"
    doppler = np.asarray(doppler, dtype=np.float64)
    if doppler.ndim != 1:
        raise RuntimeError(f"{name}: doppler must have shape (n_sat,)")
    n_sat = doppler.size
    if n_sat < 1:
        raise RuntimeError(f"{name} requires at least one satellite")

    sat_ecef = _sat_matrix(name, "sat_ecef", sat_ecef, n_sat)
    sat_vel = _sat_matrix(name, "sat_vel", sat_vel, n_sat)

    rx_pos = np.asarray(rx_pos, dtype=np.float64)
    if rx_pos.ndim != 1 or rx_pos.size != 3:
        raise RuntimeError(f"{name}: rx_pos must have shape (3,)")

    if weights is None:
        weights = np.ones(n_sat, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or weights.size != n_sat:
            raise RuntimeError(f"{name}: weights must have shape (n_sat,)")

    _validate_solver_options(name, wavelength, max_iter, tol)

    if not (np.isfinite(sat_ecef).all() and np.isfinite(sat_vel).all()):
        raise RuntimeError(f"{name}: satellite positions and velocities must be finite")
    if not np.isfinite(doppler).all():
        raise RuntimeError(f"{name}: doppler values must be finite")
    if not np.isfinite(rx_pos).all():
        raise RuntimeError(f"{name}: rx_pos must be finite")
    if not (np.isfinite(weights).all() and np.all(weights >= 0.0)):
        raise RuntimeError(f"{name}: weights must be finite and nonnegative")

    return (
        sat_ecef,
        sat_vel,
        np.ascontiguousarray(doppler, dtype=np.float64),
        np.ascontiguousarray(rx_pos, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
    )


def _validate_doppler_batch(sat_ecef, sat_vel, doppler, rx_pos, weights, wavelength, max_iter, tol):
    name = "doppler_velocity_batch"
    sat_ecef = np.asarray(sat_ecef, dtype=np.float64)
    if sat_ecef.ndim != 3 or sat_ecef.shape[2] != 3:
        raise RuntimeError(f"{name}: sat_ecef must have shape (n_epoch, n_sat, 3)")
    n_epoch, n_sat, _ = sat_ecef.shape
    if n_epoch < 1:
        raise RuntimeError(f"{name}: n_epoch must be >= 1")
    if n_sat < 4:
        raise RuntimeError(f"{name} requires at least 4 satellites")

    sat_vel = np.asarray(sat_vel, dtype=np.float64)
    doppler = np.asarray(doppler, dtype=np.float64)
    rx_pos = np.asarray(rx_pos, dtype=np.float64)
    if sat_vel.ndim != 3 or sat_vel.shape != sat_ecef.shape:
        raise RuntimeError(f"{name}: sat_vel shape must match sat_ecef")
    if doppler.ndim != 2 or doppler.shape != (n_epoch, n_sat):
        raise RuntimeError(f"{name}: doppler must have shape (n_epoch, n_sat)")
    if rx_pos.ndim != 2 or rx_pos.shape != (n_epoch, 3):
        raise RuntimeError(f"{name}: rx_pos must have shape (n_epoch, 3)")

    if weights is None:
        weights = np.ones((n_epoch, n_sat), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 2 or weights.shape != (n_epoch, n_sat):
            raise RuntimeError(f"{name}: weights must have shape (n_epoch, n_sat)")

    _validate_solver_options(name, wavelength, max_iter, tol)

    if not (np.isfinite(sat_ecef).all() and np.isfinite(sat_vel).all()):
        raise RuntimeError(f"{name}: satellite positions and velocities must be finite")
    if not np.isfinite(doppler).all():
        raise RuntimeError(f"{name}: doppler values must be finite")
    if not np.isfinite(rx_pos).all():
        raise RuntimeError(f"{name}: rx_pos must be finite")
    if not (np.isfinite(weights).all() and np.all(weights >= 0.0)):
        raise RuntimeError(f"{name}: weights must be finite and nonnegative")

    return (
        np.ascontiguousarray(sat_ecef, dtype=np.float64),
        np.ascontiguousarray(sat_vel, dtype=np.float64),
        np.ascontiguousarray(doppler, dtype=np.float64),
        np.ascontiguousarray(rx_pos, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
    )


def doppler_velocity(sat_ecef, sat_vel, doppler, rx_pos, weights=None,
                     wavelength=L1_WAVELENGTH, max_iter=10, tol=1e-6):
    """Estimate receiver velocity from Doppler measurements using WLS.

    Parameters
    ----------
    sat_ecef : array_like, shape (n_sat, 3) or flat length n_sat*3
        Satellite ECEF positions [m].
    sat_vel : array_like, shape (n_sat, 3) or flat length n_sat*3
        Satellite ECEF velocities [m/s].
    doppler : array_like, shape (n_sat,)
        Doppler frequency measurements [Hz].
    rx_pos : array_like, shape (3,)
        Known receiver ECEF position [m].
    weights : array_like, shape (n_sat,), optional
        Observation weights (1/sigma^2). Defaults to ones.
    wavelength : float
        Carrier wavelength [m]. Default is GPS L1.
    max_iter : int
        Maximum Gauss-Newton iterations.
    tol : float
        Convergence tolerance [m/s].

    Returns
    -------
    result : numpy.ndarray, shape (4,)
        Estimated (vx, vy, vz, clock_drift) in ECEF [m/s].
    iters : int
        Number of iterations used.
    """
    sat_ecef, sat_vel, doppler, rx_pos, weights = _validate_doppler_single(
        sat_ecef, sat_vel, doppler, rx_pos, weights, wavelength, max_iter, tol
    )

    if _HAS_NATIVE:
        return _doppler_velocity(
            sat_ecef, sat_vel, doppler, rx_pos, weights,
            wavelength, max_iter, tol)

    # Pure-Python fallback
    return _doppler_velocity_py(
        sat_ecef, sat_vel, doppler, rx_pos, weights,
        wavelength, max_iter, tol)


def doppler_velocity_batch(sat_ecef, sat_vel, doppler, rx_pos, weights=None,
                           wavelength=L1_WAVELENGTH, max_iter=10, tol=1e-6):
    """Batch Doppler velocity estimation (GPU parallel).

    Parameters
    ----------
    sat_ecef : array_like, shape (n_epoch, n_sat, 3)
        Satellite ECEF positions [m].
    sat_vel : array_like, shape (n_epoch, n_sat, 3)
        Satellite ECEF velocities [m/s].
    doppler : array_like, shape (n_epoch, n_sat)
        Doppler frequency measurements [Hz].
    rx_pos : array_like, shape (n_epoch, 3)
        Known receiver ECEF positions [m].
    weights : array_like, shape (n_epoch, n_sat), optional
        Observation weights. Defaults to ones.
    wavelength : float
        Carrier wavelength [m].
    max_iter : int
        Maximum iterations per epoch.
    tol : float
        Convergence tolerance [m/s].

    Returns
    -------
    results : numpy.ndarray, shape (n_epoch, 4)
        Estimated (vx, vy, vz, clock_drift) per epoch [m/s].
    iters : numpy.ndarray, shape (n_epoch,)
        Iterations used per epoch.
    """
    sat_ecef, sat_vel, doppler, rx_pos, weights = _validate_doppler_batch(
        sat_ecef, sat_vel, doppler, rx_pos, weights, wavelength, max_iter, tol
    )
    n_epoch = sat_ecef.shape[0]

    if _HAS_NATIVE:
        return _doppler_velocity_batch(
            sat_ecef, sat_vel, doppler, rx_pos, weights,
            wavelength, max_iter, tol)

    # Pure-Python fallback (loop over epochs)
    results = np.zeros((n_epoch, 4), dtype=np.float64)
    iters = np.zeros(n_epoch, dtype=np.int32)
    for i in range(n_epoch):
        r, it = _doppler_velocity_py(
            sat_ecef[i], sat_vel[i], doppler[i], rx_pos[i], weights[i],
            wavelength, max_iter, tol)
        results[i] = r
        iters[i] = it
    return results, iters


def _doppler_velocity_py(sat_ecef, sat_vel, doppler, rx_pos, weights,
                         wavelength, max_iter, tol):
    """Pure-Python Doppler velocity WLS (fallback)."""
    n_sat = len(doppler)
    if n_sat < 4:
        return np.zeros(4), -1

    rx = rx_pos[0]
    ry = rx_pos[1]
    rz = rx_pos[2]

    vx, vy, vz, cd = 0.0, 0.0, 0.0, 0.0

    for it in range(max_iter):
        HTWH = np.zeros((4, 4))
        HTWdy = np.zeros(4)

        for s in range(n_sat):
            dx = sat_ecef[s, 0] - rx
            dy = sat_ecef[s, 1] - ry
            dz = sat_ecef[s, 2] - rz
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            if r < 1e-6:
                continue

            lx, ly, lz = dx / r, dy / r, dz / r

            pred = ((sat_vel[s, 0] - vx) * lx +
                    (sat_vel[s, 1] - vy) * ly +
                    (sat_vel[s, 2] - vz) * lz + cd)
            obs = doppler[s] * wavelength
            residual = obs - pred
            w = weights[s]

            H = np.array([-lx, -ly, -lz, 1.0])
            for a in range(4):
                HTWdy[a] += H[a] * w * residual
                for b in range(4):
                    HTWH[a, b] += H[a] * w * H[b]

        try:
            delta = np.linalg.solve(HTWH, HTWdy)
        except np.linalg.LinAlgError:
            break

        vx += delta[0]
        vy += delta[1]
        vz += delta[2]
        cd += delta[3]

        if np.linalg.norm(delta) < tol:
            return np.array([vx, vy, vz, cd]), it + 1

    return np.array([vx, vy, vz, cd]), max_iter
