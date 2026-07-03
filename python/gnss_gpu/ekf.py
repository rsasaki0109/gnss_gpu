"""Extended Kalman Filter positioning module for GNSS.

``EKFPositioner`` selects one of two internal state backends at ``initialize()``:

- **Native** (``_NativeState``): when ``_gnss_gpu_ekf`` is importable. State is
  held as plain ``numpy`` arrays (``x`` length 8, ``P`` length 64) because
  ``ekf_predict`` / ``ekf_update`` mutate buffers in-place — avoiding pybind11
  struct-copy semantics on ``EKFState``.
- **Pure Python** (``_PureState``): when the CUDA extension is absent (CI smoke,
  editable install without rebuild). Implements the same predict/update math in
  NumPy.

Both backends expose the same public API (``get_position``, ``get_velocity``,
``get_covariance``). Callers must not depend on ``ekf.state`` type. Unifying
into a single class is deferred: behaviour parity is tested via ``tests/test_ekf.py``
and ``tests/test_ekf_wrapper.py``, not by merging implementations.
"""

import numpy as np

from gnss_gpu.input_validation import (
    as_position_ecef,
    as_sat_ecef_matrix,
    finite_1d_array,
    finite_float,
    nonnegative_1d_array,
    positive_float,
)

try:
    from gnss_gpu._gnss_gpu_ekf import (
        EKFConfig,
        EKFState,
        ekf_initialize,
        ekf_predict,
        ekf_update,
        ekf_batch,
    )
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


# Native backend: numpy buffers passed to in-place ekf_predict / ekf_update.
class _NativeState:
    """Holds EKF state as numpy arrays for the ``_gnss_gpu_ekf`` backend."""

    def __init__(self, ekf_state):
        # Extract arrays from the native EKFState returned by ekf_initialize
        self.x = np.asarray(ekf_state.get_state(), dtype=np.float64).copy()
        self.P = np.asarray(ekf_state.get_covariance(), dtype=np.float64).ravel().copy()


class EKFPositioner:
    """Extended Kalman Filter for GNSS positioning.

    State vector: ``[x, y, z, vx, vy, vz, clock_bias, clock_drift]``.

    Uses a native CUDA backend when available, otherwise a pure-Python fallback
    with identical validation and array contracts (see module docstring).

    Parameters
    ----------
    sigma_pr : float
        Pseudorange measurement noise standard deviation [m].
    sigma_pos : float
        Position process noise [m/s^2].
    sigma_vel : float
        Velocity process noise [m/s^3].
    sigma_clk : float
        Clock bias process noise [m/s].
    sigma_drift : float
        Clock drift process noise [m/s^2].
    """

    def __init__(self, sigma_pr=5.0, sigma_pos=1.0, sigma_vel=0.1,
                 sigma_clk=100.0, sigma_drift=10.0):
        if _HAS_NATIVE:
            self.config = EKFConfig(
                sigma_pos=sigma_pos, sigma_vel=sigma_vel,
                sigma_clk=sigma_clk, sigma_drift=sigma_drift,
                sigma_pr=sigma_pr,
            )
        else:
            self.config = {
                'sigma_pos': sigma_pos,
                'sigma_vel': sigma_vel,
                'sigma_clk': sigma_clk,
                'sigma_drift': sigma_drift,
                'sigma_pr': sigma_pr,
            }
        self.sigma_pr = sigma_pr
        self.state = None
        self.initialized = False

    def initialize(self, position_ecef, clock_bias=0.0,
                   sigma_pos=100.0, sigma_cb=1000.0):
        """Initialize EKF state from an initial ECEF position.

        Parameters
        ----------
        position_ecef : array_like, shape (3,)
            Initial ECEF position [m].
        clock_bias : float
            Initial receiver clock bias [m].
        sigma_pos : float
            Initial position uncertainty [m].
        sigma_cb : float
            Initial clock bias uncertainty [m].
        """
        pos = as_position_ecef(position_ecef)
        sigma_pos = positive_float("sigma_pos", sigma_pos)
        sigma_cb = positive_float("sigma_cb", sigma_cb)
        clock_bias = finite_float("clock_bias", clock_bias)

        if _HAS_NATIVE:
            native = ekf_initialize(pos, clock_bias, sigma_pos, sigma_cb)
            self.state = _NativeState(native)
        else:
            self.state = _PureState(pos, clock_bias, sigma_pos, sigma_cb)

        self.initialized = True

    def predict(self, dt=1.0):
        """Run the EKF prediction step.

        Parameters
        ----------
        dt : float
            Time step [s].
        """
        if not self.initialized:
            raise RuntimeError("EKF not initialized. Call initialize() first.")

        dt = positive_float("dt", dt)

        if _HAS_NATIVE:
            ekf_predict(self.state.x, self.state.P, dt, self.config)
        else:
            self.state.predict(dt, self.config)

    def update(self, sat_ecef, pseudoranges, weights=None):
        """Run the EKF measurement update step.

        Parameters
        ----------
        sat_ecef : array_like, shape (n_sat, 3)
            Satellite ECEF positions [m].
        pseudoranges : array_like, shape (n_sat,)
            Observed pseudoranges [m].
        weights : array_like, shape (n_sat,), optional
            Measurement weights (1/sigma^2). If None, uses 1/sigma_pr^2.
        """
        if not self.initialized:
            raise RuntimeError("EKF not initialized. Call initialize() first.")

        pr = finite_1d_array("pseudoranges", pseudoranges, min_size=1)
        n_sat = pr.size
        sat = as_sat_ecef_matrix(sat_ecef, n_sat)

        if weights is None:
            w = np.full(n_sat, 1.0 / (self.sigma_pr ** 2), dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).ravel()
            if w.size != n_sat:
                raise ValueError("weights length must match pseudoranges")
            w = nonnegative_1d_array("weights", w, min_size=n_sat)

        if _HAS_NATIVE:
            ekf_update(self.state.x, self.state.P, sat, pr, w)
        else:
            self.state.update(sat.ravel(), pr, w, n_sat)

    def get_position(self):
        """Return the current ECEF position estimate [3].

        Returns
        -------
        numpy.ndarray, shape (3,)
        """
        if not self.initialized:
            raise RuntimeError("EKF not initialized.")
        x = self._get_state_vec()
        return x[:3].copy()

    def get_velocity(self):
        """Return the current ECEF velocity estimate [3].

        Returns
        -------
        numpy.ndarray, shape (3,)
        """
        if not self.initialized:
            raise RuntimeError("EKF not initialized.")
        x = self._get_state_vec()
        return x[3:6].copy()

    def get_covariance(self):
        """Return the current 8x8 state covariance matrix.

        Returns
        -------
        numpy.ndarray, shape (8, 8)
        """
        if not self.initialized:
            raise RuntimeError("EKF not initialized.")
        if _HAS_NATIVE:
            return self.state.P.reshape(8, 8).copy()
        else:
            return self.state.P.reshape(8, 8).copy()

    def _get_state_vec(self):
        if _HAS_NATIVE:
            return self.state.x.copy()
        else:
            return self.state.x.copy()


class _PureState:
    """CPU-only EKF state used when ``_gnss_gpu_ekf`` is not importable."""

    def __init__(self, pos, cb, sigma_pos, sigma_cb):
        self.x = np.zeros(8, dtype=np.float64)
        self.x[0] = pos[0]
        self.x[1] = pos[1]
        self.x[2] = pos[2]
        self.x[6] = cb
        self.P = np.zeros(64, dtype=np.float64)
        sp2 = sigma_pos ** 2
        sv2 = 100.0 ** 2
        sc2 = sigma_cb ** 2
        sd2 = 100.0 ** 2
        P = np.diag([sp2, sp2, sp2, sv2, sv2, sv2, sc2, sd2])
        self.P = P.ravel().copy()

    def predict(self, dt, config):
        cfg = config
        sp = cfg['sigma_pos']
        sv = cfg['sigma_vel']
        sc = cfg['sigma_clk']
        sd = cfg['sigma_drift']

        F = np.eye(8)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        F[6, 7] = dt

        self.x = F @ self.x

        dt2 = dt * dt
        Q = np.diag([
            sp * sp * dt2, sp * sp * dt2, sp * sp * dt2,
            sv * sv * dt, sv * sv * dt, sv * sv * dt,
            sc * sc * dt2,
            sd * sd * dt,
        ])

        P = self.P.reshape(8, 8)
        P = F @ P @ F.T + Q
        self.P = P.ravel().copy()

    def update(self, sat_ecef_flat, pseudoranges, weights, n_sat):
        rx, ry, rz = self.x[0], self.x[1], self.x[2]
        cb = self.x[6]

        H = np.zeros((n_sat, 8))
        y = np.zeros(n_sat)
        R_diag = np.zeros(n_sat)

        for s in range(n_sat):
            sx = sat_ecef_flat[s * 3 + 0]
            sy = sat_ecef_flat[s * 3 + 1]
            sz = sat_ecef_flat[s * 3 + 2]

            dx = rx - sx
            dy = ry - sy
            dz = rz - sz
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            if r < 1e-6:
                r = 1e-6

            y[s] = pseudoranges[s] - (r + cb)
            H[s, 0] = dx / r
            H[s, 1] = dy / r
            H[s, 2] = dz / r
            H[s, 6] = 1.0
            R_diag[s] = 1.0 / weights[s] if weights[s] > 1e-15 else 1e6

        P = self.P.reshape(8, 8)
        R = np.diag(R_diag)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        IKH = np.eye(8) - K @ H
        P = IKH @ P
        self.P = P.ravel().copy()
