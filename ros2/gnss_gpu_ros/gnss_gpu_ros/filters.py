"""Causal GNSS trajectory filters (NumPy only, no ROS imports).

These are streaming (causal) ports of the offline trajectory post-processing
ideas validated on the GSDC2023 Kaggle challenge (see
``docs/gsdc2023_solution.md`` in the repository root):

* :class:`CausalHampel` — trailing-window median/MAD spike gate, the causal
  twin of the offline Hampel layer (offline: −7 cm and −93% max jump on the
  41-trip train set).
* :class:`CvKalman1D` — per-axis constant-velocity Kalman filter, the forward
  half of the offline RTS smoother layer (offline: −9.6 cm, 39/41 trips won).

Both operate per axis in a local East/North tangent plane anchored at the
first fix; :class:`NavSatTrajectoryFilter` wires them together for
latitude/longitude streams.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

# WGS84 constants
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F * _WGS84_F

# MAD of 5e-7 deg latitude — the offline Hampel floor — is about 5.6 cm.
_DEFAULT_MAD_FLOOR_M = 0.056


class CausalHampel:
    """Trailing-window Hampel outlier gate for one axis.

    Keeps the last ``window`` raw values; an incoming value farther than
    ``k * 1.4826 * MAD`` (with a floor so a perfectly stationary window does
    not flag noise) from the window median is *output* as that median, but the
    raw value still enters the window. Feeding raw values (never the median)
    into the window is what lets the gate recover from genuine sustained
    motion changes — a departure from a stop or a re-acquisition jump after an
    outage shifts the median within about half a window. As a faster escape
    hatch, after ``max_consecutive`` flagged samples in a row the gate stands
    down and passes raw values until the stream looks consistent again.
    During warm-up (fewer than ``min_samples`` values) everything passes
    through.
    """

    def __init__(
        self,
        window: int = 21,
        k: float = 2.5,
        mad_floor: float = _DEFAULT_MAD_FLOOR_M,
        min_samples: int = 5,
        max_consecutive: int = 5,
    ) -> None:
        if window < 3:
            raise ValueError("window must be >= 3")
        if k <= 0 or mad_floor <= 0:
            raise ValueError("k and mad_floor must be positive")
        if max_consecutive < 1:
            raise ValueError("max_consecutive must be >= 1")
        self.k = k
        self.mad_floor = mad_floor
        self.min_samples = max(3, min_samples)
        self.max_consecutive = max_consecutive
        self._buf: deque[float] = deque(maxlen=window)
        self._streak = 0

    def reset(self) -> None:
        """Discard streaming history after a restart or long data outage."""
        self._buf.clear()
        self._streak = 0

    def update(self, value: float) -> tuple[float, bool]:
        """Feed one value; return ``(filtered_value, was_outlier)``."""
        if len(self._buf) < self.min_samples:
            self._buf.append(value)
            return value, False
        arr = np.asarray(self._buf, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = max(1.4826 * mad, self.mad_floor)
        self._buf.append(value)
        if abs(value - med) > self.k * scale:
            self._streak += 1
            if self._streak <= self.max_consecutive:
                return med, True
            # Too many in a row: this is real motion, not a spike — stand down.
            return value, False
        self._streak = 0
        return value, False


class CvKalman1D:
    """Constant-velocity Kalman filter for one axis (causal, forward only).

    Same motion model as the offline RTS smoother layer:
    ``F = [[1, dt], [0, 1]]``, ``Q = sigma_a^2 * [[dt^4/4, dt^3/2],
    [dt^3/2, dt^2]]``, scalar position measurement with ``R = sigma_z^2``.
    """

    def __init__(self, sigma_a: float = 1.0, sigma_z: float = 1.0) -> None:
        if sigma_a <= 0 or sigma_z <= 0:
            raise ValueError("sigma_a and sigma_z must be positive")
        self.sigma_a = sigma_a
        self.sigma_z = sigma_z
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        """Discard the position, velocity, covariance, and timestamp state."""
        self._x = None
        self._P = None
        self._t = None

    def update(self, t: float, z: float) -> float:
        """Feed one timestamped measurement; return the filtered position."""
        r = self.sigma_z**2
        if self._x is None or self._t is None or t <= self._t:
            # First sample, or non-monotonic timestamps: (re)initialize.
            self._x = np.array([z, 0.0])
            self._P = np.diag([r, 100.0])
            self._t = t
            return z
        dt = t - self._t
        self._t = t
        f = np.array([[1.0, dt], [0.0, 1.0]])
        q = self.sigma_a**2 * np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]]
        )
        x = f @ self._x
        p = f @ self._P @ f.T + q
        h = np.array([1.0, 0.0])
        s = float(h @ p @ h) + r
        k_gain = (p @ h) / s
        self._x = x + k_gain * (z - float(h @ x))
        self._P = (np.eye(2) - np.outer(k_gain, h)) @ p
        return float(self._x[0])


def _radii_at(lat_deg: float) -> tuple[float, float]:
    """WGS84 meridional and normal-times-cos(lat) radii at a latitude."""
    lat = math.radians(lat_deg)
    sin_lat = math.sin(lat)
    denom = 1.0 - _WGS84_E2 * sin_lat * sin_lat
    r_meridional = _WGS84_A * (1.0 - _WGS84_E2) / denom**1.5
    r_normal = _WGS84_A / math.sqrt(denom)
    return r_meridional, r_normal * math.cos(lat)


class NavSatTrajectoryFilter:
    """Hampel gate + CV Kalman on latitude/longitude fixes.

    Works in a local East/North tangent plane anchored at the first fix
    (adequate for the tens-of-km scale the filter window actually spans).
    Either stage can be disabled.
    """

    def __init__(
        self,
        hampel_window: int = 21,
        hampel_k: float = 2.5,
        kalman_sigma_a: float = 1.0,
        kalman_sigma_z: float = 1.0,
        use_hampel: bool = True,
        use_kalman: bool = True,
        max_gap_s: float = 30.0,
    ) -> None:
        if not math.isfinite(max_gap_s) or max_gap_s <= 0.0:
            raise ValueError("max_gap_s must be finite and positive")
        self.use_hampel = use_hampel
        self.use_kalman = use_kalman
        self.max_gap_s = float(max_gap_s)
        self._hampel_e = CausalHampel(hampel_window, hampel_k)
        self._hampel_n = CausalHampel(hampel_window, hampel_k)
        self._kf_e = CvKalman1D(kalman_sigma_a, kalman_sigma_z)
        self._kf_n = CvKalman1D(kalman_sigma_a, kalman_sigma_z)
        self._anchor: tuple[float, float] | None = None
        self._r_m = 0.0
        self._r_p = 0.0
        self._last_t: float | None = None

    def reset(self, *, preserve_anchor: bool = False) -> None:
        """Reset all streaming state.

        ``preserve_anchor=True`` is used for an in-process GNSS outage so the
        published local EN frame does not jump. A node restart uses the
        default and establishes a fresh anchor from its first fix.
        """
        self._hampel_e.reset()
        self._hampel_n.reset()
        self._kf_e.reset()
        self._kf_n.reset()
        self._last_t = None
        if not preserve_anchor:
            self._anchor = None
            self._r_m = 0.0
            self._r_p = 0.0

    def _to_en(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        lat0, lon0 = self._anchor  # type: ignore[misc]
        return (
            math.radians(lon_deg - lon0) * self._r_p,
            math.radians(lat_deg - lat0) * self._r_m,
        )

    def _to_llh(self, e: float, n: float) -> tuple[float, float]:
        lat0, lon0 = self._anchor  # type: ignore[misc]
        return (
            lat0 + math.degrees(n / self._r_m),
            lon0 + math.degrees(e / self._r_p),
        )

    def update(
        self, t: float, lat_deg: float, lon_deg: float
    ) -> tuple[float, float, float, float, bool]:
        """Feed one fix; return ``(lat, lon, east, north, was_outlier)``.

        ``east``/``north`` are the filtered local-plane coordinates relative
        to the first fix (useful for Path/odometry-style consumers).
        """
        if not all(math.isfinite(value) for value in (t, lat_deg, lon_deg)):
            raise ValueError("timestamp, latitude, and longitude must be finite")
        if not -90.0 <= lat_deg <= 90.0:
            raise ValueError("latitude must be in [-90, 90]")
        if not -180.0 <= lon_deg <= 180.0:
            raise ValueError("longitude must be in [-180, 180]")
        if self._last_t is not None and t - self._last_t > self.max_gap_s:
            self.reset(preserve_anchor=True)
        self._last_t = t

        if self._anchor is None:
            self._anchor = (lat_deg, lon_deg)
            self._r_m, self._r_p = _radii_at(lat_deg)
        e, n = self._to_en(lat_deg, lon_deg)
        outlier = False
        if self.use_hampel:
            e, oe = self._hampel_e.update(e)
            n, on = self._hampel_n.update(n)
            outlier = oe or on
        if self.use_kalman:
            e = self._kf_e.update(t, e)
            n = self._kf_n.update(t, n)
        lat, lon = self._to_llh(e, n)
        return lat, lon, e, n, outlier
