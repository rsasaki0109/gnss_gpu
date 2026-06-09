"""Atmospheric delay correction models for GNSS positioning.

Provides tropospheric (Saastamoinen) and ionospheric (Klobuchar) delay
correction, with GPU-accelerated batch processing.
"""

import numpy as np

try:
    from gnss_gpu._gnss_gpu_atmosphere import (
        tropo_saastamoinen,  # noqa: F401
        iono_klobuchar,  # noqa: F401
        tropo_correction_batch,
        iono_correction_batch,
    )
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


_DEFAULT_ALPHA = [1.1176e-8, -7.4506e-9, -5.9605e-8, 1.1921e-7]
_DEFAULT_BETA = [1.1264e5, -3.2768e4, -2.6214e5, 4.5875e5]


def _array(name, values):
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be numeric") from exc


def _iono_params(name, values):
    values = _array(f"AtmosphereCorrection: {name}", values)
    if values.ndim != 1 or values.size != 4:
        raise RuntimeError(f"AtmosphereCorrection: {name} must have shape (4,)")
    if not np.isfinite(values).all():
        raise RuntimeError(f"AtmosphereCorrection: {name} must be finite")
    return values.astype(np.float64, copy=False).tolist()


def _rx_lla_array(name, rx_lla):
    rx_lla = _array(f"{name}: rx_lla", rx_lla)
    if rx_lla.ndim == 1:
        if rx_lla.size != 3:
            raise RuntimeError(f"{name}: rx_lla must have shape (3,) or (n_epoch, 3)")
        n_epoch = 1
        single_epoch = True
    elif rx_lla.ndim == 2 and rx_lla.shape[1] == 3:
        n_epoch = rx_lla.shape[0]
        if n_epoch < 1:
            raise RuntimeError(f"{name}: n_epoch must be >= 1")
        single_epoch = False
    else:
        raise RuntimeError(f"{name}: rx_lla must have shape (3,) or (n_epoch, 3)")

    if not np.isfinite(rx_lla).all():
        raise RuntimeError(f"{name}: rx_lla must be finite")

    return np.ascontiguousarray(rx_lla, dtype=np.float64), single_epoch, n_epoch


def _sat_angles(name, label, values, single_epoch, n_epoch):
    values = _array(f"{name}: {label}", values)
    if single_epoch and values.ndim == 0:
        values = values.reshape(1)

    if single_epoch:
        if values.ndim != 1:
            raise RuntimeError(f"{name}: {label} must have shape (n_sat,)")
        n_sat = values.size
    else:
        if values.ndim != 2 or values.shape[0] != n_epoch:
            raise RuntimeError(f"{name}: {label} must have shape (n_epoch, n_sat)")
        n_sat = values.shape[1]

    if n_sat < 1:
        raise RuntimeError(f"{name}: n_sat must be >= 1")
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name}: {label} must be finite")

    return np.ascontiguousarray(values, dtype=np.float64), n_sat


def _gps_times(name, gps_time, n_epoch):
    gps_times = _array(f"{name}: gps_time", gps_time)
    if gps_times.ndim == 0:
        if n_epoch != 1:
            raise RuntimeError(f"{name}: gps_time must have shape (n_epoch,)")
        gps_times = gps_times.reshape(1)
    elif gps_times.ndim != 1 or gps_times.size != n_epoch:
        raise RuntimeError(f"{name}: gps_time must have shape (n_epoch,)")

    if not np.isfinite(gps_times).all():
        raise RuntimeError(f"{name}: gps_time must be finite")

    return np.ascontiguousarray(gps_times, dtype=np.float64)


def _tropo_saastamoinen_cpu(lat, alt, el):
    """Pure-Python fallback for Saastamoinen tropospheric delay."""
    P = 1013.25 * (1.0 - 2.2557e-5 * alt) ** 5.2568
    T = 15.0 - 6.5e-3 * alt + 273.15
    e_wv = 6.108 * np.exp((17.15 * (T - 273.15)) / (T - 273.15 + 234.7))
    e_wv *= 0.5

    alt_km = alt / 1000.0
    tropo_zenith = 0.002277 * (P + (1255.0 / T + 0.05) * e_wv) / \
                   (1.0 - 0.00266 * np.cos(2.0 * lat) - 0.00028 * alt_km)

    el_min = 2.0 * np.pi / 180.0
    el_eff = np.where(el > el_min, el, el_min) if hasattr(el, '__len__') else max(el, el_min)
    sin_el = np.sin(np.sqrt(el_eff ** 2 + 6.25 * (np.pi / 180.0) ** 2))
    return tropo_zenith / sin_el


def _iono_klobuchar_cpu(alpha, beta, lat, lon, az, el, gps_time):
    """Pure-Python fallback for Klobuchar ionospheric delay."""
    C_LIGHT = 299792458.0
    PI = np.pi

    lat_sc = lat / PI
    lon_sc = lon / PI
    el_sc = el / PI

    psi = 0.0137 / (el_sc + 0.11) - 0.022

    phi_i = lat_sc + psi * np.cos(az)
    phi_i = np.clip(phi_i, -0.416, 0.416)

    lambda_i = lon_sc + psi * np.sin(az) / np.cos(phi_i * PI)

    phi_m = phi_i + 0.064 * np.cos((lambda_i - 1.617) * PI)

    t = 4.32e4 * lambda_i + gps_time
    t = np.mod(t, 86400.0)

    F = 1.0 + 16.0 * (0.53 - el_sc) ** 3

    PER = sum(beta[n] * phi_m ** n for n in range(4))
    PER = np.maximum(PER, 72000.0)

    AMP = sum(alpha[n] * phi_m ** n for n in range(4))
    AMP = np.maximum(AMP, 0.0)

    x = 2.0 * PI * (t - 50400.0) / PER

    if np.isscalar(x):
        if abs(x) < 1.57:
            Tiono = F * (5.0e-9 + AMP * (1.0 - x**2 / 2.0 + x**4 / 24.0))
        else:
            Tiono = F * 5.0e-9
    else:
        Tiono = np.where(
            np.abs(x) < 1.57,
            F * (5.0e-9 + AMP * (1.0 - x**2 / 2.0 + x**4 / 24.0)),
            F * 5.0e-9
        )

    return Tiono * C_LIGHT


class AtmosphereCorrection:
    """Atmospheric delay correction for GNSS positioning.

    Combines Saastamoinen tropospheric and Klobuchar ionospheric models.
    Supports both single-point CPU and GPU-batch computation.

    Parameters
    ----------
    iono_alpha : list of 4 floats, optional
        Klobuchar alpha parameters from GPS NAV message.
    iono_beta : list of 4 floats, optional
        Klobuchar beta parameters from GPS NAV message.
    """

    def __init__(self, iono_alpha=None, iono_beta=None):
        # Default alpha/beta from GPS broadcast (typical values)
        self.alpha = _iono_params(
            "iono_alpha",
            _DEFAULT_ALPHA if iono_alpha is None else iono_alpha,
        )
        self.beta = _iono_params(
            "iono_beta",
            _DEFAULT_BETA if iono_beta is None else iono_beta,
        )

    def tropo(self, rx_lla, sat_el):
        """Compute tropospheric delay correction.

        Parameters
        ----------
        rx_lla : array_like, shape (3,) or (n_epoch, 3)
            Receiver position [lat_rad, lon_rad, alt_m].
        sat_el : array_like, shape (n_sat,) or (n_epoch, n_sat)
            Satellite elevation angles [rad].

        Returns
        -------
        numpy.ndarray
            Tropospheric delay corrections in meters.
        """
        rx_lla, single_epoch, n_epoch = _rx_lla_array(
            "AtmosphereCorrection.tropo", rx_lla
        )
        sat_el, _ = _sat_angles(
            "AtmosphereCorrection.tropo", "sat_el", sat_el, single_epoch, n_epoch
        )

        if _HAS_GPU:
            return np.asarray(tropo_correction_batch(rx_lla, sat_el))

        # CPU fallback
        if single_epoch:
            return _tropo_saastamoinen_cpu(rx_lla[0], rx_lla[2], sat_el)

        results = np.empty_like(sat_el)
        for i in range(rx_lla.shape[0]):
            results[i] = _tropo_saastamoinen_cpu(
                rx_lla[i, 0], rx_lla[i, 2], sat_el[i])
        return results

    def iono(self, rx_lla, sat_az, sat_el, gps_time):
        """Compute ionospheric delay correction.

        Parameters
        ----------
        rx_lla : array_like, shape (3,) or (n_epoch, 3)
            Receiver position [lat_rad, lon_rad, alt_m].
        sat_az : array_like, shape (n_sat,) or (n_epoch, n_sat)
            Satellite azimuth angles [rad].
        sat_el : array_like, shape (n_sat,) or (n_epoch, n_sat)
            Satellite elevation angles [rad].
        gps_time : float or array_like, shape (n_epoch,)
            GPS time of week [s].

        Returns
        -------
        numpy.ndarray
            Ionospheric delay corrections in meters (L1 frequency).
        """
        rx_lla, single_epoch, n_epoch = _rx_lla_array(
            "AtmosphereCorrection.iono", rx_lla
        )
        sat_az, n_sat = _sat_angles(
            "AtmosphereCorrection.iono", "sat_az", sat_az, single_epoch, n_epoch
        )
        sat_el, n_sat_el = _sat_angles(
            "AtmosphereCorrection.iono", "sat_el", sat_el, single_epoch, n_epoch
        )
        if n_sat_el != n_sat:
            raise RuntimeError(
                "AtmosphereCorrection.iono: sat_az and sat_el must have matching shape"
            )
        gps_times = _gps_times("AtmosphereCorrection.iono", gps_time, n_epoch)

        alpha = np.array(self.alpha, dtype=np.float64)
        beta = np.array(self.beta, dtype=np.float64)

        if _HAS_GPU:
            return np.asarray(iono_correction_batch(
                rx_lla, sat_az, sat_el, alpha, beta, gps_times))

        # CPU fallback
        if single_epoch:
            return _iono_klobuchar_cpu(
                self.alpha, self.beta,
                rx_lla[0], rx_lla[1],
                sat_az, sat_el, gps_times[0])

        results = np.empty_like(sat_el)
        for i in range(rx_lla.shape[0]):
            results[i] = _iono_klobuchar_cpu(
                self.alpha, self.beta,
                rx_lla[i, 0], rx_lla[i, 1],
                sat_az[i], sat_el[i], gps_times[i])
        return results

    def total(self, rx_lla, sat_az, sat_el, gps_time=0.0):
        """Compute total atmospheric delay (tropospheric + ionospheric).

        Parameters
        ----------
        rx_lla : array_like, shape (3,) or (n_epoch, 3)
            Receiver position [lat_rad, lon_rad, alt_m].
        sat_az : array_like, shape (n_sat,) or (n_epoch, n_sat)
            Satellite azimuth angles [rad].
        sat_el : array_like, shape (n_sat,) or (n_epoch, n_sat)
            Satellite elevation angles [rad].
        gps_time : float or array_like, shape (n_epoch,), optional
            GPS time of week [s]. Default is 0 (nighttime minimum).

        Returns
        -------
        numpy.ndarray
            Total atmospheric delay corrections in meters.
        """
        t = self.tropo(rx_lla, sat_el)
        i = self.iono(rx_lla, sat_az, sat_el, gps_time)
        return t + i
