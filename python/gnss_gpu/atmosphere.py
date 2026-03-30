"""Atmospheric delay correction models for GNSS positioning.

Provides tropospheric (Saastamoinen) and ionospheric (Klobuchar) delay
correction, with GPU-accelerated batch processing.
"""

import numpy as np

try:
    from gnss_gpu._gnss_gpu_atmosphere import (
        tropo_saastamoinen,
        iono_klobuchar,
        tropo_correction_batch,
        iono_correction_batch,
    )
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


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
        self.alpha = iono_alpha or [1.1176e-8, -7.4506e-9, -5.9605e-8, 1.1921e-7]
        self.beta = iono_beta or [1.1264e5, -3.2768e4, -2.6214e5, 4.5875e5]

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
        rx_lla = np.asarray(rx_lla, dtype=np.float64)
        sat_el = np.asarray(sat_el, dtype=np.float64)

        if _HAS_GPU:
            return np.asarray(tropo_correction_batch(
                np.ascontiguousarray(rx_lla),
                np.ascontiguousarray(sat_el)))

        # CPU fallback
        if rx_lla.ndim == 1:
            return _tropo_saastamoinen_cpu(rx_lla[0], rx_lla[2], sat_el)
        else:
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
        rx_lla = np.asarray(rx_lla, dtype=np.float64)
        sat_az = np.asarray(sat_az, dtype=np.float64)
        sat_el = np.asarray(sat_el, dtype=np.float64)
        gps_times = np.atleast_1d(np.asarray(gps_time, dtype=np.float64))

        alpha = np.array(self.alpha, dtype=np.float64)
        beta = np.array(self.beta, dtype=np.float64)

        if _HAS_GPU:
            return np.asarray(iono_correction_batch(
                np.ascontiguousarray(rx_lla),
                np.ascontiguousarray(sat_az),
                np.ascontiguousarray(sat_el),
                np.ascontiguousarray(alpha),
                np.ascontiguousarray(beta),
                np.ascontiguousarray(gps_times)))

        # CPU fallback
        if rx_lla.ndim == 1:
            return _iono_klobuchar_cpu(
                self.alpha, self.beta,
                rx_lla[0], rx_lla[1],
                sat_az, sat_el, gps_times[0])
        else:
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
